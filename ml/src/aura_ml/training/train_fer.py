"""Train the facial-emotion classifier on FER2013.

    python -m aura_ml.data.fer2013               # download + cache (133 MB)
    python -m aura_ml.training.train_fer --arch resnet18 --epochs 30

Standard protocol: Training (28,709) / PublicTest (3,589) for model selection /
PrivateTest (3,589) reported once. Test uses horizontal-flip TTA, as at inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from aura_ml.config import FER_MODEL_DIR, SEED
from aura_ml.data import fer2013
from aura_ml.eval.metrics import classification_report
from aura_ml.labels import FER2013_LABELS
from aura_ml.models.fer import INPUT_SIZE, MEAN, STD, build_model, preprocess
from aura_ml.training.common import (
    Timer,
    autocast_dtype,
    device,
    environment,
    seed_everything,
    write_json,
)


class GPUAugment(nn.Module):
    """Batched augmentation on the GPU (random resized crop, flip, rotation,
    brightness/contrast, random erasing) - keeps a laptop CPU out of the loop."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [N,1,48,48] in [0,1]
        n, dev = x.shape[0], x.device
        angle = (torch.rand(n, device=dev) - 0.5) * 2 * np.deg2rad(12)
        scale = 1 / (0.85 + 0.25 * torch.rand(n, device=dev))
        flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        tx, ty = (torch.rand(2, n, device=dev) - 0.5) * 0.2
        cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
        theta = torch.stack(
            [torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1
        )
        grid = nn.functional.affine_grid(theta, (n, 1, INPUT_SIZE, INPUT_SIZE), align_corners=False)
        x = nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
        b = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.5
        c = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.5
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = ((x - mean) * c + mean * b).clamp(0, 1)
        # random erasing (one rectangle on ~25% of images)
        erase = torch.rand(n, device=dev) < 0.25
        if erase.any():
            sz = INPUT_SIZE // 4
            ys, xs = torch.randint(0, INPUT_SIZE - sz, (2,), device=dev).tolist()
            x[erase, :, ys : ys + sz, xs : xs + sz] = torch.rand(1, device=dev)
        return x


def normalise(x: torch.Tensor) -> torch.Tensor:
    x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def evaluate(model, x: np.ndarray, y: np.ndarray, dev, tta: bool = True, bs: int = 512) -> dict:
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        batch = preprocess(x[i : i + bs]).to(dev)
        with torch.autocast(dev.type, dtype=autocast_dtype(), enabled=dev.type == "cuda"):
            p = model(batch).float().softmax(-1)
            if tta:
                p = (p + model(batch.flip(-1)).float().softmax(-1)) / 2
        preds += p.argmax(-1).tolist()
    return classification_report(
        [FER2013_LABELS[i] for i in preds], [FER2013_LABELS[i] for i in y], FER2013_LABELS
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arch", choices=["resnet18", "aura_cnn"], default="resnet18")
    ap.add_argument("--out-dir", type=Path, default=FER_MODEL_DIR)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    args = ap.parse_args()

    seed_everything(SEED)
    dev = device()
    data = fer2013.load()
    xtr, ytr = data["train"]
    train = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = build_model(args.arch).to(dev).to(memory_format=torch.channels_last)
    augment = GPUAugment()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.15
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best, history = -1.0, []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for xb, yb in tqdm(loader, desc=f"epoch {epoch}", leave=False):
                xb = xb.to(dev, non_blocking=True).float().div(255).unsqueeze(1)
                xb = normalise(augment(xb)).contiguous(memory_format=torch.channels_last)
                with torch.autocast(dev.type, dtype=autocast_dtype(), enabled=dev.type == "cuda"):
                    loss = criterion(model(xb), yb.to(dev))
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                sched.step()
                total += loss.item()
            val = evaluate(model, *data["validation"], dev)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / len(loader),
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": val["macro_f1"],
                }
            )
            print(
                f"epoch {epoch}: loss={history[-1]['train_loss']:.4f} val acc={val['accuracy']:.4f} macro-F1={val['macro_f1']:.4f}"
            )
            if val["accuracy"] > best:
                best = val["accuracy"]
                torch.save(model.state_dict(), args.out_dir / "model.pt")

    write_json(
        args.out_dir / "config.json",
        {"arch": args.arch, "labels": list(FER2013_LABELS), "input_size": INPUT_SIZE},
    )
    model.load_state_dict(torch.load(args.out_dir / "model.pt", weights_only=True))
    report = {
        "model": f"{args.arch} ({'ImageNet-pretrained' if args.arch == 'resnet18' else 'from scratch'})",
        "protocol": "FER2013 Training / PublicTest (selection) / PrivateTest (reported), hflip TTA",
        "hyperparameters": vars(args),
        "train_minutes": round(timer.seconds / 60, 2),
        "best_val_accuracy": best,
        "test": evaluate(model, *data["test"], dev),
        "history": history,
        "environment": environment(),
    }
    write_json(args.out_dir / "metrics.json", report)
    print(
        f"TEST accuracy={report['test']['accuracy']:.4f} macro-F1={report['test']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
