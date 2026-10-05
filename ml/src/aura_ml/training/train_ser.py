"""Fine-tune wav2vec2 for speech emotion recognition on RAVDESS.

    python -m aura_ml.training.train_ser --epochs 20

Speaker-independent protocol: actors 1-18 train, 19-20 validation, 21-24 test
(two male + two female test speakers who are never heard during training). The
2025 version split files at random, so the same actors appeared in train and
test and its accuracy was not a fair estimate.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_cosine_schedule_with_warmup,
)

from aura_ml.audio import SAMPLE_RATE, load_audio, trim_silence
from aura_ml.config import ARTIFACTS_DIR, HF_SER_BASE, RAVDESS_DIR, SEED, SER_MODEL_DIR
from aura_ml.eval.metrics import classification_report
from aura_ml.labels import RAVDESS_CODES
from aura_ml.training.common import (
    Timer,
    autocast_dtype,
    device,
    environment,
    seed_everything,
    write_json,
)

LABELS = sorted(set(RAVDESS_CODES.values()))
SPLITS = {"train": range(1, 19), "validation": range(19, 21), "test": range(21, 25)}


def index_ravdess(root: Path) -> dict[str, list[tuple[Path, str]]]:
    out: dict[str, list[tuple[Path, str]]] = {k: [] for k in SPLITS}
    for path in sorted(root.rglob("*.wav")):
        parts = path.stem.split(
            "-"
        )  # modality-channel-emotion-intensity-statement-repetition-actor
        if len(parts) != 7 or parts[0] != "03":
            continue
        actor = int(parts[6])
        for split, actors in SPLITS.items():
            if actor in actors:
                out[split].append((path, RAVDESS_CODES[parts[2]]))
    return out


class ClipDataset(Dataset):
    def __init__(self, items: list[tuple[Path, str]], train: bool, seconds: float):
        self.wavs = [
            trim_silence(load_audio(p)) for p, _ in tqdm(items, desc="loading audio", leave=False)
        ]
        self.labels = [LABELS.index(label) for _, label in items]
        self.train, self.n = train, int(seconds * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.wavs)

    def __getitem__(self, i: int):
        wav = self.wavs[i]
        if self.train:
            if len(wav) > self.n:
                start = random.randint(0, len(wav) - self.n)
                wav = wav[start : start + self.n]
            wav = wav * random.uniform(0.6, 1.4)
            if random.random() < 0.5:
                wav = wav + np.random.normal(0, random.uniform(0.001, 0.01), len(wav)).astype(
                    np.float32
                )
        else:
            wav = wav[: self.n]
        return wav.astype(np.float32), self.labels[i]


def make_collate(extractor):
    # wav2vec2-base checkpoints were trained without attention masks (zero padding
    # instead); only models whose extractor asks for a mask should receive one.
    use_mask = bool(getattr(extractor, "return_attention_mask", False))

    def collate(batch):
        wavs, labels = zip(*batch, strict=True)
        enc = extractor(list(wavs), sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
        mask = enc.get("attention_mask") if use_mask else None
        return enc["input_values"], mask, torch.tensor(labels)

    return collate


def _mask(mask, dev):
    return mask.to(dev) if mask is not None else None


@torch.inference_mode()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    preds, golds = [], []
    for x, mask, y in loader:
        with torch.autocast(dev.type, dtype=autocast_dtype(), enabled=dev.type == "cuda"):
            logits = model(x.to(dev), attention_mask=_mask(mask, dev)).logits
        preds += [LABELS[i] for i in logits.argmax(-1).tolist()]
        golds += [LABELS[i] for i in y.tolist()]
    return classification_report(preds, golds, LABELS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=RAVDESS_DIR)
    ap.add_argument("--out-dir", type=Path, default=SER_MODEL_DIR)
    ap.add_argument("--base", default=HF_SER_BASE)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--legacy", type=Path, default=ARTIFACTS_DIR / "legacy" / "ser_wav2vec2_v1")
    args = ap.parse_args()

    seed_everything(SEED)
    dev = device()
    index = index_ravdess(args.data_dir)
    print({k: len(v) for k, v in index.items()})
    extractor = AutoFeatureExtractor.from_pretrained(args.base)
    collate = make_collate(extractor)
    sets = {s: ClipDataset(index[s], s == "train", args.seconds) for s in SPLITS}
    loaders = {
        s: DataLoader(
            ds, batch_size=args.batch_size, shuffle=s == "train", collate_fn=collate, num_workers=0
        )
        for s, ds in sets.items()
    }

    model = AutoModelForAudioClassification.from_pretrained(
        args.base,
        num_labels=len(LABELS),
        label2id={lab: i for i, lab in enumerate(LABELS)},
        id2label=dict(enumerate(LABELS)),
        ignore_mismatched_sizes=True,
    ).to(dev)
    model.freeze_feature_encoder()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    steps = len(loaders["train"]) * args.epochs // args.grad_accum
    sched = get_cosine_schedule_with_warmup(optim, int(0.1 * steps), steps)

    best, history = -1.0, []
    with Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for step, (x, mask, y) in enumerate(
                tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False), 1
            ):
                with torch.autocast(dev.type, dtype=autocast_dtype(), enabled=dev.type == "cuda"):
                    loss = (
                        model(x.to(dev), attention_mask=_mask(mask, dev), labels=y.to(dev)).loss
                        / args.grad_accum
                    )
                loss.backward()
                total += loss.item()
                if step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
            val = evaluate(model, loaders["validation"], dev)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / len(loaders["train"]),
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": val["macro_f1"],
                }
            )
            print(
                f"epoch {epoch}: loss={history[-1]['train_loss']:.4f} val acc={val['accuracy']:.4f} macro-F1={val['macro_f1']:.4f}"
            )
            if val["macro_f1"] > best:
                best = val["macro_f1"]
                model.save_pretrained(args.out_dir)
                extractor.save_pretrained(args.out_dir)

    best_model = AutoModelForAudioClassification.from_pretrained(args.out_dir).to(dev)
    report = {
        "model": f"{args.base} fine-tuned (CNN feature encoder frozen)",
        "protocol": "RAVDESS speech, speaker-independent: actors 1-18 / 19-20 / 21-24",
        "hyperparameters": vars(args),
        "train_minutes": round(timer.seconds / 60, 2),
        "best_val_macro_f1": best,
        "test": evaluate(best_model, loaders["test"], dev),
        "history": history,
        "environment": environment(),
    }
    if (args.legacy / "config.json").exists():
        legacy = AutoModelForAudioClassification.from_pretrained(args.legacy).to(dev)
        if [legacy.config.id2label[i] for i in range(legacy.config.num_labels)] == LABELS:
            report["legacy_v1_on_same_test"] = evaluate(legacy, loaders["test"], dev)
    write_json(args.out_dir / "metrics.json", report)
    print(
        f"TEST accuracy={report['test']['accuracy']:.4f} macro-F1={report['test']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
