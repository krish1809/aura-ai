"""Train the two-head RoBERTa emotion-cause extractor.

    python -m aura_ml.data.ece_dataset          # build silver-labelled data first
    python -m aura_ml.training.train_ece --epochs 8

Model selection uses validation span-overlap F1; the chosen checkpoint is then
scored once on the test split. Results land in artifacts/ece/metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from aura_ml.config import ARTIFACTS_DIR, ECE_DATA_DIR, ECE_MODEL_DIR, HF_ECE_BASE, SEED
from aura_ml.data.bio import IGNORE_INDEX, align_bio, decode_bio
from aura_ml.data.ece_dataset import load
from aura_ml.eval.metrics import binary_prf, span_exact_prf, span_overlap_prf
from aura_ml.models.ece import RoBERTaForECE
from aura_ml.training.common import (
    Timer,
    autocast_dtype,
    device,
    environment,
    seed_everything,
    write_json,
)


def encode(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    enc = tokenizer(
        [r["text"] for r in rows],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    items = []
    for i, row in enumerate(rows):
        offsets = [tuple(o) for o in enc["offset_mapping"][i]]
        items.append(
            {
                "input_ids": enc["input_ids"][i],
                "offsets": offsets,
                "token_labels": align_bio(offsets, [tuple(s) for s in row["spans"]]),
                "clause_label": int(row["has_cause"]),
                "gold_spans": [tuple(s) for s in row["spans"]],
            }
        )
    return items


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE_INDEX)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        mask[i, :k] = 1
        labels[i, :k] = torch.tensor(b["token_labels"])
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "token_labels": labels,
        "clause_labels": torch.tensor([b["clause_label"] for b in batch]),
        "offsets": [b["offsets"] for b in batch],
        "gold_spans": [b["gold_spans"] for b in batch],
    }


@torch.inference_mode()
def evaluate(model, loader, dev, threshold: float = 0.5) -> dict:
    model.eval()
    pred_spans, gold_spans, pred_has, gold_has = [], [], [], []
    for batch in loader:
        with torch.autocast(dev.type, dtype=autocast_dtype(), enabled=dev.type == "cuda"):
            out = model(
                input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev)
            )
        clause_p = out.clause_logits.float().softmax(-1)[:, 1].tolist()
        tok = out.token_logits.argmax(-1).tolist()
        for i, offsets in enumerate(batch["offsets"]):
            spans = decode_bio(tok[i][: len(offsets)], offsets) if clause_p[i] >= threshold else []
            pred_spans.append(spans)
            pred_has.append(bool(spans))
        gold_spans.extend(batch["gold_spans"])
        gold_has.extend(bool(s) for s in batch["gold_spans"])
    return {
        "span_exact": span_exact_prf(pred_spans, gold_spans),
        "span_overlap": span_overlap_prf(pred_spans, gold_spans),
        "clause": binary_prf(pred_has, gold_has),
        "n": len(gold_has),
        "n_with_cause": sum(gold_has),
    }


THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)


def calibrate(model, val_loader, dev) -> tuple[float, dict]:
    """Pick the has-cause threshold that maximises validation span-overlap F1.

    Training uses a 1:2 cause/no-cause ratio but real conversations are closer to
    1:12, so the default 0.5 threshold over-predicts causes."""
    sweep = {t: evaluate(model, val_loader, dev, threshold=t) for t in THRESHOLDS}
    best = max(sweep, key=lambda t: sweep[t]["span_overlap"]["f1"])
    return best, {str(t): r["span_overlap"]["f1"] for t, r in sweep.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=ECE_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=ECE_MODEL_DIR)
    ap.add_argument("--base", default=HF_ECE_BASE)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument(
        "--compare-legacy", type=Path, default=ARTIFACTS_DIR / "legacy" / "ece_roberta_v1"
    )
    ap.add_argument(
        "--calibrate-only",
        action="store_true",
        help="skip training; re-run threshold calibration and test on the saved model",
    )
    args = ap.parse_args()

    seed_everything(SEED)
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    data = {
        s: encode(load(s, args.data_dir), tokenizer, args.max_length)
        for s in ("train", "validation", "test")
    }

    def loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            data[split],
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        )

    train_loader, val_loader, test_loader = (
        loader("train", True),
        loader("validation", False),
        loader("test", False),
    )

    # B/I tokens are rare next to O; up-weight them so the span head does not collapse to all-O.
    counts = torch.zeros(3)
    for item in data["train"]:
        for lab in item["token_labels"]:
            if lab != IGNORE_INDEX:
                counts[lab] += 1
    class_weights = (counts.sum() / (3 * counts.clamp(min=1))).clamp(max=10.0).to(dev)

    model = RoBERTaForECE.from_pretrained(args.base).to(dev)
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(params, lr=args.lr)
    total = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup * total), total)

    best_f1, history = -1.0, []
    if args.calibrate_only:
        previous = json.loads((args.out_dir / "metrics.json").read_text())
        best_f1, history = previous["best_val_span_overlap_f1"], previous["history"]
    with Timer() as timer:
        for epoch in range(1, 0 if args.calibrate_only else args.epochs + 1):
            model.train()
            running = 0.0
            for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                with torch.autocast(dev.type, dtype=autocast_dtype(), enabled=dev.type == "cuda"):
                    out = model(
                        input_ids=batch["input_ids"].to(dev),
                        attention_mask=batch["attention_mask"].to(dev),
                        clause_labels=batch["clause_labels"].to(dev),
                        token_labels=batch["token_labels"].to(dev),
                        token_class_weights=class_weights,
                    )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                running += out.loss.item()
            val = evaluate(model, val_loader, dev)
            f1 = val["span_overlap"]["f1"]
            history.append({"epoch": epoch, "train_loss": running / len(train_loader), "val": val})
            print(
                f"epoch {epoch}: loss={running / len(train_loader):.4f} "
                f"val overlap-F1={f1:.4f} exact-F1={val['span_exact']['f1']:.4f} clause-F1={val['clause']['f1']:.4f}"
            )
            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained(args.out_dir)
                tokenizer.save_pretrained(args.out_dir)

    best = RoBERTaForECE.from_pretrained(args.out_dir).to(dev)
    threshold, sweep = calibrate(best, val_loader, dev)
    write_json(args.out_dir / "inference.json", {"clause_threshold": threshold})
    report = {
        "model": "RoBERTaForECE (roberta-base, clause head + BIO span head)",
        "data": "ESConv seeker turns, silver labels from aura_ml.data.causes",
        "hyperparameters": vars(args) | {"token_class_weights": class_weights.tolist()},
        "train_minutes": previous["train_minutes"]
        if args.calibrate_only
        else round(timer.seconds / 60, 2),
        "best_val_span_overlap_f1": best_f1,
        "clause_threshold": threshold,
        "threshold_sweep_val_overlap_f1": sweep,
        "test": evaluate(best, test_loader, dev, threshold=threshold),
        "test_at_threshold_0.5": evaluate(best, test_loader, dev, threshold=0.5),
        "history": history,
        "environment": environment(),
    }
    if args.compare_legacy and (args.compare_legacy / "model.safetensors").exists():
        legacy = RoBERTaForECE.from_pretrained(args.compare_legacy).to(dev)
        report["legacy_v1_on_same_test"] = evaluate(legacy, test_loader, dev)
    write_json(args.out_dir / "metrics.json", {k: v for k, v in report.items()})
    t = report["test"]
    print(
        f"TEST (threshold {threshold}) overlap-F1={t['span_overlap']['f1']:.4f} exact-F1={t['span_exact']['f1']:.4f} "
        f"clause-F1={t['clause']['f1']:.4f} clause-acc={t['clause']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
