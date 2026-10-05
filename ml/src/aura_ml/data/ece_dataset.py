"""Build the emotion-cause extraction (ECE) dataset from ESConv seeker turns.

Output: data/processed/ece/{train,validation,test}.jsonl with one utterance per
line, its silver cause spans (character offsets) and conversation metadata.
Splits follow the official ESConv conversation split, so no conversation is
shared between train and test.

    python -m aura_ml.data.ece_dataset
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from aura_ml.config import ECE_DATA_DIR, ESCONV_DIR, SEED
from aura_ml.data.causes import extract_causes, is_phatic
from aura_ml.data.esconv import load_all
from aura_ml.labels import ESCONV_TO_CANONICAL, to_canonical


def build(
    esconv_dir: Path = ESCONV_DIR, out_dir: Path = ECE_DATA_DIR, neg_ratio: float = 2.0
) -> dict:
    """`neg_ratio` caps train negatives at neg_ratio x positives; val/test keep the
    natural distribution so reported metrics reflect real usage."""
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    for split, convs in load_all(esconv_dir).items():
        pos, neg = [], []
        for conv in convs:
            for i, turn in enumerate(conv.turns):
                if turn.speaker != "seeker" or is_phatic(turn.text):
                    continue
                spans = extract_causes(turn.text)
                row = {
                    "id": f"{conv.id}-t{i:02d}",
                    "conversation_id": conv.id,
                    "text": turn.text,
                    "emotion": to_canonical(conv.emotion_type, ESCONV_TO_CANONICAL),
                    "esconv_emotion": conv.emotion_type,
                    "problem_type": conv.problem_type,
                    "has_cause": bool(spans),
                    "spans": [[s.start, s.end] for s in spans],
                    "causes": [s.text for s in spans],
                    "rules": [s.rule for s in spans],
                }
                (pos if spans else neg).append(row)
        if split == "train" and len(neg) > neg_ratio * len(pos):
            neg = rng.sample(neg, int(neg_ratio * len(pos)))
        rows = pos + neg
        rng.shuffle(rows)
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = {
            "utterances": len(rows),
            "with_cause": len(pos),
            "without_cause": len(neg),
            "spans": sum(len(r["spans"]) for r in pos),
            "rules": dict(Counter(rule for r in pos for rule in r["rules"])),
        }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load(split: str, data_dir: Path = ECE_DATA_DIR) -> list[dict]:
    with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--esconv-dir", type=Path, default=ESCONV_DIR)
    ap.add_argument("--out-dir", type=Path, default=ECE_DATA_DIR)
    ap.add_argument("--neg-ratio", type=float, default=2.0)
    args = ap.parse_args()
    print(json.dumps(build(args.esconv_dir, args.out_dir, args.neg_ratio), indent=2))


if __name__ == "__main__":
    main()
