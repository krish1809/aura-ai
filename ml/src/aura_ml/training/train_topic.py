"""Train the problem-topic classifier on ESConv (CPU, seconds).

    python -m aura_ml.training.train_topic

Examples are cumulative: after each seeker turn, everything the seeker has said
so far is one example, so the model also learns to guess early in a conversation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from aura_ml.config import ARTIFACTS_DIR, ESCONV_DIR
from aura_ml.data.esconv import Conversation, load_all
from aura_ml.eval.metrics import classification_report
from aura_ml.io import write_json
from aura_ml.models.topic import MAIN_TOPICS, OTHER, build_pipeline, normalise_topic

LABELS = (*MAIN_TOPICS, OTHER)


def examples(
    convs: list[Conversation], max_turns: int = 8
) -> tuple[list[str], list[str], list[int]]:
    texts, labels, turns = [], [], []
    for conv in convs:
        so_far: list[str] = []
        for turn in conv.turns:
            if turn.speaker != "seeker":
                continue
            so_far.append(turn.text)
            if len(so_far) > max_turns:
                break
            texts.append(" ".join(so_far))
            labels.append(normalise_topic(conv.problem_type))
            turns.append(len(so_far))
    return texts, labels, turns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--esconv-dir", type=Path, default=ESCONV_DIR)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "topic")
    args = ap.parse_args()

    data = {s: examples(c) for s, c in load_all(args.esconv_dir).items()}
    best_c, best_f1 = None, -1.0
    for c in (1.0, 4.0, 16.0):
        model = build_pipeline(C=c).fit(data["train"][0], data["train"][1])
        f1 = classification_report(
            model.predict(data["validation"][0]), data["validation"][1], LABELS
        )["macro_f1"]
        print(f"C={c}: val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1
    model = build_pipeline(C=best_c).fit(
        data["train"][0] + data["validation"][0], data["train"][1] + data["validation"][1]
    )
    preds = model.predict(data["test"][0])
    test = classification_report(preds, data["test"][1], LABELS)
    # accuracy after k seeker turns: how quickly the topic becomes clear
    by_turn = {}
    for k in (1, 2, 3, 5, 8):
        hits = [
            p == g
            for p, g, t in zip(preds, data["test"][1], data["test"][2], strict=True)
            if t == k
        ]
        by_turn[k] = round(sum(hits) / len(hits), 4) if hits else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "model.joblib", compress=3)
    write_json(
        args.out_dir / "metrics.json",
        {
            "model": "TF-IDF (all seeker text so far) -> logistic regression",
            "labels": list(LABELS),
            "selected_C": best_c,
            "val_macro_f1": best_f1,
            "test": test,
            "test_accuracy_after_k_user_turns": by_turn,
        },
    )
    print(
        f"TEST accuracy={test['accuracy']:.4f} macro-F1={test['macro_f1']:.4f} by turn: {by_turn}"
    )


if __name__ == "__main__":
    main()
