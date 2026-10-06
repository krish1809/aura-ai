"""Train the support-strategy predictor on ESConv (CPU, about a minute).

python -m aura_ml.training.train_strategy
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib

from aura_ml.config import ESCONV_DIR, STRATEGY_MODEL_DIR
from aura_ml.data.esconv import Conversation, load_all
from aura_ml.eval.metrics import classification_report
from aura_ml.io import write_json
from aura_ml.labels import STRATEGIES
from aura_ml.models.strategy import StrategyContext, build_pipeline


def examples(convs: list[Conversation]) -> tuple[list[dict], list[str]]:
    """One example per supporter turn that answers a seeker turn."""
    rows, labels = [], []
    for conv in convs:
        prev_reply, prev_strategy, seeker_turns = "", "none", 0
        for i, turn in enumerate(conv.turns):
            if turn.speaker == "seeker":
                seeker_turns += 1
                continue
            if i > 0 and conv.turns[i - 1].speaker == "seeker" and turn.strategy in STRATEGIES:
                rows.append(
                    StrategyContext(
                        seeker_text=conv.turns[i - 1].text,
                        prev_supporter_text=prev_reply,
                        prev_strategy=prev_strategy,
                        turn_index=seeker_turns,
                        emotion=conv.emotion_type,
                        problem_type=conv.problem_type,
                    ).as_row()
                )
                labels.append(turn.strategy)
            prev_reply, prev_strategy = turn.text, turn.strategy or "none"
    return rows, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--esconv-dir", type=Path, default=ESCONV_DIR)
    ap.add_argument("--out-dir", type=Path, default=STRATEGY_MODEL_DIR)
    args = ap.parse_args()

    data = {s: examples(c) for s, c in load_all(args.esconv_dir).items()}
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["validation"], data["test"]
    print({s: len(v[1]) for s, v in data.items()})

    best_c, best_f1 = None, -1.0
    for c in (0.5, 1.0, 2.0, 4.0, 8.0):
        model = build_pipeline(C=c).fit(xtr, ytr)
        f1 = classification_report(model.predict(xva), yva, STRATEGIES)["macro_f1"]
        print(f"C={c}: val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1

    model = build_pipeline(C=best_c).fit(xtr + xva, ytr + yva)
    test = classification_report(model.predict(xte), yte, STRATEGIES)
    majority = Counter(ytr).most_common(1)[0][0]
    baselines = {
        "majority_class": classification_report([majority] * len(yte), yte, STRATEGIES),
        "repeat_previous_strategy": classification_report(
            [r["prev_strategy"] if r["prev_strategy"] in STRATEGIES else majority for r in xte],
            yte,
            STRATEGIES,
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "model.joblib", compress=3)
    write_json(
        args.out_dir / "metrics.json",
        {
            "model": "TF-IDF (seeker msg, previous reply) + state features -> logistic regression",
            "selected_C": best_c,
            "val_macro_f1": best_f1,
            "test": test,
            "baselines_on_test": {
                k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                for k, v in baselines.items()
            },
        },
    )
    print(
        f"TEST accuracy={test['accuracy']:.4f} macro-F1={test['macro_f1']:.4f} | "
        + " | ".join(
            f"{k}: acc={v['accuracy']:.3f} F1={v['macro_f1']:.3f}" for k, v in baselines.items()
        )
    )


if __name__ == "__main__":
    main()
