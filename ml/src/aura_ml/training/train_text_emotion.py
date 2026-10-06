"""Train the CPU text-emotion classifier on GoEmotions (Ekman 7 classes).

    python -m aura_ml.training.train_text_emotion

TF-IDF (word + character n-grams) -> class-balanced logistic regression. Small
and fast enough for a GPU-free deployment; full mode uses a DistilRoBERTa model.
The keyword lexicon it replaces is scored on the same test set for comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from aura_ml.config import ARTIFACTS_DIR
from aura_ml.data import goemotions
from aura_ml.eval.metrics import classification_report
from aura_ml.io import write_json
from aura_ml.labels import EMOTIONS
from aura_ml.models.text_emotion import LexiconEmotionClassifier


def build_pipeline(C: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "words",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000
                ),
            ),
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=3,
                    sublinear_tf=True,
                    max_features=80000,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "text_emotion")
    args = ap.parse_args()

    data = {s: goemotions.load(s) for s in goemotions.SPLITS}
    print({s: len(v[0]) for s, v in data.items()})
    best_c, best_f1 = None, -1.0
    for c in (1.0, 3.0, 10.0):
        model = build_pipeline(c).fit(*data["train"])
        f1 = classification_report(
            model.predict(data["validation"][0]), data["validation"][1], EMOTIONS
        )["macro_f1"]
        print(f"C={c}: val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1
    model = build_pipeline(best_c).fit(
        data["train"][0] + data["validation"][0], data["train"][1] + data["validation"][1]
    )
    texts, gold = data["test"]
    test = classification_report(list(model.predict(texts)), gold, EMOTIONS)
    lexicon = LexiconEmotionClassifier()
    baseline = classification_report([lexicon.predict(t)["emotion"] for t in texts], gold, EMOTIONS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "model.joblib", compress=3)
    write_json(
        args.out_dir / "metrics.json",
        {
            "model": "TF-IDF (word 1-2 + char 2-5 grams) -> class-balanced logistic regression",
            "data": "GoEmotions, official Ekman mapping, single-category comments only",
            "selected_C": best_c,
            "val_macro_f1": best_f1,
            "test": test,
            "baselines_on_test": {
                "keyword_lexicon": {
                    "accuracy": baseline["accuracy"],
                    "macro_f1": baseline["macro_f1"],
                }
            },
        },
    )
    print(
        f"TEST accuracy={test['accuracy']:.4f} macro-F1={test['macro_f1']:.4f} | "
        f"lexicon acc={baseline['accuracy']:.4f} macro-F1={baseline['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
