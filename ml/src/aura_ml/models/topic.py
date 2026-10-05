"""Problem-topic classifier (ESConv `problem_type`) over what the user has said so far."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MAIN_TOPICS = (
    "ongoing depression",
    "job crisis",
    "breakup with partner",
    "problems with friends",
    "academic pressure",
)
OTHER = "other"


def normalise_topic(problem_type: str) -> str:
    return problem_type if problem_type in MAIN_TOPICS else OTHER


def build_pipeline(C: float = 4.0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=C, max_iter=3000)),
        ]
    )


class TopicClassifier:
    def __init__(self, model_path: str | Path):
        self.pipeline: Pipeline = joblib.load(model_path)
        self.classes = list(self.pipeline.classes_)

    def predict(self, user_text_so_far: str) -> dict:
        probs = self.pipeline.predict_proba([user_text_so_far])[0]
        dist = {c: round(float(p), 4) for c, p in zip(self.classes, probs, strict=True)}
        top = max(dist, key=dist.get)
        return {"topic": top, "confidence": dist[top], "probs": dist}
