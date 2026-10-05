"""Support-strategy predictor: which of ESConv's 8 strategies should Aura use next?

A TF-IDF + logistic-regression model over the seeker's message, Aura's previous
reply, the previous strategy and coarse conversation state. It is small, fast on
CPU and its choice is shown in the UI, so it doubles as an explanation of *why*
Aura answered the way it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from aura_ml.labels import STRATEGIES


@dataclass
class StrategyContext:
    seeker_text: str
    prev_supporter_text: str = ""
    prev_strategy: str = "none"
    turn_index: int = 0  # number of seeker turns so far (1-based for the current one)
    emotion: str = ""
    problem_type: str = ""

    def as_row(self) -> dict:
        return self.__dict__


def position_bucket(turn_index: int) -> str:
    return (
        "opening"
        if turn_index <= 2
        else "exploring"
        if turn_index <= 6
        else "comforting"
        if turn_index <= 12
        else "late"
    )


class FieldText(BaseEstimator, TransformerMixin):
    def __init__(self, key: str):
        self.key = key

    def fit(self, rows, y=None):
        return self

    def transform(self, rows):
        return [r[self.key] or "" for r in rows]


class FieldCategorical(BaseEstimator, TransformerMixin):
    def fit(self, rows, y=None):
        return self

    def transform(self, rows):
        return [
            {
                "prev": r["prev_strategy"] or "none",
                "pos": position_bucket(r["turn_index"]),
                "emo": r["emotion"],
                "prob": r["problem_type"],
                "q": "?" in (r["seeker_text"] or ""),
            }
            for r in rows
        ]


def build_pipeline(C: float = 2.0) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "seeker",
                Pipeline(
                    [
                        ("f", FieldText("seeker_text")),
                        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                    ]
                ),
            ),
            (
                "prev_reply",
                Pipeline(
                    [
                        ("f", FieldText("prev_supporter_text")),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000
                            ),
                        ),
                    ]
                ),
            ),
            ("state", Pipeline([("f", FieldCategorical()), ("dv", DictVectorizer())])),
        ]
    )
    return Pipeline([("features", features), ("clf", LogisticRegression(C=C, max_iter=3000))])


class StrategyPredictor:
    def __init__(self, model_path: str | Path):
        self.pipeline: Pipeline = joblib.load(model_path)
        self.classes = list(self.pipeline.classes_)

    def predict(self, ctx: StrategyContext) -> dict:
        probs = self.pipeline.predict_proba([ctx.as_row()])[0]
        dist = {c: round(float(p), 4) for c, p in zip(self.classes, probs, strict=True)}
        top = max(dist, key=dist.get)
        return {
            "strategy": top,
            "confidence": dist[top],
            "probs": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
        }


__all__ = [
    "STRATEGIES",
    "StrategyContext",
    "StrategyPredictor",
    "build_pipeline",
    "position_bucket",
]
