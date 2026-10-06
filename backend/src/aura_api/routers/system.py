from __future__ import annotations

import json

from fastapi import APIRouter, Request

from aura_ml.safety import DEFAULT_RESOURCES

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health(request: Request) -> dict:
    ml = request.app.state.ml
    return {"status": "ok", "models_ready": ml.ready, "mode": request.app.state.settings.ml_mode}


@router.get("/system/models")
async def models(request: Request) -> dict:
    """Which models are loaded and what each one fell back to - shown on the About page."""
    ml = request.app.state.ml
    return {
        "ready": ml.ready,
        "mode": request.app.state.settings.ml_mode,
        "components": ml.status(),
    }


@router.get("/system/resources")
async def resources() -> dict:
    return {"crisis_resources": DEFAULT_RESOURCES}


def _get(d: dict, path: str):
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


# (artifact dir, title, evaluation protocol, [(label, json path, kind)])
REPORT = [
    (
        "text_emotion",
        "Emotion from text (CPU model)",
        "GoEmotions test set, 7 emotions",
        [
            ("Accuracy", "test.accuracy", "pct"),
            ("Macro-F1", "test.macro_f1", "pct"),
            (
                "Keyword-lexicon baseline, macro-F1",
                "baselines_on_test.keyword_lexicon.macro_f1",
                "pct",
            ),
        ],
    ),
    (
        "ece",
        "Emotion-cause extraction",
        "ESConv test split, silver labels",
        [
            ("Cause span F1 (overlap)", "test.span_overlap.f1", "pct"),
            ("Cause span F1 (exact)", "test.span_exact.f1", "pct"),
            ("Has-a-cause F1", "test.clause.f1", "pct"),
        ],
    ),
    (
        "ser",
        "Emotion from voice",
        "RAVDESS, 4 speakers never heard in training",
        [
            ("Accuracy (8 classes)", "test.accuracy", "pct"),
            ("Macro-F1", "test.macro_f1", "pct"),
        ],
    ),
    (
        "fer",
        "Emotion from face",
        "FER2013 PrivateTest",
        [
            ("Accuracy (7 classes)", "test.accuracy", "pct"),
            ("Macro-F1", "test.macro_f1", "pct"),
        ],
    ),
    (
        "strategy",
        "Support-strategy choice",
        "ESConv test split",
        [
            ("Accuracy (8 strategies)", "test.accuracy", "pct"),
            ("Majority-class baseline", "baselines_on_test.majority_class.accuracy", "pct"),
        ],
    ),
    (
        "topic",
        "Topic of the conversation",
        "ESConv test split",
        [
            ("Accuracy (6 topics)", "test.accuracy", "pct"),
            ("Macro-F1", "test.macro_f1", "pct"),
        ],
    ),
    (
        "llm/aura-llama-3.2-3b-lora",
        "Reply generation (Llama 3.2 3B + LoRA)",
        "ESConv test replies",
        [
            ("Perplexity, fine-tuned", "test_reply_perplexity.aura_finetuned", "num"),
            ("Perplexity, base model", "test_reply_perplexity.base_llama_3.2_3b_instruct", "num"),
        ],
    ),
]


@router.get("/system/metrics")
async def metrics(request: Request) -> dict:
    """Measured test results of every trained model (read from artifacts/*/metrics.json)."""
    root = request.app.state.settings.artifacts_dir
    out = []
    for folder, title, protocol, fields in REPORT:
        path = root / folder / "metrics.json"
        if not path.exists():
            out.append(
                {
                    "id": folder,
                    "title": title,
                    "protocol": protocol,
                    "trained": False,
                    "metrics": [],
                }
            )
            continue
        data = json.loads(path.read_text())
        out.append(
            {
                "id": folder,
                "title": title,
                "protocol": protocol,
                "trained": True,
                "metrics": [
                    {"label": label, "value": value, "kind": kind}
                    for label, key, kind in fields
                    if (value := _get(data, key)) is not None
                ],
            }
        )
    return {"models": out}
