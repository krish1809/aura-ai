"""Label spaces and the mappings between them.

Every modality is projected onto one canonical 7-way emotion space (Ekman's six
basic emotions + neutral) so text, voice and face predictions can be fused.
"""

from __future__ import annotations

EMOTIONS: tuple[str, ...] = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")

# RAVDESS filename code -> its own label (3rd field of 03-01-06-01-02-01-12.wav)
RAVDESS_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
RAVDESS_TO_CANONICAL = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": "surprise",
}

# FER2013 integer labels
FER2013_LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
FER2013_TO_CANONICAL = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}

# ESConv `emotion_type` (conversation level) -> canonical. The fine-grained
# ESConv label is kept as well because "anxiety" says more to the LLM than "fear".
ESCONV_TO_CANONICAL = {
    "anxiety": "fear",
    "nervousness": "fear",
    "fear": "fear",
    "depression": "sadness",
    "sadness": "sadness",
    "pain": "sadness",
    "shame": "sadness",
    "guilt": "sadness",
    "anger": "anger",
    "jealousy": "anger",
    "disgust": "disgust",
}

# Canonical -> the ESConv label the strategy model saw most often for that family.
CANONICAL_TO_ESCONV = {
    "fear": "anxiety",
    "sadness": "sadness",
    "anger": "anger",
    "disgust": "disgust",
}

STRATEGIES: tuple[str, ...] = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
)

STRATEGY_DESCRIPTIONS = {
    "Question": "ask a gentle, open-ended question to understand their situation better",
    "Restatement or Paraphrasing": "briefly restate what they said so they feel heard",
    "Reflection of feelings": "name and validate the feeling they are expressing",
    "Self-disclosure": "share a short, relatable experience to show they are not alone",
    "Affirmation and Reassurance": "affirm their strengths and offer reassurance",
    "Providing Suggestions": "offer one practical, concrete suggestion",
    "Information": "share useful, factual information relevant to their problem",
    "Others": "offer warm, general support",
}


def to_canonical(label: str, mapping: dict[str, str]) -> str:
    """Map a source label onto the canonical space; unknown labels become neutral."""
    return mapping.get(label.lower().strip(), "neutral")


def project_distribution(probs: dict[str, float], mapping: dict[str, str]) -> dict[str, float]:
    """Sum a source-label distribution into the canonical emotion space."""
    out = dict.fromkeys(EMOTIONS, 0.0)
    for label, p in probs.items():
        out[to_canonical(label, mapping)] += float(p)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else out
