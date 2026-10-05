"""Model hub + per-turn analysis: the "Sense -> Understand" half of Aura.

`ModelHub` loads every component it can and records why anything is missing, so
the app degrades gracefully instead of crashing:

    component      trained model                    fallback
    text_emotion   DistilRoBERTa (7 emotions)       keyword lexicon
    ece            Aura RoBERTa two-head ECE        the silver-label rules
    ner            spaCy + relation patterns        relation patterns
    topic/strategy scikit-learn models (CPU)        none
    stt/ser/fer    Whisper / wav2vec2 / ResNet-18   modality unavailable
    llm            Llama 3.2 3B + Aura LoRA         OpenAI-compatible server or mock
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aura_ml import config
from aura_ml.audio import prosody
from aura_ml.data.causes import extract_causes
from aura_ml.fusion import fuse
from aura_ml.labels import CANONICAL_TO_ESCONV
from aura_ml.models.llm import LLM, MockLLM, OpenAICompatLLM
from aura_ml.models.ner import EntityExtractor
from aura_ml.models.strategy import StrategyContext
from aura_ml.models.text_emotion import LexiconEmotionClassifier
from aura_ml.prompts import PromptContext
from aura_ml.safety import ELEVATED_INSTRUCTION, assess

log = logging.getLogger("aura_ml.hub")


class RuleCauseExtractor:
    source = "rules"

    def predict(self, text: str) -> dict:
        spans = extract_causes(text)
        return {
            "has_cause": bool(spans),
            "clause_confidence": 1.0 if spans else 0.0,
            "causes": [
                {"text": s.text, "start": s.start, "end": s.end, "confidence": None} for s in spans
            ],
            "source": self.source,
        }


@dataclass
class HubConfig:
    artifacts_dir: Path = config.ARTIFACTS_DIR
    use_torch: bool = True
    device: str | None = None
    llm_provider: str = "local"  # local | openai | mock
    llm_base: str = config.HF_LLM_BASE
    llm_adapter: Path = config.LLM_ADAPTER_DIR
    openai_base_url: str = "http://localhost:11434/v1"
    openai_model: str = "aura"
    openai_api_key: str | None = None
    stt_model: str = config.HF_STT
    text_emotion_model: str = config.HF_TEXT_EMOTION
    components: tuple[str, ...] = (
        "text_emotion",
        "ece",
        "ner",
        "topic",
        "strategy",
        "stt",
        "ser",
        "fer",
        "llm",
    )


@dataclass
class ModelHub:
    cfg: HubConfig = field(default_factory=HubConfig)
    status: dict[str, dict] = field(default_factory=dict)

    text_emotion: object = None
    ece: object = None
    ner: EntityExtractor | None = None
    topic: object = None
    strategy: object = None
    stt: object = None
    ser: object = None
    fer: object = None
    llm: LLM | None = None

    def load(self) -> ModelHub:
        for name in self.cfg.components:
            t0 = time.perf_counter()
            try:
                source = getattr(self, f"_load_{name}")()
                self.status[name] = {
                    "ready": source is not None,
                    "source": source,
                    "load_s": round(time.perf_counter() - t0, 2),
                }
            except Exception as exc:  # a missing model must never take the whole app down
                log.warning("component %s unavailable: %s", name, exc)
                self.status[name] = {
                    "ready": False,
                    "source": None,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                self._fallback(name)
        return self

    # -- loaders: return a short source label, or None when the component is off ----------------
    def _art(self, *parts: str) -> Path:
        return Path(self.cfg.artifacts_dir, *parts)

    def _load_text_emotion(self):
        if self.cfg.use_torch:
            from aura_ml.models.text_emotion import TextEmotionClassifier

            self.text_emotion = TextEmotionClassifier(self.cfg.text_emotion_model, self.cfg.device)
            return "distilroberta"
        self.text_emotion = LexiconEmotionClassifier()
        return "lexicon"

    def _load_ece(self):
        if self.cfg.use_torch and (self._art("ece") / "model.safetensors").exists():
            from aura_ml.models.ece import EmotionCauseExtractor

            self.ece = EmotionCauseExtractor(self._art("ece"), self.cfg.device)
            return "roberta-ece"
        self.ece = RuleCauseExtractor()
        return "rules"

    def _load_ner(self):
        self.ner = EntityExtractor(config.SPACY_MODEL)
        return self.ner.source

    def _load_topic(self):
        path = self._art("topic", "model.joblib")
        if not path.exists():
            return None
        from aura_ml.models.topic import TopicClassifier

        self.topic = TopicClassifier(path)
        return "tfidf-logreg"

    def _load_strategy(self):
        path = self._art("strategy", "model.joblib")
        if not path.exists():
            return None
        from aura_ml.models.strategy import StrategyPredictor

        self.strategy = StrategyPredictor(path)
        return "tfidf-logreg"

    def _load_stt(self):
        if not self.cfg.use_torch:
            return None
        from aura_ml.models.stt import WhisperSTT

        self.stt = WhisperSTT(self.cfg.stt_model, self.cfg.device)
        return self.cfg.stt_model

    def _load_ser(self):
        if not (self.cfg.use_torch and (self._art("ser") / "config.json").exists()):
            return None
        from aura_ml.models.ser import SpeechEmotionRecognizer

        self.ser = SpeechEmotionRecognizer(self._art("ser"), self.cfg.device)
        return "wav2vec2-ravdess"

    def _load_fer(self):
        if not (self.cfg.use_torch and (self._art("fer") / "model.pt").exists()):
            return None
        from aura_ml.models.fer import FacialEmotionRecognizer

        self.fer = FacialEmotionRecognizer(self._art("fer"), self.cfg.device)
        return "resnet18-fer2013"

    def _load_llm(self):
        provider = self.cfg.llm_provider
        if provider == "local" and self.cfg.use_torch:
            from aura_ml.models.llm import LocalLLM

            self.llm = LocalLLM(
                self.cfg.llm_base, self.cfg.llm_adapter, device=self.cfg.device or "cuda"
            )
            return self.llm.name
        if provider == "openai":
            self.llm = OpenAICompatLLM(
                self.cfg.openai_base_url, self.cfg.openai_model, self.cfg.openai_api_key
            )
            return self.llm.name
        self.llm = MockLLM()
        return "mock"

    def _fallback(self, name: str) -> None:
        if name == "text_emotion":
            self.text_emotion = LexiconEmotionClassifier()
            self.status[name] |= {"ready": True, "source": "lexicon (fallback)"}
        elif name == "ece":
            self.ece = RuleCauseExtractor()
            self.status[name] |= {"ready": True, "source": "rules (fallback)"}
        elif name == "ner":
            self.ner = EntityExtractor(None)
            self.status[name] |= {"ready": True, "source": "patterns (fallback)"}
        elif name == "llm":
            self.llm = MockLLM()
            self.status[name] |= {"ready": True, "source": "mock (fallback)"}


@dataclass
class TurnState:
    """What the analyzer needs to know about the conversation so far."""

    user_text_so_far: str = ""
    prev_reply: str = ""
    prev_strategy: str = "none"
    turn_index: int = 1
    last_cause: str | None = None


def _timed(timings: dict, key: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        timings[key] = round((time.perf_counter() - t0) * 1000, 1)


def analyze_turn(
    hub: ModelHub,
    text: str,
    state: TurnState,
    audio: np.ndarray | None = None,
    face: dict | None = None,
) -> dict:
    """Analyse one user message. `face` is an aggregated FER result from recent webcam frames."""
    timings: dict[str, float] = {}
    safety = assess(text)
    text_emo = (
        _timed(timings, "text_emotion", hub.text_emotion.predict, text) if text.strip() else None
    )
    voice = (
        _timed(timings, "voice_emotion", hub.ser.predict, audio)
        if audio is not None and hub.ser
        else None
    )
    pros = (
        _timed(timings, "prosody", prosody, audio, transcript=text) if audio is not None else None
    )
    fused = fuse({"text": text_emo, "voice": voice, "face": face})

    ece = _timed(timings, "cause", hub.ece.predict, text)
    if not isinstance(ece, dict):
        ece = ece.to_dict() | {"source": "model"}
    if ece["causes"]:
        cause = {"text": ece["causes"][0]["text"], "from": "this message"}
    elif state.last_cause:
        cause = {"text": state.last_cause, "from": "earlier in the conversation"}
    else:
        cause = {"text": None, "from": None}

    entities = _timed(timings, "entities", hub.ner.extract, text)
    topic = (
        _timed(timings, "topic", hub.topic.predict, f"{state.user_text_so_far} {text}".strip())
        if hub.topic
        else None
    )
    strategy = None
    if hub.strategy:
        strategy = _timed(
            timings,
            "strategy",
            hub.strategy.predict,
            StrategyContext(
                seeker_text=text,
                prev_supporter_text=state.prev_reply,
                prev_strategy=state.prev_strategy,
                turn_index=state.turn_index,
                emotion=CANONICAL_TO_ESCONV.get(fused["emotion"], ""),
                problem_type=topic["topic"] if topic else "",
            ),
        )
    return {
        "safety": safety,
        "emotion": fused,
        "modalities": {"text": text_emo, "voice": voice, "face": face},
        "prosody": pros,
        "cause": cause | {"extraction": ece},
        "entities": entities,
        "topic": topic,
        "strategy": strategy,
        "timings_ms": timings,
    }


def prompt_context(analysis: dict, memory: str | None) -> PromptContext:
    topic = analysis.get("topic") or {}
    strategy = analysis.get("strategy") or {}
    extra = ELEVATED_INSTRUCTION if analysis["safety"]["level"] == "elevated" else None
    return PromptContext(
        emotion=analysis["emotion"]["emotion"] if analysis["emotion"]["modalities"] else None,
        cause=analysis["cause"]["text"],
        topic=topic.get("topic") if topic.get("topic") not in (None, "other") else None,
        entities=[e["text"] for e in analysis.get("entities", [])],
        memory=memory,
        strategy=strategy.get("strategy"),
        extra_instruction=extra,
    )
