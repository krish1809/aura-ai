"""Loader for the ESConv emotional-support conversation dataset.

The Hugging Face export (thu-coai/esconv) stores one conversation per line as
`{"text": "<json string>"}`; the original GitHub release stores plain objects.
Both are accepted. We keep the official conversation-level train/validation/test
split everywhere so no conversation ever leaks across splits.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from aura_ml.config import ESCONV_DIR

SPLIT_FILES = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}


@dataclass
class Turn:
    speaker: str  # "seeker" | "supporter"
    text: str
    strategies: list[str] = field(default_factory=list)

    @property
    def strategy(self) -> str | None:
        return self.strategies[0] if self.strategies else None


@dataclass
class Conversation:
    id: str
    split: str
    emotion_type: str
    problem_type: str
    situation: str
    turns: list[Turn]


def _parse_record(raw: dict) -> dict:
    return json.loads(raw["text"]) if "text" in raw and isinstance(raw["text"], str) else raw


def _merge_consecutive(dialog: list[dict]) -> list[Turn]:
    """Merge back-to-back utterances from the same speaker into one turn."""
    turns: list[Turn] = []
    for utt in dialog:
        speaker = "seeker" if utt.get("speaker") in ("usr", "seeker") else "supporter"
        text = " ".join(str(utt.get("text") or utt.get("content") or "").split())
        if not text:
            continue
        strategy = utt.get("strategy") or (utt.get("annotation") or {}).get("strategy")
        if turns and turns[-1].speaker == speaker:
            turns[-1].text = f"{turns[-1].text} {text}"
            if strategy and strategy not in turns[-1].strategies:
                turns[-1].strategies.append(strategy)
        else:
            turns.append(Turn(speaker, text, [strategy] if strategy else []))
    return turns


def _iter_json_objects(text: str) -> Iterator[dict]:
    """Yield every JSON object in `text`, tolerating blank lines and records
    that were concatenated without a newline (present in the HF test split)."""
    decoder = json.JSONDecoder()
    pos, end = 0, len(text)
    while pos < end:
        while pos < end and text[pos].isspace():
            pos += 1
        if pos >= end:
            break
        obj, pos = decoder.raw_decode(text, pos)
        yield obj


def load_split(split: str, root: Path = ESCONV_DIR) -> list[Conversation]:
    path = Path(root) / SPLIT_FILES[split]
    conversations = []
    for i, raw in enumerate(_iter_json_objects(path.read_text(encoding="utf-8"))):
        rec = _parse_record(raw)
        conversations.append(
            Conversation(
                id=f"{split}-{i:04d}",
                split=split,
                emotion_type=str(rec.get("emotion_type", "")).strip().lower(),
                problem_type=str(rec.get("problem_type", "")).strip().lower(),
                situation=" ".join(str(rec.get("situation", "")).split()),
                turns=_merge_consecutive(rec.get("dialog", [])),
            )
        )
    return conversations


def load_all(root: Path = ESCONV_DIR) -> dict[str, list[Conversation]]:
    return {split: load_split(split, root) for split in SPLIT_FILES}


def iter_seeker_turns(conv: Conversation) -> Iterator[tuple[int, Turn]]:
    for i, turn in enumerate(conv.turns):
        if turn.speaker == "seeker":
            yield i, turn
