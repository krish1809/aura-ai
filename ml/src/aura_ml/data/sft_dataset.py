"""Build the hyper-contextual instruction-tuning dataset for the Aura LLM.

One example per ESConv supporter turn that answers a seeker turn:

    system    <- aura_ml.prompts.system_prompt(emotion, cause, topic, entities, memory, strategy)
    user/assistant history (previous turns, real roles)
    user      <- the seeker's message
    assistant <- the supporter's reply (the training target)

Fixes over the 2025 generator: speakers are no longer all labelled "Aura", the
history is passed as real chat turns, the gold ESConv topic and situation are
used instead of keyword guesses, and the prompt is the same function the app
uses at inference time.

    python -m aura_ml.data.sft_dataset
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from aura_ml.config import ESCONV_DIR, SEED, SFT_DATA_DIR
from aura_ml.data.causes import extract_causes
from aura_ml.data.esconv import load_all
from aura_ml.io import write_jsonl
from aura_ml.labels import ESCONV_TO_CANONICAL, STRATEGIES, to_canonical
from aura_ml.models.ner import EntityExtractor
from aura_ml.prompts import PromptContext, build_messages


def build(
    esconv_dir: Path = ESCONV_DIR,
    out_dir: Path = SFT_DATA_DIR,
    history_turns: int = 8,
    memory_prob: float = 0.5,
) -> dict:
    rng = random.Random(SEED)
    ner = EntityExtractor()
    stats: dict = {"entity_source": ner.source}
    for split, convs in load_all(esconv_dir).items():
        rows = []
        for conv in convs:
            latest_cause: str | None = None
            # the seeker's own pre-chat description stands in for long-term memory
            memory = (
                f"- {conv.situation}" if conv.situation and rng.random() < memory_prob else None
            )
            for i, turn in enumerate(conv.turns):
                if turn.speaker == "seeker":
                    spans = extract_causes(turn.text)
                    if spans:
                        latest_cause = "; ".join(s.text for s in spans[:2])
                    continue
                if i == 0 or conv.turns[i - 1].speaker != "seeker":
                    continue
                user_msg = conv.turns[i - 1].text
                history = [
                    {"role": "user" if t.speaker == "seeker" else "assistant", "content": t.text}
                    for t in conv.turns[max(0, i - 1 - history_turns) : i - 1]
                ]
                ctx = PromptContext(
                    emotion=to_canonical(conv.emotion_type, ESCONV_TO_CANONICAL),
                    cause=latest_cause,
                    topic=conv.problem_type or None,
                    entities=[e["text"] for e in ner.extract(user_msg)],
                    memory=memory,
                    strategy=turn.strategy if turn.strategy in STRATEGIES else "Others",
                )
                messages = build_messages(user_msg, history, ctx, max_history=history_turns)
                messages.append({"role": "assistant", "content": turn.text})
                rows.append(
                    {
                        "id": f"{conv.id}-t{i:02d}",
                        "conversation_id": conv.id,
                        "strategy": ctx.strategy,
                        "emotion": ctx.emotion,
                        "topic": conv.problem_type,
                        "has_memory": memory is not None,
                        "messages": messages,
                    }
                )
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        stats[split] = {
            "examples": len(rows),
            "with_cause": sum(
                "Possible cause: not clear yet" not in r["messages"][0]["content"] for r in rows
            ),
            "with_memory": sum(r["has_memory"] for r in rows),
            "strategies": dict(Counter(r["strategy"] for r in rows)),
        }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--esconv-dir", type=Path, default=ESCONV_DIR)
    ap.add_argument("--out-dir", type=Path, default=SFT_DATA_DIR)
    ap.add_argument("--history-turns", type=int, default=8)
    ap.add_argument("--memory-prob", type=float, default=0.5)
    args = ap.parse_args()
    print(
        json.dumps(
            build(args.esconv_dir, args.out_dir, args.history_turns, args.memory_prob), indent=2
        )
    )


if __name__ == "__main__":
    main()
