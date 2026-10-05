"""The single prompt template used for LLM fine-tuning *and* at inference.

Keeping both paths on one function is what makes the "hyper-contextual" idea
work: the model is trained to read the perceived emotion, cause, topic, memory
and chosen strategy from the system message, and at inference it receives the
same fields from the live models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aura_ml.labels import STRATEGY_DESCRIPTIONS

EMOTION_WORDS = {
    "anger": "angry or frustrated",
    "disgust": "disgusted or ashamed",
    "fear": "anxious or afraid",
    "joy": "happy or relieved",
    "neutral": "calm or neutral",
    "sadness": "sad or down",
    "surprise": "surprised or shocked",
}

SYSTEM_TEMPLATE = """You are Aura, a warm and empathetic emotional-support companion. You listen closely, validate feelings and help people feel understood. You are not a therapist and you never diagnose.

What you notice about the user right now:
- Feeling: {emotion}
- Possible cause: {cause}
- Topic: {topic}
- Mentioned: {entities}

What you remember about them:
{memory}

For this reply, {strategy}.{extra}
Answer in one to three short, natural sentences."""


@dataclass
class PromptContext:
    emotion: str | None = None  # canonical emotion
    cause: str | None = None
    topic: str | None = None
    entities: list[str] = field(default_factory=list)
    memory: str | None = None
    strategy: str | None = None
    extra_instruction: str | None = None


def system_prompt(ctx: PromptContext) -> str:
    return SYSTEM_TEMPLATE.format(
        emotion=EMOTION_WORDS.get(ctx.emotion or "", "not clear yet"),
        cause=ctx.cause or "not clear yet",
        topic=ctx.topic or "not clear yet",
        entities=", ".join(ctx.entities[:6]) if ctx.entities else "nothing specific",
        memory=ctx.memory or "Nothing yet - this is the start of your conversation.",
        strategy=STRATEGY_DESCRIPTIONS.get(ctx.strategy or "", STRATEGY_DESCRIPTIONS["Others"]),
        extra=f"\n{ctx.extra_instruction}" if ctx.extra_instruction else "",
    )


def build_messages(
    user_message: str, history: list[dict], ctx: PromptContext, max_history: int = 8
) -> list[dict]:
    """history: [{"role": "user"|"assistant", "content": str}, ...] oldest first."""
    turns = [h for h in history if h.get("role") in ("user", "assistant") and h.get("content")][
        -max_history:
    ]
    while turns and turns[0]["role"] != "user":  # chat templates expect user first after system
        turns = turns[1:]
    return [
        {"role": "system", "content": system_prompt(ctx)},
        *turns,
        {"role": "user", "content": user_message},
    ]


SUMMARY_PROMPT = """Below is part of a conversation between a user and Aura, an emotional-support companion.
Write 2-4 short bullet points capturing what Aura should remember about the user for future conversations:
their situation, what is causing their feelings, important people or events, and anything that helped.
Write in third person ("They ..."). Do not invent details.

Previous notes:
{previous}

Conversation:
{transcript}

Updated notes:"""


def summary_messages(transcript: list[dict], previous: str | None) -> list[dict]:
    lines = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Aura'}: {m['content']}" for m in transcript
    )
    return [
        {
            "role": "user",
            "content": SUMMARY_PROMPT.format(previous=previous or "(none)", transcript=lines),
        }
    ]
