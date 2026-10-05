"""Crisis-language detection and a safe, deterministic response.

A 3B model fine-tuned on peer-support chat is not equipped to handle someone in
danger, so when crisis language is detected Aura does not improvise: it answers
with a fixed, carefully worded message and points to real help. Detection is a
transparent pattern list (high recall on explicit statements, by design).
"""

from __future__ import annotations

import re

_CRISIS = [
    r"\b(?:kill|killing|end|ending|take|taking)\s+(?:myself|my\s+(?:own\s+)?life)\b",
    r"\bsuicid(?:e|al)\b",
    r"\b(?:want|wanna|going|planning)\s+to\s+die\b",
    r"\b(?:don'?t|do not|no longer)\s+want\s+to\s+(?:live|be alive|exist|wake up)\b",
    r"\bbetter\s+off\s+dead\b",
    r"\b(?:no|nothing\s+to)\s+(?:reason\s+)?(?:to\s+)?live\s+for\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bend\s+it\s+all\b",
    r"\bself[\s-]?harm(?:ing)?\b",
    r"\b(?:cut|cutting|hurt|hurting|harm|harming)\s+myself\b",
    r"\boverdos(?:e|ing)\b",
    r"\bnot\s+worth\s+living\b",
]
_ELEVATED = [
    r"\bhopeless\b",
    r"\bcan'?t\s+(?:go\s+on|take\s+(?:it|this)\s+any\s*more|do\s+this\s+any\s*more)\b",
    r"\b(?:no\s+one|nobody)\s+(?:would|will)\s+(?:care|miss\s+me)\b",
    r"\bwant\s+to\s+disappear\b",
    r"\bworthless\b",
    r"\bgive\s+up\s+on\s+(?:life|everything)\b",
]
_CRISIS_RE = [re.compile(p, re.IGNORECASE) for p in _CRISIS]
_ELEVATED_RE = [re.compile(p, re.IGNORECASE) for p in _ELEVATED]

DEFAULT_RESOURCES = [
    {"region": "India", "name": "Tele-MANAS (24x7, free)", "contact": "14416 or 1-800-891-4416"},
    {"region": "India", "name": "Emergency services", "contact": "112"},
    {
        "region": "United States",
        "name": "988 Suicide & Crisis Lifeline",
        "contact": "call or text 988",
    },
    {"region": "UK & Ireland", "name": "Samaritans", "contact": "116 123"},
    {
        "region": "Anywhere",
        "name": "Find a helpline near you",
        "contact": "https://findahelpline.com",
    },
]

CRISIS_RESPONSE = (
    "I'm really glad you told me, and I'm so sorry you're carrying this much pain right now. "
    "You deserve support from a real person who can be with you through this. "
    "If you are in immediate danger, please call your local emergency number. "
    "You can also reach a free, confidential crisis line right now - I've listed some below. "
    "If you can, reach out to someone you trust and let them know how you're feeling. "
    "I'm still here to listen if you want to keep talking."
)

ELEVATED_INSTRUCTION = (
    "The user may be in significant distress. Be especially gentle, validate their pain, "
    "and softly ask whether they are safe and whether there is someone they can reach out to."
)


def assess(text: str) -> dict:
    crisis = [m.group(0) for r in _CRISIS_RE if (m := r.search(text))]
    if crisis:
        return {"level": "crisis", "matches": crisis}
    elevated = [m.group(0) for r in _ELEVATED_RE if (m := r.search(text))]
    return {"level": "elevated" if elevated else "none", "matches": elevated}
