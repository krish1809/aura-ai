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


# --- Honesty -----------------------------------------------------------------------------------
# Aura's replies were learned from human volunteers, who talked about their own lives. An AI
# companion must never claim to be a person, so identity questions get a fixed truthful answer
# and generated sentences that claim a human life are removed before a reply is stored.

_IDENTITY = re.compile(
    r"\b(?:are|r)\s+(?:you|u)\s+(?:a\s+|an\s+)?(?:real|human|person|bot|robot|ai|machine|chat\s?bot|computer|alive)\b"
    r"|\b(?:are|r)\s+(?:you|u)\s+(?:in|from)\s+(?:the\s+)?(?:uk|us|usa|states|india|england|america|canada|australia)\b"
    r"|\bwho\s+(?:are|r)\s+(?:you|u)\b"
    r"|\bwhat\s+(?:are|r)\s+(?:you|u)\s*\??$"
    r"|\bwhere\s+do\s+(?:you|u)\s+(?:live|stay|come\s+from)\b"
    r"|\bwhere\s+(?:are|r)\s+(?:you|u)\s+from\b"
    r"|\bhow\s+old\s+(?:are|r)\s+(?:you|u)\b"
    r"|\bdo\s+(?:you|u)\s+have\s+(?:a\s+)?(?:family|kids|children|wife|husband|boyfriend|girlfriend|body|job)\b",
    re.IGNORECASE,
)

IDENTITY_RESPONSE = (
    "I'm Aura, an AI companion - not a person. I don't have a life, a home or a family of my own, "
    "but I'm genuinely here to listen and help you think things through. What's on your mind?"
)

HONESTY_INSTRUCTION = (
    "You are an AI: never claim to be a person or to have a home, family or past of your own."
)

_HUMAN_CLAIMS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bI(?:'m|\s+am)\s+(?:a\s+)?(?:real\s+)?(?:person|human)\b",
        r"\bI(?:'m|\s+am)\s+not\s+(?:a\s+|an\s+)?(?:bot|robot|ai|machine|computer)\b",
        r"\bI\s+live\s+(?:in|near|with|alone|by)\b",
        r"\bI(?:'m|\s+am)\s+(?:from|based\s+in)\s+[A-Z]",
        r"\bI(?:'m|\s+am)\s+in\s+(?:the\s+)?(?:UK|US|USA|States|India|Canada|England|America|Australia)\b",
        r"\bmy\s+(?:wife|husband|kids|children|son|daughter|mom|mum|dad|mother|father|boyfriend|girlfriend|ex|brother|sister|family|boss|job|house|apartment|dog|cat)\b",
        r"\bI(?:'ve|\s+have)\s+been\s+(?:there|through\s+(?:that|this|it|something|the\s+same|a\s+similar))",
        r"\bI\s+(?:went|was\s+going)\s+through\b",
        r"\bwhen\s+I\s+was\s+(?:younger|a\s+kid|a\s+child|little|in\s+(?:school|college|high\s+school|university))\b",
        r"\bI\s+felt\s+the\s+same\b",
        r"\b(?:happened|happens)\s+to\s+me\b",
    )
]
_SENTENCES = re.compile(r"[^.!?]+[.!?]*\s*")


def is_identity_question(text: str) -> bool:
    return bool(_IDENTITY.search(text))


def remove_human_claims(reply: str) -> tuple[str, list[str]]:
    """Drop sentences in which the model claims to be human or to have a personal life."""
    kept, removed = [], []
    for sentence in _SENTENCES.findall(reply):
        (removed if any(p.search(sentence) for p in _HUMAN_CLAIMS) else kept).append(sentence)
    return "".join(kept).strip(), [s.strip() for s in removed]
