"""Entity extraction for memory and the knowledge graph.

People in emotional conversations rarely name others ("my boss", "my mom"), so
alongside spaCy NER we extract relationship mentions with a pattern, which also
works when spaCy is not installed.
"""

from __future__ import annotations

import re

RELATIONS = [
    "mom",
    "mother",
    "mum",
    "dad",
    "father",
    "parents",
    "brother",
    "sister",
    "sibling",
    "son",
    "daughter",
    "kids",
    "children",
    "child",
    "wife",
    "husband",
    "partner",
    "girlfriend",
    "boyfriend",
    "fiance",
    "fiancee",
    "ex",
    "spouse",
    "grandma",
    "grandmother",
    "grandpa",
    "grandfather",
    "aunt",
    "uncle",
    "cousin",
    "family",
    "friend",
    "friends",
    "best friend",
    "roommate",
    "boss",
    "manager",
    "coworker",
    "co-worker",
    "colleague",
    "colleagues",
    "teacher",
    "professor",
    "classmate",
    "classmates",
    "therapist",
    "doctor",
    "dog",
    "cat",
    "pet",
]
_REL = re.compile(
    r"\bmy\s+(?:(?:best|old|new|younger|older|little|big|ex)\s+)?("
    + "|".join(sorted(map(re.escape, RELATIONS), key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "EVENT", "FAC", "NORP"}


class EntityExtractor:
    def __init__(self, spacy_model: str | None = "en_core_web_sm"):
        self.nlp = None
        if spacy_model:
            try:
                import spacy

                self.nlp = spacy.load(spacy_model, disable=["lemmatizer"])
            except (ImportError, OSError):
                self.nlp = None

    @property
    def source(self) -> str:
        return "spacy" if self.nlp else "patterns"

    def extract(self, text: str) -> list[dict]:
        found: dict[tuple[str, str], dict] = {}
        for m in _REL.finditer(text):
            rel = m.group(1).lower()
            found[(rel, "RELATION")] = {"text": rel, "label": "RELATION"}
        if self.nlp:
            for ent in self.nlp(text).ents:
                if ent.label_ in KEEP_LABELS and len(ent.text) > 1:
                    found.setdefault(
                        (ent.text.lower(), ent.label_), {"text": ent.text, "label": ent.label_}
                    )
        return list(found.values())
