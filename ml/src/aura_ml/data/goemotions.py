"""GoEmotions (Demszky et al., 2020): 58k Reddit comments with 27 human-annotated
emotions, mapped onto Aura's 7 canonical emotions with the dataset's official
Ekman mapping. Used to train the CPU text-emotion model for lite deployments.

    python -m aura_ml.data.goemotions
"""

from __future__ import annotations

import csv
import http.client
import io
import time
import urllib.request
from pathlib import Path

from aura_ml.config import RAW_DIR

BASE = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
SPLITS = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
GOEMOTIONS_DIR = RAW_DIR / "goemotions"

LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# Official Ekman mapping shipped with the dataset (ekman_mapping.json), in Aura's label names.
EKMAN = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy",
        "amusement",
        "approval",
        "excitement",
        "gratitude",
        "love",
        "optimism",
        "relief",
        "pride",
        "admiration",
        "desire",
        "caring",
    ],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
TO_EKMAN = {fine: coarse for coarse, fines in EKMAN.items() for fine in fines}


def _fetch(url: str, dest: Path, attempts: int = 4) -> None:
    """Download to a temp file and rename, retrying flaky connections."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                tmp.write_bytes(r.read())
            tmp.replace(dest)
            return
        except (OSError, http.client.HTTPException) as exc:
            print(f"  attempt {attempt} failed: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


def download(root: Path = GOEMOTIONS_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in SPLITS.values():
        dest = root / name
        if not dest.exists():
            print(f"downloading {name} ...")
            _fetch(f"{BASE}/{name}", dest)


def load(split: str, root: Path = GOEMOTIONS_DIR) -> tuple[list[str], list[str]]:
    """Texts and single Ekman labels. Comments whose labels map to more than one
    Ekman category are dropped (they have no single correct answer)."""
    if not (root / SPLITS[split]).exists():
        download(root)
    texts, labels = [], []
    reader = csv.reader(
        io.StringIO((root / SPLITS[split]).read_text(encoding="utf-8")),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    )
    for row in reader:
        if len(row) < 2:
            continue
        coarse = {TO_EKMAN[LABELS[int(i)]] for i in row[1].split(",")}
        if len(coarse) == 1:
            texts.append(row[0])
            labels.append(coarse.pop())
    return texts, labels


if __name__ == "__main__":
    download()
    for s in SPLITS:
        t, y = load(s)
        print(s, len(t))
