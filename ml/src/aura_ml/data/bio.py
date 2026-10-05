"""BIO tagging between character spans and sub-word tokens.

Works on the `offset_mapping` produced by any Hugging Face *fast* tokenizer, so
the gold labels and the decoded predictions are both expressed as character
offsets into the original text (never as detokenised strings, which drift).
"""

from __future__ import annotations

from collections.abc import Sequence

O, B, I = 0, 1, 2  # noqa: E741  (I is the conventional BIO tag name)
ID2TAG = {O: "O", B: "B-CAUSE", I: "I-CAUSE"}
TAG2ID = {v: k for k, v in ID2TAG.items()}
IGNORE_INDEX = -100

Span = tuple[int, int]


def align_bio(offsets: Sequence[Span], spans: Sequence[Span]) -> list[int]:
    """Token labels for character `spans`. Special tokens (offset (0, 0)) get IGNORE_INDEX."""
    labels: list[int] = []
    started = [False] * len(spans)
    for tok_start, tok_end in offsets:
        if tok_start == tok_end:
            labels.append(IGNORE_INDEX)
            continue
        label = O
        for k, (s, e) in enumerate(spans):
            if tok_start < e and tok_end > s:
                label = I if started[k] else B
                started[k] = True
                break
        labels.append(label)
    return labels


def decode_bio(labels: Sequence[int], offsets: Sequence[Span]) -> list[Span]:
    """Character spans from predicted token labels. An I without a preceding B opens a span."""
    spans: list[Span] = []
    cur: list[int] | None = None
    for label, (tok_start, tok_end) in zip(labels, offsets, strict=False):
        if tok_start == tok_end:
            continue
        if label == B or (label == I and cur is None):
            if cur:
                spans.append((cur[0], cur[1]))
            cur = [tok_start, tok_end]
        elif label == I and cur is not None:
            cur[1] = tok_end
        else:
            if cur:
                spans.append((cur[0], cur[1]))
            cur = None
    if cur:
        spans.append((cur[0], cur[1]))
    return spans
