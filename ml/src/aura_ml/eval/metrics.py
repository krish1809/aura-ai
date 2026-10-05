"""Evaluation metrics that do not depend on torch."""

from __future__ import annotations

from collections.abc import Sequence

Span = tuple[int, int]


def _prf(tp: float, fp: float, fn: float) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def span_exact_prf(
    preds: Sequence[Sequence[Span]], golds: Sequence[Sequence[Span]]
) -> dict[str, float]:
    """Exact-match span precision/recall/F1 over a corpus."""
    tp = fp = fn = 0
    for pred, gold in zip(preds, golds, strict=True):
        p, g = set(map(tuple, pred)), set(map(tuple, gold))
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return _prf(tp, fp, fn) | {"tp": tp, "fp": fp, "fn": fn}


def _chars(spans: Sequence[Span]) -> set[int]:
    return {i for s, e in spans for i in range(s, e)}


def span_overlap_prf(
    preds: Sequence[Sequence[Span]], golds: Sequence[Sequence[Span]]
) -> dict[str, float]:
    """Proportional (character-overlap) precision/recall/F1 - gives partial credit
    when a predicted cause is a little longer or shorter than the silver label."""
    inter = pred_n = gold_n = 0
    for pred, gold in zip(preds, golds, strict=True):
        p, g = _chars(pred), _chars(gold)
        inter += len(p & g)
        pred_n += len(p)
        gold_n += len(g)
    return _prf(inter, pred_n - inter, gold_n - inter)


def binary_prf(preds: Sequence[bool], golds: Sequence[bool]) -> dict[str, float]:
    tp = sum(p and g for p, g in zip(preds, golds, strict=True))
    fp = sum(p and not g for p, g in zip(preds, golds, strict=True))
    fn = sum(g and not p for p, g in zip(preds, golds, strict=True))
    acc = sum(p == g for p, g in zip(preds, golds, strict=True)) / max(len(golds), 1)
    return _prf(tp, fp, fn) | {"accuracy": acc}


def _tokens(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c == "'" else " " for c in text).split()


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 via longest common subsequence over lowercase word tokens."""
    a, b = _tokens(pred), _tokens(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Share of unique n-grams across all generations (higher = less repetitive)."""
    grams = [
        tuple(t[i : i + n])
        for text in texts
        for t in [_tokens(text)]
        for i in range(len(t) - n + 1)
    ]
    return len(set(grams)) / len(grams) if grams else 0.0


def classification_report(
    preds: Sequence[str], golds: Sequence[str], labels: Sequence[str]
) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1 and a confusion matrix (rows = gold)."""
    idx = {label: i for i, label in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for p, g in zip(preds, golds, strict=True):
        cm[idx[g]][idx[p]] += 1
    per_class = {}
    for label, i in idx.items():
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels))) - tp
        fn = sum(cm[i]) - tp
        per_class[label] = _prf(tp, fp, fn) | {"support": sum(cm[i])}
    supported = [v["f1"] for v in per_class.values() if v["support"]]
    return {
        "accuracy": sum(p == g for p, g in zip(preds, golds, strict=True)) / max(len(golds), 1),
        "macro_f1": sum(supported) / max(len(supported), 1),
        "per_class": per_class,
        "labels": list(labels),
        "confusion_matrix": cm,
    }
