import json

import pytest

from aura_ml.data.bio import IGNORE_INDEX, B, I, O, align_bio, decode_bio
from aura_ml.data.causes import extract_causes, is_phatic
from aura_ml.data.esconv import load_split
from aura_ml.eval.metrics import (
    binary_prf,
    classification_report,
    distinct_n,
    rouge_l,
    span_exact_prf,
    span_overlap_prf,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I am sad because my partner and i broke up", ["my partner and i broke up"]),
        ("I lost my job last week so I am really stressed", ["I lost my job last week"]),
        (
            "My boss yells at me every day and it makes me feel worthless",
            ["My boss yells at me every day"],
        ),
        (
            "I am stressed about losing my job due to this pandemic.",
            ["losing my job", "this pandemic"],
        ),
        ("I am so angry at my brother for lying", ["my brother for lying"]),
    ],
)
def test_explicit_causes(text, expected):
    assert [c.text for c in extract_causes(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        "I will call him because he asked me to",  # reason for an action, not a feeling
        "So it is hard to tell what is the underlying cause I think",  # 'cause' the noun
        "We cant go through life being upset at each other.",  # pronoun-only object
        "i am feeling very sad at the moment",
        "Okay, thanks!",
    ],
)
def test_non_causes(text):
    assert extract_causes(text) == []


def test_spans_are_character_offsets():
    text = "Honestly I'm worried about my mom's health."
    (span,) = extract_causes(text)
    assert text[span.start : span.end] == span.text == "my mom's health"


def test_phatic():
    assert is_phatic("hello!") and is_phatic("ok") and not is_phatic("I can't sleep at night")


def test_bio_roundtrip_uses_training_label_ids():
    # O=0, B=1, I=2 - the 2025 inference code decoded these as B=0, I=1, O=2
    assert (O, B, I) == (0, 1, 2)
    offsets = [
        (0, 0),
        (0, 1),
        (2, 4),
        (5, 10),
        (11, 13),
        (14, 18),
        (0, 0),
    ]  # <s> I am worried my exam </s>
    labels = align_bio(offsets, [(11, 18)])
    assert labels == [IGNORE_INDEX, O, O, O, B, I, IGNORE_INDEX]
    assert decode_bio(labels, offsets) == [(11, 18)]


def test_decode_orphan_inside_tag_opens_span():
    offsets = [(0, 3), (4, 7), (8, 11)]
    assert decode_bio([O, I, I], offsets) == [(4, 11)]
    assert decode_bio([B, O, B], offsets) == [(0, 3), (8, 11)]


def test_span_metrics():
    preds, golds = [[(0, 5)], [], [(3, 9)]], [[(0, 5)], [(1, 4)], [(3, 6)]]
    exact = span_exact_prf(preds, golds)
    assert (exact["tp"], exact["fp"], exact["fn"]) == (1, 1, 2)
    overlap = span_overlap_prf(preds, golds)
    assert overlap["precision"] == pytest.approx(8 / 11)
    assert overlap["recall"] == pytest.approx(8 / 11)
    assert binary_prf([True, False, True], [True, True, True])["recall"] == pytest.approx(2 / 3)


def test_classification_report_confusion_matrix():
    rep = classification_report(["a", "b", "b"], ["a", "a", "b"], ["a", "b"])
    assert rep["confusion_matrix"] == [[1, 1], [0, 1]]
    assert rep["accuracy"] == pytest.approx(2 / 3)


def test_generation_metrics():
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l("the cat", "a dog") == 0.0
    assert distinct_n(["a a a"], 1) == pytest.approx(1 / 3)


def test_esconv_loader_handles_concatenated_records(tmp_path):
    conv = {
        "emotion_type": "anxiety",
        "problem_type": "job crisis",
        "situation": "worried about job",
        "dialog": [
            {"speaker": "usr", "text": "Hi"},
            {"speaker": "sys", "text": "Hello.", "strategy": "Question"},
            {"speaker": "sys", "text": "How are you?", "strategy": "Question"},
            {"speaker": "usr", "text": "Anxious about my job."},
        ],
    }
    line = json.dumps({"text": json.dumps(conv)})
    (tmp_path / "test.jsonl").write_text(
        "\n" + line + line + "\n"
    )  # blank line + two glued records
    convs = load_split("test", tmp_path)
    assert len(convs) == 2
    turns = convs[0].turns
    assert [t.speaker for t in turns] == ["seeker", "supporter", "seeker"]
    assert turns[1].text == "Hello. How are you?" and turns[1].strategies == ["Question"]
