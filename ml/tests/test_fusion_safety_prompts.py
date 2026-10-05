import pytest

from aura_ml.fusion import average_distributions, fuse
from aura_ml.labels import EMOTIONS, RAVDESS_TO_CANONICAL, project_distribution
from aura_ml.models.llm import MockLLM
from aura_ml.models.ner import EntityExtractor
from aura_ml.prompts import PromptContext, build_messages, system_prompt
from aura_ml.safety import assess


def dist(**kw):
    d = dict.fromkeys(EMOTIONS, 0.0)
    d.update(kw)
    return {"available": True, "probs": d}


def test_fusion_renormalises_missing_modalities():
    out = fuse({"text": dist(fear=0.8, neutral=0.2), "voice": None, "face": {"available": False}})
    assert (
        out["emotion"] == "fear"
        and out["modalities"] == ["text"]
        and out["weights"] == {"text": 1.0}
    )


def test_fusion_weighted_vote_and_agreement():
    out = fuse({"text": dist(sadness=1.0), "voice": dist(anger=1.0), "face": dist(sadness=1.0)})
    assert out["emotion"] == "sadness"
    assert out["probs"]["sadness"] == pytest.approx(0.75)
    assert out["agreement"] == pytest.approx(2 / 3, abs=1e-3)
    assert not out["mixed"]


def test_fusion_flags_mixed_signals():
    out = fuse({"text": dist(joy=0.5, sadness=0.5)})
    assert out["mixed"]


def test_fusion_without_signals_is_neutral():
    assert fuse({})["emotion"] == "neutral"


def test_projection_merges_calm_into_neutral():
    p = project_distribution({"calm": 0.3, "neutral": 0.2, "fearful": 0.5}, RAVDESS_TO_CANONICAL)
    assert p["neutral"] == pytest.approx(0.5) and p["fear"] == pytest.approx(0.5)


def test_average_distributions():
    avg = average_distributions([dist(joy=1.0)["probs"], dist(sadness=1.0)["probs"]])
    assert avg["joy"] == avg["sadness"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("I want to kill myself", "crisis"),
        ("sometimes I think everyone is better off dead without me", "crisis"),
        ("I've been self-harming again", "crisis"),
        ("I feel so hopeless about everything", "elevated"),
        ("this exam is killing me lol", "none"),
        ("I'm sad about my breakup", "none"),
    ],
)
def test_safety_levels(text, level):
    assert assess(text)["level"] == level


def test_system_prompt_fills_every_field():
    ctx = PromptContext(
        emotion="fear",
        cause="my exams",
        topic="academic pressure",
        entities=["mom"],
        memory="- They study medicine.",
        strategy="Question",
    )
    sp = system_prompt(ctx)
    for piece in (
        "anxious or afraid",
        "my exams",
        "academic pressure",
        "mom",
        "They study medicine",
        "open-ended question",
    ):
        assert piece in sp


def test_system_prompt_defaults():
    sp = system_prompt(PromptContext())
    assert "not clear yet" in sp and "start of your conversation" in sp


def test_build_messages_starts_history_with_user():
    history = [
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "how are you?"},
    ]
    msgs = build_messages("not great", history, PromptContext())
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "not great"


def test_mock_llm_follows_strategy_line():
    msgs = build_messages("I failed", [], PromptContext(strategy="Question"))
    assert "?" in "".join(MockLLM().stream(msgs))


def test_relation_entities_without_spacy():
    ents = EntityExtractor(None).extract("My best friend and my Mom had a fight")
    assert {e["text"] for e in ents} == {"friend", "mom"}
