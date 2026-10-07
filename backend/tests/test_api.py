import base64
import io
import wave

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from .conftest import chat_turn, register


def test_health_and_model_status(client):
    assert client.get("/api/health").json()["status"] == "ok"
    comps = client.get("/api/system/models").json()["components"]
    assert comps["components"]["ece"]["source"] == "rules"
    assert comps["capabilities"]["voice"] is False


def test_register_login_logout(client):
    user = register(client)
    assert client.get("/api/auth/me").json()["email"] == user["email"]
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401
    assert (
        client.post(
            "/api/auth/login", json={"email": user["email"], "password": "wrong password"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login", json={"email": "RISHI@example.com", "password": "correct horse"}
        ).status_code
        == 200
    )


def test_duplicate_email_and_weak_password(client):
    register(client)
    assert (
        client.post(
            "/api/auth/register",
            json={"email": "rishi@example.com", "name": "x", "password": "longenough"},
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/auth/register",
            json={"email": "new@example.com", "name": "x", "password": "short"},
        ).status_code
        == 422
    )


def test_conversations_are_private(client, user):
    conv = client.post("/api/conversations", json={}).json()
    client.post("/api/auth/logout")
    register(client, email="other@example.com")
    assert client.get(f"/api/conversations/{conv['id']}").status_code == 404
    assert client.get("/api/conversations").json() == []


def test_chat_turn_streams_analysis_and_reply(client, user):
    conv = client.post("/api/conversations", json={}).json()
    events = chat_turn(
        client,
        conv["id"],
        "I'm so anxious about my exams because my mom keeps comparing me to my sister",
    )
    kinds = [e["type"] for e in events]
    assert kinds[0] == "analysis" and kinds[-1] == "done" and "token" in kinds

    analysis = events[0]["message"]["analysis"]
    assert analysis["emotion"]["emotion"] == "fear"
    assert analysis["cause"]["text"] == "my exams"
    assert {e["text"] for e in analysis["entities"]} == {"mom", "sister"}
    assert analysis["topic"]["topic"] == "academic pressure"
    assert analysis["strategy"]["strategy"]

    reply = events[-1]["message"]
    assert (
        reply["role"] == "assistant"
        and reply["content"] == "".join(e["text"] for e in events if e["type"] == "token").strip()
    )
    assert reply["meta"]["strategy"] == analysis["strategy"]["strategy"]
    assert events[-1]["conversation"]["title"].startswith("I'm so anxious about my exams")

    detail = client.get(f"/api/conversations/{conv['id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["created_at"].endswith("Z") or "+00:00" in detail["created_at"]


def test_cause_is_carried_across_turns_and_graph_is_built(client, user):
    conv = client.post("/api/conversations", json={}).json()
    chat_turn(client, conv["id"], "I've been really down since my boyfriend broke up with me")
    events = chat_turn(client, conv["id"], "I just don't know what to do anymore")
    cause = events[0]["message"]["analysis"]["cause"]
    assert cause == cause | {
        "text": "my boyfriend broke up with me",
        "from": "earlier in the conversation",
    }

    graph = client.get("/api/memory/graph").json()
    labels = {(n["kind"], n["label"]) for n in graph["nodes"]}
    assert ("person", "boyfriend") in labels and (
        "cause",
        "my boyfriend broke up with me",
    ) in labels
    assert any(link["relation"] == "involved_in" for link in graph["links"])


def test_memory_notes_after_summary_and_forget(client, user):
    conv = client.post("/api/conversations", json={}).json()
    chat_turn(client, conv["id"], "Work has been so stressful lately")
    chat_turn(
        client, conv["id"], "My manager keeps piling on deadlines"
    )  # summarize_every=2 -> summary
    notes = client.get("/api/memory/notes").json()["notes"]
    assert "They said: My manager keeps piling on deadlines" in notes
    client.delete("/api/memory")
    assert client.get("/api/memory/notes").json()["notes"] == ""
    assert client.get("/api/memory/graph").json()["nodes"] == []


def test_crisis_message_gets_safe_template_and_resources(client, user):
    conv = client.post("/api/conversations", json={}).json()
    events = chat_turn(client, conv["id"], "I don't want to live anymore, I want to end my life")
    safety = next(e for e in events if e["type"] == "safety")
    assert safety["level"] == "crisis" and any("14416" in r["contact"] for r in safety["resources"])
    assert events[-1]["message"]["meta"]["model"] == "safety-template"


def test_voice_message_needs_stt_in_lite_mode(client, user):
    conv = client.post("/api/conversations", json={}).json()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.sin(np.linspace(0, 2000, 16000)) * 3000).astype("<i2").tobytes())
    events = chat_turn(client, conv["id"], audio=base64.b64encode(buf.getvalue()).decode())
    assert events[-1]["type"] == "error" and "speech-to-text" in events[-1]["detail"]


def test_websocket_rejects_strangers(client, user):
    conv = client.post("/api/conversations", json={}).json()
    client.cookies.clear()  # otherwise the valid session cookie is used
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"/api/ws/conversations/{conv['id']}?token=bad") as ws,
    ):
        ws.receive_json()
    register(client, email="intruder@example.com")
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"/api/ws/conversations/{conv['id']}") as ws,
    ):
        ws.receive_json()


def test_insights_and_account_deletion(client, user):
    conv = client.post("/api/conversations", json={}).json()
    chat_turn(client, conv["id"], "I'm angry because my coworker took credit for my work")
    data = client.get("/api/insights").json()
    assert data["messages"] == 1 and data["emotions"] == {"anger": 1}
    assert data["top_causes"][0]["text"] == "my coworker took credit for my work"
    assert client.delete("/api/auth/me").status_code == 204
    assert client.get("/api/auth/me").status_code == 401
    assert (
        client.post(
            "/api/auth/login", json={"email": user["email"], "password": "correct horse"}
        ).status_code
        == 401
    )


def test_spa_serving_blocks_path_traversal(settings, tmp_path):
    from fastapi.testclient import TestClient

    from aura_api.main import create_app

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Aura</title>")
    (tmp_path / "secret.txt").write_text("top secret")
    with TestClient(create_app(settings.model_copy(update={"frontend_dist": dist}))) as c:
        assert "Aura" in c.get("/chat/123").text  # client-side route -> index.html
        assert "top secret" not in c.get("/..%2Fsecret.txt").text
        assert c.get("/api/health").status_code == 200


def test_metrics_report_reads_trained_artifacts(client):
    models = {m["id"]: m for m in client.get("/api/system/metrics").json()["models"]}
    assert set(models) >= {"ece", "ser", "fer", "strategy", "topic"}
    if models["strategy"]["trained"]:  # trained in this checkout
        labels = {m["label"] for m in models["strategy"]["metrics"]}
        assert "Accuracy (8 strategies)" in labels


def test_back_to_back_messages_are_queued_not_dropped(client, user):
    """Regression: a message sent right after 'done' used to be rejected - even a crisis message."""
    conv = client.post("/api/conversations", json={}).json()
    token = client.cookies.get("aura_session")
    texts = ["hey", "I'm stressed about work", "I want to end my life"]
    with client.websocket_connect(f"/api/ws/conversations/{conv['id']}?token={token}") as ws:
        assert ws.receive_json()["type"] == "ready"
        for t in texts:
            ws.send_json({"type": "message", "text": t, "audio": None})
        done, events = [], []
        while len(done) < len(texts):
            e = ws.receive_json()
            events.append(e["type"])
            assert e["type"] != "error", e
            if e["type"] == "done":
                done.append(e["message"])
    assert "safety" in events
    assert done[-1]["meta"]["model"] == "safety-template"


def test_identity_question_gets_the_honest_template(client, user):
    conv = client.post("/api/conversations", json={}).json()
    events = chat_turn(client, conv["id"], "are you a real person?")
    reply = events[-1]["message"]
    assert "not a person" in reply["content"] and reply["meta"]["model"] == "honesty-template"
