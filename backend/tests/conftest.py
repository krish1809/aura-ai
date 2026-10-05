import time

import pytest
from fastapi.testclient import TestClient

from aura_api.config import Settings
from aura_api.main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(
        environment="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        ml_mode="lite",
        llm_provider="mock",
        summarize_every=2,
        cors_origins=["http://testserver"],
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        deadline = time.time() + 10
        while not app.state.ml.ready and time.time() < deadline:
            time.sleep(0.05)
        yield c


def register(client, email="rishi@example.com", name="Rishi", password="correct horse"):
    r = client.post("/api/auth/register", json={"email": email, "name": name, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def user(client):
    return register(client)


def chat_turn(client, conversation_id: str, text: str = "", audio: str | None = None) -> list[dict]:
    """Send one message over the WebSocket and collect events until the reply is done."""
    token = client.cookies.get("aura_session")
    events = []
    with client.websocket_connect(f"/api/ws/conversations/{conversation_id}?token={token}") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "message", "text": text, "audio": audio})
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("done", "error"):
                break
    client.portal.call(client.app.state.chat.drain)  # let background memory work finish
    return events
