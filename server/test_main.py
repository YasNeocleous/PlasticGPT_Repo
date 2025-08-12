from fastapi.testclient import TestClient
from main import app


def test_health():
	client = TestClient(app)
	r = client.get("/health")
	assert r.status_code == 200
	assert r.json()["status"] == "ok"


def test_chat_stub(monkeypatch):
	# Force stub mode by clearing API key
	monkeypatch.delenv("OPENAI_API_KEY", raising=False)
	client = TestClient(app)
	r = client.post("/api/chat", json={"question": "What is microsurgery?"})
	assert r.status_code == 200
	data = r.json()
	assert "response" in data
	assert "STUB" in data["response"]

