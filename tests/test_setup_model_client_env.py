
import pytest
from spoox.utils import ModelClientId, setup_model_client


def test_setup_model_client_missing_env(monkeypatch):
    for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_HOST"]:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(Exception):
        setup_model_client(ModelClientId.OPENAI, "dummy-model")


def test_setup_model_client_with_minimal_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    client = setup_model_client(ModelClientId.OPENAI, "gpt-3-test")
    assert client is not None
