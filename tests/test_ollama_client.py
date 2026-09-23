import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features.llm import ollama_client as oc


class FakeClient:
    def __init__(self, models=("mistral:latest",), list_error=None, gen_error=None, response="hello"):
        self.models, self.list_error, self.gen_error, self.response = models, list_error, gen_error, response
        self.list_calls = 0

    def list(self):
        self.list_calls += 1
        if self.list_error:
            raise self.list_error
        return {"models": [{"model": m} for m in self.models]}

    def generate(self, **kwargs):
        if self.gen_error:
            raise self.gen_error
        return {"response": f"  {self.response}  "}


def client_with(fake, **kw):
    c = oc.OllamaClient(**kw)
    c._client = fake
    return c


def test_uses_ollama_when_available():
    c = client_with(FakeClient())
    r = c.generate("hi")
    assert (r.source, r.text, r.model, c.fallback_mode) == ("ollama", "hello", "mistral:latest", False)


def test_server_down_falls_back_without_raising():
    c = client_with(FakeClient(list_error=ConnectionError("refused")))
    r = c.generate("Question\nThe context line.")
    assert r.source == "fallback" and c.fallback_mode and "refused" in r.error
    assert "The context line." in r.text


def test_model_not_pulled_falls_back():
    c = client_with(FakeClient(models=("llama3:latest",)))
    assert not c.is_available(force=True)
    assert "ollama pull mistral:latest" in c.last_error


def test_generation_failure_mid_session_falls_back():
    c = client_with(FakeClient(gen_error=TimeoutError("timed out")))
    r = c.generate("hi", fallback=lambda p: "custom")
    assert (r.source, r.text, c.fallback_mode) == ("fallback", "custom", True)


def test_recovers_after_recheck_interval():
    fake = FakeClient(list_error=ConnectionError("down"))
    c = client_with(fake, recheck_interval_sec=0)
    assert c.generate("x").source == "fallback"
    fake.list_error = None
    assert c.generate("x").source == "ollama" and not c.fallback_mode


def test_availability_is_cached_within_interval():
    fake = FakeClient()
    c = client_with(fake, recheck_interval_sec=3600)
    c.is_available(); c.is_available(); c.generate("x")
    assert fake.list_calls == 1


def test_missing_library_means_fallback(monkeypatch):
    monkeypatch.setattr(oc, "ollama", None)
    c = oc.OllamaClient()
    assert not c.is_available(force=True)
    assert c.generate("x").source == "fallback" and "not installed" in c.last_error


@pytest.mark.parametrize("host", ["http://api.example.com:11434", "http://10.0.0.5:11434"])
def test_non_local_host_rejected(host):
    with pytest.raises(ValueError, match="local-only"):
        oc.OllamaClient(host=host)


def test_defaults_to_mistral():
    assert oc.OllamaClient().model == "mistral:latest"
