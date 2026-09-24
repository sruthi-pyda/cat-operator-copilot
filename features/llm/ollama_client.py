"""
Local LLM access via Ollama — offline-first, with deterministic fallback.

Scope: language tasks only (phrasing Buddy answers, summarising evidence,
explaining a plan). NEVER use this for safety decisions — those are
deterministic rules in features/safety (docs/ARCHITECTURE.md).

Guarantees:
  * Local only. The host must be a loopback address; anything else is
    rejected at construction, so no prompt can leave the machine.
  * Never raises for availability problems. If the `ollama` library is not
    installed, the server is not running, the model is not pulled, or a
    request fails/times out, the client sets `fallback_mode = True` and
    answers with simple deterministic logic instead.
  * Recovers automatically: availability is re-checked after
    `recheck_interval_sec`, so starting Ollama mid-session is picked up.

Usage:
    client = OllamaClient()
    result = client.generate("Summarise: ...", fallback=lambda p: "No summary available.")
    result.text, result.source   # source is "ollama" or "fallback"
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS_PATH = os.path.join(ROOT, "config", "settings.yaml")

try:
    import ollama  # optional dependency; absence means fallback mode
except ImportError:  # pragma: no cover - depends on environment
    ollama = None

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
DEFAULTS: Dict[str, Any] = {
    "model": "mistral:latest",
    "host": "http://localhost:11434",
    "timeout_sec": 30,
    "recheck_interval_sec": 60,
    "temperature": 0.2,
    "max_tokens": 256,
}

SOURCE_OLLAMA, SOURCE_FALLBACK = "ollama", "fallback"


@dataclass
class GenerationResult:
    text: str
    source: str                   # "ollama" | "fallback"
    model: Optional[str]          # None when the fallback answered
    latency_ms: float
    error: Optional[str] = None   # why the fallback was used, if it was


def load_llm_settings(path: str = SETTINGS_PATH) -> Dict[str, Any]:
    """`llm:` section of config/settings.yaml merged over DEFAULTS (missing file/section is fine)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            section = (yaml.safe_load(f) or {}).get("llm") or {}
    except OSError:
        section = {}
    return {**DEFAULTS, **section}


def simple_fallback(prompt: str, max_chars: int = 280) -> str:
    """
    Default non-LLM answer: the first meaningful lines of the prompt's
    context, clearly labelled. Deterministic and never invents content.
    """
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    body = " ".join(lines[1:] if len(lines) > 1 else lines)
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0] + "..."
    return f"[Offline mode - AI assistant unavailable] {body}" if body else \
           "[Offline mode - AI assistant unavailable]"


class OllamaClient:
    """Offline-first client for a local Ollama server."""

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None,
                 timeout_sec: Optional[float] = None, recheck_interval_sec: Optional[float] = None,
                 options: Optional[Dict[str, Any]] = None):
        cfg = load_llm_settings()
        self.model = model or cfg["model"]
        self.host = host or cfg["host"]
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else cfg["timeout_sec"])
        self.recheck_interval_sec = float(recheck_interval_sec if recheck_interval_sec is not None
                                          else cfg["recheck_interval_sec"])
        self.options = {"temperature": cfg["temperature"], "num_predict": cfg["max_tokens"], **(options or {})}

        hostname = urlparse(self.host).hostname
        if hostname not in LOCAL_HOSTS:
            raise ValueError(f"OllamaClient is local-only; refusing non-loopback host {self.host!r}")

        self.fallback_mode = True          # pessimistic until proven available
        self.last_error: Optional[str] = None
        self._last_check = float("-inf")
        self._client = None
        if ollama is not None:
            self._client = ollama.Client(host=self.host, timeout=self.timeout_sec)
        else:
            self.last_error = "ollama Python library not installed (pip install ollama)"

    # ── availability ──────────────────────────────────────────────────────────

    def _enter_fallback(self, reason: str) -> None:
        self.fallback_mode = True
        self.last_error = reason
        self._last_check = time.monotonic()

    def is_available(self, force: bool = False) -> bool:
        """
        True when the server responds and the configured model is pulled.
        Result is cached for `recheck_interval_sec` unless `force=True`.
        """
        if self._client is None:
            return False
        if not force and time.monotonic() - self._last_check < self.recheck_interval_sec:
            return not self.fallback_mode

        try:
            listing = self._client.list()
        except Exception as exc:  # connection refused, timeout, DNS, HTTP errors
            self._enter_fallback(f"Ollama server not reachable at {self.host}: {exc}")
            return False

        models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
        names = {(m.get("model") or m.get("name")) if isinstance(m, dict) else
                 (getattr(m, "model", None) or getattr(m, "name", None)) for m in models}
        wanted = {self.model, self.model if ":" in self.model else f"{self.model}:latest"}
        if not names & wanted:
            self._enter_fallback(f"model {self.model!r} not pulled (run: ollama pull {self.model})")
            return False

        self.fallback_mode = False
        self.last_error = None
        self._last_check = time.monotonic()
        return True

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, system: Optional[str] = None,
                 fallback: Optional[Callable[[str], str]] = None,
                 options: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """
        Generate text locally. On any availability or request failure the
        `fallback` callable (default: simple_fallback) answers instead and
        `fallback_mode` is set. Never raises for connectivity problems.
        """
        fallback = fallback or simple_fallback
        start = time.perf_counter()

        if self.is_available():
            try:
                resp = self._client.generate(model=self.model, prompt=prompt, system=system or "",
                                             options={**self.options, **(options or {})})
                text = resp.get("response", "") if isinstance(resp, dict) else getattr(resp, "response", "")
                return GenerationResult(text=text.strip(), source=SOURCE_OLLAMA, model=self.model,
                                        latency_ms=round((time.perf_counter() - start) * 1000, 1))
            except Exception as exc:  # server died mid-session, timeout, model error
                self._enter_fallback(f"generation failed: {exc}")

        return GenerationResult(text=fallback(prompt), source=SOURCE_FALLBACK, model=None,
                                latency_ms=round((time.perf_counter() - start) * 1000, 1),
                                error=self.last_error)

    def status(self) -> Dict[str, Any]:
        """Snapshot for dashboards/logs."""
        return {"model": self.model, "host": self.host, "fallback_mode": self.fallback_mode,
                "library_installed": ollama is not None, "last_error": self.last_error}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    client = OllamaClient()
    print("available:", client.is_available(force=True))
    print("status:", client.status())
    r = client.generate("Explain the next task to the operator.\nTask T00010: excavation at LOC003, "
                        "ETA 101 min, deadline met with 45.6 h margin.")
    print(f"[{r.source}] {r.text}  ({r.latency_ms} ms)")
