"""
openmdm-agent · agent/backends/ollama_backend.py
Ollama local backend — completely air-gapped, zero network calls.

Data residency: Customer's on-prem server or private cloud VM
HIPAA eligible: Yes — data never leaves the machine
PHI boundary:   Air-gapped, no internet required

Setup required:
  1. Install Ollama: https://ollama.com
  2. Pull a model: ollama pull llama3.1
  3. Ollama runs automatically as a local server on port 11434

Environment variables:
  OLLAMA_BASE_URL = http://localhost:11434  (default)
  OLLAMA_MODEL    = llama3.1               (or mistral, phi3, etc.)

Recommended models for MDM stewardship:
  llama3.1      — Best reasoning quality  (4.7GB)
  mistral       — Fast, good JSON output  (4.1GB)
  phi3:medium   — Smallest, still capable (2.3GB)
"""

from __future__ import annotations
import json
import urllib.request
from agent.backends.base import LLMBackend


class OllamaBackend(LLMBackend):
    """
    Fully local LLM via Ollama — completely air-gapped.
    PHI never leaves the machine. No API key required.
    """

    def __init__(
        self,
        model:    str = "llama3.1",
        base_url: str = "http://localhost:11434",
    ):
        self._model    = model
        self._base_url = base_url.rstrip("/")

    def decide(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps({
            "model":  self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["message"]["content"]

    @property
    def display_name(self) -> str:
        return f"Ollama Local ({self._model})"

    @property
    def model_id(self) -> str:
        return f"ollama/{self._model}"

    @property
    def data_residency(self) -> str:
        return "On-Premises (Air-Gapped)"

    @property
    def hipaa_eligible(self) -> bool:
        return True
