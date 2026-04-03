"""
openmdm-agent · agent/backends/anthropic_backend.py
Direct Anthropic API backend — for demo and development only.
PHI crosses the public internet. Not for production PHI workloads
without a signed BAA with Anthropic.
"""

from __future__ import annotations
import anthropic
from agent.backends.base import LLMBackend


class AnthropicBackend(LLMBackend):
    """
    Calls Anthropic's public API directly.
    Suitable for: demos, development, non-PHI workloads.
    Not suitable for: production PHI without BAA.
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model  = model

    def decide(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model      = self._model,
            max_tokens = 1024,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    @property
    def display_name(self) -> str:
        return "Anthropic API (Direct)"

    @property
    def model_id(self) -> str:
        return f"anthropic/{self._model}"

    @property
    def data_residency(self) -> str:
        return "Anthropic Cloud (External)"

    @property
    def hipaa_eligible(self) -> bool:
        return False  # requires BAA — not default
