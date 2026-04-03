"""
openmdm-agent · agent/backends/base.py
Abstract base class all LLM backends must implement.
Swap the backend → everything else stays identical.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """
    Abstract LLM backend interface.
    Every backend must implement decide() with identical input/output contract.
    """

    @abstractmethod
    def decide(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and return the raw response text.

        Args:
            system_prompt: The steward agent system instructions.
            user_prompt:   Field-level evidence for the candidate pair.

        Returns:
            Raw text response from the LLM (expected to be JSON).
        """
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the UI."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier logged to the audit trail."""
        ...

    @property
    def data_residency(self) -> str:
        """Data residency description shown in the UI."""
        return "External API"

    @property
    def hipaa_eligible(self) -> bool:
        """Whether this backend is HIPAA-eligible."""
        return False
