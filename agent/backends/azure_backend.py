"""
openmdm-agent · agent/backends/azure_backend.py
Azure OpenAI backend — inference stays inside customer's Azure subscription.

Data residency: Customer's Azure region (e.g. eastus)
HIPAA eligible: Yes — covered under Microsoft HIPAA BAA
PHI boundary:   Never leaves Azure subscription boundary

Setup required:
  1. Azure subscription with OpenAI resource deployed
  2. GPT-4 or GPT-4o model deployed in Azure OpenAI Studio
  3. pip install openai

Environment variables:
  AZURE_OPENAI_ENDPOINT   = https://your-resource.openai.azure.com/
  AZURE_OPENAI_API_KEY    = your-azure-api-key
  AZURE_OPENAI_DEPLOYMENT = gpt-4o (your deployment name)
  AZURE_OPENAI_API_VERSION= 2024-02-01
"""

from __future__ import annotations
from agent.backends.base import LLMBackend


class AzureOpenAIBackend(LLMBackend):
    """
    Azure OpenAI backend — PHI stays inside customer's Azure subscription.
    Supports GPT-4, GPT-4o, and other Azure-deployed models.
    """

    def __init__(
        self,
        endpoint:    str,
        api_key:     str,
        deployment:  str = "gpt-4o",
        api_version: str = "2024-02-01",
        region:      str = "eastus",
    ):
        try:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint = endpoint,
                api_key        = api_key,
                api_version    = api_version,
            )
            self._deployment = deployment
            self._region     = region
        except ImportError:
            raise ImportError(
                "openai package is required for Azure OpenAI backend. "
                "Run: pip install openai"
            )

    def decide(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model    = self._deployment,
            messages = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            max_tokens  = 1024,
            temperature = 0,
        )
        return response.choices[0].message.content

    @property
    def display_name(self) -> str:
        return f"Azure OpenAI ({self._region})"

    @property
    def model_id(self) -> str:
        return f"azure/{self._deployment}"

    @property
    def data_residency(self) -> str:
        return f"Azure {self._region} (Customer Subscription)"

    @property
    def hipaa_eligible(self) -> bool:
        return True
