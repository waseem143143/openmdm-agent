"""
openmdm-agent · agent/backends/__init__.py
Backend factory — creates the right backend from config.
"""

from agent.backends.base             import LLMBackend
from agent.backends.anthropic_backend import AnthropicBackend
from agent.backends.bedrock_backend   import BedrockBackend
from agent.backends.azure_backend     import AzureOpenAIBackend
from agent.backends.ollama_backend    import OllamaBackend
from agent.backends.mock_backend      import MockBackend


BACKEND_OPTIONS = {
    "mock":      "AWS Bedrock — Demo Mode (No API Key Needed)",
    "anthropic": "Anthropic API (Direct — Dev/Demo Only)",
    "bedrock":   "AWS Bedrock (Enterprise — Customer AWS Account)",
    "azure":     "Azure OpenAI (Enterprise — Customer Azure Subscription)",
    "ollama":    "Ollama Local (Air-Gapped — On-Premises)",
}

BACKEND_RESIDENCY = {
    "mock":      ("AWS us-east-1 (Simulated)", True,  "🟢 No data transmitted"),
    "anthropic": ("Anthropic Cloud (External)", False, "🔴 PHI crosses public internet"),
    "bedrock":   ("AWS Region (Customer Account)", True,  "🟢 Stays inside AWS VPC"),
    "azure":     ("Azure Region (Customer Subscription)", True,  "🟢 Stays inside Azure boundary"),
    "ollama":    ("On-Premises (Air-Gapped)", True,  "🟢 Never leaves the machine"),
}


def create_backend(
    backend_type: str,
    api_key:      str  = "",
    region:       str  = "us-east-1",
    ollama_model: str  = "llama3.1",
    azure_endpoint: str = "",
    azure_key:    str  = "",
    azure_deploy: str  = "gpt-4o",
) -> LLMBackend:
    """
    Factory function — returns the right backend instance from config.

    Args:
        backend_type:    One of: mock | anthropic | bedrock | azure | ollama
        api_key:         Anthropic API key (for anthropic backend)
        region:          AWS/Azure region (for bedrock/azure/mock backends)
        ollama_model:    Ollama model name (for ollama backend)
        azure_endpoint:  Azure OpenAI endpoint URL
        azure_key:       Azure OpenAI API key
        azure_deploy:    Azure OpenAI deployment name
    """
    if backend_type == "mock":
        return MockBackend(region=region)

    elif backend_type == "anthropic":
        if not api_key:
            raise ValueError("Anthropic API key required for anthropic backend")
        return AnthropicBackend(api_key=api_key)

    elif backend_type == "bedrock":
        return BedrockBackend(region=region)

    elif backend_type == "azure":
        if not azure_endpoint or not azure_key:
            raise ValueError("Azure endpoint and API key required for azure backend")
        return AzureOpenAIBackend(
            endpoint   = azure_endpoint,
            api_key    = azure_key,
            deployment = azure_deploy,
            region     = region,
        )

    elif backend_type == "ollama":
        return OllamaBackend(model=ollama_model)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}. "
                         f"Choose from: {list(BACKEND_OPTIONS.keys())}")


__all__ = [
    "LLMBackend", "AnthropicBackend", "BedrockBackend",
    "AzureOpenAIBackend", "OllamaBackend", "MockBackend",
    "create_backend", "BACKEND_OPTIONS", "BACKEND_RESIDENCY",
]
