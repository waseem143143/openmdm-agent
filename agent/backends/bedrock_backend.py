"""
openmdm-agent · agent/backends/bedrock_backend.py
AWS Bedrock backend — inference stays inside customer's AWS account.

Data residency: Customer's AWS region (e.g. us-east-1)
HIPAA eligible: Yes — covered under AWS HIPAA BAA
PHI boundary:   Never leaves AWS account VPC

Setup required:
  1. AWS account with Bedrock enabled
  2. IAM role with bedrock:InvokeModel permission
  3. Model access approved in Bedrock console
  4. pip install boto3

Environment variables:
  AWS_REGION          = us-east-1
  BEDROCK_MODEL_ID    = anthropic.claude-haiku-4-5 (or other)
  AWS credentials via: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
                    or: IAM instance role (recommended for EC2/ECS)
"""

from __future__ import annotations
import json
from agent.backends.base import LLMBackend


class BedrockBackend(LLMBackend):
    """
    AWS Bedrock backend — PHI stays inside customer's AWS account.
    Identical prompt/response contract to AnthropicBackend.
    """

    def __init__(
        self,
        region:   str = "us-east-1",
        model_id: str = "anthropic.claude-haiku-4-5",
    ):
        try:
            import boto3
            self._client   = boto3.client(
                "bedrock-runtime",
                region_name=region,
            )
            self._model_id = model_id
            self._region   = region
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Bedrock backend. "
                "Run: pip install boto3"
            )

    def decide(self, system_prompt: str, user_prompt: str) -> str:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system":  system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        })
        response = self._client.invoke_model(
            modelId     = self._model_id,
            body        = body,
            contentType = "application/json",
            accept      = "application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    @property
    def display_name(self) -> str:
        return f"AWS Bedrock ({self._region})"

    @property
    def model_id(self) -> str:
        return f"bedrock/{self._model_id}"

    @property
    def data_residency(self) -> str:
        return f"AWS {self._region} (Customer Account)"

    @property
    def hipaa_eligible(self) -> bool:
        return True
