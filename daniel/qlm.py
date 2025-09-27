"""Quarantined Language Model with schema enforcement using Anthropic Claude."""

from typing import Any

import anthropic
from server.base_models import BaseQLM, JsonSchema


class QLM(BaseQLM):
    """QLM using Anthropic Sonnet with dynamic schema enforcement."""

    def __init__(self, client=None, model="claude-3-5-sonnet-20241022"):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def __call__(self, prompt: str, schema: JsonSchema) -> Any:
        """Query the QLM with schema enforcement."""
        # Create dynamic tool from the provided schema
        tool = {
            "name": "extract_data",
            "description": "Extract structured data matching the provided schema",
            "input_schema": schema,
        }

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],  # type: ignore
            tool_choice={"type": "tool", "name": "extract_data"},
        )

        return response.content[0].input  # type: ignore
