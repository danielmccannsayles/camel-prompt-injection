"""Base classes for PLM and QLM implementations."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict


class Message:
    """Represents a message in the conversation."""

    def __init__(self, role: str, content: str):
        self.role = role  # "system", "user", "assistant"
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Message":
        return cls(role=data["role"], content=data["content"])


def make_error_messages(code: str, error_msg: str) -> list[Message]:
    """Create error messages to send back to the PLM when code execution fails."""
    return [
        Message(role="assistant", content=code),
        Message(
            role="user",
            content=f"""Running the code gave the following error:
{error_msg}
Provide the new code with the error fixed. Provide *all the code* so that I can directly run it.""",
        ),
    ]


# TODO: this probably needs to be stricter, e.g. always required. ALso, is this the general schema or just anthropic? idrc
# JSON Schema type definitions
class JsonSchemaProperty(TypedDict):
    """Represents a JSON Schema property definition."""

    type: Literal["string", "number", "integer", "boolean", "array", "object"]
    description: NotRequired[str]
    items: NotRequired["JsonSchemaProperty"]
    properties: NotRequired[dict[str, "JsonSchemaProperty"]]
    required: NotRequired[list[str]]


class JsonSchema(TypedDict):
    """Represents a complete JSON Schema."""

    type: Literal["object"]
    properties: dict[str, JsonSchemaProperty]
    required: NotRequired[list[str]]
    description: NotRequired[str]


class BasePLM(ABC):
    """Base class for Privileged Language Model implementations."""

    @abstractmethod
    def __call__(self, messages: list[Message]) -> str:
        """Generate code given a list of messages.

        Args:
            messages: List of Message objects representing the conversation

        Returns:
            Generated code as a string
        """
        raise NotImplementedError


class BaseQLM(ABC):
    """Base class for Quarantined Language Model implementations."""

    @abstractmethod
    def __call__(self, prompt: str, schema: JsonSchema) -> Any:
        """Query the QLM with schema enforcement.

        Args:
            prompt: The prompt to send to the QLM
            schema: JSON schema to enforce on the output

        Returns:
            Structured data matching the schema
        """
        raise NotImplementedError
