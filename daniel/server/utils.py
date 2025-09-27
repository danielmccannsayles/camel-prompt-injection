"""Utility functions for serialization and type conversion."""

import logging
from typing import Any

from agentdojo import functions_runtime as ad_runtime

logger = logging.getLogger(__name__)


def serialize_functions_runtime(runtime: ad_runtime.FunctionsRuntime) -> list[dict[str, Any]]:
    """Serialize the functions runtime for websocket transmission."""
    return [
        {
            "name": func.name,
            "description": func.description,
            "parameters_schema": func.parameters.model_json_schema(),
            "full_docstring": func.full_docstring,
            "return_type": str(func.return_type) if func.return_type else None,
        }
        for func in runtime.functions.values()
    ]


def serialize_function_result(result: Any) -> Any:
    """Serialize function result for websocket transmission.

    Handles arbitrary return types by converting them to JSON-serializable forms.
    """
    # Handle common types that are already JSON serializable
    if result is None or isinstance(result, str | int | float | bool | list | dict):
        return result

    # Handle Pydantic models
    if hasattr(result, "model_dump"):
        return result.model_dump()

    # Handle objects with dict() method
    if hasattr(result, "dict"):
        return result.dict()

    # Handle dataclasses
    if hasattr(result, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(result)

    # Handle other objects by converting to string
    logger.warning(f"Serializing complex object {type(result)} as string")
    return str(result)


def deserialize_functions_runtime(
    data: list[dict[str, Any]],
) -> tuple[ad_runtime.FunctionsRuntime, dict[str, dict[str, Any]]]:
    """Deserialize functions runtime data from websocket.

    Returns:
        A tuple of (empty runtime, function metadata dict)
    """
    runtime = ad_runtime.FunctionsRuntime()

    # Store function metadata for reference and system prompt generation
    # Note: We can't execute these functions directly, they'll be proxied to client
    function_metadata = {func_data["name"]: func_data for func_data in data}

    logger.info(f"Received function metadata for {len(data)} functions")
    for func_data in data:
        logger.info(f"Function: {func_data['name']} - {func_data['description']}")

    return runtime, function_metadata
