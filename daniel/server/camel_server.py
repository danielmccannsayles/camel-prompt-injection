import logging
import os
import sys
from typing import Any, TypeVar

from agentdojo import functions_runtime
from pydantic import BaseModel

from .base_models import JsonSchema, Message
from .utils import deserialize_functions_runtime
from .websocket_wrapper import WebSocketServer

_T = TypeVar("_T", bound=str | int | float | BaseModel)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directories to path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from camel.system_prompt_generator import default_system_prompt_generator


class CamelServer:
    """Server.

    1. Implements privileged LLM logic using client PLM/QLM via websockets.
    2. Handles tool calls (TODO)
    3. Interprets code (TODO)
    4. Manages capabilities (TODO)"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.websocket_server = WebSocketServer(host, port, handle_query=self.handle_query)
        self.client_runtimes: dict[str, functions_runtime.FunctionsRuntime] = {}
        self.client_function_metadata: dict[str, dict[str, dict[str, Any]]] = {}

    async def call_plm(self, client_id: str, messages: list[Message]) -> str:
        """Call the client's PLM function via websocket."""
        message_dicts = [msg.to_dict() for msg in messages]
        message = {"type": "plm_call", "messages": message_dicts}
        response = await self.websocket_server.send_to_client(client_id, message)

        if response["type"] != "plm_response":
            raise ValueError(f"Expected plm_response, got {response['type']}")

        if "error" in response:
            raise Exception(f"PLM error: {response['error']}")

        return response["result"]

    async def call_qlm(self, client_id: str, prompt: str, output_schema: JsonSchema) -> Any:
        """Call the client's QLM function via websocket."""
        message = {"type": "qlm_call", "prompt": prompt, "output_schema": output_schema}
        response = await self.websocket_server.send_to_client(client_id, message)

        if response["type"] != "qlm_response":
            raise ValueError(f"Expected qlm_response, got {response['type']}")

        if "error" in response:
            raise Exception(f"QLM error: {response['error']}")

        return response["result"]

    async def call_function(self, client_id: str, function_name: str, args: dict[str, Any]) -> Any:
        """Call a client function via websocket."""
        message = {"type": "function_call", "function_name": function_name, "args": args}
        response = await self.websocket_server.send_to_client(client_id, message)

        if response["type"] != "function_response":
            raise ValueError(f"Expected function_response, got {response['type']}")

        if "error" in response:
            raise Exception(f"Function error: {response['error']}")

        return response["result"]

    async def handle_query(self, client_id: str, data: dict[str, Any]):
        """Handle a query by implementing privileged LLM logic."""
        query = data["query"]
        logger.info(f"Handling query from client {client_id}: {query}")

        # Store the client's function runtime if provided
        if "functions_runtime" in data:
            runtime_data = data["functions_runtime"]
            runtime, function_metadata = deserialize_functions_runtime(runtime_data)
            self.client_runtimes[client_id] = runtime
            self.client_function_metadata[client_id] = function_metadata
            logger.info(f"Registered {len(runtime.functions)} functions for client {client_id}")

        # Send immediate acknowledgment
        ack_message = {"type": "query_received"}
        await self.websocket_server.send_to_client_only(client_id, ack_message)

        try:
            # Step 1: Get the client's function runtime and create system prompt
            runtime = self.client_runtimes.get(client_id, functions_runtime.FunctionsRuntime())
            system_prompt = default_system_prompt_generator(runtime.functions.values())

            # Step 2: Create message list and call client's PLM to generate code
            messages = [Message(role="system", content=system_prompt), Message(role="user", content=query)]
            code = await self.call_plm(client_id, messages)

            logger.info(f"Generated code: {code}")

            # Step 3: Test function calling (for now, manually test if we have functions)
            function_results = []
            if client_id in self.client_function_metadata and "add_numbers" in self.client_function_metadata[client_id]:
                # Test calling the add_numbers function if available
                try:
                    func_result = await self.call_function(client_id, "add_numbers", {"a": 2, "b": 3})
                    function_results.append(f"add_numbers(2, 3) = {func_result}")
                    logger.info(f"Function call result: {func_result}")
                except Exception as e:
                    logger.error(f"Function call failed: {e}")
                    function_results.append(f"add_numbers failed: {e}")

            if client_id in self.client_function_metadata and "get_weather" in self.client_function_metadata[client_id]:
                # Test calling the get_weather function if available
                try:
                    func_result = await self.call_function(client_id, "get_weather", {"city": "San Francisco"})
                    function_results.append(f"get_weather('San Francisco') = {func_result}")
                    logger.info(f"Function call result: {func_result}")
                except Exception as e:
                    logger.error(f"Function call failed: {e}")
                    function_results.append(f"get_weather failed: {e}")

            # Step 4: Combine code and function results
            # TODO: Implement proper code execution with tool calls and QLM callbacks
            result_parts = [f"Generated code: {code}"]
            if function_results:
                result_parts.append(f"Function test results: {', '.join(function_results)}")
            result = "\n".join(result_parts)

            # Send response back to client
            response_message = {"type": "query_response", "result": result}
            await self.websocket_server.send_to_client_only(client_id, response_message)

        except Exception as e:
            logger.error(f"Error handling query: {e}")
            import traceback

            traceback.print_exc()

            error_message = {"type": "query_response", "result": f"Error: {e!s}"}
            await self.websocket_server.send_to_client_only(client_id, error_message)

    async def start(self):
        """Start the WebSocket server."""
        await self.websocket_server.start()
