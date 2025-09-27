import logging
import os
import sys
from typing import Any, TypeVar

from pydantic import BaseModel

from .base_models import JsonSchema, Message
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

    async def handle_query(self, client_id: str, data: dict[str, Any]):
        """Handle a query by implementing privileged LLM logic."""
        query = data["query"]
        logger.info(f"Handling query from client {client_id}: {query}")

        # Send immediate acknowledgment
        ack_message = {"type": "query_received"}
        await self.websocket_server.send_to_client_only(client_id, ack_message)

        try:
            # Step 1: Create system prompt for code generation
            system_prompt = default_system_prompt_generator()

            # Step 2: Create message list and call client's PLM to generate code
            messages = [Message(role="system", content=system_prompt), Message(role="user", content=query)]
            code = await self.call_plm(client_id, messages)

            logger.info(f"Generated code: {code}")

            # Step 3: Execute the code (simplified version for now)
            # TODO: Implement proper code execution with tool calls and QLM callbacks
            # For now, just return the generated code as the result
            result = f"Generated code: {code}"

            # Step 4: Handle any tool calls or QLM calls that the code needs
            # TODO: Parse the code for function calls and execute them via callbacks

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
