import logging
import os
import sys
from typing import Any, TypeVar

from .websocket_wrapper import WebSocketServer

# TODO: we need to import e.g. system prompt generator
# Add the parent directories to path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pydantic import BaseModel

_T = TypeVar("_T", bound=str | int | float | BaseModel)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CamelServer:
    """Server.

    1. Implements privileged LLM logic using client PLM/QLM via websockets.
    2. Handles tool calls (TODO)
    3. Interprets code (TODO)
    4. Manages capabilities (TODO)"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.websocket_server = WebSocketServer(host, port, handle_query=self.handle_query)

    async def call_plm(self, client_id: str, prompt: str) -> str:
        """Call the client's PLM function via websocket."""
        message = {"type": "plm_call", "prompt": prompt}
        response = await self.websocket_server.send_to_client(client_id, message)

        if response["type"] != "plm_response":
            raise ValueError(f"Expected plm_response, got {response['type']}")

        if "error" in response:
            raise Exception(f"PLM error: {response['error']}")

        return response["result"]

    async def call_qlm(self, client_id: str, prompt: str, output_schema: dict[str, Any]) -> Any:
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
        tools = data.get("tools", [])
        logger.info(f"Handling query from client {client_id}: {query}")

        # Send immediate acknowledgment
        ack_message = {"type": "query_received"}
        await self.websocket_server.send_to_client_only(client_id, ack_message)

        try:
            # Step 1: Create system prompt for code generation
            # For now, use a simplified system prompt - we can expand this later
            system_prompt = """You are a helpful AI assistant that can write Python code to solve problems.
You have access to a function called query_quarantined_llm(query: str, output_schema: type) -> Any that can answer questions.
Please write Python code to solve the user's query."""

            # Step 2: Call client's PLM to generate code
            full_prompt = f"System: {system_prompt}\n\nUser: {query}"
            code = await self.call_plm(client_id, full_prompt)

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
