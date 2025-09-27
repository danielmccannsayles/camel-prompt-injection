import asyncio
import logging
import uuid

from .base_models import BasePLM, BaseQLM, JsonSchema, Message
from .websocket_wrapper import WebSocketClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CamelClient:
    """Client that connects to CamelServer"""

    def __init__(
        self,
        plm: BasePLM,
        qlm: BaseQLM,
        tools: list | None = None,
        server_host: str = "localhost",
        server_port: int = 8765,
    ):
        """
        Initialize CamelClient.

        Args:
            plm: Privileged LLM instance that takes message list and returns response
            qlm: Quarantined LLM instance that takes prompt and schema and returns structured data
            tools: List of tools (TODO: define format)
            server_host: CamelServer host
            server_port: CamelServer port
        """
        self.websocket_client = WebSocketClient(server_host, server_port)
        self.plm = plm
        self.qlm = qlm
        self.tools = tools or []
        self.client_id = str(uuid.uuid4())

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self.websocket_client.websocket is not None

    async def connect(self, max_retries: int = 3):
        """Connect to the CamelServer with retry logic."""
        for attempt in range(max_retries):
            try:
                await self.websocket_client.connect(self.client_id)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(1)

    async def disconnect(self):
        """Disconnect from the CamelServer."""
        await self.websocket_client.disconnect()

    # TODO: handle tool calls!
    async def _handle_server_callbacks(self):
        """Handle callbacks from server for PLM/QLM calls."""
        async for data in self.websocket_client.listen_for_messages():
            message_type = data.get("type")

            if message_type == "plm_call":
                # Server is requesting a PLM call
                message_dicts = data["messages"]
                messages = [Message.from_dict(msg_dict) for msg_dict in message_dicts]
                try:
                    result = self.plm(messages)
                    response = {"type": "plm_response", "result": result}
                except Exception as e:
                    response = {"type": "plm_response", "error": str(e)}
                await self.websocket_client.respond(response)

            elif message_type == "qlm_call":
                # Server is requesting a QLM call
                prompt = data["prompt"]
                output_schema: JsonSchema = data["output_schema"]
                try:
                    result = self.qlm(prompt, output_schema)
                    response = {"type": "qlm_response", "result": result}
                except Exception as e:
                    response = {"type": "qlm_response", "error": str(e)}
                await self.websocket_client.respond(response)

            elif message_type == "query_response":
                # This is the final response to our query - return it
                return data["result"]

            elif message_type == "error":
                raise Exception(f"Server error: {data['message']}")

        return None

    async def query(self, query: str) -> str:
        """
        Send a query to the server and get the result.

        Args:
            query: The query string to process

        Returns:
            The result from the server
        """
        if not self.is_connected:
            await self.connect()

        # Send query to server and get acknowledgment
        query_message = {"type": "query", "query": query, "tools": self.tools}
        ack = await self.websocket_client.send_message(query_message)
        if ack.get("type") != "query_received":
            raise Exception(f"Unexpected server response: {ack}")

        # Handle server callbacks and wait for final response
        result = await self._handle_server_callbacks()
        return result or "No response received"
