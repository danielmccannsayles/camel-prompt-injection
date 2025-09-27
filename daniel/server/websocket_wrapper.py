import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketServer:
    """Basic WebSocket server wrapper"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        handle_query: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
    ):
        self.host = host
        self.port = port
        self.clients: dict[str, Any] = {}  # websocket connections
        self.handle_query = handle_query

    async def register_client(self, websocket, client_id: str):
        """Register a new client connection."""
        self.clients[client_id] = websocket
        logger.info(f"Client {client_id} registered")

    async def unregister_client(self, client_id: str):
        """Unregister a client connection."""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Client {client_id} unregistered")

    async def send_to_client(self, client_id: str, message: dict[str, Any]) -> dict[str, Any]:
        """Send message to client and wait for response."""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not connected")

        websocket = self.clients[client_id]
        await websocket.send(json.dumps(message))
        response = await websocket.recv()
        return json.loads(response)

    async def send_to_client_only(self, client_id: str, message: dict[str, Any]):
        """Send message to client without waiting for response."""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not connected")

        websocket = self.clients[client_id]
        await websocket.send(json.dumps(message))

    async def handle_client(self, websocket):
        """Handle a client connection"""
        client_id = None
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")

                    if message_type == "register":
                        client_id = data["client_id"]
                        await self.register_client(websocket, client_id)
                        await websocket.send(json.dumps({"type": "registered", "client_id": client_id}))

                    elif message_type == "query":
                        if not client_id:
                            await websocket.send(json.dumps({"type": "error", "message": "Client not registered"}))
                            continue

                        # This is where the server logic will handle the query
                        if self.handle_query:
                            await self.handle_query(client_id, data)
                        else:
                            await websocket.send(
                                json.dumps({"type": "error", "message": "No query handler configured"})
                            )

                    else:
                        await websocket.send(
                            json.dumps({"type": "error", "message": f"Unknown message type: {message_type}"})
                        )

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await websocket.send(json.dumps({"type": "error", "message": str(e)}))

        except Exception as e:
            logger.info(f"Client {client_id} disconnected: {e}")
        finally:
            if client_id:
                await self.unregister_client(client_id)

    async def start(self):
        """Start the WebSocket server using modern asyncio websockets."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        async with serve(self.handle_client, self.host, self.port) as server:
            await server.serve_forever()


class WebSocketClient:
    """Basic WebSocket client wrapper"""

    def __init__(self, server_host: str = "localhost", server_port: int = 8765):
        self.server_host = server_host
        self.server_port = server_port
        self.websocket = None
        self.client_id = None

    async def connect(self, client_id: str):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await connect(f"ws://{self.server_host}:{self.server_port}")
            self.client_id = client_id

            # Register with server
            register_message = {"type": "register", "client_id": client_id}
            await self.websocket.send(json.dumps(register_message))

            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data["type"] != "registered":
                raise ValueError(f"Failed to register: {response_data}")

            logger.info(f"Connected to server as client {client_id}")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def send_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send message to server and get response."""
        if not self.websocket:
            raise ValueError("Not connected to server")

        await self.websocket.send(json.dumps(message))
        response = await self.websocket.recv()
        return json.loads(response)

    async def respond(self, message: dict[str, Any]):
        """Send message to server without waiting for response. Used for responding to server, e.g. qLM response"""
        if not self.websocket:
            raise ValueError("Not connected to server")
        await self.websocket.send(json.dumps(message))

    async def listen_for_messages(self):
        """Listen for messages from server (for callbacks)."""
        try:
            if not self.websocket:
                raise Exception("No clientConnection")
            async for message in self.websocket:
                data = json.loads(message)
                yield data
        except Exception as e:
            logger.info(f"Connection to server closed: {e}")
            return
