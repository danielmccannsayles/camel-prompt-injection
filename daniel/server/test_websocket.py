import asyncio
import json
import logging

from websocket_wrapper import WebSocketClient, WebSocketServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_websocket():
    """Test basic WebSocket server/client communication."""

    # Create server
    server = WebSocketServer(host="localhost", port=8767)

    # Override handle_query for testing
    async def test_handle_query(client_id: str, data: dict):
        query = data["query"]
        logger.info(f"Server received query from {client_id}: {query}")
        # Send response back
        response = {"type": "query_response", "result": f"Server processed: {query}"}
        await server.clients[client_id].send(json.dumps(response))

    server.handle_query = test_handle_query

    # Start server in background
    server_task = asyncio.create_task(server.start())

    # Give server time to start
    await asyncio.sleep(1)

    try:
        # Create client
        client = WebSocketClient(server_host="localhost", server_port=8767)
        await client.connect("test_client")

        # Send test query
        query_message = {"type": "query", "query": "What is 2+2?"}
        logger.info("Sending test query...")

        await client.websocket.send(json.dumps(query_message))

        # Wait for response
        response = await client.websocket.recv()
        response_data = json.loads(response)

        logger.info(f"Received response: {response_data}")

        if response_data.get("type") == "query_response":
            print("✅ Basic WebSocket test successful!")
            print(f"Result: {response_data['result']}")
        else:
            print("❌ Unexpected response type")

        await client.disconnect()

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        print("❌ Test failed!")

    finally:
        # Cancel server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(test_basic_websocket())
