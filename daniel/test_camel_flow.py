import asyncio
import logging
import os
import sys

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from server import CamelClient, CamelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_camel_flow():
    """Test the basic CamelServer + CamelClient flow."""

    # Start server in background
    server = CamelServer(host="localhost", port=8766)
    server_task = asyncio.create_task(server.start())

    # Give server time to start
    await asyncio.sleep(2)

    try:
        # Define PLM and QLM functions
        def my_plm(prompt: str) -> str:
            """Simple PLM that just echoes back with prefix."""
            logger.info(f"PLM called with prompt: {prompt[:100]}...")
            return f"PLM processed: {prompt}"

        def my_qlm(prompt: str, schema: dict) -> dict:
            """Simple QLM that returns structured data."""
            logger.info(f"QLM called with prompt: {prompt[:100]}... schema: {schema}")
            return {"result": f"QLM processed: {prompt}", "schema_type": schema.get("type", "unknown")}

        # Create client with same port as server
        client = CamelClient(my_plm, my_qlm, [], server_host="localhost", server_port=8766)

        # Test query
        logger.info("Sending test query to client...")
        result = await client.query("What is 2+2?")

        logger.info(f"Received result: {result}")

        # Cleanup
        await client.disconnect()

        print("✅ Test completed successfully!")
        print(f"Result: {result}")

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
    asyncio.run(test_camel_flow())
