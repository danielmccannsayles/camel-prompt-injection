import asyncio
import logging

import anthropic
from locallms import LocalPLM, LocalQLM
from server import CamelClient, CamelServer
from server.base_models import JsonSchema, Message, make_error_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


async def test_camel_flow():
    """Test the basic CamelServer + CamelClient flow."""

    # Start server in background
    server = CamelServer(host="localhost", port=8766)
    server_task = asyncio.create_task(server.start())

    # Give server time to start
    await asyncio.sleep(2)
    try:
        # Initialize Anthropic client
        anthropic_client = anthropic.Anthropic()

        # Define PLM class

        # Create instances
        my_plm = LocalPLM(anthropic_client)
        my_qlm = LocalQLM(anthropic_client)

        # Create client with same port as server
        camel_client = CamelClient(my_plm, my_qlm, [], server_host="localhost", server_port=8766)

        # Test PLM query
        logger.info("Sending test query to client...")
        result = await camel_client.query("What is 2+2?")

        logger.info(f"Received result: {result}")

        # Test multi-message PLM handling
        logger.info("Testing PLM with multiple messages...")
        test_messages = [
            Message(role="system", content="You are a helpful coding assistant."),
            Message(role="user", content="Write Python code to calculate 5 * 3"),
            Message(role="assistant", content="print(5 * 3)"),
            Message(role="user", content="Good! Now add error handling to make it more robust."),
        ]

        multi_msg_result = my_plm(test_messages)
        logger.info(f"Multi-message PLM result: {multi_msg_result[:100]}...")
        print(f"Multi-message PLM Test: {multi_msg_result[:200]}...")

        # Test error message format (simulating code execution failure)
        logger.info("Testing PLM with error message format...")
        broken_code = "print(5 / 0)"
        error_msg = "ZeroDivisionError: division by zero"
        error_messages = make_error_messages(broken_code, error_msg)

        # Add system message and user query to simulate full context
        full_error_context = [
            Message(role="system", content="You are a helpful coding assistant."),
            Message(role="user", content="Write Python code to calculate 5 divided by 0"),
            *error_messages,
        ]

        error_fix_result = my_plm(full_error_context)
        logger.info(f"Error fix PLM result: {error_fix_result[:100]}...")
        print(f"Error Fix PLM Test: {error_fix_result[:200]}...")

        # Test QLM directly to verify dynamic schema enforcement
        logger.info("Testing QLM with basic math schema...")
        math_schema: JsonSchema = {
            "type": "object",
            "properties": {
                "answer": {"type": "integer", "description": "The numerical answer"},
                "explanation": {"type": "string", "description": "Brief explanation"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "explanation", "confidence"],
        }

        qlm_result1 = my_qlm("What is 5 + 3?", math_schema)
        logger.info(f"QLM math result: {qlm_result1}")
        print(f"QLM Math Test: {qlm_result1}")

        # Test with different schema - invoice-like structure
        logger.info("Testing QLM with invoice schema...")
        invoice_schema: JsonSchema = {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string", "description": "Invoice identifier"},
                "amount": {"type": "number", "description": "Total amount"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "number"},
                        },
                        "required": ["description", "quantity", "unit_price"],
                    },
                },
            },
            "required": ["invoice_number", "amount", "items"],
        }

        qlm_result2 = my_qlm("Create an invoice for 3 apples at $1.50 each and 2 oranges at $2.00 each", invoice_schema)
        logger.info(f"QLM invoice result: {qlm_result2}")
        print(f"QLM Invoice Test: {qlm_result2}")

        # Cleanup
        await camel_client.disconnect()

        print("✅ Test completed successfully!")
        print(f"PLM Result: {result}")
        print(f"Multi-message PLM Result: {multi_msg_result[:200]}...")
        print(f"Error Fix PLM Result: {error_fix_result[:200]}...")
        print(f"QLM Math Result: {qlm_result1}")
        print(f"QLM Invoice Result: {qlm_result2}")

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
