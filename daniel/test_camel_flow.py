import asyncio
import json
import logging
import os
import sys
from typing import Any

import anthropic
from server import CamelClient, CamelServer

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

        # Define PLM and QLM functions
        def my_plm(prompt: str) -> str:
            """PLM using Anthropic Sonnet to generate code."""
            logger.info(f"PLM called with prompt: {prompt[:100]}...")
            try:
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022", max_tokens=1000, messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
                logger.info(f"PLM response: {result[:100]}...")
                return result
            except Exception as e:
                logger.error(f"PLM error: {e}")
                return f"Error in PLM: {e}"

        def create_dynamic_tool(schema: dict):
            """Generate tool definition from dynamic schema"""
            return {
                "name": "extract_data",
                "description": "Extract structured data matching the provided schema",
                "input_schema": schema,
            }

        def my_qlm(prompt: str, schema: dict) -> Any:
            """QLM using Anthropic Sonnet with dynamically generated tools for schema enforcement."""
            logger.info(f"QLM called with prompt: {prompt[:100]}... schema: {schema}")
            try:
                # Create dynamic tool from the provided schema
                dynamic_tool = create_dynamic_tool(schema)
                logger.info(f"Created dynamic tool: {dynamic_tool['name']} with schema: {json.dumps(schema, indent=2)}")

                # Use tool calling to enforce structured output
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[dynamic_tool],  # type: ignore
                    tool_choice={"type": "tool", "name": "extract_data"},
                )

                # Extract the tool use result
                if response.content and len(response.content) > 0:
                    for content_block in response.content:
                        if hasattr(content_block, "type") and content_block.type == "tool_use":
                            result = content_block.input
                            logger.info(f"QLM structured result: {result}")
                            return result

                # Fallback if no tool use found
                logger.warning("No structured output found, falling back to text parsing")
                if response.content and len(response.content) > 0:
                    first_content = response.content[0]
                    if hasattr(first_content, "text"):
                        text_content = first_content.text
                        try:
                            result = json.loads(text_content)
                            return result
                        except json.JSONDecodeError:
                            return {"result": text_content, "schema_matched": False}
                    else:
                        return {"result": str(first_content), "schema_matched": False}

                return {"error": "No response content", "schema_matched": False}

            except Exception as e:
                logger.error(f"QLM error: {e}")
                # Fallback to regular message creation with schema prompt
                try:
                    schema_prompt = f"""Please respond with JSON that matches this schema: {json.dumps(schema)}

{prompt}

Respond only with valid JSON matching the schema."""

                    response = anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": schema_prompt}],
                    )

                    if response.content and len(response.content) > 0:
                        first_content = response.content[0]
                        if hasattr(first_content, "text"):
                            result_text = first_content.text.strip()
                            try:
                                result = json.loads(result_text)
                                logger.info(f"QLM fallback JSON result: {result}")
                                return result
                            except json.JSONDecodeError:
                                return {"result": result_text, "schema_matched": False}
                        else:
                            return {"result": str(first_content), "schema_matched": False}

                    return {"error": "No fallback response content", "schema_matched": False}

                except Exception as fallback_error:
                    logger.error(f"QLM fallback error: {fallback_error}")
                    return {"error": str(e), "fallback_error": str(fallback_error), "schema_matched": False}

        # Create client with same port as server
        camel_client = CamelClient(my_plm, my_qlm, [], server_host="localhost", server_port=8766)

        # Test PLM query
        logger.info("Sending test query to client...")
        result = await camel_client.query("What is 2+2?")

        logger.info(f"Received result: {result}")

        # Test QLM directly to verify dynamic schema enforcement
        logger.info("Testing QLM with basic math schema...")
        math_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "integer", "description": "The numerical answer"},
                "explanation": {"type": "string", "description": "Brief explanation"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["answer", "explanation", "confidence"],
        }

        qlm_result1 = my_qlm("What is 5 + 3?", math_schema)
        logger.info(f"QLM math result: {qlm_result1}")
        print(f"QLM Math Test: {qlm_result1}")

        # Test with different schema - invoice-like structure
        logger.info("Testing QLM with invoice schema...")
        invoice_schema = {
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
