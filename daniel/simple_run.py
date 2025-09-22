import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from agentdojo import agent_pipeline, benchmark, functions_runtime
from agentdojo import logging as ad_logging
from anthropic import AsyncAnthropic

# Fix some weird pydantic model defs.
try:
    from agentdojo.benchmark import TaskResults

    TaskResults.model_rebuild()
except Exception:
    pass

from dotenv import load_dotenv
from environment import task_suite
from pydantic import BaseModel

from camel import quarantined_llm
from camel.pipeline_elements import privileged_llm
from camel.pipeline_elements.security_policies import (
    ADNoSecurityPolicyEngine,
)

load_dotenv()

# Set up minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

LOG_FILE = "text_log.txt"


def log_to_file(message):
    """Simple append"""
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {message}\n")
        f.flush()


class LoggingPrivilegedLLM(privileged_llm.PrivilegedLLM):
    """Custom PrivilegedLLM with detailed logging"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = Path("debug_execution.json")
        self.execution_log = []

    def query(self, query, runtime, env=None, messages=[], extra_args={}):
        """Override query to add logging"""
        log_to_file("=== PrivilegedLLM Query Started ===")
        log_to_file(f"Query: {query}")
        log_to_file(f"Messages count: {len(messages)}")
        log_to_file(f"Messages: {[m.get('role', 'unknown') for m in messages]}")

        log_to_file("=== Full Message Contents ===")
        for i, msg in enumerate(messages):
            log_to_file(f"Message {i}: {msg}")
        log_to_file("=== End Message Contents ===")

        try:
            log_to_file("Calling super().query()...")
            log_to_file(f"include_environment_context: {self.include_environment_context}")
            result = super().query(query, runtime, env, messages, extra_args)
            log_to_file("=== PrivilegedLLM Query Completed Successfully ===")
            log_to_file(f"Result type: {type(result)}")
            if hasattr(result, "__len__"):
                log_to_file(f"Result length: {len(result)}")
            return result
        except Exception as e:
            log_to_file(f"=== PrivilegedLLM Query Failed: {type(e).__name__}: {e} ===")
            import traceback

            log_to_file(f"Traceback: {traceback.format_exc()}")
            raise

    def run_code(self, code, env, namespace, dependencies):
        """Override run_code to log detailed execution"""
        log_to_file("=== Code Execution Started ===")
        log_to_file(f"Code to execute:\n{code}")
        log_to_file(f"Environment type: {type(env)}")
        log_to_file(
            f"Namespace variables: {list(namespace.variables.keys()) if hasattr(namespace, 'variables') else 'No variables attr'}"
        )

        try:
            log_to_file("Calling super().run_code()...")
            result = super().run_code(code, env, namespace, dependencies)
            log_to_file("Code executed, processing result...")
            log_to_file(f"Result tuple length: {len(result) if result else 'None'}")

            if result:
                output, _, exception, _, _ = result
                log_to_file(f"Output: {output}")
                if exception:
                    log_to_file("Code execution had an error (full error message logged in privileged_llm)")
                else:
                    log_to_file("Code executed successfully")

            return result
        except Exception as e:
            log_to_file(f"=== Code Execution Failed: {type(e).__name__}: {e} ===")
            import traceback

            log_to_file(f"Traceback: {traceback.format_exc()}")
            raise


def run_test():
    """Main function to run the custom test"""
    with open(LOG_FILE, "w") as f:
        f.write("")

    log_to_file("=== Starting Test Execution ===")

    # Load our custom environment using AgentDojo standard method
    task_suite.load_and_inject_default_environment({})

    # Create runtime with our tools
    runtime = functions_runtime.FunctionsRuntime(task_suite.tools)

    # Add AI assistant query function (same as original)
    model = "claude-3-7-sonnet-20250219"
    _T = TypeVar("_T", bound=str | int | float | BaseModel)

    def query_ai_assistant(query: str, output_schema: type[_T]) -> _T:
        return quarantined_llm.query_quarantined_llm(
            llm=(model),
            query=query,
            output_schema=output_schema,
            retries=5,
        )

    query_ai_assistant.__doc__ = quarantined_llm.query_quarantined_llm.__doc__
    runtime.register_function(query_ai_assistant)

    # Create LLM and pipeline with logging
    llm = agent_pipeline.AnthropicLLM(
        AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        "claude-3-7-sonnet-20250219",
    )
    p_llm = LoggingPrivilegedLLM(
        llm,
        ADNoSecurityPolicyEngine,
        model,
        include_environment_context=True,
        exclude_datetime=False,
        log_function=log_to_file,
    )

    pipeline = agent_pipeline.AgentPipeline(
        [
            agent_pipeline.InitQuery(),
            p_llm,
        ]
    )

    # Set pipeline name and run specific tasks
    pipeline.name = "daniel-suite"
    specific_task_ids = ["user_task_0"]  # Start with just one task

    print(f"Running task: {specific_task_ids[0]}")

    # Create logs directory
    logdir = Path("./ad_logs")
    logdir.mkdir(exist_ok=True)

    # Run benchmark with timeout
    print("Starting benchmark...")
    try:
        with ad_logging.OutputLogger(str(logdir)):
            # Run with a shorter timeout to avoid hanging
            import signal

            def timeout_handler(_signum, _frame):
                raise TimeoutError("Benchmark execution timed out after 60 seconds")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout

            try:
                results = benchmark.benchmark_suite_without_injections(
                    agent_pipeline=pipeline,
                    suite=task_suite,
                    logdir=logdir,
                    force_rerun=True,
                    user_tasks=specific_task_ids,
                    benchmark_version="test_custom",
                )
                signal.alarm(0)  # Cancel timeout
                print("Benchmark completed successfully")
            except TimeoutError:
                print("Benchmark timed out after 60 seconds")
                return None

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None

    # Print results
    if results:
        utility_results = results["utility_results"]
        print("\nResults:")
        for task_result, score in utility_results.items():
            print(f"  {task_result}: {score}")

    return results


if __name__ == "__main__":
    run_test()
