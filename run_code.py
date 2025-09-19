# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from dotenv import load_dotenv

load_dotenv()

from agentdojo import agent_pipeline, benchmark, functions_runtime, logging
from agentdojo.task_suite import get_suite
from anthropic import AsyncAnthropic
from pydantic import BaseModel

# Fix some weird pydantic model defs.
try:
    from agentdojo.benchmark import TaskResults
    from agentdojo.functions_runtime import FunctionCall

    TaskResults.model_rebuild()
except Exception:
    pass

from camel import quarantined_llm
from camel.interpreter.namespace import Namespace
from camel.pipeline_elements import privileged_llm
from camel.pipeline_elements.agentdojo_function import (
    make_agentdojo_namespace,
)
from camel.pipeline_elements.security_policies import (
    ADNoSecurityPolicyEngine,
)

if TYPE_CHECKING:
    from agentdojo.default_suites.v1.workspace.task_suite import (
        WorkspaceEnvironment,
    )


def run_code(
    code: str,
    p_llm: privileged_llm.PrivilegedLLM,
    env: "WorkspaceEnvironment",
    namespace: Namespace,
):
    out, _, exc, namespace, _ = p_llm.run_code(code, env, namespace, ())

    print(code)

    if exc is not None:
        print([dep.raw for dep in exc.dependencies])
        print(f"{type(exc.exception).__name__}({exc.exception})")
        print(privileged_llm.format_camel_exception(exc, code))
    else:
        print(out)


def attack():
    suite = get_suite("v1.2", "workspace")
    runtime = functions_runtime.FunctionsRuntime(suite.tools)
    env: WorkspaceEnvironment = suite.load_and_inject_default_environment({})

    model = "claude-3-5-sonnet-latest"

    _T = TypeVar("_T", bound=str | int | float | BaseModel)

    def query_ai_assistant(query: str, output_schema: type[_T]) -> _T:
        return quarantined_llm.query_quarantined_llm(
            llm=(model),  # type: ignore
            query=query,
            output_schema=output_schema,
            retries=5,
        )

    query_ai_assistant.__doc__ = quarantined_llm.query_quarantined_llm.__doc__
    runtime.register_function(query_ai_assistant)

    namespace = Namespace.with_builtins()
    namespace = namespace.add_variables(make_agentdojo_namespace(namespace, runtime, env))

    llm = agent_pipeline.AnthropicLLM(
        AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        "claude-3-5-sonnet-latest",
    )
    p_llm = privileged_llm.PrivilegedLLM(llm, ADNoSecurityPolicyEngine, model)  # type: ignore

    pipeline = agent_pipeline.AgentPipeline(
        [
            # Adds the user query to the history
            agent_pipeline.InitQuery(),
            # Generates the code and writes it to `extra_args`
            p_llm,
        ]
    )

    # To get this working (formerly just pipeline.query)
    # I had to do this manually (wrap in logger, run benchmark())
    pipeline.name = "camel-prompt-injection-test"
    specific_task_ids = ["user_task_0", "user_task_1"]

    print(f"Running specific tasks: {specific_task_ids}")

    # log to logs/
    logdir = Path("./logs")
    logdir.mkdir(exist_ok=True)
    with logging.OutputLogger(str(logdir)):
        results = benchmark.benchmark_suite_without_injections(
            agent_pipeline=pipeline,
            suite=suite,
            logdir=logdir,
            force_rerun=True,
            user_tasks=specific_task_ids,
            benchmark_version="v1.2",
        )

    # Print results
    utility_results = results["utility_results"]
    print(f"\nResults for tasks {specific_task_ids}:")
    for task_result, score in utility_results.items():
        print(f"  {task_result}: {score}")


if __name__ == "__main__":
    attack()
