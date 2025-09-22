## Exploring Camel paper

Added daniel/ folder. Made some changes to the flow:

1. Added code to show the current namespace variables for the LM (on an error, so it doesn't try & redefine)
2. Prompt tweaks:
   1. Explained NotEnoughInfo differently - before the solution was phrased as "supply more info". Added that the problem might be the schema being too complex
   2. Tell LM it can't see printed text & to use query_ai_assistant more
   3. Reframe as code template, not writing code, to avoid triggering best practices
   4. Lots of reminders to simplify schemas
3. Added lots of logging to privileged_llm.

Preliminary results (one run of the first 5 tasks of travel) seem to suggest that these tweaks are a little better than the original at handling underdocumented APIs. In the paper, #2 & #4 failed due to this. #2 worked correctly, and #4 failed but due to an unrelated misstep (qllm thought 4.2 > 4.7?).

See the results in daniel/travel_log-sonnet-35

# OLD:

# `CaMeL`: [Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813)

Edoardo Debenedetti<sup>1,3</sup>, Ilia Shumailov<sup>2</sup>, Tianqi Fan<sup>1</sup>, Jamie Hayes<sup>2</sup>, Nicholas Carlini<sup>2</sup>, Daniel Fabian<sup>1</sup>, Christoph Kern<sup>1</sup>, Chongyang Shi<sup>2</sup>, Florian Tramèr<sup>3</sup>

<sup>1</sup>Google, <sup>2</sup>Google DeepMind, and <sup>3</sup>ETH Zurich

> [!WARNING]
> This is a research artifact released to reproduce the results in our paper. The interpreter implementation likely contains bugs (e.g., it might throw uncaught exceptions and crash) and the implementation might not be fully secure.
>
> This is **not** a Google product, and we are not planning to provide support for and/or maintain this codebase.

## Pre-requisites

1. Install `uv` via the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. Rename `.env.example` to `.env` and populate it with your API keys.
3. `uv` will install all dependencies as soon as you run `uv run ...`.

## Running running the defense against AgentDojo

```bash
uv run --env-file .env main.py MODEL_NAME [--use-original] [--ad_defense] [--reasoning-effort] [--thinking_budget_tokens] [--run-attack] [--replay-with-policies] [--eval_mode]
```

More details on the various CLI arguments can be found by running `uv run main.py --help`

## FAQ

> How do I try a new/different model?

You can add it to the [`models.py`](src/camel/models.py) file, in the `_supported_model_names` variable. The keys are the model names with the given provider (check the provider's API) and the values is what the model says when asked "what model are you?". Keep in mind that OpenAI reasoning models are stored in the `_oai_thinking_models` variable instead.

> If I have questions on the codebase how can I reach out?

Please open an issue in this repository. Please note that we are not planning to fix bugs as this codebase is just meant as a research artifact.

## Running tests and linters

```bash
uv run ruff check --fix
uv run format
uv run pyright
uv run pytest
```

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
