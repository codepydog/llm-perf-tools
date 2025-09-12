# llm-perf-tools

## Prerequisites

- Python 3.10+
- pyenv (recommended)
- Poetry
- OpenAI API access

## Setup

```bash
pyenv local 3.10
python -m venv .venv
source .venv/bin/activate
make install
```

### Environment

Create `.env` file:

```bash
# OpenAI config
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1/

# Optional: Langfuse config
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## Usage

Basic InferenceTracker usage:

```python
import asyncio
import os
from openai import AsyncOpenAI
from llm_perf_tools import InferenceTracker
from dotenv import load_dotenv

load_dotenv()


async def main():
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    tracker = InferenceTracker(client)

    # Track a request
    response = await tracker.create_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-5",
    )
    print(response)

    # Get performance metrics
    stats = tracker.compute_metrics()
    print("Metrics:\n")
    print(f"TTFT: {stats.avg_ttft:.3f}s")
    print(f"Throughput: {stats.rps:.2f} req/s")


asyncio.run(main())

```

## TODO

1. **Enhance batch statistics in `compute_stats`**
   - Aggregate TTFT, first-token, and end-to-end latency for lists of `RequestMetrics`.
   - Compute 50th/95th/99th percentile latencies with NumPy.
   - Extend the `InferenceStats` dataclass and update existing callers.
   - Add tests exercising batched latency and throughput reporting.

2. **Support pluggable tokenizers**
   - Introduce a tokenizer interface with a default whitespace implementation.
   - Integrate optional [`tiktoken`](https://github.com/openai/tiktoken) support for OpenAI models.
   - Allow `InferenceTracker` to accept a tokenizer via configuration.
   - Document tokenizer selection and update usage examples.

3. **Track errors and request costs**
   - Add `error_type` and `cost` fields to `RequestMetrics`.
   - Capture exceptions during requests and record prompt/completion token counts.
   - Compute costs from a model pricing table and expose totals in reports.
   - Emit cost and error summaries in JSON exports and CLI output.

4. **Provide CLI commands and visualizations**
   - Build a CLI (e.g., with Typer) exposing run/export/report commands.
   - Generate matplotlib histograms for TTFT and throughput.
   - Support saving plots to disk for later analysis.
   - Supply CLI examples and documentation.

5. **Expand test coverage**
   - Test `save_metrics_to_json` and verify exported contents.
   - Add integration tests for concurrent requests and error paths.
   - Exercise cost computations and tokenizer variations.
   - Cover CLI entry points to guard against regressions.


