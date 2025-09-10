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
from openai import AsyncOpenAI
from llm_perf_tools import InferenceTracker

async def main():
    client = AsyncOpenAI()
    tracker = InferenceTracker(client)
    
    # Track a request
    response = await tracker.create_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4"
    )
    
    # Get performance metrics
    stats = tracker.compute_metrics()
    print(f"TTFT: {stats.avg_ttft:.3f}s")
    print(f"Throughput: {stats.rps:.2f} req/s")

asyncio.run(main())
```

