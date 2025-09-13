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
import tiktoken

load_dotenv()


async def main():
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    # Use GPT-5 tokenizer for accurate token counting
    tokenizer = tiktoken.encoding_for_model("gpt-5").encode
    tracker = InferenceTracker(client, tokenizer=tokenizer)

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

### GPU Monitoring

Basic GPU usage tracking:

```python
import asyncio
from openai import AsyncOpenAI
from llm_perf_tools import InferenceTracker, monitor_gpu_usage

async def main():
    # Setup client for local LLM server (e.g., SGLang, vLLM, Ollama)
    client = AsyncOpenAI(base_url="http://localhost:30000/v1", api_key="None")
    
    with monitor_gpu_usage("gpu_metrics.csv", interval=0.1) as gpu_metrics:
        tracker = InferenceTracker(client)
        
        response = await tracker.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="meta-llama/Llama-3.2-8B-Instruct"  # Replace with your model
        )
    
    print(f"Collected {len(gpu_metrics)} GPU samples")
    print("GPU metrics saved to gpu_metrics.csv")

asyncio.run(main())
```

### Visualization

Visualize performance metrics with built-in plotting:

```python
from llm_perf_tools import plot_eval_result

# Plot metrics from saved evaluation results
inference_path = "eval_results/batch_metrics.json"
gpu_path = "eval_results/gpu_metrics.csv"

plots = plot_eval_result(inference_path, gpu_path)

# Save plots
plots[0].savefig("inference_metrics.png", dpi=300, bbox_inches='tight')
plots[1].savefig("gpu_metrics.png", dpi=300, bbox_inches='tight')

print("Performance plots saved")
```

This generates two comprehensive plots:

**Inference Metrics Plot**:
![Inference Metrics](figures/inference_metrics.png)

- Time to First Token (TTFT) distribution and timeline
- End-to-End Latency trends over time
- Tokens per Second (TPS) performance
- Inter-Token Latency (ITL) statistics

**GPU Metrics Plot**:
![GPU Metrics](figures/gpu_metrics.png)

- GPU utilization percentage over time
- Memory usage (used vs total)
- Temperature monitoring
- Power consumption tracking
