import asyncio
import time

from openai import AsyncOpenAI

from llm_perf_tools.models import RequestMetrics
from llm_perf_tools.metrics import calculate_stats


# Prerequisites: Launch Ollama server with: ollama run gpt-oss:20b
async def main():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    messages = [{"role": "user", "content": "Hello! Please say hello back."}]

    request_start = time.time()

    response = await client.chat.completions.create(
        model="gpt-oss:20b", messages=messages, stream=True
    )

    first_token_time = None
    content_chunks = []

    print("Response: ", end="", flush=True)

    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            content_chunks.append(content)

    print()  # New line after streaming
    request_end = time.time()
    full_content = "".join(content_chunks)

    input_tokens = len(" ".join(msg["content"] for msg in messages).split())
    output_tokens = len(full_content.split())

    metrics = RequestMetrics(
        request_start=request_start,
        first_token_time=first_token_time,
        request_end=request_end,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    stats = calculate_stats(metrics)

    print(f"\nInference Metrics:")
    print(
        f"  Time to First Token (TTFT): {stats.ttft:.3f}s"
        if stats.ttft
        else "  TTFT: N/A"
    )
    print(
        f"  End-to-End Latency: {stats.e2e_latency:.3f}s"
        if stats.e2e_latency
        else "  E2E Latency: N/A"
    )
    print(
        f"  Inter-token Latency (ITL): {stats.itl:.3f}s" if stats.itl else "  ITL: N/A"
    )
    print(f"  Input tokens: {metrics.input_tokens}")
    print(f"  Output tokens: {metrics.output_tokens}")

    # For single request, calculate TPS from generation time
    if stats.ttft and stats.e2e_latency and metrics.output_tokens > 0:
        generation_time = stats.e2e_latency - stats.ttft
        tps = metrics.output_tokens / generation_time if generation_time > 0 else 0
        print(f"  Tokens Per Second (TPS): {tps:.2f}")
    else:
        print(f"  Tokens Per Second (TPS): N/A")


if __name__ == "__main__":
    asyncio.run(main())
