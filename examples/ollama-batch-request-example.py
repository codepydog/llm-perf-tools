import asyncio
import time

from openai import AsyncOpenAI

from llm_perf_tools import RequestMetrics, compute_batch_metrics


async def make_request(
    client: AsyncOpenAI, messages: list[dict], model: str, request_id: int
) -> RequestMetrics:
    request_start = time.time()

    try:
        response = await client.chat.completions.create(
            model=model, messages=messages, stream=True
        )

        first_token_time = None
        content_chunks = []

        print(f"Request {request_id}: ", end="", flush=True)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                content_chunks.append(content)

        request_end = time.time()
        full_content = "".join(content_chunks)

        input_tokens = len(" ".join(msg["content"] for msg in messages).split())
        output_tokens = len(full_content.split())

        return RequestMetrics(
            request_start=request_start,
            first_token_time=first_token_time,
            request_end=request_end,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return RequestMetrics(
            request_start=request_start,
            first_token_time=None,
            request_end=time.time(),
            input_tokens=0,
            output_tokens=0,
        )


async def main():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    num_requests = 5
    concurrency = 3

    messages = [
        {
            "role": "user",
            "content": "Hello! Please say hello back and tell me a short joke.",
        }
    ]

    print(
        f"Running {num_requests} concurrent requests with concurrency limit of {concurrency}..."
    )
    print("=" * 80)

    batch_start = time.time()

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(request_id: int) -> RequestMetrics:
        async with semaphore:
            return await make_request(client, messages, "gpt-oss:20b", request_id)

    tasks = [bounded_request(i) for i in range(num_requests)]
    metrics_list = await asyncio.gather(*tasks)

    batch_end = time.time()
    batch_duration = batch_end - batch_start

    print("=" * 80)
    print(f"\nBatch completed in {batch_duration:.3f}s")

    stats = compute_batch_metrics(metrics_list, batch_duration)

    print("\nBatch Inference Metrics:")
    print(f"  Total Requests: {stats.total_requests}")
    print(f"  Successful Requests: {stats.successful_requests}")
    print(
        f"  Request Success Rate: {(stats.successful_requests / stats.total_requests * 100):.1f}%"
    )
    print(f"  Requests Per Second (RPS): {stats.rps:.2f}")

    if stats.avg_ttft:
        print("\n  Time to First Token (TTFT):")
        print(f"    Average: {stats.avg_ttft:.3f}s")
        print(f"    P50: {stats.p50_ttft:.3f}s")
        print(f"    P95: {stats.p95_ttft:.3f}s")
        print(f"    P99: {stats.p99_ttft:.3f}s")
        print(f"    Min: {stats.min_ttft:.3f}s")
        print(f"    Max: {stats.max_ttft:.3f}s")

    if stats.avg_e2e_latency:
        print("\n  End-to-End Latency:")
        print(f"    Average: {stats.avg_e2e_latency:.3f}s")
        print(f"    P50: {stats.p50_e2e_latency:.3f}s")
        print(f"    P95: {stats.p95_e2e_latency:.3f}s")
        print(f"    P99: {stats.p99_e2e_latency:.3f}s")
        print(f"    Min: {stats.min_e2e_latency:.3f}s")
        print(f"    Max: {stats.max_e2e_latency:.3f}s")

    if stats.avg_itl:
        print("\n  Inter-token Latency (ITL):")
        print(f"    Average: {stats.avg_itl:.3f}s")
        print(f"    P50: {stats.p50_itl:.3f}s")
        print(f"    P95: {stats.p95_itl:.3f}s")
        print(f"    P99: {stats.p99_itl:.3f}s")
        print(f"    Min: {stats.min_itl:.3f}s")
        print(f"    Max: {stats.max_itl:.3f}s")

    if stats.avg_tps:
        print("\n  Tokens Per Second (TPS):")
        print(f"    Average: {stats.avg_tps:.2f}")
        print(f"    P50: {stats.p50_tps:.2f}")
        print(f"    P5: {stats.p5_tps:.2f}")
        print(f"    P1: {stats.p1_tps:.2f}")
        print(f"    Min: {stats.min_tps:.2f}")
        print(f"    Max: {stats.max_tps:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
