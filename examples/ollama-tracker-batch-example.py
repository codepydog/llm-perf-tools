import asyncio

from openai import AsyncOpenAI

from llm_perf_tools import InferenceTracker


async def main():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    tracker = InferenceTracker(client)

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

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(request_id: int) -> str:
        async with semaphore:
            print(f"Request {request_id}: ", end="", flush=True)
            response = await tracker.create_chat_completion(
                model="gpt-oss:20b", messages=messages
            )
            print(response)
            return response

    tasks = [bounded_request(i) for i in range(num_requests)]
    _ = await asyncio.gather(*tasks)

    print("=" * 80)

    stats = tracker.get_batch_stats()

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
