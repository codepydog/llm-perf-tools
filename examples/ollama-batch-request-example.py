import asyncio
import time

from openai import AsyncOpenAI

from llm_perf_tools.models import RequestMetrics, BatchInferenceStats


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


def calculate_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((percentile / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def calculate_batch_stats(metrics_list: list[RequestMetrics], batch_duration: float) -> BatchInferenceStats:
    if not metrics_list:
        return BatchInferenceStats()

    successful_metrics = [m for m in metrics_list if m.request_end is not None]

    ttft_values = [
        m.first_token_time - m.request_start
        for m in successful_metrics
        if m.first_token_time is not None
    ]

    e2e_values = [
        m.request_end - m.request_start
        for m in successful_metrics
        if m.request_end is not None
    ]

    itl_values = []
    tps_values = []

    for m in successful_metrics:
        if m.first_token_time and m.request_end and m.output_tokens > 1:
            generation_time = m.request_end - m.first_token_time
            itl = generation_time / (m.output_tokens - 1)
            itl_values.append(itl)

            if generation_time > 0:
                tps = m.output_tokens / generation_time
                tps_values.append(tps)

    rps = len(successful_metrics) / batch_duration if batch_duration > 0 else 0

    return BatchInferenceStats(
        avg_ttft=sum(ttft_values) / len(ttft_values) if ttft_values else None,
        p50_ttft=calculate_percentile(ttft_values, 50) if ttft_values else None,
        p95_ttft=calculate_percentile(ttft_values, 95) if ttft_values else None,
        p99_ttft=calculate_percentile(ttft_values, 99) if ttft_values else None,
        min_ttft=min(ttft_values) if ttft_values else None,
        max_ttft=max(ttft_values) if ttft_values else None,
        avg_e2e_latency=sum(e2e_values) / len(e2e_values) if e2e_values else None,
        p50_e2e_latency=calculate_percentile(e2e_values, 50) if e2e_values else None,
        p95_e2e_latency=calculate_percentile(e2e_values, 95) if e2e_values else None,
        p99_e2e_latency=calculate_percentile(e2e_values, 99) if e2e_values else None,
        min_e2e_latency=min(e2e_values) if e2e_values else None,
        max_e2e_latency=max(e2e_values) if e2e_values else None,
        avg_itl=sum(itl_values) / len(itl_values) if itl_values else None,
        p50_itl=calculate_percentile(itl_values, 50) if itl_values else None,
        p95_itl=calculate_percentile(itl_values, 95) if itl_values else None,
        p99_itl=calculate_percentile(itl_values, 99) if itl_values else None,
        min_itl=min(itl_values) if itl_values else None,
        max_itl=max(itl_values) if itl_values else None,
        avg_tps=sum(tps_values) / len(tps_values) if tps_values else None,
        p50_tps=calculate_percentile(tps_values, 50) if tps_values else None,
        p5_tps=calculate_percentile(tps_values, 5) if tps_values else None,
        p1_tps=calculate_percentile(tps_values, 1) if tps_values else None,
        min_tps=min(tps_values) if tps_values else None,
        max_tps=max(tps_values) if tps_values else None,
        rps=rps,
        total_requests=len(metrics_list),
        successful_requests=len(successful_metrics),
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

    stats = calculate_batch_stats(metrics_list, batch_duration)

    print(f"\nBatch Inference Metrics:")
    print(f"  Total Requests: {stats.total_requests}")
    print(f"  Successful Requests: {stats.successful_requests}")
    print(
        f"  Request Success Rate: {(stats.successful_requests / stats.total_requests * 100):.1f}%"
    )
    print(f"  Requests Per Second (RPS): {stats.rps:.2f}")

    if stats.avg_ttft:
        print(f"\n  Time to First Token (TTFT):")
        print(f"    Average: {stats.avg_ttft:.3f}s")
        print(f"    P50: {stats.p50_ttft:.3f}s")
        print(f"    P95: {stats.p95_ttft:.3f}s")
        print(f"    P99: {stats.p99_ttft:.3f}s")
        print(f"    Min: {stats.min_ttft:.3f}s")
        print(f"    Max: {stats.max_ttft:.3f}s")

    if stats.avg_e2e_latency:
        print(f"\n  End-to-End Latency:")
        print(f"    Average: {stats.avg_e2e_latency:.3f}s")
        print(f"    P50: {stats.p50_e2e_latency:.3f}s")
        print(f"    P95: {stats.p95_e2e_latency:.3f}s")
        print(f"    P99: {stats.p99_e2e_latency:.3f}s")
        print(f"    Min: {stats.min_e2e_latency:.3f}s")
        print(f"    Max: {stats.max_e2e_latency:.3f}s")

    if stats.avg_itl:
        print(f"\n  Inter-token Latency (ITL):")
        print(f"    Average: {stats.avg_itl:.3f}s")
        print(f"    P50: {stats.p50_itl:.3f}s")
        print(f"    P95: {stats.p95_itl:.3f}s")
        print(f"    P99: {stats.p99_itl:.3f}s")
        print(f"    Min: {stats.min_itl:.3f}s")
        print(f"    Max: {stats.max_itl:.3f}s")

    if stats.avg_tps:
        print(f"\n  Tokens Per Second (TPS):")
        print(f"    Average: {stats.avg_tps:.2f}")
        print(f"    P50: {stats.p50_tps:.2f}")
        print(f"    P5: {stats.p5_tps:.2f}")
        print(f"    P1: {stats.p1_tps:.2f}")
        print(f"    Min: {stats.min_tps:.2f}")
        print(f"    Max: {stats.max_tps:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
