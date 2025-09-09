import asyncio

from openai import AsyncOpenAI

from llm_perf_tools import InferenceTracker, save_metrics_to_json


async def main():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    tracker = InferenceTracker(client)

    messages = [{"role": "user", "content": "Hello! Please say hello back."}]

    print("Response: ", end="", flush=True)

    response = await tracker.create_chat_completion(
        model="gpt-oss:20b", messages=messages
    )

    print(response)

    stats = tracker.compute_metrics()

    print("\nInference Metrics:")
    print(f"  Total Requests: {stats.total_requests}")
    print(f"  Successful Requests: {stats.successful_requests}")

    if stats.avg_ttft:
        print(f"  Time to First Token (TTFT): {stats.avg_ttft:.3f}s")
    else:
        print("  TTFT: N/A")

    if stats.avg_e2e_latency:
        print(f"  End-to-End Latency: {stats.avg_e2e_latency:.3f}s")
    else:
        print("  E2E Latency: N/A")

    if stats.avg_itl:
        print(f"  Inter-token Latency (ITL): {stats.avg_itl:.3f}s")
    else:
        print("  ITL: N/A")

    if stats.avg_tps:
        print(f"  Tokens Per Second (TPS): {stats.avg_tps:.2f}")
    else:
        print("  TPS: N/A")

    if tracker.metrics:
        metrics = tracker.metrics[0]
        print(f"  Input tokens: {metrics.input_tokens}")
        print(f"  Output tokens: {metrics.output_tokens}")

    saved_file = save_metrics_to_json(tracker, "ollama_single_example_metrics.json")
    print(f"\nMetrics saved to: {saved_file}")


if __name__ == "__main__":
    asyncio.run(main())
