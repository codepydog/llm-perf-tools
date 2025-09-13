import asyncio

from openai import AsyncOpenAI

from llm_perf_tools import InferenceTracker, save_metrics_to_json


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

    metrics = tracker.compute_metrics()
    print(metrics)

    saved_file = save_metrics_to_json(tracker, "ollama_batch_example_complete.json")
    print(f"\nComplete metrics saved to: {saved_file}")


if __name__ == "__main__":
    asyncio.run(main())
