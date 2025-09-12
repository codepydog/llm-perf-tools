import asyncio
from openai import AsyncOpenAI
from rich.console import Console
from llm_perf_tools import InferenceTracker, save_metrics_to_json


async def main():
    console = Console()
    client = AsyncOpenAI(base_url="http://localhost:30000/v1", api_key="None")
    tracker = InferenceTracker(client)

    requests = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "What is AI?"}],
        [{"role": "user", "content": "Explain machine learning"}],
    ]

    console.print(f"[yellow]Sending {len(requests)} requests...[/yellow]")

    tasks = [
        tracker.create_chat_completion(messages=msg, model="llama") for msg in requests
    ]
    _ = await asyncio.gather(*tasks)

    console.print("[bold green]Responses received[/bold green]")

    metrics = tracker.compute_metrics()
    console.print("\n[bold blue]Metrics:[/bold blue]")
    console.print(metrics)

    save_metrics_to_json(tracker, "sglang_batch_metrics.json")


if __name__ == "__main__":
    asyncio.run(main())
