import asyncio
from openai import AsyncOpenAI
from llm_perf_tools import InferenceTracker
from dotenv import load_dotenv
from rich.console import Console
from llm_perf_tools import save_metrics_to_json

load_dotenv()

# sglang
port = 30000
server_url = f"http://localhost:{port}/v1"
MODLE_NAME = "Llama-Primus-Nemotron-70B-Instruct-auto-awq-4WA16-g128"


async def main():
    console = Console()
    client = AsyncOpenAI(base_url=server_url, api_key="None")
    tracker = InferenceTracker(client)

    requests = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "What is AI?"}],
        [{"role": "user", "content": "Explain machine learning"}],
        [{"role": "user", "content": "Tell me about Python"}],
        [{"role": "user", "content": "What is Linux?"}],
    ]

    console.print(f"[yellow]Sending {len(requests)} requests...[/yellow]")

    tasks = [
        tracker.create_chat_completion(messages=messages, model=MODLE_NAME)
        for messages in requests
    ]

    responses = await asyncio.gather(*tasks)

    console.print("[bold green]Responses:[/bold green]")
    for i, response in enumerate(responses):
        console.print(f"[cyan]Request {i + 1}:[/cyan] {str(response)[:50]}...")

    metrics = tracker.compute_metrics()
    console.print("\n[bold blue]Batch Metrics:[/bold blue]")
    console.print(metrics)

    save_metrics_to_json(tracker=tracker, filename="sglang_batch_metrics.json")


if __name__ == "__main__":
    asyncio.run(main())
