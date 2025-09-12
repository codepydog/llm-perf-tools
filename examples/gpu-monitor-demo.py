import asyncio
from openai import AsyncOpenAI
from llm_perf_tools import InferenceTracker, monitor_gpu_usage, save_metrics_to_json
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

port = 30000
server_url = f"http://localhost:{port}/v1"
MODEL_NAME = "Llama-Primus-Nemotron-70B-Instruct-auto-awq-4WA16-g128"


async def main():
    console = Console()
    client = AsyncOpenAI(base_url=server_url, api_key="None")

    requests = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "What is AI?"}],
        [{"role": "user", "content": "Explain machine learning"}],
        [{"role": "user", "content": "Tell me about Python"}],
        [{"role": "user", "content": "What is Linux?"}],
    ]

    console.print("[bold]GPU Monitor Demo[/bold]")
    console.print(f"Sending {len(requests)} requests...")

    with monitor_gpu_usage("gpu_metrics.csv", interval=0.1) as gpu_metrics:
        tracker = InferenceTracker(client)

        tasks = [
            tracker.create_chat_completion(messages=messages, model=MODEL_NAME)
            for messages in requests
        ]

        responses = await asyncio.gather(*tasks)

    console.print(f"[green]Completed {len(responses)} requests[/green]")
    console.print(
        f"[blue]Collected {len(gpu_metrics)} GPU samples -> gpu_metrics.csv[/blue]"
    )

    metrics = tracker.compute_metrics()
    console.print(f"[cyan]Inference metrics:[/cyan] {metrics}")

    save_metrics_to_json(tracker=tracker, filename="gpu_demo_metrics.json")
    console.print("[yellow]Saved inference metrics -> gpu_demo_metrics.json[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
