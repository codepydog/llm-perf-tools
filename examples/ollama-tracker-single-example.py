import asyncio
from openai import AsyncOpenAI
from rich.console import Console
from llm_perf_tools import InferenceTracker, save_metrics_to_json


async def main():
    console = Console()
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    tracker = InferenceTracker(client)

    messages = [{"role": "user", "content": "Hello! Please say hello back."}]

    response = await tracker.create_chat_completion(
        model="gpt-oss:20b", messages=messages
    )
    console.print(f"[green]Response:[/green] {response}")

    stats = tracker.compute_metrics()
    console.print("\n[bold blue]Metrics:[/bold blue]")
    console.print(stats)

    saved_file = save_metrics_to_json(tracker, "ollama_single_example_metrics.json")
    console.print(f"[yellow]Saved to:[/yellow] {saved_file}")


if __name__ == "__main__":
    asyncio.run(main())
