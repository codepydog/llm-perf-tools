import asyncio
from openai import AsyncOpenAI
from llm_perf_tools import InferenceTracker
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

# sglang
port = 30000
server_url = f"http://localhost:{port}/v1"
MODLE_NAME = "Llama-Primus-Nemotron-70B-Instruct-auto-awq-4WA16-g128"


async def main():
    console = Console()
    client = AsyncOpenAI(base_url=server_url, api_key="None")
    tracker = InferenceTracker(client)

    response = await tracker.create_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model=MODLE_NAME,
    )

    console.print("[bold green]Response:[/bold green]")
    console.print(response)

    metrics = tracker.compute_metrics()
    console.print("\n[bold blue]Metrics:[/bold blue]")
    console.print(metrics)


if __name__ == "__main__":
    asyncio.run(main())
