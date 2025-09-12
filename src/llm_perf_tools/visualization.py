import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns

from .types import GPUMetrics
from .utils import load_inference_data, load_gpu_data

sns.set_style("whitegrid")


def plot_inference_metrics(data: dict) -> matplotlib.figure.Figure:
    batch_stats = data.get("batch_stats", {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Inference Metrics", fontsize=16)

    if batch_stats.get("avg_ttft"):
        ttft_data = [
            batch_stats["min_ttft"],
            batch_stats["p50_ttft"],
            batch_stats["avg_ttft"],
            batch_stats["p95_ttft"],
            batch_stats["max_ttft"],
        ]
        axes[0, 0].boxplot(ttft_data)
        axes[0, 0].set_title("TTFT Distribution")
        axes[0, 0].set_ylabel("Time (s)")

    if batch_stats.get("avg_e2e_latency"):
        e2e_data = [
            batch_stats["min_e2e_latency"],
            batch_stats["p50_e2e_latency"],
            batch_stats["avg_e2e_latency"],
            batch_stats["p95_e2e_latency"],
            batch_stats["max_e2e_latency"],
        ]
        axes[0, 1].boxplot(e2e_data)
        axes[0, 1].set_title("End-to-End Latency")
        axes[0, 1].set_ylabel("Time (s)")

    if batch_stats.get("avg_tps"):
        tps_data = [
            batch_stats["min_tps"],
            batch_stats["p50_tps"],
            batch_stats["avg_tps"],
            batch_stats["max_tps"],
        ]
        axes[1, 0].hist(tps_data, bins=10)
        axes[1, 0].set_title("TPS Distribution")
        axes[1, 0].set_xlabel("Tokens/sec")

    successful = batch_stats.get("successful_requests", 0)
    total = batch_stats.get("total_requests", 0)
    failed = total - successful

    axes[1, 1].pie(
        [successful, failed], labels=["Successful", "Failed"], autopct="%1.1f%%"
    )
    axes[1, 1].set_title("Request Summary")

    plt.tight_layout()
    return fig


def plot_gpu_metrics(gpu_metrics: list[GPUMetrics]) -> matplotlib.figure.Figure:
    if not gpu_metrics:
        return plt.figure()

    timestamps = [m.timestamp for m in gpu_metrics]
    start_time = min(timestamps)
    relative_times = [(t - start_time) for t in timestamps]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GPU Metrics", fontsize=16)

    axes[0, 0].plot(relative_times, [m.gpu_utilization_percent for m in gpu_metrics])
    axes[0, 0].set_title("GPU Utilization")
    axes[0, 0].set_ylabel("Utilization (%)")

    axes[0, 1].plot(relative_times, [m.memory_utilization_percent for m in gpu_metrics])
    axes[0, 1].set_title("Memory Usage")
    axes[0, 1].set_ylabel("Memory (%)")

    axes[1, 0].plot(relative_times, [m.temperature_celsius for m in gpu_metrics])
    axes[1, 0].set_title("Temperature")
    axes[1, 0].set_ylabel("Temperature (°C)")
    axes[1, 0].set_xlabel("Time (s)")

    axes[1, 1].plot(relative_times, [m.power_draw_watts for m in gpu_metrics])
    axes[1, 1].set_title("Power Draw")
    axes[1, 1].set_ylabel("Power (W)")
    axes[1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    return fig


def plot_eval_result(
    inference_path: str, gpu_path: str | None = None
) -> (
    matplotlib.figure.Figure | tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]
):
    inference_data = load_inference_data(inference_path)
    inference_fig = plot_inference_metrics(inference_data)

    if gpu_path:
        gpu_data = load_gpu_data(gpu_path)
        gpu_fig = plot_gpu_metrics(gpu_data)
        return inference_fig, gpu_fig

    return inference_fig
