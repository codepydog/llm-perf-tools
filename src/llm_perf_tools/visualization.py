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

    gpu_data = {}
    for metric in gpu_metrics:
        if metric.gpu_id not in gpu_data:
            gpu_data[metric.gpu_id] = []
        gpu_data[metric.gpu_id].append(metric)

    gpu_ids = sorted(gpu_data.keys())
    num_gpus = len(gpu_ids)

    fig, axes = plt.subplots(num_gpus, 4, figsize=(16, 4 * num_gpus))
    if num_gpus == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("GPU Metrics by GPU ID", fontsize=16)

    for i, gpu_id in enumerate(gpu_ids):
        metrics = gpu_data[gpu_id]
        timestamps = [m.timestamp for m in metrics]
        start_time = min(timestamps)
        relative_times = [(t - start_time) for t in timestamps]

        axes[i, 0].plot(relative_times, [m.gpu_utilization_percent for m in metrics])
        axes[i, 0].set_title(f"GPU {gpu_id} Utilization")
        axes[i, 0].set_ylabel("Utilization (%)")

        axes[i, 1].plot(relative_times, [m.memory_utilization_percent for m in metrics])
        axes[i, 1].set_title(f"GPU {gpu_id} Memory Usage")
        axes[i, 1].set_ylabel("Memory (%)")

        axes[i, 2].plot(relative_times, [m.temperature_celsius for m in metrics])
        axes[i, 2].set_title(f"GPU {gpu_id} Temperature")
        axes[i, 2].set_ylabel("Temperature (°C)")

        axes[i, 3].plot(relative_times, [m.power_draw_watts for m in metrics])
        axes[i, 3].set_title(f"GPU {gpu_id} Power Draw")
        axes[i, 3].set_ylabel("Power (W)")

        if i == num_gpus - 1:
            axes[i, 2].set_xlabel("Time (s)")
            axes[i, 3].set_xlabel("Time (s)")

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
