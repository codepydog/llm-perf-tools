import csv
import time
import threading
from contextlib import contextmanager
from pathlib import Path

import pynvml
from .types import GPUMetrics

POWER_WATTS_DIVISOR = 1000.0
BYTES_TO_MB = 1024 * 1024


def _collect_gpu_metrics(gpu_id: int) -> GPUMetrics:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / POWER_WATTS_DIVISOR

    return GPUMetrics(
        timestamp=time.perf_counter(),
        gpu_id=gpu_id,
        memory_used_mb=memory_info.used // BYTES_TO_MB,
        memory_total_mb=memory_info.total // BYTES_TO_MB,
        memory_utilization_percent=round(memory_info.used / memory_info.total * 100, 2),
        gpu_utilization_percent=utilization.gpu,
        temperature_celsius=temperature,
        power_draw_watts=power,
    )


def _save_metrics_to_csv(metrics: list[GPUMetrics], output_path: str) -> None:
    if not metrics:
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].model_fields.keys())
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.model_dump())


@contextmanager
def monitor_gpu_usage(
    output_path: str = "gpu_metrics.csv", interval: float = 0.1, gpu_id: int = 0
):
    pynvml.nvmlInit()
    metrics = []
    stop_event = threading.Event()

    def _monitor_loop():
        while not stop_event.is_set():
            try:
                metric = _collect_gpu_metrics(gpu_id)
                metrics.append(metric)
                time.sleep(interval)
            except (pynvml.NVMLError, OSError):
                break

    thread = threading.Thread(target=_monitor_loop, daemon=True)
    thread.start()

    try:
        yield metrics
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        _save_metrics_to_csv(metrics, output_path)
        pynvml.nvmlShutdown()
