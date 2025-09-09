from .models import RequestMetrics, InferenceStats


def calculate_ttft(metrics: RequestMetrics) -> float | None:
    if metrics.first_token_time is None:
        return None
    return metrics.first_token_time - metrics.request_start


def calculate_e2e_latency(metrics: RequestMetrics) -> float | None:
    if metrics.request_end is None:
        return None
    return metrics.request_end - metrics.request_start


def calculate_itl(metrics: RequestMetrics) -> float | None:
    if metrics.first_token_time is None or metrics.request_end is None:
        return None
    if metrics.output_tokens <= 1:
        return None
    generation_time = metrics.request_end - metrics.first_token_time
    return generation_time / (metrics.output_tokens - 1)


def calculate_tps(metrics: list[RequestMetrics]) -> float | None:
    if not metrics:
        return None

    total_tokens = sum(m.output_tokens for m in metrics)
    if total_tokens == 0:
        return None

    start_times = [m.request_start for m in metrics]
    end_times = [m.request_end for m in metrics if m.request_end is not None]

    if not end_times:
        return None

    duration = max(end_times) - min(start_times)
    return total_tokens / duration if duration > 0 else None


def calculate_rps(metrics: list[RequestMetrics], duration: float) -> float | None:
    if duration <= 0:
        return None
    completed_requests = len([m for m in metrics if m.request_end is not None])
    return completed_requests / duration


def calculate_stats(
    metrics: RequestMetrics | list[RequestMetrics], duration: float | None = None
) -> InferenceStats:
    if isinstance(metrics, RequestMetrics):
        return InferenceStats(
            ttft=calculate_ttft(metrics),
            e2e_latency=calculate_e2e_latency(metrics),
            itl=calculate_itl(metrics),
        )

    stats = InferenceStats()
    if metrics:
        stats.tps = calculate_tps(metrics)
        if duration is not None:
            stats.rps = calculate_rps(metrics, duration)

    return stats
