import time
from typing import Any

from .types import RequestMetrics, InferenceStats, BatchInferenceStats


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


def calculate_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((percentile / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def calculate_batch_stats(
    metrics_list: list[RequestMetrics], batch_duration: float
) -> BatchInferenceStats:
    if not metrics_list:
        return BatchInferenceStats()

    successful_metrics = [m for m in metrics_list if m.request_end is not None]

    ttft_values = [
        m.first_token_time - m.request_start
        for m in successful_metrics
        if m.first_token_time is not None
    ]

    e2e_values = [
        m.request_end - m.request_start
        for m in successful_metrics
        if m.request_end is not None
    ]

    itl_values = []
    tps_values = []

    for m in successful_metrics:
        if m.first_token_time and m.request_end and m.output_tokens > 1:
            generation_time = m.request_end - m.first_token_time
            itl = generation_time / (m.output_tokens - 1)
            itl_values.append(itl)

            if generation_time > 0:
                tps = m.output_tokens / generation_time
                tps_values.append(tps)

    rps = len(successful_metrics) / batch_duration if batch_duration > 0 else 0

    return BatchInferenceStats(
        avg_ttft=sum(ttft_values) / len(ttft_values) if ttft_values else None,
        p50_ttft=calculate_percentile(ttft_values, 50) if ttft_values else None,
        p95_ttft=calculate_percentile(ttft_values, 95) if ttft_values else None,
        p99_ttft=calculate_percentile(ttft_values, 99) if ttft_values else None,
        min_ttft=min(ttft_values) if ttft_values else None,
        max_ttft=max(ttft_values) if ttft_values else None,
        avg_e2e_latency=sum(e2e_values) / len(e2e_values) if e2e_values else None,
        p50_e2e_latency=calculate_percentile(e2e_values, 50) if e2e_values else None,
        p95_e2e_latency=calculate_percentile(e2e_values, 95) if e2e_values else None,
        p99_e2e_latency=calculate_percentile(e2e_values, 99) if e2e_values else None,
        min_e2e_latency=min(e2e_values) if e2e_values else None,
        max_e2e_latency=max(e2e_values) if e2e_values else None,
        avg_itl=sum(itl_values) / len(itl_values) if itl_values else None,
        p50_itl=calculate_percentile(itl_values, 50) if itl_values else None,
        p95_itl=calculate_percentile(itl_values, 95) if itl_values else None,
        p99_itl=calculate_percentile(itl_values, 99) if itl_values else None,
        min_itl=min(itl_values) if itl_values else None,
        max_itl=max(itl_values) if itl_values else None,
        avg_tps=sum(tps_values) / len(tps_values) if tps_values else None,
        p50_tps=calculate_percentile(tps_values, 50) if tps_values else None,
        p5_tps=calculate_percentile(tps_values, 5) if tps_values else None,
        p1_tps=calculate_percentile(tps_values, 1) if tps_values else None,
        min_tps=min(tps_values) if tps_values else None,
        max_tps=max(tps_values) if tps_values else None,
        rps=rps,
        total_requests=len(metrics_list),
        successful_requests=len(successful_metrics),
    )


class InferenceTracker:
    def __init__(self, client: Any):
        self.client = client
        self.metrics: list[RequestMetrics] = []
        self._start_time: float | None = None

    async def create_chat_completion(
        self,
        messages: list[dict],
        model: str,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        user: str | None = None,
        **kwargs,
    ) -> str:
        if self._start_time is None:
            self._start_time = time.time()

        request_start = time.time()

        kwargs.update(
            {
                k: v
                for k, v in {
                    "frequency_penalty": frequency_penalty,
                    "logit_bias": logit_bias,
                    "logprobs": logprobs,
                    "top_logprobs": top_logprobs,
                    "max_tokens": max_tokens,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "response_format": response_format,
                    "seed": seed,
                    "stop": stop,
                    "stream": stream,
                    "temperature": temperature,
                    "top_p": top_p,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "user": user,
                }.items()
                if v is not None
            }
        )

        try:
            response = await self.client.chat.completions.create(
                model=model, messages=messages, stream=True, **kwargs
            )

            first_token_time = None
            content_chunks = []

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    content = chunk.choices[0].delta.content
                    content_chunks.append(content)

            request_end = time.time()
            full_content = "".join(content_chunks)

            input_tokens = len(" ".join(msg["content"] for msg in messages).split())
            output_tokens = len(full_content.split())

            metrics = RequestMetrics(
                request_start=request_start,
                first_token_time=first_token_time,
                request_end=request_end,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            self.metrics.append(metrics)
            return full_content

        except Exception as e:
            request_end = time.time()
            failed_metrics = RequestMetrics(
                request_start=request_start,
                first_token_time=None,
                request_end=request_end,
                input_tokens=0,
                output_tokens=0,
            )
            self.metrics.append(failed_metrics)
            raise e

    def get_batch_stats(self) -> BatchInferenceStats:
        if not self.metrics or self._start_time is None:
            return BatchInferenceStats()

        current_time = time.time()
        batch_duration = current_time - self._start_time
        return calculate_batch_stats(self.metrics, batch_duration)

    def reset_metrics(self):
        self.metrics.clear()
        self._start_time = None
