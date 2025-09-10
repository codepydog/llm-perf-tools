import time
from typing import Any

from .types import RequestMetrics, InferenceStats, BatchInferenceStats


def time_to_first_token(metrics: RequestMetrics) -> float | None:
    """Calculate time from request start to first token received.

    Measures the latency between sending a request and receiving
    the first token of the response. Lower values indicate better
    responsiveness.

    Args:
        metrics: RequestMetrics containing timing information

    Returns:
        Time in seconds, or None if first token time not recorded
    """
    if metrics.first_token_time is None:
        return None
    return metrics.first_token_time - metrics.request_start


def end_to_end_latency(metrics: RequestMetrics) -> float | None:
    """Calculate total request processing time.

    Measures complete request duration from start to finish.
    Includes both time to first token and generation time.

    Args:
        metrics: RequestMetrics containing timing information

    Returns:
        Total time in seconds, or None if request not completed
    """
    if metrics.request_end is None:
        return None
    return metrics.request_end - metrics.request_start


def inter_token_latency(metrics: RequestMetrics) -> float | None:
    """Calculate average time between consecutive tokens.

    Measures token generation consistency by computing the average
    interval between tokens during the generation phase.

    Args:
        metrics: RequestMetrics with timing and token count data

    Returns:
        Average seconds per token, or None if insufficient data
    """
    if metrics.first_token_time is None or metrics.request_end is None:
        return None
    if metrics.output_tokens <= 1:
        return None
    generation_time = metrics.request_end - metrics.first_token_time
    return generation_time / (metrics.output_tokens - 1)


def tokens_per_second(metrics: list[RequestMetrics]) -> float | None:
    """Calculate overall token generation throughput.

    Measures tokens generated per second across multiple requests,
    accounting for parallel processing and overlapping requests.

    Args:
        metrics: List of RequestMetrics from multiple requests

    Returns:
        Tokens per second throughput, or None if no completed requests
    """
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


def requests_per_second(metrics: list[RequestMetrics], duration: float) -> float | None:
    """Calculate request processing rate.

    Measures how many requests are completed per second
    during a given time period.

    Args:
        metrics: List of RequestMetrics to analyze
        duration: Time period in seconds

    Returns:
        Completed requests per second, or None if invalid duration
    """
    if duration <= 0:
        return None
    completed_requests = len([m for m in metrics if m.request_end is not None])
    return completed_requests / duration


def compute_stats(metrics: RequestMetrics | list[RequestMetrics]) -> InferenceStats:
    """Compute inference statistics for single request or batch.

    Calculates key performance metrics including time to first token,
    end-to-end latency, and tokens per second.

    Args:
        metrics: Single RequestMetrics instance or list of RequestMetrics

    Returns:
        InferenceStats containing computed performance metrics

    Example:
        >>> from llm_perf_tools.types import RequestMetrics
        >>> request_metrics = RequestMetrics(
        ...     request_start=1000.0,
        ...     first_token_time=1001.5,
        ...     request_end=1003.0,
        ...     output_tokens=20
        ... )
        >>> stats = compute_stats(request_metrics)
        >>> stats.ttft > 0
        True
    """

    if isinstance(metrics, RequestMetrics):
        generation_time = (
            metrics.request_end - metrics.first_token_time
            if metrics.first_token_time and metrics.request_end
            else None
        )
        tps = (
            metrics.output_tokens / generation_time
            if generation_time and generation_time > 0
            else None
        )
        return InferenceStats(
            ttft=time_to_first_token(metrics),
            e2e_latency=end_to_end_latency(metrics),
            itl=inter_token_latency(metrics),
            tps=tps,
        )

    stats = InferenceStats()
    if metrics:
        stats.tps = tokens_per_second(metrics)

    return stats


def percentile(values: list[float], percentile: float) -> float:
    """Calculate percentile value from a list of numbers.

    Computes the specified percentile using linear interpolation
    method for statistical analysis of performance metrics.

    Args:
        values: List of numeric values to analyze
        percentile: Percentile to calculate (0-100)

    Returns:
        Percentile value, or 0.0 if values list is empty
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((percentile / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def compute_batch_metrics(
    metrics_list: list[RequestMetrics], batch_duration: float
) -> BatchInferenceStats:
    """Compute comprehensive batch-level performance metrics.

    Analyzes multiple requests to calculate percentiles, averages,
    and other aggregate statistics for batch processing evaluation.

    Args:
        metrics_list: List of RequestMetrics from batch requests
        batch_duration: Total time in seconds for batch processing

    Returns:
        BatchInferenceStats with percentiles, averages, and totals

    Example:
        >>> from llm_perf_tools.types import RequestMetrics
        >>> metrics = [
        ...     RequestMetrics(request_start=1000.0, first_token_time=1001.0, request_end=1003.0, output_tokens=20),
        ...     RequestMetrics(request_start=1001.0, first_token_time=1002.0, request_end=1004.0, output_tokens=25)
        ... ]
        >>> batch_stats = compute_batch_metrics(metrics, 10.5)
        >>> batch_stats.total_requests
        2
    """

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

    total_input_tokens = sum(m.input_tokens for m in successful_metrics)
    total_output_tokens = sum(m.output_tokens for m in successful_metrics)
    avg_input_tokens = (
        total_input_tokens / len(successful_metrics) if successful_metrics else None
    )
    avg_output_tokens = (
        total_output_tokens / len(successful_metrics) if successful_metrics else None
    )

    rps = len(successful_metrics) / batch_duration if batch_duration > 0 else 0

    return BatchInferenceStats(
        avg_ttft=sum(ttft_values) / len(ttft_values) if ttft_values else None,
        p50_ttft=percentile(ttft_values, 50) if ttft_values else None,
        p95_ttft=percentile(ttft_values, 95) if ttft_values else None,
        p99_ttft=percentile(ttft_values, 99) if ttft_values else None,
        min_ttft=min(ttft_values) if ttft_values else None,
        max_ttft=max(ttft_values) if ttft_values else None,
        avg_e2e_latency=sum(e2e_values) / len(e2e_values) if e2e_values else None,
        p50_e2e_latency=percentile(e2e_values, 50) if e2e_values else None,
        p95_e2e_latency=percentile(e2e_values, 95) if e2e_values else None,
        p99_e2e_latency=percentile(e2e_values, 99) if e2e_values else None,
        min_e2e_latency=min(e2e_values) if e2e_values else None,
        max_e2e_latency=max(e2e_values) if e2e_values else None,
        avg_itl=sum(itl_values) / len(itl_values) if itl_values else None,
        p50_itl=percentile(itl_values, 50) if itl_values else None,
        p95_itl=percentile(itl_values, 95) if itl_values else None,
        p99_itl=percentile(itl_values, 99) if itl_values else None,
        min_itl=min(itl_values) if itl_values else None,
        max_itl=max(itl_values) if itl_values else None,
        avg_tps=sum(tps_values) / len(tps_values) if tps_values else None,
        p50_tps=percentile(tps_values, 50) if tps_values else None,
        p5_tps=percentile(tps_values, 5) if tps_values else None,
        p1_tps=percentile(tps_values, 1) if tps_values else None,
        min_tps=min(tps_values) if tps_values else None,
        max_tps=max(tps_values) if tps_values else None,
        overall_tps=tokens_per_second(successful_metrics),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        rps=rps,
        total_requests=len(metrics_list),
        successful_requests=len(successful_metrics),
    )


class InferenceTracker:
    """Tracks performance metrics for LLM inference requests.

    Wraps an OpenAI client to measure response times, token throughput,
    and other key performance indicators automatically.

    Args:
        client: OpenAI async client for making requests

    Example:
        Track metrics for a single request:

        .. code-block:: python

            from openai import AsyncOpenAI
            from llm_perf_tools import InferenceTracker

            client = AsyncOpenAI()
            tracker = InferenceTracker(client)

            response = await tracker.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-5"
            )

            metrics = tracker.compute_metrics()
            print(f"Time to first token: {metrics.avg_ttft:.3f}s")
    """

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
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        user: str | None = None,
        **kwargs,
    ) -> str:
        """Chat completion API compatible with OpenAI client.
        
        Same interface as OpenAI's create() method, except stream=True 
        is always enforced for performance metrics collection.
        """
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

    def compute_metrics(self) -> BatchInferenceStats:
        if not self.metrics or self._start_time is None:
            return BatchInferenceStats()

        current_time = time.time()
        batch_duration = current_time - self._start_time
        return compute_batch_metrics(self.metrics, batch_duration)

    def reset(self):
        self.metrics.clear()
        self._start_time = None
