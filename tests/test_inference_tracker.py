from types import SimpleNamespace

import pytest
from llm_perf_tools.inference import InferenceTracker


@pytest.mark.asyncio
async def test_create_chat_completion_tracks_metrics(mocker):
    # Arrange
    tokens = ["hello", " world"]

    async def fake_response():
        for token in tokens:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=token))]
            )

    mock_create = mocker.AsyncMock(return_value=fake_response())
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )

    tracker = InferenceTracker(mock_client)

    mocker.patch(
        "llm_perf_tools.inference.time.perf_counter",
        side_effect=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    messages = [{"role": "user", "content": "hello world"}]
    model = "gpt-test"

    # Act
    result = await tracker.create_chat_completion(
        messages=messages, model=model, max_tokens=5
    )

    # Assert
    assert result == "hello world"
    assert len(tracker.metrics) == 1

    metric = tracker.metrics[0]
    assert metric.request_start == 1.0
    assert metric.first_token_time == 2.0
    assert metric.request_end == 3.0
    assert metric.input_tokens == 2
    assert metric.output_tokens == 2

    stats = tracker.compute_metrics()
    assert stats.total_input_tokens == 2
    assert stats.total_output_tokens == 2
    assert stats.avg_input_tokens == 2
    assert stats.avg_output_tokens == 2
    assert stats.overall_tps == pytest.approx(1.0)

    mock_create.assert_called_once_with(
        model=model, messages=messages, stream=True, max_tokens=5
    )
