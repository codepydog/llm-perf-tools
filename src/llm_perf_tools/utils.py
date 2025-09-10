import json
from datetime import datetime
from pathlib import Path


def save_metrics_to_json(
    tracker,
    filename: str | None = None,
    output_dir: str | Path = ".",
) -> str:
    """Export tracker metrics to JSON file.

    Saves both raw request data and computed batch statistics
    to a timestamped JSON file for analysis and reporting.

    Args:
        tracker: InferenceTracker instance with collected metrics
        filename: Output filename (auto-generated if None)
        output_dir: Directory to save file (default: current directory)

    Returns:
        Path to the saved JSON file

    Example:
        >>> import tempfile
        >>> from unittest.mock import MagicMock
        >>> from llm_perf_tools import InferenceTracker
        >>> from llm_perf_tools.types import RequestMetrics
        >>> tracker = InferenceTracker(MagicMock())
        >>> tracker.metrics = [RequestMetrics(request_start=1000.0, request_end=1003.0)]
        >>> tracker._start_time = 1000.0
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     filepath = save_metrics_to_json(tracker, "test.json", tmpdir)
        ...     filepath.endswith("test.json")
        True
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.json"

    if not filename.endswith(".json"):
        filename += ".json"

    file_path = output_path / filename

    batch_start_time = tracker._start_time if tracker._start_time else None
    current_time = datetime.now().timestamp()
    batch_duration = current_time - batch_start_time if batch_start_time else None

    data = {
        "type": "tracker_metrics",
        "timestamp": datetime.now().isoformat(),
        "total_requests": len(tracker.metrics),
        "batch_start_time": batch_start_time,
        "batch_end_time": current_time,
        "batch_duration": batch_duration,
        "raw_metrics": [metric.model_dump() for metric in tracker.metrics],
        "batch_stats": tracker.compute_metrics().model_dump(),
    }

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(file_path)
