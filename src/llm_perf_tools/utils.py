import json
from datetime import datetime
from pathlib import Path


def save_metrics_to_json(
    tracker,
    filename: str | None = None,
    output_dir: str | Path = ".",
) -> str:
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
