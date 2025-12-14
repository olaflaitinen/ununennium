"""Metrics logger for experiment tracking."""

import csv
import json
from pathlib import Path


class MetricsLogger:
    """Log metrics to JSON/CSV files."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: list[dict[str, float]] = []

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log a dictionary of metrics at a given step."""
        entry = {"step": step, **metrics}
        self.metrics_history.append(entry)

        # Append to CSV
        csv_path = self.log_dir / "metrics.csv"
        file_exists = csv_path.exists()

        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

    def save_summary(self):
        """Save full history to JSON."""
        with (self.log_dir / "metrics_history.json").open("w") as f:
            json.dump(self.metrics_history, f, indent=2)
