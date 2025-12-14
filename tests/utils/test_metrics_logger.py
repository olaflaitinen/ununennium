"""Tests for metrics logger."""

import pytest
import json
from ununennium.utils.metrics_logger import MetricsLogger

def test_metrics_logger(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log_metrics({"loss": 0.5}, step=1)
    logger.log_metrics({"loss": 0.4}, step=2)
    logger.save_summary()
    
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics_history.json").exists()
    
    with open(tmp_path / "metrics_history.json") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["loss"] == 0.5
