"""Tests for config manager."""

import pytest
import yaml
from ununennium.utils.config_manager import ConfigManager

def test_config_load(tmp_path):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"lr": 0.001, "batch_size": 32}, f)
        
    manager = ConfigManager(config_path)
    cfg = manager.load()
    
    assert cfg["lr"] == 0.001
    assert manager.get("batch_size") == 32
