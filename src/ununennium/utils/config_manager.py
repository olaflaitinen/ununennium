"""Configuration manager."""

import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """Manage YAML configurations."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
    def load(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
            
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        return self.config
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self.config.get(key, default)
