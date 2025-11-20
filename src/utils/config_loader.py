import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config