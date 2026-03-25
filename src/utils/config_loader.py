import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    # Convert the input path into a Path object for easier handling
    path = Path(path)

    # Open the YAML file and safely parse its contents into a Python dictionary
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Return the loaded configuration dictionary
    return config