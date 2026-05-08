import shutil
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clear_cache(cache_dir: str):
    shutil.rmtree(cache_dir, ignore_errors=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
