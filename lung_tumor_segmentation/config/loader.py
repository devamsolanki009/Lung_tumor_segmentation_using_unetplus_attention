"""Config loader — reads config/config.yaml and returns a Python dict."""

import os
from pathlib import Path
import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load YAML config and return as a nested dict.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML config file. Defaults to config/config.yaml
        next to this loader module.

    Returns
    -------
    dict
        Fully parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve relative paths relative to the project root (two levels up)
    project_root = Path(__file__).resolve().parent.parent.parent
    _resolve_paths(cfg.get("paths", {}), project_root)

    return cfg


def _resolve_paths(paths_dict: dict, project_root: Path) -> None:
    """Resolve all path values to absolute paths in-place."""
    for key, value in paths_dict.items():
        if isinstance(value, str):
            resolved = (project_root / value).resolve()
            paths_dict[key] = str(resolved)
            # Ensure output directories exist
            if key.startswith("output_"):
                resolved.mkdir(parents=True, exist_ok=True)
