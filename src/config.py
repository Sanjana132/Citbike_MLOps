"""Config loading.

Single entry point for `config/config.yaml`. Values are read once and cached.
Any `${VAR}` string in the YAML is expanded from the environment, so secrets and
host-specific URIs stay in `.env` and never get committed.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside the config."""
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    return value


class Config(dict):
    """Dict with dotted-path access, e.g. ``cfg.get_path("models.lightgbm.objective")``."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get_path(path, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required config key: {path}")
        return value


@lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the project config.

    The env var ``CITIBIKE_CONFIG`` overrides the default path, which lets tests
    and the Airflow containers point at alternate configs without code changes.
    """
    resolved = Path(path or os.environ.get("CITIBIKE_CONFIG") or DEFAULT_CONFIG_PATH)
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config(_expand_env(raw))


def data_dir(*parts: str) -> Path:
    """Resolve a path under the project data directory, creating parents."""
    base = Path(os.environ.get("CITIBIKE_DATA_DIR", PROJECT_ROOT / "data"))
    target = base.joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
