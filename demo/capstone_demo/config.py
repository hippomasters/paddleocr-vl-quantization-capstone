"""Host configuration: explicit absolute paths, bounded numeric settings."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

from .paths import CLIENT_SCRIPT, MANIFEST, SERVE_SCRIPT, PARSE_SCRIPT

VARIANT_ORDER = ("BF16", "W8A16", "W4A16")
LAYOUT_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
DEVICE_PATTERN = re.compile(r"cpu|gpu:0")

# Defaults for one 8 GB laptop GPU. They are starting points to verify on the
# host, not measured settings; the study used 0.82 / 24 seqs on a 24 GB L4.
DEFAULTS = {
    "layout_device": "cpu",
    "vllm_port": 8000,
    "gpu_index": 0,
    "gpu_memory_utilization": 0.70,
    "max_model_len": 8192,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 8192,
    "server_ready_timeout_s": 900,
    "client_timeout_s": 1800,
    "warmup": True,
    "max_upload_mb": 20,
}
INT_BOUNDS = {
    "vllm_port": (1024, 65535),
    "gpu_index": (0, 15),
    "max_model_len": (1024, 131072),
    "max_num_seqs": (1, 64),
    "max_num_batched_tokens": (1024, 131072),
    "server_ready_timeout_s": (10, 7200),
    "client_timeout_s": (10, 7200),
    "max_upload_mb": (1, 50),
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DemoConfig:
    vllm_python: str
    paddle_python: str
    layout_dir: str
    models: dict
    layout_device: str
    vllm_port: int
    gpu_index: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    server_ready_timeout_s: int
    client_timeout_s: int
    warmup: bool
    max_upload_mb: int

    @property
    def variant_order(self):
        return VARIANT_ORDER

    @property
    def max_upload_bytes(self):
        return self.max_upload_mb * 1024 * 1024


def _absolute(value, key):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ConfigError(f"{key} must be a non-empty path string")
    if not Path(value).is_absolute():
        raise ConfigError(f"{key} must be an absolute path: {value!r}")
    return value


def load_config(path) -> DemoConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config must be a JSON object")
    required = {"vllm_python", "paddle_python", "layout_dir", "models"}
    unknown = set(raw) - required - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
    missing = required - set(raw)
    if missing:
        raise ConfigError(f"Missing config keys: {sorted(missing)}")
    data = {**DEFAULTS, **raw}
    for key in ("vllm_python", "paddle_python", "layout_dir"):
        _absolute(data[key], key)
    models = data["models"]
    if not isinstance(models, dict) or set(models) != set(VARIANT_ORDER):
        raise ConfigError(f"models must map exactly {list(VARIANT_ORDER)} to directories")
    models = {name: _absolute(models[name], f"models.{name}") for name in VARIANT_ORDER}
    data["models"] = models
    if not isinstance(data["layout_device"], str) or not DEVICE_PATTERN.fullmatch(
        data["layout_device"]
    ):
        raise ConfigError("layout_device must be 'cpu' or 'gpu:0' (the same single GPU as vLLM)")
    for key, (low, high) in INT_BOUNDS.items():
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ConfigError(f"{key} must be an integer in [{low}, {high}]")
    utilization = data["gpu_memory_utilization"]
    if isinstance(utilization, bool) or not isinstance(utilization, (int, float)) or not 0.1 <= utilization <= 0.95:
        raise ConfigError("gpu_memory_utilization must be in [0.1, 0.95]")
    data["gpu_memory_utilization"] = float(utilization)
    if not isinstance(data["warmup"], bool):
        raise ConfigError("warmup must be true or false")
    return DemoConfig(**data)


def readiness(config: DemoConfig) -> list[str]:
    """Cheap existence checks. Hashes are verified by serve_model.py at launch."""
    problems = []
    for script in (SERVE_SCRIPT, PARSE_SCRIPT, CLIENT_SCRIPT, MANIFEST):
        if not script.is_file():
            problems.append(f"Missing demo/source file: {script}")
    for key in ("vllm_python", "paddle_python"):
        path = Path(getattr(config, key))
        if not path.is_file() or not os.access(path, os.X_OK):
            problems.append(f"{key} is not an executable file: {path}")
    for name in LAYOUT_FILES:
        if not (Path(config.layout_dir) / name).is_file():
            problems.append(f"Layout directory missing {name}: {config.layout_dir}")
    for variant, directory in config.models.items():
        for name in ("config.json", "model.safetensors"):
            if not (Path(directory) / name).is_file():
                problems.append(f"{variant} model directory missing {name}: {directory}")
    return problems
