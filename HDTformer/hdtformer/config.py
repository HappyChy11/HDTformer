from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "method": "HDTformer",
        "dataset": "BTC",
        "seed": 12,
        "output_dir": "outputs",
    },
    "data": {
        "path": "../Data/RV_BTC.csv",
        "target_column": "realized_volatility_5T",
        "timestamp_column": "timestamp",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "anchor_lookback": 240,
        "short_lookback": 20,
        "long_lookback": 240,
        "tasks": [1, 5, 10, 20],
        "task_stride_equals_horizon": True,
    },
    "model": {
        "embed_dim": 32,
        "dense_dim": 16,
        "num_heads": 4,
        "dropout": 0.1,
        "num_blocks": 4,
        "positional_encoding": True,
        "use_mask": False,
        "max_len": 5000,
        "msdcd": {
            "trend_kernel_size": 20,
            "seasonal_kernel_size": 7,
            "avg_pool_kernel_size": 3,
            "num_basis_kernels": 4,
        },
    },
    "training": {
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.001,
        "num_workers": 0,
        "pin_memory": True,
        "precision": "fp32",
        "gradient_clip": None,
        "scheduler_factor": 0.2,
        "scheduler_patience": 4,
        "resource_sample_interval": 0.2,
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}
    config = _deep_update(deepcopy(DEFAULT_CONFIG), user_config)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["_config_path"]).parent / path).resolve()
