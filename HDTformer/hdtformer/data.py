from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class SplitData:
    name: str
    values: np.ndarray
    targets: np.ndarray
    timestamps: np.ndarray
    global_indices: np.ndarray


@dataclass
class PreparedData:
    train: SplitData
    val: SplitData
    test: SplitData
    input_scaler: MinMaxScaler
    target_scaler: MinMaxScaler


def common_target_span(
    split_length: int, anchor_lookback: int, horizons: list[int]
) -> int:
    """Return a shared target count divisible by every forecasting horizon."""
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    available = split_length - anchor_lookback
    group_size = math.lcm(*horizons)
    span = available // group_size * group_size
    if span < group_size:
        raise ValueError("split is too short for the requested horizons")
    return span


def load_and_split_data(
    csv_path: str | Path,
    target_column: str,
    timestamp_column: str,
    train_ratio: float,
    val_ratio: float,
) -> PreparedData:
    """Load, chronologically split, then normalize using training statistics only."""
    frame = pd.read_csv(csv_path)
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    columns = [target_column]
    if timestamp_column in frame.columns:
        columns.insert(0, timestamp_column)
    frame = frame[columns].dropna(subset=[target_column]).reset_index(drop=True)

    values = frame[[target_column]].to_numpy(dtype=np.float32)
    targets = frame[[target_column]].to_numpy(dtype=np.float32)
    if timestamp_column in frame.columns:
        timestamps = frame[timestamp_column].astype(str).to_numpy()
    else:
        timestamps = np.arange(len(frame)).astype(str)
    global_indices = np.arange(len(frame), dtype=np.int64)

    train_size = int(len(frame) * train_ratio)
    val_size = int(len(frame) * val_ratio)
    boundaries = {
        "train": (0, train_size),
        "val": (train_size, train_size + val_size),
        "test": (train_size + val_size, len(frame)),
    }

    input_scaler = MinMaxScaler(feature_range=(0, 1)).fit(values[:train_size])
    target_scaler = MinMaxScaler(feature_range=(0, 1)).fit(targets[:train_size])
    scaled_values = input_scaler.transform(values).astype(np.float32)
    scaled_targets = target_scaler.transform(targets).astype(np.float32)

    def make_split(name: str) -> SplitData:
        start, end = boundaries[name]
        return SplitData(
            name=name,
            values=scaled_values[start:end],
            targets=scaled_targets[start:end],
            timestamps=timestamps[start:end],
            global_indices=global_indices[start:end],
        )

    return PreparedData(
        train=make_split("train"),
        val=make_split("val"),
        test=make_split("test"),
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )


def load_external_split(
    csv_path: str | Path,
    target_column: str,
    timestamp_column: str,
    input_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    name: str,
) -> SplitData:
    """Load an out-of-sample dataset using scalers fitted on training data."""
    frame = pd.read_csv(csv_path)
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")
    columns = [target_column]
    if timestamp_column in frame.columns:
        columns.insert(0, timestamp_column)
    frame = frame[columns].dropna(subset=[target_column]).reset_index(drop=True)
    raw = frame[[target_column]].to_numpy(dtype=np.float32)
    timestamps = (
        frame[timestamp_column].astype(str).to_numpy()
        if timestamp_column in frame.columns
        else np.arange(len(frame)).astype(str)
    )
    return SplitData(
        name=name,
        values=input_scaler.transform(raw).astype(np.float32),
        targets=target_scaler.transform(raw).astype(np.float32),
        timestamps=timestamps,
        global_indices=np.arange(len(frame), dtype=np.int64),
    )


def load_external_time_range(
    csv_path: str | Path,
    target_column: str,
    timestamp_column: str,
    input_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    name: str,
    start_timestamp: str,
    end_timestamp: str,
) -> SplitData:
    """Load an exact inclusive UTC interval and apply training-fitted scalers."""
    frame = pd.read_csv(csv_path)
    if target_column not in frame.columns or timestamp_column not in frame.columns:
        raise ValueError(
            f"{csv_path} must contain {timestamp_column} and {target_column}"
        )
    parsed = pd.to_datetime(frame[timestamp_column], utc=True)
    start = pd.Timestamp(start_timestamp)
    end = pd.Timestamp(end_timestamp)
    mask = (parsed >= start) & (parsed <= end)
    selected = frame.loc[mask, [timestamp_column, target_column]].copy()
    selected["_source_index"] = np.flatnonzero(mask)
    selected = selected.dropna(subset=[target_column]).reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No rows found in {start_timestamp} -- {end_timestamp}")
    if pd.Timestamp(selected[timestamp_column].iloc[0]) != start:
        raise ValueError(f"Missing exact start timestamp {start_timestamp}")
    if pd.Timestamp(selected[timestamp_column].iloc[-1]) != end:
        raise ValueError(f"Missing exact end timestamp {end_timestamp}")

    raw = selected[[target_column]].to_numpy(dtype=np.float32)
    return SplitData(
        name=name,
        values=input_scaler.transform(raw).astype(np.float32),
        targets=target_scaler.transform(raw).astype(np.float32),
        timestamps=selected[timestamp_column].astype(str).to_numpy(),
        global_indices=selected["_source_index"].to_numpy(dtype=np.int64),
    )


class AlignedWindowDataset(Dataset):
    """Create samples using a common anchor shared by all forecasting methods.

    A method with a shorter lookback receives the tail of the common anchor window,
    while its target starts at the same row as every other method's target.
    """

    def __init__(
        self,
        split: SplitData,
        anchor_lookback: int,
        input_lookback: int,
        horizon: int,
        stride: int,
        short_lookback: int | None = None,
        target_span: int | None = None,
    ) -> None:
        if input_lookback > anchor_lookback:
            raise ValueError("input_lookback cannot exceed anchor_lookback")
        if short_lookback is not None and short_lookback > input_lookback:
            raise ValueError("short_lookback cannot exceed input_lookback")
        if horizon < 1 or stride < 1:
            raise ValueError("horizon and stride must be positive")

        self.split = split
        self.anchor_lookback = anchor_lookback
        self.input_lookback = input_lookback
        self.short_lookback = short_lookback
        self.horizon = horizon
        self.stride = stride

        available_target_values = len(split.values) - anchor_lookback
        if target_span is None:
            target_span = available_target_values
        if target_span > available_target_values:
            raise ValueError("target_span exceeds the available target interval")
        if target_span < horizon:
            raise ValueError("target_span must be at least as large as horizon")
        self.target_span = target_span
        last_start = target_span - horizon
        self.starts = np.arange(0, last_start + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        anchor_start = int(self.starts[index])
        target_start = anchor_start + self.anchor_lookback
        input_start = target_start - self.input_lookback

        model_input = self.split.values[input_start:target_start]
        target = self.split.targets[target_start:target_start + self.horizon, 0]

        if self.short_lookback is None:
            inputs = torch.from_numpy(model_input)
        else:
            short_input = model_input[-self.short_lookback:]
            inputs = (torch.from_numpy(short_input), torch.from_numpy(model_input[:, :1]))

        return inputs, torch.from_numpy(target)

    def target_metadata(self) -> pd.DataFrame:
        """Return one row per target value, preserving forecast origin and horizon."""
        rows: list[dict[str, Any]] = []
        for sample_index, anchor_start in enumerate(self.starts):
            target_start = int(anchor_start + self.anchor_lookback)
            origin_position = target_start - 1
            for step in range(self.horizon):
                position = target_start + step
                rows.append({
                    "sample_index": sample_index,
                    "forecast_origin_index": int(self.split.global_indices[origin_position]),
                    "forecast_origin_timestamp": self.split.timestamps[origin_position],
                    "horizon_step": step + 1,
                    "target_index": int(self.split.global_indices[position]),
                    "target_timestamp": self.split.timestamps[position],
                })
        return pd.DataFrame(rows)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
