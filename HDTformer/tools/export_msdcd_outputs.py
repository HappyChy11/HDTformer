from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


def inverse_transform(scaler, values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(values.shape)


def load_model(checkpoint_path: Path, config: dict, model_ctor) -> object:
    model = model_ctor(short_lookback=int(config["data"]["short_lookback"]), **config["model"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def extract_sample(
    model,
    split,
    dataset,
    sample_index: int,
    input_scaler,
    data_config: dict,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict]:
    anchor_lookback = int(data_config["anchor_lookback"])
    long_lookback = int(data_config["long_lookback"])
    short_lookback = int(data_config["short_lookback"])

    anchor_start = int(dataset.starts[sample_index])
    target_start = anchor_start + anchor_lookback
    input_start = target_start - long_lookback
    short_start = target_start - short_lookback

    (_short_x, long_x), _target = dataset[sample_index]
    if isinstance(long_x, np.ndarray):
        long_tensor = torch.from_numpy(long_x).unsqueeze(0).float()
    else:
        long_tensor = long_x.unsqueeze(0).float()

    with torch.inference_mode():
        trend, seasonal, residual, original = model.ltv.msdcd(long_tensor)

    trend = trend.squeeze(0).cpu().numpy()
    seasonal = seasonal.squeeze(0).cpu().numpy()
    residual = residual.squeeze(0).cpu().numpy()
    original = original.squeeze(0).cpu().numpy()

    frame = pd.DataFrame({
        "dataset": dataset_name,
        "sample_index": np.full(long_lookback, sample_index, dtype=np.int64),
        "position": np.arange(long_lookback, dtype=np.int32),
        "global_index": split.global_indices[input_start:target_start],
        "timestamp": split.timestamps[input_start:target_start],
        "original_scaled": original,
        "trend_scaled": trend,
        "cyclic_scaled": seasonal,
        "residual_scaled": residual,
        "original": inverse_transform(input_scaler, original),
        "trend": inverse_transform(input_scaler, trend),
        "cyclic": inverse_transform(input_scaler, seasonal),
        "residual": inverse_transform(input_scaler, residual),
    })

    metadata = {
        "dataset": dataset_name,
        "sample_index": int(sample_index),
        "anchor_start": anchor_start,
        "input_start": input_start,
        "short_start": short_start,
        "target_start": target_start,
        "forecast_origin_timestamp": str(split.timestamps[target_start - 1]),
        "forecast_origin_index": int(split.global_indices[target_start - 1]),
        "short_start_timestamp": str(split.timestamps[short_start]),
        "short_end_timestamp": str(split.timestamps[target_start - 1]),
    }
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MSDCD decomposition for BNB/BTC windows.")
    parser.add_argument("--source-root", default="/home/chen/HDTformer/HDTformer", help="Local root containing this HDTformer copy.")
    parser.add_argument("--checkpoint-root", default="/home/chen/HDTformer/mymodel/checkpoints", help="Checkpoint root with BNB_HDTformer_multiseed and BTC_HDTformer_multiseed.")
    parser.add_argument("--output-dir", default="/home/chen/HDTformer/visualization_source_data", help="Output base directory.")
    parser.add_argument("--seed", type=int, default=12, help="Model seed.")
    parser.add_argument("--samples", type=int, default=3, help="Number of test samples to export per dataset.")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root))

    from hdtformer.config import load_config, resolve_path
    from hdtformer.data import AlignedWindowDataset, load_and_split_data
    from hdtformer.model import HDTformer

    output_dir = Path(args.output_dir).resolve()
    vis_root = output_dir / "MSDCD"
    vis_root.mkdir(parents=True, exist_ok=True)

    dataset_names = ["BNB", "BTC"]
    summary_rows = []

    for dataset_name in dataset_names:
        config_path = source_root / "configs" / f"{dataset_name.lower()}_multiseed.yaml"
        config = load_config(config_path)

        checkpoint = Path(args.checkpoint_root) / f"{dataset_name}_HDTformer_multiseed" / f"HDTformer_{dataset_name}_seed{args.seed}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        model = load_model(checkpoint, config, HDTformer)

        data_config = config["data"]
        prepared = load_and_split_data(
            resolve_path(config, data_config["path"]),
            data_config["target_column"],
            data_config["timestamp_column"],
            float(data_config["train_ratio"]),
            float(data_config["val_ratio"]),
        )

        test_dataset = AlignedWindowDataset(
            split=prepared.test,
            anchor_lookback=int(data_config["anchor_lookback"]),
            input_lookback=int(data_config["long_lookback"]),
            short_lookback=int(data_config["short_lookback"]),
            horizon=1,
            stride=1,
        )

        total = len(test_dataset)
        if args.samples <= 0 or args.samples > total:
            raise ValueError(f"samples={args.samples} invalid for {dataset_name}; test windows={total}")

        dataset_dir = vis_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for sample_index in range(args.samples):
            frame, metadata = extract_sample(
                model=model,
                split=prepared.test,
                dataset=test_dataset,
                sample_index=sample_index,
                input_scaler=prepared.input_scaler,
                data_config=data_config,
                dataset_name=dataset_name,
            )
            frame.to_csv(dataset_dir / f"{dataset_name}_sample_{sample_index:03d}_msdcd.csv", index=False)
            summary_rows.append(metadata)

    pd.DataFrame(summary_rows).to_csv(vis_root / "msdcd_summary.csv", index=False)


if __name__ == "__main__":
    main()
