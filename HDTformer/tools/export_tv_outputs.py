from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch


def compute_hema(values: np.ndarray, window: int) -> np.ndarray:
    alpha = 2.0 / (window + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx - 1] + (1 - alpha) * out[idx - 1]
    return out


def extract_window_features(
    split,
    sample_index: int,
    dataset: AlignedWindowDataset,
    data_config: dict,
    temporal_weights: np.ndarray,
    temporal_biases: np.ndarray,
    input_scaler,
    use_original: bool = True,
) -> pd.DataFrame:
    anchor_lookback = int(data_config["anchor_lookback"])
    short_lookback = int(data_config["short_lookback"])

    anchor_start = int(dataset.starts[sample_index])
    target_start = anchor_start + anchor_lookback
    short_start = target_start - short_lookback

    if split.values.shape[1] < 4:
        raise ValueError(
            "TV export requires the configured RV and three HEMA input features"
        )
    raw_features = input_scaler.inverse_transform(split.values)
    raw = raw_features[:, 0]
    scaled = split.values[:, 0]
    hema_60_scaled = split.values[:, 1]
    hema_720_scaled = split.values[:, 2]
    hema_1440_scaled = split.values[:, 3]
    hema_60 = raw_features[:, 1]
    hema_720 = raw_features[:, 2]
    hema_1440 = raw_features[:, 3]

    pos_idx = np.arange(short_start, target_start, dtype=np.int64)

    rv_scaled = scaled[pos_idx]
    h1_scaled = hema_60_scaled[pos_idx]
    h12_scaled = hema_720_scaled[pos_idx]
    h1d_scaled = hema_1440_scaled[pos_idx]

    selected_inputs = np.column_stack(
        [rv_scaled, h1_scaled, h12_scaled, h1d_scaled]
    )
    expected_shape = (short_lookback, selected_inputs.shape[1])
    if temporal_weights.shape != expected_shape or temporal_biases.shape != expected_shape:
        raise ValueError(
            "TV parameter shapes do not match the selected short-term features: "
            f"expected {expected_shape}, got weights={temporal_weights.shape}, "
            f"biases={temporal_biases.shape}"
        )
    periodic = np.sin(temporal_weights * selected_inputs + temporal_biases)
    rv_periodic, h1_periodic, h12_periodic, h1d_periodic = periodic.T

    out = pd.DataFrame({
        "dataset": split.name,
        "sample_index": int(sample_index),
        "position": np.arange(short_lookback, dtype=np.int32),
        "global_index": split.global_indices[pos_idx],
        "timestamp": split.timestamps[pos_idx],
        "rv_scaled": rv_scaled,
        "hema_1h_scaled": h1_scaled,
        "hema_12h_scaled": h12_scaled,
        "hema_1d_scaled": h1d_scaled,
        "rv": raw[pos_idx],
        "hema_1h": hema_60[pos_idx],
        "hema_12h": hema_720[pos_idx],
        "hema_1d": hema_1440[pos_idx],
        "rv_periodic": rv_periodic,
        "hema_1h_periodic": h1_periodic,
        "hema_12h_periodic": h12_periodic,
        "hema_1d_periodic": h1d_periodic,
    })
    if use_original:
        return out
    return out.drop(columns=["rv", "hema_1h", "hema_12h", "hema_1d"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TV input and extracted periodic features for BNB short windows with HEMA.")
    parser.add_argument("--source-root", default="/home/chen/HDTformer/HDTformer")
    parser.add_argument("--checkpoint-root", default="/home/chen/HDTformer/mymodel/checkpoints", help="Checkpoint root containing BNB_HDTformer_multiseed.")
    parser.add_argument("--output-dir", default="/home/chen/HDTformer/visualization_source_data")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--exclude-raw", action="store_true", help="Only keep scaled fields.")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root))

    from hdtformer.model import HDTformer
    from hdtformer.config import load_config, resolve_path
    from hdtformer.data import AlignedWindowDataset, load_and_split_data

    config_path = source_root / "configs" / "tv_multivariate_visualization.yaml"
    config = load_config(config_path)

    checkpoint = Path(args.checkpoint_root) / "BNB_HDTformer_multiseed" / f"HDTformer_BNB_seed{args.seed}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    model = HDTformer(
        short_lookback=int(config["data"]["short_lookback"]),
        **config["model"],
    )
    model.load_state_dict(state)
    model.eval()

    data_config = config["data"]
    prepared = load_and_split_data(
        resolve_path(config, data_config["path"]),
        data_config["target_column"],
        data_config["timestamp_column"],
        float(data_config["train_ratio"]),
        float(data_config["val_ratio"]),
        data_config.get("feature_engineering"),
    )

    dataset = AlignedWindowDataset(
        split=prepared.test,
        anchor_lookback=int(data_config["anchor_lookback"]),
        input_lookback=int(data_config["long_lookback"]),
        short_lookback=int(data_config["short_lookback"]),
        horizon=1,
        stride=1,
    )
    if args.samples <= 0 or args.samples > len(dataset):
        raise ValueError(f"samples={args.samples} is invalid for TV extraction; test windows={len(dataset)}")

    selected_indices = model.std.temporal_vector.periodic_feature_indices
    if selected_indices != (0, 1, 2, 3):
        raise ValueError(
            "This exporter expects periodic_feature_indices=[0, 1, 2, 3] "
            f"for RV and the three HEMA features, got {selected_indices}"
        )
    weights = model.std.temporal_vector.weights_periodic.detach().cpu().numpy()
    biases = model.std.temporal_vector.bias_periodic.detach().cpu().numpy()

    vis_root = Path(args.output_dir).resolve() / "TV" / "BNB"
    vis_root.mkdir(parents=True, exist_ok=True)

    split = prepared.test
    summary_rows = []
    for sample_index in range(args.samples):
        out = extract_window_features(
            split,
            sample_index=sample_index,
            dataset=dataset,
            data_config=data_config,
            temporal_weights=weights,
            temporal_biases=biases,
            input_scaler=prepared.input_scaler,
            use_original=not args.exclude_raw,
        )
        out.to_csv(vis_root / f"BNB_sample_{sample_index:03d}_tv_with_hema.csv", index=False)
        summary_rows.append({
            "dataset": "BNB",
            "sample_index": sample_index,
            "anchor_start": int(dataset.starts[sample_index]),
            "forecast_origin_timestamp": str(split.timestamps[int(dataset.starts[sample_index]) + int(data_config['anchor_lookback']) - 1]),
            "forecast_origin_index": int(split.global_indices[int(dataset.starts[sample_index]) + int(data_config['anchor_lookback']) - 1]),
            "short_start": int(int(dataset.starts[sample_index]) + int(data_config['anchor_lookback']) - int(data_config['short_lookback'])),
            "short_end": int(int(dataset.starts[sample_index]) + int(data_config['anchor_lookback']) - 1),
        })

    pd.DataFrame(summary_rows).to_csv(vis_root / "tv_summary.csv", index=False)


if __name__ == "__main__":
    main()
