#!/usr/bin/env python3
"""Train HDTformer once and evaluate all configured forecasting horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


def make_dataset(split, data_config: dict, horizon: int, target_span: int | None = None):
    from hdtformer.data import AlignedWindowDataset

    stride = horizon if bool(data_config["task_stride_equals_horizon"]) else 1
    return AlignedWindowDataset(
        split=split,
        anchor_lookback=int(data_config["anchor_lookback"]),
        input_lookback=int(data_config["long_lookback"]),
        short_lookback=int(data_config["short_lookback"]),
        horizon=horizon,
        stride=stride,
        target_span=target_span,
    )


def update_recursive_inputs(
    short_x: torch.Tensor,
    long_x: torch.Tensor,
    prediction: torch.Tensor,
    input_scaler,
    target_scaler,
    feature_config: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    previous_raw = input_scaler.inverse_transform(
        short_x[:, -1, :].detach().cpu().numpy()
    )
    prediction_raw = target_scaler.inverse_transform(
        prediction.detach().cpu().numpy()
    )[:, 0]

    if short_x.size(-1) == 1:
        next_raw = prediction_raw[:, None]
    else:
        if not feature_config or not bool(feature_config.get("enabled", False)):
            raise ValueError("Multivariate recursive prediction requires feature_engineering")
        spans = [int(span) for span in feature_config["hema_spans"]]
        expected_features = 2 + len(spans)
        if short_x.size(-1) != expected_features:
            raise ValueError(
                f"Expected {expected_features} short-term features, got {short_x.size(-1)}"
            )
        previous_rv = previous_raw[:, 0]
        next_hema = []
        for column, span in enumerate(spans, start=1):
            alpha = 2.0 / (span + 1.0)
            next_hema.append(
                alpha * previous_rv + (1.0 - alpha) * previous_raw[:, column]
            )
        next_difference = prediction_raw - previous_rv
        next_raw = np.column_stack(
            [prediction_raw, *next_hema, next_difference]
        )

    next_scaled = torch.as_tensor(
        input_scaler.transform(next_raw),
        dtype=short_x.dtype,
        device=short_x.device,
    ).unsqueeze(1)
    short_x = torch.cat([short_x[:, 1:, :], next_scaled], dim=1)
    long_x = torch.cat([long_x[:, 1:, :], next_scaled[:, :, :1]], dim=1)
    return short_x, long_x


def recursive_predict(
    model,
    loader,
    steps: int,
    device: torch.device,
    input_scaler,
    target_scaler,
    feature_config: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for (short_x, long_x), labels in loader:
            short_x = short_x.to(device, non_blocking=True)
            long_x = long_x.to(device, non_blocking=True)
            batch_predictions = []
            for _ in range(steps):
                prediction = model(short_x, long_x)[:, -1].unsqueeze(1)
                batch_predictions.append(prediction)
                short_x, long_x = update_recursive_inputs(
                    short_x,
                    long_x,
                    prediction,
                    input_scaler,
                    target_scaler,
                    feature_config,
                )
            predictions.append(torch.cat(batch_predictions, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))

    from hdtformer.config import load_config, resolve_path
    from hdtformer.data import common_target_span, load_and_split_data, make_loader
    from hdtformer.engine import train_model
    from hdtformer.metrics import regression_metrics
    from hdtformer.model import HDTformer
    from hdtformer.reproducibility import set_random_seed

    config = load_config(args.config)
    experiment_config = config["experiment"]
    data_config = config["data"]
    training_config = config["training"]
    seed = int(args.seed if args.seed is not None else experiment_config.get("seed", 12))
    set_random_seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_dir = args.output_dir
    if output_dir is None:
        configured = experiment_config.get("output_dir", "../outputs")
        output_dir = resolve_path(config, configured) / f"{experiment_config['dataset']}_seed{seed}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = load_and_split_data(
        resolve_path(config, data_config["path"]),
        data_config["target_column"],
        data_config["timestamp_column"],
        float(data_config["train_ratio"]),
        float(data_config["val_ratio"]),
        data_config.get("feature_engineering"),
    )
    train_dataset = make_dataset(prepared.train, data_config, horizon=1)
    val_dataset = make_dataset(prepared.val, data_config, horizon=1)
    train_loader = make_loader(
        train_dataset,
        int(training_config["batch_size"]),
        True,
        int(training_config["num_workers"]),
        bool(training_config["pin_memory"]),
    )
    val_loader = make_loader(
        val_dataset,
        int(training_config["batch_size"]),
        False,
        int(training_config["num_workers"]),
        bool(training_config["pin_memory"]),
    )

    model = HDTformer(
        short_lookback=int(data_config["short_lookback"]),
        **config["model"],
    ).to(device)
    best_state, history, training_summary = train_model(
        model, train_loader, val_loader, device, training_config
    )
    model.load_state_dict(best_state)
    torch.save(
        {"model_state_dict": best_state, "config": config, "seed": seed},
        output_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    tasks = [int(task) for task in data_config["tasks"]]
    target_span = common_target_span(
        len(prepared.test.values), int(data_config["anchor_lookback"]), tasks
    )
    all_metrics = {}
    for horizon in tasks:
        dataset = make_dataset(prepared.test, data_config, horizon, target_span)
        loader = make_loader(
            dataset,
            int(training_config["batch_size"]),
            False,
            int(training_config["num_workers"]),
            bool(training_config["pin_memory"]),
        )
        scaled_predictions, scaled_targets = recursive_predict(
            model,
            loader,
            horizon,
            device,
            prepared.input_scaler,
            prepared.target_scaler,
            data_config.get("feature_engineering"),
        )
        predictions = prepared.target_scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)
        ).reshape(scaled_predictions.shape)
        targets = prepared.target_scaler.inverse_transform(
            scaled_targets.reshape(-1, 1)
        ).reshape(scaled_targets.shape)
        all_metrics[str(horizon)] = regression_metrics(targets, predictions)

        metadata = dataset.target_metadata()
        metadata["actual"] = targets.reshape(-1)
        metadata["prediction"] = predictions.reshape(-1)
        metadata.to_csv(output_dir / f"predictions_h{horizon}.csv", index=False)

    summary = {
        "dataset": experiment_config["dataset"],
        "seed": seed,
        "device": str(device),
        "training": training_summary,
        "metrics": all_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
