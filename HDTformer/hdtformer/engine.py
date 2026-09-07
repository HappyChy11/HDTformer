from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .resources import ResourceMonitor


def _autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
) -> tuple[dict, list[dict], dict]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
    )
    use_scaler = device.type == "cuda" and config["precision"] == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    gradient_clip = config.get("gradient_clip")
    history: list[dict] = []
    best_val_loss = float("inf")
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    monitor = ResourceMonitor(float(config["resource_sample_interval"]))
    with monitor:
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            train_loss = 0.0
            for (short_x, long_x), targets in tqdm(
                train_loader,
                desc=f"Epoch {epoch} train",
                leave=False,
                mininterval=5.0,
            ):
                short_x = short_x.to(device, non_blocking=True)
                long_x = long_x.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, config["precision"]):
                    predictions = model(short_x, long_x)
                    loss = criterion(predictions, targets)
                scaler.scale(loss).backward()
                if gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(gradient_clip)
                    )
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss = validation_loss(
                model, val_loader, criterion, device, config["precision"]
            )
            scheduler.step(val_loss)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            print(
                f"Epoch {epoch}/{config['epochs']} "
                f"train={train_loss:.8f} val={val_loss:.8f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

    summary = {"best_val_loss": best_val_loss, "best_epoch": min(
        history, key=lambda item: item["val_loss"]
    )["epoch"]}
    return best_state, history, {**summary, **monitor.to_dict()}


def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    precision: str,
) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for (short_x, long_x), targets in loader:
            short_x = short_x.to(device, non_blocking=True)
            long_x = long_x.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast_context(device, precision):
                total += criterion(model(short_x, long_x), targets).item()
    return total / len(loader)


def predict(
    model: nn.Module,
    loader: DataLoader,
    steps: int,
    device: torch.device,
    precision: str,
    resource_sample_interval: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    monitor = ResourceMonitor(resource_sample_interval)
    with monitor, torch.inference_mode():
        for (short_x, long_x), labels in tqdm(
            loader, desc=f"Task {steps}", mininterval=5.0
        ):
            short_x = short_x.to(device, non_blocking=True)
            long_x = long_x.to(device, non_blocking=True)
            with _autocast_context(device, precision):
                output = model.predict_recursive(short_x, long_x, steps)
            predictions.append(output.float().cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(predictions), np.concatenate(targets), monitor.to_dict()
