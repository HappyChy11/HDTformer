#!/usr/bin/env python3
"""Download Binance 1-minute spot klines and construct rolling 5-minute RV."""

from __future__ import annotations

import argparse
import io
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return pd.period_range(start=start, end=end, freq="M").strftime("%Y-%m").tolist()


def convert_open_time(value: str) -> pd.Timestamp:
    timestamp = int(value)
    # Binance archives use microseconds from 2025 onward and milliseconds before 2025.
    if timestamp >= 100_000_000_000_000:
        timestamp //= 1_000
    return pd.Timestamp(datetime.fromtimestamp(timestamp / 1_000, tz=timezone.utc))


def download_month(symbol: str, month: str, directory: Path) -> Path:
    filename = f"{symbol}-1m-{month}.zip"
    destination = directory / filename
    url = f"{BASE_URL}/{symbol}/1m/{filename}"
    request = urllib.request.Request(url, headers={"User-Agent": "HDTformer/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    return destination


def read_month(archive: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as zipped:
        members = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {archive.name}, found {members}")
        with zipped.open(members[0]) as raw:
            frame = pd.read_csv(
                io.BytesIO(raw.read()),
                header=None,
                usecols=[0, 1, 4],
                names=["open_time", "open", "close"],
            )
    frame["timestamp"] = frame["open_time"].astype(str).map(convert_open_time)
    frame["open"] = pd.to_numeric(frame["open"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    return frame[["timestamp", "open", "close"]]


def construct_rv(
    prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, window: int
) -> pd.DataFrame:
    prices = prices.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    full_index = pd.date_range(start=start, end=end, freq="min", tz="UTC")
    prices = prices.reindex(full_index)

    missing = prices["close"].isna()
    previous_close = prices["close"].ffill()
    if previous_close.isna().any():
        raise ValueError("A leading Binance bar is missing; extend the requested start time")
    prices.loc[missing, "open"] = previous_close.loc[missing]
    prices.loc[missing, "close"] = previous_close.loc[missing]

    log_return = np.log(prices["close"] / prices["open"])
    rv = np.sqrt(log_return.pow(2).rolling(window=window, min_periods=window).sum())
    output = pd.DataFrame(
        {
            "timestamp": full_index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            f"realized_volatility_{window}T": rv.to_numpy(),
        }
    )
    return output.dropna().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, choices=["BNBUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"])
    parser.add_argument("--start", required=True, help="Inclusive UTC timestamp")
    parser.add_argument("--end", required=True, help="Inclusive UTC timestamp")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("--start and --end must include an explicit UTC offset")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if start > end or args.window < 1:
        raise ValueError("Invalid date interval or rolling window")

    with tempfile.TemporaryDirectory(prefix="hdtformer_binance_") as directory:
        temporary = Path(directory)
        frames = [
            read_month(download_month(args.symbol, month, temporary))
            for month in month_range(start, end)
        ]
    prices = pd.concat(frames, ignore_index=True)
    output = construct_rv(prices, start, end, args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} RV observations to {args.output}")


if __name__ == "__main__":
    main()
