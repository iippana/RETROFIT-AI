"""Read-only telemetry replay utility for causal stream inspection."""
from __future__ import annotations

import argparse
import os
import pandas as pd


def replay(data_dir: str, asset_id: str, start_utc: str | None = None, end_utc: str | None = None, step_s: int = 60) -> pd.DataFrame:
    tpath = os.path.join(data_dir, "telemetry", f"asset_id={asset_id}")
    if os.path.exists(tpath):
        df = pd.read_parquet(tpath)
    else:
        df = pd.read_csv(os.path.join(data_dir, "telemetry.csv.gz"))
        df = df[df["asset_id"] == asset_id]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc")
    if start_utc:
        df = df[df["timestamp_utc"] >= pd.to_datetime(start_utc, utc=True)]
    if end_utc:
        df = df[df["timestamp_utc"] <= pd.to_datetime(end_utc, utc=True)]
    if step_s > 1 and len(df):
        df = df.iloc[:: max(1, step_s // int(df["sample_interval_s"].iloc[0]))]
    return df.reset_index(drop=True)


def replay_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrofit_sim replay")
    p.add_argument("--data", required=True)
    p.add_argument("--asset", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--step-s", type=int, default=60)
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args(argv)
    df = replay(a.data, a.asset, a.start, a.end, a.step_s)
    print(df.head(a.limit).to_string(index=False))
    return 0

