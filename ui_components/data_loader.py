"""Data caching and loader module for Retrofit AI Streamlit Console.

Loads and caches assets, telemetry, 1-min features, events, alarms, and AI diagnostic pipeline predictions.
Supports both data/full and data/starter paths.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return the best available data directory (prefers full over starter)."""
    full_dir = ROOT_DIR / "data" / "full"
    if full_dir.exists() and (full_dir / "assets.csv").exists():
        return full_dir
    starter_dir = ROOT_DIR / "data" / "starter"
    if starter_dir.exists() and (starter_dir / "assets.csv").exists():
        return starter_dir
    raise FileNotFoundError(f"Neither data/full nor data/starter found in {ROOT_DIR}")


@st.cache_data(show_spinner=False)
def load_assets() -> pd.DataFrame:
    """Load the assets table with metadata and instrumentation tiers."""
    d = get_data_dir()
    df = pd.read_csv(d / "assets.csv")
    df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
    df["window_end_utc"] = pd.to_datetime(df["window_end_utc"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_events(asset_id: Optional[str] = None) -> pd.DataFrame:
    """Load known ground-truth events (train and validation splits)."""
    d = get_data_dir()
    frames = []
    train_path = d / "events_train.csv"
    if train_path.exists():
        tr = pd.read_csv(train_path)
        tr["split"] = "train"
        frames.append(tr)
    val_path = d / "events_validation.csv"
    if val_path.exists():
        val = pd.read_csv(val_path)
        val["split"] = "validation"
        frames.append(val)
    if not frames:
        return pd.DataFrame()
    events = pd.concat(frames, ignore_index=True)
    for col in ["onset_utc", "detectable_utc", "action_deadline_utc", "end_utc"]:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], utc=True)
    if asset_id is not None and "asset_id" in events.columns:
        events = events[events["asset_id"] == asset_id].reset_index(drop=True)
    return events


@st.cache_data(show_spinner=False)
def load_alarms() -> pd.DataFrame:
    """Load SCADA plant alarms."""
    d = get_data_dir()
    alarms_path = d / "alarms.csv"
    if not alarms_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(alarms_path)
    for col in ["raised_utc", "cleared_utc", "acknowledged_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_maintenance() -> pd.DataFrame:
    """Load operator maintenance work order logs."""
    d = get_data_dir()
    maint_path = d / "maintenance.csv"
    if not maint_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(maint_path)
    for col in ["start_utc", "end_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_predictions(asset_id: Optional[str] = None) -> pd.DataFrame:
    """Load AI diagnostic pipeline predictions across the fleet."""
    frames = []

    fleet_path = ROOT_DIR / "build" / "predictions_fleet.csv"
    if fleet_path.exists():
        df = pd.read_csv(fleet_path)
        frames.append(df)
    else:
        val_path = ROOT_DIR / "build" / "predictions_validation.csv"
        if val_path.exists():
            v = pd.read_csv(val_path)
            v["split"] = "validation"
            frames.append(v)
        test_path = ROOT_DIR / "predictions_test.csv"
        if not test_path.exists():
            test_path = ROOT_DIR / "submission_template" / "predictions_test.csv"
        if test_path.exists():
            t = pd.read_csv(test_path)
            t["split"] = "test"
            frames.append(t)

    if not frames:
        return pd.DataFrame()

    preds = pd.concat(frames, ignore_index=True)
    if "alert_time_utc" in preds.columns:
        preds["alert_time_utc"] = pd.to_datetime(preds["alert_time_utc"], utc=True, format="ISO8601")
    if "split" not in preds.columns:
        try:
            assets = load_assets()
            split_map = dict(zip(assets["asset_id"], assets["split"]))
            preds["split"] = preds["asset_id"].map(split_map).fillna("unknown")
        except Exception:
            preds["split"] = "unknown"
    preds = preds.sort_values("alert_time_utc").reset_index(drop=True)
    if asset_id is not None and "asset_id" in preds.columns:
        preds = preds[preds["asset_id"] == asset_id].reset_index(drop=True)
    return preds


@st.cache_data(show_spinner=False)
def load_benchmark_metrics() -> Dict[str, Any]:
    """Load AI diagnostic pipeline benchmark metrics JSON."""
    bm_json = ROOT_DIR / "build" / "benchmark_metrics.json"
    out: Dict[str, Any] = {"metrics_available": False}
    if bm_json.exists():
        try:
            with open(bm_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            out.update(data)
            out["metrics_available"] = True
        except Exception:
            pass
    return out


def run_ai_pipeline() -> Tuple[bool, str]:
    """Execute src/pipeline.py and clear cached UI data."""
    script_path = ROOT_DIR / "src" / "pipeline.py"
    if not script_path.exists():
        return False, f"Pipeline script not found at {script_path}"
    try:
        r = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        st.cache_data.clear()
        if r.returncode == 0:
            return True, r.stdout[-1500:] if r.stdout else "AI diagnostic pipeline completed successfully."
        return False, (r.stderr or r.stdout)[-1500:]
    except Exception as exc:
        return False, str(exc)


@st.cache_data(show_spinner=False)
def load_features_1min(asset_id: Optional[str] = None) -> pd.DataFrame:
    """Load 1-minute aggregated telemetry features for fast plotting.
    
    If data/full/features_1min.parquet exists, it reads that.
    Otherwise falls back to starter telemetry.csv.gz.
    """
    full_pq = ROOT_DIR / "data" / "full" / "features_1min.parquet"
    if full_pq.exists():
        try:
            if asset_id:
                df = pd.read_parquet(full_pq, filters=[("asset_id", "==", asset_id)])
            else:
                df = pd.read_parquet(full_pq)
            if "window_start_utc" in df.columns:
                df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
                df = df.rename(columns={"window_start_utc": "timestamp_utc"})
            return df
        except Exception:
            pass

    # Fallback to starter telemetry.csv.gz
    starter_csv = ROOT_DIR / "data" / "starter" / "telemetry.csv.gz"
    if starter_csv.exists():
        df = pd.read_csv(starter_csv)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        if asset_id:
            df = df[df["asset_id"] == asset_id]
        return df

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_vibration_snapshots(asset_id: Optional[str] = None) -> pd.DataFrame:
    """Load raw 10 kHz vibration snapshot waveforms if available."""
    snap_path = ROOT_DIR / "data" / "full" / "vibration_snapshots.parquet"
    if snap_path.exists():
        try:
            if asset_id:
                df = pd.read_parquet(snap_path, filters=[("asset_id", "==", asset_id)])
            else:
                df = pd.read_parquet(snap_path)
            if "timestamp_utc" in df.columns:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def get_asset_info(asset_id: str) -> Optional[pd.Series]:
    """Retrieve full specification series for an asset."""
    assets = load_assets()
    sub = assets[assets["asset_id"] == asset_id]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def get_active_events(asset_id: str, timestamp_utc: pd.Timestamp) -> pd.DataFrame:
    """Return ground truth events active at the specified timestamp."""
    ts = pd.to_datetime(timestamp_utc, utc=True)
    events = load_events()
    if events.empty:
        return pd.DataFrame()
    sub = events[events["asset_id"] == asset_id]
    active = sub[
        (sub["onset_utc"] <= ts) & 
        ((sub["end_utc"].isna()) | (sub["end_utc"] >= ts))
    ]
    return active


def get_active_alarms(asset_id: str, timestamp_utc: pd.Timestamp) -> pd.DataFrame:
    """Return SCADA alarms active at the specified timestamp."""
    ts = pd.to_datetime(timestamp_utc, utc=True)
    alarms = load_alarms()
    if alarms.empty:
        return pd.DataFrame()
    sub = alarms[alarms["asset_id"] == asset_id]
    active = sub[
        (sub["raised_utc"] <= ts) & 
        ((sub["cleared_utc"].isna()) | (sub["cleared_utc"] >= ts))
    ]
    return active


def get_active_predictions(
    asset_id: str,
    timestamp_utc: pd.Timestamp,
    lookback_hours: float = 36.0,
    lookahead_hours: float = 2.0,
) -> pd.DataFrame:
    """Return AI alerts on `asset_id` active near `timestamp_utc`, sorted by proximity."""
    ts = pd.to_datetime(timestamp_utc, utc=True)
    preds = load_predictions()
    if preds.empty:
        return pd.DataFrame()
    sub = preds[preds["asset_id"] == asset_id].copy()
    if sub.empty:
        return pd.DataFrame()
    lo = ts - pd.Timedelta(hours=lookback_hours)
    hi = ts + pd.Timedelta(hours=lookahead_hours)
    near = sub[(sub["alert_time_utc"] >= lo) & (sub["alert_time_utc"] <= hi)].copy()
    if near.empty:
        return pd.DataFrame()
    near["_dt_s"] = (near["alert_time_utc"] - ts).abs().dt.total_seconds()
    return near.sort_values("_dt_s").drop(columns=["_dt_s"]).reset_index(drop=True)
