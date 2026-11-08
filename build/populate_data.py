"""Populate missing generated dataset files in data/full and data/starter."""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retrofit_sim import dataset as D
from retrofit_sim import generate as G
from retrofit_sim.fleet import default_limits, load_fleet, _deep_merge

TEST_RAW_ASSETS = [
    {
        "asset_id": "PUMP-10",
        "split": "test",
        "site": "Vessel C (RoPax, fictional)",
        "system": "Seawater cooling, pump No. 1",
        "profile": "SHIP_SW_COOLING",
        "start_utc": "2025-07-21T00:00:00",
        "commissioned_year": 2012,
        "instrumentation_tier": "T1_RETROFIT",
        "q_bep_m3h": 450,
        "h_bep_m": 27.0,
        "eta_bep": 0.82,
        "npshr_bep_m": 4.0,
        "q_design_m3h": 430,
        "h_design_m": 26.5,
        "motor_kw": 45,
        "motor_eff": 0.945,
        "motor_pf": 0.86,
        "rated_rpm": 1780,
        "vib_base_mm_s": 1.1,
        "unbalance_base_mm_s": 0.5,
        "hydraulic_vib_mm_s": 0.9,
        "cavitation_vib_mm_s": 3.0,
        "accel_base_g": 0.15,
        "scenario": [
            {"type": "SUCTION_STRAINER_CLOGGING", "onset_h": 36, "end_h": 92, "repair_h": 1.5, "params": {"rate_per_h": 0.055, "c_max": 3.2, "shape": "linear"}},
            {"type": "SENSOR_DRIFT", "sensor_tag": "suction_pressure_kpa", "onset_h": 30, "end_h": 92, "params": {"drift_per_h": 0.25}},
            {"type": "COMMUNICATION_LOSS", "sensor_tag": "GATEWAY", "onset_h": 116, "end_h": 118.5, "jitter_h": 0.5},
        ],
    },
    {
        "asset_id": "PUMP-11",
        "split": "test",
        "site": "Plant E (paper converting, fictional)",
        "system": "Cooling-water circulation pump CW-11",
        "profile": "INDUSTRIAL_CW",
        "start_utc": "2026-04-20T00:00:00",
        "commissioned_year": 2003,
        "instrumentation_tier": "T0_LEGACY",
        "q_bep_m3h": 400,
        "h_bep_m": 30.0,
        "eta_bep": 0.82,
        "npshr_bep_m": 3.8,
        "q_design_m3h": 380,
        "h_design_m": 29.0,
        "motor_kw": 55,
        "motor_eff": 0.948,
        "motor_pf": 0.86,
        "rated_rpm": 1485,
        "vib_base_mm_s": 1.3,
        "unbalance_base_mm_s": 0.6,
        "hydraulic_vib_mm_s": 1.0,
        "cavitation_vib_mm_s": 3.0,
        "accel_base_g": 0.16,
        "scenario": [
            {"type": "DISCHARGE_RESTRICTION", "onset_h": 42, "end_h": 110, "repair_h": 2.0, "params": {"rate_per_h": 0.04, "extra_max": 1.1, "shape": "exp"}},
            {"type": "SENSOR_CALIBRATION_SHIFT", "sensor_tag": "discharge_pressure_kpa", "onset_h": 124, "end_h": 160, "params": {"step": 22.0}},
        ],
    },
    {
        "asset_id": "PUMP-12",
        "split": "test",
        "site": "Plant D (chemical process, fictional)",
        "system": "Hot process-water transfer pump PT-3",
        "profile": "PROCESS_TRANSFER",
        "start_utc": "2025-11-24T00:00:00",
        "commissioned_year": 2011,
        "instrumentation_tier": "T2_FULL",
        "q_bep_m3h": 52,
        "h_bep_m": 46.0,
        "eta_bep": 0.69,
        "npshr_bep_m": 3.0,
        "q_design_m3h": 50,
        "h_design_m": 45.0,
        "motor_kw": 11,
        "motor_eff": 0.915,
        "motor_pf": 0.84,
        "rated_rpm": 2935,
        "vib_base_mm_s": 1.4,
        "unbalance_base_mm_s": 0.7,
        "hydraulic_vib_mm_s": 1.3,
        "cavitation_vib_mm_s": 3.5,
        "accel_base_g": 0.24,
        "scenario": [
            {"type": "SUCTION_STRAINER_CLOGGING", "onset_h": 48, "end_h": 102, "repair_h": 2.0, "params": {"rate_per_h": 0.12, "c_max": 6.0, "shape": "linear"}},
            {"type": "BEARING_WEAR", "onset_h": 15, "params": {"rate_per_h": 0.011, "t_fail_h": 220}},
        ],
    },
]


def main():
    t0 = time.time()
    cfg, public_assets = load_fleet(str(ROOT / "configs" / "fleet_public.yaml"))
    profiles = cfg["profiles"]
    test_assets = []
    for raw in TEST_RAW_ASSETS:
        base = profiles[raw["profile"]]
        a = _deep_merge(base, raw)
        a.setdefault("design_supply_temp_c", 32.0)
        a["limits"] = _deep_merge(default_limits(a), raw.get("limits", {}))
        a["_limit_overrides"] = sorted(raw.get("limits", {}).keys())
        test_assets.append(a)

    all_assets = public_assets + test_assets
    seeds = {"train": int(cfg["public_seed"]), "validation": int(cfg["public_seed"]), "test": 20261107}

    full_dir = ROOT / "data" / "full"
    starter_dir = ROOT / "data" / "starter"
    starter_dir.mkdir(parents=True, exist_ok=True)
    starter_ids = {"PUMP-01", "PUMP-03", "PUMP-04", "PUMP-06", "PUMP-08"}

    feat_list = []
    ctx_list = []
    snap_list = []
    starter_tele_list = []

    for a in all_assets:
        a = D.enrich_asset(a)
        t1 = time.time()
        res = G.simulate_asset(a, seeds[a["split"]], 7.0, want_detectability=False)
        tele = G.telemetry_frame(res)
        D.write_telemetry(tele, full_dir / "telemetry")
        feat_list.append(D.features_1min(tele))
        ctx_list.append(G.context_frame(res))
        snap_list += D.snapshots_for(res)

        if a["asset_id"] in starter_ids:
            starter_tele_list.append(D.downsample(tele, 10))

        print(f"  Generated {a['asset_id']} ({a['split']}) rows={len(tele)} in {time.time() - t1:.1f}s")

    # Write features_1min.parquet, operating_context.csv, vibration_snapshots.parquet in data/full
    feat_df = pd.concat(feat_list, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(feat_df, preserve_index=False), full_dir / "features_1min.parquet", compression="zstd")

    ctx_df = pd.concat(ctx_list, ignore_index=True)
    ctx_df["timestamp_utc"] = ctx_df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    D._write_csv(ctx_df, full_dir / "operating_context.csv")

    s_df = pd.DataFrame(snap_list)
    tbl_s = pa.table({k: (pa.array(list(s_df[k]), type=pa.list_(pa.float32())) if k == "samples" else pa.array(s_df[k])) for k in s_df.columns})
    pq.write_table(tbl_s, full_dir / "vibration_snapshots.parquet", compression="zstd")

    # Write starter dataset files in data/starter
    starter_tele = pd.concat(starter_tele_list, ignore_index=True)
    starter_tele.to_csv(
        starter_dir / "telemetry.csv.gz",
        index=False,
        float_format="%.4g",
        compression={"method": "gzip", "mtime": 0},
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )

    for fname in ("assets.csv", "events_train.csv", "events_validation.csv", "benign_windows_train_validation.csv", "alarms.csv", "maintenance.csv", "operating_context.csv"):
        df = pd.read_csv(full_dir / fname)
        df_sub = df[df["asset_id"].isin(starter_ids)]
        D._write_csv(df_sub, starter_dir / fname)

    for doc_name in ("data_dictionary.csv", "provenance.md", "schema.json", "simulation_assumptions.md", "units.md"):
        if (full_dir / doc_name).exists():
            shutil.copy2(full_dir / doc_name, starter_dir / doc_name)

    print(f"Done populating data/full and data/starter in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

