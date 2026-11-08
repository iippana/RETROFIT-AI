"""Instrument models: channel specifications, transmitter lag, noise, quantization, sparse updates, and sensor faults."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fleet import rated_current_a
from .plant import _first_order_lag

QUALITY_BITS = {
    "COMM_LOSS": 1,
    "OUT_OF_RANGE": 2,
    "STALE": 4,
    "BAD_SENSOR_STATUS": 8,
    "STARTUP_INHIBIT": 16,
}


@dataclass
class ChannelSpec:
    col: str
    truth_key: str
    lo: float
    hi: float
    sigma: float
    rel_sigma: float
    resolution: float
    tau_s: float
    update_s: int
    source: str
    offset_sd: float = 0.0
    gain_sd: float = 0.002
    sparse_nan: bool = False


def channel_specs(acfg: dict) -> dict[str, ChannelSpec]:
    """Build instrument specifications for the asset's instrumentation tier and control mode."""
    tier = acfg["instrumentation_tier"]
    ctl = acfg["control"]
    is_vfd = ctl.startswith("VFD")
    i_r = rated_current_a(acfg)
    q_bep = float(acfg["q_bep_m3h"])
    p_kw = float(acfg["motor_kw"])
    w_upd = int(acfg.get("wireless_update_s", 300)) if tier == "T1_RETROFIT" else 1
    vib_src = "WIRELESS" if tier == "T1_RETROFIT" else "IOLINK"
    sparse_w = tier == "T1_RETROFIT"

    specs: dict[str, ChannelSpec] = {
        "motor_current_a": ChannelSpec(
            "motor_current_a", "current_a", 0.0, min(600.0, round(2.5 * i_r, 0)),
            max(0.15, round(0.004 * 2.0 * i_r, 2)), 0.003, 0.1, 2.0, 1, "VFD" if is_vfd else "PLC", offset_sd=0.1
        ),
        "discharge_pressure_kpa": ChannelSpec(
            "discharge_pressure_kpa", "discharge_kpa_g", -100.0, 1600.0,
            1.2, 0.002, 0.1, 2.0, 1, "PLC", offset_sd=0.4
        ),
        "fluid_temp_c": ChannelSpec(
            "fluid_temp_c", "fluid_temp_c", -20.0, 120.0,
            0.15, 0.0, 0.1, 12.0, 1, "PLC", offset_sd=0.15
        ),
        "ambient_temp_c": ChannelSpec(
            "ambient_temp_c", "ambient_temp_c", -30.0, 70.0,
            0.15, 0.0, 0.1, 20.0, 10, "PLC", offset_sd=0.2, sparse_nan=False
        ),
    }

    if is_vfd:
        specs["motor_speed_rpm"] = ChannelSpec(
            "motor_speed_rpm", "rpm", 0.0, 4000.0,
            1.5, 0.001, 0.5, 1.0, 1, "VFD", offset_sd=0.0, gain_sd=0.0005
        )
        specs["motor_power_kw"] = ChannelSpec(
            "motor_power_kw", "input_kw", 0.0, min(200.0, round(2.0 * p_kw, 0)),
            max(0.08, round(0.005 * p_kw, 2)), 0.004, 0.05, 2.0, 1, "VFD", offset_sd=0.05
        )

    if tier in ("T1_RETROFIT", "T2_FULL"):
        specs["suction_pressure_kpa"] = ChannelSpec(
            "suction_pressure_kpa", "suction_kpa_g", -100.0, 600.0,
            0.6, 0.002, 0.1, 2.0, 1, "PLC", offset_sd=0.25
        )
        specs["bearing_temp_c"] = ChannelSpec(
            "bearing_temp_c", "bearing_temp_c", -20.0, 150.0,
            0.15, 0.0, 0.1, 15.0, w_upd, vib_src, offset_sd=0.2, sparse_nan=sparse_w
        )
        specs["vibration_rms_mm_s"] = ChannelSpec(
            "vibration_rms_mm_s", "vib_rms_mm_s", 0.0, 50.0,
            0.04, 0.015, 0.01, 3.0, w_upd, vib_src, offset_sd=0.02, sparse_nan=sparse_w
        )
        specs["vibration_accel_rms_g"] = ChannelSpec(
            "vibration_accel_rms_g", "vib_accel_rms_g", 0.0, 16.0,
            0.008, 0.015, 0.005, 3.0, w_upd, vib_src, offset_sd=0.004, sparse_nan=sparse_w
        )
        specs["vibration_crest_factor"] = ChannelSpec(
            "vibration_crest_factor", "vib_crest", 1.0, 20.0,
            0.08, 0.01, 0.01, 3.0, w_upd, vib_src, offset_sd=0.03, sparse_nan=sparse_w
        )

    if tier == "T2_FULL":
        specs["flow_m3_h"] = ChannelSpec(
            "flow_m3_h", "flow_m3h", 0.0, min(1200.0, round(1.8 * q_bep, 0)),
            max(0.2, round(0.005 * q_bep, 2)), 0.008, 0.1, 3.0, 1, "FLOW_METER", offset_sd=0.3
        )
        if "motor_power_kw" not in specs:
            specs["motor_power_kw"] = ChannelSpec(
                "motor_power_kw", "input_kw", 0.0, min(200.0, round(2.0 * p_kw, 0)),
                max(0.08, round(0.005 * p_kw, 2)), 0.004, 0.05, 2.0, 1, "POWER_METER", offset_sd=0.05
            )
        specs["motor_winding_temp_c"] = ChannelSpec(
            "motor_winding_temp_c", "winding_temp_c", -20.0, 200.0,
            0.2, 0.0, 0.1, 15.0, 1, "PLC", offset_sd=0.25
        )
        if ctl in ("VFD_PRESSURE", "FLOW_VALVE"):
            specs["valve_position_pct"] = ChannelSpec(
                "valve_position_pct", "valve_position_pct", 0.0, 100.0,
                0.2, 0.0, 0.1, 2.0, 1, "PLC", offset_sd=0.1
            )

    return specs


def observe(
    truth: dict,
    acfg: dict,
    specs: dict[str, ChannelSpec],
    sensor_faults: list,
    comm: dict,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate observed telemetry channels, quality flags, and row-presence mask."""
    n = len(truth["speed_ratio"])
    obs: dict[str, np.ndarray] = {}
    qual = np.zeros(n, dtype=np.int16)
    present = np.ones(n, dtype=bool)

    for a, b in comm.get("gateway_outages", []):
        present[max(0, int(a)) : min(n, int(b))] = False

    stopped = truth["speed_ratio"] <= 0.005

    for col in sorted(specs.keys()):
        c = specs[col]
        x = np.asarray(truth[c.truth_key], dtype=float)
        lagged = _first_order_lag(x, c.tau_s, float(x[0]))
        gain = 1.0 + float(rng.normal(0.0, c.gain_sd))
        offset = float(rng.normal(0.0, c.offset_sd))
        noise_sd = c.sigma + c.rel_sigma * np.abs(lagged)
        y = gain * lagged + offset + rng.normal(0.0, noise_sd, size=n)

        if col in ("motor_current_a", "motor_speed_rpm", "motor_power_kw", "flow_m3_h"):
            y = np.where(stopped, 0.0, y)

        # Apply channel-specific instrument faults
        for f in sensor_faults:
            if f.sensor_tag != col:
                continue
            a = max(0, int(f.onset_s))
            b = n if f.end_s is None else min(n, int(f.end_s))
            if b <= a:
                continue
            if f.fault_type == "SENSOR_DRIFT":
                rate = float(f.params.get("drift_per_h", 0.1))
                y[a:b] = y[a:b] + rate * (np.arange(b - a, dtype=float) / 3600.0)
            elif f.fault_type == "SENSOR_CALIBRATION_SHIFT":
                y[a:b] = y[a:b] + float(f.params.get("step", 10.0))
            elif f.fault_type == "SENSOR_NOISE_SPIKES":
                mult = float(f.params.get("noise_mult", 4.0))
                s_rate = float(f.params.get("spike_rate", 0.003))
                s_size = float(f.params.get("spike_size", 50.0))
                extra = rng.normal(0.0, max(mult - 1.0, 1.0) * c.sigma, size=b - a)
                spikes = (rng.random(b - a) < s_rate) * rng.choice([-1.0, 1.0], size=b - a) * s_size
                y[a:b] = y[a:b] + extra + spikes
            elif f.fault_type == "SENSOR_STUCK":
                ref_idx = max(0, a - 1)
                stuck_val = float(np.round(np.clip(y[ref_idx], c.lo, c.hi) / c.resolution) * c.resolution)
                y[a:b] = stuck_val
            elif f.fault_type == "COMMUNICATION_LOSS":
                y[a:b] = np.nan
                qual[a:b] |= QUALITY_BITS["COMM_LOSS"]

        # Range clipping and OUT_OF_RANGE bit
        oor = (y > c.hi) | (y < (c.lo - (1.0 if c.lo == 0.0 else 0.0)))
        qual[np.nan_to_num(oor, nan=False).astype(bool)] |= QUALITY_BITS["OUT_OF_RANGE"]
        y = np.clip(y, c.lo, c.hi)
        if col in ("suction_pressure_kpa", "discharge_pressure_kpa"):
            y = np.maximum(y, -100.0)

        # Quantization
        if c.resolution > 0:
            y = np.round(y / c.resolution) * c.resolution

        # Update intervals: zero-order hold for slow wired channels, sparse samples for wireless
        if c.update_s > 1:
            if c.sparse_nan:
                mask_keep = (np.arange(n) % c.update_s) == 0
                y = np.where(mask_keep, y, np.nan)
            else:
                idx_hold = (np.arange(n) // c.update_s) * c.update_s
                y = y[idx_hold]
                # Re-apply stuck fault if active so boundaries stay exact
                for f in sensor_faults:
                    if f.sensor_tag == col and f.fault_type == "SENSOR_STUCK":
                        a = max(0, int(f.onset_s))
                        b = n if f.end_s is None else min(n, int(f.end_s))
                        if b > a:
                            y[a:b] = y[max(0, a - 1)]

        # Random single-sample drops (0.15 % on non-sparse channels outside stuck windows)
        if not c.sparse_nan:
            drops = rng.random(n) < 0.0015
            for f in sensor_faults:
                if f.sensor_tag == col and f.fault_type == "SENSOR_STUCK":
                    drops[max(0, int(f.onset_s)) : (n if f.end_s is None else min(n, int(f.end_s)))] = False
            y = np.where(drops, np.nan, y)

        obs[col] = y.astype(np.float32)

    return obs, qual, present

