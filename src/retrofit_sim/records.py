"""Operator-facing records: SCADA alarm list and maintenance work-order logs."""
from __future__ import annotations

import numpy as np

ACTION = {
    "BEARING_WEAR": "BEARING_REPLACEMENT",
    "SUCTION_STRAINER_CLOGGING": "STRAINER_CLEANING",
    "CAVITATION": "STRAINER_CLEANING",
    "DISCHARGE_RESTRICTION": "VALVE_OR_HX_SERVICE",
    "MISALIGNMENT": "REALIGNMENT",
    "IMBALANCE": "IMPELLER_CLEANING_BALANCE",
    "IMPELLER_WEAR": "IMPELLER_REPLACEMENT",
    "AIR_ENTRAINMENT": "SUCTION_INSPECTION",
    "SENSOR_DRIFT": "SENSOR_RECALIBRATION",
    "SENSOR_CALIBRATION_SHIFT": "SENSOR_RECALIBRATION",
    "SENSOR_STUCK": "SENSOR_REPLACEMENT",
    "SENSOR_NOISE_SPIKES": "WIRING_REPAIR",
    "COMMUNICATION_LOSS": "GATEWAY_RESTART",
}

_OBS_TEXT = {
    "BEARING_WEAR": [
        ("Drive-end bearing noisy and warm; spalling found on outer race, bearing replaced.", "CONFIRMED"),
        ("Elevated vibration; bearing replaced as precaution, minor wear visible.", "PROBABLE"),
    ],
    "SUCTION_STRAINER_CLOGGING": [
        ("Suction strainer basket fouled with debris; cleaned and flushed.", "CONFIRMED"),
        ("Crackling noise from pump, suspected cavitation.", "UNCERTAIN"),
        ("Low suction pressure; strainer basket cleaned.", "CONFIRMED"),
    ],
    "DISCHARGE_RESTRICTION": [
        ("Discharge isolation valve found partly closed; reopened and locked.", "CONFIRMED"),
        ("Downstream plate cooler backflushed; differential pressure restored.", "PROBABLE"),
    ],
    "MISALIGNMENT": [
        ("Laser alignment check showed parallel/angular offset; motor shimmed and realigned.", "CONFIRMED"),
        ("Coupling element worn; replaced and alignment adjusted.", "PROBABLE"),
    ],
    "IMBALANCE": [
        ("Impeller inspected; rag/fouling removed from vanes, vibration normal.", "CONFIRMED"),
        ("Rotor trim-balanced on site.", "CONFIRMED"),
    ],
    "IMPELLER_WEAR": [
        ("Wear-ring clearance excessive; impeller and wear ring replaced.", "CONFIRMED"),
    ],
    "SENSOR_DRIFT": [
        ("Transmitter zero drift found against test gauge; recalibrated.", "CONFIRMED"),
    ],
    "SENSOR_CALIBRATION_SHIFT": [
        ("Transmitter offset after maintenance; zero/span trimmed.", "CONFIRMED"),
    ],
    "SENSOR_STUCK": [
        ("Signal frozen; transmitter impulse line cleared and module reset.", "CONFIRMED"),
    ],
    "SENSOR_NOISE_SPIKES": [
        ("Intermittent signal spikes; loose terminal tightened in junction box.", "CONFIRMED"),
    ],
    "COMMUNICATION_LOSS": [
        ("Gateway communication timeout; edge gateway power-cycled.", "CONFIRMED"),
    ],
}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end), ...] index intervals where boolean `mask` is True."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    padded = np.r_[False, m, False]
    diff = np.diff(padded.astype(np.int8))
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def operator_text(fault_type: str, rng: np.random.Generator, tag: str | None = None) -> tuple[str, str]:
    choices = _OBS_TEXT.get(fault_type, [("Inspected pump and instrumentation, returned to service.", "UNCERTAIN")])
    idx = int(rng.integers(0, len(choices)))
    txt, conf = choices[idx]
    if tag and fault_type.startswith("SENSOR"):
        txt = f"{tag}: {txt}"
    return txt, conf


def alarms(
    asset_id: str,
    obs: dict[str, np.ndarray],
    present: np.ndarray,
    running_cmd: np.ndarray,
    limits: dict,
    t0_ms: int,
    rng: np.random.Generator,
    qual: np.ndarray,
    night: np.ndarray,
) -> list[dict]:
    """Generate SCADA alarm rows from observed signals and fixed commissioning limits."""
    n = len(present)
    rows: list[dict] = []

    # 1. Gateway communication loss alarms (after 60 s timeout)
    for a, b in _runs(~present):
        if b - a >= 60:
            rows.append(
                {
                    "asset_id": asset_id,
                    "tag": "GATEWAY",
                    "alarm_type": "COMM_LOSS",
                    "priority": 3,
                    "raised_ms": int(t0_ms + (a + 60) * 1000),
                    "cleared_ms": int(t0_ms + b * 1000) if b < n else None,
                    "acknowledged_ms": None,
                    "value_at_raise": None,
                    "limit": None,
                }
            )

    inhibit = (qual & 16) != 0
    active_proc = running_cmd & present & (~inhibit)

    # (col, alarm_type, limit_key, priority, on_delay_s, deadband_frac, needs_running)
    rules = [
        ("vibration_rms_mm_s", "HI", "vib_alarm_mm_s", 2, 15, 0.05, True),
        ("vibration_rms_mm_s", "HIHI", "vib_trip_mm_s", 1, 10, 0.05, True),
        ("bearing_temp_c", "HI", "bearing_alarm_c", 2, 30, 0.03, False),
        ("bearing_temp_c", "HIHI", "bearing_trip_c", 1, 15, 0.03, False),
        ("motor_winding_temp_c", "HI", "winding_alarm_c", 2, 30, 0.03, False),
        ("suction_pressure_kpa", "LO", "suction_lo_kpa", 2, 10, 0.05, True),
        ("discharge_pressure_kpa", "LO", "discharge_lo_kpa", 2, 10, 0.05, True),
        ("flow_m3_h", "LO", "flow_lo_m3h", 2, 15, 0.05, True),
        ("motor_current_a", "HI", "current_hi_a", 2, 15, 0.04, True),
        ("fluid_temp_c", "HI", "fluid_hi_c", 2, 30, 0.03, True),
    ]

    for col, atype, lkey, prio, delay_s, db_frac, need_run in rules:
        if col not in obs or lkey not in limits or limits[lkey] is None:
            continue
        lim = float(limits[lkey])
        y = obs[col].astype(float)
        valid = ~np.isnan(y) & present & (~inhibit)
        if need_run:
            valid &= active_proc

        # Forward-fill sparse channels (e.g. wireless 300 s) for alarm state evaluation
        if np.isnan(y).mean() > 0.5:
            y_eval = y.copy()
            idx_v = np.nonzero(~np.isnan(y_eval))[0]
            if len(idx_v) == 0:
                continue
            for k_i in range(len(idx_v) - 1):
                y_eval[idx_v[k_i] : idx_v[k_i + 1]] = y_eval[idx_v[k_i]]
            y_eval[idx_v[-1] :] = y_eval[idx_v[-1]]
            valid = (~np.isnan(y_eval)) & present & (~inhibit)
            if need_run:
                valid &= active_proc
        else:
            y_eval = y

        db = max(abs(lim) * db_frac, 0.5)
        if atype in ("HI", "HIHI"):
            cond = valid & (y_eval > lim)
            clr_cond = (~valid) | (y_eval < (lim - db))
        else:
            cond = valid & (y_eval < lim)
            clr_cond = (~valid) | (y_eval > (lim + db))

        for a, b in _runs(cond):
            if b - a < delay_s:
                continue
            r_idx = a + delay_s
            # Find when it clears after r_idx
            rem_clr = np.nonzero(clr_cond[r_idx:])[0]
            c_idx = int(r_idx + rem_clr[0]) if len(rem_clr) else None
            if rows and rows[-1]["tag"] == col and rows[-1]["alarm_type"] == atype:
                prev_c = rows[-1]["cleared_ms"]
                if prev_c is None or int(t0_ms + r_idx * 1000) <= prev_c + 15000:
                    continue
            ack_delay = int(rng.uniform(120, 900) if night[min(r_idx, n - 1)] else rng.uniform(30, 300))
            ack_idx = r_idx + ack_delay
            ack_ms = int(t0_ms + ack_idx * 1000) if (ack_idx < n and rng.random() < 0.92) else None
            rows.append(
                {
                    "asset_id": asset_id,
                    "tag": col,
                    "alarm_type": atype,
                    "priority": prio,
                    "raised_ms": int(t0_ms + r_idx * 1000),
                    "cleared_ms": int(t0_ms + c_idx * 1000) if (c_idx is not None and c_idx < n) else None,
                    "acknowledged_ms": ack_ms,
                    "value_at_raise": float(y_eval[r_idx]),
                    "limit": lim,
                }
            )

    return sorted(rows, key=lambda r: r["raised_ms"])

