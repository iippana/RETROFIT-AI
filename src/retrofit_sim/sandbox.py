"""Read-only counterfactual what-if sandbox for operator decision support."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import physics as ph
from .fleet import default_limits, load_fleet

INTERVENTIONS = {
    "clean_strainer": {"stop_duration_h": 1.5, "labour_h": 3.0, "parts_eur": 50.0, "description": "Isolate pump, open basket strainer, clean and flush."},
    "replace_bearing": {"stop_duration_h": 6.0, "labour_h": 12.0, "parts_eur": 900.0, "description": "Replace drive-end rolling element bearing and regrease."},
    "realign": {"stop_duration_h": 3.0, "labour_h": 6.0, "parts_eur": 150.0, "description": "Laser shaft alignment of motor and pump coupling."},
    "speed_setpoint": {"stop_duration_h": 0.0, "labour_h": 0.25, "parts_eur": 0.0, "description": "Adjust variable-frequency drive speed ratio setpoint."},
    "valve_position": {"stop_duration_h": 0.0, "labour_h": 0.25, "parts_eur": 0.0, "description": "Adjust discharge throttling / control valve position (%)."},
    "none": {"stop_duration_h": 0.0, "labour_h": 0.0, "parts_eur": 0.0, "description": "No intervention (evaluate baseline only)."},
}

HYPOTHESIS_KEYS = {
    "clog": (0.0, 8.0),
    "bearing_wear": (0.0, 0.95),
    "restriction": (1.0, 4.0),
    "misalignment": (0.0, 1.0),
    "unbalance": (0.0, 2.0),
    "impeller_wear": (0.0, 0.15),
    "air_entrainment": (0.0, 0.08),
}

ELECTRICITY_EUR_KWH = 0.12
LABOUR_EUR_H = 85.0


def _eval_point(acfg: dict, ctx_row: dict, state: dict, speed_override: float | None = None, valve_override: float | None = None) -> dict:
    q_bep = float(acfg["q_bep_m3h"]) / 3600.0
    q_des = float(acfg["q_design_m3h"]) / 3600.0
    h_bep = float(acfg["h_bep_m"])
    h_des = float(acfg["h_design_m"])
    h0 = float(acfg.get("h0_shutoff_ratio", 1.20))
    h_static = float(acfg.get("system_static_m", 3.0))

    suction_loss_des = float(acfg.get("suction_loss_m_at_design", 0.7))
    K_s0 = suction_loss_des / max(q_des ** 2, 1e-9)
    s_share = float(acfg.get("strainer_share", 0.6))
    clog = float(state.get("clog", 0.05))
    restr = float(state.get("restriction", 1.0))
    bw = float(state.get("bearing_wear", 0.05))
    mis = float(state.get("misalignment", 0.02))
    unb = float(state.get("unbalance", 0.0))
    iw = float(state.get("impeller_wear", 0.0))
    air = float(state.get("air_entrainment", 0.0))

    K_suction = K_s0 * (1.0 + s_share * (((1.0 + clog) ** 2) - 1.0))
    K_des = max((h_des - h_static) / max(q_des ** 2, 1e-9), 10.0)
    x_des = float(acfg.get("valve_design_position", 0.85))
    v_share = float(acfg.get("valve_share_of_K", 0.20))
    K_v_des = v_share * K_des
    K_v_base = K_v_des / (50.0 ** (2.0 * (1.0 - x_des)))
    K_pipe0 = max(K_des - K_v_des - K_s0, 0.1 * K_des)
    K_pipe = K_pipe0 * restr

    demand = float(ctx_row.get("duty_demand_pct", 80.0)) / 100.0
    t_supply = float(ctx_row.get("supply_water_temp_c", 20.0))
    t_amb = float(ctx_row.get("ambient_temp_c", 28.0))
    z_s = float(ctx_row.get("suction_static_head_m", acfg.get("context", {}).get("suction_static_m", acfg.get("context", {}).get("sea_chest_submergence_m", 3.5))))
    if np.isnan(z_s):
        z_s = float(acfg.get("context", {}).get("sea_chest_submergence_m", 3.5))

    ctl = acfg["control"]
    r_min = float(acfg.get("r_min", 0.50))
    r_max = float(acfg.get("r_max", 1.0))

    if ctl == "VFD_TEMPERATURE":
        T_set = float(acfg.get("controlled_temp_setpoint_c", 36.0))
        T_des = float(acfg.get("design_supply_temp_c", 25.0))
        q_frac = float(np.clip(demand * (T_set - T_des) / max(T_set - t_supply, 2.0), float(acfg.get("min_flow_frac", 0.40)), 1.15))
        Q_req = q_frac * q_des
        valve_pos = x_des if valve_override is None else float(valve_override) / 100.0
        K_valve = K_v_base * (50.0 ** (2.0 * (1.0 - np.clip(valve_pos, 0.1, 1.0))))
        K_tot = K_pipe + K_valve + K_suction
        r_req = float(np.sqrt(max(0.0, ((h_static + K_tot * (Q_req ** 2)) / h_bep + (h0 - 1.0) * ((Q_req / q_bep) ** 2)) / h0)))
        r = float(np.clip(r_req, r_min, r_max)) if speed_override is None else float(speed_override)
    elif ctl == "VFD_PRESSURE":
        Q_req = float(np.clip(demand, 0.35, 1.15)) * q_des
        valve_pos = float(np.clip(0.35 + 0.52 * demand, 0.25, 0.98)) if valve_override is None else float(valve_override) / 100.0
        K_valve = K_v_base * (50.0 ** (2.0 * (1.0 - np.clip(valve_pos, 0.1, 1.0))))
        dp_set_m = float(acfg.get("dp_setpoint_m", 0.85 * h_des))
        Q_dp = float(np.sqrt(max(dp_set_m - h_static, 0.5) / max(K_pipe + K_valve, 1e-6)))
        Q_ctrl = max(Q_dp, 0.96 * Q_req)
        H_need = h_static + (K_pipe + K_valve + K_suction) * (Q_ctrl ** 2)
        r_req = float(np.sqrt(max(0.0, (H_need / h_bep + (h0 - 1.0) * ((Q_ctrl / q_bep) ** 2)) / h0)))
        r = float(np.clip(r_req, r_min, r_max)) if speed_override is None else float(speed_override)
    elif ctl == "FLOW_VALVE":
        Q_req = float(np.clip(demand, 0.3, 1.15)) * q_des
        r = 1.0 if speed_override is None else float(speed_override)
        if valve_override is None:
            H_avail = h_bep * (h0 - (h0 - 1.0) * ((Q_req / q_bep) ** 2))
            K_v_req = max(0.0, (H_avail - h_static) / max(Q_req ** 2, 1e-9) - K_pipe - K_suction)
            valve_pos = float(np.clip(1.0 - 0.5 * np.log(max(K_v_req / max(K_v_base, 1e-9), 1.0)) / np.log(50.0), 0.15, 1.0))
        else:
            valve_pos = float(valve_override) / 100.0
        K_valve = K_v_base * (50.0 ** (2.0 * (1.0 - np.clip(valve_pos, 0.1, 1.0))))
    else:
        Q_req = max(demand, 0.5) * q_des
        r = 1.0 if speed_override is None else float(speed_override)
        valve_pos = x_des if valve_override is None else float(valve_override) / 100.0
        K_valve = K_v_base * (50.0 ** (2.0 * (1.0 - np.clip(valve_pos, 0.1, 1.0))))

    K_total = K_pipe + K_valve + K_suction
    rho = float(ph.water_density(t_supply, float(acfg.get("salinity_psu", 0.0))))
    p_v = float(ph.vapour_pressure_pa(t_supply))
    p_atm = 101325.0
    p_loop = float(acfg.get("loop_pressurisation_kpa", 0.0)) * 1000.0

    f_non_cav = (1.0 - iw) * (1.0 - 2.5 * air)
    q_hi = float(ph.solve_operating_point(r, h_bep, q_bep, h0, h_static, K_total, f_non_cav))
    q_lo = 0.0
    for _ in range(15):
        q_mid = 0.5 * (q_lo + q_hi)
        p_tot_m = max(p_v + 150.0, p_atm + p_loop + rho * ph.G_ACC * (z_s - K_suction * (q_mid ** 2)))
        npsha_m = (p_tot_m - p_v) / (rho * ph.G_ACC)
        npshr_m = float(ph.npsh3(q_mid, r, q_bep, float(acfg["npshr_bep_m"])))
        f_cav_m, _ = ph.cavitation_head_factor(npsha_m, npshr_m)
        q_eval = float(ph.solve_operating_point(r, h_bep, q_bep, h0, h_static, K_total, f_non_cav * float(f_cav_m)))
        if q_eval >= q_mid:
            q_lo = q_mid
        else:
            q_hi = q_mid

    Q = 0.5 * (q_lo + q_hi)
    p_tot = max(p_v + 150.0, p_atm + p_loop + rho * ph.G_ACC * (z_s - K_suction * (Q ** 2)))
    npsha = (p_tot - p_v) / (rho * ph.G_ACC)
    npshr = float(ph.npsh3(Q, r, q_bep, float(acfg["npshr_bep_m"])))
    f_cav, cav_int = ph.cavitation_head_factor(npsha, npshr)
    f_cav, cav_int = float(f_cav), float(cav_int)
    npsh_margin = npsha / max(npshr, 0.1)

    f_head = f_non_cav * f_cav
    H = float(ph.pump_head(Q, r, h_bep, q_bep, h0, f_head))
    q_rel = Q / max(r * q_bep, 1e-9)

    v_suc = 2.5 * (Q / max(q_bep, 1e-9))
    suction_kpa_g = (max(p_v, p_tot - 0.5 * rho * (v_suc ** 2)) - p_atm) / 1000.0
    discharge_kpa_g = (p_tot + rho * ph.G_ACC * H - p_atm) / 1000.0

    mech_drag = 1.0 + 0.04 * (bw ** 2) + 0.02 * mis
    P_sh_w = float(ph.shaft_power(Q, H, r, q_bep, h_bep, float(acfg["eta_bep"]), h0, rho, f_head, mech_drag))
    P_in_w, _, _, I_a, lam = ph.motor_electrical(
        P_sh_w,
        float(acfg["motor_kw"]) * 1000.0,
        float(acfg["motor_eff"]),
        float(acfg["motor_pf"]),
        float(acfg["voltage_v"]),
        1.0,
        float(acfg.get("const_loss_share", 0.35)),
        is_vfd=ctl.startswith("VFD"),
        rated_vfd_eff=float(acfg.get("vfd_efficiency", 0.97)),
    )
    P_in_kw = float(P_in_w) / 1000.0
    lam = float(lam)

    bearing_temp_c = (
        0.6 * t_amb
        + 0.4 * t_supply
        + float(acfg.get("bearing_rise_k", 24.0)) * (r ** 1.3) * (0.75 + 0.25 * lam)
        + 40.0 * (max(bw - 0.4, 0.0) ** 1.6) * (0.4 + 0.6 * r)
        + 6.0 * mis * r
    )
    v_s = float(acfg.get("vib_base_mm_s", 1.2)) * (0.4 + 0.6 * (r ** 1.5))
    v_u = (float(acfg.get("unbalance_base_mm_s", 0.6)) + 4.0 * unb) * (r ** 2)
    v_m = 4.5 * mis * (0.5 + 0.5 * lam) * r
    v_h = float(acfg.get("hydraulic_vib_mm_s", 1.0)) * (r ** 2) * (abs(q_rel - 1.0) ** 1.5) * (3.0 if q_rel < 0.5 else 1.0)
    v_c = float(acfg.get("cavitation_vib_mm_s", 3.0)) * cav_int * (r ** 2) + 3.0 * air * r
    v_b = 14.0 * (max(bw - 0.4, 0.0) ** 1.5) * (0.3 + 0.7 * r)
    vib_rms = float(np.sqrt(v_s ** 2 + v_u ** 2 + v_m ** 2 + v_h ** 2 + v_c ** 2 + v_b ** 2))

    duty_sat = float(np.clip((Q / max(Q_req, 1e-6)), 0.0, 1.5))
    risk = float(np.clip(0.4 * cav_int + 0.3 * (vib_rms / 8.0) + 0.3 * max(0.0, 1.0 - duty_sat), 0.0, 1.0))

    return {
        "speed_ratio": round(r, 3),
        "valve_position_pct": round(valve_pos * 100.0, 1),
        "flow_m3h": round(Q * 3600.0, 1),
        "flow_required_m3h": round(Q_req * 3600.0, 1),
        "duty_satisfaction": round(duty_sat, 3),
        "head_m": round(H, 2),
        "suction_kpa_g": round(suction_kpa_g, 1),
        "discharge_kpa_g": round(discharge_kpa_g, 1),
        "npsha_m": round(npsha, 2),
        "npshr_m": round(npshr, 2),
        "npsh_margin": round(npsh_margin, 2),
        "shaft_kw": round(P_sh_w / 1000.0, 2),
        "input_kw": round(P_in_kw, 2),
        "current_a": round(float(I_a), 1),
        "motor_load": round(lam, 3),
        "bearing_temp_c": round(bearing_temp_c, 1),
        "vib_rms_mm_s": round(vib_rms, 2),
        "risk_proxy": round(risk, 3),
        "energy_eur_per_day": round(P_in_kw * 24.0 * ELECTRICITY_EUR_KWH, 2),
    }


def what_if(
    data_dir: str,
    asset_id: str,
    time_utc: str,
    action: str = "none",
    value: float | None = None,
    hypothesis: dict | None = None,
    root: str | None = None,
) -> dict:
    """Evaluate a counterfactual intervention in read-only mode."""
    root_path = Path(root) if root else Path(__file__).resolve().parents[2]
    data_path = Path(data_dir)
    reasons: list[str] = []
    warnings: list[str] = []

    if action not in INTERVENTIONS:
        return {"accepted": False, "rejection_reasons": [f"unknown action '{action}' (valid: {sorted(INTERVENTIONS)})"], "warnings": []}

    hyp = dict(hypothesis or {})
    for k, v in hyp.items():
        if k not in HYPOTHESIS_KEYS:
            reasons.append(f"unknown hypothesis key '{k}'")
            continue
        lo, hi = HYPOTHESIS_KEYS[k]
        if float(v) < lo or float(v) > hi:
            reasons.append(f"hypothesis '{k}'={v} outside valid range [{lo}, {hi}]")
    if reasons:
        return {"accepted": False, "rejection_reasons": reasons, "warnings": warnings}

    assets_df = pd.read_csv(data_path / "assets.csv")
    if asset_id not in set(assets_df["asset_id"]):
        return {"accepted": False, "rejection_reasons": [f"unknown asset_id '{asset_id}'"], "warnings": warnings}
    arow = assets_df[assets_df["asset_id"] == asset_id].iloc[0]

    t_req = pd.to_datetime(time_utc, utc=True)
    w_start = pd.to_datetime(arow["window_start_utc"], utc=True)
    w_end = pd.to_datetime(arow["window_end_utc"], utc=True)
    if not (w_start <= t_req <= w_end):
        return {
            "accepted": False,
            "rejection_reasons": [f"time_utc {time_utc} outside asset window [{arow['window_start_utc']}, {arow['window_end_utc']}]"],
            "warnings": warnings,
        }

    # Build asset config from public fleet or assets.csv row
    cfg_path = root_path / "configs" / "fleet_public.yaml"
    cfg, fleet_list = load_fleet(str(cfg_path))
    match = [a for a in fleet_list if a["asset_id"] == asset_id]
    if match:
        acfg = match[0]
    else:
        prof = arow["profile"]
        acfg = dict(cfg["profiles"][prof])
        acfg.update(
            {
                "asset_id": asset_id,
                "profile": prof,
                "control": arow["control"],
                "q_bep_m3h": float(arow["pump_rated_flow_m3_h"]),
                "h_bep_m": float(arow["pump_rated_head_m"]),
                "eta_bep": float(arow["pump_bep_efficiency"]),
                "npshr_bep_m": float(arow["npsh3_at_bep_m"]),
                "q_design_m3h": float(arow["design_flow_m3_h"]),
                "h_design_m": float(arow["design_head_m"]),
                "system_static_m": float(arow["static_head_m"]),
                "motor_kw": float(arow["motor_rated_kw"]),
                "rated_rpm": float(arow["motor_rated_rpm"]),
                "voltage_v": float(arow["supply_voltage_v"]),
                "supply_hz": float(arow["supply_frequency_hz"]),
                "motor_eff": 0.945,
                "motor_pf": 0.86,
            }
        )
        if pd.notna(arow.get("min_speed_ratio")):
            acfg["r_min"] = float(arow["min_speed_ratio"])
        acfg["limits"] = default_limits(acfg)

    # Load operating context at timestamp
    ctx_path = data_path / "operating_context.csv"
    ctx_row = {"duty_demand_pct": 85.0, "supply_water_temp_c": 20.0, "ambient_temp_c": 28.0, "suction_static_head_m": np.nan}
    if ctx_path.exists():
        cdf = pd.read_csv(ctx_path)
        cdf = cdf[cdf["asset_id"] == asset_id].copy()
        if len(cdf):
            cdf["_t"] = pd.to_datetime(cdf["timestamp_utc"], utc=True)
            idx_closest = (cdf["_t"] - t_req).abs().idxmin()
            ctx_row = cdf.loc[idx_closest].to_dict()

    base_state = {
        "clog": float(hyp.get("clog", 0.05)),
        "bearing_wear": float(hyp.get("bearing_wear", 0.05)),
        "restriction": float(hyp.get("restriction", 1.0)),
        "misalignment": float(hyp.get("misalignment", 0.02)),
        "unbalance": float(hyp.get("unbalance", 0.0)),
        "impeller_wear": float(hyp.get("impeller_wear", 0.0)),
        "air_entrainment": float(hyp.get("air_entrainment", 0.0)),
    }
    prop_state = dict(base_state)
    speed_ovr = None
    valve_ovr = None

    if action == "clean_strainer":
        prop_state["clog"] = 0.05
    elif action == "replace_bearing":
        prop_state["bearing_wear"] = 0.03
    elif action == "realign":
        prop_state["misalignment"] = 0.02
    elif action == "speed_setpoint":
        if not str(acfg["control"]).startswith("VFD"):
            reasons.append(f"{asset_id} is a fixed-speed ({acfg['control']}) pump; speed_setpoint cannot be changed")
        elif value is None:
            reasons.append("speed_setpoint requires a numerical value")
        else:
            r_min = float(acfg.get("r_min", 0.50))
            r_max = float(acfg.get("r_max", 1.05))
            if float(value) < r_min or float(value) > r_max:
                reasons.append(f"speed_setpoint {value} outside drive limits [{r_min}, {r_max}]")
            speed_ovr = float(value)
    elif action == "valve_position":
        if acfg["control"] not in ("FLOW_VALVE", "VFD_PRESSURE"):
            reasons.append(f"{asset_id} has no remotely actuated control valve ({acfg['control']})")
        elif value is None or float(value) < 15.0 or float(value) > 100.0:
            reasons.append(f"valve_position {value} outside valid range [15, 100] %")
        else:
            valve_ovr = float(value)

    baseline = _eval_point(acfg, ctx_row, base_state)
    proposed = _eval_point(acfg, ctx_row, prop_state, speed_override=speed_ovr, valve_override=valve_ovr)

    # Feasibility checks on proposed operating point
    if speed_ovr is not None and abs(speed_ovr - baseline["speed_ratio"]) > 0.20:
        reasons.append(f"speed step {abs(speed_ovr - baseline['speed_ratio']):.2f} exceeds safe single-step limit (0.20)")
    if proposed["duty_satisfaction"] < 0.92 and proposed["duty_satisfaction"] < baseline["duty_satisfaction"] - 0.02:
        reasons.append(f"insufficient cooling/process flow: duty_satisfaction {proposed['duty_satisfaction']:.2f} < 0.92")
    if proposed["npsh_margin"] < 1.15 and proposed["npsh_margin"] < baseline["npsh_margin"]:
        reasons.append(f"cavitation risk: proposed npsh_margin {proposed['npsh_margin']:.2f} < 1.15")
    if proposed["motor_load"] > 1.05:
        reasons.append(f"motor overload: proposed motor_load {proposed['motor_load']:.2f} > 1.05")
    if proposed["vib_rms_mm_s"] > float(acfg["limits"]["vib_alarm_mm_s"]):
        warnings.append(f"proposed vibration {proposed['vib_rms_mm_s']:.2f} mm/s exceeds alarm limit {acfg['limits']['vib_alarm_mm_s']} mm/s")
    if proposed["npsh_margin"] < 1.30:
        warnings.append(f"tight NPSH margin ({proposed['npsh_margin']:.2f} < 1.30)")

    iv_spec = INTERVENTIONS[action]
    iv_cost = round(iv_spec["labour_h"] * LABOUR_EUR_H + iv_spec["parts_eur"], 2)

    return {
        "synthetic_simulation": True,
        "asset_id": asset_id,
        "time_utc": time_utc,
        "action": action,
        "value": value,
        "hypothesis": hyp,
        "accepted": len(reasons) == 0,
        "rejection_reasons": reasons,
        "warnings": warnings,
        "baseline": baseline,
        "proposed": proposed,
        "intervention": {
            "action": action,
            "stop_duration_h": iv_spec["stop_duration_h"],
            "labour_h": iv_spec["labour_h"],
            "parts_eur": iv_spec["parts_eur"],
            "estimated_cost_eur": iv_cost,
            "assumptions": f"labour EUR {LABOUR_EUR_H}/h, electricity EUR {ELECTRICITY_EUR_KWH}/kWh (synthetic assumptions)",
        },
    }


def what_if_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrofit_sim what_if")
    p.add_argument("--data", required=True)
    p.add_argument("--asset", required=True)
    p.add_argument("--time", required=True)
    p.add_argument("--action", default="none")
    p.add_argument("--value", type=float, default=None)
    p.add_argument("--hypothesis", default="{}")
    a = p.parse_args(argv)
    hyp = json.loads(a.hypothesis) if isinstance(a.hypothesis, str) else a.hypothesis
    res = what_if(a.data, a.asset, a.time, a.action, a.value, hyp)
    print(json.dumps(res, indent=2))
    return 0

