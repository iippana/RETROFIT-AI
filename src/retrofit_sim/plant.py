"""Quasi-steady plant simulator: controller, operating point, power, thermal lags, and vibration."""
from __future__ import annotations

import numpy as np

from . import context as C
from . import physics as ph


def _first_order_lag(x: np.ndarray, tau_s: float, x0: float | None = None) -> np.ndarray:
    if tau_s <= 1.0:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    out[0] = x[0] if x0 is None else float(x0)
    a = 1.0 / float(tau_s)
    b = 1.0 - a
    for i in range(1, len(x)):
        out[i] = b * out[i - 1] + a * x[i]
    return out


def _speed_trajectory(r_target: np.ndarray, running: np.ndarray, mode: np.ndarray, acfg: dict) -> np.ndarray:
    n = len(r_target)
    r = np.zeros(n, dtype=float)
    tau_run = float(acfg.get("speed_tau_s", 15.0)) if acfg["control"].startswith("VFD") else float(acfg.get("startup_ramp_s", 5.0)) / 2.5
    tau_start = max(float(acfg.get("startup_ramp_s", 8.0)) / 2.5, 1.5)
    tau_stop = max(float(acfg.get("coastdown_tau_s", 7.0)), 1.5)
    r[0] = float(r_target[0]) if running[0] else 0.0
    for i in range(1, n):
        if not running[i]:
            tau = tau_stop
            target = 0.0
        elif mode[i] == "STARTUP" or r[i - 1] < 0.25:
            tau = tau_start
            target = r_target[i]
        else:
            tau = tau_run
            target = r_target[i]
        a = 1.0 / max(tau, 1.0)
        r[i] = (1.0 - a) * r[i - 1] + a * target
        if not running[i] and r[i] < 0.005:
            r[i] = 0.0
    return r


def simulate_truth(acfg: dict, ctx, lat, rng: np.random.Generator) -> dict:
    """Simulate 1 Hz true plant physical states over the asset window."""
    n = len(ctx.t)
    q_bep = float(acfg["q_bep_m3h"]) / 3600.0
    q_des = float(acfg["q_design_m3h"]) / 3600.0
    h_bep = float(acfg["h_bep_m"])
    h_des = float(acfg["h_design_m"])
    h0 = float(acfg.get("h0_shutoff_ratio", 1.20))
    h_static = float(acfg.get("system_static_m", 3.0))

    suction_loss_des = float(acfg.get("suction_loss_m_at_design", 0.7))
    K_s0 = suction_loss_des / max(q_des ** 2, 1e-9)
    s_share = float(acfg.get("strainer_share", 0.6))
    K_suction = K_s0 * (1.0 + s_share * (((1.0 + lat.clog) ** 2) - 1.0))

    K_des = max((h_des - h_static) / max(q_des ** 2, 1e-9), 10.0)
    x_des = float(acfg.get("valve_design_position", 0.85))
    v_share = float(acfg.get("valve_share_of_K", 0.20))
    K_v_des = v_share * K_des
    K_v_base = K_v_des / (50.0 ** (2.0 * (1.0 - x_des)))
    K_pipe0 = max(K_des - K_v_des - K_s0, 0.1 * K_des)
    K_pipe = K_pipe0 * lat.restriction

    ctl = acfg["control"]
    r_min = float(acfg.get("r_min", 0.50))
    r_max = float(acfg.get("r_max", 1.0))

    # Control law
    if ctl == "VFD_TEMPERATURE":
        T_set = float(acfg.get("controlled_temp_setpoint_c", 36.0))
        T_des = float(acfg.get("design_supply_temp_c", 25.0))
        q_frac = np.clip(
            ctx.demand * (T_set - T_des) / np.maximum(T_set - ctx.supply_temp_c, 2.0),
            float(acfg.get("min_flow_frac", 0.40)),
            1.15,
        )
        Q_req = np.where(ctx.running, q_frac * q_des, 0.0)
        valve_pos = np.full(n, x_des)
        K_valve = K_v_base * (50.0 ** (2.0 * (1.0 - valve_pos)))
        K_tot_est = K_pipe + K_valve + K_suction
        r_req = np.sqrt(
            np.maximum(0.0, ((h_static + K_tot_est * (Q_req ** 2)) / h_bep + (h0 - 1.0) * ((Q_req / q_bep) ** 2)) / h0)
        )
        r_target = np.where(ctx.running, np.clip(r_req, r_min, r_max), 0.0)
    elif ctl == "VFD_PRESSURE":
        valve_pos_raw = np.where(ctx.running, np.clip(0.35 + 0.52 * ctx.demand, 0.25, 0.98), 0.0)
        valve_pos = _first_order_lag(valve_pos_raw, 10.0, valve_pos_raw[0])
        K_valve = K_v_base * (50.0 ** (2.0 * (1.0 - np.clip(valve_pos, 0.1, 1.0))))
        dp_set_m = float(acfg.get("dp_setpoint_m", 0.85 * h_des))
        Q_req = np.where(ctx.running, np.clip(ctx.demand, 0.35, 1.15) * q_des, 0.0)
        Q_dp = np.sqrt(np.maximum(dp_set_m - h_static, 0.5) / np.maximum(K_pipe + K_valve, 1e-6))
        Q_ctrl = np.maximum(Q_dp, 0.96 * Q_req)
        H_need = h_static + (K_pipe + K_valve + K_suction) * (Q_ctrl ** 2)
        r_req = np.sqrt(np.maximum(0.0, (H_need / h_bep + (h0 - 1.0) * ((Q_ctrl / q_bep) ** 2)) / h0))
        r_target = np.where(ctx.running, np.clip(r_req, r_min, r_max), 0.0)
    elif ctl == "FLOW_VALVE":
        Q_req = np.where(ctx.running, np.clip(ctx.demand, 0.3, 1.15) * q_des, 0.0)
        r_target = np.where(ctx.running, 1.0, 0.0)
        H_avail = h_bep * (h0 - (h0 - 1.0) * ((Q_req / q_bep) ** 2))
        K_v_req = np.maximum(0.0, (H_avail - h_static) / np.maximum(Q_req ** 2, 1e-9) - K_pipe - K_suction)
        x_req = 1.0 - 0.5 * np.log(np.maximum(K_v_req / max(K_v_base, 1e-9), 1.0)) / np.log(50.0)
        x_req = np.where(ctx.running, np.clip(x_req, 0.15, 1.0), 0.0)
        valve_pos = _first_order_lag(x_req, float(acfg.get("valve_tau_s", 6.0)), x_req[0])
        K_valve = K_v_base * (50.0 ** (2.0 * (1.0 - np.clip(valve_pos, 0.1, 1.0))))
    else:  # FIXED_SPEED
        Q_req = np.full(n, np.nan)
        r_target = np.where(ctx.running, 1.0, 0.0)
        valve_pos = np.full(n, x_des)
        K_valve = K_v_base * (50.0 ** (2.0 * (1.0 - valve_pos)))

    r = _speed_trajectory(r_target, ctx.running, ctx.mode, acfg)
    K_total = K_pipe + K_valve + K_suction

    # Water properties and suction head
    rho = ph.water_density(ctx.supply_temp_c, float(acfg.get("salinity_psu", 0.0)))
    p_v = ph.vapour_pressure_pa(ctx.supply_temp_c)
    p_atm = 101325.0
    p_loop = float(acfg.get("loop_pressurisation_kpa", 0.0)) * 1000.0
    if acfg["profile"].startswith("SHIP"):
        roll_period = float(acfg.get("roll_period_s", 11.0))
        roll_m = 0.25 * ctx.sea_state * np.sin(2.0 * np.pi * ctx.t / roll_period)
    else:
        roll_m = 0.0
    z_s = ctx.suction_static_m + roll_m

    # Fixed-point bisection for operating flow Q with cavitation head drop
    f_non_cav = (1.0 - lat.impeller_wear) * (1.0 - 2.5 * lat.air_entrainment)
    q_hi = ph.solve_operating_point(r, h_bep, q_bep, h0, h_static, K_total, f_non_cav)
    q_lo = np.zeros(n, dtype=float)
    for _ in range(12):
        q_mid = 0.5 * (q_lo + q_hi)
        p_tot_m = np.maximum(p_v + 150.0, p_atm + p_loop + rho * ph.G_ACC * (z_s - K_suction * (q_mid ** 2)))
        npsha_m = (p_tot_m - p_v) / (rho * ph.G_ACC)
        npshr_m = ph.npsh3(q_mid, r, q_bep, float(acfg["npshr_bep_m"]))
        f_cav_m, _ = ph.cavitation_head_factor(npsha_m, npshr_m)
        q_eval = ph.solve_operating_point(r, h_bep, q_bep, h0, h_static, K_total, f_non_cav * f_cav_m)
        hi_mask = q_eval >= q_mid
        q_lo = np.where(hi_mask, q_mid, q_lo)
        q_hi = np.where(hi_mask, q_hi, q_mid)

    Q = 0.5 * (q_lo + q_hi)
    Q = np.where(r > 0.02, Q, 0.0)
    p_tot = np.maximum(p_v + 150.0, p_atm + p_loop + rho * ph.G_ACC * (z_s - K_suction * (Q ** 2)))
    npsha = (p_tot - p_v) / (rho * ph.G_ACC)
    npshr = ph.npsh3(Q, r, q_bep, float(acfg["npshr_bep_m"]))
    f_cav, cav_int = ph.cavitation_head_factor(npsha, npshr)
    cav_int = np.where(r > 0.2, cav_int, 0.0)
    npsh_margin = np.where(r > 0.2, npsha / np.maximum(npshr, 0.1), 5.0)

    f_head = f_non_cav * f_cav
    H = ph.pump_head(Q, r, h_bep, q_bep, h0, f_head)
    q_rel = np.where(r > 0.05, Q / np.maximum(r * q_bep, 1e-9), 0.0)

    v_suc = 2.5 * (Q / max(q_bep, 1e-9))
    p_suc_abs = np.maximum(p_v, p_tot - 0.5 * rho * (v_suc ** 2) * (r > 0.02))
    suction_kpa_g = (p_suc_abs - p_atm) / 1000.0
    p_dis_abs = p_tot + rho * ph.G_ACC * H
    discharge_kpa_g = (p_dis_abs - p_atm) / 1000.0

    # Power and electrical
    mech_drag = 1.0 + 0.04 * (lat.bearing_wear ** 2) + 0.02 * lat.misalignment
    P_sh_w = ph.shaft_power(Q, H, r, q_bep, h_bep, float(acfg["eta_bep"]), h0, rho, f_head, mech_drag)
    P_hyd_w = rho * ph.G_ACC * Q * H
    is_vfd = ctl.startswith("VFD")
    P_in_w, eta_m, pf, I_a, lam = ph.motor_electrical(
        P_sh_w,
        float(acfg["motor_kw"]) * 1000.0,
        float(acfg["motor_eff"]),
        float(acfg["motor_pf"]),
        float(acfg["voltage_v"]),
        ctx.supply_voltage_pu,
        float(acfg.get("const_loss_share", 0.35)),
        is_vfd=is_vfd,
        rated_vfd_eff=float(acfg.get("vfd_efficiency", 0.97)),
    )
    # Contactor open during coast-down (STOPPING / not ctx.running)
    P_in_w = np.where(ctx.running, P_in_w, 0.0)
    I_a = np.where(ctx.running, I_a, 0.0)
    lam = np.where(ctx.running, lam, 0.0)

    # DOL startup current inrush during first seconds of STARTUP
    if not is_vfd:
        inrush_mult = float(acfg.get("inrush_multiple", 6.5))
        i_rated = float(acfg["motor_kw"]) * 1000.0 / (float(acfg["motor_eff"]) * np.sqrt(3.0) * float(acfg["voltage_v"]) * float(acfg["motor_pf"]))
        in_start = (ctx.mode == "STARTUP") & (r < 0.85)
        I_a = np.where(in_start, np.minimum(580.0, I_a + (inrush_mult - 1.0) * i_rated * np.maximum(0.0, 1.0 - r)), I_a)

    slip = float(acfg.get("rated_slip", 0.015))
    rpm = np.where(r > 0.005, r * float(acfg["rated_rpm"]) * (1.0 - slip * np.clip(lam, 0.0, 1.5)) / (1.0 - slip), 0.0)

    # Thermal model
    dT_fluid = np.where(
        Q > 0.05 * q_bep,
        np.clip((P_sh_w - P_hyd_w) * 0.9 / np.maximum(rho * 4184.0 * Q, 100.0), 0.0, 8.0),
        0.0,
    )
    fluid_temp_c = _first_order_lag(ctx.supply_temp_c + dT_fluid, 60.0, float(ctx.supply_temp_c[0]))

    k_b = float(acfg.get("bearing_rise_k", 24.0))
    tau_b = float(acfg.get("bearing_tau_s", 1500.0))
    T_brg_ss = (
        0.6 * ctx.ambient_c
        + 0.4 * fluid_temp_c
        + k_b * (r ** 1.3) * (0.75 + 0.25 * lam)
        + 40.0 * (np.maximum(lat.bearing_wear - 0.4, 0.0) ** 1.6) * (0.4 + 0.6 * r)
        + 6.0 * lat.misalignment * r
    )
    bearing_temp_c = _first_order_lag(T_brg_ss, tau_b, float(T_brg_ss[0]))

    k_w = float(acfg.get("winding_rise_k", 72.0))
    tau_w = float(acfg.get("winding_tau_s", 2400.0))
    T_wnd_ss = ctx.ambient_c + np.where(
        ctx.running,
        k_w * (0.25 + 0.75 * (lam ** 2)) / (np.maximum(ctx.supply_voltage_pu, 0.7) ** 2) + 3.0 * lat.misalignment * r,
        0.0,
    )
    winding_temp_c = _first_order_lag(T_wnd_ss, tau_w, float(T_wnd_ss[0]))

    # Vibration model
    v_s = float(acfg.get("vib_base_mm_s", 1.2)) * (0.4 + 0.6 * (r ** 1.5))
    v_u = (float(acfg.get("unbalance_base_mm_s", 0.6)) + 4.0 * lat.unbalance) * (r ** 2)
    v_m = 4.5 * lat.misalignment * (0.5 + 0.5 * lam) * r
    recirc = np.where(q_rel < 0.5, 3.0, 1.0)
    v_h = float(acfg.get("hydraulic_vib_mm_s", 1.0)) * (r ** 2) * (np.abs(q_rel - 1.0) ** 1.5) * recirc
    v_c = float(acfg.get("cavitation_vib_mm_s", 3.0)) * cav_int * (r ** 2) + 3.0 * lat.air_entrainment * r
    v_b = 14.0 * (np.maximum(lat.bearing_wear - 0.4, 0.0) ** 1.5) * (0.3 + 0.7 * r)
    v_rss = np.sqrt(v_s ** 2 + v_u ** 2 + v_m ** 2 + v_h ** 2 + v_c ** 2 + v_b ** 2)

    ou_v = np.exp(C._smooth_noise(rng, n, 0.04, 180.0))
    res_bump = np.where((ctx.mode == "STARTUP") & (r > 0.35) & (r < 0.55), 1.30, 1.0)
    vib_rms_mm_s = np.where(r > 0.02, v_rss * ou_v * res_bump, 0.04)

    a_base = float(acfg.get("accel_base_g", 0.16)) * (0.3 + 0.7 * (r ** 1.8))
    a_brg = 2.2 * (np.maximum(lat.bearing_wear - 0.12, 0.0) ** 1.3) * np.maximum(r, 0.3)
    a_cav = 1.8 * cav_int * (r ** 1.5) + 4.5 * lat.air_entrainment * r
    a_mech = 0.06 * (v_u + v_m)
    vib_accel_rms_g = np.where(r > 0.02, np.sqrt(a_base ** 2 + a_brg ** 2 + a_cav ** 2 + a_mech ** 2) * ou_v, 0.01)

    w_loc = np.maximum(lat.bearing_wear - 0.08, 0.0) / 0.18
    cf_brg = 2.8 * w_loc * np.exp(1.0 - w_loc)
    cf_noise = C._smooth_noise(rng, n, 0.06, 180.0)
    vib_crest = np.where(r > 0.02, np.clip(3.2 + cf_brg + 0.8 * cav_int + cf_noise, 1.5, 18.0), 3.0)

    valve_pct = np.where(
        ctl in ("FLOW_VALVE", "VFD_PRESSURE"),
        np.clip(valve_pos * 100.0, 0.0, 100.0),
        np.full(n, x_des * 100.0),
    )

    return {
        "speed_ratio": r,
        "rpm": rpm,
        "flow_m3h": Q * 3600.0,
        "flow_required_m3h": Q_req * 3600.0 if not np.all(np.isnan(Q_req)) else np.full(n, np.nan),
        "head_m": H,
        "q_rel": q_rel,
        "suction_kpa_g": suction_kpa_g,
        "discharge_kpa_g": discharge_kpa_g,
        "npsha_m": npsha,
        "npshr_m": npshr,
        "npsh_margin": npsh_margin,
        "shaft_kw": P_sh_w / 1000.0,
        "input_kw": P_in_w / 1000.0,
        "hydraulic_kw": P_hyd_w / 1000.0,
        "current_a": I_a,
        "motor_load": lam,
        "fluid_temp_c": fluid_temp_c,
        "bearing_temp_c": bearing_temp_c,
        "winding_temp_c": winding_temp_c,
        "ambient_temp_c": ctx.ambient_c,
        "vib_rms_mm_s": vib_rms_mm_s,
        "vib_accel_rms_g": vib_accel_rms_g,
        "vib_crest": vib_crest,
        "valve_position_pct": valve_pct,
        "cavitation_intensity": cav_int,
        "K_total": K_total,
        "K_suction": K_suction,
        "bearing_wear": lat.bearing_wear,
        "unbalance": lat.unbalance,
        "misalignment": lat.misalignment,
        "air_entrainment": lat.air_entrainment,
    }

