"""Module 4: Digital Sandbox 'What-If' Counterfactual Simulator.

Allows human operators to test proposed maintenance interventions (cleaning strainer,
replacing bearings, adjusting speed setpoints) against a sealed physics-based digital twin
before taking action in the real world. Strictly read-only simulation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from retrofit_sim.sandbox import what_if, INTERVENTIONS, HYPOTHESIS_KEYS
    WHAT_IF_AVAILABLE = True
except Exception as e:
    WHAT_IF_AVAILABLE = False
    WHAT_IF_ERROR = str(e)

from .data_loader import load_assets, get_asset_info, get_data_dir, load_features_1min, load_predictions, load_events


def _ensure_valid_asset_timestamp(asset_id: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Ensure scrubber_time falls inside the selected asset's 7-day window; snap to an incident/alert if out of range."""
    df = load_features_1min(asset_id)
    if not df.empty:
        t_min = pd.to_datetime(df["timestamp_utc"].min(), utc=True)
        t_max = pd.to_datetime(df["timestamp_utc"].max(), utc=True)
    else:
        info = get_asset_info(asset_id)
        t_min = pd.to_datetime(info.get("window_start_utc", "2025-10-06T00:00:00Z"), utc=True)
        t_max = pd.to_datetime(info.get("window_end_utc", "2025-10-12T23:59:00Z"), utc=True)

    cur = pd.to_datetime(st.session_state.get("scrubber_time", t_min + (t_max - t_min) / 2), utc=True)
    if cur < t_min or cur > t_max:
        # Snap to the first AI alert or ground-truth event for this asset, or window midpoint
        preds = load_predictions(asset_id)
        events = load_events(asset_id)
        if not preds.empty:
            cand = pd.to_datetime(preds.iloc[0]["alert_time_utc"], utc=True)
            cur = min(max(cand, t_min), t_max)
        elif not events.empty:
            cand = pd.to_datetime(events.iloc[0]["onset_utc"], utc=True) + pd.Timedelta(hours=6)
            cur = min(max(cand, t_min), t_max)
        else:
            cur = t_min + (t_max - t_min) / 2
        st.session_state["scrubber_time"] = cur
    return cur, t_min, t_max


def render_what_if_view():
    """Render the Digital Sandbox 'What-If' Simulator."""
    st.markdown("### 🧪 Digital Sandbox 'What-If' Counterfactual Simulator")
    st.markdown(
        "Safely simulate prospective operational interventions and maintenance decisions against the "
        "first-principles digital twin. Evaluates feasibility, NPSH cavitation margins, motor load constraints, "
        "and energy economics before any physical work is dispatched."
    )

    if not WHAT_IF_AVAILABLE:
        st.error(f"Sandbox module failed to load: {WHAT_IF_ERROR}")
        return

    assets = load_assets()
    selected_asset = st.session_state.get("selected_asset", "PUMP-06")
    asset_list = list(assets["asset_id"].unique())
    asset_idx = asset_list.index(selected_asset) if selected_asset in asset_list else 0

    def _on_whatif_asset_change():
        new_a = st.session_state.get("whatif_asset_select")
        if new_a:
            st.session_state["_pending_asset"] = new_a
            st.session_state.pop("last_sim_res", None)
            st.session_state.pop("whatif_hypothesis", None)
            st.session_state.pop("whatif_action", None)

    col_a, col_t = st.columns([1.2, 2.3])
    with col_a:
        asset_id = st.selectbox(
            "Simulated Machine",
            asset_list,
            index=asset_idx,
            key="whatif_asset_select",
            on_change=_on_whatif_asset_change,
        )

    asset_row = get_asset_info(asset_id)
    scrubber_time, t_min, t_max = _ensure_valid_asset_timestamp(asset_id)

    with col_t:
        selected_dt = st.slider(
            f"Simulation Timestamp ({asset_id} Operating Window UTC)",
            min_value=t_min.to_pydatetime(),
            max_value=t_max.to_pydatetime(),
            value=scrubber_time.to_pydatetime(),
            format="YYYY-MM-DD HH:mm",
        )
        scrubber_time = pd.to_datetime(selected_dt, utc=True)
        st.session_state["scrubber_time"] = scrubber_time

    # Infer smart defaults from the pump's detected AI alerts if not explicitly preset
    pump_preds = load_predictions(asset_id)
    detected_faults = set(pump_preds["fault_type"].unique()) if not pump_preds.empty else set()
    auto_clog = 3.5 if ("SUCTION_STRAINER_CLOGGING" in detected_faults or "CAVITATION" in detected_faults) else 0.0
    auto_bw = 0.55 if "BEARING_WEAR" in detected_faults else 0.05
    auto_restr = 2.2 if "DISCHARGE_RESTRICTION" in detected_faults else 1.0
    auto_mis = 0.50 if ("MISALIGNMENT" in detected_faults or "IMBALANCE" in detected_faults) else 0.0
    if "SUCTION_STRAINER_CLOGGING" in detected_faults or "CAVITATION" in detected_faults:
        auto_action = "clean_strainer"
    elif "BEARING_WEAR" in detected_faults:
        auto_action = "replace_bearing"
    elif "MISALIGNMENT" in detected_faults or "IMBALANCE" in detected_faults:
        auto_action = "realign"
    elif "DISCHARGE_RESTRICTION" in detected_faults and str(asset_row["control"]) in ("FLOW_VALVE", "VFD_PRESSURE"):
        auto_action = "valve_position"
    else:
        auto_action = "clean_strainer"

    st.markdown("---")

    # Action & Hypothesis Input Section
    col_input1, col_input2 = st.columns([1.2, 1.2])

    with col_input1:
        with st.container(border=True):
            st.markdown("#### 🛠️ 1. Proposed Operator Intervention")
            action_options = list(INTERVENTIONS.keys())
            default_action = st.session_state.get("whatif_action", auto_action)
            act_idx = action_options.index(default_action) if default_action in action_options else 0

            action = st.selectbox("Intervention Action", action_options, index=act_idx)

            action_val = None
            if action == "speed_setpoint":
                if not asset_row["control"].startswith("VFD"):
                    st.warning(f"⚠️ {asset_id} is a Fixed-Speed Direct-On-Line (DOL) pump! Speed setpoint changes will be rejected.")
                action_val = st.slider("Target Speed Ratio", min_value=0.50, max_value=1.05, value=0.80, step=0.02)
            elif action == "valve_position":
                action_val = st.slider("Target Valve Position (%)", min_value=15, max_value=100, value=80, step=5)
            elif action == "clean_strainer":
                st.info("Cleans inlet filter basket. Restores suction differential pressure. Estimated labor: 3h, Parts: €50.")
            elif action == "replace_bearing":
                st.info("Replaces drive-end rolling bearing. Restores vibration baseline. Estimated labor: 12h, Parts: €900.")
            elif action == "realign":
                st.info("Laser realignment of pump and motor shafts. Estimated labor: 6h, Parts: €150.")

    with col_input2:
        with st.container(border=True):
            st.markdown("#### 🔬 2. Hypothesized Machine Degradation State")
            st.caption("Operator's diagnostic hypothesis of current physical degradation:")

            preset_hyp = st.session_state.get("whatif_hypothesis", {})

            clog_val = st.slider(
                "Suction Clogging Severity (0 = clean, 8 = blocked)",
                min_value=0.0, max_value=8.0,
                value=float(preset_hyp.get("clog", auto_clog)),
                step=0.25
            )
            bearing_wear_val = st.slider(
                "Bearing Damage (0 = new, 0.95 = severe spalling)",
                min_value=0.0, max_value=0.95,
                value=float(preset_hyp.get("bearing_wear", auto_bw)),
                step=0.05
            )
            restr_val = st.slider(
                "Discharge Restriction Factor (1 = open, 4 = throttled)",
                min_value=1.0, max_value=4.0,
                value=float(preset_hyp.get("restriction", auto_restr)),
                step=0.20
            )
            misalign_val = st.slider(
                "Shaft Misalignment (0 = aligned, 1.0 = heavy)",
                min_value=0.0, max_value=1.0,
                value=float(preset_hyp.get("misalignment", auto_mis)),
                step=0.05
            )

    hypothesis = {
        "clog": clog_val,
        "bearing_wear": bearing_wear_val,
        "restriction": restr_val,
        "misalignment": misalign_val,
    }

    # Run Simulation Button
    if st.button("🚀 Run Digital Twin Simulation", type="primary", use_container_width=True):
        data_dir = str(get_data_dir())
        time_iso = scrubber_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        with st.spinner("Simulating multi-physics hydrodynamic state..."):
            try:
                sim_res = what_if(
                    data_dir=data_dir,
                    asset_id=asset_id,
                    time_utc=time_iso,
                    action=action,
                    value=action_val,
                    hypothesis=hypothesis,
                    root=str(ROOT_DIR)
                )
                st.session_state["last_sim_res"] = sim_res
            except Exception as ex:
                st.error(f"Simulation execution failed: {ex}")
                return

    sim_res = st.session_state.get("last_sim_res", None)
    if sim_res is None:
        st.info("Adjust intervention parameters and click 'Run Digital Twin Simulation' to view results.")
        return

    st.markdown("---")
    st.markdown("### 📊 Simulation Results & Counterfactual Feasibility")

    # Acceptance Banner
    accepted = sim_res.get("accepted", False)
    rejection_reasons = sim_res.get("rejection_reasons", [])
    warnings = sim_res.get("warnings", [])

    if accepted:
        st.success("✅ **INTERVENTION ACCEPTED: Physically Feasible & Safe within Operational Envelopes**")
    else:
        st.error(f"❌ **INTERVENTION REJECTED: Safety / Operational Limit Violation**")
        for r in rejection_reasons:
            st.markdown(f"• **Reason:** `{r}`")

    if warnings:
        for w in warnings:
            st.warning(f"⚠️ **Caution:** {w}")

    # Before vs After Comparison Table
    if "baseline" in sim_res and "proposed" in sim_res:
        base = sim_res["baseline"]
        prop = sim_res["proposed"]

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            b_q, p_q = base.get("flow_m3h", 0), prop.get("flow_m3h", 0)
            delta_q = p_q - b_q if b_q and p_q else 0
            st.metric("Flow Delivery (m³/h)", f"{p_q:.1f}" if p_q else "N/A", delta=f"{delta_q:+.1f} m³/h")

        with m_col2:
            b_npsh, p_npsh = base.get("npsh_margin", 0), prop.get("npsh_margin", 0)
            delta_npsh = p_npsh - b_npsh if b_npsh and p_npsh else 0
            st.metric("NPSH Cavitation Margin (m)", f"{p_npsh:.2f}" if p_npsh else "N/A", delta=f"{delta_npsh:+.2f} m")

        with m_col3:
            b_vib, p_vib = base.get("vib_rms_mm_s", 0), prop.get("vib_rms_mm_s", 0)
            delta_vib = p_vib - b_vib if b_vib and p_vib else 0
            st.metric("Vibration RMS (mm/s)", f"{p_vib:.2f}" if p_vib else "N/A", delta=f"{delta_vib:+.2f} mm/s", delta_color="inverse")

        with m_col4:
            b_cost, p_cost = base.get("energy_eur_per_day", 0), prop.get("energy_eur_per_day", 0)
            delta_cost = p_cost - b_cost if b_cost and p_cost else 0
            st.metric("Energy Cost (€/day)", f"€{p_cost:.1f}" if p_cost else "N/A", delta=f"€{delta_cost:+.1f}/day", delta_color="inverse")

        # Detailed Comparison Dataframe
        comparison_rows = [
            {"Parameter": "Motor Speed Ratio", "Baseline (Current)": f"{base.get('speed_ratio', 'N/A')}", "Proposed Action": f"{prop.get('speed_ratio', 'N/A')}", "Unit": "ratio"},
            {"Parameter": "Flow Delivery", "Baseline (Current)": f"{base.get('flow_m3h', 'N/A')}", "Proposed Action": f"{prop.get('flow_m3h', 'N/A')}", "Unit": "m³/h"},
            {"Parameter": "Pump Head", "Baseline (Current)": f"{base.get('head_m', 'N/A')}", "Proposed Action": f"{prop.get('head_m', 'N/A')}", "Unit": "m"},
            {"Parameter": "Suction Pressure", "Baseline (Current)": f"{base.get('suction_kpa_g', 'N/A')}", "Proposed Action": f"{prop.get('suction_kpa_g', 'N/A')}", "Unit": "kPa(g)"},
            {"Parameter": "Discharge Pressure", "Baseline (Current)": f"{base.get('discharge_kpa_g', 'N/A')}", "Proposed Action": f"{prop.get('discharge_kpa_g', 'N/A')}", "Unit": "kPa(g)"},
            {"Parameter": "NPSH Margin", "Baseline (Current)": f"{base.get('npsh_margin', 'N/A')}", "Proposed Action": f"{prop.get('npsh_margin', 'N/A')}", "Unit": "m (>1.3 safe)"},
            {"Parameter": "Motor Input Power", "Baseline (Current)": f"{base.get('input_kw', 'N/A')}", "Proposed Action": f"{prop.get('input_kw', 'N/A')}", "Unit": "kW"},
            {"Parameter": "Motor Load Factor", "Baseline (Current)": f"{base.get('motor_load', 'N/A')}", "Proposed Action": f"{prop.get('motor_load', 'N/A')}", "Unit": "ratio (<=1.0)"},
            {"Parameter": "Bearing Temperature", "Baseline (Current)": f"{base.get('bearing_temp_c', 'N/A')}", "Proposed Action": f"{prop.get('bearing_temp_c', 'N/A')}", "Unit": "°C"},
            {"Parameter": "Vibration Velocity RMS", "Baseline (Current)": f"{base.get('vib_rms_mm_s', 'N/A')}", "Proposed Action": f"{prop.get('vib_rms_mm_s', 'N/A')}", "Unit": "mm/s"},
            {"Parameter": "Daily Energy Cost", "Baseline (Current)": f"€{base.get('energy_eur_per_day', 'N/A')}", "Proposed Action": f"€{prop.get('energy_eur_per_day', 'N/A')}", "Unit": "EUR/day"},
        ]
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    # Intervention Cost Estimates
    if "intervention" in sim_res:
        interv = sim_res["intervention"]
        with st.container(border=True):
            st.markdown("#### 💶 Estimated Work-Order Economics")
            c_e1, c_e2, c_e3 = st.columns(3)
            with c_e1:
                st.metric("Estimated Stop Duration", f"{interv.get('stop_duration_h', 0):.1f} hours")
            with c_e2:
                st.metric("Estimated Labor Required", f"{interv.get('labour_h', 0):.1f} hours")
            with c_e3:
                st.metric("Total Direct Intervention Cost", f"€{interv.get('estimated_cost_eur', 0):.0f}")

