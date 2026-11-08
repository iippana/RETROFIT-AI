"""Module 2: Interactive Incident Replay & Multi-Signal Telemetry Viewer.

Features synchronized Plotly time-series charts across hydraulics, electrical drive,
and mechanical vibration channels, with scrubbable timeline, event overlays, and incident bookmarks.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from .data_loader import (
    load_assets,
    load_events,
    load_alarms,
    load_predictions,
    load_features_1min,
    load_vibration_snapshots,
    get_asset_info,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Famous Incidents Bookmarks
INCIDENT_PRESETS = [
    {
        "label": "🔴 PUMP-06: Strainer Clog ➔ Severe Cavitation",
        "asset_id": "PUMP-06",
        "start_iso": "2025-10-08T12:00:00Z",
        "end_iso": "2025-10-10T23:59:00Z",
        "scrub_iso": "2025-10-09T21:45:00Z",
        "desc": "Chemical Plant D: Strainer slowly blocks; operator does not notice until suction pressure plunges and pump cavitates violently.",
    },
    {
        "label": "🟡 PUMP-04: Bearing Wear Spike ➔ Restriction",
        "asset_id": "PUMP-04",
        "start_iso": "2025-06-16T00:00:00Z",
        "end_iso": "2025-06-22T12:00:00Z",
        "scrub_iso": "2025-06-18T12:00:00Z",
        "desc": "Food Plant C: Stage-2 bearing wear shows early in vibration crest factor days before temperature or velocity rises.",
    },
    {
        "label": "🔵 PUMP-01: Seawater Clog + Gateway Comms Loss",
        "asset_id": "PUMP-01",
        "start_iso": "2025-08-11T12:00:00Z",
        "end_iso": "2025-08-16T12:00:00Z",
        "scrub_iso": "2025-08-14T13:30:00Z",
        "desc": "Vessel A (RoPax): Suction strainer clogged while underway; wireless gateway communication loss on Aug 14.",
    },
    {
        "label": "⚪ PUMP-03: T0 Legacy Severe Valve Restriction",
        "asset_id": "PUMP-03",
        "start_iso": "2026-03-02T12:00:00Z",
        "end_iso": "2026-03-07T12:00:00Z",
        "scrub_iso": "2026-03-04T00:00:00Z",
        "desc": "Vessel A (RoPax): Legacy pump with no vibration sensor; discharge valve restriction causes head rise and current drop.",
    },
    {
        "label": "🧪 PUMP-08: Validation Clogging & Sensor Spikes",
        "asset_id": "PUMP-08",
        "start_iso": "2025-09-08T12:00:00Z",
        "end_iso": "2025-09-14T12:00:00Z",
        "scrub_iso": "2025-09-10T15:50:00Z",
        "desc": "Vessel B (Validation): Bearing damage + suction strainer clogging caught early before SCADA alarm.",
    },
    {
        "label": "🧪 PUMP-09: Validation Misalignment & Restriction",
        "asset_id": "PUMP-09",
        "start_iso": "2026-02-09T12:00:00Z",
        "end_iso": "2026-02-15T12:00:00Z",
        "scrub_iso": "2026-02-10T12:00:00Z",
        "desc": "Plant C (Validation): Shaft misalignment increases vibration RMS velocity; followed by discharge restriction.",
    },
]


def render_telemetry_view():
    """Render the Interactive Incident Replay & Telemetry View."""
    if not PLOTLY_AVAILABLE:
        st.error(
            "Plotly is not installed yet! Run: `pip install plotly` in your terminal, then reload this page."
        )
        return

    assets = load_assets()
    events = load_events()
    alarms = load_alarms()
    predictions = load_predictions()

    st.markdown("### 📈 Interactive Incident Replay & Multi-Signal Telemetry")
    st.markdown(
        "Synchronized multi-channel telemetry viewer with regime-normal baseline context, "
        "vibration tracking, SCADA alarms, and AI anomaly alerts."
    )

    # 1. Incident Bookmarks Quick-Jump Bar
    st.markdown("##### 📌 Incident Bookmarks (Click to load incident):")
    p_cols = st.columns(len(INCIDENT_PRESETS))
    for i, preset in enumerate(INCIDENT_PRESETS):
        with p_cols[i]:
            if st.button(preset["label"].split(":")[0], key=f"preset_btn_{i}", help=preset["desc"], use_container_width=True):
                st.session_state["_pending_asset"] = preset["asset_id"]
                st.session_state["replay_start"] = pd.to_datetime(preset["start_iso"], utc=True)
                st.session_state["replay_end"] = pd.to_datetime(preset["end_iso"], utc=True)
                st.session_state["scrubber_time"] = pd.to_datetime(preset["scrub_iso"], utc=True)
                st.session_state.pop("time_range_slider", None)
                st.session_state.pop("scrubber_slider", None)
                st.rerun()

    # 2. Asset Selector Header
    selected_asset = st.session_state.get("selected_asset", "PUMP-06")
    asset_list = list(assets["asset_id"].unique())
    asset_idx = asset_list.index(selected_asset) if selected_asset in asset_list else 0

    def _on_telemetry_asset_change():
        new_a = st.session_state.get("_telemetry_asset_select")
        if new_a and new_a != st.session_state.get("selected_asset"):
            st.session_state["selected_asset"] = new_a
            st.session_state.pop("replay_start", None)
            st.session_state.pop("replay_end", None)
            st.session_state.pop("time_range_slider", None)
            st.session_state.pop("scrubber_slider", None)

    c_asset, c_info, c_tier = st.columns([1.5, 3, 1.5])
    with c_asset:
        selected_asset = st.selectbox(
            "Active Monitored Asset",
            asset_list,
            index=asset_idx,
            key="_telemetry_asset_select",
            on_change=_on_telemetry_asset_change,
        )

    asset_row = get_asset_info(selected_asset)
    if asset_row is None:
        st.warning("Asset not found.")
        return

    with c_info:
        st.markdown(f"**{asset_row['system']}** ({asset_row['site']})")
        st.caption(f"Profile: `{asset_row['profile']}` · Control: `{asset_row['control']}` · Rated: {asset_row['pump_rated_flow_m3_h']} m³/h @ {asset_row['pump_rated_head_m']} m head")

    with c_tier:
        tier_str = asset_row["instrumentation_tier"]
        tier_color = "#004085" if "T1" in tier_str else "#155724" if "T2" in tier_str else "#856404"
        st.markdown(
            f"<div style='text-align: right; padding-top: 10px;'>"
            f"<span style='background:#f0f4f8; color:{tier_color}; border:1px solid {tier_color}; padding:6px 12px; border-radius:15px; font-weight:bold; font-size:13px;'>{tier_str}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # 3. Load Telemetry Data
    with st.spinner(f"Loading telemetry for {selected_asset}..."):
        df = load_features_1min(selected_asset)

    if df.empty:
        st.warning(f"No telemetry data available for asset {selected_asset}.")
        return

    # Ensure timestamp sorting
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    min_time = pd.to_datetime(df["timestamp_utc"].min(), utc=True)
    max_time = pd.to_datetime(df["timestamp_utc"].max(), utc=True)

    # Determine window
    default_start = pd.to_datetime(st.session_state.get("replay_start", min_time), utc=True)
    default_end = pd.to_datetime(st.session_state.get("replay_end", max_time), utc=True)
    if default_start < min_time or default_start > max_time:
        default_start = min_time
    if default_end > max_time or default_end < min_time or default_end <= default_start:
        default_end = max_time

    # Scrubber controls
    s_col1, s_col2 = st.columns([3, 1.5])
    with s_col1:
        time_range = st.slider(
            "7-Day Replay Window Range",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(),
            value=(default_start.to_pydatetime(), default_end.to_pydatetime()),
            format="YYYY-MM-DD HH:mm",
            key="time_range_slider"
        )
        start_dt, end_dt = pd.to_datetime(time_range[0], utc=True), pd.to_datetime(time_range[1], utc=True)
        st.session_state["replay_start"] = start_dt
        st.session_state["replay_end"] = end_dt

    with s_col2:
        default_scrub = pd.to_datetime(st.session_state.get("scrubber_time", start_dt + (end_dt - start_dt) / 2), utc=True)
        if default_scrub < start_dt or default_scrub > end_dt:
            default_scrub = start_dt + (end_dt - start_dt) / 2

        scrub_dt = st.slider(
            "Inspection Scrubber Cursor (Time Point)",
            min_value=start_dt.to_pydatetime(),
            max_value=end_dt.to_pydatetime(),
            value=default_scrub.to_pydatetime(),
            format="MM-DD HH:mm",
            key="scrubber_slider"
        )
        scrubber_time = pd.to_datetime(scrub_dt, utc=True)
        st.session_state["scrubber_time"] = scrubber_time

    # Filter data to active window
    w_df = df[(df["timestamp_utc"] >= start_dt) & (df["timestamp_utc"] <= end_dt)].copy()
    if w_df.empty:
        st.warning("No data points in selected window range.")
        return

    # Real-time Telemetry Snapshot at Scrubber Time
    scrub_point = w_df.iloc[(w_df["timestamp_utc"] - scrubber_time).abs().argsort()[:1]]
    if not scrub_point.empty:
        pt = scrub_point.iloc[0]
        st.markdown(f"##### ⏱️ Scrubber Snapshot at `{scrubber_time.strftime('%Y-%m-%d %H:%M:%S UTC')}`:")
        kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
        with kpi_c1:
            st.metric("Operating Mode", pt.get("operating_mode", "UNKNOWN"))
        with kpi_c2:
            p_disc = pt.get("discharge_pressure_kpa__mean", pt.get("discharge_pressure_kpa", np.nan))
            st.metric("Discharge P", f"{p_disc:.1f} kPa" if pd.notna(p_disc) else "N/A")
        with kpi_c3:
            p_suct = pt.get("suction_pressure_kpa__mean", pt.get("suction_pressure_kpa", np.nan))
            st.metric("Suction P", f"{p_suct:.1f} kPa" if pd.notna(p_suct) else "N/A (T0)")
        with kpi_c4:
            speed = pt.get("motor_speed_rpm__mean", pt.get("motor_speed_rpm", np.nan))
            st.metric("Speed", f"{speed:.0f} RPM" if pd.notna(speed) else "Fixed (DOL)")
        with kpi_c5:
            vib_rms = pt.get("vibration_rms_mm_s__mean", pt.get("vibration_rms_mm_s", np.nan))
            alarm_lim = asset_row.get("vibration_alarm_mm_s", 6.4)
            delta_str = f"Limit: {alarm_lim} mm/s"
            st.metric("Vib RMS", f"{vib_rms:.2f} mm/s" if pd.notna(vib_rms) else "N/A (T0)", delta=delta_str, delta_color="off")
        with kpi_c6:
            crest = pt.get("vibration_crest_factor__mean", pt.get("vibration_crest_factor", np.nan))
            st.metric("Crest Factor", f"{crest:.2f}" if pd.notna(crest) else "N/A (T0)")

    # 4. Build Synchronized Plotly Subplots
    st.markdown(
        "<div style='display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 8px 0; font-size:12px;'>"
        "<span style='background:#b26a00; color:#fff; padding:3px 9px; border-radius:10px; border:1px solid #ffb74d;'>🟠 Ground-Truth Machine Fault (Row 1)</span>"
        "<span style='background:#01579b; color:#fff; padding:3px 9px; border-radius:10px; border:1px solid #4fc3f7;'>🔵 Ground-Truth Sensor Fault (Row 1)</span>"
        "<span style='background:#004d40; color:#e0f2f1; padding:3px 9px; border-radius:10px; border:1px solid #00e676;'>🟢 🧠 AI Diagnostic Alert (Row 2)</span>"
        "<span style='background:#880e4f; color:#fff; padding:3px 9px; border-radius:10px; border:1px solid #ff80ab;'>🔴 🚨 Plant SCADA Alarm (Row 3)</span>"
        "<span style='background:#ffd740; color:#111; padding:3px 9px; border-radius:10px; font-weight:bold;'>🟡 ◀ Active Inspection Cursor</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Hydraulics & Pressures</b> (Suction, Discharge, Flow)",
            "<b>2. Electrical Drive & Speed</b> (Speed RPM, Current A, Power kW)",
            "<b>3. Mechanical Condition & Vibration</b> (Velocity RMS, Crest Factor, Temp)"
        ),
        row_heights=[0.34, 0.33, 0.33]
    )

    t_series = w_df["timestamp_utc"]

    # Helper for channel extraction
    def get_chan(col_name: str):
        if f"{col_name}__mean" in w_df.columns:
            return w_df[f"{col_name}__mean"]
        if col_name in w_df.columns:
            return w_df[col_name]
        return None

    # Subplot 1: Hydraulics
    p_disc = get_chan("discharge_pressure_kpa")
    if p_disc is not None:
        fig.add_trace(go.Scatter(x=t_series, y=p_disc, name="Discharge P (kPa)", line=dict(color="#29b6f6", width=2)), row=1, col=1)
    p_suct = get_chan("suction_pressure_kpa")
    if p_suct is not None:
        fig.add_trace(go.Scatter(x=t_series, y=p_suct, name="Suction P (kPa)", line=dict(color="#00e5ff", width=2)), row=1, col=1)
    flow = get_chan("flow_m3_h")
    if flow is not None:
        fig.add_trace(go.Scatter(x=t_series, y=flow, name="Flow (m³/h)", line=dict(color="#66bb6a", dash="dash", width=1.6)), row=1, col=1)

    # Subplot 2: Electrical
    speed = get_chan("motor_speed_rpm")
    if speed is not None:
        fig.add_trace(go.Scatter(x=t_series, y=speed, name="Speed (RPM)", line=dict(color="#ffa726", width=2)), row=2, col=1)
    current = get_chan("motor_current_a")
    if current is not None:
        fig.add_trace(go.Scatter(x=t_series, y=current, name="Current (A)", line=dict(color="#ab47bc", width=1.6)), row=2, col=1)
    power = get_chan("motor_power_kw")
    if power is not None:
        fig.add_trace(go.Scatter(x=t_series, y=power, name="Power (kW)", line=dict(color="#a1887f", width=1.6)), row=2, col=1)

    # Subplot 3: Vibration & Mechanical
    vib_rms = get_chan("vibration_rms_mm_s")
    if vib_rms is not None:
        fig.add_trace(go.Scatter(x=t_series, y=vib_rms, name="Vib Velocity RMS (mm/s)", line=dict(color="#ff5252", width=2.4)), row=3, col=1)
        # Add ISO Alarm Threshold line with high-contrast badge
        alarm_val = float(asset_row.get("vibration_alarm_mm_s", 6.4))
        fig.add_hline(y=alarm_val, line=dict(color="#ff5252", dash="dash", width=1.5), row=3, col=1)
        fig.add_annotation(
            x=end_dt, y=alarm_val, xref="x3", yref="y3",
            text=f"ISO Limit ({alarm_val} mm/s)",
            showarrow=False, xanchor="right", yanchor="bottom", yshift=3,
            font=dict(color="#ffffff", size=10),
            bgcolor="rgba(183, 28, 28, 0.88)", bordercolor="#ff5252", borderwidth=1, borderpad=2,
        )
    crest = get_chan("vibration_crest_factor")
    if crest is not None:
        fig.add_trace(go.Scatter(x=t_series, y=crest, name="Crest Factor", line=dict(color="#f48fb1", width=1.6, dash="dot")), row=3, col=1)
    bearing_temp = get_chan("bearing_temp_c")
    if bearing_temp is not None:
        fig.add_trace(go.Scatter(x=t_series, y=bearing_temp, name="Bearing Temp (°C)", line=dict(color="#dce775", width=1.6)), row=3, col=1)

    # Overlays: Ground Truth Events in window (band across all rows, high-contrast badge ONLY on Row 1)
    asset_events = events[events["asset_id"] == selected_asset].sort_values("onset_utc") if not events.empty else pd.DataFrame()
    ev_idx = 0
    for _, ev in asset_events.iterrows():
        ev_start = ev["onset_utc"]
        ev_end = ev["end_utc"] if pd.notna(ev["end_utc"]) else end_dt
        if ev_end >= start_dt and ev_start <= end_dt:
            is_mach = ev["event_class"] == "MACHINE"
            fill_col = "rgba(255, 152, 0, 0.14)" if is_mach else "rgba(0, 229, 255, 0.14)"
            line_col = "#ffb74d" if is_mach else "#4fc3f7"
            bg_col = "rgba(178, 106, 0, 0.92)" if is_mach else "rgba(1, 87, 155, 0.92)"
            x0_clamp = max(ev_start, start_dt)
            x1_clamp = min(ev_end, end_dt)
            fig.add_vrect(
                x0=x0_clamp, x1=x1_clamp,
                fillcolor=fill_col, opacity=0.45, line_width=1, line_color=line_col,
                row="all", col=1
            )
            y_pos = 0.96 if (ev_idx % 2 == 0) else 0.78
            fig.add_annotation(
                x=x0_clamp, y=y_pos, xref="x", yref="y domain",
                text=f"<b>{ev['event_class']}: {ev['fault_type']}</b>",
                showarrow=False, xanchor="left", yanchor="top", xshift=4,
                font=dict(color="#ffffff", size=10),
                bgcolor=bg_col, bordercolor=line_col, borderwidth=1, borderpad=3,
            )
            ev_idx += 1

    # Overlays: AI Model Alerts in window (bright neon green line across all rows, staggered badge ONLY on Row 2)
    if not predictions.empty:
        asset_preds = predictions[predictions["asset_id"] == selected_asset].sort_values("alert_time_utc")
        w_preds = asset_preds[(asset_preds["alert_time_utc"] >= start_dt) & (asset_preds["alert_time_utc"] <= end_dt)]
        if not w_preds.empty:
            y_levels_ai = [0.96, 0.74, 0.52]
            for ai_idx, (_, alert) in enumerate(w_preds.iterrows()):
                al_t = alert["alert_time_utc"]
                fig.add_vline(
                    x=al_t,
                    line=dict(color="#00e676", width=1.8, dash="dashdot"),
                    row="all", col=1
                )
                y_ai = y_levels_ai[ai_idx % len(y_levels_ai)]
                conf_pct = int(float(alert.get("confidence", 0.9)) * 100)
                fig.add_annotation(
                    x=al_t, y=y_ai, xref="x2", yref="y2 domain",
                    text=f"<b>🧠 AI: {alert['fault_type']} ({conf_pct}%)</b>",
                    showarrow=False, xanchor="left", yanchor="top", xshift=4,
                    font=dict(color="#e0f2f1", size=10),
                    bgcolor="rgba(0, 77, 64, 0.94)", bordercolor="#00e676", borderwidth=1, borderpad=3,
                )

    # Overlays: SCADA Alarms in window (red dotted line across all rows, badge ONLY on Row 3)
    asset_alarms = alarms[alarms["asset_id"] == selected_asset].sort_values("raised_utc") if not alarms.empty else pd.DataFrame()
    scada_idx = 0
    for _, al in asset_alarms.iterrows():
        al_time = al["raised_utc"]
        if start_dt <= al_time <= end_dt:
            fig.add_vline(
                x=al_time, line=dict(color="#ff5252", width=1.5, dash="dot"),
                row="all", col=1
            )
            y_sc = 0.95 if (scada_idx % 2 == 0) else 0.74
            fig.add_annotation(
                x=al_time, y=y_sc, xref="x3", yref="y3 domain",
                text=f"<b>🚨 SCADA: {al['tag']}</b>",
                showarrow=False, xanchor="left", yanchor="top", xshift=4,
                font=dict(color="#ffffff", size=9.5),
                bgcolor="rgba(136, 14, 79, 0.92)", bordercolor="#ff80ab", borderwidth=1, borderpad=3,
            )
            scada_idx += 1

    # Scrubber Cursor Vertical Line (high-contrast gold/amber line + single badge at bottom of Row 1)
    fig.add_vline(
        x=scrubber_time,
        line=dict(color="#ffd740", width=2.5, dash="dash"),
        row="all", col=1
    )
    fig.add_annotation(
        x=scrubber_time, y=0.04, xref="x", yref="y domain",
        text="<b>◀ ACTIVE CURSOR</b>",
        showarrow=False, xanchor="left", yanchor="bottom", xshift=5,
        font=dict(color="#111111", size=10.5),
        bgcolor="#ffd740", bordercolor="#ffab00", borderwidth=1, borderpad=3,
    )

    fig.update_layout(
        height=850,
        margin=dict(l=55, r=35, t=65, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Pressure / Flow", row=1, col=1)
    fig.update_yaxes(title_text="Drive / Motor", row=2, col=1)
    fig.update_yaxes(title_text="Vibration / Temp", row=3, col=1)
    fig.update_xaxes(title_text="Timestamp (UTC)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 4b. AI Alerts Table
    if not predictions.empty:
        asset_preds_all = predictions[predictions["asset_id"] == selected_asset].copy()
        if not asset_preds_all.empty:
            st.markdown(f"##### 🧠 Active AI Diagnostic Alerts for `{selected_asset}` ({len(asset_preds_all)} alerts):")
            disp_cols = [c for c in ["alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"] if c in asset_preds_all.columns]
            st.dataframe(asset_preds_all[disp_cols], use_container_width=True, hide_index=True)

    # 5. Raw High-Frequency Vibration Waveform & FFT Spectrum Inspector
    st.markdown("---")
    with st.expander("🔬 High-Frequency Vibration Waveform & FFT Harmonic Spectrum Inspector", expanded=False):
        snapshots = load_vibration_snapshots(selected_asset)
        tier_str = asset_row.get("instrumentation_tier", "")

        if "T0" in tier_str or snapshots.empty:
            st.info(
                f"ℹ️ **{selected_asset} is currently in tier `{tier_str}`** (No raw high-frequency vibration sensor installed).\n\n"
                f"Legacy T0 pumps rely solely on motor current and discharge pressure, making early subsurface bearing fatigue "
                f"and shaft imbalance invisible until audible noise or seizure occurs. Upgrading to a **T1 Retrofit (€2.4k)** "
                f"installs a wireless vibration probe providing high-frequency acceleration and crest factor tracking."
            )
        else:
            st.markdown(
                "Inspect raw 10.24 kHz acceleration waveforms and FFT spectral peaks. "
                "Harmonic peaks at $BPFO$ (ball pass frequency outer race) or $BPFI$ indicate localized bearing damage "
                "days before overall velocity or temperature triggers conventional alarms."
            )

            # Snapshots list
            snap_ids = list(snapshots["snapshot_id"].unique())
            # Find closest snapshot to scrubber time
            time_diffs = (snapshots["timestamp_utc"] - scrubber_time).abs()
            closest_idx = time_diffs.argmin()
            default_snap_id = snapshots.iloc[closest_idx]["snapshot_id"]

            snap_c1, snap_c2, snap_c3 = st.columns([2, 2, 2])
            with snap_c1:
                chosen_snap_id = st.selectbox("Select Vibration Burst Snapshot", snap_ids, index=snap_ids.index(default_snap_id))
            snap_data = snapshots[snapshots["snapshot_id"] == chosen_snap_id].iloc[0]

            with snap_c2:
                st.markdown(f"**Capture Time:** `{snap_data['timestamp_utc'].strftime('%Y-%m-%d %H:%M:%S UTC')}`")
                st.caption(f"Sample Rate: `{snap_data['sample_rate_hz']} Hz` · Axis: `{snap_data['axis']}`")

            with snap_c3:
                rpm = float(snap_data.get("rpm_at_capture", asset_row.get("motor_rated_rpm", 1780)))
                st.markdown(f"**Shaft Speed:** `{rpm:.1f} RPM`")
                f_1x = rpm / 60.0
                st.caption(f"Running Frequency (1X): `{f_1x:.2f} Hz`")

            # Extract samples and calculate FFT
            raw_samples = np.array(snap_data["samples"], dtype=np.float32)
            n_samples = len(raw_samples)
            fs = int(snap_data["sample_rate_hz"])
            t_ms = (np.arange(n_samples) / fs) * 1000.0  # ms

            # FFT
            fft_amp = np.abs(np.fft.rfft(raw_samples)) * (2.0 / n_samples)
            freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)

            # Bearing frequencies from asset spec
            bpfo_order = float(asset_row.get("bearing_bpfo_order", 3.057))
            bpfi_order = float(asset_row.get("bearing_bpfi_order", 4.943))
            vanes = int(asset_row.get("impeller_vanes", 6))

            f_bpfo = f_1x * bpfo_order
            f_bpfi = f_1x * bpfi_order
            f_2x = f_1x * 2.0
            f_vpf = f_1x * vanes

            col_w1, col_w2 = st.columns(2)

            with col_w1:
                # Plot Time Waveform (first 100 ms for crisp clarity)
                n_display = min(1024, n_samples)
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=t_ms[:n_display], y=raw_samples[:n_display],
                    line=dict(color="#d62728", width=1.5),
                    name="Acceleration (g)"
                ))
                fig_time.update_layout(
                    title="<b>Raw Waveform (First 100 ms)</b>",
                    xaxis_title="Time (ms)",
                    yaxis_title="Acceleration (g)",
                    height=360,
                    margin=dict(l=40, r=20, t=40, b=30)
                )
                st.plotly_chart(fig_time, use_container_width=True)

            with col_w2:
                # Plot FFT Spectrum (0 to 1500 Hz)
                mask = freqs <= 1500.0
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(
                    x=freqs[mask], y=fft_amp[mask],
                    line=dict(color="#1f77b4", width=1.5),
                    name="FFT Amplitude (g)"
                ))

                # Vertical Harmonic Markers (staggered with high-contrast badges for Dark & Light mode)
                markers = [
                    (f_1x, "1X Shaft", "#4fc3f7", 0.96),
                    (f_2x, "2X Misalign", "#ffb74d", 0.80),
                    (f_bpfo, f"BPFO ({f_bpfo:.1f}Hz)", "#ff5252", 0.64),
                    (f_bpfi, f"BPFI ({f_bpfi:.1f}Hz)", "#ce93d8", 0.48),
                    (f_vpf, f"VPF ({f_vpf:.1f}Hz)", "#69f0ae", 0.32),
                ]
                for f_val, label, col, y_pos in markers:
                    if f_val <= 1500.0:
                        fig_fft.add_vline(
                            x=f_val,
                            line=dict(color=col, width=1.4, dash="dash"),
                        )
                        fig_fft.add_annotation(
                            x=f_val, y=y_pos, xref="x", yref="paper",
                            text=f"<b>{label}</b>",
                            showarrow=False, xanchor="left", xshift=3,
                            font=dict(color=col, size=9.5),
                            bgcolor="rgba(20, 25, 35, 0.88)", bordercolor=col, borderwidth=1, borderpad=2,
                        )

                fig_fft.update_layout(
                    title="<b>FFT Spectrum & Defect Harmonics</b>",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Amplitude (g)",
                    height=360,
                    margin=dict(l=40, r=20, t=40, b=30)
                )
                st.plotly_chart(fig_fft, use_container_width=True)

            # Technical Interpretation
            st.markdown(
                f"**Bearing Order Reference ({asset_row.get('bearing_designation', '6311')} Deep-Groove Ball Bearing):**\n"
                f"• **1X Shaft Speed:** `{f_1x:.1f} Hz` · **2X Harmonics:** `{f_2x:.1f} Hz`\n"
                f"• **BPFO (Outer Race):** `{bpfo_order:.3f} × 1X = {f_bpfo:.1f} Hz`\n"
                f"• **BPFI (Inner Race):** `{bpfi_order:.3f} × 1X = {f_bpfi:.1f} Hz`\n"
                f"• **Vane Pass (VPF):** `{vanes} × 1X = {f_vpf:.1f} Hz`"
            )


