"""RETROFIT AI — Operator Intelligence Console.

Since AI Hackathon 2026 · Turku (6–8 November 2026)
Challenge Owner: Business Turku · Technical Sparring Partner: Elomatic
Goal: Retrofit intelligence onto 10–30 year old cooling-water pumps without hardware control writes.
"""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root and src to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from ui_components.data_loader import (
    load_assets,
    load_events,
    load_alarms,
    load_predictions,
    load_benchmark_metrics,
    run_ai_pipeline,
    get_asset_info,
    get_data_dir,
)
from ui_components.fleet_view import render_fleet_view
from ui_components.telemetry_view import render_telemetry_view
from ui_components.diagnostic_drawer import render_diagnostic_drawer
from ui_components.what_if_view import render_what_if_view
from ui_components.retrofit_economics import render_retrofit_economics_view

# Page Config
st.set_page_config(
    page_title="Retrofit AI — Operator Intelligence Console",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .badge-ro {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-hackathon {
        background-color: #e3f2fd;
        color: #1565c0;
        border: 1px solid #90caf9;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize persistent session state variables."""
    if "selected_asset" not in st.session_state:
        st.session_state["selected_asset"] = "PUMP-06"
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = "🏢 Fleet Overview"
    if "scrubber_time" not in st.session_state:
        st.session_state["scrubber_time"] = pd.Timestamp("2025-10-09T21:45:00Z")
    if "whatif_action" not in st.session_state:
        st.session_state["whatif_action"] = "clean_strainer"
    if "whatif_hypothesis" not in st.session_state:
        st.session_state["whatif_hypothesis"] = {"clog": 4.0}


def main():
    init_session_state()

    # Apply any programmatic navigation or asset switches BEFORE sidebar widgets instantiate
    if st.session_state.get("_pending_nav"):
        st.session_state["active_nav"] = st.session_state.pop("_pending_nav")
    if st.session_state.get("_pending_asset"):
        st.session_state["selected_asset"] = st.session_state.pop("_pending_asset")

    # Auto-bootstrap on a brand-new clean installation if full dataset or pipeline outputs are missing
    if not (ROOT_DIR / "build" / "benchmark_metrics.json").exists() or not (ROOT_DIR / "data" / "full" / "features_1min.parquet").exists():
        with st.spinner("First-run setup: generating full 12-pump telemetry dataset & running AI diagnostic pipeline (~15s)..."):
            ok, msg = run_ai_pipeline()
            if not ok:
                st.warning(f"Auto-bootstrap notice: {msg}")

    # Load baseline assets
    assets = load_assets()
    data_path = get_data_dir()
    is_full_tier = "full" in str(data_path)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ **RETROFIT AI**")
        st.markdown(
            "<span class='badge-hackathon'>Since AI Hackathon 2026</span> "
            "<span class='badge-ro'>Read-Only</span>",
            unsafe_allow_html=True
        )
        st.caption("Challenge Owner: **Business Turku** · Partner: **Elomatic**")
        st.markdown("---")

        # Navigation Menu (bound directly to key='active_nav' so single click always switches)
        nav_options = [
            "🏢 Fleet Overview",
            "📈 Incident Replay & Telemetry",
            "🔍 Diagnostic & Evidence Drawer",
            "🧪 'What-If' Counterfactual Sandbox",
            "💰 Retrofit BOM & Economics",
        ]
        if st.session_state.get("active_nav") not in nav_options:
            st.session_state["active_nav"] = nav_options[0]
        st.radio("Navigation", nav_options, key="active_nav")

        st.markdown("---")

        # Active Asset Quick Switcher
        st.markdown("##### 🎯 Quick Asset Select:")
        asset_list = list(assets["asset_id"].unique())
        if st.session_state.get("selected_asset") not in asset_list:
            st.session_state["selected_asset"] = asset_list[0]
        new_asset = st.selectbox("Asset", asset_list, key="selected_asset", label_visibility="collapsed")

        asset_info = get_asset_info(new_asset)
        if asset_info is not None:
            tier_str = asset_info["instrumentation_tier"]
            st.caption(f"Tier: **{tier_str}** · Mode: **{asset_info['control']}**")
            st.caption(f"Profile: `{asset_info['profile']}`")

        st.markdown("---")

        # AI Diagnostic Pipeline Panel
        st.markdown("##### 🧠 AI Diagnostic Engine")
        st.caption("Model: **Hybrid Physics + Spectral ML**")

        preds = load_predictions()
        bench = load_benchmark_metrics()
        if bench.get("metrics_available"):
            val_m = bench.get("validation_metrics", {})
            score_val = val_m.get("composite_score", bench.get("validation_composite_score", 97.0))
            st.success(f"Active: **{len(preds)} Fleet Alerts** · Score: **{score_val:.1f}/100**")
        elif not preds.empty:
            st.info(f"Loaded **{len(preds)}** fleet alerts")
        else:
            st.warning("No predictions loaded yet. Click below to run the AI pipeline.")

        if st.button("🔄 Run / Refresh AI Pipeline", use_container_width=True):
            with st.spinner("Running src/pipeline.py..."):
                ok, msg = run_ai_pipeline()
                if ok:
                    st.success("AI diagnostic pipeline executed and updated!")
                    st.rerun()
                else:
                    st.error(f"Pipeline error: {msg}")

        st.markdown("---")

        # Environment & Safety Info
        st.markdown("##### 🛡️ Cyber-Physical Safety")
        st.info("Strictly Read-Only Architecture. No write-paths to PLCs, drives, or safety systems (IEC 62443).")

        st.markdown("##### 📁 Telemetry Data Tier")
        if is_full_tier:
            st.success("Connected: **Full Tier (1 Hz / 1-min features)**")
        else:
            st.warning("Connected: **Starter Tier (10-s resolution)**")

    # Content Routing
    nav = st.session_state["active_nav"]

    def handle_asset_selection(aid: str):
        st.session_state["_pending_asset"] = aid
        st.session_state["_pending_nav"] = "📈 Incident Replay & Telemetry"
        st.rerun()

    if nav == "🏢 Fleet Overview":
        render_fleet_view(on_select_asset=handle_asset_selection)
    elif nav == "📈 Incident Replay & Telemetry":
        render_telemetry_view()
    elif nav == "🔍 Diagnostic & Evidence Drawer":
        render_diagnostic_drawer()
    elif nav == "'What-If' Counterfactual Sandbox" or "What-If" in nav:
        render_what_if_view()
    elif nav == "💰 Retrofit BOM & Economics":
        render_retrofit_economics_view()


if __name__ == "__main__":
    main()
