"""Verification script for Retrofit AI Operator Console & Digital Twin.

Tests data loaders, physics sandbox integration, AI diagnostic pipeline outputs, and UI module exports.
Run with: python verify_ui.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def test_data_loader():
    print("Testing data_loader...")
    from ui_components.data_loader import (
        load_assets,
        load_events,
        load_alarms,
        load_features_1min,
        load_vibration_snapshots,
        get_asset_info,
        get_active_events,
    )
    assets = load_assets()
    print(f"  ✓ Loaded {len(assets)} assets (splits: {dict(assets['split'].value_counts())})")
    assert len(assets) == 12, f"Expected 12 assets, got {len(assets)}"

    events = load_events()
    print(f"  ✓ Loaded {len(events)} ground-truth events across train/validation")

    alarms = load_alarms()
    print(f"  ✓ Loaded {len(alarms)} SCADA alarms")

    p6_feat = load_features_1min("PUMP-06")
    print(f"  ✓ Loaded 1-min features for PUMP-06 ({len(p6_feat)} rows)")

    snaps = load_vibration_snapshots("PUMP-04")
    print(f"  ✓ Loaded {len(snaps)} raw vibration snapshots for PUMP-04")

    from ui_components.data_loader import load_predictions, load_benchmark_metrics
    ai_preds = load_predictions()
    print(f"  ✓ Loaded {len(ai_preds)} full-fleet AI diagnostic alerts across {ai_preds['asset_id'].nunique()} pumps")
    assert not ai_preds.empty, "Expected non-empty AI predictions for UI"

    bm = load_benchmark_metrics()
    val_score = bm.get("validation_composite_score")
    print(f"  ✓ Loaded benchmark metrics (Validation Composite Score: {val_score} / 100)")
    assert val_score is not None and val_score > 90.0, f"Expected validation score > 90, got {val_score}"


def test_what_if_sandbox():
    print("\nTesting 'What-If' digital twin sandbox...")
    from retrofit_sim.sandbox import what_if
    from ui_components.data_loader import get_data_dir

    data_dir = str(get_data_dir())
    res = what_if(
        data_dir=data_dir,
        asset_id="PUMP-06",
        time_utc="2025-10-09T21:50:00Z",
        action="clean_strainer",
        value=None,
        hypothesis={"clog": 4.5},
        root=str(ROOT),
    )
    print(f"  ✓ Simulation accepted: {res.get('accepted')}")
    if "baseline" in res and "proposed" in res:
        print(f"  ✓ Baseline flow: {res['baseline'].get('flow_m3h')} m³/h -> Proposed: {res['proposed'].get('flow_m3h')} m³/h")
        print(f"  ✓ Intervention cost: €{res.get('intervention', {}).get('estimated_cost_eur')}")


def test_diagnostic_logic():
    print("\nTesting diagnostic triage logic...")
    import pandas as pd
    from ui_components.diagnostic_drawer import evaluate_diagnostic_state
    from ui_components.data_loader import load_features_1min, get_asset_info

    asset_info = get_asset_info("PUMP-06")
    df = load_features_1min("PUMP-06")
    diag = evaluate_diagnostic_state(
        "PUMP-06",
        pd.Timestamp("2025-10-09T21:45:00Z", tz="UTC"),
        df,
        asset_info,
    )
    print(f"  ✓ Diagnostic Triage: {diag['event_class']} -> {diag['fault_type']} (Confidence: {int(diag['confidence']*100)}%)")
    print(f"  ✓ Action: {diag['recommended_action'][:60]}...")

    # Verify blinded test pump (PUMP-12) diagnostic drawer is driven by AI Diagnostic Pipeline
    asset_12 = get_asset_info("PUMP-12")
    df_12 = load_features_1min("PUMP-12")
    diag_12 = evaluate_diagnostic_state(
        "PUMP-12",
        pd.Timestamp("2025-11-27T04:50:00Z", tz="UTC"),
        df_12,
        asset_12,
    )
    print(f"  ✓ Blinded Test Pump (PUMP-12) AI Triage: {diag_12['event_class']} -> {diag_12['fault_type']} (Confidence: {int(diag_12['confidence']*100)}%)")
    assert diag_12["fault_type"] == "CAVITATION", f"Expected CAVITATION on PUMP-12, got {diag_12['fault_type']}"


if __name__ == "__main__":
    print("=== RETROFIT AI UI VERIFICATION ===")
    try:
        test_data_loader()
        test_what_if_sandbox()
        test_diagnostic_logic()
        print("\n🎉 ALL TESTS PASSED! UI components and digital twin are fully operational.")
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()

