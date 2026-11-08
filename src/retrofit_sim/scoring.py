"""Benchmark scorer: burst grouping, event-window matching, anchored false-alert episodes, and composite score."""
from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import pandas as pd

from .faults import MACHINE_FAULTS, SENSOR_FAULTS

FAULT_TYPES = set(MACHINE_FAULTS) | set(SENSOR_FAULTS) | {"UNKNOWN"}
VALID_CLASSES = {"MACHINE", "SENSOR"}


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"asset_id", "alert_time_utc", "event_class", "fault_type", "confidence"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file {path} missing required columns: {sorted(missing)}")
    return df


def _to_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ("true", "1", "yes")


def _make_bursts(pred: pd.DataFrame) -> list[dict]:
    """Group alerts on the same asset within 5 minutes of the burst start into a single burst."""
    if pred is None or len(pred) == 0:
        return []
    df = pred.copy()
    bad_cls = set(df["event_class"].dropna().astype(str).unique()) - VALID_CLASSES
    if bad_cls or df["event_class"].isna().any():
        raise ValueError(f"Invalid event_class values: {bad_cls}; must be in {VALID_CLASSES}")
    df["_t"] = pd.to_datetime(df["alert_time_utc"], utc=True, format="ISO8601")
    df = df.sort_values(["asset_id", "_t"]).reset_index(drop=True)

    bursts: list[dict] = []
    for aid, g in df.groupby("asset_id", sort=False):
        b_start = None
        b_classes: set[str] = set()
        b_types: set[str] = set()
        b_conf = 0.0
        for _, r in g.iterrows():
            t = r["_t"]
            cls = str(r["event_class"])
            ft = str(r["fault_type"])
            conf = float(r.get("confidence", 0.5)) if pd.notna(r.get("confidence", 0.5)) else 0.5
            if b_start is None or (t - b_start) >= pd.Timedelta(minutes=5):
                if b_start is not None:
                    bursts.append(
                        {
                            "asset_id": aid,
                            "time": b_start,
                            "event_class": next(iter(b_classes)) if len(b_classes) == 1 else None,
                            "fault_type": next(iter(b_types)) if len(b_types) == 1 else None,
                            "confidence": b_conf,
                        }
                    )
                b_start = t
                b_classes = {cls}
                b_types = {ft}
                b_conf = conf
            else:
                b_classes.add(cls)
                b_types.add(ft)
                b_conf = max(b_conf, conf)
        if b_start is not None:
            bursts.append(
                {
                    "asset_id": aid,
                    "time": b_start,
                    "event_class": next(iter(b_classes)) if len(b_classes) == 1 else None,
                    "fault_type": next(iter(b_types)) if len(b_types) == 1 else None,
                    "confidence": b_conf,
                }
            )
    return bursts


def _anchored_episodes(times: list[pd.Timestamp], window_min: int = 30) -> list[pd.Timestamp]:
    """Form non-chained `window_min`-minute episodes anchored at the first alert of each episode."""
    if not times:
        return []
    anchors: list[pd.Timestamp] = []
    w = pd.Timedelta(minutes=window_min)
    for t in sorted(times):
        if not anchors or (t - anchors[-1]) >= w:
            anchors.append(t)
    return anchors


def score(
    pred: pd.DataFrame,
    events: pd.DataFrame,
    assets: pd.DataFrame,
    benign: pd.DataFrame | None = None,
    split: str | None = None,
) -> dict:
    """Score predictions against ground-truth events according to BENCHMARK_RULES.md."""
    assets_scope = assets[assets["split"] == split].copy() if split is not None else assets.copy()
    scope_ids = set(assets_scope["asset_id"])
    ws_map = dict(zip(assets_scope["asset_id"], pd.to_datetime(assets_scope["window_start_utc"], utc=True, format="ISO8601")))
    we_map = dict(zip(assets_scope["asset_id"], pd.to_datetime(assets_scope["window_end_utc"], utc=True, format="ISO8601")))
    asset_days = float(sum((we_map[a] - ws_map[a]).total_seconds() / 86400.0 for a in scope_ids))

    pred_scope = pred[pred["asset_id"].isin(scope_ids)].copy() if (pred is not None and len(pred)) else pd.DataFrame(columns=["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence"])
    if pred is not None and len(pred) and len(pred_scope) == 0:
        # Validate event_class even if asset filter emptied it
        _make_bursts(pred)
    bursts_all = _make_bursts(pred_scope)
    bursts = [b for b in bursts_all if ws_map[b["asset_id"]] <= b["time"] <= we_map[b["asset_id"]]]

    ev_scope = events[events["asset_id"].isin(scope_ids)].copy() if (events is not None and len(events)) else pd.DataFrame()
    ev_records: list[dict] = []
    for _, e in ev_scope.iterrows():
        aid = e["asset_id"]
        onset = pd.to_datetime(e["onset_utc"], utc=True, format="ISO8601")
        det = pd.to_datetime(e["detectable_utc"], utc=True, format="ISO8601") if pd.notna(e.get("detectable_utc")) and str(e.get("detectable_utc")) != "" else None
        dl = pd.to_datetime(e["action_deadline_utc"], utc=True, format="ISO8601") if pd.notna(e.get("action_deadline_utc")) and str(e.get("action_deadline_utc")) != "" else None
        end = pd.to_datetime(e["end_utc"], utc=True, format="ISO8601") if pd.notna(e.get("end_utc")) and str(e.get("end_utc")) != "" else None
        pre = _to_bool(e.get("pre_existing", False))
        obs_ok = _to_bool(e.get("observable", True))

        w_start = ws_map[aid] if pre else (max(onset, det - pd.Timedelta(hours=6)) if det is not None else onset)
        if end is None:
            w_end = we_map[aid]
        elif str(e["event_class"]) == "MACHINE" and str(e["fault_type"]) not in ("CAVITATION", "AIR_ENTRAINMENT"):
            w_end = end
        else:
            w_end = min(we_map[aid], end + pd.Timedelta(minutes=30))

        ev_records.append(
            {
                "event_id": e["event_id"],
                "asset_id": aid,
                "event_class": str(e["event_class"]),
                "fault_type": str(e["fault_type"]),
                "onset": onset,
                "detectable": det,
                "deadline": dl,
                "end": end,
                "w_start": w_start,
                "w_end": w_end,
                "observable": obs_ok,
                "matched_bursts": [],
            }
        )

    false_bursts_by_asset: dict[str, list[pd.Timestamp]] = {a: [] for a in scope_ids}

    for b in bursts:
        aid = b["asset_id"]
        t = b["time"]
        cands = [e for e in ev_records if e["asset_id"] == aid and e["observable"] and e["w_start"] <= t <= e["w_end"]]
        if cands:
            cands.sort(
                key=lambda e: (
                    0 if (b["fault_type"] is not None and b["fault_type"] == e["fault_type"]) else 1,
                    0 if (b["event_class"] is not None and b["event_class"] == e["event_class"]) else 1,
                    0 if len(e["matched_bursts"]) == 0 else 1,
                    e["onset"],
                )
            )
            cands[0]["matched_bursts"].append(b)
        else:
            unobs = [e for e in ev_records if e["asset_id"] == aid and (not e["observable"]) and e["w_start"] <= t <= e["w_end"]]
            if not unobs:
                false_bursts_by_asset[aid].append(t)

    scored_events = [e for e in ev_records if e["observable"]]
    n_scored = len(scored_events)
    tp = 0
    type_hits = 0
    class_hits = 0
    timeliness_sum = 0.0
    excess_repeats = 0
    lead_times_h: list[float] = []
    per_event: list[dict] = []

    for e in scored_events:
        m = e["matched_bursts"]
        if m:
            tp += 1
            first = m[0]
            t_det = first["time"]
            ft_ok = bool(first["fault_type"] is not None and first["fault_type"] in (FAULT_TYPES - {"UNKNOWN"}) and first["fault_type"] == e["fault_type"])
            cls_ok = bool(first["event_class"] is not None and first["event_class"] == e["event_class"])
            type_hits += int(ft_ok)
            class_hits += int(cls_ok)

            ref_t = e["deadline"] if e["deadline"] is not None else ((e["detectable"] if e["detectable"] is not None else e["onset"]) + pd.Timedelta(hours=6))
            ref_t = min(ref_t, e["w_end"])
            if t_det <= ref_t:
                t_score = 1.0
            else:
                span = max((e["w_end"] - ref_t).total_seconds(), 1.0)
                t_score = float(np.clip(1.0 - 0.9 * ((t_det - ref_t).total_seconds() / span), 0.1, 1.0))
            timeliness_sum += t_score

            lead_h = None
            if e["deadline"] is not None:
                lead_h = float((e["deadline"] - t_det).total_seconds() / 3600.0)
                lead_times_h.append(lead_h)

            # Repeat episodes after the detecting burst
            rep_anchors = _anchored_episodes([x["time"] for x in m[1:]], 30)
            last_free = t_det
            for ra in rep_anchors:
                if (ra - last_free) >= pd.Timedelta(hours=12):
                    last_free = ra
                else:
                    excess_repeats += 1

            per_event.append(
                {
                    "event_id": e["event_id"],
                    "asset_id": e["asset_id"],
                    "fault_type": e["fault_type"],
                    "detected": True,
                    "first_alert_utc": t_det.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "predicted_type": first["fault_type"],
                    "type_correct": ft_ok,
                    "class_correct": cls_ok,
                    "timeliness": round(t_score, 3),
                    "lead_time_h": round(lead_h, 2) if lead_h is not None else None,
                }
            )
        else:
            per_event.append(
                {
                    "event_id": e["event_id"],
                    "asset_id": e["asset_id"],
                    "fault_type": e["fault_type"],
                    "detected": False,
                    "first_alert_utc": None,
                    "predicted_type": None,
                    "type_correct": False,
                    "class_correct": False,
                    "timeliness": 0.0,
                    "lead_time_h": None,
                }
            )

    false_episodes = 0
    nuisance_episodes = 0
    bn_df = None
    if benign is not None and len(benign):
        bn_df = benign[benign["asset_id"].isin(scope_ids)].copy()
        bn_df["_s"] = pd.to_datetime(bn_df["start_utc"], utc=True, format="ISO8601")
        bn_df["_e"] = pd.to_datetime(bn_df["end_utc"], utc=True, format="ISO8601")

    for aid, ftimes in false_bursts_by_asset.items():
        f_anchors = _anchored_episodes(ftimes, 30)
        false_episodes += len(f_anchors)
        if bn_df is not None and len(f_anchors):
            b_sub = bn_df[bn_df["asset_id"] == aid]
            for fa in f_anchors:
                if ((b_sub["_s"] <= fa) & (fa <= b_sub["_e"])).any():
                    nuisance_episodes += 1

    recall = tp / n_scored if n_scored > 0 else 1.0
    denom_p = tp + false_episodes + excess_repeats
    precision = tp / denom_p if denom_p > 0 else (1.0 if n_scored == 0 else 0.0)
    f1 = (2.0 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    fa_per_day = (false_episodes + excess_repeats) / max(asset_days, 1e-6)
    burden = max(0.0, 1.0 - fa_per_day / 2.0)
    timeliness = timeliness_sum / n_scored if n_scored > 0 else 0.0
    ft_recall = type_hits / n_scored if n_scored > 0 else 0.0
    cls_recall = class_hits / n_scored if n_scored > 0 else 0.0
    composite = 100.0 * (0.35 * f1 + 0.20 * timeliness + 0.25 * ft_recall + 0.10 * cls_recall + 0.10 * burden)

    mach_ev = [e for e in scored_events if e["event_class"] == "MACHINE"]
    sens_ev = [e for e in scored_events if e["event_class"] == "SENSOR"]
    mach_rec = sum(1 for e in mach_ev if e["matched_bursts"]) / len(mach_ev) if mach_ev else np.nan
    sens_rec = sum(1 for e in sens_ev if e["matched_bursts"]) / len(sens_ev) if sens_ev else np.nan

    return {
        "split": split or "all",
        "assets_scored": len(scope_ids),
        "asset_days": round(asset_days, 2),
        "events_scored": n_scored,
        "alerts_submitted": int(len(pred_scope)),
        "alerts_scored": int(len(bursts)),
        "alert_bursts": int(len(bursts)),
        "true_positive_events": tp,
        "false_alert_episodes": false_episodes,
        "nuisance_alert_episodes": nuisance_episodes,
        "excess_repeat_episodes": excess_repeats,
        "event_recall": float(recall),
        "machine_event_recall": float(mach_rec) if not np.isnan(mach_rec) else np.nan,
        "sensor_event_recall": float(sens_rec) if not np.isnan(sens_rec) else np.nan,
        "alert_precision": float(precision),
        "event_f1": float(f1),
        "false_alerts_per_asset_day": float(fa_per_day),
        "alert_burden": float(burden),
        "timeliness": float(timeliness),
        "fault_type_recall": float(ft_recall),
        "class_recall": float(cls_recall),
        "median_lead_time_h": float(np.median(lead_times_h)) if lead_times_h else np.nan,
        "composite_score": float(composite),
        "per_event": per_event,
    }


def format_report(r: dict) -> str:
    lead = f"{r['median_lead_time_h']:.1f} h" if not pd.isna(r.get("median_lead_time_h", np.nan)) else "-"
    lines = [
        f"Composite score : {r['composite_score']:.1f} / 100",
        f"Event F1        : {r['event_f1']:.3f}  (recall {r['event_recall']:.3f}, precision {r['alert_precision']:.3f}; TP={r['true_positive_events']}/{r['events_scored']})",
        f"Diagnosis       : fault-type recall {r['fault_type_recall']:.3f}, class recall {r['class_recall']:.3f}",
        f"Timeliness      : {r['timeliness']:.3f}  (median lead to action deadline: {lead})",
        f"Alert burden    : {r['alert_burden']:.3f}  (false+excess episodes/asset-day: {r['false_alerts_per_asset_day']:.3f}; FP={r['false_alert_episodes']}, excess={r['excess_repeat_episodes']})",
    ]
    return "\n".join(lines)


def score_main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrofit_sim score")
    p.add_argument("--predictions", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--benign", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--json", default=None, help="Write JSON report to file.")
    a = p.parse_args(argv)
    pred = load_predictions(a.predictions)
    ev = pd.read_csv(a.events)
    assets = pd.read_csv(a.assets)
    bn = pd.read_csv(a.benign) if a.benign else None
    res = score(pred, ev, assets, bn, a.split)
    print(format_report(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=str)
    return 0

