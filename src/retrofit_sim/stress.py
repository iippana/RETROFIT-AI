"""Stress-tier random fleet sampler for robustness testing."""
from __future__ import annotations

import copy
import numpy as np

from .fleet import default_limits


def sample_fleet(cfg: dict, seed: int, n_assets: int = 6, days: float = 3.0) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    base_assets = cfg["assets"]
    profiles = cfg["profiles"]
    out = []
    for k in range(n_assets):
        template = copy.deepcopy(base_assets[int(rng.integers(0, len(base_assets)))])
        prof = template["profile"]
        a = copy.deepcopy(profiles[prof])
        a.update(template)
        a["asset_id"] = f"STRESS-{k + 1:02d}"
        a["split"] = "stress"
        a["limits"] = default_limits(a)
        out.append(a)
    return out

