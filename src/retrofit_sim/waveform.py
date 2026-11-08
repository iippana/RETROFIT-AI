"""Synthetic high-frequency (10.24 kHz) vibration acceleration waveform snapshots."""
from __future__ import annotations

import numpy as np


def snapshot_schedule(n: int, running: np.ndarray, interval_s: int, rng: np.random.Generator) -> list[int]:
    """Select one running second every `interval_s` seconds across the window."""
    indices: list[int] = []
    for start in range(600, n - 60, interval_s):
        end = min(n - 1, start + interval_s)
        cand = np.nonzero(running[start:end])[0]
        if cand.size > 0:
            pick = int(start + cand[int(rng.integers(0, min(len(cand), 600)))])
            indices.append(pick)
    return indices


def snapshot(truth: dict, idx: int, acfg: dict, rng: np.random.Generator, fs: int = 10240, n_samples: int = 10240) -> np.ndarray:
    """Synthesize a 1-second radial acceleration waveform [g] at `fs` Hz for sample index `idx`."""
    t = np.arange(n_samples, dtype=float) / float(fs)
    rpm = float(truth["rpm"][idx])
    fr = max(rpm / 60.0, 1.0)
    vanes = int(acfg.get("impeller_vanes", 6))
    bpfo_order = float(acfg.get("bearing_bpfo_order", 3.057))

    w = float(truth["bearing_wear"][idx]) if "bearing_wear" in truth else 0.05
    u = float(truth["unbalance"][idx]) if "unbalance" in truth else 0.0
    m = float(truth["misalignment"][idx]) if "misalignment" in truth else 0.02
    cav = float(truth["cavitation_intensity"][idx]) if "cavitation_intensity" in truth else 0.0
    target_rms = float(truth["vib_accel_rms_g"][idx])

    # 1x, 2x, 3x rotational and vane-pass harmonics
    ph1, ph2, ph3, phv = rng.uniform(0, 2 * np.pi, 4)
    a1 = 0.04 * (1.0 + 1.5 * u + 0.5 * m)
    a2 = 0.025 * (1.0 + 2.5 * m)
    a3 = 0.01 * (1.0 + m)
    av = 0.05 * (1.0 + 0.5 * cav)
    sig = (
        a1 * np.sin(2 * np.pi * fr * t + ph1)
        + a2 * np.sin(2 * np.pi * 2 * fr * t + ph2)
        + a3 * np.sin(2 * np.pi * 3 * fr * t + ph3)
        + av * np.sin(2 * np.pi * vanes * fr * t + phv)
    )

    # Outer-race defect impulse train at BPFO exciting 2.8 kHz housing resonance
    if w > 0.08 and fr > 2.0:
        bpfo = bpfo_order * fr
        period = 1.0 / bpfo
        t_imp = np.arange(rng.uniform(0, period), t[-1], period)
        amp_imp = 2.8 * (max(w - 0.08, 0.0) ** 1.15)
        ring_len = int(0.008 * fs)
        tau_ring = np.arange(ring_len, dtype=float) / float(fs)
        kernel = np.exp(-tau_ring * 950.0) * np.sin(2.0 * np.pi * 2800.0 * tau_ring)
        imp_train = np.zeros(n_samples + ring_len, dtype=float)
        for ti in t_imp:
            k0 = int(ti * fs)
            if 0 <= k0 < n_samples:
                imp_train[k0 : k0 + ring_len] += amp_imp * (1.0 + float(rng.normal(0.0, 0.08))) * kernel
        sig = sig + imp_train[:n_samples]

    # Band-limited cavitation noise (1.5 - 4.2 kHz)
    if cav > 0.01:
        white = rng.normal(0.0, 1.0, n_samples)
        W = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
        band = np.exp(-0.5 * ((freqs - 2600.0) / 800.0) ** 2)
        cav_sig = np.fft.irfft(W * band, n=n_samples)
        cav_sig = cav_sig / max(np.std(cav_sig), 1e-6) * (0.45 * cav)
        sig = sig + cav_sig

    # Broadband sensor & flow floor
    sig = sig + rng.normal(0.0, 0.04, n_samples)
    cur_rms = float(np.sqrt(np.mean(sig ** 2)))
    if cur_rms > 1e-6 and target_rms > 0:
        sig = sig * (target_rms / cur_rms)

    return sig.astype(np.float32)

