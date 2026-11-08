"""Reduced-order pump, system-curve, cavitation, and motor-drive physics relations."""
from __future__ import annotations

import numpy as np

G_ACC = 9.81
G = G_ACC


def water_density(temp_c: np.ndarray | float, salinity_psu: float = 0.0) -> np.ndarray:
    """Water density [kg/m^3] as a function of temperature [degC] and salinity [PSU]."""
    t = np.asarray(temp_c, dtype=float)
    rho_fw = 1000.0 * (1.0 - ((t + 288.9414) / (508929.2 * (t + 68.12963))) * ((t - 3.9863) ** 2))
    return rho_fw + 0.78 * float(salinity_psu)


def vapour_pressure_pa(temp_c: np.ndarray | float) -> np.ndarray:
    """Water vapour pressure [Pa] via the Antoine equation."""
    t = np.asarray(temp_c, dtype=float)
    return 133.322 * (10.0 ** (8.07131 - 1730.63 / (233.426 + t)))


def pump_head(
    Q: np.ndarray | float,
    r: np.ndarray | float,
    h_bep: float,
    q_bep: float,
    h0_ratio: float = 1.20,
    f_head: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Pump head [m] for volumetric flow Q [m^3/s] and speed ratio r = n / n_rated."""
    Q = np.asarray(Q, dtype=float)
    r = np.asarray(r, dtype=float)
    f_head = np.asarray(f_head, dtype=float)
    h = h_bep * (h0_ratio * (r ** 2) - (h0_ratio - 1.0) * ((Q / max(q_bep, 1e-9)) ** 2)) * f_head
    return np.where(r > 1e-4, np.maximum(h, 0.0), 0.0)


def solve_operating_point(
    r: np.ndarray | float,
    h_bep: float,
    q_bep: float,
    h0_ratio: float,
    h_static: np.ndarray | float,
    K_total: np.ndarray | float,
    f_head: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Analytical intersection of the quadratic pump curve and system curve H_sys = H_static + K_total * Q^2."""
    r = np.asarray(r, dtype=float)
    K_total = np.asarray(K_total, dtype=float)
    f_head = np.asarray(f_head, dtype=float)
    h_static = np.asarray(h_static, dtype=float)
    num = np.maximum(0.0, h_bep * f_head * h0_ratio * (r ** 2) - h_static)
    den = K_total + h_bep * f_head * (h0_ratio - 1.0) / (max(q_bep, 1e-9) ** 2)
    q = np.sqrt(num / np.maximum(den, 1e-9))
    return np.where(r > 1e-4, q, 0.0)


def npsh3(Q: np.ndarray | float, r: np.ndarray | float, q_bep: float, npshr_bep: float) -> np.ndarray:
    """Required NPSH3 [m] scaled with speed squared and flow ratio q = Q / (r Q_BEP)."""
    Q = np.asarray(Q, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.where(r > 0.05, Q / np.maximum(r * q_bep, 1e-9), 0.0)
    q_clip = np.clip(q, 0.0, 2.0)
    return npshr_bep * (np.maximum(r, 0.2) ** 2) * (0.55 + 0.45 * (q_clip ** 2.6))


def cavitation_head_factor(npsha: np.ndarray | float, npshr: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return (f_cav, cavitation_intensity) from available and required NPSH [m]."""
    npsha = np.asarray(npsha, dtype=float)
    npshr = np.asarray(npshr, dtype=float)
    sigma = npsha / np.maximum(npshr, 0.1)
    f_cav = np.clip(1.0 - 0.03 * np.exp(-6.0 * (sigma - 1.0)), 0.55, 1.0)
    intensity = np.clip((1.25 - sigma) / 0.45, 0.0, 1.0)
    return f_cav, intensity


def shaft_power(
    Q: np.ndarray | float,
    H: np.ndarray | float,
    r: np.ndarray | float,
    q_bep: float,
    h_bep: float,
    eta_bep: float,
    h0_ratio: float = 1.20,
    rho: np.ndarray | float = 1000.0,
    f_head: np.ndarray | float = 1.0,
    mech_Drag: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Shaft power [W], algebraically equal to rho*g*Q*H / eta and finite at shut-off."""
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    f_head = np.asarray(f_head, dtype=float)
    r_safe = np.maximum(r, 0.1)
    q = np.where(r > 1e-4, Q / np.maximum(r_safe * q_bep, 1e-9), 0.0)
    q_eff = np.clip(q, 0.0, 1.65)
    eta_r = 1.0 - (1.0 - eta_bep) * (np.clip(r_safe, 0.35, 1.2) ** -0.1)
    h_q = np.where(
        r > 1e-4,
        np.maximum(H / np.maximum(h_bep * (r_safe ** 2), 1e-6), 0.15 * f_head),
        0.0,
    )
    p = (rho * G_ACC * q_bep * h_bep / np.maximum(eta_r, 0.2)) * (r ** 3) * (h_q / (2.0 - q_eff)) * mech_Drag
    return np.where(r > 1e-4, np.maximum(p, 0.0), 0.0)


def vfd_efficiency(load_frac: np.ndarray | float, rated_vfd_eff: float = 0.97) -> np.ndarray:
    """Drive efficiency as a function of load fraction (US DOE reference curve)."""
    lam = np.clip(np.asarray(load_frac, dtype=float), 0.0, 1.5)
    xp = np.array([0.0, 0.125, 0.25, 0.50, 1.0, 1.5])
    fp = np.array([0.82, 0.88, 0.93, 0.96, 0.97, 0.97]) * (rated_vfd_eff / 0.97)
    return np.interp(lam, xp, fp)


def motor_electrical(
    P_sh: np.ndarray | float,
    P_rated: float,
    eta_rated: float,
    pf_rated: float,
    voltage_v: float,
    v_pu: np.ndarray | float = 1.0,
    const_loss_share: float = 0.35,
    is_vfd: bool = False,
    rated_vfd_eff: float = 0.97,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (P_in [W], eta_m, pf, I [A], load_factor) from shaft power [W]."""
    P_sh = np.asarray(P_sh, dtype=float)
    v_pu = np.asarray(v_pu, dtype=float)
    lam = np.maximum(P_sh, 0.0) / max(P_rated, 1.0)
    L_rated = P_rated * (1.0 / max(eta_rated, 0.5) - 1.0)
    L = L_rated * (const_loss_share + (1.0 - const_loss_share) * ((lam / np.maximum(v_pu, 0.5)) ** 2))
    P_motor = np.where(P_sh > 0.0, P_sh + L, 0.0)
    if is_vfd:
        eta_vfd = vfd_efficiency(lam, rated_vfd_eff)
        P_in = np.where(P_sh > 0.0, P_motor / np.maximum(eta_vfd, 0.5), 0.0)
    else:
        P_in = P_motor
    eta_m = np.where(P_in > 0.0, P_sh / np.maximum(P_in, 1e-9), 0.0)
    pf = pf_rated * (1.0 - np.exp(-np.maximum(lam, 0.02) / 0.3)) / (1.0 - np.exp(-1.0 / 0.3))
    pf = np.where(P_sh > 0.0, np.clip(pf, 0.15, 0.98), 0.0)
    I = np.where(P_sh > 0.0, P_in / (np.sqrt(3.0) * voltage_v * np.maximum(v_pu, 0.5) * np.maximum(pf, 0.15)), 0.0)
    return P_in, eta_m, pf, I, lam
