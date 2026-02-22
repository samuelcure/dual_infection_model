"""
Author: Samuel Curé, February 2026.
A simple implementation of the model by Abu-Raddad et al. (Science 2006).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Tuple


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ----------------------------
# Model
# ----------------------------
# Per group: S, I0, Y1, I1, Y2, I2a, I2b, Y3, I3
# Vector: V_S, V_I
STATE_DIM = 20

@dataclass(frozen=True)
class Assumptions:
    # HIV transmission (per coital act) by stage
    p_act_acute: float
    p_act_chronic: float
    p_act_advanced: float

    # HIV stage durations
    dur_acute_months: float
    dur_chronic_years: float
    dur_advanced_years: float

    # Viral load -> transmissibility
    rr_per_log10_vl: float
    log_inc_chronic_clin: float
    log_inc_chronic_nonclin: float
    log_inc_advanced: float

    # HIV -> malaria susceptibility (percent increase)
    sus_inc_chronic_pct: float
    sus_inc_advanced_pct: float

    # Malaria infectious period (also used as "duration of heightened viral load") in days
    tau_gamet_days: float

    # Sexual activity reduction during malaria (percent)
    red_sex_clin_pct: float
    red_sex_nonclin_pct: float

    # Fraction clinical malaria (percent)
    frac_clin_hiv_neg_pct: float
    frac_clin_hiv_pos_pct: float

    # Malaria-enhanced HIV mortality in advanced stage (percent increase; 0 in stable malaria)
    mort_inc_adv_coinf_pct: float


@dataclass
class Params:
    # demographics
    N0: np.ndarray  # [general, core]
    mu: float       # per day

    # HIV progression & behavior
    omega1: float   # per day (acute -> chronic)
    omega2: float   # per day (chronic -> advanced)
    omega3: float   # per day (advanced -> death)
    rho: np.ndarray # partners/day [general, core]
    assort_e: float
    tau_p_months: float
    n_month: Dict[int, float]   # coital acts/month by HIV stage
    p_act: Dict[int, float]     # per-act HIV transmission prob by HIV stage

    # malaria / vectors
    nu: float
    theta: float
    t_vh: float
    t_hv: float
    mu_v: float
    tau_incub: float
    D_avg: float
    ratio_v: float

    # interaction
    g: Dict[int, float]         # malaria susceptibility multipliers by HIV stage (1..3)
    log_inc: Dict[str, float]   # log10 VL increments for I-compartments
    rr_per_log: float
    f_clin_hiv_neg: float
    f_clin_hiv_pos: float
    r_clin: float
    r_nonclin: float
    d_mort: float               # multiplier on omega3 for I3

    # cached
    z: Dict[str, Dict[str, float]] | None = None


# ----------------------------
# Fixed Kisumu setting params (Fig. 1)
# ----------------------------
def build_params(a: Assumptions, *, interaction: bool) -> Params:
    # Population / risk groups
    N_total = 200_000.0
    frac_core = 0.113
    N0 = np.array([(1.0 - frac_core) * N_total, frac_core * N_total], dtype=float)

    # Natural turnover from sexually active class: 35 years
    mu = 1.0 / (35.0 * 365.0)

    # HIV progression rates from durations (per day)
    omega1 = (12.0 / a.dur_acute_months) / 365.0
    omega2 = 1.0 / (a.dur_chronic_years * 365.0)
    omega3 = 1.0 / (a.dur_advanced_years * 365.0)

    # Sexual behavior (Table S2)
    rho_general_y = 0.93
    rho = np.array([rho_general_y, 10.0 * rho_general_y], dtype=float) / 365.0
    assort_e = 0.1
    tau_p_months = 6.0
    n_month = {1: 10.6, 2: 11.0, 3: 7.1}

    # HIV transmission probabilities (Table S2; exposed in UI)
    p_act = {1: a.p_act_acute, 2: a.p_act_chronic, 3: a.p_act_advanced}

    # Malaria / vector (Table S1)
    tau_gamet = max(1.0, a.tau_gamet_days)
    nu = 1.0 / tau_gamet
    theta = 0.4
    t_vh = 0.026
    t_hv = 0.495
    mu_v = 0.167
    tau_incub = 16.0
    D_avg = 59.1
    ratio_v = 2.0

    # Interaction (Table S4)
    rr_per_log = a.rr_per_log10_vl
    f_clin_hiv_neg = a.frac_clin_hiv_neg_pct / 100.0
    f_clin_hiv_pos = a.frac_clin_hiv_pos_pct / 100.0
    r_clin = a.red_sex_clin_pct / 100.0
    r_nonclin = a.red_sex_nonclin_pct / 100.0
    d_mort = 1.0 + a.mort_inc_adv_coinf_pct / 100.0

    if interaction:
        g = {
            1: 1.0,
            2: 1.0 + a.sus_inc_chronic_pct / 100.0,
            3: 1.0 + a.sus_inc_advanced_pct / 100.0,
        }
        log_inc = {
            "I1": 0.0,
            "I2a": a.log_inc_chronic_clin,
            "I2b": a.log_inc_chronic_nonclin,
            "I3": a.log_inc_advanced,
        }
    else:
        g = {1: 1.0, 2: 1.0, 3: 1.0}
        log_inc = {"I1": 0.0, "I2a": 0.0, "I2b": 0.0, "I3": 0.0}
        r_clin = 0.0
        r_nonclin = 0.0
        d_mort = 1.0

    return Params(
        N0=N0, mu=mu,
        omega1=omega1, omega2=omega2, omega3=omega3,
        rho=rho, assort_e=assort_e, tau_p_months=tau_p_months, n_month=n_month, p_act=p_act,
        nu=nu, theta=theta, t_vh=t_vh, t_hv=t_hv, mu_v=mu_v, tau_incub=tau_incub,
        D_avg=D_avg, ratio_v=ratio_v,
        g=g, log_inc=log_inc, rr_per_log=rr_per_log,
        f_clin_hiv_neg=f_clin_hiv_neg, f_clin_hiv_pos=f_clin_hiv_pos,
        r_clin=r_clin, r_nonclin=r_nonclin, d_mort=d_mort,
    )


def vector_density(p: Params, t_days: float) -> float:
    """Seasonal mosquito density (mosquito per human), mean p.D_avg."""
    b = 0.5 * (1.0 - 1.0 / p.ratio_v)
    return p.D_avg / (1.0 - b) * ((1.0 - b) + b * math.sin(2.0 * math.pi * t_days / 365.0))


def precompute_z(p: Params) -> Dict[str, Dict[str, float]]:
    """Per-partnership HIV transmission probability z (Eq. 9), including morbidity effects."""
    r_hiv_neg = p.f_clin_hiv_neg * p.r_clin + (1.0 - p.f_clin_hiv_neg) * p.r_nonclin
    r_hiv_pos = p.f_clin_hiv_pos * p.r_clin + (1.0 - p.f_clin_hiv_pos) * p.r_nonclin

    r = {
        "S": 0.0,
        "I0": r_hiv_neg,
        "Y1": 0.0, "Y2": 0.0, "Y3": 0.0,
        "I1": r_hiv_pos,
        "I2a": p.r_clin,
        "I2b": p.r_nonclin,
        "I3": r_hiv_pos,
    }
    stage = {"Y1": 1, "I1": 1, "Y2": 2, "I2a": 2, "I2b": 2, "Y3": 3, "I3": 3}

    p_act = {
        "Y1": p.p_act[1],
        "Y2": p.p_act[2],
        "Y3": p.p_act[3],
        "I1": p.p_act[1] * (p.rr_per_log ** p.log_inc["I1"]),
        "I2a": p.p_act[2] * (p.rr_per_log ** p.log_inc["I2a"]),
        "I2b": p.p_act[2] * (p.rr_per_log ** p.log_inc["I2b"]),
        "I3": p.p_act[3] * (p.rr_per_log ** p.log_inc["I3"]),
    }

    z = {"S": {}, "I0": {}}
    for sus in ("S", "I0"):
        for inf in ("Y1", "Y2", "Y3", "I1", "I2a", "I2b", "I3"):
            acts = p.n_month[stage[inf]] * p.tau_p_months
            acts *= (1.0 - r[inf]) * (1.0 - r[sus])
            z[sus][inf] = 1.0 - (1.0 - p_act[inf]) ** acts
    return z


def deriv(t_days: float, y: np.ndarray, p: Params, y_past: Callable[[float], np.ndarray]) -> np.ndarray:
    """DDE RHS; only the vector infectious compartment has a delay term (Eq. 2)."""
    host = y[:18].reshape(2, 9)
    S, I0, Y1, I1, Y2, I2a, I2b, Y3, I3 = host.T
    N = host.sum(axis=1)
    N_total = N.sum()
    V_S, V_I = y[18], y[19]

    # Malaria FOI to humans (Eq. 10)
    lam_M = p.theta * p.t_vh * (V_I / N_total)

    # HIV mixing matrix (Eq. 8)
    M = p.rho * N
    mix = (M / M.sum()) if M.sum() > 0 else np.array([0.5, 0.5])
    G = (1.0 - p.assort_e) * np.tile(mix, (2, 1)) + p.assort_e * np.eye(2)

    # HIV FOI (Eqs. 6-7)
    assert p.z is not None
    frac = np.zeros((2, 7))
    # order: Y1,Y2,Y3,I1,I2a,I2b,I3
    frac[:, 0] = Y1 / N
    frac[:, 1] = Y2 / N
    frac[:, 2] = Y3 / N
    frac[:, 3] = I1 / N
    frac[:, 4] = I2a / N
    frac[:, 5] = I2b / N
    frac[:, 6] = I3 / N

    zS = np.array([p.z["S"]["Y1"], p.z["S"]["Y2"], p.z["S"]["Y3"], p.z["S"]["I1"], p.z["S"]["I2a"], p.z["S"]["I2b"], p.z["S"]["I3"]])
    zI0 = np.array([p.z["I0"]["Y1"], p.z["I0"]["Y2"], p.z["I0"]["Y3"], p.z["I0"]["I1"], p.z["I0"]["I2a"], p.z["I0"]["I2b"], p.z["I0"]["I3"]])

    inf_S = frac @ zS
    inf_I0 = frac @ zI0

    lam_HIV_S = p.rho * (G @ inf_S)
    lam_HIV_I0 = p.rho * (G @ inf_I0)

    # Malaria FOI to vectors (Eq. 11)
    I_tot = (I0 + I1 + I2a + I2b + I3).sum()
    lam_HV = p.theta * p.t_hv * (I_tot / N_total)

    # Delayed inflow into infectious vectors (Eq. 2)
    y_tau = y_past(t_days - p.tau_incub)
    host_tau = y_tau[:18].reshape(2, 9)
    I0_t, I1_t, I2a_t, I2b_t, I3_t = host_tau[:, 1], host_tau[:, 3], host_tau[:, 5], host_tau[:, 6], host_tau[:, 8]
    N_tau = host_tau.sum()
    I_tot_tau = (I0_t + I1_t + I2a_t + I2b_t + I3_t).sum()
    lam_HV_tau = p.theta * p.t_hv * (I_tot_tau / N_tau)
    inflow_VI = lam_HV_tau * y_tau[18] * math.exp(-p.mu_v * p.tau_incub)

    # Vector demography with seasonal density (Eq. 3)
    V_total = vector_density(p, t_days) * N_total
    dV_S = p.mu_v * V_total - p.mu_v * V_S - lam_HV * V_S
    dV_I = inflow_VI - p.mu_v * V_I

    # Host dynamics (Eq. 1)
    dhost = np.zeros_like(host)
    for i in range(2):
        dhost[i, 0] = p.mu * p.N0[i] + p.nu * I0[i] - p.mu * S[i] - lam_M * S[i] - lam_HIV_S[i] * S[i]
        dhost[i, 1] = lam_M * S[i] - p.mu * I0[i] - p.nu * I0[i] - lam_HIV_I0[i] * I0[i]
        dhost[i, 2] = lam_HIV_S[i] * S[i] + p.nu * I1[i] - p.mu * Y1[i] - p.omega1 * Y1[i] - p.g[1] * lam_M * Y1[i]
        dhost[i, 3] = lam_HIV_I0[i] * I0[i] + p.g[1] * lam_M * Y1[i] - p.mu * I1[i] - p.nu * I1[i] - p.omega1 * I1[i]
        dhost[i, 4] = p.omega1 * Y1[i] + p.nu * I2a[i] + p.nu * I2b[i] - p.mu * Y2[i] - p.omega2 * Y2[i] - p.g[2] * lam_M * Y2[i]
        dhost[i, 5] = p.f_clin_hiv_pos * (p.g[2] * lam_M * Y2[i] + p.omega1 * I1[i]) - p.mu * I2a[i] - p.nu * I2a[i] - p.omega2 * I2a[i]
        dhost[i, 6] = (1.0 - p.f_clin_hiv_pos) * (p.g[2] * lam_M * Y2[i] + p.omega1 * I1[i]) - p.mu * I2b[i] - p.nu * I2b[i] - p.omega2 * I2b[i]
        dhost[i, 7] = p.omega2 * Y2[i] + p.nu * I3[i] - p.mu * Y3[i] - p.omega3 * Y3[i] - p.g[3] * lam_M * Y3[i]
        dhost[i, 8] = p.g[3] * lam_M * Y3[i] + p.omega2 * (I2a[i] + I2b[i]) - p.mu * I3[i] - p.nu * I3[i] - p.d_mort * p.omega3 * I3[i]

    dy = np.zeros_like(y)
    dy[:18] = dhost.reshape(-1)
    dy[18] = dV_S
    dy[19] = dV_I
    return dy


def simulate(
    p: Params,
    *,
    year_start: float = 1960.0,
    year_end: float = 2010.0,
    seed_year: float = 1980.0,
    dt_days: float = 1.0,
    hiv_seed_core: float = 50.0,
    hiv_seed_gen: float = 13.0,
    init_mal_prev: float = 0.35,
    init_vec_inf: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Daily-grid method-of-steps RK4."""
    p.z = precompute_z(p)

    t0 = (year_start - seed_year) * 365.0
    t1 = (year_end - seed_year) * 365.0
    t = np.arange(t0, t1 + dt_days, dt_days)
    y = np.zeros((len(t), STATE_DIM), dtype=float)

    # Host ICs (no HIV, malaria endemic guess)
    for gi in range(2):
        N0 = p.N0[gi]
        I0_init = init_mal_prev * N0
        y[0, gi * 9 + 0] = N0 - I0_init
        y[0, gi * 9 + 1] = I0_init

    # Vector ICs
    N_total0 = y[0, :18].sum()
    V_total0 = vector_density(p, t[0]) * N_total0
    y[0, 18] = (1.0 - init_vec_inf) * V_total0
    y[0, 19] = init_vec_inf * V_total0

    seed_idx = int(round((0.0 - t0) / dt_days))

    def y_past_factory(cur_idx: int) -> Callable[[float], np.ndarray]:
        def y_past(t_query: float) -> np.ndarray:
            if t_query <= t[0]:
                return y[0]
            pos = (t_query - t[0]) / dt_days
            i0 = int(math.floor(pos))
            if i0 >= cur_idx:
                return y[cur_idx]
            frac = pos - i0
            return (1.0 - frac) * y[i0] + frac * y[i0 + 1]
        return y_past

    for k in range(len(t) - 1):
        dt = dt_days
        y_past = y_past_factory(k)

        k1 = deriv(t[k], y[k], p, y_past)
        k2 = deriv(t[k] + dt / 2.0, y[k] + dt * k1 / 2.0, p, y_past)
        k3 = deriv(t[k] + dt / 2.0, y[k] + dt * k2 / 2.0, p, y_past)
        k4 = deriv(t[k] + dt, y[k] + dt * k3, p, y_past)

        y[k + 1] = np.clip(y[k] + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0, None)

        # Seed HIV at seed_year (move S -> Y1)
        if k + 1 == seed_idx:
            for (s_idx, y1_idx, seed) in ((0, 2, hiv_seed_gen), (9, 11, hiv_seed_core)):
                seed = min(seed, y[k + 1, s_idx])
                y[k + 1, s_idx] -= seed
                y[k + 1, y1_idx] += seed

    years = seed_year + t / 365.0
    return years, y


def prevalences(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    host = y[:, :18].reshape(-1, 2, 9)
    N = host.sum(axis=(1, 2))

    hiv = host[:, :, 2] + host[:, :, 3] + host[:, :, 4] + host[:, :, 5] + host[:, :, 6] + host[:, :, 7] + host[:, :, 8]
    mal = host[:, :, 1] + host[:, :, 3] + host[:, :, 5] + host[:, :, 6] + host[:, :, 8]
    dual = host[:, :, 3] + host[:, :, 5] + host[:, :, 6] + host[:, :, 8]

    return 100.0 * hiv.sum(axis=1) / N, 100.0 * mal.sum(axis=1) / N, 100.0 * dual.sum(axis=1) / N


# HIV data points used in the original Figure 1 reproduction script
ANC_POINTS = [
    (1990, 19.23), (1991, 20.04), (1992, 20.00), (1994, 30.40), (1995, 27.25),
    (1996, 27.30), (1997, 33.50), (1998, 30.61), (1999, 27.20), (2000, 35.00), (2001, 28.50),
]
POP_POINT = (1997.75, 26.0)


@st.cache_data(show_spinner=False)
def run_scenario(a: Assumptions, interaction: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = build_params(a, interaction=interaction)
    years, y = simulate(p)
    hiv, mal, dual = prevalences(y)
    mask = (years >= 1980.0) & (years <= 2010.0)
    return years[mask], hiv[mask], mal[mask], dual[mask]


def fig1_plot(
    years: np.ndarray,
    hiv_no: np.ndarray, mal_no: np.ndarray, dual_no: np.ndarray,
    hiv_int: np.ndarray, mal_int: np.ndarray, dual_int: np.ndarray,
    show_points: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)

    c_no, c_int = "black", "#b30000"
    lw = 2.4

    # Disease encodes linestyle; scenario encodes color.
    ax.plot(years, mal_no, color=c_no, lw=lw, ls="-")
    ax.plot(years, hiv_no, color=c_no, lw=lw, ls="--")
    ax.plot(years, dual_no, color=c_no, lw=lw, ls=":")

    ax.plot(years, mal_int, color=c_int, lw=lw, ls="-")
    ax.plot(years, hiv_int, color=c_int, lw=lw, ls="--")
    ax.plot(years, dual_int, color=c_int, lw=lw, ls=":")

    if show_points:
        xs, ys = zip(*ANC_POINTS)
        ax.scatter(xs, ys, marker="^", s=42, color="#1f77b4", edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter([POP_POINT[0]], [POP_POINT[1]], marker="D", s=54, color="#2ca02c", edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_xlim(1980, 2010)
    ax.set_ylim(0, 60)
    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Prevalence (%)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compact legends (slide-friendly)
    from matplotlib.lines import Line2D

    disease = [
        Line2D([0], [0], color="black", lw=lw, ls="-", label="Malaria"),
        Line2D([0], [0], color="black", lw=lw, ls="--", label="HIV"),
        Line2D([0], [0], color="black", lw=lw, ls=":", label="Dual infection"),
    ]
    scenario = [
        Line2D([0], [0], color=c_no, lw=lw, label="No interaction"),
        Line2D([0], [0], color=c_int, lw=lw, label="Interaction"),
    ]
    if show_points:
        scenario += [
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4", markeredgecolor="white",
                   markeredgewidth=0.6, markersize=8, label="ANC survey (Kisumu)"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#2ca02c", markeredgecolor="white",
                   markeredgewidth=0.6, markersize=8, label="Population survey (1997/98)"),
        ]

    leg1 = ax.legend(handles=disease, loc="upper left", frameon=False, fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=scenario, loc="lower right", frameon=False, fontsize=10)

    fig.tight_layout()
    return fig


def excess_bar_plot(hiv_no: np.ndarray, mal_no: np.ndarray, hiv_int: np.ndarray, mal_int: np.ndarray) -> Tuple[plt.Figure, float, float]:
    dhiv = hiv_int - hiv_no
    dmal = mal_int - mal_no
    hiv_excess = float(np.max(dhiv))
    mal_excess = float(np.max(dmal))

    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=200)
    ax.bar(["HIV", "Malaria"], [hiv_excess, mal_excess])
    ax.set_ylabel("Excess prevalence (percentage points)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(0.1, 1.15 * max(hiv_excess, mal_excess)))

    for x, v in zip([0, 1], [hiv_excess, mal_excess]):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    return fig, hiv_excess, mal_excess

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="HIV–Malaria Coupled Model (Abu-Raddad et al.)",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Main parameters")

    with st.expander("HIV natural history / transmission", expanded=False):
        p_act_acute = st.number_input("HIV per-act transmission (acute)", min_value=0.0, max_value=1.0, value=0.0107, format="%.4f")
        p_act_chronic = st.number_input("HIV per-act transmission (chronic)", min_value=0.0, max_value=1.0, value=0.0008, format="%.4f")
        p_act_advanced = st.number_input("HIV per-act transmission (advanced)", min_value=0.0, max_value=1.0, value=0.0042, format="%.4f")

        dur_acute_months = st.number_input("Acute stage duration (months)", min_value=0.5, max_value=24.0, value=2.5, step=0.1)
        dur_chronic_years = st.number_input("Chronic stage duration (years)", min_value=0.1, max_value=30.0, value=7.59, step=0.1)
        dur_advanced_years = st.number_input("Advanced stage duration (years)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

    with st.expander("Malaria → HIV infectiousness (viral load)", expanded=False):
        rr_per_log10_vl = st.number_input("RR per +1 log10 viral load", min_value=1.0, max_value=10.0, value=2.45, step=0.05)
        log_inc_chronic_clin = st.number_input("Δlog10 VL: chronic + clinical malaria", min_value=0.0, max_value=3.0, value=0.82, step=0.01)
        log_inc_chronic_nonclin = st.number_input("Δlog10 VL: chronic + non-clinical malaria", min_value=0.0, max_value=3.0, value=0.08, step=0.01)
        log_inc_advanced = st.number_input("Δlog10 VL: advanced + malaria", min_value=0.0, max_value=3.0, value=0.20, step=0.01)

    with st.expander("HIV → malaria susceptibility", expanded=False):
        sus_inc_chronic_pct = st.slider("Chronic HIV: susceptibility increase (%)", 0.0, 300.0, 44.0, 1.0)
        sus_inc_advanced_pct = st.slider("Advanced HIV: susceptibility increase (%)", 0.0, 500.0, 103.0, 1.0)

    with st.expander("Shared timescale / behavior", expanded=False):
        tau_gamet_days = st.number_input("Malaria infectious period / heightened VL (days)", min_value=7.0, max_value=200.0, value=42.0, step=1.0)
        red_sex_clin_pct = st.slider("Sex activity reduction: clinical malaria (%)", 0.0, 100.0, 10.0, 1.0)
        red_sex_nonclin_pct = st.slider("Sex activity reduction: non-clinical malaria (%)", 0.0, 100.0, 3.0, 1.0)

    with st.expander("Clinical malaria fractions", expanded=False):
        frac_clin_hiv_neg_pct = st.slider("Clinical fraction among HIV− (%)", 0.0, 100.0, 16.0, 1.0)
        frac_clin_hiv_pos_pct = st.slider("Clinical fraction among HIV+ (%)", 0.0, 100.0, 31.0, 1.0)

    with st.expander("Malaria → HIV mortality (advanced stage)", expanded=False):
        mort_inc_adv_coinf_pct = st.slider("Mortality increase in advanced co-infection (%)", 0.0, 100.0, 0.0, 1.0)

    show_points = st.checkbox("Show Kisumu HIV data points (as in Fig. 1)", value=False)

    a = Assumptions(
        p_act_acute=p_act_acute,
        p_act_chronic=p_act_chronic,
        p_act_advanced=p_act_advanced,
        dur_acute_months=dur_acute_months,
        dur_chronic_years=dur_chronic_years,
        dur_advanced_years=dur_advanced_years,
        rr_per_log10_vl=rr_per_log10_vl,
        log_inc_chronic_clin=log_inc_chronic_clin,
        log_inc_chronic_nonclin=log_inc_chronic_nonclin,
        log_inc_advanced=log_inc_advanced,
        sus_inc_chronic_pct=sus_inc_chronic_pct,
        sus_inc_advanced_pct=sus_inc_advanced_pct,
        tau_gamet_days=tau_gamet_days,
        red_sex_clin_pct=red_sex_clin_pct,
        red_sex_nonclin_pct=red_sex_nonclin_pct,
        frac_clin_hiv_neg_pct=frac_clin_hiv_neg_pct,
        frac_clin_hiv_pos_pct=frac_clin_hiv_pos_pct,
        mort_inc_adv_coinf_pct=mort_inc_adv_coinf_pct,
    )


st.title("HIV-Malaria dual infection model")
st.write("An implementation of the model by *Abu-Raddad, et at. (2006) Science, 314(5805), 1603-1606.*")

with st.spinner("Simulating model (no interaction vs interaction)…"):
    years_no, hiv_no, mal_no, dual_no = run_scenario(a, interaction=False)
    years_in, hiv_in, mal_in, dual_in = run_scenario(a, interaction=True)

# Main panel: two figures side-by-side (stay visible while scrolling sidebar)
col_plot, col_bar = st.columns([2.35, 1.0], gap="large")

with col_plot:
    st.subheader("Time evolution of HIV and Malaria prevalence")
    fig = fig1_plot(years_no, hiv_no, mal_no, dual_no, hiv_in, mal_in, dual_in, show_points=show_points)
    st.pyplot(fig, use_container_width=True)

with col_bar:
    st.subheader("Excess prevalence")
    fig_bar, hiv_excess, mal_excess = excess_bar_plot(hiv_no, mal_no, hiv_in, mal_in)
    st.pyplot(fig_bar, use_container_width=True)
    st.caption(f"Max (interaction − no interaction) over 1980–2010.\nHIV: {hiv_excess:.2f} pp, Malaria: {mal_excess:.2f} pp.")
