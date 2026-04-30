"""
cordis/channel/pathloss.py
==========================
3GPP TR 38.901-compliant path loss and LoS probability models for the
CORDIS simulation framework.

Supported scenarios
-------------------
UMi-StreetCanyon
    Primary scenario for the CORDIS paper.  Valid for outdoor sub-6 GHz
    and mmWave frequencies (0.5–100 GHz).

UMa (Urban Macro)
    Provided for comparison / sensitivity analysis.

The module also implements:
- Distance-dependent LoS probability (3GPP Table 7.4.2-1)
- Shadow fading (log-normal, scenario-dependent σ values)
- Correlated shadow fading across UEs (exponential spatial correlation)
- Large-scale fading coefficient β_{a,u} = PL × shadow fading

References
----------
3GPP TR 38.901 V17.0.0 (Jun. 2022), Tables 7.4.1-1, 7.4.2-1, 7.5-6.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.utils.config import CORDISConfig
from cordis.utils.logger import get_logger
from cordis.channel.topology import NetworkTopology

logger = get_logger(__name__)


# =============================================================================
# Constants (3GPP TR 38.901)
# =============================================================================

# Breakpoint distance parameters for UMi-StreetCanyon LoS
_UMI_DBP_COEFF = 4.0           # d_BP = 4 h_BS h_UT f_c / c  [TR eq. 7.4-1]
_SPEED_OF_LIGHT = 3.0e8        # m/s

# Shadow fading standard deviations [dB] (TR Table 7.4.1-1)
_SHADOW_STD = {
    "UMi": {"LoS": 4.0,  "NLoS": 7.82},
    "UMa": {"LoS": 4.0,  "NLoS": 6.0},
    "RMa": {"LoS": 4.0,  "NLoS": 8.0},
}

# Minimum 2-D distance constraint (UMi Table 7.4.1-1, note 1)
_D2D_MIN = 10.0                # metres


# =============================================================================
# LoS probability
# =============================================================================

def los_probability_umi(d_2d: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    3GPP TR 38.901 Table 7.4.2-1 — UMi-StreetCanyon LoS probability.

        P_LoS = min(18/d, 1) * (1 − exp(−d/36)) + exp(−d/36)

    Parameters
    ----------
    d_2d : np.ndarray
        2-D link distance(s) in metres.  Must be > 0.

    Returns
    -------
    np.ndarray
        P_LoS ∈ [0, 1] for each input distance.
    """
    d = np.asarray(d_2d, dtype=float)
    d = np.maximum(d, 1e-3)    # avoid division by zero
    p = np.minimum(18.0 / d, 1.0) * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)
    return np.clip(p, 0.0, 1.0)


def los_probability_uma(d_2d: NDArray[np.float64],
                        h_ut: float = 1.5) -> NDArray[np.float64]:
    """
    3GPP TR 38.901 Table 7.4.2-1 — UMa LoS probability.

        C(d, h_UT) = ((1.25e-6) d² exp(−d/150))^3  if h_UT ≤ 13 m
        P_LoS = (min(18/d, 1) * (1−exp(−d/63)) + exp(−d/63)) * (1 + C)

    Simplified version (h_UT ≤ 13 m assumed for UE).
    """
    d = np.asarray(d_2d, dtype=float)
    d = np.maximum(d, 1e-3)
    C = ((1.25e-6) * d**2 * np.exp(-d / 150.0))**3
    base = np.minimum(18.0 / d, 1.0) * (1.0 - np.exp(-d / 63.0)) + np.exp(-d / 63.0)
    return np.clip(base * (1.0 + C), 0.0, 1.0)


def sample_los_state(
    p_los: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """
    Draw independent Bernoulli LoS/NLoS realisations.

    Parameters
    ----------
    p_los : np.ndarray
        Array of LoS probabilities (any shape).
    rng : np.random.Generator

    Returns
    -------
    np.ndarray[bool]
        ``True`` = LoS,  ``False`` = NLoS.  Same shape as ``p_los``.
    """
    return rng.random(p_los.shape) < p_los


# =============================================================================
# Path loss models (deterministic, no shadow fading)
# =============================================================================

def _breakpoint_distance_umi(
    fc_hz: float,
    h_bs: float = 10.0,
    h_ut: float = 1.5,
) -> float:
    """
    3GPP UMi effective environment height is h_E = 1 m.

        h_BS' = h_BS − h_E,   h_UT' = h_UT − h_E
        d_BP  = 4 h_BS' h_UT' f_c / c

    Returns
    -------
    float
        Breakpoint distance in metres.
    """
    h_e  = 1.0
    hbsp = h_bs - h_e
    hutp = h_ut - h_e
    return 4.0 * hbsp * hutp * fc_hz / _SPEED_OF_LIGHT


def path_loss_umi_los(
    d_2d: NDArray[np.float64],
    d_3d: NDArray[np.float64],
    fc_hz: float,
    h_bs: float = 10.0,
    h_ut: float = 1.5,
) -> NDArray[np.float64]:
    """
    3GPP TR 38.901 Table 7.4.1-1 — UMi-StreetCanyon **LoS** path loss [dB].

    Two-slope model with breakpoint distance d_BP:

        PL1 = 32.4 + 21 log10(d_3D) + 20 log10(f_c[GHz])
                                               for 10 m ≤ d_2D ≤ d_BP

        PL2 = 32.4 + 40 log10(d_3D) + 20 log10(f_c[GHz])
              − 9.5 log10(d_BP² + (h_BS − h_UT)²)
                                               for d_BP < d_2D ≤ 5000 m

    Parameters
    ----------
    d_2d : np.ndarray
        2-D link distance(s) in metres.
    d_3d : np.ndarray
        3-D link distance(s) in metres.
    fc_hz : float
        Carrier frequency in Hz.
    h_bs, h_ut : float
        BS (AP) and UT (UE) heights in metres.

    Returns
    -------
    np.ndarray
        Path loss in dB (positive values, larger = more loss).
    """
    d2 = np.maximum(np.asarray(d_2d, dtype=float), _D2D_MIN)
    d3 = np.maximum(np.asarray(d_3d, dtype=float), _D2D_MIN)
    fc_ghz = fc_hz / 1e9

    d_bp = _breakpoint_distance_umi(fc_hz, h_bs, h_ut)
    h_diff = h_bs - h_ut

    # PL1
    pl1 = 32.4 + 21.0 * np.log10(d3) + 20.0 * np.log10(fc_ghz)

    # PL2
    pl2 = (32.4 + 40.0 * np.log10(d3) + 20.0 * np.log10(fc_ghz)
           - 9.5 * np.log10(d_bp**2 + h_diff**2))

    return np.where(d2 <= d_bp, pl1, pl2)


def path_loss_umi_nlos(
    d_2d: NDArray[np.float64],
    d_3d: NDArray[np.float64],
    fc_hz: float,
    h_bs: float = 10.0,
    h_ut: float = 1.5,
) -> NDArray[np.float64]:
    """
    3GPP TR 38.901 Table 7.4.1-1 — UMi-StreetCanyon **NLoS** path loss [dB].

        PL_NLoS = max(PL_UMi-LoS, PL'_NLoS)

        PL'_NLoS = 35.3 log10(d_3D) + 22.4 + 21.3 log10(f_c[GHz])
                   − 0.3 (h_UT − 1.5)

    Parameters
    ----------
    (same as :func:`path_loss_umi_los`)

    Returns
    -------
    np.ndarray
        Path loss in dB.
    """
    d2 = np.maximum(np.asarray(d_2d, dtype=float), _D2D_MIN)
    d3 = np.maximum(np.asarray(d_3d, dtype=float), _D2D_MIN)
    fc_ghz = fc_hz / 1e9

    pl_los  = path_loss_umi_los(d2, d3, fc_hz, h_bs, h_ut)
    pl_nlos_prime = (35.3 * np.log10(d3) + 22.4
                     + 21.3 * np.log10(fc_ghz)
                     - 0.3 * (h_ut - 1.5))
    return np.maximum(pl_los, pl_nlos_prime)


def path_loss_uma_los(
    d_2d: NDArray[np.float64],
    d_3d: NDArray[np.float64],
    fc_hz: float,
    h_bs: float = 25.0,
    h_ut: float = 1.5,
) -> NDArray[np.float64]:
    """
    3GPP TR 38.901 Table 7.4.1-1 — UMa **LoS** path loss [dB].

        h_E ~ Uniform({12, …, h_UT−1.5})  (simplified: h_E = 1 m here)
        d_BP = 4 h_BS' h_UT' f_c / c

        PL1 = 28.0 + 22 log10(d_3D) + 20 log10(f_c[GHz])
        PL2 = 28.0 + 40 log10(d_3D) + 20 log10(f_c[GHz])
              − 9 log10(d_BP² + (h_BS−h_UT)²)
    """
    d2 = np.maximum(np.asarray(d_2d, dtype=float), _D2D_MIN)
    d3 = np.maximum(np.asarray(d_3d, dtype=float), _D2D_MIN)
    fc_ghz = fc_hz / 1e9

    h_e  = 1.0
    hbsp = h_bs - h_e
    hutp = h_ut - h_e
    d_bp = 4.0 * hbsp * hutp * fc_hz / _SPEED_OF_LIGHT
    h_diff = h_bs - h_ut

    pl1 = 28.0 + 22.0 * np.log10(d3) + 20.0 * np.log10(fc_ghz)
    pl2 = (28.0 + 40.0 * np.log10(d3) + 20.0 * np.log10(fc_ghz)
           - 9.0 * np.log10(d_bp**2 + h_diff**2))
    return np.where(d2 <= d_bp, pl1, pl2)


def path_loss_uma_nlos(
    d_2d: NDArray[np.float64],
    d_3d: NDArray[np.float64],
    fc_hz: float,
    h_bs: float = 25.0,
    h_ut: float = 1.5,
) -> NDArray[np.float64]:
    """
    3GPP TR 38.901 Table 7.4.1-1 — UMa **NLoS** path loss [dB].

        PL_NLoS = max(PL_UMa-LoS,
                      13.54 + 39.08 log10(d_3D) + 20 log10(f_c[GHz])
                      − 0.6 (h_UT − 1.5))
    """
    d2 = np.maximum(np.asarray(d_2d, dtype=float), _D2D_MIN)
    d3 = np.maximum(np.asarray(d_3d, dtype=float), _D2D_MIN)
    fc_ghz = fc_hz / 1e9

    pl_los      = path_loss_uma_los(d2, d3, fc_hz, h_bs, h_ut)
    pl_nlos_pr  = (13.54 + 39.08 * np.log10(d3)
                   + 20.0 * np.log10(fc_ghz)
                   - 0.6 * (h_ut - 1.5))
    return np.maximum(pl_los, pl_nlos_pr)


# =============================================================================
# Shadow fading
# =============================================================================

def _shadow_std(scenario: str, is_los: bool) -> float:
    state = "LoS" if is_los else "NLoS"
    return _SHADOW_STD.get(scenario, _SHADOW_STD["UMi"])[state]


def _correlated_shadow_fading(
    ue_positions_2d: NDArray[np.float64],
    std_db: float,
    d_corr: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Generate spatially correlated shadow fading samples for N_ue UEs
    (viewed from a single AP).

    The covariance between UEs j and k is

        Cov(ψ_j, ψ_k) = σ² exp(−d_{j,k} / D_corr)

    where d_{j,k} is the 2-D inter-UE distance and D_corr is the
    decorrelation distance (default 50 m per 3GPP).

    Parameters
    ----------
    ue_positions_2d : np.ndarray, shape (N_ue, 2)
    std_db : float
        Shadow fading standard deviation σ [dB].
    d_corr : float
        Decorrelation distance D_corr [m].
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (N_ue,)
        Correlated shadow fading realizations in dB.
    """
    n_ue = len(ue_positions_2d)
    if n_ue == 1:
        return rng.normal(0.0, std_db, size=1)

    # Inter-UE distance matrix
    diff = (ue_positions_2d[:, np.newaxis, :]
            - ue_positions_2d[np.newaxis, :, :])    # (N, N, 2)
    d_ij = np.linalg.norm(diff, axis=-1)             # (N, N)

    # Correlation matrix
    R = std_db**2 * np.exp(-d_ij / d_corr)          # (N, N)

    # Cholesky decomposition (add small jitter for numerical stability)
    jitter = 1e-8 * np.eye(n_ue)
    try:
        L = np.linalg.cholesky(R + jitter)
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue decomposition if Cholesky fails
        eigvals, eigvecs = np.linalg.eigh(R + jitter)
        eigvals = np.maximum(eigvals, 0.0)
        L = eigvecs * np.sqrt(eigvals)

    return L @ rng.standard_normal(n_ue)


# =============================================================================
# Large-scale fading (path loss + shadow fading)
# =============================================================================

@dataclass
class LargeScaleFading:
    """
    Container for pre-computed large-scale fading coefficients.

    Attributes
    ----------
    beta_db : np.ndarray, shape (N_ap, N_ue)
        Large-scale fading coefficient β_{a,u} in dB for all AP-UE pairs.
        β = path_loss [dB] + shadow_fading [dB]  (both positive → more loss)
    beta_lin : np.ndarray, shape (N_ap, N_ue)
        β in linear scale (β_lin = 10^(−β_db/10) so larger = stronger link).
    los_state : np.ndarray[bool], shape (N_ap, N_ue)
        LoS (True) / NLoS (False) state for each link.
    path_loss_db : np.ndarray, shape (N_ap, N_ue)
        Deterministic path loss component [dB].
    shadow_db : np.ndarray, shape (N_ap, N_ue)
        Shadow fading component [dB].

    For sensing (AP-target links):
    beta_tg_db : np.ndarray, shape (N_ap, N_tg)  or None
    beta_tg_lin : np.ndarray, shape (N_ap, N_tg)  or None
    los_state_tg : np.ndarray[bool], shape (N_ap, N_tg)  or None
    """

    # AP-UE links
    beta_db: NDArray[np.float64]         # (N_ap, N_ue)
    beta_lin: NDArray[np.float64]        # (N_ap, N_ue)
    los_state: NDArray[np.bool_]         # (N_ap, N_ue)
    path_loss_db: NDArray[np.float64]    # (N_ap, N_ue)
    shadow_db: NDArray[np.float64]       # (N_ap, N_ue)

    # AP-target links (optional)
    beta_tg_db: Optional[NDArray[np.float64]] = None    # (N_ap, N_tg)
    beta_tg_lin: Optional[NDArray[np.float64]] = None   # (N_ap, N_tg)
    los_state_tg: Optional[NDArray[np.bool_]] = None    # (N_ap, N_tg)


def compute_large_scale_fading(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    rng: np.random.Generator,
    include_targets: bool = True,
) -> LargeScaleFading:
    """
    Compute large-scale fading coefficients β_{a,u} for all AP-UE
    (and optionally AP-target) links.

    The computation follows the 3GPP TR 38.901 UMi-StreetCanyon model:

    1. Determine LoS probability from distance.
    2. Draw Bernoulli LoS/NLoS state.
    3. Compute deterministic path loss (LoS or NLoS formula).
    4. Add spatially correlated (across UEs) shadow fading.
    5. Compute β_lin = 10^(−β_db / 10) for use in channel generation.

    Parameters
    ----------
    topo : NetworkTopology
    cfg : CORDISConfig
    rng : np.random.Generator
    include_targets : bool
        If True, also compute AP-target large-scale fading.

    Returns
    -------
    LargeScaleFading
    """
    ch  = cfg.channel
    t   = cfg.topology
    frq = cfg.frequency

    fc_hz = frq.carrier_freq_hz
    scenario = ch.scenario           # "UMi" | "UMa"
    d_corr   = ch.shadow_corr_distance_m

    n_ap = topo.n_ap
    n_ue = topo.n_ue

    d_2d = topo.ap_ue_distances_2d()   # (N_ap, N_ue)
    d_3d = topo.ap_ue_distances_3d()   # (N_ap, N_ue)

    ue_pos_2d = topo.ue_positions[:, :2]    # (N_ue, 2)

    # ── LoS probability ───────────────────────────────────────────────────
    if scenario.startswith("UMi"):
        p_los = los_probability_umi(d_2d)   # (N_ap, N_ue)
    else:
        p_los = los_probability_uma(d_2d, h_ut=t.h_ue_m)

    los_state = sample_los_state(p_los, rng)   # (N_ap, N_ue) bool

    # ── Deterministic path loss ───────────────────────────────────────────
    pl_los_all  = path_loss_umi_los(d_2d, d_3d, fc_hz, t.h_ap_m, t.h_ue_m) \
                  if scenario.startswith("UMi") else \
                  path_loss_uma_los(d_2d, d_3d, fc_hz, t.h_ap_m, t.h_ue_m)

    pl_nlos_all = path_loss_umi_nlos(d_2d, d_3d, fc_hz, t.h_ap_m, t.h_ue_m) \
                  if scenario.startswith("UMi") else \
                  path_loss_uma_nlos(d_2d, d_3d, fc_hz, t.h_ap_m, t.h_ue_m)

    path_loss_db = np.where(los_state, pl_los_all, pl_nlos_all)   # (N_ap, N_ue)

    # ── Shadow fading (correlated across UEs, independent across APs) ─────
    shadow_db = np.zeros((n_ap, n_ue), dtype=float)
    for a in range(n_ap):
        # Use per-AP LoS/NLoS mix for σ (take majority state as proxy)
        n_los = int(los_state[a].sum())
        mostly_los = n_los >= n_ue / 2
        std_los  = _shadow_std(scenario, True)
        std_nlos = _shadow_std(scenario, False)

        # Draw correlated shadow fading for LoS UEs
        # and independently for NLoS UEs at this AP
        sf_los  = _correlated_shadow_fading(ue_pos_2d, std_los,  d_corr, rng)
        sf_nlos = _correlated_shadow_fading(ue_pos_2d, std_nlos, d_corr, rng)
        shadow_db[a] = np.where(los_state[a], sf_los, sf_nlos)

    # ── Combine: β [dB] = PL + SF  (positive = more loss) ───────────────
    beta_db  = path_loss_db + shadow_db                             # (N_ap, N_ue)
    beta_lin = 10.0 ** (-beta_db / 10.0)                           # (N_ap, N_ue)

    logger.debug(
        "Large-scale fading: β_lin mean=%.2e, min=%.2e, max=%.2e  "
        "LoS fraction=%.2f",
        beta_lin.mean(), beta_lin.min(), beta_lin.max(),
        los_state.mean(),
    )

    # ── AP-target links ───────────────────────────────────────────────────
    beta_tg_db  = None
    beta_tg_lin = None
    los_state_tg = None

    if include_targets and topo.n_targets > 0:
        n_tg = topo.n_targets
        d_tg_2d = np.zeros((n_ap, n_tg))
        d_tg_3d = np.zeros((n_ap, n_tg))
        for a_idx, ap in enumerate(topo.aps):
            for tg_idx, tg in enumerate(topo.targets):
                d_tg_2d[a_idx, tg_idx] = ap.distance_2d_to(tg.pos)
                d_tg_3d[a_idx, tg_idx] = ap.distance_to(tg.pos)

        if scenario.startswith("UMi"):
            p_los_tg = los_probability_umi(d_tg_2d)
            pl_los_tg  = path_loss_umi_los(d_tg_2d, d_tg_3d, fc_hz,
                                           t.h_ap_m, t.h_tg_m)
            pl_nlos_tg = path_loss_umi_nlos(d_tg_2d, d_tg_3d, fc_hz,
                                            t.h_ap_m, t.h_tg_m)
        else:
            p_los_tg = los_probability_uma(d_tg_2d, h_ut=t.h_tg_m)
            pl_los_tg  = path_loss_uma_los(d_tg_2d, d_tg_3d, fc_hz,
                                           t.h_ap_m, t.h_tg_m)
            pl_nlos_tg = path_loss_uma_nlos(d_tg_2d, d_tg_3d, fc_hz,
                                            t.h_ap_m, t.h_tg_m)

        los_state_tg = sample_los_state(p_los_tg, rng)     # (N_ap, N_tg)
        pl_tg = np.where(los_state_tg, pl_los_tg, pl_nlos_tg)

        # Shadow fading for targets (independent, scalar σ)
        std_tg = _shadow_std(scenario, True)  # assume LoS for targets
        sf_tg = rng.normal(0.0, std_tg, size=(n_ap, n_tg))
        beta_tg_db  = pl_tg + sf_tg
        beta_tg_lin = 10.0 ** (-beta_tg_db / 10.0)

        logger.debug(
            "Target large-scale fading: β_tg_lin mean=%.2e, LoS frac=%.2f",
            beta_tg_lin.mean(), los_state_tg.mean(),
        )

    return LargeScaleFading(
        beta_db=beta_db,
        beta_lin=beta_lin,
        los_state=los_state,
        path_loss_db=path_loss_db,
        shadow_db=shadow_db,
        beta_tg_db=beta_tg_db,
        beta_tg_lin=beta_tg_lin,
        los_state_tg=los_state_tg,
    )


# =============================================================================
# Noise power helper
# =============================================================================

def noise_power_watts(
    bandwidth_hz: float,
    noise_figure_db: float = 7.0,
    temp_k: float = 290.0,
) -> float:
    """
    Thermal noise power:  N_0 = k_B T B × NF

    Parameters
    ----------
    bandwidth_hz : float
        System bandwidth (or subcarrier bandwidth) in Hz.
    noise_figure_db : float
        Receiver noise figure in dB.
    temp_k : float
        Noise temperature in Kelvin.

    Returns
    -------
    float
        Noise power in Watts.
    """
    k_b = 1.380649e-23           # Boltzmann constant [J/K]
    nf_lin = 10.0 ** (noise_figure_db / 10.0)
    return k_b * temp_k * bandwidth_hz * nf_lin


def snr_to_tx_power(
    snr_db: float,
    noise_power_w: float,
) -> float:
    """
    Compute the transmit power P_max (Watts) corresponding to a given
    P_max / σ²_z ratio in dB.

    Parameters
    ----------
    snr_db : float
        P_max / σ²_z in dB.
    noise_power_w : float
        Noise power σ²_z in Watts.

    Returns
    -------
    float
        P_max in Watts.
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    return snr_lin * noise_power_w

