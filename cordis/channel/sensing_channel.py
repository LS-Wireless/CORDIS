"""
cordis/channel/sensing_channel.py
==================================
Multi-static STAP sensing channel model and expected SCNR computation
for the CORDIS simulation framework.

Implements Section II-B (Sensing and Clutter Channel Model) and
Proposition 3 (Expected STAP SCNR) from the CORDIS journal paper.

Sensing channel model
---------------------
For target t observed through transmit AP a_t and receive AP a_r:

    H_{a_t a_r}^sens[τ] = H_{a_t a_r}^tgt + H_{a_t a_r}^clt[τ]

Target channel (rank-one LoS):
    H_{a_t a_r}^tgt = s_t sqrt(β_{a_t a_r}^tgt) ζ_t a_{a_r} a_{a_t}^H
    where ζ_t ~ CN(0, σ_RCS^2)  [Swerling-I],  s_t ∈ {0, 1}  [LoS availability]

Clutter channel (separable space-time):
    H_{a_t a_r}^clt[τ] = σ_clt C_{a_r}^{1/2} Q[τ] C_{a_t}^{1/2}
    vec(Q[τ]) ~ CN(0, I),  temporal correlation ρ_clt(Δτ)

Expected STAP SCNR (Proposition 3)
-----------------------------------
After averaging over random transmit symbols:

    R_{g_{a_r}} = σ_clt^2 Σ_{a_t} tr(W_{a_t}^H C_{a_t} W_{a_t}) C_{a_r}
                  + σ_{n,a_r}^2 I_{M_r}

    SCNR_{a_r} = σ_RCS^2 T Σ_{a_t} β_{a_t a_r}^tgt ‖a_{a_t}^H W_{a_t}‖^2
                 × (a_{a_r}^H R_{g_{a_r}}^{-1} a_{a_r})

This expression decouples into:
  - Transmit beamforming gain:  Σ_{a_t} β ‖a_{a_t}^H W_{a_t}‖^2
  - Spatial STAP gain:          a_{a_r}^H R_{g_{a_r}}^{-1} a_{a_r}

The Doppler (temporal) processing gain T is a multiplicative factor,
so optimizing the spatial SCNR is equivalent to optimizing the full
joint STAP SCNR.

Note: The optimization objective in CORDIS uses the clutter-aware linear
surrogate rather than the exact SCNR fraction, for tractability.
This module provides both the exact SCNR (for evaluation) and the
quantities needed for the surrogate (for optimization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.utils.config import CORDISConfig
from cordis.utils.logger import get_logger
from cordis.utils.math_utils import (
    compute_wavelength, steering_vector, nearest_psd
)
from cordis.channel.topology import NetworkTopology, AccessPoint, SensingTarget
from cordis.channel.pathloss import LargeScaleFading, noise_power_watts

logger = get_logger(__name__)


# =============================================================================
# Clutter spatial correlation matrix  C_{a}  at direction (φ, θ)
# =============================================================================

def compute_clutter_spatial_correlation(
    array_type: str,
    n_ant: int,
    az_mean_deg: float,
    el_mean_deg: float,
    clutter_as_deg: float,
    wavelength: float,
    spacing_factor: float,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.complex128]:
    """
    Compute the spatial clutter covariance C_a ∈ C^{n_ant × n_ant} at AP a
    by integrating the steering vector outer product over the clutter angular
    spread, modeled as a truncated Laplacian PAS.

    C_a satisfies tr(C_a) = n_ant.

    Parameters
    ----------
    array_type : {"ULA", "UCA"}
    n_ant : int
    az_mean_deg, el_mean_deg : float
        Central clutter direction in degrees.
    clutter_as_deg : float
        Clutter angular spread standard deviation [°].
    wavelength, spacing_factor : float
    n_samples : int
    rng : np.random.Generator or None
        Uses np.random.default_rng() if None.

    Returns
    -------
    np.ndarray, shape (n_ant, n_ant), dtype complex128
    """
    from cordis.channel.rician import _sample_laplacian_angles
    if rng is None:
        rng = np.random.default_rng()

    az_rad = _sample_laplacian_angles(rng, n_samples, az_mean_deg, clutter_as_deg)
    el_rad = _sample_laplacian_angles(rng, n_samples, el_mean_deg, clutter_as_deg,
                                       clip_range_deg=89.0)
    el_rad = np.clip(el_rad, np.deg2rad(1.0), np.deg2rad(179.0))

    A = np.column_stack([
        np.sqrt(n_ant) * steering_vector(
            array_type, n_ant, az_rad[i], el_rad[i], wavelength, spacing_factor
        )
        for i in range(n_samples)
    ])
    C = (A @ A.conj().T) / n_samples
    C = (C + C.conj().T) / 2.0
    tr = np.real(np.trace(C))
    if tr > 1e-12:
        C *= n_ant / tr
    return nearest_psd(C).astype(np.complex128)


# =============================================================================
# SensingChannelStatistics  (second-order, computed once per topology)
# =============================================================================

@dataclass
class SensingChannelStatistics:
    """
    Pre-computed sensing channel statistics for all AP-pair/target combinations.

    Attributes
    ----------
    beta_bistatic : dict[(at_idx, ar_idx, tg_idx), float]
        Bi-static path loss β_{a_t a_r}^tgt in linear scale.
    a_tx : dict[(ap_idx, tg_idx), np.ndarray (Mt,)]
        Transmit steering vectors a_{a_t}(φ_t) toward each target.
        Un-normalised: ‖a‖^2 = Mt.
    a_rx : dict[(ap_idx, tg_idx), np.ndarray (Mr,)]
        Receive steering vectors a_{a_r}(φ_t) at each receive AP.
        Un-normalised: ‖a‖^2 = Mr.
    C_tx : dict[ap_idx, np.ndarray (Mt, Mt)]
        Transmit clutter covariance C_{a_t} at each TX AP.
    C_rx : dict[ap_idx, np.ndarray (Mr, Mr)]
        Receive clutter covariance C_{a_r} at each RX AP.
    los_state_tg : dict[(ap_idx, tg_idx), bool]
        LoS availability s_t for each AP-target pair.
    sigma_rcs_sq : float
        RCS power σ_RCS^2 (linear).
    sigma_clt_sq : float
        Clutter channel gain squared σ_clt^2.
    """

    beta_bistatic: Dict[Tuple[int, int, int], float]
    a_tx:          Dict[Tuple[int, int], NDArray[np.complex128]]
    a_rx:          Dict[Tuple[int, int], NDArray[np.complex128]]
    C_tx:          Dict[int, NDArray[np.complex128]]
    C_rx:          Dict[int, NDArray[np.complex128]]
    los_state_tg:  Dict[Tuple[int, int], bool]
    sigma_rcs_sq:  float
    sigma_clt_sq:  float


def compute_sensing_statistics(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    lsf: LargeScaleFading,
    rng: np.random.Generator,
) -> SensingChannelStatistics:
    """
    Pre-compute sensing channel statistics for all AP-pair/target combinations.

    Called once per topology realization.

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    lsf  : LargeScaleFading
    rng  : np.random.Generator

    Returns
    -------
    SensingChannelStatistics
    """
    s_cfg = cfg.sensing
    t_cfg = cfg.topology
    frq   = cfg.frequency

    lam     = frq.wavelength_m
    n_ant_t = t_cfg.n_ant   # Mt (all APs currently have the same array size)
    n_ant_r = t_cfg.n_ant   # Mr
    array_t = t_cfg.array_type
    spacing = t_cfg.antenna_spacing_factor

    sigma_rcs_sq = 10.0 ** (s_cfg.sigma_rcs_sq_db / 10.0)
    # σ_clt^2 from CNR: CNR = σ_clt^2 / σ_n^2
    sigma_n_sq   = noise_power_watts(frq.bandwidth_hz, cfg.channel.noise_figure_db)
    sigma_clt_sq = (10.0 ** (s_cfg.clutter_cnr_db / 10.0)) * sigma_n_sq

    beta_bistatic_dict: Dict[Tuple[int, int, int], float] = {}
    a_tx_dict:          Dict[Tuple[int, int], NDArray]    = {}
    a_rx_dict:          Dict[Tuple[int, int], NDArray]    = {}
    C_tx_dict:          Dict[int, NDArray]                = {}
    C_rx_dict:          Dict[int, NDArray]                = {}
    los_state_tg_dict:  Dict[Tuple[int, int], bool]      = {}

    # ── Clutter covariances at TX and RX APs ────────────────────────────────
    for ap in topo.tx_aps:
        # Clutter direction: use the centroid of all targets as a proxy
        if topo.n_targets > 0:
            tg_centroid = topo.target_positions.mean(axis=0)
            diff = tg_centroid - ap.pos
            az = float(np.arctan2(diff[1], diff[0]))
            el = float(np.arccos(np.clip(diff[2] / (np.linalg.norm(diff) + 1e-9), -1, 1)))
        else:
            az, el = 0.0, np.pi / 2

        C_tx_dict[ap.idx] = compute_clutter_spatial_correlation(
            array_t, n_ant_t,
            float(np.rad2deg(az)), float(np.rad2deg(el)),
            s_cfg.clutter_as_deg, lam, spacing,
            n_samples=500, rng=rng,
        )

    for ap in topo.rx_aps:
        if topo.n_targets > 0:
            tg_centroid = topo.target_positions.mean(axis=0)
            diff = tg_centroid - ap.pos
            az = float(np.arctan2(diff[1], diff[0]))
            el = float(np.arccos(np.clip(diff[2] / (np.linalg.norm(diff) + 1e-9), -1, 1)))
        else:
            az, el = 0.0, np.pi / 2

        C_rx_dict[ap.idx] = compute_clutter_spatial_correlation(
            array_t, n_ant_r,
            float(np.rad2deg(az)), float(np.rad2deg(el)),
            s_cfg.clutter_as_deg, lam, spacing,
            n_samples=500, rng=rng,
        )

    # ── Per-target, per-AP statistics ────────────────────────────────────────
    for tg_idx, tg in enumerate(topo.targets):
        for ap in topo.tx_aps:
            key_t = (ap.idx, tg_idx)

            # LoS state for this AP-target pair
            if lsf.los_state_tg is not None:
                los = bool(lsf.los_state_tg[ap.idx, tg_idx])
            else:
                los = True
            los_state_tg_dict[key_t] = los

            # Transmit steering vector toward target
            diff = tg.pos - ap.pos
            dist  = np.linalg.norm(diff) + 1e-9
            az_t  = float(np.arctan2(diff[1], diff[0]))
            el_t  = float(np.arccos(np.clip(diff[2] / dist, -1, 1)))
            a_at  = np.sqrt(n_ant_t) * steering_vector(
                array_t, n_ant_t, az_t, el_t, lam, spacing
            )
            a_tx_dict[key_t] = a_at.astype(np.complex128)

        for ap in topo.rx_aps:
            key_r = (ap.idx, tg_idx)

            # Receive steering vector toward target
            diff = tg.pos - ap.pos
            dist  = np.linalg.norm(diff) + 1e-9
            az_r  = float(np.arctan2(diff[1], diff[0]))
            el_r  = float(np.arccos(np.clip(diff[2] / dist, -1, 1)))
            a_ar  = np.sqrt(n_ant_r) * steering_vector(
                array_t, n_ant_r, az_r, el_r, lam, spacing
            )
            a_rx_dict[key_r] = a_ar.astype(np.complex128)

        # ── Bi-static path loss β_{a_t a_r}^tgt ─────────────────────────────
        for ap_t in topo.tx_aps:
            for ap_r in topo.rx_aps:
                # Use large-scale fading from pathloss.py for the two legs
                # β_bistatic = β_{a_t → target} × β_{target → a_r}
                # We approximate this from lsf.beta_tg_lin if available
                if lsf.beta_tg_lin is not None:
                    beta_at = lsf.beta_tg_lin[ap_t.idx, tg_idx]
                    beta_ar = lsf.beta_tg_lin[ap_r.idx, tg_idx]
                else:
                    # Fallback: free-space path loss
                    d_at = ap_t.distance_to(tg.pos)
                    d_ar = ap_r.distance_to(tg.pos)
                    lam_  = frq.wavelength_m
                    beta_at = (lam_ / (4 * np.pi * max(d_at, 1.0))) ** 2
                    beta_ar = (lam_ / (4 * np.pi * max(d_ar, 1.0))) ** 2

                # Bi-static PL: geometric mean of two legs
                beta_bi = np.sqrt(beta_at * beta_ar)
                beta_bistatic_dict[(ap_t.idx, ap_r.idx, tg_idx)] = float(beta_bi)

    logger.debug(
        "Sensing statistics computed: %d TX-RX pairs × %d targets, "
        "σ_RCS^2=%.2e, σ_clt^2=%.2e",
        len(topo.tx_aps) * len(topo.rx_aps), topo.n_targets,
        sigma_rcs_sq, sigma_clt_sq,
    )
    return SensingChannelStatistics(
        beta_bistatic=beta_bistatic_dict,
        a_tx=a_tx_dict,
        a_rx=a_rx_dict,
        C_tx=C_tx_dict,
        C_rx=C_rx_dict,
        los_state_tg=los_state_tg_dict,
        sigma_rcs_sq=sigma_rcs_sq,
        sigma_clt_sq=sigma_clt_sq,
    )


# =============================================================================
# Per-symbol clutter-plus-noise covariance  R_{g_{a_r}}
# =============================================================================

def compute_clutter_noise_covariance(
    ap_r: AccessPoint,
    W_tx: Dict[int, NDArray[np.complex128]],  # {at_idx: W_{a_t} (Mt, D)}
    sensing_stats: SensingChannelStatistics,
    sigma_n_sq: float,
    n_ant_r: int,
) -> NDArray[np.complex128]:
    """
    Compute the per-symbol clutter-plus-noise spatial covariance at
    receive AP a_r (Proposition 3, journal paper):

        R_{g_{a_r}} = σ_clt^2 Σ_{a_t} tr(W_{a_t}^H C_{a_t} W_{a_t}) C_{a_r}
                      + σ_{n,a_r}^2 I_{M_r}

    Parameters
    ----------
    ap_r : AccessPoint
        The receive AP.
    W_tx : dict[at_idx, np.ndarray (Mt, D)]
        Precoding matrices at all transmit APs.
    sensing_stats : SensingChannelStatistics
    sigma_n_sq : float
        Noise variance σ_{n,a_r}^2 at the receive AP.
    n_ant_r : int
        Number of receive antennas M_r.

    Returns
    -------
    np.ndarray, shape (M_r, M_r), dtype complex128
        Positive definite clutter-plus-noise covariance.
    """
    sigma_clt_sq = sensing_stats.sigma_clt_sq
    C_ar = sensing_stats.C_rx.get(ap_r.idx,
                                   np.eye(n_ant_r, dtype=complex))  # (Mr, Mr)

    # Σ_{a_t} tr(W_{a_t}^H C_{a_t} W_{a_t})
    total_clutter_power = 0.0
    for at_idx, W_at in W_tx.items():
        C_at = sensing_stats.C_tx.get(at_idx, np.eye(W_at.shape[0], dtype=complex))
        WCW = W_at.conj().T @ C_at @ W_at       # (D, D)
        total_clutter_power += float(np.real(np.trace(WCW)))

    R_g = sigma_clt_sq * total_clutter_power * C_ar + sigma_n_sq * np.eye(n_ant_r)
    return R_g.astype(np.complex128)


# =============================================================================
# Expected STAP SCNR  (Proposition 3 evaluation)
# =============================================================================

def compute_expected_scnr(
    ap_r: AccessPoint,
    tg_idx: int,
    W_tx: Dict[int, NDArray[np.complex128]],
    sensing_stats: SensingChannelStatistics,
    n_snapshots: int,
    sigma_n_sq: float,
    n_ant_r: int,
) -> float:
    """
    Evaluate the expected STAP SCNR at receive AP a_r for target t
    (Proposition 3, journal paper):

        SCNR_{a_r,t} = σ_RCS^2 × T
                       × Σ_{a_t} β_{a_t a_r}^tgt ‖a_{a_t}^H W_{a_t}‖^2
                       × (a_{a_r}^H R_{g_{a_r}}^{-1} a_{a_r})

    where T = n_snapshots is the temporal processing gain.

    Parameters
    ----------
    ap_r : AccessPoint
        Receive AP.
    tg_idx : int
        Target index.
    W_tx : dict[at_idx, np.ndarray (Mt, D)]
        Precoding matrices at all transmit APs.
    sensing_stats : SensingChannelStatistics
    n_snapshots : int
        T — number of slow-time sensing snapshots.
    sigma_n_sq : float
        Noise variance σ_{n,a_r}^2.
    n_ant_r : int
        Number of receive antennas M_r.

    Returns
    -------
    float
        Expected SCNR (linear scale, ≥ 0).
    """
    sigma_rcs_sq = sensing_stats.sigma_rcs_sq
    a_ar_key = (ap_r.idx, tg_idx)

    if a_ar_key not in sensing_stats.a_rx:
        return 0.0

    a_ar = sensing_stats.a_rx[a_ar_key]    # (Mr,), ‖a‖^2 = Mr

    # ── Clutter-plus-noise covariance R_{g_{a_r}} ─────────────────────────
    R_g = compute_clutter_noise_covariance(
        ap_r, W_tx, sensing_stats, sigma_n_sq, n_ant_r
    )

    # ── Spatial STAP gain: a_{a_r}^H R_{g}^{-1} a_{a_r} ──────────────────
    R_g_inv = np.linalg.solve(R_g, np.eye(n_ant_r))   # (Mr, Mr)
    spatial_gain = float(np.real(a_ar.conj() @ R_g_inv @ a_ar))

    # ── Transmit beamforming gain: Σ_{a_t} β_{a_t a_r}^tgt ‖a_{a_t}^H W_{a_t}‖^2
    tx_gain = 0.0
    for at_idx, W_at in W_tx.items():
        key_bi  = (at_idx, ap_r.idx, tg_idx)
        key_a_t = (at_idx, tg_idx)

        if key_bi not in sensing_stats.beta_bistatic:
            continue
        if key_a_t not in sensing_stats.a_tx:
            continue
        # LoS availability
        if not sensing_stats.los_state_tg.get(key_a_t, False):
            continue

        beta_bi = sensing_stats.beta_bistatic[key_bi]   # scalar
        a_at    = sensing_stats.a_tx[key_a_t]           # (Mt,)
        # ‖a_{a_t}^H W_{a_t}‖^2  (sum over all D beams)
        a_at_unit = a_at / np.sqrt(np.dot(a_at.conj(), a_at).real)
        beam_gain = float(np.real(
            np.linalg.norm(a_at_unit.conj() @ W_at) ** 2
        ))
        tx_gain += beta_bi * beam_gain

    scnr = sigma_rcs_sq * n_snapshots * tx_gain * spatial_gain
    return float(np.maximum(scnr, 0.0))


def compute_network_scnr(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    W_tx: Dict[int, NDArray[np.complex128]],
    sensing_stats: SensingChannelStatistics,
    sigma_n_sq: float,
) -> Dict[Tuple[int, int], float]:
    """
    Compute expected SCNR for every (receive AP, target) pair.

    Parameters
    ----------
    topo         : NetworkTopology
    cfg          : CORDISConfig
    W_tx         : dict[at_idx, np.ndarray (Mt, D)]
    sensing_stats: SensingChannelStatistics
    sigma_n_sq   : float

    Returns
    -------
    dict[(ar_idx, tg_idx), float]
        SCNR in linear scale for each receive AP / target combination.
    """
    n_snapshots = cfg.sensing.n_snapshots
    n_ant_r     = cfg.topology.n_ant

    scnr_dict: Dict[Tuple[int, int], float] = {}
    for ap_r in topo.rx_aps:
        for tg_idx in range(topo.n_targets):
            scnr = compute_expected_scnr(
                ap_r, tg_idx, W_tx, sensing_stats,
                n_snapshots, sigma_n_sq, n_ant_r,
            )
            scnr_dict[(ap_r.idx, tg_idx)] = scnr

    return scnr_dict


# =============================================================================
# Clutter-aware linear sensing surrogate  (used in CORDIS optimization)
# =============================================================================

def compute_sensing_surrogate(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    W_tx: Dict[int, NDArray[np.complex128]],
    sensing_stats: SensingChannelStatistics,
    omega: Optional[Dict[int, float]] = None,
) -> float:
    """
    Evaluate the clutter-aware linear sensing surrogate used as the
    optimization objective in CORDIS-Split and CORDIS-ADMM
    (eq. admm-linear-objective, journal paper):

        U_cpu^sens = Σ_{a_t} [ Σ_{t} ω̄_t β̄_{a_t}^t ‖a_{a_t t}^H W_{a_t}‖^2
                               - κ tr(W_{a_t}^H C_{a_t} W_{a_t}) ]

    where  ω̄_t = ω_t T σ_RCS^2  and  β̄_{a_t}^t = Σ_{a_r} β_{a_t a_r}^t.

    Parameters
    ----------
    topo         : NetworkTopology
    cfg          : CORDISConfig
    W_tx         : dict[at_idx, np.ndarray (Mt, D)]
    sensing_stats: SensingChannelStatistics
    omega        : dict[tg_idx, float] or None
        Target priority weights ω_t.  Equal weights (ω_t = 1) if None.

    Returns
    -------
    float
        Scalar value of the sensing surrogate.
    """
    kappa       = cfg.algorithm.admm.kappa
    n_snapshots = cfg.sensing.n_snapshots
    sigma_rcs_sq = sensing_stats.sigma_rcs_sq

    if omega is None:
        omega = {tg_idx: 1.0 for tg_idx in range(topo.n_targets)}

    total = 0.0
    for at_idx, W_at in W_tx.items():
        C_at = sensing_stats.C_tx.get(at_idx,
               np.eye(W_at.shape[0], dtype=complex))

        # Clutter penalty: κ tr(W_{a_t}^H C_{a_t} W_{a_t})
        clutter_penalty = kappa * float(np.real(np.trace(
            W_at.conj().T @ C_at @ W_at
        )))

        # Target echo power: Σ_t ω̄_t β̄_{a_t}^t ‖a_{a_t}^H W_{a_t}‖^2
        echo_power = 0.0
        for tg_idx in range(topo.n_targets):
            w_t  = omega.get(tg_idx, 1.0)
            omega_bar = w_t * n_snapshots * sigma_rcs_sq

            # β̄_{a_t}^t = Σ_{a_r} β_{a_t a_r}^t
            beta_bar = sum(
                sensing_stats.beta_bistatic.get((at_idx, ap_r.idx, tg_idx), 0.0)
                for ap_r in topo.rx_aps
            )

            a_at_key = (at_idx, tg_idx)
            if a_at_key not in sensing_stats.a_tx:
                continue
            if not sensing_stats.los_state_tg.get(a_at_key, False):
                continue

            a_at = sensing_stats.a_tx[a_at_key]
            a_at_unit = a_at / np.sqrt(np.dot(a_at.conj(), a_at).real)
            beam_gain = float(np.real(
                np.linalg.norm(a_at_unit.conj() @ W_at) ** 2
            ))
            echo_power += omega_bar * beta_bar * beam_gain

        total += echo_power - clutter_penalty

    return total

