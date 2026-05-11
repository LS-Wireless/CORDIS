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
            los_cfg = cfg.sensing.los_model.lower().strip()
            if los_cfg == "always":
                los = True
            elif los_cfg == "never":
                los = False
            elif lsf.los_state_tg is not None:
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
# SCNR computation has moved to cordis/metrics/scnr.py
# =============================================================================
# The following functions now live in cordis.metrics.scnr:
#   - compute_scnr (replaces compute_expected_scnr / compute_network_scnr)
#   - compute_sensing_surrogate
# This separation keeps physical channel models in cordis/channel/ and
# performance metrics in cordis/metrics/.

