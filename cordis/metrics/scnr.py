"""
cordis/metrics/scnr.py
======================
SCNR computation for the CORDIS framework.

Implements the expected STAP SCNR from Proposition 3 (Section III-B):

    SCNR_{a_r,t} = σ_RCS² · T · [ Σ_{a_t} β_{a_t a_r}^tgt · ‖a_{a_t}^H W_{a_t}‖² ]
                              · ( a_{a_r}^H R_{g_{a_r}}^{-1} a_{a_r} )

where:
    R_{g_{a_r}} = σ_clt² · Σ_{a_t} tr(W_{a_t}^H C_{a_t} W_{a_t}) · C_{a_r}
                  + σ_n² · I_{M_r}

and the steering vectors satisfy ‖a_{a_t}‖² = M_t,  ‖a_{a_r}‖² = M_r.

The summation over transmit APs is restricted to the bistatic links
defined by the :class:`SensingAssociation` — i.e., for each receive AP
a_r and target t, only the transmit APs explicitly assigned to
(a_t, a_r, t) contribute to the target echo.  The clutter covariance,
however, integrates contributions from ALL active transmit APs because
clutter is a broadcast phenomenon.

This module also provides the clutter-aware linear sensing surrogate
(eq. admm-linear-objective) that is used as the optimization objective
in CORDIS-Split (P-Split) and CORDIS-ADMM (P-Local).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.utils.config import CORDISConfig
from cordis.utils.logger import get_logger
from cordis.channel.topology import NetworkTopology, SensingAssociation
from cordis.channel.sensing_channel import (
    SensingChannelStatistics,
    compute_clutter_noise_covariance,
)

logger = get_logger(__name__)


# =============================================================================
# Per-trial metrics
# =============================================================================

@dataclass
class SCNRMetrics:
    """
    Per-trial SCNR metrics for a single channel realization.

    Attributes
    ----------
    scnr_per_pair : dict[(ar_idx, tg_idx), float]
        Expected STAP SCNR (linear scale) for every assigned
        (receive AP, target) pair.
    weights : dict[tg_idx, float]
        Target priority weights ω_t used in the weighted-sum aggregate.
    surrogate : float
        Value of the clutter-aware linear sensing surrogate
        (eq. admm-linear-objective) — same quantity that the CORDIS
        algorithms maximize.

    Derived properties expose:
      - per-pair SCNR in dB
      - per-target SCNR (sum over receive APs)
      - weighted sum, min, mean (linear and dB)
    """

    scnr_per_pair:  Dict[Tuple[int, int], float]   # (ar_idx, tg_idx) -> linear
    weights:        Dict[int, float]               # tg_idx -> ω_t
    surrogate:      float

    # ── Per-pair / per-target derived ────────────────────────────────────

    @property
    def scnr_per_pair_db(self) -> Dict[Tuple[int, int], float]:
        return {k: 10.0 * np.log10(max(v, 1e-30))
                for k, v in self.scnr_per_pair.items()}

    @property
    def scnr_per_target(self) -> Dict[int, float]:
        """Total SCNR per target, summed across all receiving APs."""
        out: Dict[int, float] = {}
        for (_ar, tg), v in self.scnr_per_pair.items():
            out[tg] = out.get(tg, 0.0) + v
        return out

    @property
    def scnr_per_target_db(self) -> Dict[int, float]:
        return {tg: 10.0 * np.log10(max(v, 1e-30))
                for tg, v in self.scnr_per_target.items()}

    # ── Scalar aggregates ────────────────────────────────────────────────

    @property
    def weighted_sum_scnr(self) -> float:
        """Σ_{a_r, t} ω_t · SCNR_{a_r, t}  (linear)."""
        return float(sum(
            self.weights.get(tg, 1.0) * v
            for (_ar, tg), v in self.scnr_per_pair.items()
        ))

    @property
    def weighted_sum_scnr_db(self) -> float:
        return 10.0 * np.log10(max(self.weighted_sum_scnr, 1e-30))

    @property
    def min_scnr(self) -> float:
        if not self.scnr_per_pair:
            return 0.0
        return float(min(self.scnr_per_pair.values()))

    @property
    def min_scnr_db(self) -> float:
        return 10.0 * np.log10(max(self.min_scnr, 1e-30))

    @property
    def mean_scnr(self) -> float:
        if not self.scnr_per_pair:
            return 0.0
        return float(np.mean(list(self.scnr_per_pair.values())))

    @property
    def mean_scnr_db(self) -> float:
        return 10.0 * np.log10(max(self.mean_scnr, 1e-30))


# =============================================================================
# Main computation
# =============================================================================

def compute_scnr(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    W_tx: Dict[int, NDArray[np.complex128]],
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    sigma_n_sq: float,
    omega: Optional[Dict[int, float]] = None,
) -> SCNRMetrics:
    """
    Compute expected STAP SCNR (Proposition 3) for every assigned
    (receive AP, target) pair, plus the clutter-aware linear surrogate.

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    W_tx : dict[ap_idx, np.ndarray (Mt, N_ue + N_t)]
        Precoding matrices at all transmit APs.
    sensing_stats : SensingChannelStatistics
        Pre-computed sensing channel statistics from
        ``compute_sensing_statistics``.
    association : SensingAssociation
        Per-target AP association produced by
        ``cordis.channel.sensing_assignment.assign_sensing``.  The SCNR
        is computed only for the bistatic pairs (a_t, a_r, t) that are
        present in this association.
    sigma_n_sq : float
        Noise variance σ_n² at the receive APs.
    omega : dict[tg_idx, float] or None
        Target priority weights ω_t.  If None, equal weights (ω_t = 1)
        are used for all targets.

    Returns
    -------
    SCNRMetrics
    """
    n_ant_r       = cfg.topology.n_ant
    T_snapshots   = cfg.sensing.n_snapshots
    sigma_rcs_sq  = sensing_stats.sigma_rcs_sq

    if omega is None:
        omega = {tg: 1.0 for tg in range(topo.n_targets)}

    # ── Pre-compute R_{g_{a_r}} for each receive AP ──────────────────────
    # R_g uses ALL active transmit APs (clutter is broadcast)
    R_g_cache: Dict[int, NDArray[np.complex128]] = {}
    for ar_idx in association.all_rx_ap_indices:
        ap_r = topo.aps[ar_idx]
        R_g_cache[ar_idx] = compute_clutter_noise_covariance(
            ap_r, W_tx, sensing_stats, sigma_n_sq, n_ant_r,
        )

    # ── Per-(a_r, target) SCNR ───────────────────────────────────────────
    scnr_dict: Dict[Tuple[int, int], float] = {}

    for tg_idx in range(topo.n_targets):
        rx_aps = association.rx_aps_for(tg_idx)
        tx_aps = association.tx_aps_for(tg_idx)
        if not rx_aps or not tx_aps:
            continue

        for ar_idx in rx_aps:
            a_ar_key = (ar_idx, tg_idx)
            if a_ar_key not in sensing_stats.a_rx:
                continue
            a_ar = sensing_stats.a_rx[a_ar_key]   # ‖a‖² = M_r

            # Spatial STAP gain  a_{a_r}^H R_g^{-1} a_{a_r}
            R_g = R_g_cache[ar_idx]
            try:
                R_g_inv_a = np.linalg.solve(R_g, a_ar)
            except np.linalg.LinAlgError:
                logger.warning(
                    "R_g singular at AP %d, target %d", ar_idx, tg_idx,
                )
                continue
            spatial_gain = float(np.real(a_ar.conj() @ R_g_inv_a))

            # Transmit beamforming gain  Σ_{a_t} β_{a_t a_r}^tgt ‖a_{a_t}^H W_{a_t}‖²
            tx_gain = 0.0
            for at_idx in tx_aps:
                if at_idx not in W_tx:
                    continue
                key_bi = (at_idx, ar_idx, tg_idx)
                key_at = (at_idx, tg_idx)
                if key_bi not in sensing_stats.beta_bistatic:
                    continue
                if not sensing_stats.los_state_tg.get(key_at, False):
                    # No LoS path → s_t = 0 → no target echo
                    continue
                if key_at not in sensing_stats.a_tx:
                    continue

                beta_bi = sensing_stats.beta_bistatic[key_bi]
                a_at    = sensing_stats.a_tx[key_at]      # ‖a‖² = M_t
                W_at    = W_tx[at_idx]
                # ‖a_{a_t}^H W_{a_t}‖²  — un-normalised steering, paper convention
                beam_gain = float(np.linalg.norm(a_at.conj() @ W_at) ** 2)
                tx_gain  += beta_bi * beam_gain

            scnr = sigma_rcs_sq * T_snapshots * tx_gain * spatial_gain
            scnr_dict[(ar_idx, tg_idx)] = max(scnr, 0.0)

    # ── Clutter-aware linear surrogate (CORDIS optimisation objective) ───
    surrogate = compute_sensing_surrogate(
        topo, cfg, W_tx, sensing_stats, association, omega
    )

    return SCNRMetrics(
        scnr_per_pair=scnr_dict,
        weights=omega,
        surrogate=surrogate,
    )


# =============================================================================
# Clutter-aware linear sensing surrogate  (CORDIS optimisation objective)
# =============================================================================

def compute_sensing_surrogate(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    W_tx: Dict[int, NDArray[np.complex128]],
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    omega: Optional[Dict[int, float]] = None,
) -> float:
    """
    Evaluate the clutter-aware linear sensing surrogate
    (eq. admm-linear-objective):

        U_cpu^sens(W) = Σ_{a_t} [
            Σ_{t} ω̄_t · β̄_{a_t}^t · ‖a_{a_t t}^H W_{a_t}‖²
          - κ · tr(W_{a_t}^H C_{a_t} W_{a_t})
        ]

    where ω̄_t = ω_t · T · σ_RCS²  and  β̄_{a_t}^t = Σ_{a_r ∈ A_r(t)} β_{a_t a_r}^t.

    This is the same value that CORDIS-Split (in P-Split) and
    CORDIS-ADMM (after SCA linearisation, in P-Local) maximize.

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    W_tx : dict[ap_idx, np.ndarray]
    sensing_stats : SensingChannelStatistics
    association : SensingAssociation
        Used to determine which (a_r, t) pairs contribute to β̄_{a_t}^t.
    omega : dict[tg_idx, float] or None
        Target priority weights ω_t.  Equal weights if None.

    Returns
    -------
    float
        Scalar value of the surrogate objective.
    """
    kappa        = cfg.algorithm.admm.kappa
    T_snapshots  = cfg.sensing.n_snapshots
    sigma_rcs_sq = sensing_stats.sigma_rcs_sq

    if omega is None:
        omega = {tg: 1.0 for tg in range(topo.n_targets)}

    total = 0.0
    for at_idx, W_at in W_tx.items():
        # Clutter penalty  κ · tr(W^H C_{a_t} W)
        C_at = sensing_stats.C_tx.get(
            at_idx, np.eye(W_at.shape[0], dtype=complex)
        )
        clutter_penalty = kappa * float(np.real(np.trace(
            W_at.conj().T @ C_at @ W_at
        )))

        # Target echo power  Σ_t ω̄_t · β̄_{a_t}^t · ‖a_{a_t}^H W_{a_t}‖²
        echo_power = 0.0
        for tg_idx in range(topo.n_targets):
            # Only targets that this AP is assigned to
            if at_idx not in association.tx_aps_for(tg_idx):
                continue

            a_at_key = (at_idx, tg_idx)
            if a_at_key not in sensing_stats.a_tx:
                continue
            if not sensing_stats.los_state_tg.get(a_at_key, False):
                continue

            # β̄_{a_t}^t = Σ_{a_r assigned to t} β_{a_t a_r}^tgt
            rx_aps_for_t = association.rx_aps_for(tg_idx)
            beta_bar = sum(
                sensing_stats.beta_bistatic.get((at_idx, ar_idx, tg_idx), 0.0)
                for ar_idx in rx_aps_for_t
            )

            omega_bar = omega.get(tg_idx, 1.0) * T_snapshots * sigma_rcs_sq
            a_at      = sensing_stats.a_tx[a_at_key]
            beam_gain = float(np.linalg.norm(a_at.conj() @ W_at) ** 2)
            echo_power += omega_bar * beta_bar * beam_gain

        total += echo_power - clutter_penalty

    return float(total)

