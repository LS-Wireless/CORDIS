"""
cordis/metrics/sinr.py
======================
SINR computation for the CORDIS framework.

Implements the journal SINR expression (eq. sinr, Section III-A):

    SINR_u(Ĥ) = P_CDS / (P_MUI + P_S2CI + P_CSI-Error + P_Noise)

where for each user u:

    P_CDS       = |Σ_{a_t} ĥ_{a_t u}^H w_{a_t u}|²            (signal)
    P_MUI       = Σ_{k≠u} |Σ_{a_t} ĥ_{a_t u}^H w_{a_t k}|²    (multi-user int.)
    P_S2CI      = Σ_{t∈T} |Σ_{a_t} ĥ_{a_t u}^H w_{a_t t}|²    (sensing leakage)
    P_CSI-Error = Σ_{a_t} tr(W_{a_t}^H R̃_{a_t u} W_{a_t})    (CSI mismatch)
    P_Noise     = σ_n²

The CSI-Error term is what distinguishes the journal SINR from the
conference-paper SINR; it captures the residual signal distortion arising
from imperfect channel estimation, using the error covariance R̃_{a_t u}
produced by the linear MMSE estimator.

This module also provides aggregate metrics (min-SINR, sum-rate) and
ergodic statistics across multiple trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from cordis.channel.estimation import EstimationResult
from cordis.channel.topology import NetworkTopology
from cordis.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Per-trial metrics
# =============================================================================

@dataclass
class SINRMetrics:
    """
    Per-trial SINR metrics for a single channel realization.

    All array attributes have shape ``(N_ue,)``.  Derived scalar
    quantities (min-SINR, sum-rate, …) are exposed as ``@property``.

    Attributes
    ----------
    sinr_per_user : np.ndarray, shape (N_ue,)
        Linear-scale SINR for each user.
    p_cds : np.ndarray
        Communication desired-signal power |Σ_{a_t} ĥ^H w_au|² for each user.
    p_mui : np.ndarray
        Total multi-user interference power for each user.
    p_s2ci : np.ndarray
        Total sensing-to-comm interference power for each user.
    p_csi_error : np.ndarray
        Total CSI estimation-error contribution for each user.
    p_noise : float
        Noise variance σ_n² (same for all users).
    """

    sinr_per_user: NDArray[np.float64]    # (N_ue,)  linear scale
    p_cds:         NDArray[np.float64]    # (N_ue,)
    p_mui:         NDArray[np.float64]    # (N_ue,)
    p_s2ci:        NDArray[np.float64]    # (N_ue,)
    p_csi_error:   NDArray[np.float64]    # (N_ue,)
    p_noise:       float

    # ── Per-user derived quantities ──────────────────────────────────────

    @property
    def sinr_per_user_db(self) -> NDArray[np.float64]:
        """Per-user SINR in dB."""
        return 10.0 * np.log10(np.maximum(self.sinr_per_user, 1e-30))

    @property
    def rate_per_user(self) -> NDArray[np.float64]:
        """Per-user achievable rate log2(1 + SINR) in bits/s/Hz."""
        return np.log2(1.0 + np.maximum(self.sinr_per_user, 0.0))

    # ── Scalar aggregates ────────────────────────────────────────────────

    @property
    def min_sinr(self) -> float:
        """Minimum SINR across users (linear)."""
        return float(self.sinr_per_user.min())

    @property
    def min_sinr_db(self) -> float:
        return float(self.sinr_per_user_db.min())

    @property
    def mean_sinr(self) -> float:
        """Arithmetic mean of per-user SINRs (linear)."""
        return float(self.sinr_per_user.mean())

    @property
    def mean_sinr_db(self) -> float:
        return float(self.sinr_per_user_db.mean())

    @property
    def sum_rate(self) -> float:
        """Sum of per-user rates Σ_u log2(1+SINR_u) [bits/s/Hz]."""
        return float(self.rate_per_user.sum())

    @property
    def min_rate(self) -> float:
        """Minimum per-user rate [bits/s/Hz]."""
        return float(self.rate_per_user.min())

    @property
    def mean_rate(self) -> float:
        """Mean per-user rate [bits/s/Hz]."""
        return float(self.rate_per_user.mean())


# =============================================================================
# Main computation
# =============================================================================

def compute_sinr(
    W_tx: Dict[int, NDArray[np.complex128]],
    est: EstimationResult,
    topo: NetworkTopology,
    sigma_n_sq: float,
) -> SINRMetrics:
    """
    Compute per-user instantaneous SINR (journal eq. sinr).

    Parameters
    ----------
    W_tx : dict[ap_idx, np.ndarray (Mt, N_ue + N_t)]
        Precoding matrix at each transmit AP.  The first ``N_ue`` columns
        are communication beamformers (one per user); the next ``N_t``
        columns are sensing beamformers (one per target).  Sensing beams
        for unassigned targets should be the zero vector.
    est : EstimationResult
        Channel estimation output containing ``h_hat`` and ``R_tilde``.
        Use the perfect-CSI mode of the estimator if you want noise-free
        SINR (R̃ = 0 for all links → P_CSI-Error = 0).
    topo : NetworkTopology
    sigma_n_sq : float
        Receiver noise variance σ_n² (same for all UEs).

    Returns
    -------
    SINRMetrics
        Per-user SINR and full power decomposition.

    Notes
    -----
    The function expects ``W_tx`` to contain entries for every AP listed
    in ``topo.tx_aps``.  Any AP not present is silently treated as
    inactive (W = 0), which is the natural behavior when a sensing
    assignment excludes some APs from transmission.
    """
    n_ue = topo.n_ue
    n_tg = topo.n_targets
    tx_ap_indices = [ap.idx for ap in topo.tx_aps if ap.idx in W_tx]

    if not tx_ap_indices:
        raise ValueError(
            "compute_sinr: no transmit APs with precoders in W_tx."
        )

    # Pre-compute, for each (a_t, u), the inner products  ĥ_{a_t u}^H W_{a_t}
    # which give a (N_ue + N_t,) vector per (AP, user).  This avoids
    # recomputing the same dot products inside the inner loops below.
    inner = np.zeros(
        (len(tx_ap_indices), n_ue, n_ue + n_tg), dtype=np.complex128
    )
    for a_idx, at_idx in enumerate(tx_ap_indices):
        W_at = W_tx[at_idx]                # (Mt, N_ue + N_t)
        for u in range(n_ue):
            h_hat = est.h_hat[(at_idx, u)]            # (Mt,)
            inner[a_idx, u, :] = h_hat.conj() @ W_at  # (N_ue + N_t,)

    # Sum across transmit APs:  for each (u, beam_idx),
    #   coherent_sum[u, k] = Σ_{a_t} ĥ_{a_t u}^H w_{a_t k}
    coherent_sum = inner.sum(axis=0)       # (N_ue, N_ue + N_t)

    # ── Power decomposition per user ──────────────────────────────────────
    p_cds = np.zeros(n_ue)
    p_mui = np.zeros(n_ue)
    p_s2ci = np.zeros(n_ue)
    p_csi = np.zeros(n_ue)

    for u in range(n_ue):
        # CDS_u = |coherent_sum[u, u]|²
        p_cds[u] = float(np.abs(coherent_sum[u, u]) ** 2)

        # MUI_u = Σ_{k ≠ u, k ∈ U} |coherent_sum[u, k]|²
        for k in range(n_ue):
            if k == u:
                continue
            p_mui[u] += float(np.abs(coherent_sum[u, k]) ** 2)

        # S2CI_u = Σ_{t ∈ T} |coherent_sum[u, N_ue + t]|²
        for t_idx in range(n_tg):
            p_s2ci[u] += float(np.abs(coherent_sum[u, n_ue + t_idx]) ** 2)

        # CSI-Error_u = Σ_{a_t} tr(W_{a_t}^H R̃_{a_t u} W_{a_t})
        for at_idx in tx_ap_indices:
            W_at = W_tx[at_idx]
            R_tilde_au = est.R_tilde[(at_idx, u)]
            # tr(W^H R̃ W) — compute as sum over columns d of w_d^H R̃ w_d
            # Equivalent to: real(trace(R̃ @ W @ W^H))
            term = np.real(np.trace(W_at.conj().T @ R_tilde_au @ W_at))
            p_csi[u] += float(term)

    # SINR per user
    denom = p_mui + p_s2ci + p_csi + sigma_n_sq
    sinr = p_cds / np.maximum(denom, 1e-30)

    return SINRMetrics(
        sinr_per_user=sinr,
        p_cds=p_cds,
        p_mui=p_mui,
        p_s2ci=p_s2ci,
        p_csi_error=p_csi,
        p_noise=float(sigma_n_sq),
    )


# =============================================================================
# Convenience wrappers
# =============================================================================

def sinr_to_rate(sinr_lin: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shannon rate log2(1 + SINR) in bits/s/Hz, element-wise."""
    return np.log2(1.0 + np.maximum(np.asarray(sinr_lin), 0.0))


def rate_to_sinr(rate_bps_hz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of sinr_to_rate: 2^rate − 1."""
    return 2.0 ** np.asarray(rate_bps_hz) - 1.0

