"""
cordis/algorithms/beamforming.py
================================
Beamforming design for the CORDIS framework.

Signal model convention
-----------------------
The paper uses the downlink model:   y_u = Σ h_{au}^H w_{au} s_u + …

H_hat is stored as shape (N_ue, M_t) with  H_hat[u, :] = h_u  (row u is
the channel vector for user u, NOT its conjugate).

The effective channel matrix that appears in the system equation y = H_eff x is:
    H_eff[u, :] = h_u^H  →  H_eff = H_hat.conj()

All precoder formulas are derived from this H_eff convention.

Previous bug
------------
The old code used  H_hat.conj().T  (= H^H, columns = h_u^*) everywhere,
giving

    h_u^H w_u ∝ h_u^H h_u^* = Σ_m (h_m^*)²   ← complex, random phase

instead of the correct

    h_u^H h_u = ||h_u||²   ← real, positive, adds coherently across APs.

That random phase prevented coherent combining across APs, causing
SINR ≈ −10 dB regardless of SNR or BF method.

Correct local formulas (one AP):
    MRT      : W = H_hat.T                              col u = h_u ✓
    ZF       : W = H_hat.T (H_hat.conj() H_hat.T)^{-1}
    RZF      : W = H_hat.T (H_hat.conj() H_hat.T + εI)^{-1}
    LR-MMSE  : W = (H_hat.T H_hat.conj() + R̃_H + εI)^{-1} H_hat.T   [spatial Gram]

Benchmark methods
-----------------
Local (one AP at a time):  lr_mmse, rzf, mrt, random
Global (all APs jointly):  global_zf, global_mrt
Sensing:                   ns_c
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.utils.config import CORDISConfig
from cordis.utils.math_utils import channel_quality_metric
from cordis.utils.logger import get_logger
from cordis.channel.topology import NetworkTopology, SensingAssociation
from cordis.channel.estimation import EstimationResult
from cordis.channel.sensing_channel import SensingChannelStatistics

logger = get_logger(__name__)


# =============================================================================
# Internal helpers
# =============================================================================

def _eps_scaled(gram: NDArray, eps_rel: float) -> float:
    """
    Channel-power-relative regularization.

        ε_abs = ε_rel × tr(Gram) / n

    Ensures ε is comparable to the Gram eigenvalues regardless of path
    loss magnitude.  With absolute ε = 0.01 and β ≈ 1e-8 the old code
    had ε >> all eigenvalues, making every method degenerate to the same
    result.
    """
    n  = gram.shape[0]
    tr = float(np.real(np.trace(gram)))
    return eps_rel * max(tr, 1e-30) / n


def _normalise(W: NDArray) -> NDArray:
    """Divide W by its Frobenius norm; return zeros if near-zero."""
    n = np.linalg.norm(W, 'fro')
    return (W / n if n > 1e-15 else W).astype(np.complex128)


# =============================================================================
# Local communication beamforming
# =============================================================================

def lr_mmse(
    H_hat: NDArray[np.complex128],
    R_tilde_H: NDArray[np.complex128],
    alpha: float,
    epsilon: float = 1e-2,
) -> NDArray[np.complex128]:
    """
    Local Robust MMSE (journal eq. split-comm-bf).

    Spatial-domain (M×M) form so R̃_H enters naturally:

        W̃ = (H.T H.conj() + R̃_H + ε I)^{-1} H.T · α

    Signal: h_u^H w_u = real, positive ✓

    Parameters
    ----------
    H_hat     : (N_ue, M_t)  rows = h_u
    R_tilde_H : (M_t, M_t)  Σ_u R̃_{au}
    alpha     : float        channel quality weight
    epsilon   : float        relative regularization coefficient

    Returns
    -------
    W : (M_t, N_ue), ||W||_F = 1
    """
    Mt   = H_hat.shape[1]
    gram = H_hat.T @ H_hat.conj()                           # (M_t, M_t)
    gram = gram + R_tilde_H + _eps_scaled(gram, epsilon) * np.eye(Mt)
    W    = np.linalg.solve(gram, H_hat.T) * alpha           # (M_t, N_ue)
    return _normalise(W)


def rzf(
    H_hat: NDArray[np.complex128],
    alpha: float,
    epsilon: float = 1e-2,
) -> NDArray[np.complex128]:
    """
    Local Regularized Zero-Forcing (MMSE without CSI-error robustness).

    User-domain (K×K) form — smaller inversion when K < M_t:

        W̃ = H.T (H.conj() H.T + ε I)^{-1} · α

    Returns
    -------
    W : (M_t, N_ue), ||W||_F = 1
    """
    K    = H_hat.shape[0]
    gram = H_hat.conj() @ H_hat.T                           # (K, K)
    gram = gram + _eps_scaled(gram, epsilon) * np.eye(K)
    W    = H_hat.T @ np.linalg.solve(gram, np.eye(K)) * alpha
    return _normalise(W)


def zf(
    H_hat: NDArray[np.complex128],
    alpha: float,
) -> NDArray[np.complex128]:
    """
    Local Zero-Forcing (right pseudo-inverse).

        W̃ = H.T (H.conj() H.T)^{-1} · α

    Falls back to RZF with tiny ε if Gram is ill-conditioned.

    Returns
    -------
    W : (M_t, N_ue), ||W||_F = 1
    """
    K    = H_hat.shape[0]
    gram = H_hat.conj() @ H_hat.T                           # (K, K)
    cond = np.linalg.cond(gram)
    if not np.isfinite(cond) or cond > 1e10:
        logger.debug("ZF: ill-conditioned Gram (%.1e), using RZF fallback.", cond)
        return rzf(H_hat, alpha, epsilon=1e-8)
    W = H_hat.T @ np.linalg.solve(gram, np.eye(K)) * alpha
    return _normalise(W)


def mrt(
    H_hat: NDArray[np.complex128],
    alpha: float,
) -> NDArray[np.complex128]:
    """
    Local Maximum Ratio Transmission for the model  y = h^H x.

        W̃ = H.T · α      (column u = h_u, NOT h_u^*)

    Signal: h_u^H h_u / ||H||_F = ||h_u||² / ||H||_F  (real, positive ✓)

    Returns
    -------
    W : (M_t, N_ue), ||W||_F = 1
    """
    return _normalise(H_hat.T * alpha)


def random_bf(
    n_ant: int,
    n_beams: int,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """
    Isotropic random Gaussian (lower bound baseline).

    Returns
    -------
    W : (M_t, n_beams), ||W||_F = 1
    """
    W = (rng.standard_normal((n_ant, n_beams))
         + 1j * rng.standard_normal((n_ant, n_beams))) / np.sqrt(2.0)
    return _normalise(W)


# =============================================================================
# Global communication beamforming (all APs jointly)
# =============================================================================

def global_zf(
    est: EstimationResult,
    topo: NetworkTopology,
    epsilon: float = 1e-2,
) -> Dict[int, NDArray[np.complex128]]:
    """
    Global Zero-Forcing using the concatenated channel matrix.

    H_global ∈ ℂ^{N_ue × (M_t N_tx)}  — all APs' channels side by side.

        W_global = H_global.T (H_global.conj() H_global.T + ε I)^{-1}

    The GLOBAL Frobenius norm is used for normalization so all APs share
    a consistent power budget.

    Returns
    -------
    dict[ap_idx → (M_t, N_ue)],  global ||concat(W)||_F = 1
    """
    K   = topo.n_ue
    M_t = est.H_hat[topo.tx_aps[0].idx].shape[1]

    H_global = np.hstack([est.H_hat[ap.idx] for ap in topo.tx_aps])  # (K, M_t N_tx)
    gram     = H_global.conj() @ H_global.T                           # (K, K)
    gram    += _eps_scaled(gram, epsilon) * np.eye(K)
    W_global = H_global.T @ np.linalg.solve(gram, np.eye(K))          # (M_t N_tx, K)

    g_norm = np.linalg.norm(W_global, 'fro')
    if g_norm > 1e-15:
        W_global /= g_norm

    return {
        ap.idx: W_global[i * M_t: (i + 1) * M_t, :].astype(np.complex128)
        for i, ap in enumerate(topo.tx_aps)
    }


def global_mrt(
    est: EstimationResult,
    topo: NetworkTopology,
) -> Dict[int, NDArray[np.complex128]]:
    """
    Global MRT with joint power normalization across all TX APs.

    Local MRT normalises each AP by its own ||H_at||_F, giving different
    power shares between APs.  Global MRT normalises by the GLOBAL
    Frobenius norm of the concatenated precoder, which is the correct
    matched filter under a sum-power constraint.

    Returns
    -------
    dict[ap_idx → (M_t, N_ue)],  global ||concat(W)||_F = 1
    """
    W_raw = {ap.idx: est.H_hat[ap.idx].T.copy() for ap in topo.tx_aps}
    g_norm = np.linalg.norm(np.vstack(list(W_raw.values())), 'fro')
    return {
        a: (W / g_norm if g_norm > 1e-15 else W).astype(np.complex128)
        for a, W in W_raw.items()
    }


COMM_BF_METHODS = {
    "lr_mmse":    "Local Robust MMSE (journal Phase I default)",
    "rzf":        "Local Regularised Zero-Forcing",
    "zf":         "Local Zero-Forcing",
    "mrt":        "Local MRT",
    "random":     "Isotropic random Gaussian (lower bound)",
    "global_zf":  "Global Zero-Forcing (centralised)",
    "global_mrt": "Global MRT (joint normalisation)",
}


# =============================================================================
# Sensing beamforming: NS-C with priority-aware allocation
# =============================================================================

def ns_c(
    H_hat: NDArray[np.complex128],
    a_targets: Dict[int, NDArray[np.complex128]],
    omega: Optional[Dict[int, float]] = None,
    epsilon: float = 1e-3,
) -> Tuple[NDArray[np.complex128], Dict[int, float]]:
    """
    Null-Space Conjugate sensing beamformer (journal §IV-A).

    For each assigned target t:
        1. P_perp = I − H.T (H.conj() H.T + ε I)^{-1} H.conj()
           → satisfies  H.conj() P_perp = 0  (null space of H_eff)
        2. w̃_t = P_perp a_t
        3. λ_t = ω_t ||w̃_t||² / Σ_j ω_j ||w̃_j||²
        4. ŵ_t = √λ_t · w̃_t / ||w̃_t||

    Returns
    -------
    W_sens : (M_t, N_assigned), ||W||_F = 1
    lambda_alloc : dict[tg_idx, float]
    """
    Mt = H_hat.shape[1]
    if not a_targets:
        return np.zeros((Mt, 0), dtype=complex), {}

    if omega is None:
        omega = {tg: 1.0 for tg in a_targets}

    # Null-space projector (corrected convention)
    K    = H_hat.shape[0]
    gram = H_hat.conj() @ H_hat.T + _eps_scaled(
        H_hat.conj() @ H_hat.T, epsilon
    ) * np.eye(K)
    # P_perp = I - H.T inv(gram) H.conj()
    P_perp = np.eye(Mt, dtype=complex) - H_hat.T @ np.linalg.solve(gram, H_hat.conj())

    tg_order  = sorted(a_targets.keys())
    w_tilde:  Dict[int, NDArray] = {}
    norms_sq: Dict[int, float]   = {}
    for tg in tg_order:
        w = P_perp @ a_targets[tg]
        w_tilde[tg]  = w
        norms_sq[tg] = float(np.real(w.conj() @ w))

    total = sum(omega.get(t, 1.0) * norms_sq[t] for t in tg_order)
    lambda_alloc: Dict[int, float] = {}
    cols: List[NDArray] = []

    for tg in tg_order:
        nsq  = norms_sq[tg]
        om   = omega.get(tg, 1.0)
        if total < 1e-15 or nsq < 1e-15:
            lambda_alloc[tg] = 0.0
            cols.append(np.zeros(Mt, dtype=complex))
        else:
            lam = om * nsq / total
            lambda_alloc[tg] = float(lam)
            cols.append(np.sqrt(lam) * w_tilde[tg] / np.sqrt(nsq))

    W_s = np.column_stack(cols) if cols else np.zeros((Mt, 0), dtype=complex)
    return _normalise(W_s), lambda_alloc


# =============================================================================
# Channel quality weights
# =============================================================================

def compute_alpha_weights(
    est: EstimationResult,
    topo: NetworkTopology,
) -> Dict[int, float]:
    """
    α_{a_t} = η(a_t) / Σ_b η(b_t),  η = reciprocal condition number of Ĥ.

    Returns dict summing to 1, all values positive.
    """
    eta   = {ap.idx: channel_quality_metric(est.H_hat[ap.idx]) for ap in topo.tx_aps}
    total = sum(eta.values())
    if total < 1e-15:
        n = len(topo.tx_aps)
        return {ap.idx: 1.0 / n for ap in topo.tx_aps}
    return {a: e / total for a, e in eta.items()}


# =============================================================================
# PhaseIResult
# =============================================================================

@dataclass
class PhaseIResult:
    """
    Output of CORDIS-Split Phase I (distributed BF at TX APs).

    W_comm_hat : dict[ap_idx → (M_t, N_ue)] normalized comm precoders
    W_sens_hat : dict[ap_idx → (M_t, N_t)]   normalised sensing precoders
    alpha      : dict[ap_idx, float]          channel quality weights
    lambda_alloc : dict[ap_idx, dict[tg, float]]
    comm_bf_method : str
    """

    W_comm_hat:     Dict[int, NDArray[np.complex128]]
    W_sens_hat:     Dict[int, NDArray[np.complex128]]
    alpha:          Dict[int, float]
    lambda_alloc:   Dict[int, Dict[int, float]]
    comm_bf_method: str

    def build_W_tx(
        self,
        psr: Dict[int, float],
        Pmax: float,
    ) -> Dict[int, NDArray[np.complex128]]:
        """
        Scale by PSR and P_max:  W_at = [√(ρ P) Ŵ^(c)  |  √((1-ρ) P) Ŵ^(s)]
        """
        W_tx: Dict[int, NDArray] = {}
        for ap_idx, W_c in self.W_comm_hat.items():
            rho = float(psr.get(ap_idx, 0.5))
            W_s = self.W_sens_hat[ap_idx]
            W_tx[ap_idx] = np.hstack([
                np.sqrt(rho * Pmax) * W_c,
                np.sqrt((1.0 - rho) * Pmax) * W_s,
            ]).astype(np.complex128)
        return W_tx

    def build_W_tx_equal_psr(
        self,
        psr: float,
        Pmax: float,
    ) -> Dict[int, NDArray[np.complex128]]:
        """Uniform ρ across all APs."""
        return self.build_W_tx({ap: psr for ap in self.W_comm_hat}, Pmax)


# =============================================================================
# Phase I assembly
# =============================================================================

def design_phase_i(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    est: EstimationResult,
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    comm_bf_method: str = "lr_mmse",
    omega: Optional[Dict[int, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> PhaseIResult:
    """
    CORDIS-Split Phase I: beamforming design at each TX AP.

    For global methods ("global_zf", "global_mrt") the comm precoder is
    computed jointly across all APs, then NS-C runs locally per AP.

    Parameters
    ----------
    topo, cfg, est, sensing_stats, association : standard pipeline objects
    comm_bf_method : str  — key in COMM_BF_METHODS
    omega          : dict[tg_idx, float] or None
    rng            : required for "random"

    Returns
    -------
    PhaseIResult
    """
    n_ant    = cfg.topology.n_ant
    n_t      = topo.n_targets
    eps_comm = cfg.algorithm.split.epsilon_reg
    eps_sens = cfg.algorithm.split.epsilon_nsc
    method   = comm_bf_method.lower().strip()

    if method not in COMM_BF_METHODS:
        raise ValueError(
            f"Unknown BF method '{comm_bf_method}'. "
            f"Choose from: {list(COMM_BF_METHODS)}."
        )

    alpha_dict = compute_alpha_weights(est, topo)

    # Pre-compute global comm precoders (all APs at once)
    W_comm_global: Optional[Dict[int, NDArray]] = None
    if method == "global_zf":
        W_comm_global = global_zf(est, topo, epsilon=eps_comm)
    elif method == "global_mrt":
        W_comm_global = global_mrt(est, topo)

    W_comm_hat_dict: Dict[int, NDArray] = {}
    W_sens_hat_dict: Dict[int, NDArray] = {}
    lambda_dict:     Dict[int, Dict[int, float]] = {}

    for ap in topo.tx_aps:
        a_idx     = ap.idx
        H_hat     = est.H_hat[a_idx]         # (N_ue, M_t)
        R_tilde_H = est.R_tilde_H[a_idx]     # (M_t, M_t)
        alpha     = alpha_dict[a_idx]

        # Communication BF
        if W_comm_global is not None:
            W_c = W_comm_global[a_idx]
        elif method == "lr_mmse":
            W_c = lr_mmse(H_hat, R_tilde_H, alpha, epsilon=eps_comm)
        elif method == "rzf":
            W_c = rzf(H_hat, alpha, epsilon=eps_comm)
        elif method == "zf":
            W_c = zf(H_hat, alpha)
        elif method == "mrt":
            W_c = mrt(H_hat, alpha)
        elif method == "random":
            if rng is None:
                raise ValueError("random BF requires rng.")
            W_c = random_bf(n_ant, topo.n_ue, rng)
        else:
            raise ValueError(f"Unhandled method: {method}")

        W_comm_hat_dict[a_idx] = W_c

        # Sensing BF (NS-C)
        a_targets = {
            tg_idx: sensing_stats.a_tx[(a_idx, tg_idx)]
            for tg_idx in range(n_t)
            if a_idx in association.tx_aps_for(tg_idx)
            and (a_idx, tg_idx) in sensing_stats.a_tx
        }
        W_s_cols, lam = ns_c(H_hat, a_targets, omega=omega, epsilon=eps_sens)

        # Pad to full (M_t, N_t) with zeros for unassigned targets
        W_s_full = np.zeros((n_ant, n_t), dtype=complex)
        for col, tg in enumerate(sorted(a_targets.keys())):
            if col < W_s_cols.shape[1]:
                W_s_full[:, tg] = W_s_cols[:, col]
        s_norm = np.linalg.norm(W_s_full, 'fro')
        if s_norm > 1e-15:
            W_s_full /= s_norm

        W_sens_hat_dict[a_idx] = W_s_full.astype(np.complex128)
        lambda_dict[a_idx]     = lam

    logger.debug(
        "Phase I (%s): %d TX APs, %d targets, α ∈ [%.3f, %.3f]",
        comm_bf_method, len(topo.tx_aps), n_t,
        min(alpha_dict.values()), max(alpha_dict.values()),
    )
    return PhaseIResult(
        W_comm_hat=W_comm_hat_dict,
        W_sens_hat=W_sens_hat_dict,
        alpha=alpha_dict,
        lambda_alloc=lambda_dict,
        comm_bf_method=comm_bf_method,
    )


# =============================================================================
# Compressed scalars for CORDIS-Split Phase II
# =============================================================================

@dataclass
class SplitScalars:
    """
    Compressed uplink scalars for Algorithm 1 (§IV).

    beta_hat : dict[ap → (N_ue,) complex]
    g_tilde  : dict[ap → (N_ue,) float]   MUI + CSI-error
    e_sens   : dict[ap → (N_ue,) float]   sensing CSI-error leakage
    z        : dict[ap, float]             target gain difference
    q_tilde  : dict[ap, float]             clutter illumination difference
    """
    beta_hat: Dict[int, NDArray[np.complex128]]
    g_tilde:  Dict[int, NDArray[np.float64]]
    e_sens:   Dict[int, NDArray[np.float64]]
    z:        Dict[int, float]
    q_tilde:  Dict[int, float]


def compute_split_scalars(
    phase_i: PhaseIResult,
    est: EstimationResult,
    topo: NetworkTopology,
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    omega: Optional[Dict[int, float]] = None,
    kappa: float = 1.0,
) -> SplitScalars:
    """
    Compute the compressed scalars sent by each AP to the CPU in Phase I
    (Algorithm 1, lines 6–7).
    """
    n_ue = topo.n_ue
    n_t  = topo.n_targets
    if omega is None:
        omega = {tg: 1.0 for tg in range(n_t)}

    beta_hat_dict: Dict[int, NDArray] = {}
    g_tilde_dict:  Dict[int, NDArray] = {}
    e_sens_dict:   Dict[int, NDArray] = {}
    z_dict:        Dict[int, float]   = {}
    q_tilde_dict:  Dict[int, float]   = {}

    for ap in topo.tx_aps:
        a_idx   = ap.idx
        W_c_hat = phase_i.W_comm_hat[a_idx]    # (M_t, N_ue)
        W_s_hat = phase_i.W_sens_hat[a_idx]    # (M_t, N_t)

        beta_hat = np.zeros(n_ue, dtype=complex)
        g_tilde  = np.zeros(n_ue, dtype=float)
        e_sens   = np.zeros(n_ue, dtype=float)

        for u in range(n_ue):
            h      = est.h_hat[(a_idx, u)]        # (M_t,)
            R_tilde = est.R_tilde[(a_idx, u)]      # (M_t, M_t)

            # β̂ = h_u^H ŵ^(c)_u  (now real and positive for MRT/ZF)
            beta_hat[u] = h.conj() @ W_c_hat[:, u]

            # MUI from comm beams + CSI-error from comm beams
            mui = sum(
                float(abs(h.conj() @ W_c_hat[:, k]) ** 2)
                for k in range(n_ue) if k != u
            )
            e_c = float(np.real(np.trace(W_c_hat.conj().T @ R_tilde @ W_c_hat)))
            g_tilde[u] = mui + e_c

            # CSI-error from sensing beams
            e_sens[u] = float(np.real(np.trace(W_s_hat.conj().T @ R_tilde @ W_s_hat)))

        beta_hat_dict[a_idx] = beta_hat
        g_tilde_dict[a_idx]  = g_tilde
        e_sens_dict[a_idx]   = e_sens

        # z_{a_t}: target gain difference
        z_val  = 0.0
        sr_sq  = sensing_stats.sigma_rcs_sq
        for tg in range(n_t):
            if a_idx not in association.tx_aps_for(tg):
                continue
            key = (a_idx, tg)
            if key not in sensing_stats.a_tx:
                continue
            a_at     = sensing_stats.a_tx[key]
            beta_bar = sum(
                sensing_stats.beta_bistatic.get((a_idx, ar, tg), 0.0)
                for ar in association.rx_aps_for(tg)
            )
            gc = float(np.linalg.norm(a_at.conj() @ W_c_hat) ** 2)
            gs = float(np.linalg.norm(a_at.conj() @ W_s_hat) ** 2)
            z_val += omega.get(tg, 1.0) * sr_sq * beta_bar * (gc - gs)
        z_dict[a_idx] = z_val

        # q̃_{a_t}: clutter illumination difference
        C = sensing_stats.C_tx.get(a_idx, np.eye(W_c_hat.shape[0], dtype=complex))
        q_c = float(np.real(np.trace(W_c_hat.conj().T @ C @ W_c_hat)))
        q_s = float(np.real(np.trace(W_s_hat.conj().T @ C @ W_s_hat)))
        q_tilde_dict[a_idx] = q_c - q_s

    return SplitScalars(
        beta_hat=beta_hat_dict,
        g_tilde=g_tilde_dict,
        e_sens=e_sens_dict,
        z=z_dict,
        q_tilde=q_tilde_dict,
    )

