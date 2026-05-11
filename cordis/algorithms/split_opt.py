"""
cordis/algorithms/split_opt.py
===============================
CORDIS-Split algorithm (Algorithm 1 in the paper).

Phase I  — Distributed beamforming at each AP (LR-MMSE + NS-C).
           Already implemented in ``beamforming.py``.
Phase II — Centralized power allocation at CPU (P-Split).
           Solves a strictly convex program for the PSRs ρ_{a_t}.

Two solver backends:

    **CVXPY** (primary, recommended):
        Expresses S_u(ρ) via the concave geometric-mean decomposition
        and uses interior-point methods (CLARABEL / SCS / ECOS).

    **scipy** (fallback):
        Uses the p = √ρ substitution and ``trust-constr`` with
        analytical Jacobians.

Usage::

    W_tx, phase_i, res = run_cordis_split(topo, cfg, est, ...)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.utils.logger import get_logger
from cordis.utils.config import CORDISConfig
from cordis.utils.math_utils import db2lin, lin2db
from cordis.channel.topology import NetworkTopology
from cordis.channel.estimation import EstimationResult
from cordis.channel.sensing_channel import SensingChannelStatistics
from cordis.channel.sensing_assignment import SensingAssociation
from cordis.algorithms.beamforming import (
    PhaseIResult, SplitScalars,
    design_phase_i, compute_split_scalars,
)

logger = get_logger(__name__)

# ── Optional CVXPY import ──────────────────────────────────────────────────────
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


# =============================================================================
# Result container
# =============================================================================

@dataclass
class SplitOptResult:
    """
    Output of CORDIS-Split.

    rho_opt     : dict[TX-AP index → optimal PSR ∈ [0, 1]]
    objective   : final value of the P-Split objective
    sinr_u_est  : (N_ue,) estimated per-user SINR from the model
    slack       : (N_ue,) slack ε_u  (> 0 ⟹ SINR_u < γ_u)
    converged   : solver success flag
    n_evals     : function evaluations (scipy) or iterations (CVXPY)
    solver      : solver name
    """
    rho_opt:    Dict[int, float]
    objective:  float
    sinr_u_est: NDArray[np.float64]
    slack:      NDArray[np.float64]
    converged:  bool
    n_evals:    int
    solver:     str


# =============================================================================
# Data preparation helpers
# =============================================================================

def _build_p_split_data(
    scalars: SplitScalars,
    tx_ap_indices: List[int],
    n_ue: int,
    sigma_n_sq: float,
    Pmax: float,
    gamma: NDArray[np.float64],
    kappa: float,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, float]:
    """Stack compressed scalars into dense arrays + constraint normalisation."""
    n_tx   = len(tx_ap_indices)
    noise_f = sigma_n_sq / Pmax           # raw noise floor

    # ── Stack raw scalars ──────────────────────────────────────────────────
    beta_hat = np.zeros((n_tx, n_ue), dtype=complex)
    g_tilde  = np.zeros((n_tx, n_ue), dtype=float)
    e_s      = np.zeros((n_tx, n_ue), dtype=float)
    c_vec    = np.zeros(n_tx, dtype=float)

    for i, a_idx in enumerate(tx_ap_indices):
        beta_hat[i, :] = scalars.beta_hat[a_idx]
        g_tilde[i, :]  = scalars.g_tilde[a_idx]
        e_s[i, :]      = scalars.e_sens[a_idx]
        c_vec[i]       = scalars.z[a_idx] - kappa * scalars.q_tilde[a_idx]

    # ── Normalize constraints so all values are O(1) ───────────────────────
    # Without this, 3GPP path-loss makes S_u ~ 1e-12, D_u ~ 1e-9,
    # rendering the slack penalty ξ ineffective.
    if noise_f > 1e-30:
        cnorm     = 1.0 / noise_f        # = Pmax / σ²
        beta_hat *= np.sqrt(cnorm)
        g_tilde  *= cnorm
        e_s      *= cnorm
        noise_f   = 1.0

    return beta_hat, g_tilde, e_s, c_vec, gamma, noise_f


def _signal_power(rho, beta_hat):
    """S_u(ρ) = |Σ √ρ β̂|² for each u.  (N_ue,) float."""
    sqrt_rho = np.sqrt(np.maximum(rho, 0.0))
    f = sqrt_rho @ beta_hat           # (N_ue,) complex
    return np.abs(f) ** 2


def _interf_power(rho, g_tilde, e_sens, noise_f):
    """D_u(ρ) = Σ [ρ g̃ + (1−ρ) e_s] + noise.  (N_ue,) float."""
    delta = g_tilde - e_sens
    return rho @ delta + e_sens.sum(axis=0) + noise_f


# =============================================================================
# CVXPY solver (primary)
# =============================================================================

def _solve_p_split_cvxpy(
    beta_hat: NDArray[np.complex128],
    g_tilde:  NDArray[np.float64],
    e_sens:   NDArray[np.float64],
    c_vec:    NDArray[np.float64],
    gamma_u:  NDArray[np.float64],
    noise_f:  float,
    xi:       float,
    solver:   str = "CLARABEL",
) -> Tuple[NDArray, NDArray, float, bool, int]:
    """
    Solve P-Split via CVXPY interior-point.

    S_u(ρ) is expressed via its concave decomposition:
        S_u = Σ_i ρ_i |β̂_iu|² + 2 Σ_{i<j} β̂_iu β̂_ju √(ρ_i ρ_j)

    Since β̂ is real-positive for LR-MMSE, all cross-terms are positive
    and each √(ρ_i ρ_j) = geo_mean(ρ_i, ρ_j) is concave.  The resulting
    S_u is a sum of affine and concave terms ⟹ concave. ✓
    """
    n_tx, n_ue = beta_hat.shape

    # β̂ should be real-positive for LR-MMSE.  Phase-align if needed.
    beta_r = np.real(beta_hat)
    if np.any(beta_r < -1e-12):
        logger.warning("β̂ has negative real parts — applying phase alignment")
        for u in range(n_ue):
            phase = np.angle(beta_hat[:, u].sum())
            beta_hat[:, u] *= np.exp(-1j * phase)
        beta_r = np.real(beta_hat)
    beta_r = np.maximum(beta_r, 0.0)

    # ── Variables ──────────────────────────────────────────────────────────
    rho = cp.Variable(n_tx, nonneg=True)
    eps = cp.Variable(n_ue, nonneg=True)

    # ── Objective: max c^T ρ − ξ 1^T ε ───────────────────────────────────
    objective = cp.Maximize(c_vec @ rho - xi * cp.sum(eps))

    # ── Constraints ────────────────────────────────────────────────────────
    constraints = [rho <= 1.0]

    delta = g_tilde - e_sens                   # (N_tx, N_ue)
    e_sum = e_sens.sum(axis=0)                 # (N_ue,)

    for u in range(n_ue):
        # Build concave expression for S_u(ρ)
        b = beta_r[:, u]                       # (N_tx,) real positive

        # Diagonal terms: Σ ρ_i b_i²   (affine in ρ)
        S_u = b ** 2 @ rho

        # Cross-terms: 2 Σ_{i<j} b_i b_j √(ρ_i ρ_j)  (concave)
        for i in range(n_tx):
            for j in range(i + 1, n_tx):
                if b[i] > 1e-15 and b[j] > 1e-15:
                    S_u = S_u + 2.0 * b[i] * b[j] * cp.geo_mean(
                        cp.vstack([rho[i], rho[j]])
                    )

        # D_u(ρ) = δ^T ρ + e_sum + noise   (affine)
        D_u = delta[:, u] @ rho + e_sum[u] + noise_f

        # S_u + ε_u ≥ γ_u D_u   (concave + affine ≥ affine → DCP ✓)
        constraints.append(S_u + eps[u] >= gamma_u[u] * D_u)

    # ── Solve ──────────────────────────────────────────────────────────────
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver, verbose=False)
    except (cp.SolverError, Exception) as exc:
        logger.warning("CVXPY %s failed: %s — trying SCS", solver, exc)
        try:
            prob.solve(solver="SCS", verbose=False)
        except Exception:
            logger.error("CVXPY solve failed entirely")
            return (0.5 * np.ones(n_tx), np.zeros(n_ue), 0.0, False, 0)

    success = prob.status in ("optimal", "optimal_inaccurate")
    if rho.value is None:
        return (0.5 * np.ones(n_tx), np.zeros(n_ue), 0.0, False, 0)

    rho_opt = np.clip(rho.value, 0.0, 1.0)
    eps_opt = np.maximum(eps.value, 0.0)
    obj_val = float(prob.value) if prob.value is not None else 0.0

    return rho_opt, eps_opt, obj_val, success, int(prob.solver_stats.num_iters or 0)


# =============================================================================
# scipy solver (fallback)
# =============================================================================

def _solve_p_split_scipy(
    beta_hat: NDArray[np.complex128],
    g_tilde:  NDArray[np.float64],
    e_sens:   NDArray[np.float64],
    c_vec:    NDArray[np.float64],
    gamma_u:  NDArray[np.float64],
    noise_f:  float,
    xi:       float,
    maxiter:  int = 1000,
) -> Tuple[NDArray, NDArray, float, bool, int]:
    """
    Solve P-Split via scipy trust-constr with p = √ρ substitution.
    """
    from scipy.optimize import minimize, NonlinearConstraint

    n_tx, n_ue = beta_hat.shape
    n_vars     = n_tx + n_ue

    beta_re = beta_hat.real
    beta_im = beta_hat.imag
    delta   = g_tilde - e_sens
    e_sum   = e_sens.sum(axis=0)

    p_init = 0.5 * np.ones(n_tx)
    x0     = np.concatenate([p_init, 0.01 * np.ones(n_ue)])
    bounds = [(0.0, 1.0)] * n_tx + [(0.0, None)] * n_ue

    def objective(x):
        p, eps = x[:n_tx], x[n_tx:]
        return -(c_vec @ (p ** 2) - xi * eps.sum())

    def objective_jac(x):
        g = np.zeros(n_vars)
        g[:n_tx] = -2.0 * c_vec * x[:n_tx]
        g[n_tx:] = xi
        return g

    def cons_vec(x):
        p, eps = x[:n_tx], x[n_tx:]
        f_re = p @ beta_re;  f_im = p @ beta_im
        S = f_re ** 2 + f_im ** 2
        D = (p ** 2) @ delta + e_sum + noise_f
        return S + eps - gamma_u * D

    def cons_jac(x):
        p = x[:n_tx]
        J  = np.zeros((n_ue, n_vars))
        f_re = p @ beta_re;  f_im = p @ beta_im
        dS = 2.0 * (beta_re * f_re[None, :] + beta_im * f_im[None, :])
        dD = 2.0 * p[:, None] * delta
        J[:, :n_tx] = (dS - gamma_u[None, :] * dD).T
        J[:, n_tx:] = np.eye(n_ue)
        return J

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            objective, x0, method="trust-constr",
            jac=objective_jac, bounds=bounds,
            constraints=NonlinearConstraint(cons_vec, 0.0, np.inf, jac=cons_jac),
            options={"maxiter": maxiter, "gtol": 1e-8, "verbose": 0},
        )

    p_opt   = np.clip(res.x[:n_tx], 0.0, 1.0)
    rho_opt = p_opt ** 2
    eps_opt = np.maximum(res.x[n_tx:], 0.0)
    return rho_opt, eps_opt, -(res.fun), bool(res.success), int(res.nfev)


# =============================================================================
# Public API — solve P-Split
# =============================================================================

def solve_p_split(
    scalars: SplitScalars,
    topo: NetworkTopology,
    cfg: CORDISConfig,
    sigma_n_sq: float,
    Pmax: float,
    gamma_u_db: Optional[NDArray[np.float64]] = None,
    use_cvxpy: Optional[bool] = None,
) -> SplitOptResult:
    """
    Solve CORDIS-Split Phase II (P-Split).

    Parameters
    ----------
    scalars    : compressed Phase I scalars
    topo       : network topology
    cfg        : simulation config
    sigma_n_sq : noise power [W]
    Pmax       : per-AP power budget [W]
    gamma_u_db : per-user SINR target [dB] (None → config default)
    use_cvxpy  : True = force CVXPY, False = force scipy, None = auto-detect

    Returns
    -------
    SplitOptResult
    """
    n_ue          = topo.n_ue
    n_t           = topo.n_targets
    tx_ap_indices = [ap.idx for ap in topo.tx_aps]
    kappa         = cfg.algorithm.split.kappa
    xi            = cfg.algorithm.split.xi_penalty

    # ── Short-circuit: no targets → ρ = 1 ──────────────────────────────────
    if n_t == 0:
        rho_dict = {a_idx: 1.0 for a_idx in tx_ap_indices}
        return SplitOptResult(
            rho_opt=rho_dict, objective=0.0,
            sinr_u_est=np.zeros(n_ue), slack=np.zeros(n_ue),
            converged=True, n_evals=0, solver="trivial (no targets)",
        )

    # ── Per-user SINR threshold ────────────────────────────────────────────
    if gamma_u_db is None:
        gamma_lin = db2lin(cfg.algorithm.split.gamma_db) * np.ones(n_ue)
    else:
        gamma_lin = db2lin(np.asarray(gamma_u_db, dtype=float))

    beta_hat, g_tilde, e_sens, c_vec, gamma_lin, noise_f = _build_p_split_data(
        scalars, tx_ap_indices, n_ue, sigma_n_sq, Pmax, gamma_lin, kappa,
    )

    # ── Select solver ──────────────────────────────────────────────────────
    want_cvxpy = use_cvxpy if use_cvxpy is not None else _HAS_CVXPY
    cvxpy_solver = cfg.algorithm.split.solver

    if want_cvxpy and _HAS_CVXPY:
        rho_opt, eps_opt, obj_val, success, n_ev = _solve_p_split_cvxpy(
            beta_hat, g_tilde, e_sens, c_vec, gamma_lin, noise_f, xi,
            solver=cvxpy_solver,
        )
        solver_name = f"CVXPY-{cvxpy_solver}"
    else:
        if want_cvxpy and not _HAS_CVXPY:
            logger.warning("CVXPY requested but not installed — using scipy")
        rho_opt, eps_opt, obj_val, success, n_ev = _solve_p_split_scipy(
            beta_hat, g_tilde, e_sens, c_vec, gamma_lin, noise_f, xi,
        )
        solver_name = "scipy-trust-constr"

    # ── Evaluate SINR from model ───────────────────────────────────────────
    S = _signal_power(rho_opt, beta_hat)
    D = _interf_power(rho_opt, g_tilde, e_sens, noise_f)
    sinr_est = S / np.maximum(D, 1e-30)

    rho_dict = {a_idx: float(rho_opt[i])
                for i, a_idx in enumerate(tx_ap_indices)}

    logger.debug(
        "P-Split: obj=%.4f  converged=%s  ρ=[%s]  slack_max=%.2e  "
        "min_sinr=%.1f dB  solver=%s",
        obj_val, success,
        ", ".join(f"{v:.3f}" for v in rho_opt),
        np.max(eps_opt),
        lin2db(np.min(sinr_est)),
        solver_name,
    )

    return SplitOptResult(
        rho_opt=rho_dict, objective=obj_val,
        sinr_u_est=sinr_est, slack=eps_opt,
        converged=success, n_evals=n_ev, solver=solver_name,
    )


# =============================================================================
# Public API — full CORDIS-Split pipeline
# =============================================================================

def run_cordis_split(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    est: EstimationResult,
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    sigma_n_sq: float,
    Pmax: float,
    comm_bf_method: str = "lr_mmse",
    gamma_u_db: Optional[NDArray[np.float64]] = None,
    omega: Optional[Dict[int, float]] = None,
    use_cvxpy: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[int, NDArray[np.complex128]], PhaseIResult, SplitOptResult]:
    """
    Full CORDIS-Split: Phase I → compressed scalars → P-Split → W_tx.

    Returns
    -------
    W_tx      : dict[ap_idx → (M_t, D)]  final beamforming matrices
    phase_i   : PhaseIResult
    split_res : SplitOptResult
    """
    phase_i = design_phase_i(
        topo, cfg, est, sensing_stats, association,
        comm_bf_method=comm_bf_method, omega=omega, rng=rng,
    )
    scalars = compute_split_scalars(
        phase_i, est, topo, sensing_stats, association,
        omega=omega, kappa=cfg.algorithm.split.kappa,
    )
    split_res = solve_p_split(
        scalars, topo, cfg, sigma_n_sq, Pmax,
        gamma_u_db=gamma_u_db, use_cvxpy=use_cvxpy,
    )
    W_tx = phase_i.build_W_tx(split_res.rho_opt, Pmax)
    return W_tx, phase_i, split_res

