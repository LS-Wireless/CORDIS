"""
cordis/algorithms/centralized.py
=================================
Centralized benchmark for joint BF + PA optimization.

Solves P-Global (eq. admm-p-global) by collecting all APs' beamformers
into a single joint optimisation.  This provides the performance upper
bound for the distributed CORDIS-ADMM algorithm.

The sensing objective ||a^H W||² is non-convex; we apply Successive
Convex Approximation (SCA) with a first-order Taylor expansion to
obtain a concave surrogate at each iteration.  The SINR constraints
are handled via either:
  - CVXPY SOC formulation (primary, when CVXPY is installed)
  - scipy trust-constr with direct SINR inequality (fallback)

Usage::

    W_tx, result = run_centralized(topo, cfg, est, ...)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
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
from cordis.algorithms.beamforming import design_phase_i, PhaseIResult
from cordis.metrics.sinr import compute_sinr

logger = get_logger(__name__)

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


# =============================================================================
# Module constants
# =============================================================================

# Iteration cap for the inner CVXPY SOCP at each SCA step.
#
# Default solver caps are too tight for this problem size (CLARABEL ≈ 200,
# SCS ≈ 2500): when the inner solve hits its limit, CVXPY returns
# ``status="user_limit"``, the SCA outer loop falls back to the previous
# iterate, and convergence stalls.  Empirically 10 000 is generous enough
# for problems up to ~10 APs × 16 antennas × 8 streams without slowing
# typical solves (CLARABEL terminates well before this cap when the
# problem is well-conditioned).
_CVXPY_INNER_MAX_ITER: int = 10_000


def _cvxpy_solve_opts(solver: str) -> Dict[str, object]:
    """
    Return solver-specific keyword arguments for ``prob.solve``.

    The iteration-cap keyword name differs between solvers
    (``max_iter`` for CLARABEL, ``max_iters`` for SCS), so we route on
    the solver name.  Unknown solvers receive only ``verbose=False`` —
    safe defaults that CVXPY accepts universally.
    """
    opts: Dict[str, object] = {"verbose": False}
    if solver == "CLARABEL":
        opts["max_iter"] = _CVXPY_INNER_MAX_ITER
    elif solver == "SCS":
        opts["max_iters"] = _CVXPY_INNER_MAX_ITER
    return opts


# =============================================================================
# Result container
# =============================================================================

@dataclass
class CentralizedResult:
    """
    Output of the centralized benchmark solver.

    W_tx              : dict[ap_idx → (M_t, D)] optimised beamformers
    objective_history : list of true sensing utility values per SCA iteration
    sinr_history      : list of per-user SINR (linear) per SCA iteration
    power_history     : list of per-AP power utilization ratios per SCA iter
    converged         : SCA loop converged
    n_sca_iters       : number of SCA iterations taken
    solver            : solver name used for inner problem
    """
    W_tx:              Dict[int, NDArray[np.complex128]]
    objective_history: List[float]
    sinr_history:      List[NDArray[np.float64]]    = field(default_factory=list)
    power_history:     List[NDArray[np.float64]]    = field(default_factory=list)
    converged:         bool                          = False
    n_sca_iters:       int                           = 0
    solver:            str                           = ""
    feasible:          bool                          = True
    inner_failures:    int                           = 0


# =============================================================================
# Vectorisation helpers (complex W ↔ real vector)
# =============================================================================

def _W_to_vec(W_dict: Dict[int, NDArray], ap_order: List[int]) -> NDArray:
    """Stack all complex W matrices into a single real vector."""
    parts = []
    for a in ap_order:
        W = W_dict[a]
        parts.append(W.real.ravel())
        parts.append(W.imag.ravel())
    return np.concatenate(parts)


def _vec_to_W(
    x: NDArray, ap_order: List[int], Mt: int, D: int,
) -> Dict[int, NDArray[np.complex128]]:
    """Unpack real vector back into dict of complex matrices."""
    blk = Mt * D
    W_dict = {}
    for i, a in enumerate(ap_order):
        base = i * 2 * blk
        W_re = x[base:base + blk].reshape(Mt, D)
        W_im = x[base + blk:base + 2 * blk].reshape(Mt, D)
        W_dict[a] = (W_re + 1j * W_im).astype(np.complex128)
    return W_dict


# =============================================================================
# Sensing objective helpers
# =============================================================================

def _compute_sca_gradient(
    W_prev: Dict[int, NDArray],
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    topo: NetworkTopology,
    omega: Dict[int, float],
    n_snapshots: int,
) -> Dict[int, NDArray[np.complex128]]:
    """
    Compute the SCA gradient matrix G_{a_t} for each TX AP.

    G_{a_t} = Σ_t  ω̄_t β̄_{a_t}^t  a_{a_t t}  a_{a_t t}^H  W_{a_t}^{(n)}

    where ω̄_t = ω_t T σ²_RCS  and  β̄_{a_t}^t = Σ_{a_r} β_{a_t a_r}^t.
    """
    G: Dict[int, NDArray] = {}
    sr_sq = sensing_stats.sigma_rcs_sq
    n_t   = topo.n_targets

    for ap in topo.tx_aps:
        a_idx = ap.idx
        W = W_prev[a_idx]
        Mt, D = W.shape
        G_a = np.zeros((Mt, D), dtype=complex)

        for tg in range(n_t):
            key_t = (a_idx, tg)
            if key_t not in sensing_stats.a_tx:
                continue
            a_at = sensing_stats.a_tx[key_t]             # (Mt,)
            beta_bar = sum(
                sensing_stats.beta_bistatic.get((a_idx, ar, tg), 0.0)
                for ar in association.rx_aps_for(tg)
            )
            omega_bar = omega.get(tg, 1.0) * n_snapshots * sr_sq
            # Rank-1 outer product:  a a^H W^(n)
            G_a += omega_bar * beta_bar * np.outer(a_at, a_at.conj() @ W)

        G[a_idx] = G_a
    return G


def _sensing_objective(
    W_dict: Dict[int, NDArray],
    G_sca: Dict[int, NDArray],
    sensing_stats: SensingChannelStatistics,
    kappa: float,
) -> float:
    """
    Evaluate the SCA-linearized sensing utility:
        Σ_{a_t} [ 2 Re{tr(G^H W)} − κ tr(W^H C W) ]
    """
    val = 0.0
    for a_idx, W in W_dict.items():
        G = G_sca.get(a_idx)
        if G is not None:
            val += 2.0 * np.real(np.trace(G.conj().T @ W))
        C = sensing_stats.C_tx.get(a_idx)
        if C is not None and kappa > 0:
            val -= kappa * np.real(np.trace(W.conj().T @ C @ W))
    return float(val)


def _true_sensing_utility(
    W_dict: Dict[int, NDArray],
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    topo: NetworkTopology,
    omega: Dict[int, float],
    kappa: float,
    n_snapshots: int,
) -> float:
    """Evaluate the true (non-linearised) sensing utility."""
    sr_sq = sensing_stats.sigma_rcs_sq
    val = 0.0
    for ap in topo.tx_aps:
        a = ap.idx
        W = W_dict[a]
        for tg in range(topo.n_targets):
            key = (a, tg)
            if key not in sensing_stats.a_tx:
                continue
            a_at = sensing_stats.a_tx[key]
            beta_bar = sum(
                sensing_stats.beta_bistatic.get((a, ar, tg), 0.0)
                for ar in association.rx_aps_for(tg)
            )
            omega_bar = omega.get(tg, 1.0) * n_snapshots * sr_sq
            val += omega_bar * beta_bar * float(np.linalg.norm(a_at.conj() @ W) ** 2)
        C = sensing_stats.C_tx.get(a)
        if C is not None and kappa > 0:
            val -= kappa * float(np.real(np.trace(W.conj().T @ C @ W)))
    return val


# =============================================================================
# SINR computation from W matrices (direct, for constraints)
# =============================================================================

def _compute_sinr_direct(
    W_dict: Dict[int, NDArray],
    est: EstimationResult,
    topo: NetworkTopology,
    sigma_n_sq: float,
) -> NDArray[np.float64]:
    """
    Compute per-user SINR directly from W matrices and channel estimates.

    MUI and S2CI use the COHERENT sum across APs:
        MUI_k = |Σ_a ĥ_{a,u}^H w_{a,k}|²
    This is critical for the centralized solver, which achieves
    cross-AP coherent MUI suppression — per-AP terms can be large but
    cancel out coherently.

    Returns (N_ue,) linear-scale SINR.
    """
    n_ue = topo.n_ue
    n_t  = topo.n_targets
    D    = n_ue + n_t
    tx_ap_indices = [ap.idx for ap in topo.tx_aps]

    sinr_arr = np.zeros(n_ue)
    for u in range(n_ue):
        # ── Signal: |Σ_a ĥ^H w_u|²  (coherent sum across APs) ────────────
        sig = 0.0 + 0j
        for a in tx_ap_indices:
            h = est.h_hat[(a, u)]
            sig += h.conj() @ W_dict[a][:, u]
        P_sig = float(abs(sig) ** 2)

        # ── MUI + S2CI: coherent sum across APs, then magnitude squared ──
        P_interf = 0.0
        for k in range(D):
            if k == u:
                continue
            mui_k = 0.0 + 0j
            for a in tx_ap_indices:
                h = est.h_hat[(a, u)]
                mui_k += h.conj() @ W_dict[a][:, k]
            P_interf += float(abs(mui_k) ** 2)

        # ── CSI error: Σ_a tr(W_a^H R̃_{a,u} W_a)  (incoherent — correct) ──
        for a in tx_ap_indices:
            R_tilde = est.R_tilde.get((a, u))
            if R_tilde is not None:
                W = W_dict[a]
                P_interf += float(np.real(np.trace(W.conj().T @ R_tilde @ W)))

        sinr_arr[u] = P_sig / max(P_interf + sigma_n_sq, 1e-30)
    return sinr_arr


# =============================================================================
# scipy inner solver (fallback)
# =============================================================================

def _solve_inner_scipy(
    W_prev: Dict[int, NDArray],
    G_sca: Dict[int, NDArray],
    sensing_stats: SensingChannelStatistics,
    est: EstimationResult,
    topo: NetworkTopology,
    gamma_lin: NDArray[np.float64],
    sigma_n_sq: float,
    Pmax: float,
    kappa: float,
    maxiter: int = 200,
) -> Tuple[Dict[int, NDArray[np.complex128]], bool]:
    """Solve one SCA inner problem via scipy SLSQP + power projection."""
    from scipy.optimize import minimize

    tx_ap_order = [ap.idx for ap in topo.tx_aps]
    n_tx = len(tx_ap_order)
    Mt   = topo.tx_aps[0].n_ant
    D    = topo.n_ue + topo.n_targets
    n_ue = topo.n_ue
    blk  = Mt * D

    x0 = _W_to_vec(W_prev, tx_ap_order)

    def objective(x):
        W = _vec_to_W(x, tx_ap_order, Mt, D)
        return -_sensing_objective(W, G_sca, sensing_stats, kappa)

    def sinr_constraint_u(x, u):
        W = _vec_to_W(x, tx_ap_order, Mt, D)
        sinr = _compute_sinr_direct(W, est, topo, sigma_n_sq)
        return float(sinr[u] - gamma_lin[u])

    def power_constraint_a(x, i):
        base = i * 2 * blk
        return float(Pmax - np.sum(x[base:base + 2 * blk] ** 2))

    constraints = []
    for u in range(n_ue):
        constraints.append({"type": "ineq", "fun": sinr_constraint_u, "args": (u,)})
    for i in range(n_tx):
        constraints.append({"type": "ineq", "fun": power_constraint_a, "args": (i,)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            objective, x0, method="SLSQP",
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-8},
        )

    W_out = _vec_to_W(res.x, tx_ap_order, Mt, D)

    # ── Project onto per-AP power ball (safety net) ────────────────────────
    for a in tx_ap_order:
        W = W_out[a]
        p = np.linalg.norm(W, "fro") ** 2
        if p > Pmax:
            W_out[a] = W * np.sqrt(Pmax / p)

    return W_out, bool(res.success)


# =============================================================================
# CVXPY inner solver (primary)
# =============================================================================

def _solve_inner_cvxpy(
    W_prev: Dict[int, NDArray],
    G_sca: Dict[int, NDArray],
    sensing_stats: SensingChannelStatistics,
    est: EstimationResult,
    topo: NetworkTopology,
    gamma_lin: NDArray[np.float64],
    sigma_n_sq: float,
    Pmax: float,
    kappa: float,
    solver: str = "CLARABEL",
) -> Tuple[Dict[int, NDArray[np.complex128]], bool]:
    """
    Solve one SCA inner problem via CVXPY SOCP with unit-power normalization.

    Returns
    -------
    W_dict  : updated beamformers (or W_prev on failure)
    success : True iff the inner SOCP was solved to optimality

    Numerical conditioning
    ----------------------
    Raw values in this problem span ~22 orders of magnitude (path-loss
    factors make β̄ ~ 1e-13, σ² ~ 1e-13, P_max ~ 1e-4).  Without
    rescaling, the solver operates below its precision threshold and
    returns near-zero W ≈ 0 with random oscillation.

    We substitute  W = √P_max · W̃,  ‖W̃‖²_F ≤ 1, and divide all
    constraint terms by σ², giving:

        SOC:   ‖ṽ_u‖₂  ≤  Re{s̃_u} / √γ_u
        where  s̃_u = √(P_max/σ²) · Σ_a ĥ_au^H W̃_a[:,u]
        and    ṽ_u stacks rescaled MUI/S2CI scalars, R̃̃^{1/2} W̃ vectors,
               and a unit noise entry (since σ²/σ² = 1).

    The SCA objective (in normalized W̃ units) is:

        max  2√P_max · Re{tr(G^H W̃)}  −  κ P_max ‖C^{1/2} W̃‖²_F

    A common multiplicative obj_scale (1/(2|⟨G,W_prev⟩|_max)) brings
    both terms to O(1).
    """
    if not _HAS_CVXPY:
        raise RuntimeError("CVXPY not available")

    tx_ap_order = [ap.idx for ap in topo.tx_aps]
    Mt   = topo.tx_aps[0].n_ant
    D    = topo.n_ue + topo.n_targets
    n_ue = topo.n_ue

    sqrt_pmax = np.sqrt(Pmax)
    snr_scale = Pmax / sigma_n_sq            # P_max / σ²  (large, e.g. 10⁹)
    sqrt_snr  = np.sqrt(snr_scale)

    # ── Variables: unit-norm W̃ ─────────────────────────────────────────────
    W_var = {a: cp.Variable((Mt, D), complex=True) for a in tx_ap_order}

    # ── Power: ‖W̃‖²_F ≤ 1 ──────────────────────────────────────────────
    constraints = []
    for a in tx_ap_order:
        constraints.append(cp.norm(W_var[a], "fro") <= 1.0)

    # ── Per-user SOC SINR (in SNR-normalised units, noise → 1) ───────────
    for u in range(n_ue):
        # Scaled signal  s̃_u = √(P_max/σ²) · Σ ĥ^H W̃[:,u]
        signal = sum(
            (est.h_hat[(a, u)] * sqrt_snr).conj() @ W_var[a][:, u]
            for a in tx_ap_order
        )

        interf_parts = []

        # MUI + S2CI: scaled complex scalars
        for k in range(D):
            if k == u:
                continue
            mui = sum(
                (est.h_hat[(a, u)] * sqrt_snr).conj() @ W_var[a][:, k]
                for a in tx_ap_order
            )
            interf_parts.append(cp.reshape(mui, (1,), order='F'))

        # CSI error: ‖R̃_norm^{1/2} W̃‖_F  with R̃_norm = R̃ · P_max/σ²
        for a in tx_ap_order:
            R_tilde = est.R_tilde.get((a, u))
            if R_tilde is not None:
                R_norm = R_tilde * snr_scale
                R_sqrt = _hermitian_sqrt(R_norm)
                v_csi = cp.reshape(R_sqrt @ W_var[a], (Mt * D,), order='F')
                interf_parts.append(v_csi)

        # Noise → 1 (since σ²/σ² = 1)
        interf_parts.append(cp.Constant(np.array([1.0 + 0j])))

        interf_vec = cp.hstack(interf_parts)

        # SOC: ‖ṽ‖₂ ≤ Re{s̃} / √γ
        constraints.append(
            cp.norm(interf_vec, 2) <= cp.real(signal) / np.sqrt(gamma_lin[u])
        )

    # ── Objective: balanced normalisation so values are O(1) ──────────────
    # Reference scale: typical |⟨G, W_prev⟩| at warm start
    g_ref = 1e-30
    for a in tx_ap_order:
        G = G_sca.get(a)
        if G is not None:
            g_ref = max(g_ref, abs(np.trace(G.conj().T @ W_prev[a])))

    obj_scale = 1.0 / (2.0 * sqrt_pmax * g_ref + 1e-30)

    # Auto-balanced κ per AP (mirrors joint_opt.py's Stage-6c logic).
    #
    # The raw clutter term  κ · Pmax · ‖C^{1/2} W̃‖²  scales with Pmax,
    # whereas the bare sensing gradient term scales as √Pmax · |G·W̃|.
    # At 90 dB SNR (Pmax ≫ 1) the raw clutter term dwarfs the sensing
    # term for any sane κ, so SCA collapses W to "minimize clutter
    # response" — which, in geometries where clutter PAS overlaps the
    # target direction, also kills the target response.
    #
    # Following joint_opt.py: use the per-AP natural sensing magnitude
    #     sensing_ref[a] = |2 Re tr(G^H W_prev)|
    # as the κ multiplier, so user-facing κ = 1 means "clutter penalty
    # ≈ sensing-signal magnitude" — strong but not overwhelming.
    kappa_eff: Dict[int, float] = {}
    for a in tx_ap_order:
        G = G_sca.get(a)
        if G is None:
            kappa_eff[a] = 0.0
        else:
            sensing_ref_a = 2.0 * abs(float(np.real(np.trace(G.conj().T @ W_prev[a]))))
            kappa_eff[a] = kappa * max(sensing_ref_a, 1e-30)

    obj_terms = []
    for a in tx_ap_order:
        G = G_sca.get(a)
        if G is not None:
            # 2 Re{tr(G^H W)} = 2√P_max Re{tr(G^H W̃)}
            obj_terms.append(
                obj_scale * 2.0 * sqrt_pmax
                * cp.real(cp.trace(G.conj().T @ W_var[a]))
            )
        C = sensing_stats.C_tx.get(a)
        if C is not None and kappa_eff[a] > 0:
            C_sqrt = _hermitian_sqrt(C)
            # κ_eff · ‖C^{1/2} W‖² = κ_eff · P_max · ‖C^{1/2} W̃‖²_F
            obj_terms.append(
                -obj_scale * kappa_eff[a] * Pmax
                * cp.square(cp.norm(C_sqrt @ W_var[a], "fro"))
            )

    objective = cp.Maximize(sum(obj_terms)) if obj_terms else cp.Maximize(0)

    # ── Solve ──────────────────────────────────────────────────────────────
    prob = cp.Problem(objective, constraints)

    solver_used = solver
    try:
        prob.solve(solver=solver, **_cvxpy_solve_opts(solver))
    except Exception as e_primary:
        try:
            prob.solve(solver="SCS", **_cvxpy_solve_opts("SCS"))
            solver_used = "SCS"
        except Exception as e_fallback:
            logger.error(
                "CVXPY inner solve failed — returning previous W. "
                "primary[%s]: %s: %s | fallback[SCS]: %s: %s",
                solver,
                type(e_primary).__name__, str(e_primary)[:200],
                type(e_fallback).__name__, str(e_fallback)[:200],
            )
            return W_prev, False

    if prob.status not in ("optimal", "optimal_inaccurate"):
        logger.warning(
            "CVXPY inner solve [%s] status=%s — returning previous W",
            solver_used, prob.status,
        )
        return W_prev, False

    # Check that all W values are present
    for a in tx_ap_order:
        if W_var[a].value is None:
            logger.warning(
                "CVXPY returned None for AP %d — returning previous W", a,
            )
            return W_prev, False

    # ── Denormalize: W = √P_max · W̃ ──────────────────────────────────────
    W_out = {
        a: sqrt_pmax * np.asarray(W_var[a].value, dtype=np.complex128)
        for a in tx_ap_order
    }
    return W_out, True


def _hermitian_sqrt(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Robust Hermitian matrix square root via eigendecomposition.

    Returns S such that S^H S = M (when M is Hermitian PSD).  Handles
    rank-deficient or numerically non-PSD inputs by clipping negative
    eigenvalues to zero.
    """
    M_sym = 0.5 * (M + M.conj().T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.maximum(eigvals.real, 0.0)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T


# =============================================================================
# Public API — centralised SCA solver
# =============================================================================

def solve_centralized(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    est: EstimationResult,
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    sigma_n_sq: float,
    Pmax: float,
    gamma_u_db: Optional[NDArray[np.float64]] = None,
    omega: Optional[Dict[int, float]] = None,
    n_sca_max: int = 20,
    sca_tol: float = 1e-4,
    warm_start_psr: float = 0.5,
    use_cvxpy: Optional[bool] = None,
    sca_damping: float = 0.0,
    verbose: bool = False,
) -> CentralizedResult:
    """
    Centralized benchmark: joint BF+PA via SCA with full network knowledge.

    Parameters
    ----------
    topo, cfg, est, sensing_stats, association : standard pipeline objects
    sigma_n_sq     : noise power [W]
    Pmax           : per-AP power budget [W]
    gamma_u_db     : per-user SINR target [dB] (None → config default)
    omega          : target priority weights (None → equal)
    n_sca_max      : maximum SCA iterations (default 20)
    sca_tol        : relative convergence tolerance on true sensing utility
    warm_start_psr : initial PSR for Phase I warm start
    use_cvxpy      : True/False/None (auto-detect)
    sca_damping    : SCA damping coefficient α ∈ [0, 1).  W_new ← (1-α)W* + αW_prev.
                     0 = un-damped (default).  Increase if SCA oscillates.
    verbose        : print per-iteration diagnostics
    """
    n_ue   = topo.n_ue
    n_t    = topo.n_targets
    kappa  = cfg.algorithm.admm.kappa
    n_snap = cfg.sensing.n_snapshots

    if omega is None:
        omega = {tg: 1.0 for tg in range(n_t)}
    if gamma_u_db is None:
        gamma_lin = db2lin(cfg.algorithm.split.gamma_db) * np.ones(n_ue)
    else:
        gamma_lin = db2lin(np.asarray(gamma_u_db, dtype=float))

    want_cvxpy = use_cvxpy if use_cvxpy is not None else _HAS_CVXPY
    solver_name = ("CVXPY-" + cfg.algorithm.admm.solver) if want_cvxpy else "scipy"

    # ── Warm start from Split Phase I ──────────────────────────────────────
    phase_i = design_phase_i(topo, cfg, est, sensing_stats, association, omega=omega)
    W_cur   = phase_i.build_W_tx_equal_psr(warm_start_psr, Pmax)

    obj_history:   List[float]    = []
    sinr_history:  List[NDArray]  = []
    power_history: List[NDArray]  = []
    converged = False
    tx_ap_order = [ap.idx for ap in topo.tx_aps]

    # ── Infeasibility tracking ────────────────────────────────────────────
    inner_failures        = 0
    any_inner_success     = False
    consecutive_failures  = 0
    MAX_CONSECUTIVE_FAILS = 2     # break SCA loop after this many in a row

    if verbose:
        print(f"  [SCA] solver={solver_name}, n_sca_max={n_sca_max}, "
              f"tol={sca_tol}, γ={lin2db(gamma_lin)}")

    for sca_iter in range(n_sca_max):
        # 1. Compute SCA gradient around current W
        G_sca = _compute_sca_gradient(
            W_cur, sensing_stats, association, topo, omega, n_snap,
        )

        # 2. Solve inner problem
        if want_cvxpy and _HAS_CVXPY:
            W_solve, inner_ok = _solve_inner_cvxpy(
                W_cur, G_sca, sensing_stats, est, topo,
                gamma_lin, sigma_n_sq, Pmax, kappa,
                solver=cfg.algorithm.admm.solver,
            )
        else:
            W_solve, inner_ok = _solve_inner_scipy(
                W_cur, G_sca, sensing_stats, est, topo,
                gamma_lin, sigma_n_sq, Pmax, kappa,
            )

        # Track inner-solve failures
        if not inner_ok:
            inner_failures += 1
            consecutive_failures += 1
        else:
            any_inner_success = True
            consecutive_failures = 0

        # 3. Optional SCA damping for stability
        if sca_damping > 0.0:
            W_new = {
                a: (1.0 - sca_damping) * W_solve[a] + sca_damping * W_cur[a]
                for a in tx_ap_order
            }
        else:
            W_new = W_solve

        # 4. Evaluate diagnostics on the new iterate
        obj_true = _true_sensing_utility(
            W_new, sensing_stats, association, topo, omega, kappa, n_snap,
        )
        sinr_arr  = _compute_sinr_direct(W_new, est, topo, sigma_n_sq)
        power_arr = np.array([
            float(np.linalg.norm(W_new[a], "fro") ** 2 / Pmax)
            for a in tx_ap_order
        ])

        obj_history.append(obj_true)
        sinr_history.append(sinr_arr)
        power_history.append(power_arr)

        if verbose:
            print(f"  [SCA {sca_iter+1:2d}] obj={obj_true:+.4e}  "
                  f"min-SINR={lin2db(np.min(sinr_arr)):+.2f} dB  "
                  f"max-pwr={power_arr.max():.3f}×Pmax  "
                  f"inner_ok={inner_ok}")

        # 5. Bail out if the inner SOCP is persistently infeasible
        if consecutive_failures >= MAX_CONSECUTIVE_FAILS:
            if verbose:
                print(f"  [SCA] aborting after {consecutive_failures} "
                      f"consecutive inner failures (problem likely infeasible)")
            W_cur = W_new
            break

        # 6. Check convergence on the true sensing utility
        if sca_iter > 0:
            prev = obj_history[-2]
            denom = max(abs(prev), 1e-12)
            rel_change = abs(obj_true - prev) / denom
            if rel_change < sca_tol and inner_ok:
                converged = True
                W_cur = W_new
                break

        W_cur = W_new

    # The problem is "feasible" if the inner SOCP succeeded at least once.
    feasible = any_inner_success

    logger.debug(
        "Centralized: %d SCA iters, converged=%s, feasible=%s, "
        "inner_failures=%d, obj_final=%.4e, solver=%s",
        len(obj_history), converged, feasible, inner_failures,
        obj_history[-1] if obj_history else 0.0,
        solver_name,
    )

    return CentralizedResult(
        W_tx=W_cur,
        objective_history=obj_history,
        sinr_history=sinr_history,
        power_history=power_history,
        converged=converged,
        n_sca_iters=len(obj_history),
        solver=solver_name,
        feasible=feasible,
        inner_failures=inner_failures,
    )


# =============================================================================
# Public API — full centralised pipeline
# =============================================================================

def run_centralized(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    est: EstimationResult,
    sensing_stats: SensingChannelStatistics,
    association: SensingAssociation,
    sigma_n_sq: float,
    Pmax: float,
    gamma_u_db: Optional[NDArray[np.float64]] = None,
    omega: Optional[Dict[int, float]] = None,
    use_cvxpy: Optional[bool] = None,
    **kwargs,
) -> Tuple[Dict[int, NDArray[np.complex128]], CentralizedResult]:
    """
    Full centralized pipeline: warm-start → SCA solve → W_tx.

    Returns
    -------
    W_tx   : dict[ap_idx → (M_t, D)]  optimised beamformers
    result : CentralizedResult
    """
    result = solve_centralized(
        topo, cfg, est, sensing_stats, association,
        sigma_n_sq, Pmax,
        gamma_u_db=gamma_u_db, omega=omega,
        use_cvxpy=use_cvxpy, **kwargs,
    )
    return result.W_tx, result

