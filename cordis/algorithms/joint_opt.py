"""
cordis/algorithms/joint_opt.py
================================
Stage 6c — CORDIS-ADMM (Journal Paper formulation, Section 5).

Decentralised joint BF + PA via consensus ADMM with structured per-user
contribution vectors.  Unlike the conference-paper variant, the consensus
variable is *per-user* and *vector-valued*; the SOC constraint that
enforces SINR is solved entirely at the CPU, not approximated locally
through Minkowski inequalities.

Per-user local contribution at AP a_t (paper eq. admm-local-contribution):

    l_{a_t,u}(W_a) = [ x_{a_t,u},  (i^(c)_{a_t,u})^T,  (i^(s)_{a_t,u})^T,  e_{a_t,u} ]

where
    x_{a_t,u}     = ĥ_{a_t u}^H W_a[:, u]                         (complex scalar)
    i^(c)_{a_t,u} = col_{k≠u} { ĥ_{a_t u}^H W_a[:, k] }           (complex vector, N_ue-1)
    i^(s)_{a_t,u} = col_{t∈T}{ ĥ_{a_t u}^H W_a[:, n_ue+t] }       (complex vector, N_t)
    e_{a_t,u}     = ‖R̃_{a_t u}^{1/2} W_a‖_F                       (real scalar, ≥ 0)

    NOTE: The paper writes e_au as the SQUARED Frobenius norm (a power).  This
    paper-faithful definition makes (e_au + Σ_err)² quartic in W_a (breaking
    the QCQP claim) AND introduces √z^err inside the CPU SOC, which has no
    DCP-compliant CVXPY encoding (any of `z_err ≤ aux²`, `aux ≥ √z_err`, etc.
    violates CVXPY's `convex ≤ concave` rule).  We instead use the AMPLITUDE
    form e_au = ‖R̃^{1/2} W_a‖_F, which is:
       • DCP-friendly at the CPU: z^err appears directly in the SOC stack, and
         the built-in squaring inside the L2 norm gives (z^err)² as the
         CSI-error contribution to ‖v_u‖².
       • Conservative: by the triangle inequality,
            (z^err)² = (Σ_a ‖R̃^{1/2} W_a‖_F)²  ≥  Σ_a ‖R̃^{1/2} W_a‖²_F
         = true total CSI-error power.  So the SOC upper-bounds the true
         interference, providing the same SINR protection or stronger.

Consensus:                 z_u = Σ_{a_t} l_{a_t,u}.

Augmented Lagrangian:
    L_ρ = Σ_a Ũ_a^sens(W_a) − (ρ/2) Σ_u ‖Σ_a l_{a,u}(W_a) − z_u + ν_u‖²

Per ADMM iteration n:
    Phase I  — Parallel per-AP QCQP:  P-Local (eq. admm-p-local)
                  max Ũ_a^sens(W_a) − (ρ/2) Σ_u ‖l_{a,u}(W_a) + Σ_{a,u}^(n)‖²
                  s.t.  ‖W_a‖²_F ≤ Pmax
                with Σ_{a,u}^(n) = Σ̃_u^(n) − l_{a,u}^(n).

    Phase II — Per-user CPU SOCP projection:  P-Central (eq. admm-p-central)
                  min ‖Σ_a l_{a,u}^(n+1) + ν_u^(n) − z_u‖² + ξ ε_u
                  s.t.  ‖[z^MUI; z^S2CI; z^err; 1]‖_2  ≤  Re{z^CDS}/√γ + ε_u,
                        Im{z_u^CDS} = 0,  z_u^err ≥ 0,  ε_u ≥ 0.

    Dual:  ν_u^(n+1) = ν_u^(n) + Σ_a l_{a,u}^(n+1) − z_u^(n+1).
    Stop:  r_pri, r_dual below tolerances.

Numerical handling
------------------
1. Same unit-power / SNR-normalisation as Stage 6b: W̃_a = W_a/√Pmax,
   ĥ̃ = ĥ √(Pmax/σ²), R̃̃ = R̃ Pmax/σ², σ_n → 1.  All consensus quantities
   live in SNR-amplitude units (CDS/MUI/S2CI/err all amplitudes).

2. The e_au term is the Frobenius norm of an affine function of W_a — convex
   but not affine.  Direct CVXPY use of (e_au + Σ_err)² would not be DCP.
   We SCA-linearise around the previous iterate W̃_a^(n):
       T(W̃) = Re{tr((W̃^(n))^H R̃̃ W̃)} / f_n,    f_n = ‖R̃̃^{1/2} W̃^(n)‖_F.
   T is affine, T(W̃^(n)) = f_n (exact at the expansion point), so the local
   QCQP is a true quadratic program with quadratic constraints.
   The *exact* e_au^(n+1) = ‖R̃̃^{1/2} W̃_a^(n+1)‖_F is computed post-solve
   for transmission to the CPU.

3. The SOC at the CPU is a clean L2-norm inequality with z^err in the cone
   stack directly — no √, no rotated cone, no auxiliary variable.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# CLARABEL emits a UserWarning ("Solution may be inaccurate ...") when it
# returns status `optimal_inaccurate`.  The solver wrappers in this module
# explicitly accept that status (the residuals are still tiny in absolute
# terms; only the strict tolerance was not hit), so the warning adds noise
# without conveying actionable information for end users.
warnings.filterwarnings(
    "ignore",
    message=r"Solution may be inaccurate.*",
    category=UserWarning,
)

from cordis.algorithms.beamforming import design_phase_i
from cordis.algorithms.centralized import (
    _compute_sca_gradient, _hermitian_sqrt,
)
from cordis.channel.estimation import EstimationResult
from cordis.channel.sensing_assignment import SensingAssociation
from cordis.channel.sensing_channel import SensingChannelStatistics
from cordis.channel.topology import NetworkTopology
from cordis.utils.config import CORDISConfig
from cordis.utils.math_utils import db2lin, lin2db

logger = logging.getLogger("cordis." + __name__)

try:
    import cvxpy as cp  # noqa: F401
    _HAS_CVXPY = True
except ImportError:
    cp = None  # type: ignore
    _HAS_CVXPY = False


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class LocalContribution:
    """
    Per-(AP, user) contribution l_{a_t, u} in SNR-normalised units.

    cds  : complex scalar     — ĥ̃^H W̃_a[:,u]
    mui  : complex (N_ue-1,)  — ĥ̃^H W̃_a[:, k≠u]
    s2ci : complex (N_t,)     — ĥ̃^H W̃_a[:, n_ue+t]
    err  : real scalar        — ‖R̃̃^{1/2} W̃_a‖_F   (AMPLITUDE form, ≥ 0)

    NOTE on the err component: the journal paper defines e_au = ‖R̃^{1/2} W_a‖²_F
    (power).  However, this leads to (e_au + const)² being quartic in W_a inside
    the ADMM penalty (breaking the QCQP claim) AND requires √z^err inside the
    SOC at the CPU which cannot be encoded as a DCP-compliant CVXPY constraint
    (any of `z_err ≤ aux²`, `aux ≥ √z_err`, etc. fails DCP).

    We therefore work with the AMPLITUDE form  err = ‖R̃^{1/2} W_a‖_F  throughout:
        • At the CPU, the SOC stack contains z^err directly (no √), and the
          built-in squaring inside ‖·‖² yields (z^err)² as the CSI-error term.
          Since z^err = Σ_a ‖R̃^{1/2} W_a‖_F, by the triangle inequality
          (z^err)² ≥ Σ_a ‖R̃^{1/2} W_a‖²_F = true total CSI-error power.
          So the SOC is a CONSERVATIVE (tighter) upper bound on the original
          SINR constraint — same protection, slightly more pessimistic.
        • At the AP, the err component is convex (a norm), so the consensus
          penalty (err + Σ_err)² is not DCP directly; we use an SCA tangent
          to linearise it, matching the SCA scheme already used for sensing.
    """
    cds:  complex
    mui:  NDArray[np.complex128]
    s2ci: NDArray[np.complex128]
    err:  float


@dataclass
class ConsensusVector:
    """Global per-user consensus vector z_u (same fields as LocalContribution).

    All components live in SNR-normalised units; err is an AMPLITUDE (≥ 0).
    """
    cds:  complex
    mui:  NDArray[np.complex128]
    s2ci: NDArray[np.complex128]
    err:  float


@dataclass
class ADMMResult:
    """
    Output of CORDIS-ADMM (Stage 6c).

    Histories are appended once per outer iteration.
    """
    W_tx:                 Dict[int, NDArray[np.complex128]]
    primal_res_history:   List[float] = field(default_factory=list)
    dual_res_history:     List[float] = field(default_factory=list)
    sensing_obj_history:  List[float] = field(default_factory=list)
    sinr_history:         List[NDArray[np.float64]] = field(default_factory=list)
    slack_history:        List[NDArray[np.float64]] = field(default_factory=list)
    z_norm_history:       List[float] = field(default_factory=list)
    converged:            bool        = False
    n_admm_iters:         int         = 0
    solver:               str         = ""
    feasible:             bool        = True
    inner_failures:       int         = 0


# =============================================================================
# Local contribution helpers (in SNR-normalised units)
# =============================================================================

def _compute_local_contribution(
    a_idx: int,
    u: int,
    W_a_tilde: NDArray[np.complex128],     # W̃_a (unit-power)
    h_tilde:   NDArray[np.complex128],     # ĥ̃_{a u} (SNR-amplitude)
    R_tilde_norm: Optional[NDArray[np.complex128]],   # R̃̃ = R̃ × Pmax/σ²
    n_ue: int,
    n_t:  int,
) -> LocalContribution:
    """
    Compute l_{a_t, u}(W̃_a) in SNR-normalised units.

    Used both inside the local QCQP (for the CVXPY penalty terms) and
    post-solve to ship the new l^(n+1) to the CPU.  Here we use numpy
    on a concrete W̃_a; the CVXPY version is inlined in the QCQP build.
    """
    # x_au:  ĥ̃^H W̃[:, u]
    cds = complex(h_tilde.conj() @ W_a_tilde[:, u])

    # i^(c)_au:  ĥ̃^H W̃[:, k] for k ∈ U \ {u}
    k_idx = [k for k in range(n_ue) if k != u]
    if k_idx:
        mui = np.asarray(h_tilde.conj() @ W_a_tilde[:, k_idx], dtype=np.complex128)
    else:
        mui = np.zeros(0, dtype=np.complex128)

    # i^(s)_au:  ĥ̃^H W̃[:, n_ue + t] for t ∈ T
    if n_t > 0:
        s2ci = np.asarray(
            h_tilde.conj() @ W_a_tilde[:, n_ue:n_ue + n_t], dtype=np.complex128,
        )
    else:
        s2ci = np.zeros(0, dtype=np.complex128)

    # e_au:  ‖R̃̃^{1/2} W̃_a‖_F  (AMPLITUDE form; sqrt of squared Frobenius)
    if R_tilde_norm is not None:
        q_val = float(np.real(np.trace(W_a_tilde.conj().T @ R_tilde_norm @ W_a_tilde)))
        err_val = float(np.sqrt(max(q_val, 0.0)))
    else:
        err_val = 0.0

    return LocalContribution(cds=cds, mui=mui, s2ci=s2ci, err=err_val)


def _zeros_consensus(n_ue_minus_1: int, n_t: int) -> ConsensusVector:
    return ConsensusVector(
        cds=0.0 + 0.0j,
        mui=np.zeros(n_ue_minus_1, dtype=np.complex128),
        s2ci=np.zeros(n_t, dtype=np.complex128),
        err=0.0,
    )


def _l_norm_sq(l: LocalContribution) -> float:
    """‖l‖² in mixed-type space (treating err as a real coordinate)."""
    s  = float(abs(l.cds) ** 2)
    s += float(np.sum(np.abs(l.mui)  ** 2))
    s += float(np.sum(np.abs(l.s2ci) ** 2))
    s += float(l.err ** 2)
    return s


def _sum_local_contributions(
    l_dict: Dict[Tuple[int, int], LocalContribution],
    tx_ap_indices: List[int],
    u: int,
    n_ue_minus_1: int,
    n_t: int,
) -> ConsensusVector:
    """Σ_{a_t} l_{a_t, u}  →  acts as the empirical consensus target."""
    out = _zeros_consensus(n_ue_minus_1, n_t)
    for a in tx_ap_indices:
        l = l_dict[(a, u)]
        out.cds  += l.cds
        out.mui  += l.mui
        out.s2ci += l.s2ci
        out.err  += l.err
    return out


# =============================================================================
# Phase I — Local QCQP at each TX AP
# =============================================================================

def _solve_local_qcqp(
    a_idx:           int,
    W_a_prev:        NDArray[np.complex128],
    G_sca_a:         Optional[NDArray[np.complex128]],
    sensing_stats:   SensingChannelStatistics,
    est:             EstimationResult,
    topo:            NetworkTopology,
    Sigma_au:        Dict[int, ConsensusVector],   # Σ_{a,u}^(n)  per user (constants)
    e_au_prev:       Dict[int, float],              # f_n = ‖R̃̃^{1/2} W̃^(n)‖_F per user
    sigma_n_sq:      float,
    Pmax:            float,
    rho_effective:   float,
    kappa_effective: float,
    sensing_ref:     float,
    solver:          str,
) -> Tuple[NDArray[np.complex128], bool]:
    """
    Solve the per-AP QCQP P-Local (eq. admm-p-local):

        max  Ũ_a^sens(W̃_a)  −  (ρ_eff/2) Σ_u ‖l_au(W̃_a) + Σ_au‖²
        s.t. ‖W̃_a‖²_F ≤ 1

    where l_au components are computed in SNR-normalised units, and the
    quadratic-in-W̃ e_au is replaced by its SCA tangent around W̃_prev.

    Both ``rho_effective`` and ``kappa_effective`` are pre-scaled to the
    natural problem dimensions; see solve_cordis_admm for the derivation
    from user-facing rho_admm and kappa.

    Returns (W_a_new_physical, success).  On failure returns (W_a_prev, False).
    """
    if not _HAS_CVXPY:
        raise RuntimeError("Stage 6c CORDIS-ADMM requires CVXPY.")

    n_ue   = topo.n_ue
    n_t    = topo.n_targets
    D      = n_ue + n_t
    Mt     = topo.tx_aps[0].n_ant
    sigma_n = float(np.sqrt(sigma_n_sq))

    # ── Normalisation ────────────────────────────────────────────────────
    sqrt_pmax = float(np.sqrt(Pmax))
    snr_scale = Pmax / sigma_n_sq
    sqrt_snr  = float(np.sqrt(snr_scale))
    W_tilde_prev = W_a_prev / sqrt_pmax        # SNR-units; ‖W̃_prev‖²_F ≤ 1

    # ── Variable ─────────────────────────────────────────────────────────
    W_var = cp.Variable((Mt, D), complex=True)

    constraints = [cp.norm(W_var, "fro") <= 1.0]

    # ── Sensing SCA tangent (reuse G_sca_a built in physical units) ──────
    # 2 Re{tr(G^H W)} = 2 √Pmax Re{tr(G^H W̃)}.  Clutter term uses pre-scaled
    # kappa_effective so the user-facing kappa=1 means "clutter penalty equals
    # sensing reward at warm start"; without that scaling, raw κ·Pmax·M ~ 10⁻³
    # would dwarf raw sensing ~ 10⁻¹⁴ for any sane κ.
    if G_sca_a is not None:
        sensing_term = 2.0 * sqrt_pmax * cp.real(cp.trace(G_sca_a.conj().T @ W_var))
        C = sensing_stats.C_tx.get(a_idx)
        if C is not None and kappa_effective > 0:
            C_sqrt = _hermitian_sqrt(C)
            sensing_term = (
                sensing_term
                - kappa_effective
                * cp.square(cp.norm(C_sqrt @ W_var, "fro"))
            )
    else:
        sensing_term = cp.Constant(0.0)

    # ── Per-user consensus penalty ──────────────────────────────────────
    penalty_per_u = []

    for u in range(n_ue):
        sigma_au_u = Sigma_au[u]            # constant ConsensusVector

        # Channel for AP a, user u (SNR-amplitude)
        h_tilde = est.h_hat[(a_idx, u)] * sqrt_snr     # (Mt,)

        # ---- CDS:  |x̃_au + Σ_cds|² ------------------------------------
        x_cvx     = h_tilde.conj() @ W_var[:, u]                     # complex scalar
        cds_resid = x_cvx + complex(sigma_au_u.cds)
        pen_cds   = cp.square(cp.abs(cds_resid))

        # ---- MUI:  ‖i_c̃_au + Σ_mui‖² ----------------------------------
        k_idx = [k for k in range(n_ue) if k != u]
        if k_idx:
            mui_cvx   = h_tilde.conj() @ W_var[:, k_idx]              # (n_ue-1,)
            mui_resid = mui_cvx + sigma_au_u.mui
            pen_mui   = cp.sum_squares(cp.abs(mui_resid))
        else:
            pen_mui = cp.Constant(0.0)

        # ---- S2CI:  ‖i_s̃_au + Σ_s2ci‖² --------------------------------
        if n_t > 0:
            s2ci_cvx   = h_tilde.conj() @ W_var[:, n_ue:n_ue + n_t]    # (n_t,)
            s2ci_resid = s2ci_cvx + sigma_au_u.s2ci
            pen_s2ci   = cp.sum_squares(cp.abs(s2ci_resid))
        else:
            pen_s2ci = cp.Constant(0.0)

        # ---- err:  (ẽ_au + Σ_err)²  with SCA tangent on AMPLITUDE ───────
        # The err component is the Frobenius norm  f(W̃) = ‖R̃̃^{1/2} W̃‖_F
        # which is convex but not affine.  SCA-linearise at W̃_prev:
        #     T(W̃) = Re{tr((W̃_prev)^H R̃̃ W̃)} / f_n,    f_n = ‖R̃̃^{1/2} W̃_prev‖_F
        # T is affine in W̃; T(W̃_prev) = f_n (exact at the tangent point).
        R_tilde = est.R_tilde.get((a_idx, u))
        if R_tilde is not None:
            R_norm  = R_tilde * snr_scale                              # R̃̃
            f_n     = float(e_au_prev[u])                              # ‖R̃̃^{1/2} W̃^(n)‖_F
            if f_n > 1e-15:
                err_tangent = (
                    cp.real(cp.trace(W_tilde_prev.conj().T @ R_norm @ W_var))
                    / f_n
                )
                err_resid = err_tangent + float(sigma_au_u.err)
                pen_err   = cp.square(err_resid)
            else:
                # Degenerate case (zero CSI error at W̃_prev); penalty drops out.
                pen_err = cp.Constant(0.0)
        else:
            pen_err = cp.Constant(0.0)

        penalty_per_u.append(pen_cds + pen_mui + pen_s2ci + pen_err)

    total_penalty = cp.sum(cp.hstack(penalty_per_u)) if penalty_per_u else cp.Constant(0.0)

    # ── Objective ────────────────────────────────────────────────────────
    # Sensing utility is normalised to O(1) at the warm start (sensing_ref ≈
    # |2√Pmax Re tr(G^H W_warm)| ~ 10⁻¹⁴ in physical units).  The penalty is
    # left in raw SNR-squared units; the caller folds the natural problem
    # scale into ``rho_effective`` so that rho_admm = 1 corresponds to
    # balanced weighting between sensing and consensus at a typical
    # residual ≈ 1 (in SNR amplitude units).  See solve_cordis_admm for
    # the explicit auto-rho derivation.
    obj = (
        sensing_term / max(sensing_ref, 1e-30)
        - 0.5 * rho_effective * total_penalty
    )

    prob = cp.Problem(cp.Maximize(obj), constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception as e_primary:
        try:
            prob.solve(solver="SCS", verbose=False)
        except Exception as e_fb:
            logger.error(
                "AP %d local QCQP failed.  primary[%s]: %s | fallback[SCS]: %s",
                a_idx, solver, str(e_primary)[:160], str(e_fb)[:160],
            )
            return W_a_prev, False

    if prob.status not in ("optimal", "optimal_inaccurate") or W_var.value is None:
        logger.debug("AP %d local QCQP status=%s", a_idx, prob.status)
        return W_a_prev, False

    W_a_new = sqrt_pmax * np.asarray(W_var.value, dtype=np.complex128)
    return W_a_new, True


# =============================================================================
# Phase II — CPU SOCP projection per user
# =============================================================================

def _solve_central_socp(
    u: int,
    sum_l_u:    ConsensusVector,        # Σ_{a_t} l_{a_t, u}^(n+1)  (constant)
    nu_u:       ConsensusVector,        # ν_u^(n)  (constant)
    gamma_u:    float,                  # SINR target (linear)
    n_ue_minus_1: int,
    n_t:        int,
    xi_slack:   float,
    solver:     str,
    nu_sinr_u:  float = 0.0,            # SOC-slack ALM multiplier
) -> Tuple[ConsensusVector, float, bool]:
    """
    Solve the per-user CPU SOCP P-Central (eq. admm-p-central) in
    SNR-normalised units (noise → 1), using the AMPLITUDE convention for
    z_u^err (see LocalContribution docstring for the rationale):

        min  ‖sum_l_u + nu_u − z_u‖²
             + ν^sinr_u · ε_u + (ξ/2) · ε_u²        ← Augmented Lagrangian
        s.t. ‖[z^MUI; z^S2CI; z^err; 1]‖_2  ≤  Re{z^CDS}/√γ_u  +  ε_u
             Im{z_u^CDS} = 0,   z_u^err ≥ 0,   ε_u ≥ 0

    The Augmented Lagrangian on ε_u (Method of Multipliers, Bertsekas
    §4.2.4) supersedes the pure linear penalty ξ·ε of the exterior
    penalty method.  The multiplier ν^sinr_u is updated outside the
    SOCP via  ν^sinr ← max(0, ν^sinr + ξ · ε*),  which accumulates the
    constraint violation and drives ε → 0 at *fixed* ξ — unlike pure
    penalty which would need ξ → ∞.  The quadratic ε² term is the
    augmentation that gives linear convergence under regularity.

    The squaring inside ‖·‖² gives (z^err)² as the CSI-error contribution,
    consistent with z^err being an aggregate amplitude (≥ Σ_a ‖R̃^{1/2} W_a‖_F).
    Returns (z_u, eps_u, success).
    """
    # CVXPY variables — note that putting z^err directly in the SOC stack
    # (no √-aux, no rotated-cone) is the key DCP-safe encoding here.
    z_cds_re = cp.Variable()                                # Im{z_cds}=0 ⇒ real var
    z_mui    = cp.Variable(n_ue_minus_1, complex=True) if n_ue_minus_1 > 0 else None
    z_s2ci   = cp.Variable(n_t, complex=True)                if n_t > 0           else None
    z_err    = cp.Variable(nonneg=True)                      # AMPLITUDE, ≥ 0
    eps_u    = cp.Variable(nonneg=True)

    constraints: List = []

    # ── Build SOC stack: [Re/Im of z_MUI, z_S2CI; z_err; 1]  (all affine) ──
    parts = []
    if z_mui is not None:
        parts.append(cp.real(z_mui))
        parts.append(cp.imag(z_mui))
    if z_s2ci is not None:
        parts.append(cp.real(z_s2ci))
        parts.append(cp.imag(z_s2ci))
    parts.append(cp.reshape(z_err, (1,), order='C'))         # scalar z_err in stack
    parts.append(np.array([1.0]))                            # σ_n → 1 (normalised)
    soc_vec = cp.hstack(parts)

    # ‖soc_vec‖_2  ≤  Re{z_CDS}/√γ + ε_u    (standard convex SOC, DCP-OK)
    constraints.append(
        cp.norm(soc_vec, 2) <= z_cds_re / float(np.sqrt(gamma_u)) + eps_u
    )

    # ── Objective:  ‖sum_l + nu − z‖²  + ξ ε ──
    #   CDS residual is complex; with z_cds_re real we split real/imag:
    cds_const   = complex(sum_l_u.cds + nu_u.cds)
    pen_cds     = cp.square(np.real(cds_const) - z_cds_re) + (np.imag(cds_const)) ** 2

    if z_mui is not None:
        mui_const = (sum_l_u.mui + nu_u.mui).astype(np.complex128)
        pen_mui   = cp.sum_squares(cp.abs(mui_const - z_mui))
    else:
        pen_mui = cp.Constant(0.0)

    if z_s2ci is not None:
        s2ci_const = (sum_l_u.s2ci + nu_u.s2ci).astype(np.complex128)
        pen_s2ci   = cp.sum_squares(cp.abs(s2ci_const - z_s2ci))
    else:
        pen_s2ci = cp.Constant(0.0)

    err_const = float(sum_l_u.err + nu_u.err)
    pen_err   = cp.square(err_const - z_err)

    # Augmented Lagrangian penalty on the SOC slack ε_u:
    #     ν^sinr_u · ε_u  +  (ξ/2) · ε_u²
    # Multiplier ν^sinr_u is accumulated by the outer loop after this
    # solve; ξ stays bounded.  Both terms are convex in ε_u and accepted
    # by CLARABEL/SCS without reformulation.
    slack_pen = nu_sinr_u * eps_u + 0.5 * xi_slack * cp.square(eps_u)
    obj = pen_cds + pen_mui + pen_s2ci + pen_err + slack_pen
    prob = cp.Problem(cp.Minimize(obj), constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception as e_primary:
        try:
            prob.solve(solver="SCS", verbose=False)
        except Exception as e_fb:
            logger.error(
                "CPU SOCP (u=%d) failed. primary[%s]: %s | fallback[SCS]: %s",
                u, solver, str(e_primary)[:160], str(e_fb)[:160],
            )
            return _zeros_consensus(n_ue_minus_1, n_t), 0.0, False

    if prob.status not in ("optimal", "optimal_inaccurate") or z_cds_re.value is None:
        logger.debug("CPU SOCP (u=%d) status=%s", u, prob.status)
        return _zeros_consensus(n_ue_minus_1, n_t), 0.0, False

    z_out = ConsensusVector(
        cds  = complex(float(z_cds_re.value), 0.0),
        mui  = (np.asarray(z_mui.value,  dtype=np.complex128)
                if z_mui is not None else np.zeros(0, dtype=np.complex128)),
        s2ci = (np.asarray(z_s2ci.value, dtype=np.complex128)
                if z_s2ci is not None else np.zeros(0, dtype=np.complex128)),
        err  = float(z_err.value),
    )
    eps_out = float(eps_u.value) if eps_u.value is not None else 0.0
    return z_out, eps_out, True


# =============================================================================
# Residuals & SINR (helpers)
# =============================================================================

def _consensus_residual(l_sum: ConsensusVector, z: ConsensusVector) -> float:
    """‖Σ_a l_au − z_u‖₂ in SNR-normalised mixed-type space."""
    r2 = abs(l_sum.cds - z.cds) ** 2
    r2 += float(np.sum(np.abs(l_sum.mui - z.mui) ** 2))
    r2 += float(np.sum(np.abs(l_sum.s2ci - z.s2ci) ** 2))
    r2 += (l_sum.err - z.err) ** 2
    return float(np.sqrt(r2))


def _z_diff_norm(z_new: ConsensusVector, z_old: ConsensusVector) -> float:
    """‖z^(n+1) − z^(n)‖₂."""
    d2 = abs(z_new.cds - z_old.cds) ** 2
    d2 += float(np.sum(np.abs(z_new.mui - z_old.mui) ** 2))
    d2 += float(np.sum(np.abs(z_new.s2ci - z_old.s2ci) ** 2))
    d2 += (z_new.err - z_old.err) ** 2
    return float(np.sqrt(d2))


def _compute_sinr_coherent(
    W_dict: Dict[int, NDArray],
    est:    EstimationResult,
    topo:   NetworkTopology,
    sigma_n_sq: float,
) -> NDArray[np.float64]:
    """
    Per-user SINR with coherent cross-AP combining (same fix as Stage 6b).
        SINR_u = |Σ_a ĥ^H W_a[:,u]|² / (Σ_k≠u |Σ_a ĥ^H W_a[:,k]|² + CSI-err + σ²)
    """
    n_ue = topo.n_ue
    D    = n_ue + topo.n_targets
    tx_idx = [ap.idx for ap in topo.tx_aps]

    sinr = np.zeros(n_ue, dtype=np.float64)
    for u in range(n_ue):
        sig = 0.0 + 0j
        for a in tx_idx:
            sig += est.h_hat[(a, u)].conj() @ W_dict[a][:, u]
        P_sig = float(abs(sig) ** 2)

        P_int = 0.0
        for k in range(D):
            if k == u:
                continue
            mui_k = 0.0 + 0j
            for a in tx_idx:
                mui_k += est.h_hat[(a, u)].conj() @ W_dict[a][:, k]
            P_int += float(abs(mui_k) ** 2)

        for a in tx_idx:
            R = est.R_tilde.get((a, u))
            if R is not None:
                W_a = W_dict[a]
                P_int += float(np.real(np.trace(W_a.conj().T @ R @ W_a)))

        sinr[u] = P_sig / max(P_int + sigma_n_sq, 1e-30)
    return sinr


# =============================================================================
# Main ADMM driver
# =============================================================================

def solve_cordis_admm(
    topo: NetworkTopology,
    cfg:  CORDISConfig,
    est:  EstimationResult,
    sensing_stats: SensingChannelStatistics,
    association:   SensingAssociation,
    sigma_n_sq: float,
    Pmax:       float,
    *,
    gamma_u_db: Optional[NDArray[np.float64]] = None,
    omega:      Optional[Dict[int, float]] = None,
    n_admm_max: int   = 50,
    eps_pri:    float = 1.0,
    eps_dual:   float = 1.0,
    rho_admm:   float = 1.0,
    kappa:      float = 0.0,
    xi_slack:   float = 1e4,
    slack_tol:  float = 1e-3,
    n_snapshots: int  = 1,
    verbose:    bool  = False,
) -> ADMMResult:
    """
    Run CORDIS-ADMM (journal paper Section 5).

    Parameters
    ----------
    eps_pri, eps_dual : convergence tolerances on primal / dual residuals.
        With the auto-balanced ρ (rho_admm=1), residuals naturally settle
        near 1 in SNR-amplitude units, so tolerances around 0.3-1.0 match
        the algorithm's equilibrium.  Going below ~0.1 typically requires
        rho_admm > 1 (which can destabilise SCA — see Test 6).
    rho_admm : ADMM penalty parameter ρ, unit-less.  Internally rescaled by
        2/(N_ue · ⟨‖ĥ̃‖²⟩) so rho_admm=1 means "balanced at residual≈1".
    kappa    : clutter penalty weight, unit-less.  Internally rescaled by
        sensing_ref so κ=1 means "strong clutter avoidance".  Useful
        range [0, 2].  NOTE: in scenarios where the clutter direction
        overlaps the target direction (e.g., ground clutter around a
        rooftop target), penalising clutter inherently penalises sensing
        gain, so SCNR is only weakly affected by κ — this is geometric,
        not an implementation issue.
    xi_slack : *upper cap* on the SOC slack weight ξ in P-Central.  Inside
        the loop ξ is adapted via the exterior penalty method
        (Bertsekas, "Nonlinear Programming", §4.2.1): start at 10 % of
        the cap and grow multiplicatively (factor 1.5) whenever the SOC
        slack ``max_u ε_u`` exceeds ``slack_tol`` and is not shrinking
        fast (``> 0.95 × prev``).  ξ relaxes by the same factor only
        when slack is comfortably below tolerance.  This drives ε → 0
        rather than letting the algorithm equilibrate at a sensing-
        favoured operating point with SINR < γ.
    slack_tol : SOC-slack tolerance for ξ adaptation, default 1e-3.
        Slack is measured in normalised SNR-amplitude units (same as the
        SOC RHS), so 1e-3 means "constraint satisfied within 0.001 SNR-
        amplitude units".  Increase to relax SINR enforcement; decrease
        to tighten it (at the cost of more iterations and lower SCNR).
    """
    if not _HAS_CVXPY:
        raise RuntimeError("CORDIS-ADMM requires CVXPY.")

    n_ue       = topo.n_ue
    n_t        = topo.n_targets
    D          = n_ue + n_t
    tx_idx     = [ap.idx for ap in topo.tx_aps]
    n_tx       = len(tx_idx)
    solver_nm  = "CVXPY-" + cfg.algorithm.admm.solver

    # ── SINR targets ─────────────────────────────────────────────────────
    if gamma_u_db is None:
        gamma_lin = db2lin(cfg.algorithm.split.gamma_db) * np.ones(n_ue)
    else:
        gamma_lin = db2lin(np.asarray(gamma_u_db, dtype=np.float64))

    if omega is None:
        omega = {t: 1.0 for t in range(n_t)}

    sigma_n   = float(np.sqrt(sigma_n_sq))
    sqrt_pmax = float(np.sqrt(Pmax))
    snr_scale = Pmax / sigma_n_sq
    sqrt_snr  = float(np.sqrt(snr_scale))

    # ── Warm-start W^(0) from Phase I of CORDIS-Split ───────────────────
    p1    = design_phase_i(topo, cfg, est, sensing_stats, association)
    W_cur = p1.build_W_tx_equal_psr(0.5, Pmax)

    # ── Initial l^(0), z^(0), ν^(0) ─────────────────────────────────────
    def _compute_all_l(W_dict):
        l_dict: Dict[Tuple[int, int], LocalContribution] = {}
        e_dict: Dict[Tuple[int, int], float] = {}
        for a in tx_idx:
            W_tilde = W_dict[a] / sqrt_pmax
            for u in range(n_ue):
                h_tilde = est.h_hat[(a, u)] * sqrt_snr
                R = est.R_tilde.get((a, u))
                R_norm = R * snr_scale if R is not None else None
                l_dict[(a, u)] = _compute_local_contribution(
                    a, u, W_tilde, h_tilde, R_norm, n_ue, n_t,
                )
                e_dict[(a, u)] = l_dict[(a, u)].err
        return l_dict, e_dict

    l_cur, e_cur = _compute_all_l(W_cur)

    # ── Auto-scaling of ADMM penalty weight ─────────────────────────────
    # The raw sensing utility 2√Pmax|Re tr(G^H W̃)| ~ 10⁻¹⁴ (small physical
    # gain), while the consensus penalty in SNR-amplitude units involves
    # ‖ĥ̃‖² ~ β M·Pmax/σ² ~ 10⁶.  In the normalised objective
    #     obj = sensing/sensing_ref  −  (ρ_eff/2) · total_penalty
    # balance at residual ≈ r₀ (SNR amplitude units) requires
    #     ρ_eff ≈ 2 / (r₀² · N_ue · ⟨‖ĥ̃‖²⟩).
    # Setting r₀ = 1, we expose a unit-less knob rho_admm so that
    # rho_admm = 1 corresponds to "balance at residual ≈ 1".
    sqrt_snr_init = float(np.sqrt(Pmax / sigma_n_sq))
    hh_sq_total = 0.0
    hh_count = 0
    for a in tx_idx:
        for u in range(n_ue):
            h_hat = est.h_hat.get((a, u))
            if h_hat is not None:
                hh_sq_total += float(np.sum(np.abs(h_hat * sqrt_snr_init) ** 2))
                hh_count += 1
    hh_sq_avg = hh_sq_total / max(hh_count, 1)
    auto_rho_factor = 2.0 / (max(n_ue, 1) * max(hh_sq_avg, 1e-30))
    rho_effective = rho_admm * auto_rho_factor
    logger.debug(
        "ADMM auto-rho: ⟨‖ĥ̃‖²⟩=%.3e, auto_factor=%.3e, rho_eff=%.3e",
        hh_sq_avg, auto_rho_factor, rho_effective,
    )

    n_mui = n_ue - 1
    z_cur: Dict[int, ConsensusVector] = {}
    nu_cur: Dict[int, ConsensusVector] = {}
    for u in range(n_ue):
        z_cur[u]  = _sum_local_contributions(l_cur, tx_idx, u, n_mui, n_t)
        nu_cur[u] = _zeros_consensus(n_mui, n_t)                  # ν^(0) = 0

    # Initial broadcast Σ̃_u^(0) = Σ_a l_au^(0) − z_u^(0) + ν_u^(0) = 0
    Sigma_tilde_cur: Dict[int, ConsensusVector] = {
        u: _zeros_consensus(n_mui, n_t) for u in range(n_ue)
    }

    # Histories
    prim_hist:  List[float] = []
    dual_hist:  List[float] = []
    sens_hist:  List[float] = []
    sinr_hist:  List[NDArray] = []
    slack_hist: List[NDArray] = []
    znorm_hist: List[float] = []

    inner_failures = 0
    any_inner_ok   = False
    consec_fail    = 0
    MAX_CONSEC_FAIL = 3

    # ── Best-iterate fallback ────────────────────────────────────────────
    # At infeasible γ the algorithm doesn't converge: r_pri rises after
    # some early iters as the CPU pushes z toward an unreachable SOC and
    # ν compounds.  Without tracking, we'd return the *last* (drifted)
    # iterate.  We save the iterate with the HIGHEST min-SINR seen so far
    # (the closest to feasibility for the SINR cone), breaking ties by
    # lower r_pri.  This proxies feasibility by the metric the user
    # actually cares about (per-user SINR), avoiding the failure mode
    # where a tight-consensus iterate (low r_pri) has drifted into
    # low-SINR territory.  At feasible γ the best iterate IS the
    # converged one, so this is a no-regret change.
    best_min_sinr   = -float("inf")
    best_r_pri      = float("inf")
    best_slack_norm = float("inf")
    W_best     : Dict[int, NDArray[np.complex128]] = {
        a: W_cur[a].copy() for a in tx_idx
    }
    best_iter  = 0
    best_sens  = 0.0
    best_sinr  : Optional[NDArray[np.float64]] = None
    best_slack : Optional[NDArray[np.float64]] = None

    # ── Adaptive SOC-slack weight ξ — exterior penalty method ────────────
    # The user-supplied xi_slack is treated as the *cap*; we start at 10 %
    # of the cap and grow ξ multiplicatively whenever the SOC slack
    # (max_u ε_u) persists above ``slack_tol``.  This is the standard
    # exterior-penalty / monotone-penalty heuristic
    # (Bertsekas, "Nonlinear Programming", §4.2.1): drive the penalty
    # parameter up until the constraint is tight, then hold or relax.
    #
    # We deliberately track slack rather than r_pri here.  At high κ the
    # APs reach internal consensus on a sensing-favoured local optimum
    # (r_pri → 0) but with ε_u > 0 baked in — i.e. consensus achieved by
    # mutual agreement to violate the SINR target.  Adapting ξ on r_pri
    # would *relax* the penalty in exactly this regime; tracking slack
    # makes the adaptation reflect the violation it's supposed to
    # penalise.
    xi_cap   = max(xi_slack, 1.0)
    xi_floor = max(xi_cap / 100.0, 1.0)
    xi_cur   = max(xi_cap / 10.0, xi_floor)
    xi_grow  = 1.5
    prev_slack_norm = float("inf")

    # ── SOC-slack Augmented Lagrangian multipliers ν^sinr ────────────────
    # Per-user multipliers for the SINR cone constraint, updated outside
    # the central SOCP via  ν^sinr_u ← max(0, ν^sinr_u + ξ_cur · ε_u*).
    # This is the Method of Multipliers (Hestenes-Powell; Bertsekas
    # §4.2.4): the multiplier accumulates the running SOC slack so that
    # ε_u → 0 holds at FIXED ξ — unlike the pure exterior-penalty method
    # which would need ξ → ∞.  ν^sinr is capped at ``nu_sinr_max`` to
    # prevent runaway when γ is truly infeasible (in which case Stage 6c's
    # best-iterate fallback handles the exit).
    nu_sinr      = np.zeros(n_ue, dtype=np.float64)
    nu_sinr_max  = 1e8                                  # safety cap

    if verbose:
        print(f"  [ADMM] rho={rho_admm}, kappa={kappa}, xi∈[{xi_floor:.1e}, "
              f"{xi_cap:.1e}], n_max={n_admm_max}, gamma_dB={lin2db(gamma_lin)}")

    # =====================================================================
    # ADMM main loop
    # =====================================================================
    for n_iter in range(n_admm_max):
        # ── Build SCA sensing gradient at current W (per AP) ─────────────
        G_sca = _compute_sca_gradient(
            W_cur, sensing_stats, association, topo, omega, n_snapshots,
        )

        # Sensing references (per AP): use |2 Re tr(G^H W)| as natural scale
        sensing_ref: Dict[int, float] = {}
        for a in tx_idx:
            G_a = G_sca.get(a)
            if G_a is None:
                sensing_ref[a] = 1.0
            else:
                val = 2.0 * abs(float(np.real(np.trace(G_a.conj().T @ W_cur[a]))))
                sensing_ref[a] = max(val, 1e-30)

        # Auto-scaled κ per AP: with κ_eff = κ × sensing_ref and the
        # clutter term ``-κ_eff · ‖C^{1/2} W̃‖²``, the clutter contribution
        # in the normalised objective becomes  κ · ‖C^{1/2} W̃‖²  -- so
        # user-facing κ=1 is already a strong clutter penalty (typical
        # ‖C^{1/2} W̃‖² ~ M_t · ‖W̃[:,target]‖² ~ a few units for sensing-
        # aligned beams).  This gives a meaningful SCNR trade-off across
        # κ ∈ [0, 2].  Without this auto-scaling, raw κ·Pmax·M_t ~ 10⁻³
        # would dwarf raw sensing ~ 10⁻¹⁴ for any sane κ.
        kappa_effective: Dict[int, float] = {}
        for a in tx_idx:
            kappa_effective[a] = kappa * sensing_ref[a]

        # ── Phase I: Parallel per-AP QCQP ────────────────────────────────
        W_new: Dict[int, NDArray] = {}
        all_ok = True
        for a in tx_idx:
            # Reconstruct AP-specific residual: Σ_au^(n) = Σ̃_u^(n) − l_au^(n)
            Sigma_au: Dict[int, ConsensusVector] = {}
            e_prev_a: Dict[int, float] = {}
            for u in range(n_ue):
                s_t = Sigma_tilde_cur[u]
                l_au = l_cur[(a, u)]
                Sigma_au[u] = ConsensusVector(
                    cds  = s_t.cds  - l_au.cds,
                    mui  = s_t.mui  - l_au.mui,
                    s2ci = s_t.s2ci - l_au.s2ci,
                    err  = s_t.err  - l_au.err,
                )
                e_prev_a[u] = e_cur[(a, u)]

            W_a_new, ok = _solve_local_qcqp(
                a_idx=a,
                W_a_prev=W_cur[a],
                G_sca_a=G_sca.get(a),
                sensing_stats=sensing_stats,
                est=est,
                topo=topo,
                Sigma_au=Sigma_au,
                e_au_prev=e_prev_a,
                sigma_n_sq=sigma_n_sq,
                Pmax=Pmax,
                rho_effective=rho_effective,
                kappa_effective=kappa_effective[a],
                sensing_ref=sensing_ref[a],
                solver=cfg.algorithm.admm.solver,
            )
            W_new[a] = W_a_new
            if not ok:
                inner_failures += 1
                all_ok = False
            else:
                any_inner_ok = True

        # Compute new exact l^(n+1) and e^(n+1) from W^(n+1)
        l_new, e_new = _compute_all_l(W_new)

        # ── Phase II: Per-user CPU SOCP ──────────────────────────────────
        z_new:    Dict[int, ConsensusVector] = {}
        eps_new   = np.zeros(n_ue, dtype=np.float64)
        cpu_ok    = True
        for u in range(n_ue):
            sum_l_u = _sum_local_contributions(l_new, tx_idx, u, n_mui, n_t)
            z_u, eps_u, ok = _solve_central_socp(
                u=u,
                sum_l_u=sum_l_u,
                nu_u=nu_cur[u],
                gamma_u=float(gamma_lin[u]),
                n_ue_minus_1=n_mui,
                n_t=n_t,
                xi_slack=xi_cur,
                solver=cfg.algorithm.admm.solver,
                nu_sinr_u=float(nu_sinr[u]),
            )
            z_new[u]   = z_u
            eps_new[u] = eps_u
            if not ok:
                cpu_ok = False
                inner_failures += 1
            else:
                any_inner_ok = True

        all_ok = all_ok and cpu_ok
        consec_fail = consec_fail + 1 if not all_ok else 0

        # ── Dual update:  ν_u^(n+1) = ν_u^(n) + Σ_a l_au^(n+1) − z_u^(n+1) ──
        nu_new: Dict[int, ConsensusVector] = {}
        for u in range(n_ue):
            sum_l_u = _sum_local_contributions(l_new, tx_idx, u, n_mui, n_t)
            z_u     = z_new[u]
            nu_new[u] = ConsensusVector(
                cds  = nu_cur[u].cds  + (sum_l_u.cds  - z_u.cds),
                mui  = nu_cur[u].mui  + (sum_l_u.mui  - z_u.mui),
                s2ci = nu_cur[u].s2ci + (sum_l_u.s2ci - z_u.s2ci),
                err  = nu_cur[u].err  + (sum_l_u.err  - z_u.err),
            )

        # New broadcast Σ̃_u^(n+1) = Σ_a l_au^(n+1) − z_u^(n+1) + ν_u^(n+1)
        Sigma_tilde_new: Dict[int, ConsensusVector] = {}
        for u in range(n_ue):
            sum_l_u = _sum_local_contributions(l_new, tx_idx, u, n_mui, n_t)
            Sigma_tilde_new[u] = ConsensusVector(
                cds  = sum_l_u.cds  - z_new[u].cds  + nu_new[u].cds,
                mui  = sum_l_u.mui  - z_new[u].mui  + nu_new[u].mui,
                s2ci = sum_l_u.s2ci - z_new[u].s2ci + nu_new[u].s2ci,
                err  = sum_l_u.err  - z_new[u].err  + nu_new[u].err,
            )

        # ── Residuals ────────────────────────────────────────────────────
        r_pri = 0.0
        r_dual_inner = 0.0
        z_norm_sum = 0.0
        for u in range(n_ue):
            sum_l_u = _sum_local_contributions(l_new, tx_idx, u, n_mui, n_t)
            r_pri        += _consensus_residual(sum_l_u, z_new[u])
            r_dual_inner += _z_diff_norm(z_new[u], z_cur[u])
            z_norm_sum   += _consensus_residual(z_new[u],
                                                 _zeros_consensus(n_mui, n_t))
        r_dual = rho_admm * r_dual_inner

        # Sensing utility (matching the QCQP objective, summed across APs).
        # The optimizer maximises  2·Re{tr(Gᴴ W)} − κ_eff · ‖C^½ W̃‖²_F
        # where W̃ = W/√Pmax; equivalently  − (κ_eff/Pmax)·tr(WᴴCW)  in
        # physical W units.  Reporting this same quantity (rather than the
        # user-facing raw κ·tr(WᴴCW)) keeps the history consistent with
        # what the algorithm actually optimises.
        sens_val = 0.0
        for a in tx_idx:
            G_a = G_sca.get(a)
            if G_a is not None:
                sens_val += 2.0 * float(np.real(np.trace(G_a.conj().T @ W_new[a])))
                if kappa > 0:
                    C = sensing_stats.C_tx.get(a)
                    if C is not None:
                        clutter_pwr = float(
                            np.real(np.trace(W_new[a].conj().T @ C @ W_new[a]))
                        )
                        sens_val -= (kappa_effective[a] / max(Pmax, 1e-30)) * clutter_pwr

        sinr_vec = _compute_sinr_coherent(W_new, est, topo, sigma_n_sq)

        prim_hist.append(r_pri)
        dual_hist.append(r_dual)
        sens_hist.append(sens_val)
        sinr_hist.append(sinr_vec)
        slack_hist.append(eps_new.copy())
        znorm_hist.append(z_norm_sum)

        # ── Best-iterate snapshot (highest min-SINR seen) ────────────────
        # Track the iterate that achieves the highest min-SINR — the
        # metric the user actually cares about.  Ties broken by lower
        # r_pri.  At feasible γ the converged iterate has min-SINR ≈ γ
        # and is also the best by this rule, so this never regresses
        # against the previous behaviour.  At infeasible γ, or when the
        # algorithm hits ``n_admm_max`` before convergence, this returns
        # the iterate that came closest to the SINR target rather than
        # an iterate that drifted into deeper violation.
        cand_min_sinr = float(sinr_vec.min()) if sinr_vec.size else 0.0
        cand_slack    = float(np.max(eps_new)) if eps_new.size else 0.0
        take = False
        if all_ok:
            if cand_min_sinr > best_min_sinr + 1e-12:
                take = True                              # strictly higher min-SINR
            elif (abs(cand_min_sinr - best_min_sinr) < 1e-12
                  and r_pri < best_r_pri):
                take = True                              # same min-SINR, tighter consensus
        if take:
            best_min_sinr   = cand_min_sinr
            best_r_pri      = r_pri
            best_slack_norm = cand_slack
            W_best     = {a: W_new[a].copy() for a in tx_idx}
            best_iter  = n_iter + 1
            best_sens  = sens_val
            best_sinr  = sinr_vec.copy()
            best_slack = eps_new.copy()

        # ── Adaptive ξ — exterior penalty method, slack-driven ───────────
        # Grow ξ when the max SOC slack max_u ε_u persists above
        # ``slack_tol``; relax it only when the slack is comfortably
        # below tolerance.  This drives ε → 0 rather than letting it
        # equilibrate at a sensing-favoured operating point with
        # SINR < γ.  Standard exterior penalty heuristic — see Bertsekas
        # "Nonlinear Programming", §4.2.1.
        slack_norm = float(np.max(eps_new)) if eps_new.size else 0.0
        if n_iter >= 1:
            if slack_norm > slack_tol:
                # Constraint still violated — penalise harder.
                # Gate on "not shrinking fast" so we don't keep ramping
                # if the algorithm is already making good progress.
                if slack_norm > 0.95 * prev_slack_norm:
                    xi_cur = min(xi_cur * xi_grow, xi_cap)
                # else: violation is shrinking fast, hold ξ steady
            elif slack_norm < 0.1 * slack_tol:
                # Comfortably feasible — can relax ξ slightly
                xi_cur = max(xi_cur / xi_grow, xi_floor)
        prev_slack_norm = slack_norm

        # ── ALM multiplier update for SOC slack (Method of Multipliers) ─
        # Bertsekas §4.2.4:  ν^sinr_u ← max(0, ν^sinr_u + ξ · ε_u*).
        # When ε_u > 0 (SINR cone violated), the multiplier accumulates,
        # adding a stronger linear pull toward feasibility next iter.
        # When ε_u = 0 (constraint tight), it holds steady — i.e.
        # remembers the optimal dual variable for the constraint.
        # Capped to prevent runaway when γ is truly infeasible.
        for u in range(n_ue):
            nu_sinr[u] = min(
                max(0.0, nu_sinr[u] + xi_cur * float(eps_new[u])),
                nu_sinr_max,
            )

        if verbose:
            print(f"  [ADMM {n_iter+1:2d}]  r_pri={r_pri:.3e}  r_dual={r_dual:.3e}  "
                  f"minSINR={lin2db(sinr_vec.min()):+.2f}dB  "
                  f"maxSlack={eps_new.max():.2e}  ξ={xi_cur:.2e}  "
                  f"maxν^sinr={nu_sinr.max():.2e}")

        # ── Advance state ────────────────────────────────────────────────
        W_cur            = W_new
        l_cur            = l_new
        e_cur            = e_new
        z_cur            = z_new
        nu_cur           = nu_new
        Sigma_tilde_cur  = Sigma_tilde_new

        # ── Convergence ──────────────────────────────────────────────────
        # Original criterion: consensus tight (r_pri, r_dual) and all
        # inner solves succeeded.  The ALM multipliers + adaptive ξ keep
        # slack ε close to zero at the converged iterate; we don't add
        # an explicit slack check here because that previously caused
        # the algorithm to refuse to declare convergence on perfectly
        # good iterates with tiny but non-zero slack, which then forced
        # the fallback to drift through the full ``n_admm_max`` window.
        if r_pri < eps_pri and r_dual < eps_dual and all_ok:
            if verbose:
                print(f"  [ADMM] converged at iter {n_iter+1} "
                      f"(slack={slack_norm:.2e})")
            return ADMMResult(
                W_tx=W_cur,
                primal_res_history=prim_hist,
                dual_res_history=dual_hist,
                sensing_obj_history=sens_hist,
                sinr_history=sinr_hist,
                slack_history=slack_hist,
                z_norm_history=znorm_hist,
                converged=True,
                n_admm_iters=n_iter + 1,
                solver=solver_nm,
                feasible=any_inner_ok,
                inner_failures=inner_failures,
            )

        if consec_fail >= MAX_CONSEC_FAIL:
            if verbose:
                print(f"  [ADMM] aborting after {consec_fail} consecutive "
                      f"iterations with inner failures")
            break

    # ── Non-converged exit: return BEST iterate (lowest r_pri seen) ──────
    # At infeasible γ, the last iterate is typically the most drifted; the
    # smallest-r_pri iterate is the closest to a feasible consensus.  At
    # feasible γ, both are identical (final iter = best).
    if best_r_pri < float("inf"):
        if verbose:
            print(f"  [ADMM] no convergence — returning best iterate "
                  f"({best_iter}/{len(prim_hist)}, r_pri={best_r_pri:.3e})")
        return ADMMResult(
            W_tx=W_best,
            primal_res_history=prim_hist,
            dual_res_history=dual_hist,
            sensing_obj_history=sens_hist,
            sinr_history=sinr_hist,
            slack_history=slack_hist,
            z_norm_history=znorm_hist,
            converged=False,
            n_admm_iters=len(prim_hist),
            solver=solver_nm,
            feasible=any_inner_ok,
            inner_failures=inner_failures,
        )

    return ADMMResult(
        W_tx=W_cur,
        primal_res_history=prim_hist,
        dual_res_history=dual_hist,
        sensing_obj_history=sens_hist,
        sinr_history=sinr_hist,
        slack_history=slack_hist,
        z_norm_history=znorm_hist,
        converged=False,
        n_admm_iters=len(prim_hist),
        solver=solver_nm,
        feasible=any_inner_ok,
        inner_failures=inner_failures,
    )


# =============================================================================
# Public API
# =============================================================================

def run_cordis_admm(
    topo: NetworkTopology,
    cfg:  CORDISConfig,
    est:  EstimationResult,
    sensing_stats: SensingChannelStatistics,
    association:   SensingAssociation,
    sigma_n_sq:    float,
    Pmax:          float,
    **kwargs,
) -> Tuple[Dict[int, NDArray[np.complex128]], ADMMResult]:
    """
    Full CORDIS-ADMM pipeline (journal paper Section 5).

    Returns
    -------
    W_tx   : dict[ap_idx → (M_t, D)]  decentralised beamformers
    result : ADMMResult with full primal/dual residual histories
    """
    result = solve_cordis_admm(
        topo, cfg, est, sensing_stats, association,
        sigma_n_sq, Pmax, **kwargs,
    )
    return result.W_tx, result

