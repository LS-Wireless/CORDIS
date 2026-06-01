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

CSI-error aggregation (Stage 23b — exact per-AP block)
-----------------------------------------------------
The per-(AP,user) contribution carries the AMPLITUDE e_{a_t,u} = ‖R̃^{1/2} W_a‖_F
(not the squared form), which keeps the local QCQP a true QCQP and the CPU
SOC DCP-clean.  The CSI errors at different transmit APs are *statistically
independent*, so their powers add:

    P_CSI-Error(u) = Σ_{a_t} ‖R̃_{a_t u}^{1/2} W_{a_t}‖_F² = Σ_{a_t} e_{a_t u}²   .

We therefore keep the CSI-error contributions PER-AP in the consensus vector
(a length-N_tx block, gathered rather than coherently summed) and stack them
individually in the CPU SOC.  The L2-norm squaring then yields exactly
Σ_{a_t} e_{a_t u}², i.e. the SOC is the EXACT per-user SINR constraint.

    (Previously the block was a single scalar z^err = Σ_a e_{a u}; squaring it
     in the SOC gave (Σ_a e_{a u})² ≥ Σ_a e_{a u}², an over-count of the
     CSI-error power by all the cross terms — up to a factor N_tx.  That made
     the SINR constraint conservatively tight and manufactured infeasibility.
     The per-AP block removes that conservatism.)

The CDS / MUI / S2CI components are still coherently summed across APs
(z^CDS = Σ_a x_{a u}, etc.) because those signals combine coherently at the
user; only the CSI-error block is gathered.

Consensus:  z_u^{CDS,MUI,S2CI} = Σ_{a_t} l_{a_t,u}^{·} ;  z_u^err = (e_{a_t u})_{a_t} .

Augmented Lagrangian:
    L_ρ = Σ_a Ũ_a^sens(W_a) − (ρ/2) Σ_u ‖Σ_a l_{a,u}(W_a) − z_u + ν_u‖²
(with the err block contributing the per-AP terms Σ_a (e_{a,u} − z_u^err[a] + ν_u^err[a])²).

Per ADMM iteration n:
    Phase I  — Parallel per-AP QCQP:  P-Local (eq. admm-p-local)
                  max Ũ_a^sens(W_a) − (ρ/2) Σ_u ‖l_{a,u}(W_a) + Σ_{a,u}^(n)‖²
                  s.t.  ‖W_a‖²_F ≤ Pmax
                with Σ_{a,u}^(n) = Σ̃_u^(n) − l_{a,u}^(n)  (err block uses only AP a's slot).

    Phase II — Per-user CPU SOCP projection:  P-Central (eq. admm-p-central)
                  min ‖Σ_a l_{a,u}^(n+1) + ν_u^(n) − z_u‖² + ξ ε_u
                  s.t.  ‖[z^MUI; z^S2CI; z^err_1..z^err_Ntx; 1]‖_2  ≤  Re{z^CDS}/√γ + ε_u,
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

3. The SOC at the CPU is a clean L2-norm inequality with the per-AP z^err
   amplitudes in the cone stack directly — no √, no rotated cone, no
   auxiliary variable.  Stacking the per-AP entries makes ‖·‖² reproduce
   Σ_a e_au² = the exact CSI-error power.
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

    NOTE on the err component (Stage 23b): we use the AMPLITUDE form
    err = ‖R̃^{1/2} W_a‖_F (not the squared form) so the local QCQP stays a
    true QCQP and the CPU SOC is DCP-clean.  Crucially, the per-AP errors are
    NOT coherently summed into a single consensus scalar — they are gathered
    into a per-AP block in :class:`ConsensusVector` and stacked individually
    in the CPU SOC.  Because the CSI errors are independent across APs, their
    powers add, so the SOC's squared norm reproduces Σ_a ‖R̃^{1/2} W_a‖_F²
    EXACTLY (the true per-user CSI-error power), rather than the conservative
    (Σ_a ‖R̃^{1/2} W_a‖_F)² over-count produced by summing the amplitudes.

    At the AP, the err component is convex (a norm), so the consensus penalty
    (err − z^err[a] + ν^err[a])² is linearised via an SCA tangent around the
    previous iterate, matching the SCA scheme used for the sensing utility.
    """
    cds:  complex
    mui:  NDArray[np.complex128]
    s2ci: NDArray[np.complex128]
    err:  float


@dataclass
class ConsensusVector:
    """Global per-user consensus vector z_u.

    cds  : complex scalar         — coherent sum Σ_a x_{a u}
    mui  : complex (N_ue-1,)      — coherent sum Σ_a i^(c)_{a u}
    s2ci : complex (N_t,)         — coherent sum Σ_a i^(s)_{a u}
    err  : real (N_tx,)           — PER-AP CSI-error amplitudes (gathered, NOT
                                    summed).  Stage 23b: each transmit AP owns
                                    one slot; the CPU SOC stacks all of them so
                                    ‖z^err‖² = Σ_a e_{a u}² = exact CSI-error
                                    power.

    All components live in SNR-normalised units; err entries are AMPLITUDES (≥ 0).

    NOTE: the per-AP residual objects fed to the local QCQP (``Sigma_au``) reuse
    this dataclass for cds/mui/s2ci, but their err slot is unused there — the
    local QCQP receives its scalar per-AP err residual via a dedicated
    ``sigma_err_au`` argument.  Only the GLOBAL consensus objects (z_u, ν_u,
    Σ̃_u, and the output of :func:`_sum_local_contributions`) carry a true
    length-N_tx err vector.
    """
    cds:  complex
    mui:  NDArray[np.complex128]
    s2ci: NDArray[np.complex128]
    err:  NDArray[np.float64]


@dataclass
class ADMMResult:
    """
    Output of CORDIS-ADMM (Stage 6c).

    Histories are appended once per outer iteration.

    ``best_iter`` (1-indexed) is the outer-iteration index whose
    consensus residual was lowest — useful for diagnostics when the
    solver doesn't fully converge and returns the best iterate seen.
    Set to 0 if no iteration has been recorded.
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
    best_iter:            int         = 0
    rho_history:          List[float] = field(default_factory=list)
    early_stopped:        bool        = False


# =============================================================================
# Best-iterate selection (Stage 23a) — pure, unit-testable decision rule
# =============================================================================

def _should_take_iterate(
    criterion: str,
    *,
    cand_feasible:   bool,
    cand_min_sinr:   float,
    cand_resid_norm: float,
    cand_r_pri:      float,
    best_feasible:   bool,
    best_min_sinr:   float,
    best_resid_norm: float,
    best_r_pri:      float,
) -> bool:
    """Return True if the candidate iterate should replace the incumbent.

    Three criteria:

      * ``"residual_norm"`` (default): pick the iterate with the smallest
        combined primal+dual residual (the ADMM-natural optimality measure),
        ties broken by higher worst-user SINR.

      * ``"min_sinr"`` (legacy): pick the iterate with the highest worst-user
        SINR, ties broken by lower primal residual.

      * ``"feasible_then_residual"`` (Stage 23a): prefer iterates that are
        *feasible* (min-SINR ≥ γ for every user), ranked by worst-user SINR;
        among iterates that are all infeasible, fall back to lowest residual
        (closest to a feasible consensus), ties broken by higher min-SINR.  A
        feasible incumbent is NEVER replaced by an infeasible candidate.  This
        returns the best feasible beamformer the algorithm actually produced
        (the "best feasible incumbent" pattern) instead of discarding a
        transiently-feasible iterate because the run ended infeasible.
    """
    if criterion == "feasible_then_residual":
        if cand_feasible and not best_feasible:
            return True
        if cand_feasible and best_feasible:
            if cand_min_sinr > best_min_sinr + 1e-12:
                return True
            return (abs(cand_min_sinr - best_min_sinr) < 1e-12
                    and cand_resid_norm < best_resid_norm - 1e-12)
        if (not cand_feasible) and (not best_feasible):
            if cand_resid_norm < best_resid_norm - 1e-12:
                return True
            return (abs(cand_resid_norm - best_resid_norm) < 1e-12
                    and cand_min_sinr > best_min_sinr + 1e-12)
        # candidate infeasible, incumbent feasible → never downgrade
        return False

    if criterion == "residual_norm":
        if cand_resid_norm < best_resid_norm - 1e-12:
            return True
        return (abs(cand_resid_norm - best_resid_norm) < 1e-12
                and cand_min_sinr > best_min_sinr)

    # "min_sinr"
    if cand_min_sinr > best_min_sinr + 1e-12:
        return True
    return (abs(cand_min_sinr - best_min_sinr) < 1e-12
            and cand_r_pri < best_r_pri)


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


def _zeros_consensus(n_ue_minus_1: int, n_t: int, n_tx: int) -> ConsensusVector:
    """Zero consensus vector with a length-``n_tx`` per-AP err block (Stage 23b)."""
    return ConsensusVector(
        cds=0.0 + 0.0j,
        mui=np.zeros(n_ue_minus_1, dtype=np.complex128),
        s2ci=np.zeros(n_t, dtype=np.complex128),
        err=np.zeros(n_tx, dtype=np.float64),
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
    """Aggregate the per-AP contributions for user ``u`` into z_u's target.

    CDS / MUI / S2CI are coherently SUMMED across APs.  The CSI-error block
    is GATHERED per-AP (Stage 23b): ``err[slot]`` holds AP ``tx_ap_indices[slot]``'s
    own amplitude e_{a u}, so the CPU SOC reproduces Σ_a e_{a u}² exactly.
    """
    n_tx = len(tx_ap_indices)
    out = _zeros_consensus(n_ue_minus_1, n_t, n_tx)
    for slot, a in enumerate(tx_ap_indices):
        l = l_dict[(a, u)]
        out.cds  += l.cds
        out.mui  += l.mui
        out.s2ci += l.s2ci
        out.err[slot] = l.err          # GATHER (per-AP), not coherent sum
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
    Sigma_au:        Dict[int, ConsensusVector],   # Σ_{a,u}^(n)  per user (cds/mui/s2ci)
    sigma_err_au:    Dict[int, float],             # Σ_{a,u}^err[a] per user (scalar, Stage 23b)
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

    Stage 23b: the err consensus is per-AP, so the err residual seen by this AP
    is the scalar ``sigma_err_au[u] = ν_u^err[a] − z_u^err[a]`` (AP ``a``'s own
    slot only); the penalty term is (e_au(W̃) + sigma_err_au[u])².

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
        sigma_au_u = Sigma_au[u]            # constant ConsensusVector (cds/mui/s2ci)

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

        # ---- err:  (ẽ_au + Σ_err[a])²  with SCA tangent on AMPLITUDE ────
        # Stage 23b: per-AP err residual is the scalar sigma_err_au[u]
        # ( = ν_u^err[a] − z_u^err[a] ), AP a's own slot only.  The err
        # component f(W̃) = ‖R̃̃^{1/2} W̃‖_F is convex but not affine, so we
        # SCA-linearise at W̃_prev:
        #     T(W̃) = Re{tr((W̃_prev)^H R̃̃ W̃)} / f_n,  f_n = ‖R̃̃^{1/2} W̃_prev‖_F.
        R_tilde = est.R_tilde.get((a_idx, u))
        if R_tilde is not None:
            R_norm  = R_tilde * snr_scale                              # R̃̃
            f_n     = float(e_au_prev[u])                              # ‖R̃̃^{1/2} W̃^(n)‖_F
            if f_n > 1e-15:
                err_tangent = (
                    cp.real(cp.trace(W_tilde_prev.conj().T @ R_norm @ W_var))
                    / f_n
                )
                err_resid = err_tangent + float(sigma_err_au[u])
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
    sum_l_u:    ConsensusVector,        # Σ_{a_t} l_{a_t, u}^(n+1)  (constant; err gathered)
    nu_u:       ConsensusVector,        # ν_u^(n)  (constant; err per-AP)
    gamma_u:    float,                  # SINR target (linear)
    n_ue_minus_1: int,
    n_t:        int,
    n_tx:       int,                    # number of TX APs (err block length, Stage 23b)
    xi_slack:   float,
    solver:     str,
    nu_sinr_u:  float = 0.0,            # SOC-slack ALM multiplier
) -> Tuple[ConsensusVector, float, bool]:
    """
    Solve the per-user CPU SOCP P-Central (eq. admm-p-central) in
    SNR-normalised units (noise → 1), using the per-AP CSI-error block
    (Stage 23b; see module docstring):

        min  ‖sum_l_u + nu_u − z_u‖²
             + ν^sinr_u · ε_u + (ξ/2) · ε_u²        ← Augmented Lagrangian
        s.t. ‖[z^MUI; z^S2CI; z^err_1..z^err_Ntx; 1]‖_2  ≤  Re{z^CDS}/√γ_u + ε_u
             Im{z_u^CDS} = 0,   z_u^err ≥ 0,   ε_u ≥ 0

    The per-AP err amplitudes z^err_a are stacked individually in the SOC, so
    the built-in squaring inside ‖·‖² yields Σ_a (z^err_a)² = the EXACT
    per-user CSI-error power Σ_a ‖R̃^{1/2} W_a‖_F² (no cross-term over-count).

    The Augmented Lagrangian on ε_u (Method of Multipliers, Bertsekas §4.2.4)
    supersedes the pure linear penalty ξ·ε of the exterior penalty method.
    The multiplier ν^sinr_u is updated outside the SOCP via
    ν^sinr ← max(0, ν^sinr + ξ · ε*), driving ε → 0 at fixed ξ.

    Returns (z_u, eps_u, success).
    """
    # CVXPY variables — note that putting each z^err_a directly in the SOC
    # stack (no √-aux, no rotated-cone) is the key DCP-safe encoding here.
    z_cds_re = cp.Variable()                                # Im{z_cds}=0 ⇒ real var
    z_mui    = cp.Variable(n_ue_minus_1, complex=True) if n_ue_minus_1 > 0 else None
    z_s2ci   = cp.Variable(n_t, complex=True)                if n_t > 0           else None
    z_err    = cp.Variable(n_tx, nonneg=True)               # per-AP AMPLITUDES, ≥ 0
    eps_u    = cp.Variable(nonneg=True)

    constraints: List = []

    # ── Build SOC stack: [Re/Im z_MUI, Re/Im z_S2CI; z_err_1..z_err_Ntx; 1] ──
    parts = []
    if z_mui is not None:
        parts.append(cp.real(z_mui))
        parts.append(cp.imag(z_mui))
    if z_s2ci is not None:
        parts.append(cp.real(z_s2ci))
        parts.append(cp.imag(z_s2ci))
    parts.append(z_err)                                      # per-AP err vector in stack
    parts.append(np.array([1.0]))                            # σ_n → 1 (normalised)
    soc_vec = cp.hstack(parts)

    # ‖soc_vec‖_2  ≤  Re{z_CDS}/√γ + ε_u    (standard convex SOC, DCP-OK)
    constraints.append(
        cp.norm(soc_vec, 2) <= z_cds_re / float(np.sqrt(gamma_u)) + eps_u
    )

    # ── Objective:  ‖sum_l + nu − z‖²  + slack penalty ──
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

    # Per-AP err block (Stage 23b): err_const is a length-N_tx vector.
    err_const = np.asarray(sum_l_u.err + nu_u.err, dtype=np.float64)
    pen_err   = cp.sum_squares(err_const - z_err)

    # Augmented Lagrangian penalty on the SOC slack ε_u:
    #     ν^sinr_u · ε_u  +  (ξ/2) · ε_u²
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
            return _zeros_consensus(n_ue_minus_1, n_t, n_tx), 0.0, False

    if prob.status not in ("optimal", "optimal_inaccurate") or z_cds_re.value is None:
        logger.debug("CPU SOCP (u=%d) status=%s", u, prob.status)
        return _zeros_consensus(n_ue_minus_1, n_t, n_tx), 0.0, False

    z_out = ConsensusVector(
        cds  = complex(float(z_cds_re.value), 0.0),
        mui  = (np.asarray(z_mui.value,  dtype=np.complex128)
                if z_mui is not None else np.zeros(0, dtype=np.complex128)),
        s2ci = (np.asarray(z_s2ci.value, dtype=np.complex128)
                if z_s2ci is not None else np.zeros(0, dtype=np.complex128)),
        err  = np.asarray(z_err.value, dtype=np.float64).reshape(n_tx),
    )
    eps_out = float(eps_u.value) if eps_u.value is not None else 0.0
    return z_out, eps_out, True


# =============================================================================
# Residuals & SINR (helpers)
# =============================================================================

def _consensus_residual(l_sum: ConsensusVector, z: ConsensusVector) -> float:
    """‖Σ_a l_au − z_u‖₂ in SNR-normalised mixed-type space (err is per-AP)."""
    r2 = abs(l_sum.cds - z.cds) ** 2
    r2 += float(np.sum(np.abs(l_sum.mui - z.mui) ** 2))
    r2 += float(np.sum(np.abs(l_sum.s2ci - z.s2ci) ** 2))
    r2 += float(np.sum((l_sum.err - z.err) ** 2))      # per-AP err vector
    return float(np.sqrt(r2))


def _z_diff_norm(z_new: ConsensusVector, z_old: ConsensusVector) -> float:
    """‖z^(n+1) − z^(n)‖₂."""
    d2 = abs(z_new.cds - z_old.cds) ** 2
    d2 += float(np.sum(np.abs(z_new.mui - z_old.mui) ** 2))
    d2 += float(np.sum(np.abs(z_new.s2ci - z_old.s2ci) ** 2))
    d2 += float(np.sum((z_new.err - z_old.err) ** 2))  # per-AP err vector
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

    Note the CSI-error term here is the EXACT Σ_a tr(W_a^H R̃ W_a) — the same
    quantity the per-AP SOC block now enforces (Stage 23b).
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
    best_iter_criterion: Optional[str] = None,
    verbose:    bool  = False,
    progress:   bool  = False,
) -> ADMMResult:
    """
    Run CORDIS-ADMM (journal paper Section 5).

    Parameters
    ----------
    best_iter_criterion : {"residual_norm", "min_sinr", "feasible_then_residual"} or None
        Criterion for selecting the returned iterate (Stage 23a).  When None
        (default), it is read from ``cfg.algorithm.admm.best_iter_criterion``
        so the JSON config stays authoritative; an explicit value overrides.
        ``"feasible_then_residual"`` returns the highest-min-SINR iterate that
        met γ (the best feasible incumbent), falling back to lowest-residual
        when no iterate was feasible.
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
    ap_slot    = {a: i for i, a in enumerate(tx_idx)}   # AP idx → err-block slot (Stage 23b)
    solver_nm  = "CVXPY-" + cfg.algorithm.admm.solver

    # ── Resolve best-iterate criterion from cfg when not passed (Stage 23a) ─
    # Default None means "let the JSON config decide", keeping the config
    # field authoritative; an explicit kwarg still wins.
    if best_iter_criterion is None:
        best_iter_criterion = getattr(
            cfg.algorithm.admm, "best_iter_criterion", "residual_norm"
        )

    # ── SINR targets ─────────────────────────────────────────────────────
    if gamma_u_db is None:
        gamma_lin = db2lin(cfg.algorithm.gamma_db) * np.ones(n_ue)
    else:
        gamma_lin = db2lin(np.asarray(gamma_u_db, dtype=np.float64))

    if omega is None:
        omega = {t: 1.0 for t in range(n_t)}

    sigma_n   = float(np.sqrt(sigma_n_sq))
    sqrt_pmax = float(np.sqrt(Pmax))
    snr_scale = Pmax / sigma_n_sq
    sqrt_snr  = float(np.sqrt(snr_scale))

    # ── Warm-start W^(0): honor cfg.algorithm.admm.warm_start_from_split ───
    # Stage 20 (was Stage 19b "dead config field"): when True (default),
    # run the full CORDIS-Split pipeline to get the optimal PSR-based W;
    # this typically lands ADMM in a region that already meets γ, dramatically
    # reducing infeasibility and divergence on hard channel realizations.
    # When False, fall back to the original equal-PSR (50/50) warm-start.
    use_split_warmstart = bool(
        getattr(cfg.algorithm.admm, "warm_start_from_split", True)
    )
    if use_split_warmstart:
        # run_cordis_split returns (W_tx, PhaseIResult, SplitOptResult).
        # Local import to avoid module-level circular dependency.
        from cordis.algorithms.split_opt import run_cordis_split
        W_cur, _phase_i, _split_res = run_cordis_split(
            topo, cfg, est, sensing_stats, association,
            sigma_n_sq, Pmax,
            gamma_u_db=gamma_u_db, omega=omega,
        )
    else:
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
        nu_cur[u] = _zeros_consensus(n_mui, n_t, n_tx)            # ν^(0) = 0

    # Initial broadcast Σ̃_u^(0) = Σ_a l_au^(0) − z_u^(0) + ν_u^(0) = 0
    Sigma_tilde_cur: Dict[int, ConsensusVector] = {
        u: _zeros_consensus(n_mui, n_t, n_tx) for u in range(n_ue)
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

    # ── Best-iterate selection (Stage 22a + Stage 23a) ────────────────────
    # Criteria (see _should_take_iterate):
    #   "residual_norm" (default): smallest r_pri+r_dual; tie → higher min-SINR.
    #   "min_sinr" (legacy): highest worst-user SINR; tie → lower r_pri.
    #   "feasible_then_residual" (Stage 23a): best feasible incumbent (highest
    #       min-SINR among iterates meeting γ); fall back to lowest residual
    #       when none feasible.  See the ADMMConfig docstring.
    _ALLOWED_CRITERIA = ("residual_norm", "min_sinr", "feasible_then_residual")
    if best_iter_criterion not in _ALLOWED_CRITERIA:
        raise ValueError(
            f"best_iter_criterion={best_iter_criterion!r} not in "
            f"{_ALLOWED_CRITERIA}.  See ADMMConfig.best_iter_criterion."
        )
    best_min_sinr   = -float("inf")
    best_r_pri      = float("inf")
    best_slack_norm = float("inf")
    best_resid_norm = float("inf")     # r_pri + r_dual tracker
    best_feasible   = False            # Stage 23a: did the incumbent meet γ?
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

    # ── Adaptive ρ state (Stage 22b) ─────────────────────────────────────
    # Boyd-Parikh-Chu (2011) §3.4.1 adaptive penalty parameter scheme.
    # Uses the relative-residual variant because the dual residual in
    # this code is computed with user-facing ``rho_admm`` rather than
    # ``rho_effective``, so r_dual is systematically small vs r_pri —
    # normalising by tolerances cancels that scaling skew.
    #
    # Stability bounds are tight: empirically, scaling rho_admm by ≥5×
    # the input value destabilises SCA (the err-term tangent loses
    # validity outside a trust region around W^(n)).  We cap at 3×
    # (default) and use τ=1.5 (gentler than Boyd's 2.0) to creep up
    # slowly.  Cooldown + warmup + instability detector provide
    # further protection.
    adaptive_rho       = bool(getattr(cfg.algorithm.admm, "adaptive_rho", True))
    rho_tau            = float(getattr(cfg.algorithm.admm, "rho_tau", 1.5))
    rho_mu_balance     = float(getattr(cfg.algorithm.admm, "rho_mu_balance", 10.0))
    rho_max_factor     = float(getattr(cfg.algorithm.admm, "rho_max_factor", 3.0))
    rho_min_factor     = float(getattr(cfg.algorithm.admm, "rho_min_factor", 0.5))
    rho_adapt_warmup   = int  (getattr(cfg.algorithm.admm, "rho_adapt_warmup", 5))
    rho_adapt_interval = int  (getattr(cfg.algorithm.admm, "rho_adapt_interval", 3))

    rho_factor_current   = 1.0      # multiplier on top of input rho_admm
    last_rho_adapt_iter  = -10**9   # any value < -rho_adapt_interval works
    rho_history: List[float] = []

    # ── Patience-based early stopping (Stage 22b) ────────────────────────
    early_stop_patience   = int(getattr(cfg.algorithm.admm, "early_stop_patience", 15))
    early_stop_min_iters  = int(getattr(cfg.algorithm.admm, "early_stop_min_iters", 30))
    early_stopped = False

    if verbose:
        print(f"  [ADMM] rho={rho_admm}, kappa={kappa}, xi∈[{xi_floor:.1e}, "
              f"{xi_cap:.1e}], n_max={n_admm_max}, gamma_dB={lin2db(gamma_lin)}")
        print(f"  [ADMM] best_iter_criterion={best_iter_criterion}")
        print(f"  [ADMM] adaptive_rho={adaptive_rho} (τ={rho_tau}, "
              f"μ={rho_mu_balance}, range=[{rho_min_factor}, "
              f"{rho_max_factor}]·rho_admm)")
        if early_stop_patience > 0:
            print(f"  [ADMM] early_stop: patience={early_stop_patience}, "
                  f"min_iters={early_stop_min_iters}")

    # =====================================================================
    # ADMM main loop
    # =====================================================================
    # Optional per-iteration progress bar.  Used by convergence_trace
    # which runs ONE solve with up to ``n_admm_max`` outer iterations;
    # for Monte Carlo runners we'd just be re-drawing the bar thousands
    # of times, so the kwarg defaults to False.
    _iter_range = range(n_admm_max)
    if progress:
        try:
            from tqdm import tqdm
            _iter_range = tqdm(
                _iter_range, desc="ADMM iter", leave=True,
                total=n_admm_max, unit="iter",
            )
        except ImportError:
            pass  # silently no-op if tqdm isn't installed

    for n_iter in _iter_range:
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
            slot_a = ap_slot[a]
            # Reconstruct AP-specific residual: Σ_au^(n) = Σ̃_u^(n) − l_au^(n)
            # For cds/mui/s2ci this is the usual sum-consensus residual; for
            # the err block (Stage 23b) only AP a's own slot matters, giving
            # the scalar  Σ_au^err[a] = Σ̃_u^err[slot_a] − e_au^(n)
            #            ( = ν_u^err[a] − z_u^err[a] ).
            Sigma_au: Dict[int, ConsensusVector] = {}
            sigma_err_au: Dict[int, float] = {}
            e_prev_a: Dict[int, float] = {}
            for u in range(n_ue):
                s_t = Sigma_tilde_cur[u]
                l_au = l_cur[(a, u)]
                Sigma_au[u] = ConsensusVector(
                    cds  = s_t.cds  - l_au.cds,
                    mui  = s_t.mui  - l_au.mui,
                    s2ci = s_t.s2ci - l_au.s2ci,
                    err  = np.zeros(n_tx, dtype=np.float64),   # unused by local QCQP
                )
                sigma_err_au[u] = float(s_t.err[slot_a]) - l_au.err
                e_prev_a[u] = e_cur[(a, u)]

            W_a_new, ok = _solve_local_qcqp(
                a_idx=a,
                W_a_prev=W_cur[a],
                G_sca_a=G_sca.get(a),
                sensing_stats=sensing_stats,
                est=est,
                topo=topo,
                Sigma_au=Sigma_au,
                sigma_err_au=sigma_err_au,
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
                n_tx=n_tx,
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
                err  = nu_cur[u].err  + (sum_l_u.err  - z_u.err),   # per-AP vector
            )

        # New broadcast Σ̃_u^(n+1) = Σ_a l_au^(n+1) − z_u^(n+1) + ν_u^(n+1)
        Sigma_tilde_new: Dict[int, ConsensusVector] = {}
        for u in range(n_ue):
            sum_l_u = _sum_local_contributions(l_new, tx_idx, u, n_mui, n_t)
            Sigma_tilde_new[u] = ConsensusVector(
                cds  = sum_l_u.cds  - z_new[u].cds  + nu_new[u].cds,
                mui  = sum_l_u.mui  - z_new[u].mui  + nu_new[u].mui,
                s2ci = sum_l_u.s2ci - z_new[u].s2ci + nu_new[u].s2ci,
                err  = sum_l_u.err  - z_new[u].err  + nu_new[u].err,   # per-AP vector
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
                                                 _zeros_consensus(n_mui, n_t, n_tx))
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

        # ── Best-iterate snapshot (Stage 22a + Stage 23a) ─────────────────
        # Delegates the decision to the pure _should_take_iterate rule so
        # the three criteria stay testable in isolation.  Feasibility is
        # the SAME definition used by the downstream infeasibility metric:
        # every user meets its γ, i.e. np.all(SINR_u ≥ γ_u).
        cand_min_sinr   = float(sinr_vec.min()) if sinr_vec.size else 0.0
        cand_slack      = float(np.max(eps_new)) if eps_new.size else 0.0
        cand_resid_norm = float(r_pri + r_dual)
        cand_feasible   = bool(np.all(sinr_vec >= gamma_lin)) if sinr_vec.size else False
        take = all_ok and _should_take_iterate(
            best_iter_criterion,
            cand_feasible=cand_feasible,
            cand_min_sinr=cand_min_sinr,
            cand_resid_norm=cand_resid_norm,
            cand_r_pri=r_pri,
            best_feasible=best_feasible,
            best_min_sinr=best_min_sinr,
            best_resid_norm=best_resid_norm,
            best_r_pri=best_r_pri,
        )
        if take:
            best_min_sinr   = cand_min_sinr
            best_r_pri      = r_pri
            best_resid_norm = cand_resid_norm
            best_slack_norm = cand_slack
            best_feasible   = cand_feasible
            W_best     = {a: W_new[a].copy() for a in tx_idx}
            best_iter  = n_iter + 1
            best_sens  = sens_val
            best_sinr  = sinr_vec.copy()
            best_slack = eps_new.copy()

        # ── Adaptive ρ — Boyd §3.4.1 (Stage 22b) ─────────────────────────
        # Use relative residuals (r/ε) for balance because the dual
        # residual uses user-facing rho_admm (=1 by default) and is
        # systematically small.  Normalising by tolerances cancels skew.
        #
        # Safeguards (any one disables the update for this iter):
        #   1. Warmup (rho_adapt_warmup): don't adapt while SCA settles.
        #   2. Cooldown (rho_adapt_interval): wait between changes.
        #   3. Bounds [rho_min_factor, rho_max_factor]: cap at 3× of
        #      input rho_admm (half the empirical instability threshold).
        #   4. Instability detector: if r_pri grew >5 % iter-over-iter,
        #      NEVER increase ρ — decrease instead (SCA trust-region
        #      violation symptom).
        if adaptive_rho and (n_iter + 1) >= rho_adapt_warmup:
            iters_since_last_adapt = (n_iter + 1) - last_rho_adapt_iter
            can_adapt = iters_since_last_adapt >= rho_adapt_interval

            # Relative-residual balance test
            r_pri_rel  = r_pri  / max(eps_pri,  1e-30)
            r_dual_rel = r_dual / max(eps_dual, 1e-30)

            # Instability: primal residual growing iter-over-iter
            r_pri_growing = (
                len(prim_hist) >= 2
                and prim_hist[-2] > 0
                and r_pri > 1.05 * prim_hist[-2]
            )

            new_factor = rho_factor_current
            if can_adapt:
                if r_pri_growing:
                    # SCA trust region likely violated — back off only.
                    if rho_factor_current > rho_min_factor + 1e-9:
                        new_factor = max(
                            rho_factor_current / rho_tau, rho_min_factor
                        )
                elif r_pri_rel > rho_mu_balance * r_dual_rel:
                    # Primal-dominated → increase ρ
                    new_factor = min(
                        rho_factor_current * rho_tau, rho_max_factor
                    )
                elif r_dual_rel > rho_mu_balance * r_pri_rel:
                    # Dual-dominated → decrease ρ
                    new_factor = max(
                        rho_factor_current / rho_tau, rho_min_factor
                    )

            if abs(new_factor - rho_factor_current) > 1e-9:
                # Rescale scaled-form dual ν when ρ changes.  Scaled
                # ADMM has u = λ/ρ; to hold λ constant when ρ → τρ,
                # set u ← u/τ.  Formula: nu_rescale = ρ_old/ρ_new.
                nu_rescale = rho_factor_current / new_factor
                for u_ in range(n_ue):
                    nu_cur[u_] = ConsensusVector(
                        cds  = nu_cur[u_].cds  * nu_rescale,
                        mui  = nu_cur[u_].mui  * nu_rescale,
                        s2ci = nu_cur[u_].s2ci * nu_rescale,
                        err  = nu_cur[u_].err  * nu_rescale,
                    )
                rho_factor_current  = new_factor
                rho_effective       = rho_admm * rho_factor_current * auto_rho_factor
                last_rho_adapt_iter = n_iter + 1
                if verbose:
                    print(f"  [ADMM] ρ-adapt iter {n_iter+1}: "
                          f"factor={rho_factor_current:.3f}, "
                          f"ρ_eff={rho_effective:.3e}, "
                          f"r_pri/ε={r_pri_rel:.2f}, "
                          f"r_dual/ε={r_dual_rel:.2f}, "
                          f"r_pri_growing={r_pri_growing}")

        rho_history.append(rho_factor_current)

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
                  f"feas={cand_feasible}  "
                  f"maxSlack={eps_new.max():.2e}  ξ={xi_cur:.2e}  "
                  f"maxν^sinr={nu_sinr.max():.2e}")

        # ── Advance state ────────────────────────────────────────────────
        W_cur            = W_new
        l_cur            = l_new
        e_cur            = e_new
        z_cur            = z_new
        nu_cur           = nu_new
        Sigma_tilde_cur  = Sigma_tilde_new

        # ── Patience-based early stop (Stage 22b; feasibility-gated in the
        #    Stage 23 follow-up) ────────────────────────────────────────────
        # Bails out when no new best iterate has been recorded for
        # `early_stop_patience` consecutive iters, after a warmup of
        # `early_stop_min_iters`, and only if a best iterate exists.
        # Applies to the residual-based criteria (residual_norm and
        # feasible_then_residual); the legacy min_sinr criterion runs full
        # n_max (its swing peaks make "no improvement" unreliable).  Set
        # early_stop_patience=0 to disable.
        #
        # FEASIBILITY GATE (`and best_feasible`): the patience clock only runs
        # once we hold a FEASIBLE incumbent.  Without it, the residual reaches
        # a local minimum early (consensus momentarily tight) while min-SINR is
        # still far below γ, so patience would fire on that early low-residual
        # iterate and return a sub-γ beamformer.  The residual and the SINR
        # feasibility converge on very different timescales (feasibility is
        # reached much later), so gating on feasibility is essential: fast
        # trials still stop early once feasible, slow trials run until feasible,
        # and genuinely-infeasible trials run the full n_max (correctly flagged
        # rather than early-stopped into a bad iterate).
        if (
            best_iter_criterion in ("residual_norm", "feasible_then_residual")
            and early_stop_patience > 0
            and best_iter > 0
            and best_feasible
            and (n_iter + 1) >= early_stop_min_iters
        ):
            iters_since_best = (n_iter + 1) - best_iter
            if iters_since_best >= early_stop_patience:
                if verbose:
                    print(f"  [ADMM] early stop at iter {n_iter+1}: "
                          f"no improvement for {iters_since_best} iters "
                          f"(best at {best_iter})")
                early_stopped = True
                break

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
            # Return the best iterate recorded (per the active criterion)
            # rather than the just-converged one; they coincide in the
            # common case, but the best-iterate may have higher min-SINR
            # (feasible_then_residual) or tighter consensus.  Falls back to
            # W_cur if no iterate was ever recorded as best.
            W_return = W_best if best_iter > 0 else W_cur
            return ADMMResult(
                W_tx=W_return,
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
                best_iter=best_iter,
                rho_history=rho_history,
                early_stopped=False,
            )

        if consec_fail >= MAX_CONSEC_FAIL:
            if verbose:
                print(f"  [ADMM] aborting after {consec_fail} consecutive "
                      f"iterations with inner failures")
            break

    # ── Non-converged exit: return BEST iterate ──────────────────────────
    # Under feasible_then_residual this is the best feasible incumbent (or
    # lowest-residual iterate if none was feasible); under residual_norm the
    # lowest-residual iterate; under min_sinr the highest-min-SINR iterate.
    if best_iter > 0:
        if verbose:
            print(f"  [ADMM] no convergence — returning best iterate "
                  f"({best_iter}/{len(prim_hist)}, r_pri={best_r_pri:.3e}, "
                  f"feasible={best_feasible})")
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
            best_iter=best_iter,
            rho_history=rho_history,
            early_stopped=early_stopped,
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
        best_iter=best_iter,
        rho_history=rho_history,
        early_stopped=early_stopped,
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

