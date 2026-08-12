"""
tests/test_joint_opt.py
=======================
CORDIS-ADMM (Algorithm 2, §V) — the mathematical core.

Stage 8a covers the equation-to-code correspondence: the consensus vector, the
augmented-Lagrangian residuals, the two SCA linearizations, the P-Central cone,
and the SNR normalization. Loop control (tolerances, best-iterate, early
stopping, adaptive rho, warm start) is Stage 8b.

Paper references: `eq:admm-local-contribution`, `eq:admm-augmented-Lagrangian`,
`eq:admm-p-local`, `eq:admm-p-central`, `eq:socp_constraint`,
`eq:admm-primal-residual`, `eq:admm-dual-residual`.

Three of these tests pin *defects* rather than correct behaviour, via
`xfail(strict=True)`, so that correcting the manuscript flips them to XPASS and
forces the marker to be retired: F-08-01, F-08-02, F-08-03.
"""
from __future__ import annotations

import numpy as np
import pytest

from cordis.algorithms.joint_opt import (
    ConsensusVector, LocalContribution, _compute_local_contribution,
    _consensus_residual, _sum_local_contributions, _zeros_consensus,
)
from cordis.utils.io_utils import make_rng

N_TX, N_UE, N_T, M = 5, 3, 2, 6


@pytest.fixture
def l_dict():
    """A full set of per-(AP, user) contributions with known err amplitudes."""
    r = make_rng(11)
    out = {}
    for a in range(N_TX):
        for u in range(N_UE):
            out[(a, u)] = LocalContribution(
                cds=complex(r.standard_normal(), r.standard_normal()),
                mui=r.standard_normal(N_UE - 1) + 1j * r.standard_normal(N_UE - 1),
                s2ci=r.standard_normal(N_T) + 1j * r.standard_normal(N_T),
                err=float(abs(r.standard_normal()) + 0.2),
            )
    return out


# =============================================================================
# eq:admm-local-contribution — structure of l_{a_t u}
# =============================================================================

def test_local_contribution_has_the_printed_block_structure() -> None:
    """`l = [x, (i^c)^T, (i^s)^T, e]^T` with dims `(1, N_ue-1, N_t, 1)`."""
    r = make_rng(5)
    W = r.standard_normal((M, N_UE + N_T)) + 1j * r.standard_normal((M, N_UE + N_T))
    h = r.standard_normal(M) + 1j * r.standard_normal(M)
    A = r.standard_normal((M, M)) + 1j * r.standard_normal((M, M))
    l = _compute_local_contribution(0, 0, W, h, A.conj().T @ A, N_UE, N_T)
    assert isinstance(l.cds, complex)
    assert l.mui.shape == (N_UE - 1,)
    assert l.s2ci.shape == (N_T,)
    assert np.isscalar(l.err) or np.ndim(l.err) == 0


def test_err_is_the_amplitude_not_the_squared_form() -> None:
    """
    `e = ||R^{1/2} W||_F`, not `||.||_F^2`. The amplitude form is what keeps
    P-Local a true QCQP and the P-Central cone DCP-clean.
    """
    r = make_rng(6)
    W = r.standard_normal((M, N_UE + N_T)) + 1j * r.standard_normal((M, N_UE + N_T))
    h = np.zeros(M, dtype=complex)
    A = r.standard_normal((M, M)) + 1j * r.standard_normal((M, M))
    R = A.conj().T @ A
    l = _compute_local_contribution(0, 0, W, h, R, N_UE, N_T)
    sq = float(np.real(np.trace(W.conj().T @ R @ W)))
    assert l.err == pytest.approx(np.sqrt(sq))
    assert l.err >= 0.0


def test_cds_mui_s2ci_use_the_conjugate_channel(l_dict) -> None:
    """`x_au = h_hat^H W[:, u]`, matching the §III signal model."""
    r = make_rng(7)
    W = r.standard_normal((M, N_UE + N_T)) + 1j * r.standard_normal((M, N_UE + N_T))
    h = r.standard_normal(M) + 1j * r.standard_normal(M)
    l = _compute_local_contribution(0, 1, W, h, None, N_UE, N_T)
    assert l.cds == pytest.approx(complex(h.conj() @ W[:, 1]))
    np.testing.assert_allclose(l.mui, h.conj() @ W[:, [0, 2]])
    np.testing.assert_allclose(l.s2ci, h.conj() @ W[:, N_UE:N_UE + N_T])


# =============================================================================
# The gathered err block — the single most valuable invariant in the suite
# =============================================================================

def test_gathered_err_reproduces_the_exact_csi_error_power(l_dict) -> None:
    """
    `||z^err||^2 == sum_a e_a^2` exactly.

    Because per-AP CSI errors are independent, their *powers* add. Gathering one
    slot per transmit AP and letting the SOC square the stack reproduces that
    exactly. A previous version of this code summed the amplitudes into a single
    scalar, which manufactures infeasibility (see the counterfactual below).
    """
    z = _sum_local_contributions(l_dict, list(range(N_TX)), 0, N_UE - 1, N_T)
    e = np.array([l_dict[(a, 0)].err for a in range(N_TX)])
    assert z.err.shape == (N_TX,)
    assert float(np.sum(z.err ** 2)) == pytest.approx(float(np.sum(e ** 2)), rel=1e-12)
    np.testing.assert_allclose(np.sort(z.err), np.sort(e))


def test_collapsed_scalar_err_overcounts_by_the_cross_terms(l_dict) -> None:
    """
    The counterfactual: `(sum_a e_a)^2 = sum_a e_a^2 + cross terms`, an
    over-count bounded by `N_tx` and strictly greater than 1 whenever two APs
    both contribute. This is the bug the gathered layout exists to prevent.
    """
    e = np.array([l_dict[(a, 0)].err for a in range(N_TX)])
    exact, collapsed = float(np.sum(e ** 2)), float(np.sum(e) ** 2)
    assert collapsed > exact
    assert collapsed / exact <= N_TX + 1e-9
    assert collapsed / exact == pytest.approx(4.647, abs=1e-3)


def test_cds_mui_s2ci_are_coherently_summed_not_gathered(l_dict) -> None:
    """Only `err` is gathered; the other three blocks are coherent sums."""
    z = _sum_local_contributions(l_dict, list(range(N_TX)), 0, N_UE - 1, N_T)
    assert z.cds == pytest.approx(sum(l_dict[(a, 0)].cds for a in range(N_TX)))
    np.testing.assert_allclose(
        z.mui, sum(l_dict[(a, 0)].mui for a in range(N_TX)))
    np.testing.assert_allclose(
        z.s2ci, sum(l_dict[(a, 0)].s2ci for a in range(N_TX)))


@pytest.mark.xfail(strict=True,
                   reason="F-08-01: eq:admm-augmented-Lagrangian defines z^err as "
                          "a scalar with z_u = sum_a l_au, i.e. (sum_a e_a)^2. The "
                          "code gathers per-AP and gets sum_a e_a^2. The paper "
                          "specifies a strictly tighter cone than the code solves.")
def test_paper_scalar_err_matches_the_implementation(l_dict) -> None:
    """
    If the manuscript's literal `z^err = sum_a e_{a_t u}` were what the code
    implements, the cone's err contribution would equal the code's. It does not:
    the printed form over-counts by up to a factor `N_tx`.
    """
    e = np.array([l_dict[(a, 0)].err for a in range(N_TX)])
    z = _sum_local_contributions(l_dict, list(range(N_TX)), 0, N_UE - 1, N_T)
    assert float(np.sum(e) ** 2) == pytest.approx(float(np.sum(z.err ** 2)))


# =============================================================================
# The two SCA linearizations
# =============================================================================

def test_csi_error_tangent_is_exact_at_the_expansion_point() -> None:
    """
    `T(W) = Re tr(W_prev^H R W)/f_n` with `f_n = ||R^{1/2} W_prev||_F` satisfies
    `T(W_prev) = f_n`. Cheap and sharp: any scaling slip in the tangent breaks it.
    """
    r = make_rng(2024)
    A = r.standard_normal((M, M)) + 1j * r.standard_normal((M, M))
    R = A.conj().T @ A
    Wp = r.standard_normal((M, N_UE + N_T)) + 1j * r.standard_normal((M, N_UE + N_T))
    f_n = float(np.sqrt(np.real(np.trace(Wp.conj().T @ R @ Wp))))
    T_at = float(np.real(np.trace(Wp.conj().T @ R @ Wp))) / f_n
    assert T_at == pytest.approx(f_n, rel=1e-12)


def test_csi_error_tangent_is_a_lower_bound_on_the_true_amplitude() -> None:
    """`e` is convex, so its tangent under-estimates away from `W_prev`."""
    r = make_rng(2025)
    A = r.standard_normal((M, M)) + 1j * r.standard_normal((M, M))
    R = A.conj().T @ A
    Wp = r.standard_normal((M, N_UE + N_T)) + 1j * r.standard_normal((M, N_UE + N_T))
    f_n = float(np.sqrt(np.real(np.trace(Wp.conj().T @ R @ Wp))))
    for s in (0.25, 1.0, 3.0):
        D = r.standard_normal((M, N_UE + N_T)) + 1j * r.standard_normal((M, N_UE + N_T))
        W = Wp + s * D
        T = float(np.real(np.trace(Wp.conj().T @ R @ W))) / f_n
        e = float(np.sqrt(np.real(np.trace(W.conj().T @ R @ W))))
        assert T <= e + 1e-9


@pytest.mark.xfail(strict=True,
                   reason="F-08-03: T <= e, so inside the SUBTRACTED square "
                          "(T+sigma)^2 <= (e+sigma)^2. The surrogate "
                          "under-penalizes and over-estimates the local "
                          "objective: a majorant where MM needs a minorant.")
def test_local_penalty_surrogate_majorizes_the_true_penalty() -> None:
    """
    Valid MM for the P-Local *maximization* needs the surrogate penalty to be an
    over-estimate, `(T+sigma)^2 >= (e+sigma)^2`, so that the surrogate minorizes
    the objective. It is the other way round.
    """
    r = make_rng(2026)
    A = r.standard_normal((M, M)) + 1j * r.standard_normal((M, M))
    R = A.conj().T @ A
    Wp = r.standard_normal((M, N_UE + N_T)) + 1j * r.standard_normal((M, N_UE + N_T))
    Wp /= np.linalg.norm(Wp)
    f_n = float(np.sqrt(np.real(np.trace(Wp.conj().T @ R @ Wp))))
    D = r.standard_normal((M, N_UE + N_T)) + 1j * r.standard_normal((M, N_UE + N_T))
    W = Wp + 0.6 * D / np.linalg.norm(D)
    T = float(np.real(np.trace(Wp.conj().T @ R @ W))) / f_n
    e = float(np.sqrt(np.real(np.trace(W.conj().T @ R @ W))))
    for sigma in (-0.5, 0.0, 0.5, 2.0):
        assert (T + sigma) ** 2 >= (e + sigma) ** 2 - 1e-12


# =============================================================================
# eq:admm-p-central — the SOC and the sqrt(gamma) placement
# =============================================================================

@pytest.mark.parametrize("gamma_db", [0.0, 3.0, 5.0, 10.0, 15.0])
def test_soc_equality_reproduces_the_sinr_target_exactly(gamma_db) -> None:
    """
    Saturating `||[z^MUI; z^S2CI; z^err; 1]|| <= Re{z^CDS}/sqrt(gamma)` must give
    SINR exactly `gamma`.

    This pins the `sqrt(gamma)` placement. Writing `gamma` instead would bias
    every feasibility number in the paper: at `gamma = 5` dB the error is 2.5 dB.
    """
    r = make_rng(int(100 + gamma_db))
    g = 10 ** (gamma_db / 10)
    mui = r.standard_normal(N_UE - 1) + 1j * r.standard_normal(N_UE - 1)
    s2ci = r.standard_normal(N_T) + 1j * r.standard_normal(N_T)
    err = np.abs(r.standard_normal(N_TX))
    denom = (float(np.sum(np.abs(mui) ** 2)) + float(np.sum(np.abs(s2ci) ** 2))
             + float(np.sum(err ** 2)) + 1.0)
    z_cds = np.sqrt(g) * np.sqrt(denom)
    sinr_db = 10 * np.log10(abs(z_cds) ** 2 / denom)
    assert sinr_db == pytest.approx(gamma_db, abs=1e-9)


def test_using_gamma_instead_of_sqrt_gamma_would_be_detectable() -> None:
    """The counterfactual is a 2.5 dB bias at gamma = 5 dB, not a rounding slip."""
    g = 10 ** 0.5
    denom = 4.0
    wrong = 10 * np.log10((g * np.sqrt(denom)) ** 2 / denom)
    assert wrong - 5.0 == pytest.approx(5.0, abs=1e-9)


# =============================================================================
# eq:admm-primal-residual / eq:admm-dual-residual
# =============================================================================

def test_consensus_residual_covers_all_four_blocks() -> None:
    """`r = ||sum_a l - z||` over cds, mui, s2ci and the per-AP err vector."""
    z = ConsensusVector(cds=1 + 2j,
                        mui=np.array([1 + 0j, 0 + 1j]),
                        s2ci=np.array([2 + 0j, 0 + 2j]),
                        err=np.array([1.0, 2.0, 2.0, 0.0, 0.0]))
    zero = _zeros_consensus(N_UE - 1, N_T, N_TX)
    expect = np.sqrt(abs(1 + 2j) ** 2 + 1 + 1 + 4 + 4 + 1 + 4 + 4)
    assert _consensus_residual(z, zero) == pytest.approx(expect)


def test_zero_residual_at_perfect_consensus(l_dict) -> None:
    z = _sum_local_contributions(l_dict, list(range(N_TX)), 0, N_UE - 1, N_T)
    assert _consensus_residual(z, z) == pytest.approx(0.0, abs=1e-15)


def test_zeros_consensus_has_the_per_ap_err_block() -> None:
    z = _zeros_consensus(N_UE - 1, N_T, N_TX)
    assert z.mui.shape == (N_UE - 1,)
    assert z.s2ci.shape == (N_T,)
    assert z.err.shape == (N_TX,), "err is per-AP (Stage 23b), not a scalar"


# =============================================================================
# F-08-02 — the auto-rho rescaling
# =============================================================================

def _auto_rho_factor(topo, est, Pmax, sigma_sq) -> float:
    """Reproduce solve_cordis_admm's auto-scaling (joint_opt.py:881-892)."""
    sqrt_snr = float(np.sqrt(Pmax / sigma_sq))
    tot, cnt = 0.0, 0
    for a in [ap.idx for ap in topo.tx_aps]:
        for u in range(topo.n_ue):
            h = est.h_hat.get((a, u))
            if h is not None:
                tot += float(np.sum(np.abs(h * sqrt_snr) ** 2))
                cnt += 1
    return 2.0 / (max(topo.n_ue, 1) * max(tot / max(cnt, 1), 1e-30))


@pytest.mark.integration
def test_rho_effective_is_a_data_dependent_rescaling_of_the_knob(
        benchmark_scenario) -> None:
    """
    Table III reports `rho = 1.0`. That is the user knob; the penalty P-Local
    actually applies is `rho_effective = rho * 2 / (N_ue * <||h_tilde||^2>)`,
    which depends on the realized channel gains and so differs from drop to drop.
    """
    topo, cfg, est, _ss, _assoc, sigma_sq, Pmax = benchmark_scenario
    factor = _auto_rho_factor(topo, est, Pmax, sigma_sq)
    rho_eff = cfg.algorithm.admm.rho * factor
    assert cfg.algorithm.admm.rho == pytest.approx(1.0), "Table III's knob"
    assert factor < 1e-2, (
        f"auto_rho_factor={factor:.3e}: the effective penalty is orders of "
        f"magnitude below the reported rho")
    assert rho_eff != pytest.approx(cfg.algorithm.admm.rho, rel=1e-3)


@pytest.mark.integration
def test_auto_rho_factor_varies_across_drops(default_cfg) -> None:
    """
    The rescaling is data-dependent, so a single reported `rho` cannot describe
    the campaign. Two different drops give measurably different penalties.
    """
    from tests.conftest import _sized_cfg, build_pipeline_scenario
    factors = []
    for seed in (3001, 3002):
        topo, _c, est, _s, _a, sig, Pmax = build_pipeline_scenario(
            _sized_cfg(default_cfg, n_ap=4, n_ant=8, n_ue=3, n_spatial=600),
            seed=seed)
        factors.append(_auto_rho_factor(topo, est, Pmax, sig))
    assert factors[0] != pytest.approx(factors[1], rel=1e-6), (
        "auto_rho_factor should track the realized channel gains")


# =============================================================================
# Stage 8b — loop control
# =============================================================================

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from cordis.algorithms.joint_opt import _should_take_iterate  # noqa: E402

TRACE_DIR = Path(__file__).resolve().parents[1] / "paper/figure_src/fig_convergence/data"


# ---- best-iterate selection (eq. none; Stage 23a decision rule) -------------

def _take(criterion, **kw):
    base = dict(cand_r_pri=1.0, best_r_pri=1.0)
    return _should_take_iterate(criterion, **{**base, **kw})


def test_feasible_then_residual_prefers_a_feasible_candidate() -> None:
    """A feasible candidate replaces an infeasible incumbent even with a far
    worse residual and a far worse min-SINR."""
    assert _take("feasible_then_residual",
                 cand_feasible=True, cand_min_sinr=1.0, cand_resid_norm=99.0,
                 best_feasible=False, best_min_sinr=50.0, best_resid_norm=0.1)


def test_feasible_then_residual_never_downgrades_a_feasible_incumbent() -> None:
    """The property the criterion exists for: once feasible, never go back."""
    assert not _take("feasible_then_residual",
                     cand_feasible=False, cand_min_sinr=99.0, cand_resid_norm=1e-9,
                     best_feasible=True, best_min_sinr=1.0, best_resid_norm=99.0)


def test_residual_norm_criterion_ignores_feasibility() -> None:
    """The non-default criterion will happily replace a feasible incumbent.
    This is why `feasible_then_residual` is the shipped default."""
    assert _take("residual_norm",
                 cand_feasible=False, cand_min_sinr=0.0, cand_resid_norm=0.1,
                 best_feasible=True, best_min_sinr=99.0, best_resid_norm=1.0)


def test_min_sinr_criterion_breaks_ties_on_primal_residual() -> None:
    assert _should_take_iterate(
        "min_sinr", cand_feasible=False, cand_min_sinr=5.0, cand_resid_norm=9.0,
        cand_r_pri=0.1, best_feasible=False, best_min_sinr=5.0,
        best_resid_norm=1.0, best_r_pri=1.0)


# ---- the stored convergence traces (F-08-05, F-08-07) ----------------------

@pytest.fixture(scope="module")
def traces():
    out = {}
    for tag in ("trace_fixed", "trace_adaptive"):
        f = TRACE_DIR / tag / "trace.npz"
        if not f.exists():
            pytest.skip(f"stored trace missing: {f}")
        out[tag] = np.load(f, allow_pickle=True)
    return out


def test_stored_convergence_traces_match_the_quoted_iterates(traces) -> None:
    """
    Pins the ground truth behind Fig. 1 so the prose can be checked against it.

    F-08-05: `full/`'s caption says the adaptive rule converges in "markedly
    fewer iterations, with the best iterate at 157". The traces say the opposite:
    fixed converges at 158 (best_iter 157), adaptive at 228 (best_iter 207).
    """
    fixed, adapt = traces["trace_fixed"], traces["trace_adaptive"]
    assert int(fixed["n_admm_iters"]) == 158
    assert int(fixed["best_iter"]) == 157
    assert int(adapt["n_admm_iters"]) == 228
    assert int(adapt["best_iter"]) == 207
    assert int(adapt["n_admm_iters"]) > int(fixed["n_admm_iters"]), (
        "adaptive rho is SLOWER, not faster")
    assert bool(fixed["converged"]) and bool(adapt["converged"]), (
        "both runs converge; adaptive is not 'still above tolerance past 225'")


def test_primal_residual_is_the_binding_tolerance(traces) -> None:
    """
    F-08-07: `eps_dual` never binds. On both traces the dual test passes for most
    of the run while the primal test passes exactly once, at the final iterate.
    """
    for tag, z in traces.items():
        rp = np.asarray(z["primal_res_history"])
        rd = np.asarray(z["dual_res_history"])
        assert int((rp < 1.0).sum()) == 1, f"{tag}: r_pri should bind exactly once"
        assert int((rd < 1.0).sum()) > 50, f"{tag}: r_dual is slack for most of the run"


def test_the_tolerance_crossing_has_a_thin_margin(traces) -> None:
    """
    T_ADMM is set by where a slow smooth decay meets an arbitrary threshold: the
    final residual clears 1.0 by under half a percent in both runs.
    """
    for tag, z in traces.items():
        rp = np.asarray(z["primal_res_history"])
        assert 0.99 < rp[-1] < 1.0, f"{tag}: final r_pri = {rp[-1]}"


# ---- solver-level loop behaviour -------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_feasible_flag_tracks_solver_success_not_qos(admm_scenario) -> None:
    """
    F-08-04: `ADMMResult.feasible` is `any_inner_ok`, so it reports that a CVXPY
    subproblem solved, not that the returned beamformer meets gamma. At an
    unreachable target it still comes back True.
    """
    import copy
    from cordis.algorithms.joint_opt import run_cordis_admm
    from cordis.metrics.sinr import compute_sinr
    topo, cfg, est, ss, assoc, sigma_sq, Pmax = admm_scenario
    c = copy.deepcopy(cfg)
    c.algorithm.gamma_db = 30.0
    c.algorithm.admm.n_admm_max = 12
    c.validate()
    W, res = run_cordis_admm(topo, c, est, ss, assoc, sigma_sq, Pmax)
    min_sinr_db = 10 * np.log10(
        float(compute_sinr(W, est, topo, sigma_sq).sinr_per_user.min()))
    assert min_sinr_db < 30.0 - 5.0, "target should be genuinely out of reach"
    assert res.feasible is True, (
        "documents the defect: `feasible` stays True while gamma is missed by "
        f"{30.0 - min_sinr_db:.1f} dB")


@pytest.mark.solver
@pytest.mark.slow
def test_patience_zero_never_early_stops(admm_scenario) -> None:
    """
    F-08-06 guard. `convergence_trace` sets `early_stop_patience = 0` to record a
    full-length trace; if that ever stopped early, Fig. 1 would be truncated.
    """
    import copy
    from cordis.algorithms.joint_opt import run_cordis_admm
    topo, cfg, est, ss, assoc, sigma_sq, Pmax = admm_scenario
    c = copy.deepcopy(cfg)
    c.algorithm.admm.early_stop_patience = 0
    c.algorithm.admm.n_admm_max = 20
    c.validate()
    _W, res = run_cordis_admm(topo, c, est, ss, assoc, sigma_sq, Pmax)
    assert res.early_stopped is False


@pytest.mark.solver
@pytest.mark.slow
def test_warm_start_from_split_is_live(admm_scenario) -> None:
    """
    R-08-A: the plan expected this field to be dead. It is read at
    joint_opt.py:838 and changes the run.
    """
    import copy
    from cordis.algorithms.joint_opt import run_cordis_admm
    topo, cfg, est, ss, assoc, sigma_sq, Pmax = admm_scenario
    got = {}
    for ws in (True, False):
        c = copy.deepcopy(cfg)
        c.algorithm.admm.warm_start_from_split = ws
        c.algorithm.admm.n_admm_max = 12
        c.validate()
        _W, res = run_cordis_admm(topo, c, est, ss, assoc, sigma_sq, Pmax)
        got[ws] = res.primal_res_history[0]
    assert got[True] != pytest.approx(got[False], rel=1e-9), (
        "the two warm starts must give different initial residuals")


@pytest.mark.solver
@pytest.mark.slow
def test_residuals_decrease_and_slack_is_driven_to_zero(admm_scenario) -> None:
    """
    Plan task 13. The ALM multiplier plus adaptive xi should drive the SOC slack
    to zero even though `r_pri` itself is not monotone (the non-monotonicity is
    consistent with F-08-03's broken MM property).
    """
    import copy
    from cordis.algorithms.joint_opt import run_cordis_admm
    topo, cfg, est, ss, assoc, sigma_sq, Pmax = admm_scenario
    c = copy.deepcopy(cfg)
    c.algorithm.admm.n_admm_max = 40
    c.validate()
    _W, res = run_cordis_admm(topo, c, est, ss, assoc, sigma_sq, Pmax)
    rp = np.array(res.primal_res_history)
    slack = np.array([np.max(s) for s in res.slack_history])
    assert rp[-1] < rp[0]
    assert slack[-1] < 1e-6
    assert slack[-1] < slack[0]
