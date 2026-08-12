"""
tests/test_centralized.py
=========================
The centralized benchmark (`cordis/algorithms/centralized.py`), which the
paper presents as **P-Global** and as the performance ceiling that
CORDIS-ADMM approaches (§V, Fig. 2, Fig. 3).

Two properties matter for the manuscript's claims and are pinned here:

1. **It is a ceiling.** Its sensing utility must be at least CORDIS-Split's
   on the same scenario, otherwise "approaches the centralized bound" is
   meaningless.
2. **It honors the per-user SINR floor `gamma`.** §VI-C shows Centralized
   tracking the `y = x` guide in Fig. 2, which is only true if the returned
   `W` actually satisfies the constraint it was given.

Property 2 holds at the shipped geometry but *not* unconditionally, and the
returned `feasible` flag does not detect the difference (F-07-02).

These tests run real CVXPY solves and are marked `solver` + `slow`.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from cordis.algorithms.centralized import (
    CentralizedResult, run_centralized, solve_centralized,
)
from cordis.algorithms.split_opt import run_cordis_split
from cordis.metrics.sinr import compute_sinr

pytestmark = [pytest.mark.solver, pytest.mark.slow]


# =============================================================================
# Helpers
# =============================================================================

def _min_sinr_db(W, est, topo, sigma) -> float:
    return float(10.0 * np.log10(np.maximum(
        compute_sinr(W, est, topo, sigma).sinr_per_user, 1e-30)).min())


@pytest.fixture(scope="module")
def cent(centralized_scenario):
    """Run the centralized solver once; the solve dominates the runtime."""
    topo, cfg, est, ss, assoc, sigma, Pmax = centralized_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W, res = run_centralized(topo, cfg, est, ss, assoc, sigma, Pmax)
    return W, res


# =============================================================================
# Shape and power feasibility
# =============================================================================

def test_returns_one_beamformer_per_transmit_ap(cent, centralized_scenario) -> None:
    topo, cfg, *_ = centralized_scenario
    W, res = cent
    assert isinstance(res, CentralizedResult)
    assert set(W) == {ap.idx for ap in topo.tx_aps}
    for ap in topo.tx_aps:
        assert W[ap.idx].shape == (cfg.topology.n_ant,
                                   topo.n_ue + topo.n_targets)


def test_respects_the_per_ap_power_budget(cent, centralized_scenario) -> None:
    """
    `||W_a||_F^2 <= Pmax` for every AP. This is the constraint that makes the
    benchmark a legitimate comparison rather than an unconstrained bound.
    """
    topo, _, _, _, _, _, Pmax = centralized_scenario
    W, _ = cent
    for ap in topo.tx_aps:
        assert np.linalg.norm(W[ap.idx], "fro") ** 2 <= Pmax * (1 + 1e-6)


def test_diagnostics_are_populated(cent) -> None:
    _, res = cent
    assert len(res.objective_history) == res.n_sca_iters >= 1
    assert len(res.sinr_history) == len(res.power_history) == res.n_sca_iters
    assert res.solver != ""
    assert res.inner_failures >= 0


# =============================================================================
# The SINR floor
# =============================================================================

def test_honors_gamma_at_the_shipped_geometry(cent, centralized_scenario) -> None:
    """
    The headline claim behind Fig. 2. Measured over five drops at
    `n_ap=10, M=16` the achieved min SINR lands within 0.01 dB of `gamma`
    every time, so a 0.25 dB tolerance is generous.
    """
    topo, cfg, est, _, _, sigma, _ = centralized_scenario
    W, _ = cent
    assert _min_sinr_db(W, est, topo, sigma) >= cfg.algorithm.gamma_db - 0.25


def test_gamma_is_active_not_incidental(centralized_scenario) -> None:
    """
    Raising `gamma` raises the achieved min SINR, i.e. the constraint is
    binding rather than satisfied by accident.
    """
    topo, cfg, est, ss, assoc, sigma, Pmax = centralized_scenario
    got = []
    for g in (0.0, 5.0, 10.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W, _ = run_centralized(topo, cfg, est, ss, assoc, sigma, Pmax,
                                   gamma_u_db=np.full(topo.n_ue, g))
        got.append(_min_sinr_db(W, est, topo, sigma))
    assert got[0] < got[1] < got[2]
    for g, s in zip((0.0, 5.0, 10.0), got):
        assert s >= g - 0.25, f"gamma={g} dB: achieved only {s:.2f} dB"


def test_an_infeasible_instance_returns_the_phase_i_warm_start(
        small_centralized_scenario) -> None:
    """
    F-07-02, the mechanism. At `n_ap=4, M=6` for 3 users plus a target the
    inner SOCP fails every time. `_solve_inner` returns `W_prev` on failure
    (`centralized.py:537/544/552`), so after two SCA iterations the function
    hands back the Phase-I warm start **bit-identically**: LR-MMSE with a
    uniform `rho = 0.5`.

    That is exactly the `lr_mmse_fixed` baseline, which the manuscript plots
    as a *separate curve in the same figures*. The solver is honest about it
    (`feasible=False`), so the defect is not here but downstream, in
    `test_the_feasible_flag_is_dropped_before_aggregation`.
    """
    from cordis.algorithms.beamforming import design_phase_i
    topo, cfg, est, ss, assoc, sigma, Pmax = small_centralized_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W, res = run_centralized(topo, cfg, est, ss, assoc, sigma, Pmax)

    assert not res.feasible and not res.converged
    assert res.inner_failures == res.n_sca_iters

    warm = design_phase_i(topo, cfg, est, ss, assoc, comm_bf_method="lr_mmse"
                          ).build_W_tx_equal_psr(0.5, Pmax)
    for ap in topo.tx_aps:
        np.testing.assert_array_equal(W[ap.idx], warm[ap.idx])

    # It violates the SINR floor, and the power constraint is still met,
    # so nothing that only checks power would notice.
    assert _min_sinr_db(W, est, topo, sigma) < cfg.algorithm.gamma_db - 5.0
    for ap in topo.tx_aps:
        assert np.linalg.norm(W[ap.idx], "fro") ** 2 <= Pmax * (1 + 1e-6)


@pytest.mark.xfail(strict=True,
                   reason="F-07-02: AlgorithmResult records only `converged`; "
                          "`feasible` and `inner_failures` are computed per "
                          "trial, put on TrialOutput.extra, and then dropped at "
                          "aggregation, so no manifest can show whether the "
                          "centralized curve fell back to its warm start")
def test_the_feasible_flag_is_dropped_before_aggregation() -> None:
    """
    `_dispatch_centralized` (`scenario.py:463-470`) builds
    `extra = {converged, iters, feasible, inner_failures, solver, objective}`
    and `TrialOutput` carries it. But `AlgorithmResult`
    (`simulation/result.py:147-161`) declares only `iters`, `runtime_s`,
    `converged` and `power_ratios`, and `_aggregate` never reads `extra`.

    Consequence: a campaign in which the centralized benchmark silently
    reverted to an LR-MMSE `rho=0.5` warm start is indistinguishable, after
    the fact, from one in which it solved. The shipped campaigns cannot be
    audited for this.
    """
    import dataclasses
    from cordis.simulation.result import AlgorithmResult
    fields = {f.name for f in dataclasses.fields(AlgorithmResult)}
    assert "feasible" in fields or "inner_failures" in fields


# =============================================================================
# The ceiling property
# =============================================================================

def test_is_a_ceiling_over_cordis_split(centralized_scenario) -> None:
    """
    §V presents P-Global as the bound CORDIS-ADMM approaches. Verified here
    on the sensing utility both algorithms actually optimize.
    """
    from cordis.metrics.scnr import compute_sensing_surrogate
    topo, cfg, est, ss, assoc, sigma, Pmax = centralized_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W_c, _ = run_centralized(topo, cfg, est, ss, assoc, sigma, Pmax)
        W_s, _, _ = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)
    u_c = compute_sensing_surrogate(topo, cfg, W_c, ss, assoc)
    u_s = compute_sensing_surrogate(topo, cfg, W_s, ss, assoc)
    assert u_c >= u_s - 1e-9 * max(abs(u_c), 1.0)


# =============================================================================
# kappa
# =============================================================================

def test_kappa_moves_the_centralized_solution(centralized_scenario) -> None:
    """
    Unlike CORDIS-Split, which ignores `admm.kappa` entirely (F-06-02), the
    centralized solver does read it, so the kappa sweep of Fig. 4 is
    meaningful for this curve. The effect is small (about 1.4 dB of SCNR
    across `kappa` in [0, 2]) because of the auto-balancing described in
    F-07-03.
    """
    import copy
    from cordis.metrics.scnr import compute_scnr
    topo, cfg, est, ss, assoc, sigma, Pmax = centralized_scenario
    out = []
    for k in (0.0, 2.0):
        c = copy.deepcopy(cfg)
        c.algorithm.admm.kappa = k
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W, _ = run_centralized(topo, c, est, ss, assoc, sigma, Pmax)
        out.append(np.mean(list(
            compute_scnr(topo, c, W, ss, assoc, sigma).scnr_per_pair.values())))
    assert not np.isclose(out[0], out[1], rtol=1e-6)


# =============================================================================
# Dispatcher contract
# =============================================================================

def test_warm_start_psr_is_the_real_parameter_name() -> None:
    """
    `simulation/scenario.py:459` forwards a key named `warm_start`, but
    `solve_centralized` calls it `warm_start_psr` (F-07-04). The docstring
    of `experiments/specs.py:74` already records the discrepancy; this test
    pins which spelling the function actually has.
    """
    params = inspect.signature(solve_centralized).parameters
    assert "warm_start_psr" in params
    assert "warm_start" not in params


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="F-07-04: the dispatcher's fwd_keys contains "
                          "'warm_start', which run_centralized does not accept; "
                          "any spec that sets it raises TypeError at dispatch")
def test_dispatcher_key_warm_start_is_accepted(centralized_scenario) -> None:
    topo, cfg, est, ss, assoc, sigma, Pmax = centralized_scenario
    run_centralized(topo, cfg, est, ss, assoc, sigma, Pmax, warm_start=0.7)


@pytest.mark.parametrize("fn_name", ["run_centralized", "run_cordis_split"])
def test_a_scalar_gamma_is_rejected_rather_than_crashing(
        centralized_scenario, fn_name) -> None:
    """
    F-07-05. `specs.py:54` records that `gamma_u_db` "is required to be an
    array of length n_ue", and every experiment path builds one through
    `_gamma_vec`, so no shipped campaign hits this. But a scalar is accepted
    at the boundary and then fails deep inside the solver build with
    `IndexError: invalid index to scalar variable` (`centralized.py:464`),
    because `db2lin(np.asarray(5.0))` is 0-d and `gamma_lin[u]` indexes it.

    A `TypeError`/`ValueError` at the entry point would be the right
    behavior. This test documents the current behavior instead of asserting
    the desired one, so it stays green until F-07-05 is corrected.
    """
    fn = {"run_centralized": run_centralized,
          "run_cordis_split": run_cordis_split}[fn_name]
    topo, cfg, est, ss, assoc, sigma, Pmax = centralized_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((IndexError, TypeError, ValueError)):
            fn(topo, cfg, est, ss, assoc, sigma, Pmax, gamma_u_db=5.0)


def test_warm_start_psr_does_not_change_the_converged_answer(
        centralized_scenario) -> None:
    """
    SCA from different Phase-I initializations should land in the same
    neighborhood. A large spread would mean the benchmark's "ceiling" is an
    artifact of where it started.
    """
    topo, cfg, est, ss, assoc, sigma, Pmax = centralized_scenario
    got = []
    for w in (0.3, 0.5, 0.7):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W, _ = run_centralized(topo, cfg, est, ss, assoc, sigma, Pmax,
                                   warm_start_psr=w)
        got.append(_min_sinr_db(W, est, topo, sigma))
    assert max(got) - min(got) < 1.0, f"warm start changes min SINR: {got}"
