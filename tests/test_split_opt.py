"""
tests/test_split_opt.py
=======================
CORDIS-Split, Algorithm 1 (`cordis/algorithms/split_opt.py`).

Paper reference: §IV, `alg:cordis-split`, `eq:pa-comm-utility`, and
`eq:split-pa-opt{,-obj,-const-sinr,-const-box}`.

P-Split maximizes `c^T rho - xi 1^T eps` over the per-AP power-splitting ratios
subject to `S_u(rho) + eps_u >= gamma_u D_u(rho)` and `rho in [0, 1]`. The paper
argues the problem is strictly convex because `S_u` expands into affine terms
plus concave geometric means `sqrt(rho_i rho_j)`, which only holds when
`beta_hat_au` is real and positive. That chain is tested end to end here.

Solver-backed tests are marked `solver`; the full-pipeline ones are also `slow`.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from cordis.algorithms.beamforming import compute_split_scalars, design_phase_i
from cordis.algorithms.split_opt import run_cordis_split, solve_p_split
from cordis.channel.estimation import design_pilot_sequences, run_channel_estimation
from cordis.channel.pathloss import (
    compute_large_scale_fading, noise_power_watts, snr_to_tx_power,
)
from cordis.channel.rician import (
    compute_channel_statistics, generate_channel_realization,
)
from cordis.channel.sensing_assignment import assign_sensing
from cordis.channel.sensing_channel import compute_sensing_statistics
from cordis.channel.topology import generate_topology
from cordis.metrics.scnr import compute_scnr
from cordis.metrics.sinr import compute_sinr
from cordis.utils.io_utils import child_rng, make_rng

from conftest import skip_if_no_cvxpy


# =============================================================================
# Scenario
# =============================================================================

def _small_cfg(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.n_ap = 4
    cfg.topology.n_ant = cfg.topology.n_rf_chains = 8
    cfg.topology.n_ue = 3
    cfg.topology.n_targets = 1
    cfg.topology.n_sensing_rx = 1
    cfg.channel.n_spatial_samples = 800
    cfg.validate()
    return cfg


def _scenario(cfg, seed: int = 808):
    r = make_rng(seed)
    topo = generate_topology(cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
    st = compute_channel_statistics(topo, cfg, lsf, child_rng(r))
    real = generate_channel_realization(topo, cfg, lsf, st, child_rng(r))
    Phi, cs = design_pilot_sequences(topo.n_ue, cfg.channel.tau_p, child_rng(r))
    est = run_channel_estimation(topo, cfg, lsf, st, real, child_rng(r), Phi, cs)
    ss = compute_sensing_statistics(topo, cfg, lsf, child_rng(r))
    assoc = assign_sensing(topo, cfg, lsf)
    sigma = noise_power_watts(cfg.frequency.bandwidth_hz,
                              cfg.channel.noise_figure_db,
                              cfg.channel.noise_temp_k)
    return topo, est, ss, assoc, sigma, snr_to_tx_power(cfg.channel.snr_db, sigma)


@pytest.fixture(scope="module")
def split_run(request):
    """One full CORDIS-Split solve, reused across tests."""
    skip_if_no_cvxpy()
    from cordis.utils.config import load_config
    cfg = _small_cfg(load_config("configs/default.json"))
    topo, est, ss, assoc, sigma, pmax = _scenario(cfg)
    W_tx, phase_i, res = run_cordis_split(topo, cfg, est, ss, assoc, sigma, pmax)
    return cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res


def _rho(res):
    return np.array([res.rho_opt[a] for a in sorted(res.rho_opt)])


# =============================================================================
# Phase I outputs
# =============================================================================

@pytest.mark.solver
@pytest.mark.slow
def test_phase_i_beamformers_are_normalized(split_run) -> None:
    """
    Phase I returns unit-norm `W_comm_hat` and `W_sens_hat`; the power split
    `sqrt(rho Pmax)` / `sqrt((1-rho) Pmax)` is applied afterwards (§IV-A).
    """
    cfg, topo, *_rest, phase_i, res = split_run
    for a in phase_i.W_comm_hat:
        assert np.linalg.norm(phase_i.W_comm_hat[a]) == pytest.approx(1.0)
        assert np.linalg.norm(phase_i.W_sens_hat[a]) == pytest.approx(1.0)


@pytest.mark.solver
@pytest.mark.slow
def test_exchanged_scalars_have_the_documented_shapes(split_run) -> None:
    """
    Table IV's payload: three per-user vectors plus two per-AP scalars, with the
    returned `rho*_at` making the third per-AP scalar. `3 N_ue + 3 = 12` here.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    sc = compute_split_scalars(phase_i, est, topo, ss, assoc,
                               kappa=cfg.algorithm.split.kappa)
    n_ue = topo.n_ue
    for a in [ap.idx for ap in topo.tx_aps]:
        assert sc.beta_hat[a].shape == (n_ue,)
        assert sc.g_tilde[a].shape == (n_ue,)
        assert sc.e_sens[a].shape == (n_ue,)
        assert np.isscalar(sc.z[a]) or np.ndim(sc.z[a]) == 0
        assert np.isscalar(sc.q_tilde[a]) or np.ndim(sc.q_tilde[a]) == 0
    per_ap = 3 * n_ue + 3
    assert per_ap == 3 * n_ue + 3


@pytest.mark.solver
@pytest.mark.slow
def test_beta_hat_is_real_and_positive(split_run) -> None:
    """
    The convexity of P-Split rests on this. `beta_hat_au = h_u^H w_u^(c)` must
    be real-positive so that `S_u(rho)` is a sum of affine terms and concave
    geometric means; a negative or complex `beta_hat` would break the DCP form
    that `_solve_p_split_cvxpy` builds.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    sc = compute_split_scalars(phase_i, est, topo, ss, assoc,
                               kappa=cfg.algorithm.split.kappa)
    for a, b in sc.beta_hat.items():
        assert np.all(np.real(b) > 0)
        assert np.allclose(np.imag(b), 0.0, atol=1e-9 * np.abs(b).max())


@pytest.mark.solver
@pytest.mark.slow
def test_csi_error_scalars_are_non_negative(split_run) -> None:
    """`g_tilde` and `e_sens` are powers, so they cannot be negative."""
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    sc = compute_split_scalars(phase_i, est, topo, ss, assoc,
                               kappa=cfg.algorithm.split.kappa)
    for a in sc.g_tilde:
        assert np.all(sc.g_tilde[a] >= 0.0)
        assert np.all(sc.e_sens[a] >= 0.0)


# =============================================================================
# Phase II: P-Split
# =============================================================================

@pytest.mark.solver
@pytest.mark.slow
def test_psr_stays_in_the_box(split_run) -> None:
    """`eq:split-pa-opt-const-box`: `0 <= rho_at <= 1` for every AP."""
    *_rest, res = split_run
    rho = _rho(res)
    assert np.all(rho >= -1e-9) and np.all(rho <= 1.0 + 1e-9)


@pytest.mark.solver
@pytest.mark.slow
def test_power_budget_is_met_exactly(split_run) -> None:
    """
    `||W_at||_F^2 = Pmax` for every AP: the split allocates `rho Pmax` to
    communication and `(1-rho) Pmax` to sensing, so the budget binds.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    for a, W in W_tx.items():
        assert np.linalg.norm(W) ** 2 / pmax == pytest.approx(1.0, rel=1e-9)


@pytest.mark.solver
@pytest.mark.slow
def test_the_model_sinr_matches_the_realized_sinr(split_run) -> None:
    """
    `eq:pa-comm-utility` is a compressed surrogate built from five scalars per
    (AP, user). It must reproduce the SINR that `eq:sinr` computes from the full
    beamformers, otherwise the CPU is optimizing the wrong quantity.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    realized = compute_sinr(W_tx, est, topo, sigma).sinr_per_user
    model = res.sinr_u_est
    r_db = 10 * np.log10(np.maximum(realized, 1e-30))
    m_db = 10 * np.log10(np.maximum(model, 1e-30))
    np.testing.assert_allclose(m_db, r_db, atol=0.25)


@pytest.mark.solver
@pytest.mark.slow
def test_the_qos_floor_is_met_when_the_slack_is_zero(split_run) -> None:
    """
    Zero slack means `eq:split-pa-opt-const-sinr` holds with the true `gamma`,
    so every user should clear the floor in the realized SINR too.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    if np.sum(res.slack) > 1e-9:
        pytest.skip("this drop needed slack; the floor is not claimed to hold")
    realized_db = 10 * np.log10(np.maximum(
        compute_sinr(W_tx, est, topo, sigma).sinr_per_user, 1e-30))
    assert np.all(realized_db >= cfg.algorithm.gamma_db - 0.5)


@pytest.mark.solver
@pytest.mark.slow
def test_cvxpy_and_scipy_agree(split_run) -> None:
    """
    The interior-point and trust-region paths solve the same problem. A
    disagreement would mean one of the two formulations is not the one §IV-B
    describes.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    sc = compute_split_scalars(phase_i, est, topo, ss, assoc,
                               kappa=cfg.algorithm.split.kappa)
    a = solve_p_split(sc, topo, cfg, sigma, pmax, use_cvxpy=True)
    b = solve_p_split(sc, topo, cfg, sigma, pmax, use_cvxpy=False)
    assert a.converged and b.converged
    np.testing.assert_allclose(_rho(a), _rho(b), atol=5e-4)
    assert a.objective == pytest.approx(b.objective, rel=1e-3)


@pytest.mark.solver
@pytest.mark.slow
def test_a_harder_floor_pushes_power_toward_communication(split_run) -> None:
    """
    Raising `gamma` must not lower the communication share on average: the QoS
    constraint tightens and `rho` can only move toward 1 or pick up slack.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    sc = compute_split_scalars(phase_i, est, topo, ss, assoc,
                               kappa=cfg.algorithm.split.kappa)
    lo = solve_p_split(sc, topo, cfg, sigma, pmax,
                       gamma_u_db=np.full(topo.n_ue, -10.0))
    hi = solve_p_split(sc, topo, cfg, sigma, pmax,
                       gamma_u_db=np.full(topo.n_ue, 20.0))
    assert _rho(hi).mean() >= _rho(lo).mean() - 1e-6


@pytest.mark.solver
@pytest.mark.slow
@pytest.mark.parametrize("gamma_db", [10.0, 20.0, 40.0])
def test_an_infeasible_floor_is_absorbed_by_the_slack(split_run, gamma_db) -> None:
    """
    `xi_penalty` keeps P-Split feasible: a floor the network cannot meet
    produces positive slack rather than a solver failure (§IV-B).

    Verified up to `gamma = 40` dB, well past the `{0 ... 15}` dB sweep of
    Fig. 2. Beyond about 60 dB the solve itself fails; see the next test.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    sc = compute_split_scalars(phase_i, est, topo, ss, assoc,
                               kappa=cfg.algorithm.split.kappa)
    out = solve_p_split(sc, topo, cfg, sigma, pmax,
                        gamma_u_db=np.full(topo.n_ue, gamma_db))
    assert out.converged
    assert np.sum(out.slack) > 0.0
    assert np.all(_rho(out) >= -1e-9) and np.all(_rho(out) <= 1.0 + 1e-9)


@pytest.mark.solver
@pytest.mark.slow
def test_solver_failure_returns_an_equal_split_with_zero_slack(split_run) -> None:
    """
    Pins F-06-03. When the CVXPY solve fails, `_solve_p_split_cvxpy` returns
    `rho = 0.5` for every AP, `objective = 0`, and **`slack = 0`**.

    Zero slack is the signal that the QoS floor was met, so a failed solve
    reports "no QoS violation" while having solved nothing. The fallback only
    triggers at extreme targets (about 60 dB and above) and never fires at the
    shipped `gamma = 5` dB across 25 random drops, so no published number is
    affected; it is a latent misreport.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    sc = compute_split_scalars(phase_i, est, topo, ss, assoc,
                               kappa=cfg.algorithm.split.kappa)
    out = solve_p_split(sc, topo, cfg, sigma, pmax,
                        gamma_u_db=np.full(topo.n_ue, 80.0))
    assert not out.converged
    np.testing.assert_allclose(_rho(out), 0.5)
    assert float(np.sum(out.slack)) == 0.0
    assert out.objective == 0.0


# =============================================================================
# The two kappa fields (F-06-02)
# =============================================================================

@pytest.mark.solver
@pytest.mark.slow
def test_split_ignores_the_admm_kappa(split_run) -> None:
    """
    Pins F-06-02, the mechanism behind Fig. 4's flat CORDIS-Split curve.

    `run_kappa_sweep` varies `cfg.algorithm.admm.kappa`. CORDIS-Split reads
    `cfg.algorithm.split.kappa`, a different field pinned at 1.0, so its
    solution does not move at all across the sweep: measured
    `max|d rho| = 0` and an identical SCNR to four decimals.

    §VI-D attributes the flat curve to the fixed null-space-projected sensing
    beam being unable to react to a CPU-level penalty. That explanation is not
    what is happening here.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, *_rest = split_run
    ref = None
    for k in (0.0, 0.08, 1.0, 2.0):
        c = copy.deepcopy(cfg)
        c.algorithm.admm.kappa = k
        W, _, r = run_cordis_split(topo, c, est, ss, assoc, sigma, pmax)
        rho = _rho(r)
        if ref is None:
            ref = rho
        np.testing.assert_allclose(rho, ref, atol=1e-12)


@pytest.mark.solver
@pytest.mark.slow
def test_split_does_respond_to_its_own_kappa(split_run) -> None:
    """
    The counterpart: CORDIS-Split is **not** structurally blind to a clutter
    penalty. Varying `cfg.algorithm.split.kappa` moves `rho` substantially, so
    the flatness in Fig. 4 is about which variable is swept, not about the
    algorithm's structure.
    """
    cfg, topo, est, ss, assoc, sigma, pmax, *_rest = split_run
    sols = {}
    for k in (0.0, 1.0):
        c = copy.deepcopy(cfg)
        c.algorithm.split.kappa = k
        W, _, r = run_cordis_split(topo, c, est, ss, assoc, sigma, pmax)
        sols[k] = _rho(r)
    assert np.max(np.abs(sols[0.0] - sols[1.0])) > 0.1


@pytest.mark.solver
@pytest.mark.slow
def test_the_two_kappa_fields_differ_in_the_shipped_config(default_cfg) -> None:
    """`split.kappa = 1.0` while `admm.kappa = 0.08`; neither is in Table III."""
    assert default_cfg.algorithm.split.kappa == pytest.approx(1.0)
    assert default_cfg.algorithm.admm.kappa == pytest.approx(0.08)


# =============================================================================
# Determinism
# =============================================================================

@pytest.mark.solver
@pytest.mark.slow
def test_split_is_reproducible(split_run) -> None:
    cfg, topo, est, ss, assoc, sigma, pmax, W_tx, phase_i, res = split_run
    W2, _, res2 = run_cordis_split(topo, cfg, est, ss, assoc, sigma, pmax)
    np.testing.assert_allclose(_rho(res2), _rho(res), atol=1e-9)
    for a in W_tx:
        np.testing.assert_allclose(W2[a], W_tx[a], atol=1e-9)
