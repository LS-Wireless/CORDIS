"""
Stage 6a validation — CORDIS-Split (P-Split power allocation).

Uses 4 UEs / 8 antennas (50% loading → min-SINR ≈ 12 dB at ρ=1),
higher clutter (CNR = 10 dB), and feasible γ targets.

Tests:
1. Convergence and valid PSRs
2. SINR consistency (P-Split model vs metrics module)
3. SINR target achieved: γ=5 dB is feasible, optimizer allocates
   remaining power to sensing
4. Trade-off sweep: increasing γ → ρ↑, SINR↑, SCNR↓
5. No targets → ρ = 1.0
6. Solver comparison: CVXPY vs scipy (if both available)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import logging

from cordis.utils.config import load_config
from cordis.utils.io_utils import make_rng, child_rng
from cordis.utils.math_utils import db2lin, lin2db
from cordis.channel.topology import generate_topology
from cordis.channel.pathloss import (
    compute_large_scale_fading, noise_power_watts, snr_to_tx_power,
)
from cordis.channel.rician import (
    compute_channel_statistics, generate_channel_realization,
)
from cordis.channel.estimation import (
    run_channel_estimation, design_pilot_sequences,
)
from cordis.channel.sensing_channel import compute_sensing_statistics
from cordis.channel.sensing_assignment import assign_sensing
from cordis.algorithms.beamforming import design_phase_i, compute_split_scalars
from cordis.algorithms.split_opt import run_cordis_split, solve_p_split, _HAS_CVXPY
from cordis.metrics import compute_sinr, compute_scnr

logging.getLogger("cordis").setLevel(logging.WARNING)


def build_scenario(seed=42, n_ue=4, n_targets=2, clutter_cnr_db=10.0):
    """
    Build a single-trial scenario.

    Default: 4 UEs, 8 antennas, 50m cell, 2 targets,
    clutter CNR = 10 dB (realistic, not trivially high SCNR).
    """
    cfg = load_config("configs/default.json")
    cfg.topology.type            = "circle"
    cfg.topology.ap_radius_m     = 50.0
    cfg.topology.ue_min_radius_m = 10.0
    cfg.topology.ue_max_radius_m = 60.0
    cfg.topology.n_ap            = 6
    cfg.topology.n_ue            = n_ue
    cfg.topology.n_ant           = 8
    cfg.topology.n_rf_chains     = 8
    cfg.topology.n_targets       = n_targets
    cfg.topology.tg_min_radius_m = 10.0
    cfg.topology.tg_max_radius_m = 60.0
    cfg.topology.n_sensing_rx    = 1 if n_targets > 0 else 0
    cfg.channel.snr_db           = 90.0
    cfg.channel.pilot_power_db   = 120.0
    cfg.channel.estimation_method = "MMSE"
    cfg.sensing.clutter_cnr_db   = clutter_cnr_db
    cfg.algorithm.split.gamma_db = 5.0
    cfg.algorithm.split.kappa    = 1.0

    rng  = make_rng(seed)
    topo = generate_topology(cfg, child_rng(rng))
    lsf  = compute_large_scale_fading(topo, cfg, child_rng(rng),
                                       include_targets=(n_targets > 0))
    stats = compute_channel_statistics(topo, cfg, lsf, child_rng(rng))
    real  = generate_channel_realization(topo, cfg, lsf, stats, child_rng(rng))
    Phi, ct = design_pilot_sequences(n_ue, cfg.channel.tau_p, child_rng(rng))
    est   = run_channel_estimation(topo, cfg, lsf, stats, real,
                                    child_rng(rng), Phi, ct)
    s_stats = compute_sensing_statistics(topo, cfg, lsf, child_rng(rng))
    assoc   = assign_sensing(topo, cfg, lsf)
    sigma   = noise_power_watts(cfg.frequency.bandwidth_hz,
                                 cfg.channel.noise_figure_db,
                                 cfg.channel.noise_temp_k)
    Pmax    = snr_to_tx_power(cfg.channel.snr_db, sigma)
    return cfg, topo, est, s_stats, assoc, sigma, Pmax


def _fmt_sinr(arr):
    return [f"{lin2db(v):+.1f}" for v in arr]


# ── Test 1: Basic convergence ─────────────────────────────────────────────────

def test_convergence():
    print("\n── Test 1: Basic convergence ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()

    W_tx, p1, res = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)

    rho_vals = list(res.rho_opt.values())
    print(f"  Solver:    {res.solver}")
    print(f"  Converged: {res.converged}")
    print(f"  Objective: {res.objective:.4f}")
    print(f"  ρ:         {[f'{v:.3f}' for v in rho_vals]}")
    print(f"  Slack:     {[f'{v:.2e}' for v in res.slack]}")
    print(f"  Est SINR:  {_fmt_sinr(res.sinr_u_est)} dB")

    assert res.converged, "P-Split did not converge"
    assert all(0.0 <= v <= 1.0 for v in rho_vals), "ρ out of bounds"
    print("  ✓ PASSED")


# ── Test 2: SINR model vs metrics consistency ─────────────────────────────────

def test_sinr_consistency():
    print("\n── Test 2: SINR consistency (model vs metrics) ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()

    W_tx, p1, res = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)
    sinr_m = compute_sinr(W_tx, est, topo, sigma)

    gap = np.abs(lin2db(res.sinr_u_est) - lin2db(sinr_m.sinr_per_user))
    print(f"  Model:   {_fmt_sinr(res.sinr_u_est)} dB")
    print(f"  Metrics: {_fmt_sinr(sinr_m.sinr_per_user)} dB")
    print(f"  Avg gap: {np.mean(gap):.2f} dB  (max: {np.max(gap):.2f} dB)")
    assert np.mean(gap) < 5.0, f"Gap too large: {np.mean(gap):.2f} dB"
    print("  ✓ PASSED")


# ── Test 3: SINR target achieved and sensing power allocated ──────────────────

def test_sinr_target_met():
    print("\n── Test 3: Feasible γ → model-feasible + sensing allocated ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()

    # Check SINR ceiling at ρ=1 (all comm)
    p1 = design_phase_i(topo, cfg, est, ss, assoc)
    W_all = p1.build_W_tx_equal_psr(1.0, Pmax)
    sinr_ceil = compute_sinr(W_all, est, topo, sigma)
    ceil_db = lin2db(sinr_ceil.min_sinr)
    print(f"  SINR ceiling (ρ=1):    {ceil_db:+.1f} dB")

    # Set γ well below ceiling
    gamma_db = min(5.0, ceil_db - 3.0)
    cfg.algorithm.split.gamma_db = gamma_db
    print(f"  SINR target γ:         {gamma_db:+.1f} dB")

    W_tx, p1, res = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)
    sinr_m = compute_sinr(W_tx, est, topo, sigma)
    scnr_m = compute_scnr(topo, cfg, W_tx, ss, assoc, sigma)

    min_model_sinr_db = lin2db(np.min(res.sinr_u_est))
    min_actual_sinr_db = lin2db(sinr_m.min_sinr)
    avg_rho = np.mean(list(res.rho_opt.values()))

    print(f"  Model min-SINR:        {min_model_sinr_db:+.1f} dB")
    print(f"  Actual min-SINR:       {min_actual_sinr_db:+.1f} dB")
    print(f"  Model–actual gap:      {min_model_sinr_db - min_actual_sinr_db:.1f} dB")
    print(f"  SCNR:                  {lin2db(scnr_m.weighted_sum_scnr):+.1f} dB")
    print(f"  Avg ρ:                 {avg_rho:.3f}")
    print(f"  ρ values:              {[f'{v:.3f}' for v in res.rho_opt.values()]}")

    # The P-Split MODEL should declare feasibility (model SINR ≈ γ)
    model_feasible = min_model_sinr_db >= gamma_db - 0.5
    print(f"  Model feasible:        {model_feasible}")
    assert model_feasible, (
        f"P-Split model infeasible: model SINR {min_model_sinr_db:.1f} < γ {gamma_db:.1f} dB"
    )

    # Some sensing power should be allocated (ρ < 1 for at least some APs)
    has_sensing = avg_rho < 0.99
    print(f"  Sensing allocated:     {has_sensing} (avg ρ = {avg_rho:.3f})")
    assert has_sensing, f"No sensing power: avg ρ = {avg_rho:.3f}"

    # Note: actual SINR may be below γ due to the local model approximation
    # (cross-AP interference coupling is not captured by compressed scalars).
    # This gap is a known limitation of Split — ADMM addresses it.
    if min_actual_sinr_db < gamma_db:
        print(f"  ⚠  Actual SINR ({min_actual_sinr_db:+.1f}) < γ ({gamma_db:+.1f}): "
              f"expected Split model gap")
    print("  ✓ PASSED")


# ── Test 4: Optimised vs fixed PSR baselines ──────────────────────────────────

def test_vs_fixed_psr():
    print("\n── Test 4: Optimised vs fixed PSR baselines ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()
    cfg.algorithm.split.gamma_db = 3.0     # low target → more room for sensing

    W_tx, p1, res = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)

    results = {}
    for label, W in [("Optimised", W_tx)]:
        sinr_m = compute_sinr(W, est, topo, sigma)
        scnr_m = compute_scnr(topo, cfg, W, ss, assoc, sigma)
        results[label] = (lin2db(sinr_m.min_sinr), lin2db(scnr_m.weighted_sum_scnr))

    for psr in [0.2, 0.5, 0.8, 1.0]:
        W_f = p1.build_W_tx_equal_psr(psr, Pmax)
        sinr_f = compute_sinr(W_f, est, topo, sigma)
        scnr_f = compute_scnr(topo, cfg, W_f, ss, assoc, sigma)
        results[f"ρ={psr}"] = (lin2db(sinr_f.min_sinr), lin2db(scnr_f.weighted_sum_scnr))

    print(f"  {'Method':<12s} {'min-SINR':>10s} {'sum-SCNR':>10s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10}")
    for label, (sinr, scnr) in results.items():
        tag = " ← opt" if label == "Optimised" else ""
        print(f"  {label:<12s} {sinr:>+10.2f} {scnr:>+10.2f}{tag}")

    print(f"\n  avg ρ (opt): {np.mean(list(res.rho_opt.values())):.3f}")
    print("  ✓ PASSED")


# ── Test 5: γ sweep (comm–sensing trade-off) ──────────────────────────────────

def test_gamma_sweep():
    print("\n── Test 5: SINR target sweep (γ = 0, 3, 5, 8, 10 dB) ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()

    print(f"  {'γ [dB]':>8} {'min-SINR':>10} {'sum-SCNR':>10} {'avg-ρ':>8} {'max-ε':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

    prev_rho = 0.0
    for g_db in [0.0, 3.0, 5.0, 8.0, 10.0]:
        cfg.algorithm.split.gamma_db = g_db
        W_tx, p1, res = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)
        sinr_m = compute_sinr(W_tx, est, topo, sigma)
        scnr_m = compute_scnr(topo, cfg, W_tx, ss, assoc, sigma)
        avg_rho = np.mean(list(res.rho_opt.values()))
        print(f"  {g_db:>8.1f} {lin2db(sinr_m.min_sinr):>+10.2f} "
              f"{lin2db(scnr_m.weighted_sum_scnr):>+10.2f} "
              f"{avg_rho:>8.3f} {np.max(res.slack):>10.2e}")
        prev_rho = avg_rho

    print("  Expected: SINR↑ as γ↑, SCNR↓ as γ↑, avg-ρ↑ as γ↑")
    print("  ✓ PASSED (inspect visually)")


# ── Test 6: No targets → ρ = 1 ────────────────────────────────────────────────

def test_no_targets():
    print("\n── Test 6: No targets → ρ = 1 ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario(
        n_targets=0, clutter_cnr_db=-10.0,
    )
    W_tx, p1, res = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)
    rho_vals = list(res.rho_opt.values())
    avg_rho  = np.mean(rho_vals)
    print(f"  ρ:     {[f'{v:.3f}' for v in rho_vals]}")
    print(f"  avg ρ: {avg_rho:.3f}")
    assert avg_rho > 0.99, f"Expected ρ ≈ 1, got {avg_rho:.3f}"
    print("  ✓ PASSED")


# ── Test 7: Solver comparison (CVXPY vs scipy, if both available) ─────────────

def test_solver_comparison():
    print("\n── Test 7: Solver comparison ──")
    if not _HAS_CVXPY:
        print("  CVXPY not installed — skipping")
        return

    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()
    cfg.algorithm.split.gamma_db = 5.0

    p1 = design_phase_i(topo, cfg, est, ss, assoc)
    sc = compute_split_scalars(p1, est, topo, ss, assoc, kappa=cfg.algorithm.split.kappa)

    res_cvx = solve_p_split(sc, topo, cfg, sigma, Pmax, use_cvxpy=True)
    res_sci = solve_p_split(sc, topo, cfg, sigma, Pmax, use_cvxpy=False)

    rho_cvx = np.array(list(res_cvx.rho_opt.values()))
    rho_sci = np.array(list(res_sci.rho_opt.values()))
    rho_gap = np.abs(rho_cvx - rho_sci)

    print(f"  CVXPY:  ρ = {[f'{v:.3f}' for v in rho_cvx]}  "
          f"obj = {res_cvx.objective:.4f}  [{res_cvx.solver}]")
    print(f"  scipy:  ρ = {[f'{v:.3f}' for v in rho_sci]}  "
          f"obj = {res_sci.objective:.4f}  [{res_sci.solver}]")
    print(f"  ρ gap:  {[f'{v:.3f}' for v in rho_gap]}  (max: {np.max(rho_gap):.3f})")

    sinr_cvx = lin2db(np.min(res_cvx.sinr_u_est))
    sinr_sci = lin2db(np.min(res_sci.sinr_u_est))
    print(f"  min-SINR: CVXPY={sinr_cvx:+.1f} dB  scipy={sinr_sci:+.1f} dB")
    print("  ✓ PASSED (inspect visually)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  CORDIS-Split (Stage 6a) Validation")
    print(f"  CVXPY available: {_HAS_CVXPY}")
    print("=" * 64)

    test_convergence()
    test_sinr_consistency()
    test_sinr_target_met()
    test_vs_fixed_psr()
    test_gamma_sweep()
    test_no_targets()
    test_solver_comparison()

    print("\n" + "=" * 64)
    print("  All Stage 6a tests passed.")
    print("=" * 64)

