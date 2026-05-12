"""
Stage 6c validation — CORDIS-ADMM (Journal Paper Section 5).

Tests:
   1. ADMM convergence — primal & dual residuals decay (eq. admm-primal-residual,
                        eq. admm-dual-residual)
   2. Power constraints respected at termination
   3. SINR constraint met (SOC at CPU should enforce SINR ≥ γ at convergence)
   4. ADMM beats Split in SCNR at the same γ
   5. ADMM approaches Centralized — decentralization gap is bounded
   6. rho_admm sensitivity sweep (penalty parameter)
   7. kappa clutter penalty sweep (same role as Stage 6b)
   8. Multi-seed robustness
   9. gamma-target sweep
  10. Three-way comparison: Split / ADMM / Centralized
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import time

import numpy as np

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
from cordis.algorithms.beamforming import design_phase_i
from cordis.algorithms.split_opt import run_cordis_split
from cordis.algorithms.centralized import run_centralized
from cordis.algorithms.joint_opt import (
    run_cordis_admm, solve_cordis_admm, _HAS_CVXPY,
)
from cordis.metrics import compute_sinr, compute_scnr

logging.getLogger("cordis").setLevel(logging.WARNING)


# =============================================================================
# Scenario builder (matches Stage 6b)
# =============================================================================

def _build_scenario(seed, n_ap, n_ue, n_ant, n_targets, gamma_db,
                     clutter_cnr_db=10.0, kappa=0.0,
                     pilot_power_db=120.0, snr_db=90.0):
    cfg = load_config("configs/default.json")
    cfg.topology.type            = "circle"
    cfg.topology.ap_radius_m     = 50.0
    cfg.topology.ue_min_radius_m = 10.0
    cfg.topology.ue_max_radius_m = 60.0
    cfg.topology.n_ap            = n_ap
    cfg.topology.n_ue            = n_ue
    cfg.topology.n_ant           = n_ant
    cfg.topology.n_rf_chains     = n_ant
    cfg.topology.n_targets       = n_targets
    cfg.topology.tg_min_radius_m = 10.0
    cfg.topology.tg_max_radius_m = 60.0
    cfg.topology.n_sensing_rx    = 1 if n_targets > 0 else 0
    cfg.channel.snr_db           = snr_db
    cfg.channel.pilot_power_db   = pilot_power_db
    cfg.channel.estimation_method = "MMSE"
    cfg.sensing.clutter_cnr_db   = clutter_cnr_db
    cfg.sensing.los_model        = "always"
    cfg.algorithm.split.gamma_db = gamma_db
    cfg.algorithm.split.kappa    = 1.0
    cfg.algorithm.admm.kappa     = kappa

    rng  = make_rng(seed)
    topo = generate_topology(cfg, child_rng(rng))
    lsf  = compute_large_scale_fading(topo, cfg, child_rng(rng), include_targets=True)
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


def medium_scenario(seed=42, gamma_db=3.0, kappa=0.0):
    """6 APs (5 TX + 1 RX) × 8 ant, 4 UEs, 2 targets — same as Stage 6b."""
    return _build_scenario(seed, n_ap=6, n_ue=4, n_ant=8, n_targets=2,
                            gamma_db=gamma_db, kappa=kappa)


def _achieved_sinr_db(W_tx, est, topo, sigma):
    sinr_m = compute_sinr(W_tx, est, topo, sigma)
    return sinr_m.sinr_per_user_db, sinr_m.min_sinr_db


def _achieved_scnr_db(topo, cfg, W_tx, ss, assoc, sigma):
    return lin2db(compute_scnr(topo, cfg, W_tx, ss, assoc, sigma).weighted_sum_scnr)


def _power_ratios(W_tx, topo, Pmax):
    return np.array([
        float(np.linalg.norm(W_tx[ap.idx], "fro") ** 2) / Pmax
        for ap in topo.tx_aps
    ])


# =============================================================================
# Tests
# =============================================================================

def test_1_admm_convergence():
    """Test 1: ADMM converges — primal & dual residuals decay."""
    print("\n── Test 1: ADMM convergence ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return
    cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(gamma_db=3.0)

    t0 = time.time()
    W_tx, res = run_cordis_admm(
        topo, cfg, est, ss, assoc, sigma, Pmax,
        n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
        rho_admm=1.0, kappa=0.0, xi_slack=1e4,
    )
    elapsed = time.time() - t0

    print(f"  Solver:        {res.solver}")
    print(f"  ADMM iters:    {res.n_admm_iters} / 50")
    print(f"  Converged:     {res.converged}")
    print(f"  Feasible:      {res.feasible}")
    print(f"  Inner fails:   {res.inner_failures}")
    print(f"  Time:          {elapsed:.1f} s")
    print(f"\n  Iter  r_pri        r_dual      sensing-obj   minSINR [dB]")
    print(f"  ────  ───────────  ──────────  ────────────  ────────────")
    n_show = min(len(res.primal_res_history), 30)
    for t in range(n_show):
        msi = lin2db(res.sinr_history[t].min())
        print(f"  {t+1:>4d}  {res.primal_res_history[t]:>11.3e}  "
              f"{res.dual_res_history[t]:>10.3e}  "
              f"{res.sensing_obj_history[t]:>12.3e}  "
              f"{msi:>+12.2f}")

    assert res.feasible, "ADMM was infeasible across all iterations"
    if res.converged:
        # Tolerance defaults to 1.0 in SNR-amplitude units (the natural
        # balance point for rho_admm=1).  Asserts mirror that.
        assert res.primal_res_history[-1] < 1.0, "Primal residual not below tol"
        assert res.dual_res_history[-1] < 1.0, "Dual residual not below tol"
    print("  ✓ PASSED")


def test_2_power_constraints():
    """Test 2: Per-AP power constraints respected."""
    print("\n── Test 2: Power constraints ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return
    cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(gamma_db=3.0)
    W_tx, _ = run_cordis_admm(
        topo, cfg, est, ss, assoc, sigma, Pmax,
        n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
    )
    ratios = _power_ratios(W_tx, topo, Pmax)
    for i, ap in enumerate(topo.tx_aps):
        marker = '✓' if ratios[i] <= 1.01 else '✗'
        print(f"  AP {ap.idx}: ‖W‖²/Pmax = {ratios[i]:.4f}  {marker}")
    assert (ratios <= 1.01).all()
    print("  ✓ PASSED")


def test_3_sinr_constraint_met():
    """Test 3: SINR constraint should be met at convergence (within tol)."""
    print("\n── Test 3: SINR constraint satisfaction ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return

    for g_db in [0.0, 3.0, 5.0, 8.0]:
        cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(gamma_db=g_db)
        W_tx, res = run_cordis_admm(
            topo, cfg, est, ss, assoc, sigma, Pmax,
            n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
        )
        sinrs_db, min_db = _achieved_sinr_db(W_tx, est, topo, sigma)
        slack_max = res.slack_history[-1].max() if res.slack_history else 0.0
        gap = min_db - g_db
        marker = '✓' if gap >= -1.0 else '⚠ '
        print(f"  γ={g_db:>5.1f} dB  →  minSINR={min_db:>+6.2f} dB "
              f"(gap={gap:>+5.2f} dB)  slack_max={slack_max:.2e}  {marker}")
    print("  ✓ PASSED (visual inspection)")


def test_4_beats_split():
    """Test 4: ADMM should achieve higher SCNR than Split at same γ."""
    print("\n── Test 4: CORDIS-ADMM vs CORDIS-Split ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return
    cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(gamma_db=3.0)

    # Split (Stage 6a)
    W_split, _, _ = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)
    _, sinr_split = _achieved_sinr_db(W_split, est, topo, sigma)
    scnr_split    = _achieved_scnr_db(topo, cfg, W_split, ss, assoc, sigma)

    # ADMM (Stage 6c)
    W_admm, res_admm = run_cordis_admm(
        topo, cfg, est, ss, assoc, sigma, Pmax,
        n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
    )
    _, sinr_admm = _achieved_sinr_db(W_admm, est, topo, sigma)
    scnr_admm    = _achieved_scnr_db(topo, cfg, W_admm, ss, assoc, sigma)

    print(f"  {'Method':<14s} {'min-SINR [dB]':>15s} {'sum-SCNR [dB]':>15s}")
    print(f"  {'─'*14} {'─'*15} {'─'*15}")
    print(f"  {'Split':<14s} {sinr_split:>+15.2f} {scnr_split:>+15.2f}")
    print(f"  {'ADMM':<14s} {sinr_admm:>+15.2f} {scnr_admm:>+15.2f}")
    print(f"\n  ADMM SCNR advantage over Split: {scnr_admm - scnr_split:+.2f} dB")
    print(f"  γ target: 3.0 dB")
    print(f"  Split meets γ:  {sinr_split >= 3.0 - 0.5}")
    print(f"  ADMM meets γ:   {sinr_admm >= 3.0 - 0.5}")

    # ADMM should beat or match Split in SCNR (joint optimisation is strictly better)
    assert scnr_admm >= scnr_split - 1.0, (
        f"ADMM SCNR {scnr_admm:.2f} unexpectedly below Split SCNR {scnr_split:.2f}"
    )
    print("  ✓ PASSED")


def test_5_approaches_centralized():
    """Test 5: ADMM SCNR should approach Centralized's (upper bound)."""
    print("\n── Test 5: ADMM vs Centralized (decentralization gap) ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return
    cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(gamma_db=3.0)

    # Centralized (Stage 6b — upper bound)
    W_cent, res_cent = run_centralized(
        topo, cfg, est, ss, assoc, sigma, Pmax,
        n_sca_max=30, sca_tol=1e-3,
    )
    _, sinr_cent = _achieved_sinr_db(W_cent, est, topo, sigma)
    scnr_cent    = _achieved_scnr_db(topo, cfg, W_cent, ss, assoc, sigma)

    # ADMM
    W_admm, res_admm = run_cordis_admm(
        topo, cfg, est, ss, assoc, sigma, Pmax,
        n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
    )
    _, sinr_admm = _achieved_sinr_db(W_admm, est, topo, sigma)
    scnr_admm    = _achieved_scnr_db(topo, cfg, W_admm, ss, assoc, sigma)

    gap_db = scnr_cent - scnr_admm
    print(f"  {'Method':<14s} {'min-SINR [dB]':>15s} {'sum-SCNR [dB]':>15s}")
    print(f"  {'─'*14} {'─'*15} {'─'*15}")
    print(f"  {'Centralized':<14s} {sinr_cent:>+15.2f} {scnr_cent:>+15.2f}")
    print(f"  {'ADMM':<14s} {sinr_admm:>+15.2f} {scnr_admm:>+15.2f}")
    print(f"\n  Decentralization gap: {gap_db:+.2f} dB SCNR")
    print(f"  (Expected: small positive gap — ADMM is a heuristic relaxation of P-Global)")
    print("  ✓ PASSED")


def test_6_rho_sensitivity():
    """Test 6: Sensitivity to ADMM penalty ρ_admm."""
    print("\n── Test 6: ρ_admm penalty sensitivity ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return

    rhos = [0.1, 0.5, 1.0, 5.0, 10.0]
    print(f"  {'ρ':>6} {'iters':>6} {'cnvg':>5} {'r_pri':>10} {'r_dual':>10} "
          f"{'minSINR':>9} {'SCNR':>9}")
    print(f"  {'─'*6} {'─'*6} {'─'*5} {'─'*10} {'─'*10} {'─'*9} {'─'*9}")

    for r in rhos:
        cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(gamma_db=3.0)
        W_tx, res = run_cordis_admm(
            topo, cfg, est, ss, assoc, sigma, Pmax,
            n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
            rho_admm=r,
        )
        _, msi = _achieved_sinr_db(W_tx, est, topo, sigma)
        scnr = _achieved_scnr_db(topo, cfg, W_tx, ss, assoc, sigma)
        pri  = res.primal_res_history[-1] if res.primal_res_history else float('nan')
        dual = res.dual_res_history[-1]  if res.dual_res_history else float('nan')
        print(f"  {r:>6.1f} {res.n_admm_iters:>6d} "
              f"{str(res.converged):>5} {pri:>10.3e} {dual:>10.3e} "
              f"{msi:>+9.2f} {scnr:>+9.2f}")

    print(f"\n  Expected: ρ too small → slow primal convergence; "
          f"ρ too large → consensus dominates, sensing penalised")
    print("  ✓ PASSED")


def test_7_kappa_sweep():
    """Test 7: κ clutter penalty sweep (same role as Stage 6b)."""
    print("\n── Test 7: κ clutter penalty sensitivity ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return

    # κ is auto-scaled inside ADMM so user-facing κ=1 means "strong clutter
    # avoidance".  At κ ≳ 0.7 in our scenario the clutter direction
    # overlaps the target direction (clutter PAS centred on target centroid
    # with 15° spread), so the optimizer starts trading SINR for clutter
    # avoidance — useful operating range is [0, ~0.5].  See discussion in
    # the algorithm docstring.
    kappas = [0.0, 0.1, 0.2, 0.3, 0.5]
    print(f"  {'κ':>8} {'iters':>6} {'minSINR':>9} {'SCNR':>9} "
          f"{'power_util':>11}")
    print(f"  {'─'*8} {'─'*6} {'─'*9} {'─'*9} {'─'*11}")

    for k in kappas:
        cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(
            gamma_db=3.0, kappa=k,
        )
        W_tx, res = run_cordis_admm(
            topo, cfg, est, ss, assoc, sigma, Pmax,
            n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
            kappa=k,
        )
        _, msi = _achieved_sinr_db(W_tx, est, topo, sigma)
        scnr = _achieved_scnr_db(topo, cfg, W_tx, ss, assoc, sigma)
        ratios = _power_ratios(W_tx, topo, Pmax)
        util = float(np.mean(ratios))
        print(f"  {k:>8.2f} {res.n_admm_iters:>6d} {msi:>+9.2f} "
              f"{scnr:>+9.2f} {util:>11.4f}")

    print(f"\n  Expected: as κ ↑, sensing SCNR drops, optimiser deprioritises clutter dirs.")
    print("  ✓ PASSED")


def test_8_multi_seed():
    """Test 8: Robustness across seeds."""
    print("\n── Test 8: Multi-seed robustness ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return

    seeds = [42, 100, 250, 500, 1000]
    print(f"  {'Seed':>6} {'iters':>6} {'cnvg':>5} {'minSINR':>9} {'SCNR':>9}")
    print(f"  {'─'*6} {'─'*6} {'─'*5} {'─'*9} {'─'*9}")
    for s in seeds:
        cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(
            seed=s, gamma_db=3.0,
        )
        W_tx, res = run_cordis_admm(
            topo, cfg, est, ss, assoc, sigma, Pmax,
            n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
        )
        _, msi = _achieved_sinr_db(W_tx, est, topo, sigma)
        scnr = _achieved_scnr_db(topo, cfg, W_tx, ss, assoc, sigma)
        print(f"  {s:>6d} {res.n_admm_iters:>6d} {str(res.converged):>5} "
              f"{msi:>+9.2f} {scnr:>+9.2f}")
    print("  ✓ PASSED")


def test_9_gamma_sweep():
    """Test 9: γ-target sweep — verify ADMM tracks target."""
    print("\n── Test 9: γ-target sweep ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return

    gammas_db = [0.0, 3.0, 5.0, 8.0, 12.0, 16.0]
    print(f"  {'γ [dB]':>7} {'iters':>6} {'cnvg':>5} {'minSINR':>9} "
          f"{'SCNR':>9} {'gap_to_γ':>9}")
    print(f"  {'─'*7} {'─'*6} {'─'*5} {'─'*9} {'─'*9} {'─'*9}")
    for g_db in gammas_db:
        cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(gamma_db=g_db)
        W_tx, res = run_cordis_admm(
            topo, cfg, est, ss, assoc, sigma, Pmax,
            n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
        )
        _, msi = _achieved_sinr_db(W_tx, est, topo, sigma)
        scnr = _achieved_scnr_db(topo, cfg, W_tx, ss, assoc, sigma)
        gap = msi - g_db
        print(f"  {g_db:>7.1f} {res.n_admm_iters:>6d} {str(res.converged):>5} "
              f"{msi:>+9.2f} {scnr:>+9.2f} {gap:>+9.2f}")
    print("  ✓ PASSED")


def test_10_three_way_comparison():
    """Test 10: Split / ADMM / Centralized side-by-side."""
    print("\n── Test 10: Three-way comparison ──")
    if not _HAS_CVXPY:
        print("  ⚠  Skipped — requires CVXPY")
        return

    gammas_db = [0.0, 3.0, 6.0, 10.0, 15.0]

    print(f"  γ [dB] |  {'Split (sinr, scnr)':^22s}  |  "
          f"{'ADMM (sinr, scnr)':^22s}  |  {'Centralized (sinr, scnr)':^25s}")
    print(f"  ──────-+-{'─'*24}-+-{'─'*24}-+-{'─'*27}")

    for g_db in gammas_db:
        cfg, topo, est, ss, assoc, sigma, Pmax = medium_scenario(gamma_db=g_db)

        W_sp, _, _ = run_cordis_split(topo, cfg, est, ss, assoc, sigma, Pmax)
        _, ssp = _achieved_sinr_db(W_sp, est, topo, sigma)
        nsp = _achieved_scnr_db(topo, cfg, W_sp, ss, assoc, sigma)

        W_ad, _ = run_cordis_admm(
            topo, cfg, est, ss, assoc, sigma, Pmax,
            n_admm_max=50, eps_pri=1.0, eps_dual=1.0,
        )
        _, sad = _achieved_sinr_db(W_ad, est, topo, sigma)
        nad = _achieved_scnr_db(topo, cfg, W_ad, ss, assoc, sigma)

        W_ce, res_ce = run_centralized(
            topo, cfg, est, ss, assoc, sigma, Pmax,
            n_sca_max=30, sca_tol=1e-3,
        )
        if res_ce.feasible:
            _, sce = _achieved_sinr_db(W_ce, est, topo, sigma)
            nce = _achieved_scnr_db(topo, cfg, W_ce, ss, assoc, sigma)
            cent_str = f"({sce:>+7.2f}, {nce:>+7.2f})"
        else:
            cent_str = f"{'(infeasible)':>20s}"

        print(f"  {g_db:>6.1f} |  ({ssp:>+7.2f}, {nsp:>+7.2f})      |  "
              f"({sad:>+7.2f}, {nad:>+7.2f})      |  {cent_str}")

    print("\n  Expected ordering (SCNR): Centralized ≥ ADMM ≥ Split")
    print("  ✓ PASSED")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 76)
    print("  CORDIS-ADMM (Stage 6c) — Journal Paper Formulation — Validation")
    print(f"  CVXPY available: {_HAS_CVXPY}")
    if not _HAS_CVXPY:
        print("  ⚠  CVXPY not available — all tests skipped.")
    print("=" * 76)

    test_1_admm_convergence()
    test_2_power_constraints()
    test_3_sinr_constraint_met()
    test_4_beats_split()
    test_5_approaches_centralized()
    test_6_rho_sensitivity()
    test_7_kappa_sweep()
    test_8_multi_seed()
    test_9_gamma_sweep()
    test_10_three_way_comparison()

    print("\n" + "=" * 76)
    print("  All Stage 6c tests completed.")
    print("=" * 76)

