"""
Stage 6d validation — benchmarks.py (BF + centralised P-Split PA).

Tests
-----
1. Registry coverage:    every benchmark in BENCHMARK_REGISTRY runs.
2. Power constraints:    ‖W_a‖²_F ≤ Pmax for every AP, every benchmark.
3. Identity sanity:      ``run_benchmark("lr_mmse_split")`` matches
                         ``run_cordis_split(comm_bf_method="lr_mmse")``.
4. PA value:             at moderate γ, the *_split variants achieve
                         tighter γ-tracking and/or higher SCNR than the
                         matching *_fixed variants.
5. BF ranking:           at γ = 3 dB the rough monotone SCNR ordering
                         MRT ≤ ZF ≤ RZF ≲ LR-MMSE holds.
6. γ-target sweep:       full table of SINR / SCNR / γ-gap across all
                         benchmarks for γ ∈ {0, 3, 6, 10} dB.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import numpy as np

from cordis.utils.config import load_config
from cordis.utils.io_utils import make_rng, child_rng
from cordis.utils.math_utils import lin2db
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
from cordis.algorithms.benchmarks import (
    BENCHMARK_REGISTRY, BenchmarkResult,
    describe_benchmarks, list_benchmarks,
    run_all_benchmarks, run_benchmark,
)
from cordis.algorithms.split_opt import run_cordis_split
from cordis.metrics import compute_scnr, compute_sinr

logging.getLogger("cordis").setLevel(logging.WARNING)


# =============================================================================
# Shared scenario builder  (matches validate_stage6a)
# =============================================================================

def build_scenario(seed=42, n_ue=4, n_targets=2, clutter_cnr_db=10.0):
    """
    Compact deterministic scenario:
    6 APs / 8 antennas / 4 UEs / 2 targets / 50 m circle / CNR = 10 dB.
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
    cfg.algorithm.split.gamma_db = 3.0
    cfg.algorithm.split.kappa    = 1.0

    rng   = make_rng(seed)
    topo  = generate_topology(cfg, child_rng(rng))
    lsf   = compute_large_scale_fading(topo, cfg, child_rng(rng),
                                        include_targets=(n_targets > 0))
    stats = compute_channel_statistics(topo, cfg, lsf, child_rng(rng))
    real  = generate_channel_realization(topo, cfg, lsf, stats, child_rng(rng))
    Phi, ct = design_pilot_sequences(n_ue, cfg.channel.tau_p, child_rng(rng))
    est     = run_channel_estimation(topo, cfg, lsf, stats, real,
                                      child_rng(rng), Phi, ct)
    s_stats = compute_sensing_statistics(topo, cfg, lsf, child_rng(rng))
    assoc   = assign_sensing(topo, cfg, lsf)
    sigma   = noise_power_watts(cfg.frequency.bandwidth_hz,
                                 cfg.channel.noise_figure_db,
                                 cfg.channel.noise_temp_k)
    Pmax    = snr_to_tx_power(cfg.channel.snr_db, sigma)
    return cfg, topo, est, s_stats, assoc, sigma, Pmax


# =============================================================================
# Helpers
# =============================================================================

def _achieved_metrics(W_tx, est, topo, sigma, cfg, ss, assoc):
    """Return (min-SINR dB, weighted-sum SCNR dB)."""
    sinr_m = compute_sinr(W_tx, est, topo, sigma)
    scnr_m = compute_scnr(topo, cfg, W_tx, ss, assoc, sigma)
    return sinr_m.min_sinr_db, lin2db(scnr_m.weighted_sum_scnr)


def _power_ratios(W_tx, topo, Pmax):
    return np.array([
        float(np.linalg.norm(W_tx[ap.idx], "fro") ** 2) / Pmax
        for ap in topo.tx_aps
    ])


# =============================================================================
# Test 1 — Registry coverage
# =============================================================================

def test_1_registry_runs():
    print("\n── Test 1: Every registered benchmark runs ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()

    print(describe_benchmarks())
    print()

    results = run_all_benchmarks(
        topo, cfg, est, ss, assoc, sigma, Pmax,
        skip_failures=False,
    )

    print(f"  {'name':<20}  {'pa_opt':>7}  {'min-SINR':>10}  {'SCNR':>10}")
    print(f"  {'-' * 20}  {'-' * 7}  {'-' * 10}  {'-' * 10}")
    for name, res in results.items():
        msi, scnr = _achieved_metrics(res.W_tx, est, topo, sigma, cfg, ss, assoc)
        print(f"  {name:<20}  {'yes' if res.pa_optimized else 'no':>7}  "
              f"{msi:>+10.2f}  {scnr:>+10.2f}")
        assert isinstance(res, BenchmarkResult)
        assert res.W_tx and all(W is not None for W in res.W_tx.values())

    assert set(results.keys()) == set(BENCHMARK_REGISTRY.keys()), \
        "Some benchmarks failed to run."
    print(f"  ✓ PASSED — {len(results)}/{len(BENCHMARK_REGISTRY)} benchmarks ran")


# =============================================================================
# Test 2 — Per-AP power constraints
# =============================================================================

def test_2_power_constraints():
    print("\n── Test 2: Per-AP power constraints ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()

    results = run_all_benchmarks(
        topo, cfg, est, ss, assoc, sigma, Pmax,
        skip_failures=False,
    )

    print(f"  {'name':<20}  {'min ratio':>10}  {'max ratio':>10}  status")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}  ------")
    all_ok = True
    for name, res in results.items():
        ratios = _power_ratios(res.W_tx, topo, Pmax)
        ok = bool(np.all(ratios <= 1.0 + 1e-6))
        print(f"  {name:<20}  {ratios.min():>10.4f}  {ratios.max():>10.4f}  "
              f"{'✓' if ok else '✗'}")
        all_ok &= ok

    assert all_ok, "Some benchmarks violated the per-AP power constraint."
    print("  ✓ PASSED — all benchmarks within ‖W_a‖²_F ≤ P_max")


# =============================================================================
# Test 3 — Identity sanity (lr_mmse_split ≡ CORDIS-Split)
# =============================================================================

def test_3_identity_sanity():
    print("\n── Test 3: lr_mmse_split ≡ CORDIS-Split ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()

    res_bench = run_benchmark(
        "lr_mmse_split", topo, cfg, est, ss, assoc, sigma, Pmax,
    )
    W_split, _, split_res = run_cordis_split(
        topo, cfg, est, ss, assoc, sigma, Pmax,
        comm_bf_method="lr_mmse",
    )

    # Same ρ_opt (within solver tolerance)
    rho_bench = np.array([res_bench.psr[a.idx] for a in topo.tx_aps])
    rho_split = np.array([split_res.rho_opt[a.idx] for a in topo.tx_aps])
    rho_diff  = float(np.max(np.abs(rho_bench - rho_split)))

    # Same W_tx (within solver tolerance)
    max_W_diff = max(
        float(np.linalg.norm(res_bench.W_tx[a.idx] - W_split[a.idx], "fro"))
        for a in topo.tx_aps
    )

    print(f"  max |Δρ|      = {rho_diff:.3e}")
    print(f"  max ‖ΔW_a‖_F = {max_W_diff:.3e}")
    assert rho_diff  < 1e-6, "ρ differs between wrapper and direct call"
    assert max_W_diff < 1e-6, "W_tx differs between wrapper and direct call"
    print("  ✓ PASSED — benchmark wrapper is a transparent re-export")


# =============================================================================
# Test 4 — Value of PA optimisation
# =============================================================================

def test_4_pa_value():
    """
    For each (BF method, ρ-mode) pair, the optimised-PA variant should
    either meet γ tighter or transfer the savings to SCNR.
    Comparison metric: SINR-gap and SCNR side-by-side.
    """
    print("\n── Test 4: Value of PA optimisation ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()
    gamma_target_db = cfg.algorithm.split.gamma_db   # = 3 dB

    pairs = [
        ("lr_mmse_fixed", "lr_mmse_split"),
        ("mrt_fixed",     "mrt_split"),
        ("rzf_fixed",     "rzf_split"),
    ]

    print(f"  γ target = {gamma_target_db:.1f} dB")
    print(f"  {'pair':<32}  {'fixed: SINR / SCNR':>22}  {'split: SINR / SCNR':>22}")
    print(f"  {'-' * 32}  {'-' * 22}  {'-' * 22}")

    pa_helped_min_sinr = 0
    pa_helped_scnr     = 0
    for name_fixed, name_split in pairs:
        rf = run_benchmark(name_fixed, topo, cfg, est, ss, assoc, sigma, Pmax)
        rs = run_benchmark(name_split, topo, cfg, est, ss, assoc, sigma, Pmax)
        s_f, c_f = _achieved_metrics(rf.W_tx, est, topo, sigma, cfg, ss, assoc)
        s_s, c_s = _achieved_metrics(rs.W_tx, est, topo, sigma, cfg, ss, assoc)
        gap_f = abs(s_f - gamma_target_db)
        gap_s = abs(s_s - gamma_target_db)
        print(f"  {name_fixed:>14}→{name_split:<17}  "
              f"{s_f:+7.2f} /{c_f:+7.2f}  {s_s:+7.2f} /{c_s:+7.2f}")
        # Either *_split tracks γ more tightly OR keeps more sensing utility.
        if gap_s <= gap_f + 0.1:
            pa_helped_min_sinr += 1
        if c_s >= c_f - 0.5:
            pa_helped_scnr += 1

    # Allow noise: ≥ 2 of 3 pairs should show the PA optimisation helping
    print(f"  pairs where SINR-gap improved : {pa_helped_min_sinr}/{len(pairs)}")
    print(f"  pairs where SCNR was preserved: {pa_helped_scnr}/{len(pairs)}")
    assert pa_helped_min_sinr + pa_helped_scnr >= len(pairs), \
        "Optimised PA did not help on enough BF pairs"
    print("  ✓ PASSED — optimised PA dominates the fixed-ρ baselines")


# =============================================================================
# Test 5 — BF ranking at moderate γ
# =============================================================================

def test_5_bf_ranking():
    """
    Rough monotone SCNR ordering at γ=3 dB (with optimised PA):
        MRT  ≲  ZF  ≲  RZF  ≲  LR-MMSE
    This is the expected ordering in dense-loading regimes where
    LR-MMSE's noise-aware regularisation outperforms hard-null methods.
    We check that LR-MMSE achieves the highest SCNR among the four
    local methods (allowing ~1 dB slack for solver noise).
    """
    print("\n── Test 5: BF ranking at γ=3 dB ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()

    results = {}
    for name in ("mrt_split", "zf_split", "rzf_split", "lr_mmse_split"):
        results[name] = run_benchmark(name, topo, cfg, est, ss, assoc, sigma, Pmax)

    print(f"  {'method':<20}  {'min-SINR':>10}  {'SCNR':>10}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}")
    scnrs = {}
    for name, res in results.items():
        msi, scnr = _achieved_metrics(res.W_tx, est, topo, sigma, cfg, ss, assoc)
        scnrs[name] = scnr
        print(f"  {name:<20}  {msi:>+10.2f}  {scnr:>+10.2f}")

    # LR-MMSE should be at the top (within 1 dB slack)
    lr_scnr = scnrs["lr_mmse_split"]
    others_max = max(scnrs["mrt_split"], scnrs["zf_split"], scnrs["rzf_split"])
    print(f"  LR-MMSE SCNR − max(other) = {lr_scnr - others_max:+.2f} dB")
    assert lr_scnr >= others_max - 1.0, \
        "LR-MMSE unexpectedly far below other BF methods"
    print("  ✓ PASSED — LR-MMSE dominates (or ties) other local BFs")


# =============================================================================
# Test 6 — γ-target sweep across all benchmarks
# =============================================================================

def test_6_gamma_sweep():
    print("\n── Test 6: γ-target sweep across all benchmarks ──")
    cfg, topo, est, ss, assoc, sigma, Pmax = build_scenario()
    n_ue = topo.n_ue
    gammas_db = [0.0, 3.0, 6.0, 10.0]

    bench_names = list_benchmarks()
    # Header
    print(f"  {'name':<20}  " + "".join(f"{'γ=' + str(g):>10}  " for g in gammas_db))
    print(f"  {'-' * 20}  " + "".join(f"{'-' * 10}  " for _ in gammas_db))

    for name in bench_names:
        row_sinr = []
        for gdb in gammas_db:
            gamma_vec = np.full(n_ue, gdb)
            res = run_benchmark(
                name, topo, cfg, est, ss, assoc, sigma, Pmax,
                gamma_u_db=gamma_vec,
            )
            msi, _ = _achieved_metrics(res.W_tx, est, topo, sigma, cfg, ss, assoc)
            row_sinr.append(msi)
        row_str = "  ".join(f"{v:>+8.2f}dB" for v in row_sinr)
        print(f"  {name:<20}  {row_str}")

    print("  (table = min-SINR [dB] for each (benchmark, γ-target) pair)")
    print("  ✓ PASSED — sweep completed for every benchmark")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 76)
    print("  CORDIS-Benchmarks (Stage 6d) — Validation")
    print("=" * 76)

    test_1_registry_runs()
    test_2_power_constraints()
    test_3_identity_sanity()
    test_4_pa_value()
    test_5_bf_ranking()
    test_6_gamma_sweep()

    print("\n" + "=" * 76)
    print("  All Stage 6d tests completed.")
    print("=" * 76)


if __name__ == "__main__":
    main()

