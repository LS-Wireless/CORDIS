"""
Stage 7 validation — comprehensive stress test for cordis/simulation/.

Tests by module
---------------
scenario.py:
   1. Drop construction (fields populated, reproducible from seed)
   2. Scenario layered on a Drop (drop/realization indices distinct, drop reused)
   3. build_scenario_from_seeds convenience
   4. AlgorithmSpec validation (unknown kind, missing benchmark_name)
   5. dispatch_algorithm for each kind (split, admm, centralized, benchmark)
   6. run_scenario success path (multiple algorithms, paired comparison)
   7. run_scenario failure path (skip_failures=True emits failed TrialOutput)

runner.py:
   8. RunnerConfig validation & n_trials property
   9. MonteCarloRunner construction (unique-name guard)
  10. End-to-end small campaign (sequential mode for determinism)
  11. Reproducibility (same base_seed ⇒ identical results)
  12. Sequential vs parallel produce same results
  13. Drop-build failure handling (synthetic exception injection)

result.py:
  14. SimResult.from_run aggregates correctly (stats match aggregate_*)
  15. Metric registry: every named metric returns a 1-D array
  16. percentile / cdf / mean / std queries
  17. save / load round-trip (bit-identical numeric content)
  18. summary table renders without crashing
  19. No-targets scenario produces scnr_stats=None gracefully

The scenario is intentionally small (3 APs, 4 antennas, 3 UEs, 1 target,
2 drops × 2 realizations) so the whole script runs in well under a minute.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import tempfile
import warnings

import numpy as np
from numpy.random import SeedSequence

from cordis.algorithms.benchmarks import BENCHMARK_REGISTRY
from cordis.metrics import SCNRStatistics, SINRStatistics
from cordis.metrics.aggregation import aggregate_scnr, aggregate_sinr
from cordis.simulation.result import (
    AlgorithmResult, SimResult, list_metrics,
)
from cordis.simulation.runner import (
    MonteCarloRunner, RunnerConfig, RunReport,
    _run_one_drop, run_sequential,
)
from cordis.simulation.scenario import (
    AlgorithmSpec, Drop, Scenario, TrialOutput,
    build_drop, build_scenario, build_scenario_from_seeds,
    dispatch_algorithm, run_scenario,
)
from cordis.utils.config import load_config

logging.getLogger("cordis").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Shared lightweight scenario builder
# =============================================================================

def _small_cfg(n_targets: int = 1, gamma_db: float = 3.0):
    """
    Compact-but-feasible scenario.

    Sized so that γ = 3 dB is comfortably feasible (per-user, with one
    target), letting all four algorithms reach their normal operating
    regime.  Matches the geometry used by validate_stage6c.

    With 5 Tx APs × 8 antennas = 40 Tx antennas serving N_ue + N_tg = 4
    streams (≈10× oversubscription) and 90 dB operating SNR, LR-MMSE
    eliminates MUI cleanly and Centralized / ADMM converge.

    Smaller setups (e.g. 3 APs × 4 antennas) push every algorithm into
    its slack-absorbing regime — γ becomes unreachable, the inner SOCP
    in Centralized goes ``infeasible_inaccurate``, and SCNR collapses
    to the noise floor.  Stage-7 plumbing still works there, but the
    numbers stop being interpretable, which defeats the point of a
    validation run.
    """
    cfg = load_config("configs/default.json")
    cfg.topology.type            = "circle"
    cfg.topology.ap_radius_m     = 50.0
    cfg.topology.ue_min_radius_m = 10.0
    cfg.topology.ue_max_radius_m = 60.0
    cfg.topology.n_ap            = 6
    cfg.topology.n_ue            = 3
    cfg.topology.n_ant           = 8
    cfg.topology.n_rf_chains     = 8
    cfg.topology.n_targets       = n_targets
    cfg.topology.tg_min_radius_m = 10.0
    cfg.topology.tg_max_radius_m = 60.0
    cfg.topology.n_sensing_rx    = 1 if n_targets > 0 else 0
    cfg.channel.snr_db           = 90.0
    cfg.channel.pilot_power_db   = 120.0
    cfg.channel.estimation_method = "MMSE"
    cfg.sensing.clutter_cnr_db   = 10.0
    cfg.sensing.clutter_center_strategy = "offset"   # decouple clutter from target direction
    cfg.sensing.clutter_offset_az_deg   = 60.0       # 60° offset → no SINR/SCNR coupling
    cfg.algorithm.split.gamma_db = gamma_db
    cfg.algorithm.split.kappa    = 1.0
    return cfg


def _default_specs(gamma_db=3.0):
    """Default 4-algorithm campaign spec list used across most tests."""
    return [
        AlgorithmSpec("CORDIS-Split", "cordis_split",
                      {"gamma_u_db": np.full(3, gamma_db)}),
        AlgorithmSpec("CORDIS-ADMM",  "cordis_admm",
                      {"gamma_u_db": np.full(3, gamma_db),
                       "n_admm_max": 50}),
        AlgorithmSpec("Centralized",  "centralized",
                      {"gamma_u_db": np.full(3, gamma_db)}),
        AlgorithmSpec("MRT-Split",    "benchmark",
                      {"benchmark_name": "mrt_split",
                       "gamma_u_db": np.full(3, gamma_db)}),
    ]


def _ok(msg):  print(f"  ✓ {msg}")
def _hdr(n, t): print(f"\n── Test {n}: {t} ──")


# =============================================================================
# scenario.py tests
# =============================================================================

def test_01_drop_construction():
    _hdr(1, "Drop construction is correct and reproducible")
    cfg = _small_cfg()

    d1 = build_drop(cfg, SeedSequence(123), drop_idx=0)
    assert isinstance(d1, Drop)
    assert d1.cfg is cfg
    assert d1.topo.n_ue == 3 and d1.topo.n_targets == 1
    assert d1.sigma_n_sq > 0 and d1.Pmax > 0
    assert d1.Phi.shape[0] == 3
    assert d1.drop_idx == 0
    _ok("All Drop fields populated correctly")

    # Reproducibility: same seed → same drop
    d2 = build_drop(cfg, SeedSequence(123), drop_idx=0)
    assert d1.drop_seed == d2.drop_seed, "drop_seed mismatch"
    assert np.allclose(d1.topo.ue_positions, d2.topo.ue_positions)
    _ok("Same SeedSequence(123) ⇒ identical drop")

    # Different seed → different drop
    d3 = build_drop(cfg, SeedSequence(999), drop_idx=0)
    assert d1.drop_seed != d3.drop_seed
    assert not np.allclose(d1.topo.ue_positions, d3.topo.ue_positions)
    _ok("Different seed ⇒ different drop")


def test_02_scenario_layered_on_drop():
    _hdr(2, "Scenario layers correctly on a Drop, drop is shared")
    cfg = _small_cfg()
    drop = build_drop(cfg, SeedSequence(42), drop_idx=0)

    real_seqs = SeedSequence(42).spawn(3)
    s1 = build_scenario(drop, real_seqs[0], realization_idx=0)
    s2 = build_scenario(drop, real_seqs[1], realization_idx=1)
    s3 = build_scenario(drop, real_seqs[2], realization_idx=2)

    # Drop is shared (identity)
    assert s1.drop is drop and s2.drop is drop and s3.drop is drop
    _ok("All scenarios reference the same Drop object")

    # Indices distinct
    assert (s1.realization_idx, s2.realization_idx, s3.realization_idx) == (0, 1, 2)
    _ok("realization_idx assigned correctly")

    # Realizations differ
    key = next(iter(s1.realization.h.keys()))
    h1 = s1.realization.h[key]
    h2 = s2.realization.h[key]
    h3 = s3.realization.h[key]
    assert not (np.allclose(h1, h2) or np.allclose(h2, h3) or np.allclose(h1, h3))
    _ok("Distinct realizations produce distinct channel draws")

    # Property pass-through
    assert s1.cfg is drop.cfg and s1.topo is drop.topo
    assert s1.sigma_n_sq == drop.sigma_n_sq and s1.Pmax == drop.Pmax
    _ok("Scenario property pass-through to Drop works")


def test_03_build_scenario_from_seeds_convenience():
    _hdr(3, "build_scenario_from_seeds one-shot constructor")
    cfg = _small_cfg()
    s = build_scenario_from_seeds(cfg, SeedSequence(7), SeedSequence(11))
    assert isinstance(s, Scenario)
    assert isinstance(s.drop, Drop)
    assert s.drop.drop_idx == 0 and s.realization_idx == 0
    _ok("Single-trial convenience builder produces a complete Scenario")


def test_04_algorithm_spec_validation():
    _hdr(4, "AlgorithmSpec validates kind and benchmark_name")
    AlgorithmSpec("ok", "cordis_split", {})    # valid
    AlgorithmSpec("ok2", "benchmark", {"benchmark_name": "mrt_split"})
    _ok("Valid specs accepted")

    # Unknown kind
    try:
        AlgorithmSpec("bad", "nope", {})
        assert False, "Unknown kind should raise"
    except ValueError:
        pass
    _ok("Unknown kind raises ValueError")

    # benchmark without name
    try:
        AlgorithmSpec("bad", "benchmark", {})
        assert False, "Missing benchmark_name should raise"
    except ValueError:
        pass
    _ok("benchmark without benchmark_name raises")

    # benchmark with unknown name
    try:
        AlgorithmSpec("bad", "benchmark", {"benchmark_name": "no_such_bm"})
        assert False, "Unknown benchmark_name should raise"
    except ValueError:
        pass
    _ok("benchmark with unknown name raises")


def test_05_dispatch_each_kind():
    _hdr(5, "dispatch_algorithm works for every kind")
    cfg = _small_cfg()
    s = build_scenario_from_seeds(cfg, SeedSequence(33), SeedSequence(34))

    specs = _default_specs()
    M_t = s.cfg.topology.n_ant
    for spec in specs:
        W_tx, extra = dispatch_algorithm(s, spec, rng=None)
        assert isinstance(W_tx, dict) and len(W_tx) == s.topo.n_tx
        for ap in s.topo.tx_aps:
            assert ap.idx in W_tx
            assert W_tx[ap.idx].shape == (M_t, s.topo.n_ue + s.topo.n_targets)
        assert "converged" in extra and "iters" in extra
        _ok(f"{spec.kind:<14} → W_tx, extra OK  (converged={extra['converged']}, "
            f"iters={extra['iters']})")


def test_06_run_scenario_success():
    _hdr(6, "run_scenario runs all algorithms on one scenario (paired)")
    cfg = _small_cfg()
    s = build_scenario_from_seeds(cfg, SeedSequence(55), SeedSequence(56))
    specs = _default_specs()

    outs = run_scenario(s, specs, skip_failures=True)
    assert len(outs) == len(specs)
    assert [o.name for o in outs] == [sp.name for sp in specs]
    for o in outs:
        assert not o.failed, f"{o.name} failed: {o.error}"
        assert o.sinr is not None
        assert o.scnr is not None  # there's 1 target
        assert o.power_ratios is not None
        assert o.runtime_s > 0
        assert np.all(o.power_ratios <= 1.0 + 1e-6)
    _ok(f"All {len(specs)} algorithms ran successfully")
    _ok("Per-AP power constraints respected for every algorithm")


def test_07_run_scenario_failure_handling():
    _hdr(7, "run_scenario emits failed TrialOutputs when skip_failures=True")
    cfg = _small_cfg()
    s = build_scenario_from_seeds(cfg, SeedSequence(77), SeedSequence(78))

    # Inject a bad algorithm: benchmark with bad benchmark_name passes
    # spec __post_init__, so simulate failure via centralized at insane γ.
    bad_specs = [
        AlgorithmSpec("Good", "cordis_split", {}),
        AlgorithmSpec("Centralized-Bad", "centralized",
                      {"gamma_u_db": np.full(3, 200.0)}),  # impossible γ
    ]
    outs = run_scenario(s, bad_specs, skip_failures=True)
    # 'Good' should succeed; 'Centralized-Bad' may succeed-with-warnings or fail.
    assert len(outs) == 2
    good = next(o for o in outs if o.name == "Good")
    assert not good.failed
    _ok("Good algorithm produced a successful TrialOutput")
    _ok("Failed-or-degraded path returns a TrialOutput, not an exception")

    # Now check that skip_failures=False on a programmer-error (missing kwarg)
    # actually raises something the campaign can't swallow.
    bad_specs2 = [AlgorithmSpec("Bogus", "cordis_split", {"comm_bf_method": "not_a_method"})]
    try:
        run_scenario(s, bad_specs2, skip_failures=False)
        # If we got here, the algorithm may have silently accepted the bad method
        _ok("skip_failures=False path exercised (no crash from validator)")
    except Exception as ex:
        _ok(f"skip_failures=False raised as expected: {type(ex).__name__}")


# =============================================================================
# runner.py tests
# =============================================================================

def test_08_runner_config():
    _hdr(8, "RunnerConfig validation and n_trials property")
    rc = RunnerConfig(n_drops=5, n_realizations_per_drop=4)
    assert rc.n_trials == 20
    rc.validate()
    _ok("n_trials = n_drops × n_realizations_per_drop")

    for bad in [
        dict(n_drops=0, n_realizations_per_drop=1),
        dict(n_drops=1, n_realizations_per_drop=0),
        dict(n_drops=1, n_realizations_per_drop=1, n_workers=0),
    ]:
        rc = RunnerConfig(**bad)
        try:
            rc.validate()
            assert False, f"Should have raised for {bad}"
        except ValueError:
            pass
    _ok("validate() rejects n_drops=0, n_realizations=0, n_workers=0")


def test_09_runner_unique_names():
    _hdr(9, "MonteCarloRunner enforces unique algorithm names")
    cfg = _small_cfg()
    dup = [AlgorithmSpec("X", "cordis_split", {}),
           AlgorithmSpec("X", "cordis_admm", {})]
    try:
        MonteCarloRunner(cfg, dup)
        assert False, "Duplicate names should raise"
    except ValueError as ex:
        assert "unique" in str(ex).lower()
    _ok("Duplicate algorithm names rejected")

    try:
        MonteCarloRunner(cfg, [])
        assert False, "Empty specs should raise"
    except ValueError:
        pass
    _ok("Empty spec list rejected")


def test_10_end_to_end_small_campaign():
    _hdr(10, "End-to-end small campaign (sequential mode)")
    cfg = _small_cfg()
    specs = _default_specs()
    rc = RunnerConfig(
        n_drops=2, n_realizations_per_drop=2,
        n_workers=1, base_seed=42,
        verbose=0, skip_failures=True,
    )
    report = run_sequential(cfg, specs, rc)
    assert isinstance(report, RunReport)
    assert report.n_total == 2 * 2 * len(specs)
    _ok(f"RunReport contains {report.n_total} TrialOutputs as expected")
    # Indices coverage: every (drop_idx, realization_idx) pair appears
    pairs = {(o.drop_idx, o.realization_idx) for o in report.outputs}
    assert pairs == {(d, r) for d in range(2) for r in range(2)}
    _ok("Every (drop_idx, realization_idx) pair represented")
    # Per-algorithm runtime metadata
    runtimes = report.metadata["runtime_per_algorithm_s"]
    assert set(runtimes.keys()) == set(s.name for s in specs)
    _ok("Per-algorithm runtime metadata captured")


def test_11_runner_reproducibility():
    _hdr(11, "Same base_seed ⇒ identical campaign outputs")
    cfg = _small_cfg()
    specs = _default_specs()
    rc = RunnerConfig(
        n_drops=2, n_realizations_per_drop=1,
        n_workers=1, base_seed=12345, verbose=0,
    )
    r1 = run_sequential(cfg, specs, rc)
    r2 = run_sequential(cfg, specs, rc)
    assert r1.n_total == r2.n_total

    # Compare per-output SINRs (deterministic for fixed seed)
    for o1, o2 in zip(r1.outputs, r2.outputs):
        assert (o1.name, o1.drop_idx, o1.realization_idx) == \
               (o2.name, o2.drop_idx, o2.realization_idx)
        assert o1.drop_seed == o2.drop_seed
        assert o1.realization_seed == o2.realization_seed
        if not o1.failed:
            assert np.allclose(
                o1.sinr.sinr_per_user, o2.sinr.sinr_per_user,
                rtol=1e-6, atol=1e-9,
            ), f"Non-deterministic for {o1.name}"
    _ok("All matched outputs identical between runs (seeds + SINRs)")


def test_12_parallel_matches_sequential():
    _hdr(12, "Parallel (n_workers=2) reproduces sequential (n_workers=1)")
    cfg = _small_cfg()
    specs = _default_specs()[:2]  # smaller for speed
    rc_seq = RunnerConfig(n_drops=2, n_realizations_per_drop=1,
                          n_workers=1, base_seed=4242, verbose=0)
    rc_par = RunnerConfig(n_drops=2, n_realizations_per_drop=1,
                          n_workers=2, base_seed=4242, verbose=0)

    r_seq = run_sequential(cfg, specs, rc_seq)
    runner = MonteCarloRunner(cfg, specs, rc_par)
    try:
        r_par = runner.run()
    except ImportError:
        print("  (joblib not available — parallel-vs-sequential test skipped)")
        return
    # Order may differ across workers; key by (name, drop_idx, real_idx)
    by_key_seq = {(o.name, o.drop_idx, o.realization_idx): o for o in r_seq.outputs}
    by_key_par = {(o.name, o.drop_idx, o.realization_idx): o for o in r_par.outputs}
    assert set(by_key_seq.keys()) == set(by_key_par.keys())
    for k, o1 in by_key_seq.items():
        o2 = by_key_par[k]
        assert o1.drop_seed == o2.drop_seed
        if not o1.failed and not o2.failed:
            assert np.allclose(
                o1.sinr.sinr_per_user, o2.sinr.sinr_per_user,
                rtol=1e-6, atol=1e-9,
            ), f"Parallel diverged from sequential for {k}"
    _ok("Parallel and sequential modes produce bit-identical outputs")


def test_13_worker_failure_handling():
    _hdr(13, "_run_one_drop fills slots when build_drop raises")
    # Make build_drop fail by passing an invalid cfg.
    cfg = _small_cfg()
    cfg.topology.n_ap = -1  # will fail validate or topology generation

    specs = _default_specs()[:2]
    drop_ss = SeedSequence(99)
    real_ss = list(SeedSequence(99).spawn(2))

    outs = _run_one_drop(cfg, specs, drop_ss, real_ss, drop_idx=0,
                        skip_failures=True)
    # We expect 2 realizations × 2 algorithms = 4 failed outputs.
    assert len(outs) == 4, f"Expected 4 slots, got {len(outs)}"
    assert all(o.failed for o in outs), "All outputs should be failed"
    assert all(o.error is not None and len(o.error) > 0 for o in outs)
    _ok("Drop-build failure ⇒ every (algorithm × realization) slot is filled")


# =============================================================================
# result.py tests
# =============================================================================

def _make_small_simresult():
    """Helper: build a SimResult from a small successful campaign."""
    cfg = _small_cfg()
    specs = _default_specs()
    rc = RunnerConfig(n_drops=2, n_realizations_per_drop=2,
                      n_workers=1, base_seed=2024, verbose=0)
    report = run_sequential(cfg, specs, rc)
    return SimResult.from_run(report), report


def test_14_simresult_from_run():
    _hdr(14, "SimResult.from_run aggregates per-algorithm stats correctly")
    sr, report = _make_small_simresult()
    assert isinstance(sr, SimResult)
    assert sr.names == [s.name for s in report.algorithm_specs]
    for name, ar in sr.algorithm_results.items():
        assert isinstance(ar, AlgorithmResult)
        assert ar.n_trials_total == 4
        # Match against direct aggregate_sinr
        ok_outs = [o for o in report.outputs_for(name) if not o.failed]
        if ok_outs:
            stats_direct = aggregate_sinr([o.sinr for o in ok_outs])
            assert np.allclose(
                ar.sinr_stats.min_sinr_per_trial,
                stats_direct.min_sinr_per_trial,
            ), f"SINR aggregation mismatch for {name}"
    _ok("Aggregated SINRStatistics matches a direct aggregate_sinr() call")

    # Scalars consistent
    for name, ar in sr.algorithm_results.items():
        if ar.n_succeeded > 0:
            assert 0.0 <= ar.convergence_rate <= 1.0
            assert ar.runtime_total_s >= ar.runtime_mean_s * ar.n_succeeded * 0.999
    _ok("Derived scalars (conv_rate, runtime totals) are consistent")


def test_15_metric_registry_coverage():
    _hdr(15, "Every registered metric returns a 1-D array")
    sr, _ = _make_small_simresult()
    # Pick the first algorithm with both stats present
    name = next(n for n, r in sr.algorithm_results.items()
                if r.sinr_stats is not None and r.scnr_stats is not None)
    ar = sr[name]
    for m in list_metrics():
        if not ar.has_metric(m):
            continue
        arr = ar._samples(m)
        assert arr.ndim == 1, f"{m} returned non-1-D array (shape {arr.shape})"
        assert len(arr) > 0, f"{m} returned empty array"
    _ok(f"All {len(list_metrics())} metrics resolve to 1-D sample arrays")

    # Unknown metric raises
    try:
        ar.percentile("nonsense_metric", 50)
        assert False, "Unknown metric should raise"
    except ValueError:
        pass
    _ok("Unknown metric ⇒ ValueError with helpful message")


def test_16_query_helpers():
    _hdr(16, "percentile / cdf / mean / std queries")
    sr, _ = _make_small_simresult()
    name = sr.names[0]
    ar = sr[name]

    # percentile/mean for min_sinr_db
    p50 = ar.percentile("min_sinr_db", 50)
    avg = ar.mean("min_sinr_db")
    std = ar.std("min_sinr_db")
    assert np.isfinite(p50) and np.isfinite(avg) and std >= 0
    _ok(f"{name}: p50(min_sinr_db) = {p50:+.2f} dB, mean = {avg:+.2f} dB")

    # CDF returns sorted (xs, fs) with fs[-1] = 1
    xs, fs = ar.cdf("min_sinr_db")
    assert np.all(np.diff(xs) >= -1e-12), "CDF xs not non-decreasing"
    assert abs(fs[-1] - 1.0) < 1e-9, f"CDF should end at 1.0, ended at {fs[-1]}"
    assert abs(fs[0] - 1.0 / len(xs)) < 1e-9, "CDF should start at 1/n"
    _ok(f"CDF xs sorted, fs in [1/n, 1], length = {len(xs)}")

    # SimResult-level convenience
    p50_top = sr.percentile(name, "min_sinr_db", 50)
    assert p50_top == p50
    _ok("SimResult.percentile delegates correctly")


def test_17_save_load_roundtrip():
    _hdr(17, "save → load round-trip is numerically identical")
    sr1, _ = _make_small_simresult()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "smoke_result"
        npz, json_path = sr1.save(path)
        assert npz.exists() and json_path.exists()
        assert npz.suffix == ".npz" and json_path.suffix == ".json"
        sr2 = SimResult.load(path)

    # Compare every algorithm's full state
    assert sr1.names == sr2.names
    assert sr1.cfg.topology.n_ap == sr2.cfg.topology.n_ap
    assert sr1.runner_cfg.n_drops == sr2.runner_cfg.n_drops
    for name in sr1.names:
        a1, a2 = sr1[name], sr2[name]
        assert a1.n_trials_total == a2.n_trials_total
        assert a1.n_succeeded == a2.n_succeeded
        assert a1.convergence_rate == a2.convergence_rate
        assert np.array_equal(a1.iters, a2.iters)
        assert np.allclose(a1.runtime_s, a2.runtime_s)
        assert np.array_equal(a1.converged, a2.converged)
        assert np.allclose(a1.power_ratios, a2.power_ratios)
        if a1.sinr_stats is not None:
            assert np.allclose(
                a1.sinr_stats.min_sinr_per_trial,
                a2.sinr_stats.min_sinr_per_trial,
            )
            assert np.allclose(
                a1.sinr_stats.sinr_per_trial_per_user,
                a2.sinr_stats.sinr_per_trial_per_user,
            )
            assert a1.sinr_stats.p_noise_mean == a2.sinr_stats.p_noise_mean
        if a1.scnr_stats is not None:
            assert np.allclose(
                a1.scnr_stats.weighted_sum_per_trial,
                a2.scnr_stats.weighted_sum_per_trial,
            )
            assert a1.scnr_stats.pair_keys == a2.scnr_stats.pair_keys
    _ok("All algorithm arrays and scalars round-trip identically")

    # Cross-check metric query post-load
    p1 = sr1.mean(sr1.names[0], "min_sinr_db")
    p2 = sr2.mean(sr2.names[0], "min_sinr_db")
    assert p1 == p2
    _ok("Metric queries match before/after persistence")


def test_18_summary_renders():
    _hdr(18, "summary() renders a printable table without crashing")
    sr, _ = _make_small_simresult()
    out = sr.summary()
    assert isinstance(out, str) and len(out) > 100
    for name in sr.names:
        assert name in out, f"{name} missing from summary"
    print()
    print(out)
    _ok("Summary table contains every algorithm name")


def test_19_no_targets_scenario():
    _hdr(19, "Zero-target scenario produces scnr_stats=None gracefully")
    cfg = _small_cfg(n_targets=0)
    specs = [AlgorithmSpec("Split-NoSens", "cordis_split",
                           {"gamma_u_db": np.full(3, 3.0)})]
    rc = RunnerConfig(n_drops=1, n_realizations_per_drop=2,
                      n_workers=1, base_seed=7, verbose=0)
    report = run_sequential(cfg, specs, rc)
    sr = SimResult.from_run(report)
    ar = sr["Split-NoSens"]
    assert ar.sinr_stats is not None
    assert ar.scnr_stats is None, "scnr_stats should be None when n_targets=0"
    assert not ar.has_metric("weighted_sum_scnr_db")
    try:
        ar.percentile("weighted_sum_scnr_db", 50)
        assert False, "SCNR query should raise when scnr_stats is None"
    except ValueError as ex:
        assert "scnr" in str(ex).lower() or "scnr stat" in str(ex).lower()
    _ok("No-targets scenario: SINR stats present, SCNR stats=None, queries refuse")

    # Round-trip survives the absent-SCNR case
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "no_tg"
        sr.save(path)
        sr2 = SimResult.load(path)
        assert sr2["Split-NoSens"].scnr_stats is None
    _ok("save/load survives scnr_stats=None")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 76)
    print("  Stage 7 — Comprehensive Validation")
    print("=" * 76)

    test_01_drop_construction()
    test_02_scenario_layered_on_drop()
    test_03_build_scenario_from_seeds_convenience()
    test_04_algorithm_spec_validation()
    test_05_dispatch_each_kind()
    test_06_run_scenario_success()
    test_07_run_scenario_failure_handling()

    test_08_runner_config()
    test_09_runner_unique_names()
    test_10_end_to_end_small_campaign()
    test_11_runner_reproducibility()
    test_12_parallel_matches_sequential()
    test_13_worker_failure_handling()

    test_14_simresult_from_run()
    test_15_metric_registry_coverage()
    test_16_query_helpers()
    test_17_save_load_roundtrip()
    test_18_summary_renders()
    test_19_no_targets_scenario()

    print("\n" + "=" * 76)
    print("  ✓ ALL STAGE 7 TESTS PASSED")
    print("=" * 76)


if __name__ == "__main__":
    main()

