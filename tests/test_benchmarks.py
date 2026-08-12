"""
tests/test_benchmarks.py
========================
The classical baselines (`cordis/algorithms/benchmarks.py`).

`BENCHMARK_REGISTRY` holds eleven entries. The manuscript shows four curves
overall, two of which come from here: **Global ZF** and **Global MRT**
(`global_zf_split`, `global_mrt_split`), plus **LR-MMSE (rho=0.5)**
(`lr_mmse_fixed`). Fig. 3, Fig. 5 and Fig. 6 all carry them.

The property that decides whether those curves are a fair comparison is the
power budget: `PhaseIResult.build_W_tx` scales each AP by `sqrt(rho Pmax)`,
which delivers `rho Pmax` per AP only when the comm precoder is normalized
**per AP**. The local methods are; the two global ones are normalized across
the concatenation, so they radiate `1/N_tx` of the network budget (F-07-01).

Solver-backed tests are marked `solver` + `slow`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from cordis.algorithms.beamforming import COMM_BF_METHODS, design_phase_i
from cordis.algorithms.benchmarks import (
    BENCHMARK_REGISTRY, BenchmarkResult, describe_benchmarks, list_benchmarks,
    run_benchmark,
)
from cordis.algorithms.split_opt import run_cordis_split
from cordis.metrics.sinr import compute_sinr

pytestmark = [pytest.mark.solver, pytest.mark.slow]

_LOCAL = ("lr_mmse", "mrt", "zf", "rzf")
_GLOBAL = ("global_zf", "global_mrt")


def _min_sinr_db(W, est, topo, sigma) -> float:
    return float(10.0 * np.log10(np.maximum(
        compute_sinr(W, est, topo, sigma).sinr_per_user, 1e-30)).min())


# =============================================================================
# Registry integrity
# =============================================================================

def test_registry_metadata_is_self_consistent() -> None:
    """Every entry names a real BF method and pairs `pa_optimized` with `psr`."""
    for name, (desc, bf, pa_opt, psr) in BENCHMARK_REGISTRY.items():
        assert bf in COMM_BF_METHODS, f"{name}: unknown BF method {bf!r}"
        assert desc.strip()
        if pa_opt:
            assert psr is None, f"{name}: optimized PA must not pin a rho"
        else:
            assert psr is not None and 0.0 <= psr <= 1.0, f"{name}: bad rho"


def test_list_and_describe_agree_with_the_registry() -> None:
    assert list_benchmarks() == list(BENCHMARK_REGISTRY)
    text = describe_benchmarks()
    for name in BENCHMARK_REGISTRY:
        assert name in text


def test_the_paper_facing_names_are_present() -> None:
    """
    The three registry entries the manuscript plots. A rename would silently
    drop a curve from `all_algorithms`.
    """
    for name in ("global_zf_split", "global_mrt_split", "lr_mmse_fixed"):
        assert name in BENCHMARK_REGISTRY


def test_unknown_benchmark_raises(benchmark_scenario) -> None:
    with pytest.raises(ValueError, match="nope"):
        run_benchmark("nope", *benchmark_scenario)


# =============================================================================
# Every entry runs and respects the per-AP power constraint
# =============================================================================

@pytest.mark.parametrize("name", list(BENCHMARK_REGISTRY))
def test_every_benchmark_runs_and_is_power_feasible(
        benchmark_scenario, name) -> None:
    topo, cfg, est, ss, assoc, sigma, Pmax = benchmark_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        br = run_benchmark(name, *benchmark_scenario)
    assert isinstance(br, BenchmarkResult)
    assert br.name == name
    assert set(br.W_tx) == {ap.idx for ap in topo.tx_aps}
    for ap in topo.tx_aps:
        assert np.linalg.norm(br.W_tx[ap.idx], "fro") ** 2 <= Pmax * (1 + 1e-6)


@pytest.mark.parametrize("name,rho", [
    ("lr_mmse_fixed", 0.5), ("lr_mmse_fixed_p020", 0.2),
    ("lr_mmse_fixed_p080", 0.8), ("mrt_fixed", 0.5), ("rzf_fixed", 0.5),
])
def test_fixed_psr_benchmarks_use_the_advertised_ratio(
        benchmark_scenario, name, rho) -> None:
    """
    The comm block carries `rho Pmax` and the sensing block `(1-rho) Pmax`
    at every AP. This is what makes "LR-MMSE (rho=0.5)" mean what the legend
    says.
    """
    topo, cfg, est, ss, assoc, sigma, Pmax = benchmark_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        br = run_benchmark(name, *benchmark_scenario)
    assert br.psr == pytest.approx(rho)
    assert not br.pa_optimized and br.split_res is None
    n_ue = topo.n_ue
    for ap in topo.tx_aps:
        W = br.W_tx[ap.idx]
        comm = np.linalg.norm(W[:, :n_ue], "fro") ** 2 / Pmax
        sens = np.linalg.norm(W[:, n_ue:], "fro") ** 2 / Pmax
        assert comm == pytest.approx(rho, abs=1e-9)
        assert sens == pytest.approx(1.0 - rho, abs=1e-9)


def test_optimized_benchmarks_carry_a_split_result(benchmark_scenario) -> None:
    """
    Note the asymmetry in `BenchmarkResult.psr`: a `float` for the fixed-PSR
    family, but the solved per-AP `dict` for the optimized family. The
    registry's fourth tuple element is `None` for the latter; the result
    object reports what P-Split chose.
    """
    topo = benchmark_scenario[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        br = run_benchmark("zf_split", *benchmark_scenario)
    assert br.pa_optimized
    assert br.split_res is not None
    assert set(br.split_res.rho_opt) == {ap.idx for ap in topo.tx_aps}
    assert isinstance(br.psr, dict)
    assert br.psr == pytest.approx(br.split_res.rho_opt)


# =============================================================================
# lr_mmse_split == CORDIS-Split
# =============================================================================

def test_lr_mmse_split_reproduces_cordis_split(benchmark_scenario) -> None:
    """
    The registry description asserts `lr_mmse_split` is CORDIS-Split
    ("(= CORDIS-Split)"). If it ever drifts, the `psr_baselines` spec set
    would be comparing CORDIS-Split against a near-copy of itself under a
    different name.
    """
    topo, *_ = benchmark_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        br = run_benchmark("lr_mmse_split", *benchmark_scenario)
        W_split, _, _ = run_cordis_split(*benchmark_scenario)
    for ap in topo.tx_aps:
        np.testing.assert_allclose(br.W_tx[ap.idx], W_split[ap.idx],
                                   rtol=1e-9, atol=1e-12)


# =============================================================================
# F-07-01: the global precoders do not use the per-AP power budget
# =============================================================================

@pytest.mark.parametrize("method", _LOCAL)
def test_local_precoders_are_normalized_per_ap(benchmark_scenario,
                                               method) -> None:
    """`||W_c^a||_F = 1` at every AP, so `build_W_tx` delivers `rho Pmax`."""
    topo, cfg, est, ss, assoc, *_ = benchmark_scenario
    p1 = design_phase_i(topo, cfg, est, ss, assoc, comm_bf_method=method)
    for ap in topo.tx_aps:
        assert np.linalg.norm(p1.W_comm_hat[ap.idx], "fro") == pytest.approx(1.0)


@pytest.mark.parametrize("method", _GLOBAL)
def test_global_precoders_are_normalized_across_the_concatenation(
        benchmark_scenario, method) -> None:
    """
    Documents the actual behavior: the slice norms sum to 1 across APs
    instead of being 1 each, so the network radiates `rho Pmax` in total
    rather than `N_tx rho Pmax`.
    """
    topo, cfg, est, ss, assoc, *_ = benchmark_scenario
    p1 = design_phase_i(topo, cfg, est, ss, assoc, comm_bf_method=method)
    sq = np.array([np.linalg.norm(p1.W_comm_hat[ap.idx], "fro") ** 2
                   for ap in topo.tx_aps])
    assert sq.sum() == pytest.approx(1.0)
    assert sq.max() < 1.0


@pytest.mark.xfail(strict=True,
                   reason="F-07-01: global_zf / global_mrt normalize the "
                          "concatenated precoder, so no AP reaches its own "
                          "power limit and the network radiates 1/N_tx of the "
                          "budget the local benchmarks use")
@pytest.mark.parametrize("method", _GLOBAL)
def test_global_precoders_load_at_least_one_ap_to_its_limit(
        benchmark_scenario, method) -> None:
    """
    A per-AP-constrained design should push at least one AP to its
    constraint, otherwise the comparison hands the baseline less power than
    the scheme it is being compared against.
    """
    topo, cfg, est, ss, assoc, sigma, Pmax = benchmark_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        br = run_benchmark(method + "_split", *benchmark_scenario)
    peak = max(np.linalg.norm(br.W_tx[ap.idx], "fro") ** 2 / Pmax
               for ap in topo.tx_aps)
    assert peak == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("method,floor_db", [("global_mrt", 8.0),
                                             ("global_zf", 0.25)])
def test_renormalizing_the_global_precoders_improves_them(
        benchmark_scenario, method, floor_db) -> None:
    """
    Quantifies what F-07-01 costs the baselines, which is what a reviewer
    reimplementing them would see.

    Two natural repairs exist. Scaling by `max_a ||W_c^a||_F` keeps the
    cross-AP amplitude structure that makes global ZF work; normalizing each
    slice separately breaks that structure but fills every AP's budget, which
    is what global MRT needs since its slice norms track path loss and leave
    distant APs radiating almost nothing.

    Measured over six drops at the shipped geometry: global ZF gains about
    3.1 dB under max-slice scaling, global MRT about 14.8 dB under per-AP
    scaling. Per-drop spread is wide for global ZF (0.5 dB on this fixture's
    drop), so its threshold here is only meant to catch the sign, not the
    magnitude. The averaged figures live in
    `.local/verification/stage-07/evidence/global-bf-renorm.txt`.
    """
    topo, cfg, est, ss, assoc, sigma, Pmax = benchmark_scenario
    p1 = design_phase_i(topo, cfg, est, ss, assoc, comm_bf_method=method)
    orig = {ap.idx: p1.W_comm_hat[ap.idx].copy() for ap in topo.tx_aps}
    base = _min_sinr_db(p1.build_W_tx_equal_psr(0.5, Pmax), est, topo, sigma)

    best = base
    scales = {
        "per-ap": {k: np.linalg.norm(v, "fro") for k, v in orig.items()},
        "max-slice": {k: max(np.linalg.norm(w, "fro") for w in orig.values())
                      for k in orig},
    }
    for scale in scales.values():
        p1.W_comm_hat.update({k: v / max(scale[k], 1e-15)
                              for k, v in orig.items()})
        W = p1.build_W_tx_equal_psr(0.5, Pmax)
        for ap in topo.tx_aps:                      # still power feasible
            assert np.linalg.norm(W[ap.idx], "fro") ** 2 <= Pmax * (1 + 1e-6)
        best = max(best, _min_sinr_db(W, est, topo, sigma))

    assert best - base >= floor_db, (
        f"{method}: shipped {base:.2f} dB, best repair {best:.2f} dB")
