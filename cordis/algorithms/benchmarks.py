"""
cordis/algorithms/benchmarks.py
================================
Stage 6d — Baseline algorithms paired with the centralized Phase II
power-splitting allocator.

The journal paper compares CORDIS-Split (LR-MMSE + P-Split optimized
power allocation) and CORDIS-ADMM (joint BF + PA via consensus) against
simpler baselines.  Each baseline either

  (a) substitutes a different communication beamformer (MRT, ZF, RZF,
      global-ZF, global-MRT) into the Stage 6a Phase I + Phase II
      pipeline, isolating the *choice of communication beamformer*, or

  (b) skips the Phase II optimization entirely and uses a uniform
      power-splitting ratio ρ = ρ_fixed across all APs, isolating the
      *value of the PA optimization* itself.

Every benchmark returns a uniform :class:`BenchmarkResult` so the
simulation runner (Stage 7) can enumerate baselines without
special-casing.

Public API
----------
- ``BENCHMARK_REGISTRY``     : ordered registry of all baselines
- ``list_benchmarks()``      : list of registered names
- ``describe_benchmarks()``  : pretty-printed registry
- ``run_benchmark(name, ...)``     : run one baseline by name
- ``run_all_benchmarks(...)``      : convenience sweep
- ``BenchmarkResult``        : uniform output dataclass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from cordis.algorithms.beamforming import (
    COMM_BF_METHODS,
    PhaseIResult,
    design_phase_i,
)
from cordis.algorithms.split_opt import SplitOptResult, run_cordis_split
from cordis.channel.estimation import EstimationResult
from cordis.channel.sensing_assignment import SensingAssociation
from cordis.channel.sensing_channel import SensingChannelStatistics
from cordis.channel.topology import NetworkTopology
from cordis.utils.config import CORDISConfig

logger = logging.getLogger("cordis." + __name__)


# =============================================================================
# Benchmark registry
# =============================================================================
# Each entry: (description, comm_bf_method, pa_optimized, fixed_psr)
#   - pa_optimized=True   → uses solve_p_split (Phase II) to choose ρ_a
#   - pa_optimized=False  → uses uniform ρ = fixed_psr across all APs
#
# Insertion order preserved (Python ≥3.7 dict), which doubles as the
# canonical display order in plots / tables.

BENCHMARK_REGISTRY: Dict[str, Tuple[str, str, bool, Optional[float]]] = {
    # ── BF varied, PA optimised via P-Split ──────────────────────────────
    "lr_mmse_split":    ("LR-MMSE + P-Split PA  (≡ CORDIS-Split)",   "lr_mmse",    True,  None),
    "mrt_split":        ("Local MRT + P-Split PA",                    "mrt",        True,  None),
    "rzf_split":        ("Local RZF + P-Split PA",                    "rzf",        True,  None),
    "zf_split":         ("Local ZF + P-Split PA",                     "zf",         True,  None),
    "global_zf_split":  ("Global ZF + P-Split PA",                    "global_zf",  True,  None),
    "global_mrt_split": ("Global MRT + P-Split PA",                   "global_mrt", True,  None),

    # ── PA fixed (uniform ρ = 0.5; no PA optimisation) ───────────────────
    "lr_mmse_fixed":    ("LR-MMSE + equal ρ=0.5  (no PA opt)",         "lr_mmse",    False, 0.5),
    "mrt_fixed":        ("Local MRT + equal ρ=0.5",                    "mrt",        False, 0.5),
    "rzf_fixed":        ("Local RZF + equal ρ=0.5",                    "rzf",        False, 0.5),
}


def list_benchmarks() -> List[str]:
    """Return list of all registered benchmark names (insertion order)."""
    return list(BENCHMARK_REGISTRY.keys())


def describe_benchmarks() -> str:
    """Pretty-printed table of the registry; suitable for `print()`."""
    width = max(len(n) for n in BENCHMARK_REGISTRY) + 2
    lines = ["Benchmark registry:"]
    lines.append(f"  {'name':<{width}} description")
    lines.append(f"  {'-' * width} {'-' * 50}")
    for name, (desc, _bf, _opt, _psr) in BENCHMARK_REGISTRY.items():
        lines.append(f"  {name:<{width}} {desc}")
    return "\n".join(lines)


# =============================================================================
# Uniform result dataclass
# =============================================================================

@dataclass
class BenchmarkResult:
    """
    Uniform return type of every benchmark in this module.

    Attributes
    ----------
    name : str
        Registry key (e.g. ``"mrt_split"``).
    comm_bf_method : str
        Communication beamformer key (see ``COMM_BF_METHODS``).
    pa_optimized : bool
        ``True`` if Phase II ran (ρ_a chosen by P-Split); ``False`` if a
        uniform ρ was used (no PA optimization).
    psr : dict[int, float] or float
        Actual power-splitting ratio used.  Per-AP dict when
        ``pa_optimized`` is True; scalar float otherwise.
    W_tx : dict[int, np.ndarray (M_t, D)]
        Final beamforming matrices keyed by AP index.
    phase_i : PhaseIResult
        Always present; useful for inspecting the per-AP comm/sens
        precoders before power scaling.
    split_res : SplitOptResult or None
        Phase II diagnostics (objective, slack, ρ); only present when
        ``pa_optimized`` is True.
    """

    name:           str
    comm_bf_method: str
    pa_optimized:   bool
    psr:            Union[Dict[int, float], float]
    W_tx:           Dict[int, NDArray[np.complex128]]
    phase_i:        PhaseIResult
    split_res:      Optional[SplitOptResult] = None


# =============================================================================
# Single-benchmark driver
# =============================================================================

def run_benchmark(
    name:          str,
    topo:          NetworkTopology,
    cfg:           CORDISConfig,
    est:           EstimationResult,
    sensing_stats: SensingChannelStatistics,
    association:   SensingAssociation,
    sigma_n_sq:    float,
    Pmax:          float,
    *,
    gamma_u_db:    Optional[NDArray[np.float64]] = None,
    omega:         Optional[Dict[int, float]] = None,
    use_cvxpy:     Optional[bool] = None,
    rng:           Optional[np.random.Generator] = None,
) -> BenchmarkResult:
    """
    Run a single registered benchmark by ``name``.

    Parameters
    ----------
    name : str
        Must be a key in ``BENCHMARK_REGISTRY``.
    topo, cfg, est, sensing_stats, association :
        Standard pipeline objects (same convention as
        :func:`run_cordis_split` and :func:`run_cordis_admm`).
    sigma_n_sq : float
        Noise variance σ_n² at the receivers.
    Pmax : float
        Per-AP total power budget.
    gamma_u_db : np.ndarray (N_ue,), optional
        Per-user SINR target in dB.  Forwarded to Phase II when
        ``pa_optimized=True``; ignored when ``False`` (no PA target).
    omega : dict[int, float], optional
        Target priority weights ω_t; forwarded to Phase I.
    use_cvxpy : bool, optional
        Force CVXPY (True) or SciPy (False) solver for Phase II.  None →
        let split_opt decide based on availability.
    rng : np.random.Generator, optional
        Required if ``comm_bf_method == "random"``.

    Returns
    -------
    BenchmarkResult
    """
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'. "
            f"Choose from: {list(BENCHMARK_REGISTRY)}"
        )

    _desc, comm_bf, pa_optimized, fixed_psr = BENCHMARK_REGISTRY[name]

    if pa_optimized:
        # Identical pipeline to CORDIS-Split, only swapping the comm BF.
        W_tx, phase_i, split_res = run_cordis_split(
            topo, cfg, est, sensing_stats, association,
            sigma_n_sq, Pmax,
            comm_bf_method=comm_bf,
            gamma_u_db=gamma_u_db,
            omega=omega,
            use_cvxpy=use_cvxpy,
            rng=rng,
        )
        return BenchmarkResult(
            name=name,
            comm_bf_method=comm_bf,
            pa_optimized=True,
            psr=dict(split_res.rho_opt),
            W_tx=W_tx,
            phase_i=phase_i,
            split_res=split_res,
        )

    # Fixed-ρ branch: Phase I only, no Phase II solve.
    if fixed_psr is None:
        raise ValueError(
            f"Benchmark '{name}' is registered with pa_optimized=False but "
            f"no fixed_psr was provided in the registry."
        )

    phase_i = design_phase_i(
        topo, cfg, est, sensing_stats, association,
        comm_bf_method=comm_bf, omega=omega, rng=rng,
    )
    W_tx = phase_i.build_W_tx_equal_psr(float(fixed_psr), Pmax)
    return BenchmarkResult(
        name=name,
        comm_bf_method=comm_bf,
        pa_optimized=False,
        psr=float(fixed_psr),
        W_tx=W_tx,
        phase_i=phase_i,
        split_res=None,
    )


# =============================================================================
# Multi-benchmark sweep
# =============================================================================

def run_all_benchmarks(
    topo:          NetworkTopology,
    cfg:           CORDISConfig,
    est:           EstimationResult,
    sensing_stats: SensingChannelStatistics,
    association:   SensingAssociation,
    sigma_n_sq:    float,
    Pmax:          float,
    *,
    names:         Optional[List[str]] = None,
    gamma_u_db:    Optional[NDArray[np.float64]] = None,
    omega:         Optional[Dict[int, float]] = None,
    use_cvxpy:     Optional[bool] = None,
    rng:           Optional[np.random.Generator] = None,
    skip_failures: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Run several benchmarks on the same scenario.  Convenience for sweep
    and plot code that wants every baseline side-by-side.

    Parameters
    ----------
    names : list of str, optional
        Subset to run.  Default = every registered benchmark.
    skip_failures : bool
        If True, log warnings for benchmarks that raise and continue; if
        False, propagate the first exception.

    Returns
    -------
    dict[name → BenchmarkResult]  (missing entries on failure when
    ``skip_failures=True``).
    """
    if names is None:
        names = list_benchmarks()
    out: Dict[str, BenchmarkResult] = {}
    for name in names:
        try:
            out[name] = run_benchmark(
                name, topo, cfg, est, sensing_stats, association,
                sigma_n_sq, Pmax,
                gamma_u_db=gamma_u_db, omega=omega,
                use_cvxpy=use_cvxpy, rng=rng,
            )
        except Exception as ex:
            if skip_failures:
                logger.warning(
                    "Benchmark '%s' failed: %s", name, str(ex)[:200]
                )
            else:
                raise
    return out

