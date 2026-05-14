"""
cordis/simulation/scenario.py
==============================
Stage 7 — Single-trial scenario building and algorithm dispatch.

This module exposes three layers:

  ┌──────────────────────────────────────────────────────────────┐
  │  Drop      ← long-term randomness: positions, LSF, sensing  │
  │  └─ Scenario ← realisation-level randomness: H, estimation  │
  │      └─ AlgorithmSpec / dispatch  → TrialOutput             │
  └──────────────────────────────────────────────────────────────┘

A *Drop* captures everything that should stay fixed across multiple
small-scale fading draws (UE/target positions, large-scale fading,
sensing statistics, pilot allocation).  A *Scenario* layers a single
channel realisation and the resulting estimate on top of a Drop.  The
nested (Drop × Realisation) structure matches the ergodic averaging
convention used throughout the journal paper.

The dispatch layer maps an :class:`AlgorithmSpec` ("kind" + params) to
the appropriate algorithm runner, returning a uniform
:class:`TrialOutput` regardless of which algorithm was used.  This is
the single integration point that the Monte Carlo runner (Stage 7,
runner.py) hands off to.

Public API
----------
- ``Drop``, ``Scenario`` — frozen dataclasses
- ``build_drop(cfg, seed_seq, idx)``       → ``Drop``
- ``build_scenario(drop, seed_seq, idx)``  → ``Scenario``
- ``build_scenario_from_seeds(cfg, ...)``  convenience for single-trial code
- ``AlgorithmSpec`` — declarative algorithm config
- ``dispatch_algorithm(scenario, spec)``   → ``(W_tx, extra)``
- ``run_scenario(scenario, specs, ...)``   → ``list[TrialOutput]``
- ``TrialOutput``                          — uniform per-(alg, trial) record
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random import SeedSequence, default_rng
from numpy.typing import NDArray

# ── cordis pipeline ──────────────────────────────────────────────────────────
from cordis.algorithms.benchmarks import BENCHMARK_REGISTRY, run_benchmark
from cordis.algorithms.centralized import run_centralized
from cordis.algorithms.joint_opt import run_cordis_admm
from cordis.algorithms.split_opt import run_cordis_split
from cordis.channel.estimation import (
    EstimationResult, design_pilot_sequences, run_channel_estimation,
)
from cordis.channel.pathloss import (
    LargeScaleFading, compute_large_scale_fading,
    noise_power_watts, snr_to_tx_power,
)
from cordis.channel.rician import (
    ChannelRealization, ChannelStatistics,
    compute_channel_statistics, generate_channel_realization,
)
from cordis.channel.sensing_assignment import SensingAssociation, assign_sensing
from cordis.channel.sensing_channel import (
    SensingChannelStatistics, compute_sensing_statistics,
)
from cordis.channel.topology import NetworkTopology, generate_topology
from cordis.metrics import (
    SCNRMetrics, SINRMetrics, compute_scnr, compute_sinr,
)
from cordis.utils.config import CORDISConfig
from cordis.utils.io_utils import child_rng

logger = logging.getLogger("cordis." + __name__)


# =============================================================================
# Drop  —  long-term ergodic state
# =============================================================================

@dataclass(frozen=True)
class Drop:
    """
    Long-term ergodic state, reused across many realisations.

    Holds everything that is invariant under fresh small-scale fading
    draws: AP/UE/target positions, large-scale fading, channel and
    sensing statistics, sensing assignment, pilot allocation, and the
    derived noise and power scalars.

    Built once via :func:`build_drop`.  All fields are immutable from
    Python's perspective (``frozen=True``); the underlying arrays should
    be treated as read-only.

    Attributes
    ----------
    cfg : CORDISConfig
        Snapshot of the run configuration.
    topo : NetworkTopology
        AP / UE / target geometry.
    lsf : LargeScaleFading
        Path-loss, shadowing, and Rician K-factor per link.
    channel_stats : ChannelStatistics
        Mean / covariance structure used by MMSE estimation.
    sensing_stats : SensingChannelStatistics
        Target and clutter channel statistics.
    association : SensingAssociation
        Static assignment of transmit and receive APs to targets.
    Phi : np.ndarray, shape (n_ue, tau_p), dtype complex
        Pilot matrix.
    contamination_sets : list[list[int]]
        Per-user pilot-contamination index sets ``P_u``.
    sigma_n_sq : float
        Receiver noise variance (Watts).
    Pmax : float
        Per-AP transmit power budget (Watts).
    drop_idx : int
        Sequential index in the Monte-Carlo campaign (0-based).
    drop_seed : int
        Integer reseed used to construct this drop (for traceability).
    """

    cfg:                CORDISConfig
    topo:               NetworkTopology
    lsf:                LargeScaleFading
    channel_stats:      ChannelStatistics
    sensing_stats:      SensingChannelStatistics
    association:        SensingAssociation
    Phi:                NDArray[np.complex128]
    contamination_sets: List[List[int]]
    sigma_n_sq:         float
    Pmax:               float
    drop_idx:           int
    drop_seed:          int


# =============================================================================
# Scenario  —  single (drop, realisation) pair
# =============================================================================

@dataclass(frozen=True)
class Scenario:
    """
    Single realisation within an existing :class:`Drop`.

    Wraps a Drop plus one small-scale fading draw and the resulting
    channel estimate.  This is the unit handed to algorithm runners.

    Attributes
    ----------
    drop : Drop
        Long-term context (positions, LSF, statistics, pilots).
    realization : ChannelRealization
        One draw of small-scale fading.
    estimation : EstimationResult
        Channel estimate based on this realisation.
    realization_idx : int
        Index within the parent drop (0-based).
    realization_seed : int
        Integer reseed used for this realisation.
    """

    drop:             Drop
    realization:      ChannelRealization
    estimation:       EstimationResult
    realization_idx:  int
    realization_seed: int

    # ── Convenience accessors (read-through to drop) ────────────────────────
    @property
    def cfg(self) -> CORDISConfig:               return self.drop.cfg
    @property
    def topo(self) -> NetworkTopology:           return self.drop.topo
    @property
    def lsf(self) -> LargeScaleFading:           return self.drop.lsf
    @property
    def channel_stats(self) -> ChannelStatistics: return self.drop.channel_stats
    @property
    def sensing_stats(self) -> SensingChannelStatistics: return self.drop.sensing_stats
    @property
    def association(self) -> SensingAssociation: return self.drop.association
    @property
    def sigma_n_sq(self) -> float:               return self.drop.sigma_n_sq
    @property
    def Pmax(self) -> float:                     return self.drop.Pmax
    @property
    def drop_idx(self) -> int:                   return self.drop.drop_idx


# =============================================================================
# Drop construction
# =============================================================================

def build_drop(
    cfg:      CORDISConfig,
    seed_seq: Union[SeedSequence, int],
    drop_idx: int = 0,
) -> Drop:
    """
    Build a fresh :class:`Drop` from configuration and a seed source.

    The ``seed_seq`` may be either a :class:`SeedSequence` (preferred —
    pickles cleanly across joblib workers) or a plain integer
    (convenient for one-off scripts).  Internally everything reduces to
    a single :class:`np.random.Generator` from which sub-RNGs are
    spawned by ``child_rng`` for each pipeline step.

    Parameters
    ----------
    cfg : CORDISConfig
    seed_seq : SeedSequence or int
        Seed for the long-term randomness (positions, LSF, sensing
        statistics, pilot allocation).
    drop_idx : int, default 0
        Index used to label the drop in the Monte Carlo campaign.

    Returns
    -------
    Drop
    """
    # Normalise seed input → single integer + Generator
    if isinstance(seed_seq, SeedSequence):
        drop_seed = int(seed_seq.generate_state(1, dtype=np.uint64)[0])
        rng = default_rng(seed_seq)
    else:
        drop_seed = int(seed_seq)
        rng = default_rng(drop_seed)

    n_t = cfg.topology.n_targets

    topo = generate_topology(cfg, child_rng(rng))
    lsf  = compute_large_scale_fading(
        topo, cfg, child_rng(rng), include_targets=(n_t > 0),
    )
    channel_stats = compute_channel_statistics(topo, cfg, lsf, child_rng(rng))
    sensing_stats = compute_sensing_statistics(topo, cfg, lsf, child_rng(rng))
    association   = assign_sensing(topo, cfg, lsf)

    Phi, contam_sets = design_pilot_sequences(
        cfg.topology.n_ue, cfg.channel.tau_p, child_rng(rng),
    )

    sigma_n_sq = noise_power_watts(
        cfg.frequency.bandwidth_hz,
        cfg.channel.noise_figure_db,
        cfg.channel.noise_temp_k,
    )
    Pmax = snr_to_tx_power(cfg.channel.snr_db, sigma_n_sq)

    return Drop(
        cfg=cfg, topo=topo, lsf=lsf,
        channel_stats=channel_stats,
        sensing_stats=sensing_stats,
        association=association,
        Phi=Phi, contamination_sets=contam_sets,
        sigma_n_sq=float(sigma_n_sq), Pmax=float(Pmax),
        drop_idx=int(drop_idx), drop_seed=drop_seed,
    )


# =============================================================================
# Scenario construction
# =============================================================================

def build_scenario(
    drop:            Drop,
    seed_seq:        Union[SeedSequence, int],
    realization_idx: int = 0,
) -> Scenario:
    """
    Layer one channel realisation on top of an existing :class:`Drop`.

    Parameters
    ----------
    drop : Drop
        Pre-built long-term context.
    seed_seq : SeedSequence or int
        Seed for the realisation-level randomness (small-scale fading +
        estimation noise).
    realization_idx : int, default 0
        Index within the parent drop.

    Returns
    -------
    Scenario
    """
    if isinstance(seed_seq, SeedSequence):
        real_seed = int(seed_seq.generate_state(1, dtype=np.uint64)[0])
        rng = default_rng(seed_seq)
    else:
        real_seed = int(seed_seq)
        rng = default_rng(real_seed)

    realization = generate_channel_realization(
        drop.topo, drop.cfg, drop.lsf, drop.channel_stats, child_rng(rng),
    )
    estimation = run_channel_estimation(
        drop.topo, drop.cfg, drop.lsf, drop.channel_stats,
        realization, child_rng(rng),
        drop.Phi, drop.contamination_sets,
    )
    return Scenario(
        drop=drop, realization=realization, estimation=estimation,
        realization_idx=int(realization_idx), realization_seed=real_seed,
    )


def build_scenario_from_seeds(
    cfg:              CORDISConfig,
    drop_seed:        Union[SeedSequence, int],
    realization_seed: Union[SeedSequence, int],
) -> Scenario:
    """
    Convenience: build a fresh Drop + Scenario in one call.

    Useful for one-off scripts where there is no Monte Carlo loop.
    The Drop is discarded once the Scenario is constructed (held only
    via ``Scenario.drop``).
    """
    drop = build_drop(cfg, drop_seed, drop_idx=0)
    return build_scenario(drop, realization_seed, realization_idx=0)


# =============================================================================
# Algorithm specification and dispatch
# =============================================================================

ALGORITHM_KINDS: Tuple[str, ...] = (
    "cordis_split",
    "cordis_admm",
    "centralized",
    "benchmark",
)


@dataclass(frozen=True)
class AlgorithmSpec:
    """
    Declarative spec for one algorithm to run on a Scenario.

    Attributes
    ----------
    name : str
        Display name (used in plots, tables, and SimResult keys).  Must
        be unique within a campaign.
    kind : str
        Algorithm family.  One of :data:`ALGORITHM_KINDS`.
    params : dict
        Algorithm-time kwargs forwarded to the dispatcher.  Common keys:

        - cordis_split   : ``comm_bf_method``, ``gamma_u_db``, ``omega``,
                           ``use_cvxpy``
        - cordis_admm    : ``gamma_u_db``, ``omega``, ``rho_admm``,
                           ``kappa``, ``xi_slack``, ``n_admm_max``
        - centralized    : ``gamma_u_db``, ``omega``, ``use_cvxpy``,
                           ``warm_start``
        - benchmark      : ``benchmark_name`` (REQUIRED), ``gamma_u_db``,
                           ``omega``, ``use_cvxpy``
    """

    name:   str
    kind:   str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ALGORITHM_KINDS:
            raise ValueError(
                f"AlgorithmSpec.kind = '{self.kind}' is not recognised. "
                f"Choose from: {ALGORITHM_KINDS}"
            )
        if self.kind == "benchmark":
            bname = self.params.get("benchmark_name")
            if bname is None:
                raise ValueError(
                    f"AlgorithmSpec(name='{self.name}', kind='benchmark') "
                    f"requires params['benchmark_name']."
                )
            if bname not in BENCHMARK_REGISTRY:
                raise ValueError(
                    f"Unknown benchmark_name '{bname}'. "
                    f"Choose from: {list(BENCHMARK_REGISTRY)}"
                )


# --- per-kind dispatchers ---------------------------------------------------

def _common_kwargs(s: Scenario, p: Dict[str, Any]) -> Dict[str, Any]:
    """Common forwarded kwargs across all kinds."""
    return dict(
        topo=s.topo,
        cfg=s.cfg,
        est=s.estimation,
        sensing_stats=s.sensing_stats,
        association=s.association,
        sigma_n_sq=s.sigma_n_sq,
        Pmax=s.Pmax,
    )


def _dispatch_cordis_split(
    s: Scenario, spec: AlgorithmSpec, rng: Optional[np.random.Generator],
) -> Tuple[Dict[int, NDArray[np.complex128]], Dict[str, Any]]:
    p = spec.params
    W_tx, phase_i, split_res = run_cordis_split(
        **_common_kwargs(s, p),
        comm_bf_method=p.get("comm_bf_method", "lr_mmse"),
        gamma_u_db=p.get("gamma_u_db"),
        omega=p.get("omega"),
        use_cvxpy=p.get("use_cvxpy"),
        rng=rng,
    )
    extra = {
        "converged":  bool(split_res.converged),
        "iters":      int(split_res.n_evals),
        "objective":  float(split_res.objective),
        "rho_opt":    dict(split_res.rho_opt),
        "slack":      [float(v) for v in split_res.slack],
        "solver":     str(split_res.solver),
    }
    return W_tx, extra


def _dispatch_cordis_admm(
    s: Scenario, spec: AlgorithmSpec, rng: Optional[np.random.Generator],
) -> Tuple[Dict[int, NDArray[np.complex128]], Dict[str, Any]]:
    p = spec.params
    # Forward everything that is not None/missing; the algorithm uses
    # cfg defaults for anything omitted.
    fwd_keys = (
        "gamma_u_db", "omega", "rho_admm", "kappa", "xi_slack",
        "slack_tol", "n_admm_max", "eps_pri", "eps_dual",
        "warm_start_from_split",
    )
    kwargs = {k: p[k] for k in fwd_keys if k in p}
    W_tx, admm_res = run_cordis_admm(**_common_kwargs(s, p), **kwargs)
    pri_hist  = admm_res.primal_res_history
    dual_hist = admm_res.dual_res_history
    obj_hist  = admm_res.sensing_obj_history
    extra = {
        "converged":      bool(admm_res.converged),
        "iters":          int(admm_res.n_admm_iters),
        "feasible":       bool(admm_res.feasible),
        "inner_failures": int(admm_res.inner_failures),
        "solver":         str(admm_res.solver),
        "r_pri_final":    float(pri_hist[-1])  if pri_hist  else float("nan"),
        "r_dual_final":   float(dual_hist[-1]) if dual_hist else float("nan"),
        "objective":      float(obj_hist[-1])  if obj_hist  else float("nan"),
    }
    return W_tx, extra


def _dispatch_centralized(
    s: Scenario, spec: AlgorithmSpec, rng: Optional[np.random.Generator],
) -> Tuple[Dict[int, NDArray[np.complex128]], Dict[str, Any]]:
    p = spec.params
    fwd_keys = ("gamma_u_db", "omega", "use_cvxpy", "warm_start")
    kwargs = {k: p[k] for k in fwd_keys if k in p}
    W_tx, cent_res = run_centralized(**_common_kwargs(s, p), **kwargs)
    obj_hist = cent_res.objective_history
    extra = {
        "converged":      bool(cent_res.converged),
        "iters":          int(cent_res.n_sca_iters),
        "feasible":       bool(cent_res.feasible),
        "inner_failures": int(cent_res.inner_failures),
        "solver":         str(cent_res.solver),
        "objective":      float(obj_hist[-1]) if obj_hist else float("nan"),
    }
    return W_tx, extra


def _dispatch_benchmark(
    s: Scenario, spec: AlgorithmSpec, rng: Optional[np.random.Generator],
) -> Tuple[Dict[int, NDArray[np.complex128]], Dict[str, Any]]:
    p = spec.params
    bres = run_benchmark(
        p["benchmark_name"],
        **_common_kwargs(s, p),
        gamma_u_db=p.get("gamma_u_db"),
        omega=p.get("omega"),
        use_cvxpy=p.get("use_cvxpy"),
        rng=rng,
    )
    extra: Dict[str, Any] = {
        "benchmark_name":  p["benchmark_name"],
        "comm_bf_method":  bres.comm_bf_method,
        "pa_optimized":    bres.pa_optimized,
    }
    if bres.split_res is not None:
        extra.update({
            "converged":   bool(bres.split_res.converged),
            "iters":       int(bres.split_res.n_evals),
            "objective":   float(bres.split_res.objective),
            "rho_opt":     dict(bres.split_res.rho_opt),
            "solver":      str(bres.split_res.solver),
        })
    else:
        # Fixed-PSR branch: trivially "converged" since no solve happened.
        extra.update({
            "converged":   True,
            "iters":       0,
            "rho_fixed":   float(bres.psr),
        })
    return bres.W_tx, extra


_ALGORITHM_DISPATCHERS: Dict[
    str,
    Callable[
        [Scenario, AlgorithmSpec, Optional[np.random.Generator]],
        Tuple[Dict[int, NDArray[np.complex128]], Dict[str, Any]],
    ],
] = {
    "cordis_split": _dispatch_cordis_split,
    "cordis_admm":  _dispatch_cordis_admm,
    "centralized":  _dispatch_centralized,
    "benchmark":    _dispatch_benchmark,
}


def dispatch_algorithm(
    scenario: Scenario,
    spec:     AlgorithmSpec,
    rng:      Optional[np.random.Generator] = None,
) -> Tuple[Dict[int, NDArray[np.complex128]], Dict[str, Any]]:
    """
    Run a single algorithm on a scenario via the registered dispatcher.

    Returns
    -------
    W_tx  : dict[ap_idx → (M_t, D)]  final beamformers
    extra : dict  algorithm-specific diagnostics (converged, iters, ...)
    """
    fn = _ALGORITHM_DISPATCHERS[spec.kind]
    return fn(scenario, spec, rng)


# =============================================================================
# TrialOutput  —  single (algorithm, scenario) result
# =============================================================================

@dataclass
class TrialOutput:
    """
    Picklable, lightweight per-(algorithm, scenario) record.

    Carries enough information to compute aggregate Monte Carlo
    statistics later, without retaining the heavy beamforming matrices
    or channel objects.  Failed trials are recorded with ``failed=True``
    and metric fields set to ``None`` — the aggregator filters them out
    per-algorithm before calling :func:`aggregate_sinr` /
    :func:`aggregate_scnr`.
    """

    name:             str
    drop_idx:         int
    realization_idx:  int
    drop_seed:        int
    realization_seed: int

    # ── Success / failure ───────────────────────────────────────────────
    failed:  bool
    error:   Optional[str]

    # ── Metrics (None when failed) ──────────────────────────────────────
    sinr:    Optional[SINRMetrics]
    scnr:    Optional[SCNRMetrics]

    # ── Diagnostics ─────────────────────────────────────────────────────
    converged:    bool
    iters:        int
    runtime_s:    float
    power_ratios: Optional[NDArray[np.float64]]    # (n_tx_aps,)
    extra:        Dict[str, Any]


def _power_ratios(
    W_tx: Dict[int, NDArray[np.complex128]],
    topo: NetworkTopology,
    Pmax: float,
) -> NDArray[np.float64]:
    return np.array([
        float(np.linalg.norm(W_tx[ap.idx], "fro") ** 2) / Pmax
        for ap in topo.tx_aps
    ])


# =============================================================================
# Top-level entry point: run all algorithms on one Scenario
# =============================================================================

def run_scenario(
    scenario:      Scenario,
    specs:         List[AlgorithmSpec],
    *,
    skip_failures: bool = True,
    rng:           Optional[np.random.Generator] = None,
) -> List[TrialOutput]:
    """
    Execute every algorithm in ``specs`` on the same :class:`Scenario`.

    Each algorithm runs sequentially within this call; parallelism
    over scenarios is handled by the Monte Carlo runner.  Same-scenario
    execution is what makes the resulting Monte Carlo curves a *paired*
    head-to-head comparison (identical channel realisations across
    algorithms in a given trial), which is the standard convention for
    the journal-paper plots.

    Parameters
    ----------
    scenario : Scenario
    specs    : list of AlgorithmSpec
    skip_failures : bool
        If True, log a warning on algorithm failure and emit a
        ``failed=True`` TrialOutput.  If False, propagate the exception.
    rng : np.random.Generator, optional
        Forwarded to algorithm runners (used by e.g. ``random`` comm-BF).

    Returns
    -------
    list[TrialOutput]  in the same order as ``specs``.
    """
    outputs: List[TrialOutput] = []
    for spec in specs:
        t0 = time.perf_counter()
        try:
            W_tx, extra = dispatch_algorithm(scenario, spec, rng=rng)
            runtime = time.perf_counter() - t0

            sinr_m = compute_sinr(
                W_tx, scenario.estimation, scenario.topo, scenario.sigma_n_sq,
            )
            scnr_m: Optional[SCNRMetrics] = None
            if scenario.topo.n_targets > 0:
                scnr_m = compute_scnr(
                    scenario.topo, scenario.cfg, W_tx,
                    scenario.sensing_stats, scenario.association,
                    scenario.sigma_n_sq,
                )
            outputs.append(TrialOutput(
                name=spec.name,
                drop_idx=scenario.drop_idx,
                realization_idx=scenario.realization_idx,
                drop_seed=scenario.drop.drop_seed,
                realization_seed=scenario.realization_seed,
                failed=False, error=None,
                sinr=sinr_m, scnr=scnr_m,
                converged=bool(extra.get("converged", True)),
                iters=int(extra.get("iters", 1)),
                runtime_s=float(runtime),
                power_ratios=_power_ratios(W_tx, scenario.topo, scenario.Pmax),
                extra=extra,
            ))
        except Exception as ex:    # noqa: BLE001 — broad on purpose
            runtime = time.perf_counter() - t0
            if not skip_failures:
                raise
            msg = f"{type(ex).__name__}: {ex}"
            logger.warning(
                "Algorithm '%s' failed on trial (drop=%d, real=%d): %s",
                spec.name, scenario.drop_idx, scenario.realization_idx,
                msg[:200],
            )
            outputs.append(TrialOutput(
                name=spec.name,
                drop_idx=scenario.drop_idx,
                realization_idx=scenario.realization_idx,
                drop_seed=scenario.drop.drop_seed,
                realization_seed=scenario.realization_seed,
                failed=True, error=msg,
                sinr=None, scnr=None,
                converged=False, iters=0,
                runtime_s=float(runtime),
                power_ratios=None,
                extra={},
            ))
    return outputs

