"""
cordis/simulation/result.py
============================
Stage 7 — Per-algorithm aggregation and persistence layer.

Consumes a :class:`RunReport` (flat list of :class:`TrialOutput` from
:class:`MonteCarloRunner`) and produces a :class:`SimResult` that holds:

  * per-algorithm aggregated SINR/SCNR statistics (delegating to the
    existing :func:`aggregate_sinr` / :func:`aggregate_scnr` machinery
    so nothing is duplicated);
  * per-algorithm Monte-Carlo diagnostics (convergence rate, failure
    rate, mean iterations, total / mean runtime, per-AP power ratios);
  * uniform metric-name queries: ``.percentile(alg, metric, pct)``,
    ``.cdf(alg, metric)``, ``.mean(alg, metric)``, ``.std(alg, metric)``.

Persistence
-----------
:meth:`SimResult.save` writes a ``.npz`` file holding every per-trial
array plus a sister ``.json`` sidecar with the configuration, algorithm
specs, metadata, and scalar summaries.  :meth:`SimResult.load` round-trips
the same data back into a :class:`SimResult` with bit-identical numeric
content.

The split between binary (``.npz``) and human-readable (``.json``) is
intentional: the JSON sidecar can be diffed across runs without
touching the heavy array payload.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from cordis.metrics import SCNRStatistics, SINRStatistics
from cordis.metrics.aggregation import aggregate_scnr, aggregate_sinr
from cordis.simulation.runner import RunReport, RunnerConfig
from cordis.simulation.scenario import AlgorithmSpec, TrialOutput
from cordis.utils.config import CORDISConfig, config_from_dict

logger = logging.getLogger("cordis." + __name__)

# Bump this when persisted-format-incompatible changes happen.
SIMRESULT_FORMAT_VERSION = 1


# =============================================================================
# Metric registry  —  uniform name → per-trial sample array
# =============================================================================
#
# Each entry maps a metric name to a "source" and an "accessor".
# Source is one of {"sinr", "scnr", "diag"}; accessor is either a string
# attribute name or a callable taking the source object and returning a
# 1-D numpy array of per-trial samples.
#
# Diagnostics-only metrics (iters, runtime, power_ratio_max) live on
# AlgorithmResult itself, not in the SINR/SCNR stats blocks.

def _safe_db(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-30))


_MetricAccessor = Union[str, Callable[[Any], NDArray[np.float64]]]
_METRIC_REGISTRY: Dict[str, Tuple[str, _MetricAccessor]] = {
    # ── SINR-derived (sourced from SINRStatistics) ──────────────────────
    "min_sinr":             ("sinr", "min_sinr_per_trial"),
    "min_sinr_db":          ("sinr", "min_sinr_per_trial_db"),
    "mean_sinr":            ("sinr", "mean_sinr_per_trial"),
    "mean_sinr_db":         ("sinr", lambda s: _safe_db(s.mean_sinr_per_trial)),
    "sum_rate":             ("sinr", "sum_rate_per_trial"),
    "all_sinr_db_flat":     ("sinr", "all_sinr_db_flat"),

    # ── SCNR-derived (sourced from SCNRStatistics) ──────────────────────
    "weighted_sum_scnr":    ("scnr", "weighted_sum_per_trial"),
    "weighted_sum_scnr_db": ("scnr", "weighted_sum_per_trial_db"),
    "min_scnr":             ("scnr", "min_scnr_per_trial"),
    "min_scnr_db":          ("scnr", "min_scnr_per_trial_db"),
    "mean_scnr":            ("scnr", "mean_scnr_per_trial"),
    "mean_scnr_db":         ("scnr", lambda s: _safe_db(s.mean_scnr_per_trial)),
    "all_scnr_db_flat":     ("scnr", "all_scnr_db_flat"),

    # ── Diagnostics (sourced from AlgorithmResult) ──────────────────────
    "iters":                ("diag", "iters"),
    "runtime_s":            ("diag", "runtime_s"),
    "power_ratio_max":      ("diag", lambda r: r.power_ratios.max(axis=1)),
}


def list_metrics() -> List[str]:
    """List every metric name accepted by :meth:`AlgorithmResult.percentile`,
    :meth:`AlgorithmResult.cdf`, :meth:`AlgorithmResult.mean`, etc."""
    return list(_METRIC_REGISTRY.keys())


# =============================================================================
# AlgorithmResult
# =============================================================================

@dataclass
class AlgorithmResult:
    """
    Aggregated statistics for one algorithm across a full Monte Carlo
    campaign.

    Attributes
    ----------
    spec : AlgorithmSpec
        The spec that produced these results.
    n_trials_total : int
        Number of trials attempted (incl. failures).
    n_succeeded : int
        Number of successful trials.
    n_failed : int
        Number of failed trials.

    sinr_stats : SINRStatistics or None
        Aggregated SINR stats over successful trials.  ``None`` if
        every trial failed.
    scnr_stats : SCNRStatistics or None
        Aggregated SCNR stats.  ``None`` if there are no targets or
        every trial failed.

    iters, runtime_s, converged, power_ratios : np.ndarray
        Per-trial diagnostic arrays of length ``n_succeeded``.
        ``power_ratios`` has shape ``(n_succeeded, n_tx_aps)``.

    convergence_rate, failure_rate, iters_mean, runtime_mean_s,
    runtime_total_s : float
        Derived scalars.
    """

    spec:             AlgorithmSpec
    n_trials_total:   int
    n_succeeded:      int
    n_failed:         int

    sinr_stats:       Optional[SINRStatistics]
    scnr_stats:       Optional[SCNRStatistics]

    # Per-trial diagnostic arrays (length = n_succeeded)
    iters:            NDArray[np.int64]
    runtime_s:        NDArray[np.float64]
    converged:        NDArray[np.bool_]
    power_ratios:     NDArray[np.float64]    # (n_succeeded, n_tx_aps)

    # Derived scalars
    convergence_rate: float
    failure_rate:     float
    iters_mean:       float
    runtime_mean_s:   float
    runtime_total_s:  float

    # ── Metric access helpers ───────────────────────────────────────────

    def _samples(self, metric: str) -> NDArray[np.float64]:
        """Resolve a metric name to its per-trial sample array."""
        if metric not in _METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric '{metric}'.  Available: {list_metrics()}"
            )
        source, accessor = _METRIC_REGISTRY[metric]

        if source == "sinr":
            if self.sinr_stats is None:
                raise ValueError(
                    f"Metric '{metric}' requires SINR statistics, but "
                    f"algorithm '{self.spec.name}' has none "
                    f"(every trial failed)."
                )
            src: Any = self.sinr_stats
        elif source == "scnr":
            if self.scnr_stats is None:
                raise ValueError(
                    f"Metric '{metric}' requires SCNR statistics, but "
                    f"algorithm '{self.spec.name}' has none "
                    f"(no targets, or every trial failed)."
                )
            src = self.scnr_stats
        elif source == "diag":
            src = self
        else:
            raise RuntimeError(f"Internal error: unknown source '{source}'")

        if callable(accessor):
            return np.asarray(accessor(src), dtype=np.float64)
        return np.asarray(getattr(src, accessor), dtype=np.float64)

    def percentile(self, metric: str, pct: float) -> float:
        """Empirical percentile (``pct`` ∈ [0, 100]) of a metric."""
        arr = self._samples(metric)
        return float(np.percentile(arr, pct))

    def cdf(self, metric: str) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return ``(xs, fs)`` such that ``fs[k] = P(X ≤ xs[k])`` — i.e.
        the empirical CDF of the metric.  Both arrays are sorted by
        ``xs`` and have length equal to the number of samples.
        """
        arr = self._samples(metric)
        xs = np.sort(arr)
        n = len(xs)
        if n == 0:
            return xs, np.zeros(0)
        fs = np.arange(1, n + 1, dtype=np.float64) / n
        return xs, fs

    def mean(self, metric: str) -> float:
        """Mean of a metric over successful trials."""
        return float(np.mean(self._samples(metric)))

    def std(self, metric: str) -> float:
        """Standard deviation of a metric over successful trials."""
        return float(np.std(self._samples(metric)))

    def has_metric(self, metric: str) -> bool:
        """True if the metric can be queried (e.g. SCNR-derived needs targets)."""
        if metric not in _METRIC_REGISTRY:
            return False
        source, _ = _METRIC_REGISTRY[metric]
        if source == "sinr": return self.sinr_stats is not None
        if source == "scnr": return self.scnr_stats is not None
        return True

    # ── Infeasibility tracking (Stage 20) ────────────────────────────────

    def infeasibility_rate(
        self,
        gamma_db:    float,
        metric:      str = "min_sinr_db",
    ) -> float:
        """Fraction of successful trials with ``metric < gamma_db``.

        For an SINR-floor constrained algorithm (CORDIS-ADMM, Centralized),
        passing ``metric="min_sinr_db"`` with the experiment's γ tells you
        how often the algorithm failed to meet the per-user SINR floor.
        Returns 0.0 if there are no successful trials.

        Parameters
        ----------
        gamma_db : float
            Threshold below which a trial is counted as infeasible.
            Usually ``cfg.algorithm.gamma_db`` for the experiment.
        metric : str, default "min_sinr_db"
            Which per-trial metric to compare against ``gamma_db``.
            Any metric that ``_samples`` can resolve is allowed.
        """
        if not self.has_metric(metric):
            return 0.0
        arr = self._samples(metric)
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr < gamma_db))

    def feasible_trials_mask(
        self,
        gamma_db:    float,
        metric:      str = "min_sinr_db",
    ) -> NDArray[np.bool_]:
        """Boolean mask of successful trials that satisfy ``metric ≥ gamma_db``.

        Same selection logic as :py:meth:`infeasibility_rate` but returns
        the per-trial mask so callers can plot conditional CDFs etc.
        """
        if not self.has_metric(metric):
            return np.zeros(0, dtype=bool)
        arr = self._samples(metric)
        return arr >= gamma_db

    def cdf_conditional(
        self,
        metric:       str,
        gamma_db:     float,
        cond_metric:  str = "min_sinr_db",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Empirical CDF of ``metric`` restricted to *feasible* trials.

        A trial is feasible iff ``cond_metric ≥ gamma_db``.  Useful for
        plotting "CDF among trials where the SINR floor was met" alongside
        the full CDF — see also :py:meth:`infeasibility_rate` for the
        complementary scalar.
        """
        arr  = self._samples(metric)
        mask = self.feasible_trials_mask(gamma_db, metric=cond_metric)
        if mask.size != arr.size:
            # Different sample counts (shouldn't happen for SINR↔SINR but
            # could in theory for SCNR↔SINR conditioning).  Be safe.
            return np.zeros(0), np.zeros(0)
        xs = np.sort(arr[mask])
        n  = xs.size
        if n == 0:
            return xs, np.zeros(0)
        fs = np.arange(1, n + 1, dtype=np.float64) / n
        return xs, fs


# =============================================================================
# SimResult — top-level Monte Carlo result
# =============================================================================

@dataclass
class SimResult:
    """
    Top-level aggregated Monte Carlo result.

    Holds one :class:`AlgorithmResult` per algorithm name, the original
    :class:`CORDISConfig` and :class:`RunnerConfig`, the list of
    :class:`AlgorithmSpec`, and a metadata dict.  Serialisable to a
    ``.npz`` + ``.json`` pair via :meth:`save`/:meth:`load`.
    """

    cfg:               CORDISConfig
    runner_cfg:        RunnerConfig
    algorithm_specs:   List[AlgorithmSpec]
    algorithm_results: Dict[str, AlgorithmResult]
    metadata:          Dict[str, Any]                 = field(default_factory=dict)

    # ── Convenience queries ─────────────────────────────────────────────

    @property
    def names(self) -> List[str]:
        """List of algorithm names in registration order."""
        return [s.name for s in self.algorithm_specs]

    def __getitem__(self, name: str) -> AlgorithmResult:
        if name not in self.algorithm_results:
            raise KeyError(f"No algorithm '{name}'.  Available: {self.names}")
        return self.algorithm_results[name]

    def __iter__(self):
        return iter(self.algorithm_results.values())

    def __len__(self) -> int:
        return len(self.algorithm_results)

    def percentile(self, alg: str, metric: str, pct: float) -> float:
        return self[alg].percentile(metric, pct)

    def cdf(self, alg: str, metric: str
            ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self[alg].cdf(metric)

    def mean(self, alg: str, metric: str) -> float:
        return self[alg].mean(metric)

    def std(self, alg: str, metric: str) -> float:
        return self[alg].std(metric)

    # ── Factory: aggregate a RunReport ──────────────────────────────────

    @classmethod
    def from_run(cls, report: RunReport) -> "SimResult":
        """
        Aggregate a :class:`RunReport` into a :class:`SimResult`.

        For each algorithm, successful trial outputs are passed through
        :func:`aggregate_sinr` and :func:`aggregate_scnr`; per-trial
        diagnostic arrays (iters, runtime, power) are stacked.
        """
        per_alg: Dict[str, AlgorithmResult] = {}
        for spec in report.algorithm_specs:
            alg_outputs = [o for o in report.outputs if o.name == spec.name]
            per_alg[spec.name] = _aggregate_one_algorithm(spec, alg_outputs)

        sim_meta = dict(report.metadata)
        sim_meta["aggregated_at_iso"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        sim_meta["simresult_format_version"] = SIMRESULT_FORMAT_VERSION

        return cls(
            cfg=report.cfg,
            runner_cfg=report.runner_cfg,
            algorithm_specs=list(report.algorithm_specs),
            algorithm_results=per_alg,
            metadata=sim_meta,
        )

    # ── Persistence: save ───────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> Tuple[Path, Path]:
        """
        Persist the result to ``<path>.npz`` and ``<path>.json``.

        If ``path`` has an extension it is stripped — both files are
        written using the stem.

        Returns
        -------
        (npz_path, json_path)
        """
        p = Path(path).with_suffix("")
        p.parent.mkdir(parents=True, exist_ok=True)
        npz_path  = p.with_suffix(".npz")
        json_path = p.with_suffix(".json")

        arrays: Dict[str, NDArray[Any]] = {}
        scalars: Dict[str, Dict[str, Any]] = {}

        for name, alg_res in self.algorithm_results.items():
            prefix = f"{name}/"
            # Diagnostics
            arrays[f"{prefix}iters"]        = np.asarray(alg_res.iters)
            arrays[f"{prefix}runtime_s"]    = np.asarray(alg_res.runtime_s)
            arrays[f"{prefix}converged"]    = np.asarray(alg_res.converged)
            arrays[f"{prefix}power_ratios"] = np.asarray(alg_res.power_ratios)

            sinr_present = alg_res.sinr_stats is not None
            scnr_present = alg_res.scnr_stats is not None

            if sinr_present:
                _stash_sinr_arrays(alg_res.sinr_stats, prefix, arrays)
            if scnr_present:
                _stash_scnr_arrays(alg_res.scnr_stats, prefix, arrays)

            scalars[name] = {
                "n_trials_total":  int(alg_res.n_trials_total),
                "n_succeeded":     int(alg_res.n_succeeded),
                "n_failed":        int(alg_res.n_failed),
                "convergence_rate":float(alg_res.convergence_rate),
                "failure_rate":    float(alg_res.failure_rate),
                "iters_mean":      float(alg_res.iters_mean),
                "runtime_mean_s":  float(alg_res.runtime_mean_s),
                "runtime_total_s": float(alg_res.runtime_total_s),
                "sinr_present":    bool(sinr_present),
                "scnr_present":    bool(scnr_present),
                "sinr_n_users":    (int(alg_res.sinr_stats.n_users)
                                    if sinr_present else None),
                "sinr_p_noise_mean":(float(alg_res.sinr_stats.p_noise_mean)
                                     if sinr_present else None),
                "scnr_pair_keys":  (list(alg_res.scnr_stats.pair_keys)
                                    if scnr_present else None),
            }

        sidecar = {
            "format_version":         SIMRESULT_FORMAT_VERSION,
            "saved_at_iso":           datetime.now(timezone.utc).isoformat(
                                          timespec="seconds"),
            "cfg":                    self.cfg.to_dict(),
            "runner_cfg":             asdict(self.runner_cfg),
            "algorithm_specs":        [
                {"name": s.name, "kind": s.kind,
                 "params": _to_json_safe(s.params)}
                for s in self.algorithm_specs
            ],
            "metadata":               _to_json_safe(self.metadata),
            "per_algorithm_scalars":  scalars,
        }

        np.savez_compressed(npz_path, **arrays)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

        logger.info("SimResult saved → %s (+ %s)", npz_path, json_path.name)
        return npz_path, json_path

    # ── Persistence: load ───────────────────────────────────────────────

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimResult":
        """
        Load a previously saved SimResult from ``<path>.npz`` + ``<path>.json``.
        """
        p = Path(path).with_suffix("")
        npz_path  = p.with_suffix(".npz")
        json_path = p.with_suffix(".json")
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ payload not found: {npz_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"JSON sidecar not found: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            sidecar = json.load(f)

        fmt = sidecar.get("format_version", -1)
        if fmt != SIMRESULT_FORMAT_VERSION:
            logger.warning(
                "SimResult on disk has format_version=%d; this code expects %d. "
                "Proceeding optimistically.", fmt, SIMRESULT_FORMAT_VERSION,
            )

        npz = np.load(npz_path, allow_pickle=False)

        cfg = config_from_dict(sidecar["cfg"])
        runner_cfg = RunnerConfig(**sidecar["runner_cfg"])
        specs = [
            AlgorithmSpec(name=s["name"], kind=s["kind"], params=dict(s["params"]))
            for s in sidecar["algorithm_specs"]
        ]

        algorithm_results: Dict[str, AlgorithmResult] = {}
        for spec in specs:
            sc = sidecar["per_algorithm_scalars"][spec.name]
            prefix = f"{spec.name}/"

            sinr_stats: Optional[SINRStatistics] = None
            scnr_stats: Optional[SCNRStatistics] = None
            if sc["sinr_present"]:
                sinr_stats = _unstash_sinr_arrays(npz, prefix, sc)
            if sc["scnr_present"]:
                scnr_stats = _unstash_scnr_arrays(npz, prefix, sc)

            algorithm_results[spec.name] = AlgorithmResult(
                spec=spec,
                n_trials_total=int(sc["n_trials_total"]),
                n_succeeded=int(sc["n_succeeded"]),
                n_failed=int(sc["n_failed"]),
                sinr_stats=sinr_stats,
                scnr_stats=scnr_stats,
                iters=np.asarray(npz[f"{prefix}iters"]),
                runtime_s=np.asarray(npz[f"{prefix}runtime_s"]),
                converged=np.asarray(npz[f"{prefix}converged"]),
                power_ratios=np.asarray(npz[f"{prefix}power_ratios"]),
                convergence_rate=float(sc["convergence_rate"]),
                failure_rate=float(sc["failure_rate"]),
                iters_mean=float(sc["iters_mean"]),
                runtime_mean_s=float(sc["runtime_mean_s"]),
                runtime_total_s=float(sc["runtime_total_s"]),
            )

        return cls(
            cfg=cfg, runner_cfg=runner_cfg,
            algorithm_specs=specs,
            algorithm_results=algorithm_results,
            metadata=dict(sidecar["metadata"]),
        )

    # ── Printable summary ───────────────────────────────────────────────

    def summary(self) -> str:
        """Compact comparison table across algorithms."""
        lines: List[str] = []
        lines.append("=" * 92)
        lines.append("  SimResult summary  "
                     f"({self.runner_cfg.n_drops} drops × "
                     f"{self.runner_cfg.n_realizations_per_drop} realizations "
                     f"= {self.runner_cfg.n_trials} trials)")
        lines.append("=" * 92)
        hdr = (f"  {'algorithm':<22}  {'succ':>6}  {'conv%':>6}  "
               f"{'iters':>7}  {'time':>8}  {'min-SINR':>10}  {'W-SCNR':>10}")
        sep = (f"  {'-' * 22}  {'-' * 6}  {'-' * 6}  {'-' * 7}  "
               f"{'-' * 8}  {'-' * 10}  {'-' * 10}")
        lines.append(hdr)
        lines.append(sep)
        for name in self.names:
            r = self[name]
            min_sinr_str = (f"{r.mean('min_sinr_db'):>+8.2f}dB"
                            if r.has_metric("min_sinr_db") else "        —")
            wscnr_str = (f"{r.mean('weighted_sum_scnr_db'):>+8.2f}dB"
                         if r.has_metric("weighted_sum_scnr_db") else "        —")
            lines.append(
                f"  {name:<22}  "
                f"{r.n_succeeded:>3}/{r.n_trials_total:<2}  "
                f"{100.0 * r.convergence_rate:>5.1f}%  "
                f"{r.iters_mean:>7.1f}  "
                f"{r.runtime_mean_s:>7.3f}s  "
                f"{min_sinr_str}  {wscnr_str}"
            )
        lines.append("=" * 92)
        return "\n".join(lines)


# =============================================================================
# Internal helpers
# =============================================================================

def _aggregate_one_algorithm(
    spec:    AlgorithmSpec,
    outputs: List[TrialOutput],
) -> AlgorithmResult:
    """Aggregate every TrialOutput belonging to a single algorithm."""
    n_total = len(outputs)
    successes = [o for o in outputs if not o.failed]
    n_succ = len(successes)
    n_fail = n_total - n_succ

    # Diagnostics arrays
    iters_arr     = np.asarray([o.iters     for o in successes], dtype=np.int64)
    runtime_arr   = np.asarray([o.runtime_s for o in successes], dtype=np.float64)
    converged_arr = np.asarray([o.converged for o in successes], dtype=bool)
    if successes and successes[0].power_ratios is not None:
        power_ratios = np.stack([o.power_ratios for o in successes], axis=0)
    else:
        # Fallback: shape (0, 0) so downstream code can still take .shape[0]
        power_ratios = np.zeros((0, 0), dtype=np.float64)

    # SINR aggregation — guard against zero successes
    sinr_stats: Optional[SINRStatistics] = None
    if n_succ > 0 and successes[0].sinr is not None:
        sinr_stats = aggregate_sinr([o.sinr for o in successes])

    # SCNR aggregation — successes may have None scnr (no targets in scenario)
    scnr_metrics = [o.scnr for o in successes if o.scnr is not None]
    scnr_stats: Optional[SCNRStatistics] = None
    if scnr_metrics:
        scnr_stats = aggregate_scnr(scnr_metrics)

    # Scalars
    if n_succ > 0:
        conv_rate = float(np.mean(converged_arr))
        iters_mean = float(np.mean(iters_arr))
        runtime_mean = float(np.mean(runtime_arr))
        runtime_total = float(np.sum(runtime_arr))
    else:
        conv_rate = 0.0
        iters_mean = float("nan")
        runtime_mean = float("nan")
        runtime_total = 0.0
    fail_rate = float(n_fail) / float(n_total) if n_total > 0 else 0.0

    return AlgorithmResult(
        spec=spec,
        n_trials_total=n_total,
        n_succeeded=n_succ,
        n_failed=n_fail,
        sinr_stats=sinr_stats,
        scnr_stats=scnr_stats,
        iters=iters_arr,
        runtime_s=runtime_arr,
        converged=converged_arr,
        power_ratios=power_ratios,
        convergence_rate=conv_rate,
        failure_rate=fail_rate,
        iters_mean=iters_mean,
        runtime_mean_s=runtime_mean,
        runtime_total_s=runtime_total,
    )


# ── SINRStatistics ↔ npz arrays ─────────────────────────────────────────────

# Field names of SINRStatistics that are numpy arrays.  Scalars
# (n_trials, n_users, p_noise_mean) are stored in the JSON sidecar.
_SINR_ARRAY_FIELDS = (
    "sinr_per_trial_per_user",
    "rate_per_trial_per_user",
    "min_sinr_per_trial",
    "sum_rate_per_trial",
    "mean_sinr_per_trial",
    "ergodic_sinr_per_user",
    "ergodic_rate_per_user",
    "p_cds_mean",
    "p_mui_mean",
    "p_s2ci_mean",
    "p_csi_error_mean",
)


def _stash_sinr_arrays(
    stats:  SINRStatistics,
    prefix: str,
    out:    Dict[str, NDArray[Any]],
) -> None:
    for fname in _SINR_ARRAY_FIELDS:
        out[f"{prefix}sinr_stats/{fname}"] = np.asarray(getattr(stats, fname))


def _unstash_sinr_arrays(
    npz:    "np.lib.npyio.NpzFile",
    prefix: str,
    sc:     Dict[str, Any],
) -> SINRStatistics:
    arr = {f: np.asarray(npz[f"{prefix}sinr_stats/{f}"])
           for f in _SINR_ARRAY_FIELDS}
    n_trials = int(arr["min_sinr_per_trial"].shape[0])
    return SINRStatistics(
        n_trials=n_trials,
        n_users=int(sc["sinr_n_users"]),
        p_noise_mean=float(sc["sinr_p_noise_mean"]),
        **arr,
    )


# ── SCNRStatistics ↔ npz arrays ─────────────────────────────────────────────

_SCNR_ARRAY_FIELDS = (
    "scnr_per_trial_per_pair",
    "weighted_sum_per_trial",
    "min_scnr_per_trial",
    "mean_scnr_per_trial",
    "surrogate_per_trial",
    "ergodic_scnr_per_pair",
)


def _stash_scnr_arrays(
    stats:  SCNRStatistics,
    prefix: str,
    out:    Dict[str, NDArray[Any]],
) -> None:
    for fname in _SCNR_ARRAY_FIELDS:
        out[f"{prefix}scnr_stats/{fname}"] = np.asarray(getattr(stats, fname))


def _unstash_scnr_arrays(
    npz:    "np.lib.npyio.NpzFile",
    prefix: str,
    sc:     Dict[str, Any],
) -> SCNRStatistics:
    arr = {f: np.asarray(npz[f"{prefix}scnr_stats/{f}"])
           for f in _SCNR_ARRAY_FIELDS}
    n_trials = int(arr["weighted_sum_per_trial"].shape[0])
    # JSON serialises tuple-of-int as list-of-list; coerce back.
    pair_keys: List[Tuple[int, int]] = [
        (int(a), int(b)) for a, b in sc["scnr_pair_keys"]
    ]
    return SCNRStatistics(
        n_trials=n_trials,
        pair_keys=pair_keys,
        **arr,
    )


# ── JSON-safety walker ───────────────────────────────────────────────────────

def _to_json_safe(obj: Any) -> Any:
    """
    Recursively convert NumPy scalars, NumPy arrays, and tuples to
    JSON-serialisable plain Python types.

    Tuple becomes list; np.ndarray becomes nested list; np.integer/np.floating/
    np.bool_ become their Python counterparts.  Everything else is passed
    through (so dicts, lists, str, int, float, bool, None remain as-is).
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Anything else (e.g. custom objects in metadata) — best-effort repr.
    return repr(obj)

