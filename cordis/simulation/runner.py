"""
cordis/simulation/runner.py
============================
Stage 7 — Monte Carlo runner for cell-free ISAC simulations.

This module orchestrates a campaign of ``n_drops × n_realizations_per_drop``
trials, running every algorithm in a supplied :class:`AlgorithmSpec` list
on every (drop, realization) pair.  The nested structure matches the
ergodic-averaging convention used in the journal paper:

  * a *drop* fixes UE / target positions, large-scale fading, and pilot
    allocation (the "long-term" randomness);
  * a *realization* layers a single small-scale fading draw and the
    resulting channel estimate on top of a drop (the "short-term"
    randomness).

Setting ``n_realizations_per_drop = 1`` collapses to pure drop
randomisation; setting ``n_drops = 1`` collapses to pure realisation
randomisation.

Parallelism
-----------
The work unit handed to :mod:`joblib` is **one drop with all its
realisations**.  This amortises drop construction across the inner loop
(positions, LSF, and sensing statistics are built once per worker invocation)
and matches the natural granularity of the campaign.

Seeding
-------
Reproducibility is anchored at a single ``base_seed``.  Hierarchical
:class:`numpy.random.SeedSequence` spawning is then used to derive
statistically independent streams:

    master = SeedSequence(base_seed)
    drop_ss[d]            = master.spawn(n_drops)[d]
    realization_ss[d][r]  = drop_ss[d].spawn(n_realizations)[r]

Each worker only receives :class:`SeedSequence` objects, which pickle
cleanly across process boundaries and never share state.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from numpy.random import SeedSequence

from cordis.simulation.scenario import (
    AlgorithmSpec, TrialOutput, build_drop, build_scenario, run_scenario,
)
from cordis.utils.config import CORDISConfig

logger = logging.getLogger("cordis." + __name__)


# =============================================================================
#  Progress-bar helpers
# =============================================================================

# =============================================================================
#  Progress-bar helpers
# =============================================================================
#
# Architecture (per-trial granularity):
#
#   • The Parallel work unit is still "one drop with all its realizations"
#     (preserves the build_drop optimisation — we only build each drop once).
#   • We create a shared counter via ``multiprocessing.Manager``.
#   • That counter is passed to every worker as a kwarg of ``_run_one_drop``.
#   • Workers increment the counter after EACH realization finishes, mid-drop.
#   • A background polling thread on the parent side reads the counter
#     ~10× per second and advances the tqdm bar.
#
# This gives genuine per-trial bar updates without restructuring the
# parallelism.  Counter increments are not strictly atomic across
# workers, but the worst case is one missed tick per concurrent
# increment — negligible at the granularity we care about, and the
# campaign-level total is computed independently in the report.


@contextmanager
def _nullcontext():
    yield None


def _make_progress_context(rc: "RunnerConfig"):
    """
    Pick the right progress context for a campaign.

    Returns a triple::

        (context_manager, joblib_verbose_override, progress_counter)

    ``progress_counter`` is a multiprocessing-shared counter (or ``None``).
    Workers call ``counter.value += 1`` after each realization completes;
    a polling thread on the parent advances tqdm from that counter.

    Three fallback tiers:

    1. ``rc.progress`` False, or stderr is not a TTY → no progress display.
    2. ``tqdm`` installed → counter-driven per-trial bar.
    3. ``tqdm`` missing → joblib's own ``verbose=10`` line output (no per-trial
       granularity, but at least *something*).
    """
    if not rc.progress or not sys.stderr.isatty():
        return _nullcontext(), rc.verbose, None

    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.info("tqdm not installed — falling back to joblib verbose=10. "
                    "Install with `pip install tqdm` for a per-trial bar.")
        return _nullcontext(), max(10, rc.verbose), None

    import multiprocessing
    import threading
    import time as _time

    bar = tqdm(
        total=rc.n_trials,
        desc=f"trials ({rc.n_drops}d × {rc.n_realizations_per_drop}r)",
        unit="trial",
        leave=True,
    )

    # Shared counter — Manager proxies pickle cleanly across spawn/fork.
    # The lock makes ``counter.value += 1`` atomic across workers
    # (a Manager.Value alone is racy on read-modify-write, which loses
    # ~20-30% of ticks under contention).  Creation spawns one extra
    # (small, idle) Python process; freed when the with-block exits.
    manager = multiprocessing.Manager()
    counter = manager.Value("i", 0)
    counter_lock = manager.Lock()

    stop_event = threading.Event()

    def _poll():
        last = 0
        while not stop_event.wait(0.1):
            try:
                current = counter.value
            except Exception:
                continue   # manager momentarily unreachable; try again
            if current > last:
                bar.update(current - last)
                last = current
        # Final flush after the workers are done.
        try:
            current = counter.value
            if current > last:
                bar.update(current - last)
        except Exception:
            pass

    @contextmanager
    def _ctx():
        thread = threading.Thread(target=_poll, name="tqdm-poller", daemon=True)
        thread.start()
        try:
            yield (counter, counter_lock)
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
            bar.close()
            # The Manager process is shut down when `manager` is garbage
            # collected; leaving the with-block drops the last reference.

    return _ctx(), 0, (counter, counter_lock)


# ─── Worker-side logging suppression ────────────────────────────────────
#
# joblib workers run in fresh processes (spawn on macOS, fork on Linux).
# They never see the parent's `setup_logging`, so when worker code calls
# `logger.warning(...)`, Python's default lastResort handler kicks in
# and dumps the bare message to stderr without any level prefix.  That's
# what produces lines like
#
#     CVXPY inner solve [SCS] status=infeasible — returning previous W
#     β̂ has negative real parts — applying phase alignment
#
# during a parallel campaign.  These are diagnostic messages for known
# difficult conditions, not actionable per-trial.
#
# We install a worker-side handler that filters out WARNINGs from
# ``cordis.algorithms.*`` / ``cordis.channel.*`` (the modules that emit
# these per-trial diagnostics) but still surfaces genuine ERRORs and
# WARNINGs from other modules.  Users who need full algorithm-internal
# logging should run with ``--n-workers 1`` (everything in the parent
# process, full logging chain configured) plus ``--show-algorithm-warnings``.

_WORKER_LOG_INIT_DONE = False

_WORKER_SUPPRESS_PREFIXES = (
    "cordis.algorithms.",
    "cordis.channel.",
)


class _WorkerDiagnosticFilter(logging.Filter):
    """Drop WARNING-level diagnostic chatter from per-trial algorithm code."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True  # never silence real errors
        if any(record.name.startswith(p) for p in _WORKER_SUPPRESS_PREFIXES):
            return False
        return True


def _ensure_worker_log_quiet() -> None:
    """Install a filtering stderr handler in worker processes on first call.

    Idempotent.  Only acts if no real handlers are attached to the root
    logger (i.e. we're in a fresh worker process, not a single-process
    run where the parent already configured logging).
    """
    global _WORKER_LOG_INIT_DONE
    if _WORKER_LOG_INIT_DONE:
        return
    root = logging.getLogger()
    has_real_handlers = any(
        not isinstance(h, logging.NullHandler) for h in root.handlers
    )
    if not has_real_handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        handler.addFilter(_WorkerDiagnosticFilter())
        root.addHandler(handler)
    _WORKER_LOG_INIT_DONE = True


# =============================================================================
# Runner configuration
# =============================================================================

@dataclass
class RunnerConfig:
    """
    Configuration for a Monte Carlo campaign.

    Total number of trials = ``n_drops × n_realizations_per_drop``.  Each
    trial runs every algorithm in the spec list, so the total number of
    algorithm invocations is ``n_trials × n_algorithms``.

    Attributes
    ----------
    n_drops : int
        Number of long-term geometry/LSF draws (outer ergodic dimension).
    n_realizations_per_drop : int
        Number of small-scale fading draws per drop (inner ergodic
        dimension).
    base_seed : int
        Master seed; the entire campaign is reproducible from this scalar.
    n_workers : int
        joblib ``n_jobs`` — number of parallel processes.  ``-1`` uses
        all available cores; ``1`` runs sequentially in the main process
        (useful for debugging).
    backend : str
        joblib backend.  Default ``"loky"`` is robust on Linux and macOS;
        use ``"threading"`` for I/O-bound work or if CVXPY's solver
        already releases the GIL.
    verbose : int
        joblib verbosity (0 = silent, 1 = progress every few tasks,
        10+ = detail per task).
    progress : bool
        If True (default), show a per-drop progress bar on the console.
        Uses ``tqdm`` if installed; falls back to joblib's verbose=10
        line output otherwise.  Auto-disabled when stderr is not a TTY
        (e.g. on SLURM, in CI) so log files stay clean.
    skip_failures : bool
        If True, a failing algorithm on a given trial emits a
        ``failed=True`` TrialOutput rather than killing the campaign.
    """

    n_drops:                 int  = 10
    n_realizations_per_drop: int  = 10
    base_seed:               int  = 42
    n_workers:               int  = -1
    backend:                 str  = "loky"
    verbose:                 int  = 1
    progress:                bool = True
    skip_failures:           bool = True

    @property
    def n_trials(self) -> int:
        """Total number of (drop, realization) pairs."""
        return self.n_drops * self.n_realizations_per_drop

    def validate(self) -> None:
        if self.n_drops < 1:
            raise ValueError(f"n_drops must be ≥ 1, got {self.n_drops}")
        if self.n_realizations_per_drop < 1:
            raise ValueError(
                f"n_realizations_per_drop must be ≥ 1, got "
                f"{self.n_realizations_per_drop}"
            )
        if self.n_workers == 0:
            raise ValueError(
                "n_workers cannot be 0; use 1 for sequential or -1 for all cores"
            )


# =============================================================================
# RunReport — runner output
# =============================================================================

@dataclass
class RunReport:
    """
    Container for the flattened campaign output, returned by
    :meth:`MonteCarloRunner.run`.

    The aggregation into :class:`SINRStatistics` / :class:`SCNRStatistics`
    and the resulting :class:`SimResult` is the job of ``result.py``,
    which is built on top of this container.

    Attributes
    ----------
    cfg : CORDISConfig
        Configuration used for the campaign.
    runner_cfg : RunnerConfig
        Runner configuration used.
    algorithm_specs : list[AlgorithmSpec]
        Algorithm specs run in the campaign.
    outputs : list[TrialOutput]
        Flat list of all (algorithm × drop × realization) outputs.
        Length ≤ ``n_algorithms × n_drops × n_realizations_per_drop``
        (equal if no failures cause early drop bail-out; see
        :func:`_run_one_drop` for the failure handling contract).
    metadata : dict
        Run metadata: timestamps, host info, library versions,
        success/failure counts.
    """

    cfg:             CORDISConfig
    runner_cfg:      RunnerConfig
    algorithm_specs: List[AlgorithmSpec]
    outputs:         List[TrialOutput]
    metadata:        Dict[str, Any] = field(default_factory=dict)

    # ── Convenience queries ──────────────────────────────────────────────
    @property
    def n_total(self) -> int:
        return len(self.outputs)

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.outputs if o.failed)

    @property
    def n_succeeded(self) -> int:
        return self.n_total - self.n_failed

    def outputs_for(self, name: str) -> List[TrialOutput]:
        """Return all TrialOutputs for one algorithm name."""
        return [o for o in self.outputs if o.name == name]

    def failure_summary(self) -> Dict[str, int]:
        """Failure counts keyed by algorithm name."""
        out: Dict[str, int] = {s.name: 0 for s in self.algorithm_specs}
        for o in self.outputs:
            if o.failed:
                out[o.name] = out.get(o.name, 0) + 1
        return out


# =============================================================================
# Worker entry point — runs one drop and all its realizations
# =============================================================================

def _failed_output(
    spec_name:        str,
    drop_idx:         int,
    realization_idx:  int,
    drop_seed:        int,
    realization_seed: int,
    error_msg:        str,
) -> TrialOutput:
    """Construct a TrialOutput for a build-level failure (drop or scenario)."""
    return TrialOutput(
        name=spec_name,
        drop_idx=drop_idx,
        realization_idx=realization_idx,
        drop_seed=drop_seed,
        realization_seed=realization_seed,
        failed=True,
        error=error_msg,
        sinr=None, scnr=None,
        converged=False, iters=0, runtime_s=0.0,
        power_ratios=None, extra={},
    )


def _run_one_drop(
    cfg:                   CORDISConfig,
    algorithm_specs:       List[AlgorithmSpec],
    drop_seed_seq:         SeedSequence,
    realization_seed_seqs: List[SeedSequence],
    drop_idx:              int,
    skip_failures:         bool,
    progress_counter:      Optional[Any] = None,
) -> List[TrialOutput]:
    """
    Worker entry point: build one Drop and run every algorithm on every
    realisation under it.

    ``progress_counter``, if provided, is a ``(value_proxy, lock_proxy)``
    pair from ``multiprocessing.Manager``.  The runner increments the
    value (under the lock, for atomicity across workers) once per
    realization so the parent's tqdm bar can advance per trial rather
    than per drop.  Increments happen on the success path, the
    scenario-build failure path, AND the drop-build failure path, so
    the bar always reaches its declared total of ``n_drops × n_realizations``.

    Failure handling
    ----------------
    * If :func:`build_drop` itself raises, every (algorithm × realization)
      slot under this drop is filled with a ``failed=True`` TrialOutput
      (so the runner's accounting stays consistent).
    * If :func:`build_scenario` raises for a particular realisation,
      every algorithm slot for that realisation is filled with a
      ``failed=True`` TrialOutput.
    * If an algorithm raises during :func:`run_scenario`, the behaviour
      is governed by ``skip_failures`` (handled inside ``run_scenario``).

    Notes
    -----
    SeedSequences are passed directly to ``build_drop`` / ``build_scenario``;
    the integer seeds stored in ``Drop.drop_seed`` and
    ``Scenario.realization_seed`` are read *back* after a successful build.
    This preserves the SeedSequence-pathed RNG stream (which differs from
    ``default_rng(integer)``) and keeps the runner's traceability fields
    in lock-step with scenario.py's own seed bookkeeping.
    """
    # Suppress lastResort StreamHandler in worker processes.  Idempotent,
    # ~free after the first call.  See _ensure_worker_log_quiet docstring.
    _ensure_worker_log_quiet()

    # ── Progress-counter helper ───────────────────────────────────────
    # Called after every realization completes (any path).  Survives all
    # transient errors (manager unreachable, counter pickling glitches);
    # a missed tick is cosmetic, not a correctness issue.  ``progress_counter``
    # is a ``(value_proxy, lock_proxy)`` pair; the lock makes the read-modify-
    # write atomic across workers.
    def _tick():
        if progress_counter is None:
            return
        try:
            counter_val, counter_lock = progress_counter
            with counter_lock:
                counter_val.value += 1
        except Exception:
            pass

    # ── Drop construction ──────────────────────────────────────────────
    try:
        drop = build_drop(cfg, drop_seed_seq, drop_idx=drop_idx)
    except Exception as ex:  # noqa: BLE001
        msg = f"build_drop failed: {type(ex).__name__}: {ex}"
        logger.warning("Drop %d build failed: %s", drop_idx, msg[:200])
        # For failure-output traceability, derive the integers that
        # build_drop / build_scenario WOULD have stored.  generate_state
        # is deterministic and does not advance the SeedSequence's pool.
        drop_seed_for_log = int(drop_seed_seq.generate_state(1, dtype=np.uint64)[0])
        realization_seeds_for_log = [
            int(rss.generate_state(1, dtype=np.uint64)[0])
            for rss in realization_seed_seqs
        ]
        # Tick once per realization so the bar still completes.
        for _ in realization_seed_seqs:
            _tick()
        return [
            _failed_output(
                spec.name, drop_idx, r_idx,
                drop_seed_for_log, realization_seeds_for_log[r_idx], msg,
            )
            for r_idx in range(len(realization_seed_seqs))
            for spec in algorithm_specs
        ]

    # ── Realization loop ───────────────────────────────────────────────
    outputs: List[TrialOutput] = []
    for r_idx, real_ss in enumerate(realization_seed_seqs):
        try:
            scenario = build_scenario(drop, real_ss, realization_idx=r_idx)
        except Exception as ex:  # noqa: BLE001
            msg = f"build_scenario failed: {type(ex).__name__}: {ex}"
            logger.warning(
                "Scenario (drop=%d, real=%d) build failed: %s",
                drop_idx, r_idx, msg[:200],
            )
            real_seed_for_log = int(
                real_ss.generate_state(1, dtype=np.uint64)[0]
            )
            outputs.extend(
                _failed_output(
                    spec.name, drop_idx, r_idx,
                    drop.drop_seed, real_seed_for_log, msg,
                )
                for spec in algorithm_specs
            )
            _tick()
            continue

        outputs.extend(
            run_scenario(
                scenario, algorithm_specs,
                skip_failures=skip_failures,
            )
        )
        _tick()

    return outputs


# =============================================================================
# MonteCarloRunner
# =============================================================================

class MonteCarloRunner:
    """
    Monte Carlo campaign driver.

    Parameters
    ----------
    cfg : CORDISConfig
        Configuration shared across all trials.
    algorithm_specs : Iterable[AlgorithmSpec]
        Algorithms to run on each scenario.  Names must be unique
        (they become keys in :class:`RunReport` queries and
        :class:`SimResult` later).
    runner_cfg : RunnerConfig, optional
        Defaults to ``RunnerConfig()`` (10 drops × 10 realizations).

    Example
    -------
    >>> from cordis.simulation.runner import MonteCarloRunner, RunnerConfig
    >>> from cordis.simulation.scenario import AlgorithmSpec
    >>> specs = [
    ...     AlgorithmSpec("CORDIS-Split", "cordis_split",
    ...                   {"gamma_u_db": 3.0}),
    ...     AlgorithmSpec("CORDIS-ADMM",  "cordis_admm",
    ...                   {"gamma_u_db": 3.0}),
    ...     AlgorithmSpec("Centralized",  "centralized",
    ...                   {"gamma_u_db": 3.0}),
    ... ]
    >>> runner = MonteCarloRunner(
    ...     cfg, specs,
    ...     RunnerConfig(n_drops=20, n_realizations_per_drop=10,
    ...                  n_workers=-1, base_seed=42),
    ... )
    >>> report = runner.run()
    >>> report.n_succeeded, report.n_failed
    """

    def __init__(
        self,
        cfg:             CORDISConfig,
        algorithm_specs: Iterable[AlgorithmSpec],
        runner_cfg:      Optional[RunnerConfig] = None,
    ) -> None:
        self.cfg = cfg
        self.algorithm_specs: List[AlgorithmSpec] = list(algorithm_specs)
        self.runner_cfg = runner_cfg or RunnerConfig()
        self.runner_cfg.validate()

        if not self.algorithm_specs:
            raise ValueError("At least one AlgorithmSpec is required.")

        names = [s.name for s in self.algorithm_specs]
        if len(set(names)) != len(names):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"AlgorithmSpec names must be unique; duplicates: {set(dupes)}"
            )

    # ── Seed sequence construction ──────────────────────────────────────
    def _spawn_seed_sequences(
        self,
    ) -> List[List[SeedSequence]]:
        """
        Build a hierarchical seed tree:

            realization_ss[d][r] = master.spawn(n_drops)[d].spawn(n_realizations)[r]

        Returns
        -------
        list[list[SeedSequence]]  shape (n_drops, n_realizations_per_drop)
        """
        rc = self.runner_cfg
        master = SeedSequence(rc.base_seed)
        drop_seqs = master.spawn(rc.n_drops)
        return [list(ds.spawn(rc.n_realizations_per_drop)) for ds in drop_seqs]

    # ── Main entry ──────────────────────────────────────────────────────
    def run(self) -> RunReport:
        """
        Execute the full campaign and return a :class:`RunReport`.
        """
        rc = self.runner_cfg

        # Lazy import: keeps joblib optional for users running n_workers=1
        # via a dedicated `run_sequential()` path could be added later.
        try:
            from joblib import Parallel, delayed
        except ImportError as ex:
            raise ImportError(
                "joblib is required for MonteCarloRunner; install with "
                "`pip install joblib`."
            ) from ex

        # Hierarchical seed tree
        master = SeedSequence(rc.base_seed)
        drop_seqs = master.spawn(rc.n_drops)
        realization_seqs = [
            list(ds.spawn(rc.n_realizations_per_drop)) for ds in drop_seqs
        ]

        logger.info(
            "Starting Monte Carlo campaign: %d drops × %d realizations = %d trials "
            "× %d algorithms = %d algorithm invocations  (n_workers=%s)",
            rc.n_drops, rc.n_realizations_per_drop, rc.n_trials,
            len(self.algorithm_specs),
            rc.n_trials * len(self.algorithm_specs),
            rc.n_workers,
        )

        t_start_perf = time.perf_counter()
        t_start_iso  = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Parallelise over drops, with optional per-trial tqdm bar.
        progress_ctx, joblib_verbose, progress_counter = \
            _make_progress_context(rc)
        with progress_ctx:
            per_drop_outputs: List[List[TrialOutput]] = Parallel(
                n_jobs=rc.n_workers,
                backend=rc.backend,
                verbose=joblib_verbose,
            )(
                delayed(_run_one_drop)(
                    self.cfg, self.algorithm_specs,
                    drop_seqs[d], realization_seqs[d], d,
                    rc.skip_failures,
                    progress_counter=progress_counter,
                )
                for d in range(rc.n_drops)
            )

        # Flatten
        outputs: List[TrialOutput] = [
            o for drop_outs in per_drop_outputs for o in drop_outs
        ]

        runtime = time.perf_counter() - t_start_perf
        t_end_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        report = RunReport(
            cfg=self.cfg,
            runner_cfg=self.runner_cfg,
            algorithm_specs=self.algorithm_specs,
            outputs=outputs,
            metadata=self._build_metadata(t_start_iso, t_end_iso, runtime, outputs),
        )

        logger.info(
            "Campaign complete in %.1f s — %d/%d outputs succeeded "
            "(%d failed across %d algorithms)",
            runtime, report.n_succeeded, report.n_total,
            report.n_failed, len(self.algorithm_specs),
        )
        return report

    # ── Metadata ────────────────────────────────────────────────────────
    def _build_metadata(
        self,
        t_start_iso: str,
        t_end_iso:   str,
        runtime_s:   float,
        outputs:     List[TrialOutput],
    ) -> Dict[str, Any]:
        rc = self.runner_cfg
        meta: Dict[str, Any] = {
            "start_time_iso":         t_start_iso,
            "end_time_iso":           t_end_iso,
            "total_runtime_s":        float(runtime_s),
            "n_drops":                rc.n_drops,
            "n_realizations_per_drop":rc.n_realizations_per_drop,
            "n_trials_total":         rc.n_trials,
            "n_algorithms":           len(self.algorithm_specs),
            "n_outputs_total":        len(outputs),
            "n_failed":               sum(1 for o in outputs if o.failed),
            "base_seed":              rc.base_seed,
            "n_workers":              rc.n_workers,
            "backend":                rc.backend,
            "host":                   platform.node(),
            "python_version":         sys.version.split()[0],
            "numpy_version":          np.__version__,
            "platform":               platform.platform(),
        }
        # Optional: per-algorithm cumulative runtime
        per_alg_runtime: Dict[str, float] = {s.name: 0.0 for s in self.algorithm_specs}
        for o in outputs:
            if not o.failed:
                per_alg_runtime[o.name] += float(o.runtime_s)
        meta["runtime_per_algorithm_s"] = per_alg_runtime
        return meta


# =============================================================================
# Convenience: sequential single-drop runner (debugging / smoke testing)
# =============================================================================

def run_sequential(
    cfg:             CORDISConfig,
    algorithm_specs: Iterable[AlgorithmSpec],
    runner_cfg:      Optional[RunnerConfig] = None,
) -> RunReport:
    """
    Run the full campaign in the main process, bypassing joblib.

    Equivalent to :meth:`MonteCarloRunner.run` with ``n_workers=1`` but
    avoids the joblib import entirely.  Useful for unit tests, debugging
    tracebacks, and notebook quick-checks.
    """
    specs = list(algorithm_specs)
    rc = runner_cfg or RunnerConfig()
    rc.validate()
    if not specs:
        raise ValueError("At least one AlgorithmSpec is required.")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        dupes = [n for n in names if names.count(n) > 1]
        raise ValueError(
            f"AlgorithmSpec names must be unique; duplicates: {set(dupes)}"
        )

    master = SeedSequence(rc.base_seed)
    drop_seqs = master.spawn(rc.n_drops)
    realization_seqs = [
        list(ds.spawn(rc.n_realizations_per_drop)) for ds in drop_seqs
    ]

    t_start_perf = time.perf_counter()
    t_start_iso  = datetime.now(timezone.utc).isoformat(timespec="seconds")

    outputs: List[TrialOutput] = []
    for d in range(rc.n_drops):
        outputs.extend(
            _run_one_drop(
                cfg, specs, drop_seqs[d], realization_seqs[d], d,
                rc.skip_failures,
            )
        )

    runtime = time.perf_counter() - t_start_perf
    t_end_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Reuse the runner's metadata builder via a temporary instance
    tmp = MonteCarloRunner.__new__(MonteCarloRunner)
    tmp.cfg = cfg
    tmp.algorithm_specs = specs
    tmp.runner_cfg = rc
    return RunReport(
        cfg=cfg,
        runner_cfg=rc,
        algorithm_specs=specs,
        outputs=outputs,
        metadata=tmp._build_metadata(t_start_iso, t_end_iso, runtime, outputs),
    )

