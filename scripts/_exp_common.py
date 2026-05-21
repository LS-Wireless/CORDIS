"""
scripts/_exp_common.py
======================

Shared helpers for the eleven ``scripts/exp_<name>.py`` runner scripts.

Every paper-figure experiment in :mod:`cordis.experiments.registry`
follows the same workflow:

1. parse a uniform CLI (``--base-config``, ``--exp-config``,
   ``--n-drops``, ``--n-realizations``, ``--n-workers``, ``--seed``,
   ``--output-root``);
2. load the merged CORDIS config;
3. build a :class:`RunnerConfig` from CLI values;
4. resolve the per-experiment output directory under
   ``results/exp_<name>/<timestamp>/``;
5. configure file logging under ``<exp_dir>/logs/run.log``;
6. dispatch to the experiment's ``run_*`` function from
   :data:`REGISTRY`, passing any experiment-specific kwargs;
7. save the returned :class:`ExperimentResult` via ``result.save()``.

Each per-experiment wrapper supplies only its experiment name and
the experiment-specific CLI extras (e.g. ``--gamma-min/--gamma-max``
for ``gamma_sweep``).  Step (2)–(7) come for free.

Usage in a per-experiment wrapper::

    # scripts/exp_sinr_cdf.py
    from _exp_common import (
        build_base_parser, add_cdf_args, run_experiment,
    )

    parser = build_base_parser("sinr_cdf")
    add_cdf_args(parser, default_n_drops=50, default_n_real=4)
    args = parser.parse_args()
    extra = {"n_drops": args.n_drops, "n_realizations": args.n_realizations}
    run_experiment("sinr_cdf", args, extra)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sibling helpers importable when the script is run directly from
# anywhere (not just from the scripts/ directory).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Late imports so a missing cordis package surfaces a clean error rather
# than an import-time cascade.
def _import_cordis():
    from cordis.utils.config import load_config
    from cordis.simulation.runner import RunnerConfig
    from cordis.experiments import (
        REGISTRY, experiment_dir, log_dir,
    )
    return {
        "load_config":     load_config,
        "RunnerConfig":    RunnerConfig,
        "REGISTRY":        REGISTRY,
        "experiment_dir":  experiment_dir,
        "log_dir":         log_dir,
    }


# ─────────────────────────────────────────────────────────────────────
# CLI plumbing
# ─────────────────────────────────────────────────────────────────────

def build_base_parser(experiment_name: str) -> argparse.ArgumentParser:
    """ArgumentParser with the seven flags every experiment accepts."""
    p = argparse.ArgumentParser(
        prog=f"exp_{experiment_name}",
        description=f"Run the {experiment_name!r} experiment from "
                    f"cordis.experiments.registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-config",
                   default="configs/default.json",
                   help="Path to the base CORDIS config JSON.")
    p.add_argument("--exp-config",
                   default=None,
                   help="Optional experiment-specific overrides JSON, "
                        "deep-merged on top of --base-config.")
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Joblib workers (-1 = all cores, 1 = sequential).")
    p.add_argument("--seed", type=int, default=42,
                   help="Master Monte-Carlo seed.")
    p.add_argument("--output-root", default="results",
                   help="Root directory under which exp_<name>/<ts>/ "
                        "subdirectories are created.")
    p.add_argument("--verbose", "-v", action="count", default=0,
                   help="Increase log verbosity (use -vv for DEBUG).")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable the per-drop progress bar on stderr "
                        "(also auto-off when stderr is not a TTY).")
    p.add_argument("--show-algorithm-warnings", action="store_true",
                   help="Show per-trial WARNING messages from "
                        "cordis.algorithms.* on console. Default is to "
                        "suppress them (they always go to the log file).")
    return p


def add_drops_args(parser: argparse.ArgumentParser,
                   default_n_drops: Optional[int] = None,
                   default_n_real:  Optional[int] = None) -> None:
    """
    Add Monte-Carlo trial-count flags.

    Three knobs are exposed:

    * ``--n-trials N``     total trial count (decomposes into drops × real)
    * ``--n-drops D``      explicit drops (overrides decomposition)
    * ``--n-realizations R``  explicit realizations (overrides decomposition)

    Resolution precedence is handled by :func:`resolve_drops_real`:

    1. If BOTH ``--n-drops`` and ``--n-realizations`` are explicit, use them.
    2. Else if ``--n-trials`` is given, decompose into the two closest
       factors (with the larger one assigned to realizations).
    3. Else fall back to ``cfg.simulation.n_trials`` from the JSON config
       and decompose.
    4. Per-experiment defaults (``default_n_drops`` / ``default_n_real``)
       are last-resort fallbacks for cases where no config is loadable.

    Defaults are ``None`` (no implicit value) so that the resolver can
    distinguish "user explicitly set this" from "fall through to
    decomposition".
    """
    parser.add_argument("--n-drops", type=int, default=default_n_drops,
                        help="Independent topology drops (overrides "
                             "n_trials decomposition when set).")
    parser.add_argument("--n-realizations", type=int, default=default_n_real,
                        help="Channel realisations per drop (overrides "
                             "n_trials decomposition when set).")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Total trial count.  Decomposed into drops × "
                             "realizations using the two closest factors.  "
                             "Falls back to cfg.simulation.n_trials.")
    # --specs is universally available; run_experiment silently skips it
    # for functions whose signature doesn't include spec_set (e.g.
    # run_convergence_trace, run_fronthaul_table).  Default is None so
    # each run_* function's own historical default applies (gamma_sweep
    # → cordis_vs_centralized, antennas_sweep → cordis_vs_benchmarks,
    # everything else → all_algorithms).
    add_specs_arg(parser, default=None)


# ─────────────────────────────────────────────────────────────────────
# n_trials decomposition
# ─────────────────────────────────────────────────────────────────────

def closest_factor_pair(n: int) -> Tuple[int, int]:
    """
    Return ``(a, b)`` with ``a * b == n``, ``a <= b``, and ``|b - a|`` minimal.

    Examples
    --------
    >>> closest_factor_pair(400)   # 20 × 20
    (20, 20)
    >>> closest_factor_pair(500)   # 20 × 25
    (20, 25)
    >>> closest_factor_pair(100)   # 10 × 10
    (10, 10)
    >>> closest_factor_pair(13)    # 1 × 13  (prime → lopsided)
    (1, 13)
    """
    import math
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    a = int(math.isqrt(n))
    while a > 0 and n % a != 0:
        a -= 1
    return a, n // a


def resolve_drops_real(args: argparse.Namespace,
                       cfg_n_trials: Optional[int] = None,
                       default_n_drops: int = 10,
                       default_n_real:  int = 10) -> Tuple[int, int, str]:
    """
    Resolve ``(n_drops, n_realizations)`` from CLI args + config.

    Returns a triple ``(n_drops, n_realizations, source)`` where ``source``
    is a short string identifying which branch fired — useful for logging.

    See :func:`add_drops_args` for the precedence chain.
    """
    log = logging.getLogger(__name__)
    d, r = args.n_drops, args.n_realizations

    # Branch 1: both explicit → trust the user.
    if d is not None and r is not None:
        return d, r, "explicit drops×real"

    # Branch 2: CLI n-trials given, decompose around any explicit dim.
    n_t = args.n_trials if args.n_trials is not None else cfg_n_trials

    if n_t is not None and n_t > 0:
        if d is not None:                              # only drops explicit
            r = max(1, -(-n_t // d))                  # ceil(n_t / d)
            return d, r, f"n_trials={n_t} with explicit drops={d}"
        if r is not None:                              # only real explicit
            d = max(1, -(-n_t // r))
            return d, r, f"n_trials={n_t} with explicit real={r}"
        # Neither explicit — pure decomposition.
        a, b = closest_factor_pair(n_t)
        if a == 1 and n_t > 4:
            log.warning(
                "n_trials=%d is prime — decomposed as (1, %d).  "
                "Consider setting N_DROPS and N_REAL explicitly "
                "for better Monte Carlo balance.", n_t, n_t,
            )
        return a, b, f"n_trials={n_t} decomposed → ({a}, {b})"

    # Branch 3: nothing usable from CLI or config — last-resort defaults.
    return (
        default_n_drops if d is None else d,
        default_n_real  if r is None else r,
        "per-experiment fallback defaults",
    )


def add_specs_arg(parser: argparse.ArgumentParser,
                  default: Optional[str] = None) -> None:
    """
    Add the ``--specs`` CLI flag for selecting which spec set to run.

    Default is ``None`` — meaning "don't override, use the run_*
    function's own historical default" (which differs per experiment:
    sinr_cdf/scnr_cdf default to ``all_algorithms``, gamma/kappa/clutter
    sweeps default to ``cordis_vs_centralized``, antennas_sweep
    defaults to ``cordis_vs_benchmarks``).  Pass an explicit name to
    override.

    See :func:`cordis.experiments.registry._resolve_spec_set` for the
    runtime lookup.
    """
    try:
        from cordis.experiments import list_spec_sets
        choices = list_spec_sets()
    except Exception:
        choices = ["cordis_only", "cordis_vs_centralized",
                   "cordis_vs_benchmarks", "all_algorithms"]
    parser.add_argument(
        "--specs", type=str, default=default, choices=choices,
        metavar="SPECS",
        help=("Which named spec set to run.  When omitted, the "
              "experiment's own default applies.  Available: "
              + ", ".join(choices) + "."),
    )


def add_cdf_args(parser: argparse.ArgumentParser,
                 default_n_drops: Optional[int] = None,
                 default_n_real: Optional[int] = None) -> None:
    """CDF-experiment argument bundle (trial counts + ``--specs``).

    ``--specs`` is registered via :func:`add_drops_args`; this helper
    exists so per-experiment scripts can document CDF-specific intent.
    """
    add_drops_args(parser, default_n_drops, default_n_real)


def add_sweep_args(parser: argparse.ArgumentParser,
                   default_n_drops: Optional[int] = None,
                   default_n_real: Optional[int] = None) -> None:
    """Sweep-experiment argument bundle (trial counts + ``--specs``)."""
    add_drops_args(parser, default_n_drops, default_n_real)


def parse_value_list(s: str, kind: type = float) -> List:
    """Parse ``"1,2,3"`` or ``"1 2 3"`` into a list of numbers."""
    if not s:
        return []
    parts = s.replace(",", " ").split()
    return [kind(x) for x in parts]


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

class _AlgorithmConsoleFilter(logging.Filter):
    """
    Drop diagnostic WARNING-level messages from per-trial algorithm /
    channel code on the *console* handler.

    These warnings (e.g. "CVXPY status=infeasible — returning previous W",
    "β̂ has negative real parts") are expected behavior under known
    difficult conditions and not actionable per-trial.  They still go to
    the file log (which is DEBUG-level) for postmortem review.
    Pass ``--show-algorithm-warnings`` to re-enable on console.

    Loggers under these prefixes are filtered:
        - cordis.algorithms.*  (CVXPY infeasible, SCA fallback, etc.)
        - cordis.channel.*     (β̂ phase alignment, etc.)

    Real errors (level >= ERROR) are always surfaced.
    """
    _SUPPRESS_PREFIXES = ("cordis.algorithms.", "cordis.channel.")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        if any(record.name.startswith(p) for p in self._SUPPRESS_PREFIXES):
            return False
        return True


def setup_logging(log_path: Path, verbosity: int = 0,
                  quiet_algorithms: bool = True) -> None:
    """File + console logging under ``log_path``.

    ``verbosity`` follows -v convention: 0 → INFO console, 1 → DEBUG
    console, 2+ → DEBUG console + DEBUG everywhere.  File log is
    always DEBUG (it's the durable record).

    ``quiet_algorithms`` (default True) suppresses per-trial WARNING
    messages from ``cordis.algorithms.*`` on the console; they still
    go to the file log.  Disable with ``--show-algorithm-warnings``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # Clear any handlers Python auto-installed (so re-runs in REPLs work).
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.DEBUG)

    file_h = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(file_h)

    console_h = logging.StreamHandler(sys.stderr)
    console_h.setLevel(logging.DEBUG if verbosity >= 1 else logging.INFO)
    console_h.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    if quiet_algorithms:
        console_h.addFilter(_AlgorithmConsoleFilter())
    root.addHandler(console_h)

    # Even in verbose mode, mute matplotlib's font scanner — it's noisy
    # and irrelevant to experiment debugging.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def run_experiment(experiment_name: str,
                   args: argparse.Namespace,
                   experiment_kwargs: Optional[Dict[str, Any]] = None,
                   *,
                   fallback_n_drops: int = 10,
                   fallback_n_real:  int = 10,
                   ) -> Path:
    """End-to-end: load cfg, build runner, run, save.

    Parameters
    ----------
    experiment_name
        Key in :data:`REGISTRY`.  Determines which ``run_*`` function
        executes and the ``exp_<name>/`` output directory name.
    args
        Parsed CLI namespace from :func:`build_base_parser` (+ optional
        per-experiment additions).
    experiment_kwargs
        Forwarded to the experiment's ``run_*`` function as **kwargs.
        Typical contents: sweep ranges, seeds.  ``n_drops`` and
        ``n_realizations`` are resolved internally via
        :func:`resolve_drops_real` and overridden in this dict.
    fallback_n_drops, fallback_n_real
        Per-experiment last-resort defaults used when neither CLI args
        nor the JSON config provide a usable trial count.  Inlined into
        each runner .py by the generator from the per-experiment metadata.

    Returns
    -------
    Path
        The experiment directory the result was saved to.
    """
    cordis = _import_cordis()
    experiment_kwargs = dict(experiment_kwargs or {})

    if experiment_name not in cordis["REGISTRY"]:
        raise KeyError(f"unknown experiment {experiment_name!r}; "
                       f"registered: {sorted(cordis['REGISTRY'])}")

    # Output directory under output-root, with timestamped sub-dir.
    exp_dir = cordis["experiment_dir"](experiment_name,
                                       root=args.output_root)
    log_path = cordis["log_dir"](exp_dir) / "run.log"
    setup_logging(log_path, verbosity=args.verbose,
                  quiet_algorithms=not getattr(args,
                                               "show_algorithm_warnings",
                                               False))

    log = logging.getLogger(f"exp.{experiment_name}")
    log.info("=" * 70)
    log.info("Experiment: %s", experiment_name)
    log.info("Output dir: %s", exp_dir)
    log.info("Log file:   %s", log_path)
    log.info("=" * 70)

    # Load config.
    log.info("Loading config base=%s exp=%s", args.base_config, args.exp_config)
    cfg = cordis["load_config"](args.base_config, args.exp_config)

    # Resolve trial counts via the precedence chain:
    #   explicit drops×real → --n-trials → cfg.simulation.n_trials → fallback.
    cfg_n_trials = getattr(getattr(cfg, "simulation", None), "n_trials", None)
    n_drops, n_real, n_trials_source = resolve_drops_real(
        args,
        cfg_n_trials=cfg_n_trials,
        default_n_drops=fallback_n_drops,
        default_n_real=fallback_n_real,
    )
    log.info("Trial counts: n_drops=%d, n_realizations=%d  (%s)",
             n_drops, n_real, n_trials_source)

    # Resolved trial counts are forwarded to the experiment's run_*
    # function ONLY if it accepts them.  Sweep + CDF experiments take
    # them as explicit kwargs; convergence_trace and fronthaul_table
    # have fixed setups and don't.  Mirrors the signature-inspection
    # pattern used below for --specs.
    import inspect as _inspect
    fn = cordis["REGISTRY"][experiment_name]
    sig = _inspect.signature(fn)
    if "n_drops" in sig.parameters:
        experiment_kwargs["n_drops"] = n_drops
    else:
        log.info("--n-drops not applicable to %r (fixed-trial experiment); "
                 "ignored.", experiment_name)
    if "n_realizations" in sig.parameters:
        experiment_kwargs["n_realizations"] = n_real
    else:
        log.info("--n-realizations not applicable to %r (fixed-trial "
                 "experiment); ignored.", experiment_name)

    # Forward --specs IFF the experiment's run_* function accepts it.
    # convergence_trace and fronthaul_table don't take a spec_set kwarg
    # (their algorithm choice is fixed by design), so we silently skip
    # them rather than crash on a TypeError.
    if hasattr(args, "specs") and args.specs is not None:
        if "spec_set" in sig.parameters:
            experiment_kwargs["spec_set"] = args.specs
            log.info("spec_set=%s", args.specs)
        else:
            log.info("--specs ignored for %s (fixed algorithm set)",
                     experiment_name)

    runner_cfg = cordis["RunnerConfig"](
        n_drops=n_drops,
        n_realizations_per_drop=n_real,
        n_workers=args.n_workers,
        verbose=1 if args.verbose >= 1 else 0,
        progress=not getattr(args, "no_progress", False),
        base_seed=args.seed,
    )
    log.info("RunnerConfig: n_drops=%d n_real=%d n_workers=%d seed=%d "
             "progress=%s",
             n_drops, n_real, args.n_workers, args.seed,
             runner_cfg.progress)

    # Dispatch.
    fn = cordis["REGISTRY"][experiment_name]
    log.info("Dispatching → %s", fn.__name__)
    result = fn(cfg, runner_cfg, **experiment_kwargs)

    # Save.
    log.info("Saving %s-kind result to %s", result.kind, exp_dir)
    result.save(exp_dir)
    log.info("Done.")
    return exp_dir

