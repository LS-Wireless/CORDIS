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
from typing import Any, Dict, List, Optional

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
    return p


def add_drops_args(parser: argparse.ArgumentParser,
                   default_n_drops: int = 20,
                   default_n_real: int = 2) -> None:
    """Add Monte-Carlo ``--n-drops`` / ``--n-realizations`` flags."""
    parser.add_argument("--n-drops", type=int, default=default_n_drops,
                        help="Independent topology drops.")
    parser.add_argument("--n-realizations", type=int, default=default_n_real,
                        help="Channel realisations per drop.")


def add_cdf_args(parser: argparse.ArgumentParser,
                 default_n_drops: int = 50,
                 default_n_real: int = 4) -> None:
    """Alias of :func:`add_drops_args` with CDF-appropriate defaults."""
    add_drops_args(parser, default_n_drops, default_n_real)


def add_sweep_args(parser: argparse.ArgumentParser,
                   default_n_drops: int = 20,
                   default_n_real: int = 2) -> None:
    """Alias of :func:`add_drops_args` with sweep-appropriate defaults."""
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

def setup_logging(log_path: Path, verbosity: int = 0) -> None:
    """File + console logging under ``log_path``.

    ``verbosity`` follows -v convention: 0 → INFO console, 1 → DEBUG
    console, 2+ → DEBUG console + DEBUG everywhere.  File log is
    always DEBUG (it's the durable record).
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
        Typical contents: ``{"n_drops": ..., "n_realizations": ...}``
        plus any sweep ranges or seeds the wrapper exposes.

    Returns
    -------
    Path
        The experiment directory the result was saved to.
    """
    cordis = _import_cordis()
    experiment_kwargs = experiment_kwargs or {}

    if experiment_name not in cordis["REGISTRY"]:
        raise KeyError(f"unknown experiment {experiment_name!r}; "
                       f"registered: {sorted(cordis['REGISTRY'])}")

    # Output directory under output-root, with timestamped sub-dir.
    exp_dir = cordis["experiment_dir"](experiment_name,
                                       root=args.output_root)
    log_path = cordis["log_dir"](exp_dir) / "run.log"
    setup_logging(log_path, verbosity=args.verbose)

    log = logging.getLogger(f"exp.{experiment_name}")
    log.info("=" * 70)
    log.info("Experiment: %s", experiment_name)
    log.info("Output dir: %s", exp_dir)
    log.info("Log file:   %s", log_path)
    log.info("=" * 70)

    # Load config.
    log.info("Loading config base=%s exp=%s", args.base_config, args.exp_config)
    cfg = cordis["load_config"](args.base_config, args.exp_config)

    # Build runner config.  n_drops / n_realizations live inside
    # experiment_kwargs (each experiment knows its own defaults) — we
    # only pass them through to the runner here.
    n_drops = int(experiment_kwargs.get("n_drops", 20))
    n_real  = int(experiment_kwargs.get("n_realizations", 2))
    runner_cfg = cordis["RunnerConfig"](
        n_drops=n_drops,
        n_realizations_per_drop=n_real,
        n_workers=args.n_workers,
        verbose=1 if args.verbose >= 1 else 0,
        base_seed=args.seed,
    )
    log.info("RunnerConfig: n_drops=%d n_real=%d n_workers=%d seed=%d",
             n_drops, n_real, args.n_workers, args.seed)

    # Dispatch.
    fn = cordis["REGISTRY"][experiment_name]
    log.info("Dispatching → %s", fn.__name__)
    result = fn(cfg, runner_cfg, **experiment_kwargs)

    # Save.
    log.info("Saving %s-kind result to %s", result.kind, exp_dir)
    result.save(exp_dir)
    log.info("Done.")
    return exp_dir

