"""
Stage 8b — Experiment registry.

Each registered experiment is a function with signature::

    run_X(cfg, runner_cfg, **kwargs) -> ExperimentResult

The :data:`REGISTRY` dict maps experiment names (used on the CLI and as
the ``experiment_dir(name)`` argument) to these callables.  Each
function:

1. Optionally overrides a few fields on ``cfg`` / ``runner_cfg`` to match
   the experiment's needs (e.g. larger ``n_drops`` for CDFs, smaller for
   sweeps).
2. Builds the appropriate :class:`AlgorithmSpec` list via
   :mod:`cordis.experiments.specs`.
3. Runs the Monte-Carlo runner (or, for ``convergence_trace``, the ADMM
   solver directly) and wraps the output in :class:`ExperimentResult`.

Eleven experiments are registered:

==================  =========  =========================  ============================
name                kind       spec set                   swept axis
==================  =========  =========================  ============================
sinr_cdf            single     all_algorithms             —
scnr_cdf            single     all_algorithms             —
gamma_sweep         sweep      cordis_vs_centralized      γ [dB]
omega_sweep         sweep      cordis_vs_centralized      ω ∈ [0,1]
kappa_sweep         sweep      cordis_vs_centralized      κ
snr_sweep           sweep      all_algorithms             Pₘₐₓ/σ² [dB]
n_ue_sweep          sweep      all_algorithms             N_UE
n_ap_sweep          sweep      all_algorithms             N_AP
antennas_sweep      sweep      cordis_vs_benchmarks       M (antennas/AP)
convergence_trace   trace      CORDIS-ADMM only           ADMM iter
fronthaul_table     table      CORDIS-ADMM (avg iters)    —
==================  =========  =========================  ============================

Note: the registry name ``omega_sweep`` is used because the underlying
code parameter is ``omega``; many papers call this λ.  Display labels
use whichever LaTeX symbol the user prefers.

Use :func:`list_experiments` to introspect.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from cordis.experiments.result import ExperimentResult
from cordis.experiments.specs import (
    admm_spec,
    all_algorithms,
    cordis_vs_benchmarks,
    cordis_vs_centralized,
)
from cordis.experiments.sweeps import (
    SweepAxis,
    sweep_config_field,
    sweep_spec_factory,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _override_runner(runner_cfg: Any, **overrides) -> Any:
    """Return a shallow copy of runner_cfg with selected fields replaced.

    Uses dataclasses.replace if runner_cfg is a dataclass; otherwise
    shallow-copies and sets attributes.
    """
    if dataclasses.is_dataclass(runner_cfg):
        return dataclasses.replace(runner_cfg, **overrides)
    rc = copy.copy(runner_cfg)
    for k, v in overrides.items():
        setattr(rc, k, v)
    return rc


def _override_cfg(cfg: Any, **field_overrides) -> Any:
    """Return a deep-copy of cfg with selected fields replaced (dotted paths)."""
    if not field_overrides:
        return cfg
    out = copy.deepcopy(cfg)
    for path, value in field_overrides.items():
        parts = path.split(".")
        cursor = out
        for p in parts[:-1]:
            cursor = getattr(cursor, p)
        setattr(cursor, parts[-1], value)
    return out


def _run_single(cfg, runner_cfg, specs):
    """Build runner, run, wrap report in SimResult.  Returns SimResult."""
    # Late imports so this module imports even before the rest of the
    # codebase is fully wired up (e.g. during unit testing).
    from cordis.simulation.runner import MonteCarloRunner
    from cordis.simulation.result import SimResult
    runner = MonteCarloRunner(cfg, specs, runner_cfg)
    return SimResult.from_run(runner.run())


def _cfg_summary(cfg: Any) -> Dict[str, Any]:
    """Best-effort extraction of headline cfg fields for the manifest."""
    summary: Dict[str, Any] = {}
    for path in ("topology.n_ap", "topology.n_ue", "topology.n_targets",
                 "system.n_ant", "system.snr_db", "system.Pmax_dbm"):
        try:
            cursor = cfg
            for p in path.split("."):
                cursor = getattr(cursor, p)
            summary[path] = cursor
        except AttributeError:
            pass
    return summary


# ─────────────────────────────────────────────────────────────────────
#  Single-config experiments
# ─────────────────────────────────────────────────────────────────────

def run_sinr_cdf(
    cfg, runner_cfg,
    *,
    n_drops: int = 50,
    n_realizations: int = 4,
    spec_kwargs: Optional[Dict[str, Any]] = None,
) -> ExperimentResult:
    """Empirical CDF of min-SINR across N_drops × N_realisations trials."""
    spec_kwargs = spec_kwargs or {}
    specs = all_algorithms(**spec_kwargs)
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    sr = _run_single(cfg, rc, specs)
    return ExperimentResult(
        name="sinr_cdf",
        kind="single",
        sim_result=sr,
        metadata={
            "n_drops": n_drops,
            "n_realizations": n_realizations,
            "spec_set": "all_algorithms",
            "cfg_summary": _cfg_summary(cfg),
        },
    )


def run_scnr_cdf(
    cfg, runner_cfg,
    *,
    n_drops: int = 50,
    n_realizations: int = 4,
    spec_kwargs: Optional[Dict[str, Any]] = None,
) -> ExperimentResult:
    """Empirical CDF of sum-SCNR (sensing figure-of-merit)."""
    spec_kwargs = spec_kwargs or {}
    specs = all_algorithms(**spec_kwargs)
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    sr = _run_single(cfg, rc, specs)
    return ExperimentResult(
        name="scnr_cdf",
        kind="single",
        sim_result=sr,
        metadata={
            "n_drops": n_drops,
            "n_realizations": n_realizations,
            "spec_set": "all_algorithms",
            "cfg_summary": _cfg_summary(cfg),
        },
    )


# ─────────────────────────────────────────────────────────────────────
#  Sweep experiments — vary a SPEC-FACTORY KWARG
# ─────────────────────────────────────────────────────────────────────

def run_gamma_sweep(
    cfg, runner_cfg,
    *,
    gamma_values: Optional[List[float]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
) -> ExperimentResult:
    """Sweep per-UE SINR target γ ∈ {-3, 0, 3, 6, 10, 14, 18} dB."""
    if gamma_values is None:
        gamma_values = [-3.0, 0.0, 3.0, 6.0, 10.0, 14.0, 18.0]
    axis = SweepAxis(
        name="gamma_u_db",
        values=gamma_values,
        display=r"$\gamma$ [dB]",
        unit="dB",
    )
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_spec_factory(
        cfg, cordis_vs_centralized, "gamma_u_db", axis, rc,
    )
    return ExperimentResult(
        name="gamma_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "spec_set": "cordis_vs_centralized",
                  "cfg_summary": _cfg_summary(cfg)},
    )


def run_omega_sweep(
    cfg, runner_cfg,
    *,
    omega_values: Optional[List[float]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    display: str = r"$\omega$",
) -> ExperimentResult:
    """Sweep comm/sensing trade-off ω ∈ [0, 1].

    Paper convention uses λ; the code parameter is ``omega``.  Pass
    ``display=r"$\\lambda$"`` to render the paper label.
    """
    if omega_values is None:
        omega_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    axis = SweepAxis(
        name="omega",
        values=omega_values,
        display=display,
    )
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_spec_factory(
        cfg, cordis_vs_centralized, "omega", axis, rc,
    )
    return ExperimentResult(
        name="omega_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "spec_set": "cordis_vs_centralized",
                  "cfg_summary": _cfg_summary(cfg)},
    )


def run_kappa_sweep(
    cfg, runner_cfg,
    *,
    kappa_values: Optional[List[float]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
) -> ExperimentResult:
    """Sweep prox-regularisation weight κ.

    Only CORDIS-ADMM and Centralized actually consume κ; CORDIS-Split
    absorbs it via ``**ignored`` and produces the same output at every
    point (a flat line in the plot — informative as a control).
    """
    if kappa_values is None:
        kappa_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    axis = SweepAxis(
        name="kappa",
        values=kappa_values,
        display=r"$\kappa$",
    )
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_spec_factory(
        cfg, cordis_vs_centralized, "kappa", axis, rc,
    )
    return ExperimentResult(
        name="kappa_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "spec_set": "cordis_vs_centralized",
                  "cfg_summary": _cfg_summary(cfg)},
    )


# ─────────────────────────────────────────────────────────────────────
#  Sweep experiments — vary a CONFIG FIELD
# ─────────────────────────────────────────────────────────────────────

def run_snr_sweep(
    cfg, runner_cfg,
    *,
    snr_values_db: Optional[List[float]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    field_path: str = "system.snr_db",
) -> ExperimentResult:
    """Sweep operating SNR (Pₘₐₓ / σ²) [dB]."""
    if snr_values_db is None:
        snr_values_db = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    axis = SweepAxis(
        name="snr_db",
        values=snr_values_db,
        display="SNR [dB]",
        unit="dB",
    )
    specs = all_algorithms()
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_config_field(cfg, specs, field_path, axis, rc)
    return ExperimentResult(
        name="snr_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "field_path": field_path,
                  "spec_set": "all_algorithms",
                  "cfg_summary": _cfg_summary(cfg)},
    )


def run_n_ue_sweep(
    cfg, runner_cfg,
    *,
    n_ue_values: Optional[List[int]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    field_path: str = "topology.n_ue",
) -> ExperimentResult:
    """Sweep number of UEs (loading / scalability)."""
    if n_ue_values is None:
        n_ue_values = [2, 4, 6, 8, 10]
    axis = SweepAxis(
        name="n_ue",
        values=[float(v) for v in n_ue_values],
        display=r"$N_{\rm UE}$",
    )
    specs = all_algorithms()
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_config_field(cfg, specs, field_path, axis, rc)
    return ExperimentResult(
        name="n_ue_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "spec_set": "all_algorithms",
                  "field_path": field_path,
                  "cfg_summary": _cfg_summary(cfg)},
    )


def run_n_ap_sweep(
    cfg, runner_cfg,
    *,
    n_ap_values: Optional[List[int]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    field_path: str = "topology.n_ap",
) -> ExperimentResult:
    """Sweep number of APs (AP scalability)."""
    if n_ap_values is None:
        n_ap_values = [4, 6, 8, 10, 12]
    axis = SweepAxis(
        name="n_ap",
        values=[float(v) for v in n_ap_values],
        display=r"$N_{\rm AP}$",
    )
    specs = all_algorithms()
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_config_field(cfg, specs, field_path, axis, rc)
    return ExperimentResult(
        name="n_ap_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "field_path": field_path,
                  "spec_set": "all_algorithms",
                  "cfg_summary": _cfg_summary(cfg)},
    )


def run_antennas_sweep(
    cfg, runner_cfg,
    *,
    n_ant_values: Optional[List[int]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    field_path: str = "system.n_ant",
) -> ExperimentResult:
    """Sweep antennas per AP (M)."""
    if n_ant_values is None:
        n_ant_values = [4, 6, 8, 10, 12]
    axis = SweepAxis(
        name="n_ant",
        values=[float(v) for v in n_ant_values],
        display=r"$M$",
    )
    specs = cordis_vs_benchmarks()
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_config_field(cfg, specs, field_path, axis, rc)
    return ExperimentResult(
        name="antennas_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "field_path": field_path,
                  "spec_set": "cordis_vs_benchmarks",
                  "cfg_summary": _cfg_summary(cfg)},
    )


# ─────────────────────────────────────────────────────────────────────
#  Convergence-trace experiment (single trial, full ADMM history)
# ─────────────────────────────────────────────────────────────────────

def run_convergence_trace(
    cfg, runner_cfg,
    *,
    drop_seed: int = 42,
    realization_seed: int = 43,
    n_admm_max: int = 30,
    spec_kwargs: Optional[Dict[str, Any]] = None,
) -> ExperimentResult:
    """
    Run CORDIS-ADMM on a single trial and capture the full history.

    The Monte-Carlo runner aggregates per-trial outputs and only stores
    summary diagnostics, so this experiment bypasses the runner and
    calls :func:`run_cordis_admm` directly.

    Required APIs (already present in Stage 7):
    * ``cordis.simulation.scenario.build_scenario_from_seeds``
    * ``cordis.algorithms.joint_opt.run_cordis_admm`` returning an
      ``ADMMResult`` with ``primal_res_history``, ``dual_res_history``,
      and (where available) per-iteration slack history.
    """
    from cordis.simulation.scenario import build_scenario_from_seeds
    from cordis.algorithms.joint_opt import run_cordis_admm

    spec_kwargs = spec_kwargs or {}
    spec_kwargs.setdefault("n_admm_max", n_admm_max)
    spec_obj = admm_spec(**spec_kwargs)

    scenario = build_scenario_from_seeds(
        cfg, drop_seed=drop_seed, realization_seed=realization_seed,
    )

    # Forward only the params the ADMM solver knows about.
    fwd_keys = ("gamma_u_db", "omega", "rho_admm", "kappa", "xi_slack",
                "slack_tol", "n_admm_max", "eps_pri", "eps_dual",
                "warm_start_from_split")
    admm_kwargs = {k: spec_obj.params[k]
                   for k in fwd_keys if k in spec_obj.params}

    _W, admm_result = run_cordis_admm(
        topo=scenario.topo,
        cfg=scenario.cfg,
        est=scenario.estimation,
        sensing_stats=scenario.sensing_stats,
        association=scenario.association,
        sigma_n_sq=scenario.sigma_n_sq,
        Pmax=scenario.Pmax,
        **admm_kwargs,
    )

    md: Dict[str, Any] = {
        "drop_seed": int(drop_seed),
        "realization_seed": int(realization_seed),
        "spec_set": "cordis_admm (direct)",
        "cfg_summary": _cfg_summary(cfg),
    }
    for k, v in spec_obj.params.items():
        if isinstance(v, np.ndarray):
            md[f"{k}_uniform"] = float(v[0]) if v.size else None
        else:
            md[k] = v

    return ExperimentResult(
        name="convergence_trace", kind="trace",
        admm_result=admm_result,
        metadata=md,
    )


# ─────────────────────────────────────────────────────────────────────
#  Fronthaul-overhead table
# ─────────────────────────────────────────────────────────────────────

def run_fronthaul_table(
    cfg, runner_cfg,
    *,
    n_admm_drops: int = 20,
    n_realizations: int = 2,
) -> ExperimentResult:
    """
    Compute per-AP fronthaul coordination overhead for each algorithm.

    Centralized   : 2 × C^{M × |D|} (paper Table I baseline)
    CORDIS-Split  : 4 × R¹ (M(a), α̃, Δ̃, ρ*)
    CORDIS-ADMM   : R¹, R^|U| per outer iteration × T_ADMM

    T_ADMM is estimated by running ADMM for ``n_admm_drops`` drops and
    averaging the ``iters`` diagnostic.
    """
    # 1. Estimate average ADMM iterations.
    specs = [admm_spec()]
    rc = _override_runner(runner_cfg,
                          n_drops=n_admm_drops,
                          n_realizations_per_drop=n_realizations)
    sr = _run_single(cfg, rc, specs)
    try:
        admm_avg_iters = float(sr.mean("CORDIS-ADMM", "iters"))
    except (KeyError, AttributeError, TypeError):
        admm_avg_iters = float("nan")
        logger.warning("Could not extract 'iters' diagnostic; "
                       "leaving T_ADMM = NaN in table.")

    # 2. Read scalar topology dimensions (best-effort).
    M     = int(getattr(cfg.system,   "n_ant",     0))
    n_ue  = int(getattr(cfg.topology, "n_ue",      0))
    n_tg  = int(getattr(cfg.topology, "n_targets", 0))
    n_streams = n_ue + n_tg

    iters_nan = admm_avg_iters != admm_avg_iters  # NaN check

    # 3. Build table rows.
    table = {
        "Centralized": {
            "data_to_share": r"$\mathbf{H}_a, \mathbf{W}_a$",
            "size":          r"$2 \times \mathbb{C}^{M \times |\mathcal{D}|}$",
            "real_scalars":  2 * 2 * M * n_streams,
            "scales_with":   r"$M \cdot |\mathcal{D}|$",
            "scalable":      False,
        },
        "CORDIS-Split": {
            "data_to_share": r"$\mathcal{M}(a), \tilde{\alpha}_a, \tilde{\Delta}_a, \rho^\ast_a$",
            "size":          r"$4 \times \mathbb{R}^1$",
            "real_scalars":  4,
            "scales_with":   "constant",
            "scalable":      True,
        },
        "CORDIS-ADMM": {
            "data_to_share": r"$\gamma^{[t]}, \boldsymbol{\psi}^{[t]}$",
            "size":          r"$\mathbb{R}^1, \mathbb{R}^{|\mathcal{U}|}$",
            "real_scalars":  (None if iters_nan
                              else (1 + n_ue) * admm_avg_iters),
            "scales_with":   r"$T_{\rm ADMM} \cdot (1 + |\mathcal{U}|)$",
            "scalable":      True,
        },
    }

    return ExperimentResult(
        name="fronthaul_table", kind="table",
        table_data=table,
        metadata={
            "M": M, "n_ue": n_ue, "n_targets": n_tg, "n_streams": n_streams,
            "admm_avg_iters": admm_avg_iters,
            "n_admm_drops": n_admm_drops,
            "n_realizations": n_realizations,
            "cfg_summary": _cfg_summary(cfg),
        },
    )


# ─────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────

REGISTRY: Dict[str, Callable[..., ExperimentResult]] = {
    # Single-config CDFs
    "sinr_cdf":          run_sinr_cdf,
    "scnr_cdf":          run_scnr_cdf,
    # Spec-factory sweeps
    "gamma_sweep":       run_gamma_sweep,
    "omega_sweep":       run_omega_sweep,
    "kappa_sweep":       run_kappa_sweep,
    # Config-field sweeps
    "snr_sweep":         run_snr_sweep,
    "n_ue_sweep":        run_n_ue_sweep,
    "n_ap_sweep":        run_n_ap_sweep,
    "antennas_sweep":    run_antennas_sweep,
    # Single-trial trajectory
    "convergence_trace": run_convergence_trace,
    # Overhead table
    "fronthaul_table":   run_fronthaul_table,
}


def list_experiments() -> List[str]:
    """Return the sorted list of registered experiment names."""
    return sorted(REGISTRY.keys())


def get_experiment(name: str) -> Callable[..., ExperimentResult]:
    """Look up an experiment by name; raises KeyError with a helpful message."""
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown experiment {name!r}. "
            f"Registered: {', '.join(list_experiments())}"
        )
    return REGISTRY[name]


__all__ = [
    "REGISTRY",
    "list_experiments",
    "get_experiment",
    # Direct exports of every run_* (handy for IDE auto-complete)
    "run_sinr_cdf", "run_scnr_cdf",
    "run_gamma_sweep", "run_omega_sweep", "run_kappa_sweep",
    "run_snr_sweep", "run_n_ue_sweep", "run_n_ap_sweep",
    "run_antennas_sweep",
    "run_convergence_trace", "run_fronthaul_table",
]

