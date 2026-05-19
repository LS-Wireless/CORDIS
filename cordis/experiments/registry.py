"""
Stage 8b — Experiment registry.

Each registered experiment is a function with signature::

    run_X(cfg, runner_cfg, **kwargs) -> ExperimentResult

The :data:`REGISTRY` dict maps experiment names (used on the CLI and as
the ``experiment_dir(name)`` argument) to these callables.  Each
function:

1. Optionally overrides a few fields on ``cfg`` / ``runner_cfg`` to match
   the experiment's needs (larger ``n_drops`` for CDFs, smaller for
   sweeps).
2. Builds the appropriate :class:`AlgorithmSpec` list via
   :mod:`cordis.experiments.specs`.
3. Runs the Monte-Carlo runner (or, for ``convergence_trace``, the ADMM
   solver directly) and wraps the output in :class:`ExperimentResult`.

Eleven experiments are registered:

===================  =========  =========================  ============================
name                 kind       spec set                   swept axis
===================  =========  =========================  ============================
sinr_cdf             single     all_algorithms             —
scnr_cdf             single     all_algorithms             —
gamma_sweep          sweep      cordis_vs_centralized      γ [dB]
clutter_cnr_sweep    sweep      cordis_vs_centralized      Clutter CNR [dB]
kappa_sweep          sweep      cordis_vs_centralized      κ
snr_sweep            sweep      all_algorithms             Pₘₐₓ/σ² [dB]
n_ue_sweep           sweep      all_algorithms             N_UE
n_ap_sweep           sweep      all_algorithms             N_AP
antennas_sweep       sweep      cordis_vs_benchmarks       M (antennas/AP)
convergence_trace    trace      CORDIS-ADMM only           ADMM iter
fronthaul_table      table      CORDIS-ADMM (avg iters)    —
===================  =========  =========================  ============================

Notes
-----
* ``clutter_cnr_sweep`` probes the sensing/comm tension through the
  channel: stronger clutter raises the clutter-plus-noise floor in
  ``R_g_a_r`` (Proposition 3), so at fixed κ the algorithm trades
  raw SCNR against SINR feasibility.  Useful when one wants to
  characterise the tension geometrically rather than through a
  trade-off knob (the journal-paper formulation has no λ-style
  scalar trade-off; SINR is a hard constraint, sensing is
  maximised, and κ alone weights clutter avoidance).

* ``kappa_sweep`` updates BOTH ``cfg.algorithm.admm.kappa`` (for
  Centralized, which reads κ from cfg directly) AND rebuilds the
  spec with the new κ (for ADMM, which reads κ from spec.params via
  the dispatcher).  Without overriding cfg, Centralized would stay
  pinned at its baseline κ across the sweep.

* ``n_ue_sweep`` and ``kappa_sweep`` use custom inner loops rather
  than the generic sweep helpers because they need to mutate cfg AND
  rebuild specs per sweep point (n_ue affects the size of
  ``gamma_u_db``; κ needs to update two places).

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
    cordis_only,
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
#  Named spec sets (Stage 10)
#
#  Every run_* function that constructs a list of algorithms looks up
#  its spec factory here.  Pass `spec_set=<name>` (kwarg) or
#  `--specs <name>` (CLI) to choose a subset.  Add a new named set by
#  registering its factory below — no other code change needed.
# ─────────────────────────────────────────────────────────────────────
_SPEC_SETS = {
    "cordis_only":           cordis_only,            # Split + ADMM
    "cordis_vs_centralized": cordis_vs_centralized,  # + Centralized
    "cordis_vs_benchmarks":  cordis_vs_benchmarks,   # Split/ADMM + 4 benchmarks
    "all_algorithms":        all_algorithms,         # all 9 (default)
}


def _resolve_spec_set(name: str):
    """Look up a named spec factory; raise ValueError on unknown names."""
    if name not in _SPEC_SETS:
        raise ValueError(
            f"unknown spec_set {name!r}; available: {sorted(_SPEC_SETS)}"
        )
    return _SPEC_SETS[name]


def list_spec_sets() -> list:
    """Public accessor for the four named spec sets (e.g., for CLI choices)."""
    return list(_SPEC_SETS)


# ─────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _override_runner(runner_cfg: Any, **overrides) -> Any:
    """Return a shallow copy of runner_cfg with selected fields replaced."""
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
    """Best-effort extraction of headline cfg fields for the manifest.

    Field paths reflect the real CORDISConfig tree (see
    cordis/utils/config.py): topology.{n_ap, n_ue, n_targets, n_ant},
    channel.snr_db, algorithm.{gamma_db, admm.kappa}.
    """
    summary: Dict[str, Any] = {}
    for path in (
        "topology.n_ap", "topology.n_ue",
        "topology.n_targets", "topology.n_ant",
        "channel.snr_db",
        "algorithm.gamma_db",
        "algorithm.admm.kappa",
    ):
        try:
            cursor = cfg
            for p in path.split("."):
                cursor = getattr(cursor, p)
            summary[path] = cursor
        except AttributeError:
            pass
    return summary


def _get_n_ue(cfg: Any, default: int = 4) -> int:
    """Best-effort extraction of cfg.topology.n_ue with a sane default."""
    try:
        return int(cfg.topology.n_ue)
    except AttributeError:
        return default


# ─────────────────────────────────────────────────────────────────────
#  Single-config experiments
# ─────────────────────────────────────────────────────────────────────

def run_sinr_cdf(
    cfg, runner_cfg,
    *,
    n_drops: int = 50,
    n_realizations: int = 4,
    spec_kwargs: Optional[Dict[str, Any]] = None,
    spec_set: str = "all_algorithms",
) -> ExperimentResult:
    """Empirical CDF of min-SINR across N_drops × N_realisations trials."""
    factory = _resolve_spec_set(spec_set)
    spec_kwargs = spec_kwargs or {}
    spec_kwargs.setdefault("n_ue", _get_n_ue(cfg))
    specs = factory(**spec_kwargs)
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
            "spec_set": spec_set,
            "cfg_summary": _cfg_summary(cfg),
        },
    )


def run_scnr_cdf(
    cfg, runner_cfg,
    *,
    n_drops: int = 50,
    n_realizations: int = 4,
    spec_kwargs: Optional[Dict[str, Any]] = None,
    spec_set: str = "all_algorithms",
) -> ExperimentResult:
    """Empirical CDF of sum-SCNR (sensing figure-of-merit)."""
    factory = _resolve_spec_set(spec_set)
    spec_kwargs = spec_kwargs or {}
    spec_kwargs.setdefault("n_ue", _get_n_ue(cfg))
    specs = factory(**spec_kwargs)
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
            "spec_set": spec_set,
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
    spec_set: str = "cordis_vs_centralized",
) -> ExperimentResult:
    """Sweep per-UE SINR target γ ∈ {-3, 0, 3, 6, 10, 14, 18} dB."""
    factory = _resolve_spec_set(spec_set)
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
        cfg, factory, "gamma_u_db", axis, rc,
        n_ue=_get_n_ue(cfg),
    )
    return ExperimentResult(
        name="gamma_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "spec_set": spec_set,
                  "cfg_summary": _cfg_summary(cfg)},
    )


# ─────────────────────────────────────────────────────────────────────
#  kappa_sweep — needs custom loop because Centralized reads κ from cfg,
#  ADMM reads κ from spec.params (dispatcher gap).  Update BOTH per point.
# ─────────────────────────────────────────────────────────────────────

def run_kappa_sweep(
    cfg, runner_cfg,
    *,
    kappa_values: Optional[List[float]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    spec_set: str = "cordis_vs_centralized",
) -> ExperimentResult:
    """Sweep prox-regularisation weight κ ∈ [0, 2].

    Per joint_opt.py, κ is auto-scaled internally so user-facing κ=1
    means "strong clutter avoidance".  Useful range is [0, 2].

    Two places must be updated per sweep point:
      * ``cfg.algorithm.admm.kappa`` — read by ``solve_centralized``
        directly (the dispatcher does NOT forward κ for kind="centralized").
      * ``spec.params["kappa"]`` — forwarded by the dispatcher to
        ``solve_cordis_admm`` (the Stage-7 gap is that this isn't
        derived from cfg).
    """
    factory = _resolve_spec_set(spec_set)
    if kappa_values is None:
        kappa_values = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]
    axis = SweepAxis(
        name="kappa",
        values=kappa_values,
        display=r"$\kappa$",
    )
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    n_ue = _get_n_ue(cfg)

    results: Dict[float, Any] = {}
    for k in axis.values:
        cfg_v = _override_cfg(cfg, **{"algorithm.admm.kappa": float(k)})
        specs = factory(n_ue=n_ue, kappa=float(k))
        sr = _run_single(cfg_v, rc, specs)
        results[float(k)] = sr

    return ExperimentResult(
        name="kappa_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "spec_set": spec_set,
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
    field_path: str = "channel.snr_db",
    spec_set: str = "all_algorithms",
) -> ExperimentResult:
    """Sweep operating SNR (Pₘₐₓ / σ²) [dB]."""
    factory = _resolve_spec_set(spec_set)
    if snr_values_db is None:
        snr_values_db = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    axis = SweepAxis(
        name="snr_db",
        values=snr_values_db,
        display="SNR [dB]",
        unit="dB",
    )
    specs = factory(n_ue=_get_n_ue(cfg))
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_config_field(cfg, specs, field_path, axis, rc)
    return ExperimentResult(
        name="snr_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "field_path": field_path,
                  "spec_set": spec_set,
                  "cfg_summary": _cfg_summary(cfg)},
    )


def run_clutter_cnr_sweep(
    cfg, runner_cfg,
    *,
    cnr_values_db: Optional[List[float]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    field_path: str = "sensing.clutter_cnr_db",
    spec_set: str = "cordis_vs_centralized",
) -> ExperimentResult:
    """Sweep clutter-to-noise ratio (CNR) in dB.

    Probes the sensing/comm tension through the channel rather than
    through an algorithm knob: stronger clutter raises the
    clutter-plus-noise floor in ``R_g_a_r`` (Proposition 3), so at
    fixed κ the algorithm trades raw SCNR against SINR feasibility.
    A natural fixed-κ way to probe the journal-paper formulation,
    which has no λ-style scalar comm/sensing trade-off.
    """
    factory = _resolve_spec_set(spec_set)
    if cnr_values_db is None:
        cnr_values_db = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0]
    axis = SweepAxis(
        name="clutter_cnr_db",
        values=cnr_values_db,
        display="Clutter CNR [dB]",
        unit="dB",
    )
    specs = factory(n_ue=_get_n_ue(cfg))
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_config_field(cfg, specs, field_path, axis, rc)
    return ExperimentResult(
        name="clutter_cnr_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "field_path": field_path,
                  "spec_set": spec_set,
                  "cfg_summary": _cfg_summary(cfg)},
    )


def run_n_ue_sweep(
    cfg, runner_cfg,
    *,
    n_ue_values: Optional[List[int]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    field_path: str = "topology.n_ue",
    spec_set: str = "all_algorithms",
) -> ExperimentResult:
    """Sweep number of UEs (loading / scalability).

    Custom inner loop because n_ue affects the size of ``gamma_u_db``,
    so specs must be rebuilt per sweep point.
    """
    factory = _resolve_spec_set(spec_set)
    if n_ue_values is None:
        n_ue_values = [2, 4, 6, 8, 10]
    axis = SweepAxis(
        name="n_ue",
        values=[float(v) for v in n_ue_values],
        display=r"$N_{\rm UE}$",
    )
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results: Dict[float, Any] = {}
    for n_ue in axis.values:
        n_ue_int = int(n_ue)
        cfg_v = _override_cfg(cfg, **{field_path: n_ue_int})
        specs = factory(n_ue=n_ue_int)
        sr = _run_single(cfg_v, rc, specs)
        results[float(n_ue)] = sr
    return ExperimentResult(
        name="n_ue_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "spec_set": spec_set,
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
    spec_set: str = "all_algorithms",
) -> ExperimentResult:
    """Sweep number of APs (AP scalability)."""
    factory = _resolve_spec_set(spec_set)
    if n_ap_values is None:
        n_ap_values = [4, 6, 8, 10, 12]
    axis = SweepAxis(
        name="n_ap",
        values=[float(v) for v in n_ap_values],
        display=r"$N_{\rm AP}$",
    )
    specs = factory(n_ue=_get_n_ue(cfg))
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_config_field(cfg, specs, field_path, axis, rc)
    return ExperimentResult(
        name="n_ap_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "field_path": field_path,
                  "spec_set": spec_set,
                  "cfg_summary": _cfg_summary(cfg)},
    )


def run_antennas_sweep(
    cfg, runner_cfg,
    *,
    n_ant_values: Optional[List[int]] = None,
    n_drops: int = 20,
    n_realizations: int = 2,
    field_path: str = "topology.n_ant",
    spec_set: str = "cordis_vs_benchmarks",
) -> ExperimentResult:
    """Sweep antennas per AP (M)."""
    factory = _resolve_spec_set(spec_set)
    if n_ant_values is None:
        n_ant_values = [4, 6, 8, 10, 12]
    axis = SweepAxis(
        name="n_ant",
        values=[float(v) for v in n_ant_values],
        display=r"$M$",
    )
    specs = factory(n_ue=_get_n_ue(cfg))
    rc = _override_runner(runner_cfg,
                          n_drops=n_drops,
                          n_realizations_per_drop=n_realizations)
    results = sweep_config_field(cfg, specs, field_path, axis, rc)
    return ExperimentResult(
        name="antennas_sweep", kind="sweep",
        sweep_results=results, sweep_axis=axis,
        metadata={"n_drops": n_drops, "n_realizations": n_realizations,
                  "field_path": field_path,
                  "spec_set": spec_set,
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
    """
    from cordis.simulation.scenario import build_scenario_from_seeds
    from cordis.algorithms.joint_opt import run_cordis_admm

    spec_kwargs = spec_kwargs or {}
    spec_kwargs.setdefault("n_ue", _get_n_ue(cfg))
    spec_kwargs.setdefault("n_admm_max", n_admm_max)
    spec_obj = admm_spec(**spec_kwargs)

    scenario = build_scenario_from_seeds(
        cfg,
        drop_seed=drop_seed,
        realization_seed=realization_seed,
    )

    # Forward only the params the ADMM solver knows about.  Note:
    # warm_start_from_split is intentionally NOT in this list because
    # solve_cordis_admm does not accept it (Phase-I warm start is
    # always done internally — see joint_opt.py).
    fwd_keys = (
        "gamma_u_db", "omega", "rho_admm", "kappa",
        "xi_slack", "slack_tol",
        "n_admm_max", "eps_pri", "eps_dual",
        "n_snapshots", "verbose",
    )
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

    All sizes are journal-paper accurate:

    * **Centralized**: APs ship raw CSI and beamformers — H_a (M×N_ue
      complex) and W_a (M×|D| complex) per coordination round.
    * **CORDIS-Split** (Algorithm 1): each AP sends per-user scalars
      ``{β̂_au, g̃_au, e_au^(s)}`` plus the sensing scalars ``z_a, q̃_a``,
      then receives back ρ*_a.  Total per AP per round:
      ``3·N_ue + 3`` real scalars.
    * **CORDIS-ADMM** (Algorithm 2, Section V-C): each AP ships
      ``l_au ∈ ℂ^(|D|+1)`` to the CPU for every user u, and receives
      the broadcast residual ``Σ̃_u`` of the same structure.  The
      ``l_au`` vector is mostly complex (CDS scalar, MUI vector,
      S2CI vector) with one real entry (``e_au``), giving
      ``2(N_ue+N_t) + 1`` real scalars per ``l_au``.  Per AP per
      outer ADMM iteration, both directions combined:
      ``2 · N_ue · (2|D| + 1)`` real scalars.

    T_ADMM is estimated empirically by running ADMM for
    ``n_admm_drops`` drops and averaging the ``iters`` diagnostic.
    """
    # 1. Estimate average ADMM iterations.
    specs = [admm_spec(n_ue=_get_n_ue(cfg))]
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

    # 2. Read scalar topology dimensions from the canonical cfg paths.
    M     = int(getattr(cfg.topology, "n_ant",     0))
    n_ue  = int(getattr(cfg.topology, "n_ue",      0))
    n_tg  = int(getattr(cfg.topology, "n_targets", 0))
    n_streams = n_ue + n_tg                                   # = |D|

    iters_nan = admm_avg_iters != admm_avg_iters              # NaN check

    # 3. Per-iteration real-scalar counts.
    #    Centralized:    H_a (M×N_ue complex) + W_a (M×|D| complex)
    centralized_count = 2 * (M * n_ue + M * n_streams)
    #    Split:          β̂_au, g̃_au, e_au^(s) per user  +  z_a, q̃_a  +  ρ*_a
    split_count       = 3 * n_ue + 3
    #    ADMM:           upload l_au + download Σ̃_u, both ℂ^(|D|+1) with
    #                    one real entry (e_au) → 2(N_ue+N_t)+1 real per vector
    #                    × N_ue vectors × 2 directions × T_ADMM iterations
    admm_real_per_iter_per_user = 2 * n_streams + 1
    admm_per_iter               = 2 * n_ue * admm_real_per_iter_per_user
    admm_total = (None if iters_nan
                  else admm_per_iter * admm_avg_iters)

    # 4. Build table rows (journal-paper notation throughout).
    table = {
        "Centralized": {
            "data_to_share": r"$\widehat{\mathbf{H}}_{a_t},\ \mathbf{W}_{a_t}$",
            "size":          r"$\mathbb{C}^{M_t \times N_{\rm UE}} + \mathbb{C}^{M_t \times |\mathcal{D}|}$",
            "real_scalars":  centralized_count,
            "scales_with":   r"$M_t \cdot (N_{\rm UE} + |\mathcal{D}|)$",
            "scalable":      False,
        },
        "CORDIS-Split": {
            "data_to_share": r"$\{\widehat{\beta}_{a_t u}, \widetilde{g}_{a_t u}, e_{a_t u}^{(s)}\}_{u \in \mathcal{U}},\ z_{a_t}, \widetilde{q}_{a_t},\ \rho^\ast_{a_t}$",
            "size":          r"$(3 N_{\rm UE} + 3) \times \mathbb{R}$",
            "real_scalars":  split_count,
            "scales_with":   r"$N_{\rm UE}$",
            "scalable":      True,
        },
        "CORDIS-ADMM": {
            "data_to_share": r"$\mathbf{l}_{a_t u}\ (\mathrm{up}),\ \widetilde{\boldsymbol{\Sigma}}_u\ (\mathrm{dn})$",
            "size":          r"$2 \cdot N_{\rm UE} \cdot \mathbb{C}^{|\mathcal{D}|+1}$ per outer iter",
            "real_scalars":  admm_total,
            "scales_with":   r"$T_{\rm ADMM} \cdot N_{\rm UE} \cdot (2|\mathcal{D}| + 1)$",
            "scalable":      True,
        },
    }

    return ExperimentResult(
        name="fronthaul_table", kind="table",
        table_data=table,
        metadata={
            "M": M, "n_ue": n_ue, "n_targets": n_tg, "n_streams": n_streams,
            "admm_avg_iters": admm_avg_iters,
            "admm_per_iter_real_scalars": admm_per_iter,
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
    # Spec-factory sweep
    "gamma_sweep":       run_gamma_sweep,
    # Custom-loop sweep (cfg + spec)
    "kappa_sweep":       run_kappa_sweep,
    # Config-field sweeps
    "snr_sweep":         run_snr_sweep,
    "clutter_cnr_sweep": run_clutter_cnr_sweep,
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
    "run_gamma_sweep", "run_kappa_sweep", "run_clutter_cnr_sweep",
    "run_snr_sweep", "run_n_ue_sweep", "run_n_ap_sweep",
    "run_antennas_sweep",
    "run_convergence_trace", "run_fronthaul_table",
]

