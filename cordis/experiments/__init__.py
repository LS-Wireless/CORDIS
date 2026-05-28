"""
CORDIS experiments package.

This package bundles five orthogonal pieces of the experiment workflow:

* :mod:`cordis.experiments.io`       — output-path conventions and the
  ``exp_`` prefix (Stage 8a).
* :mod:`cordis.experiments.specs`    — canonical algorithm-spec
  builders and five spec-set factories that compose them.
* :mod:`cordis.experiments.sweeps`   — :class:`SweepAxis` plus two
  sweep helpers (vary a config field, vary a spec-factory kwarg).
* :mod:`cordis.experiments.result`   — :class:`ExperimentResult` with
  uniform save/load across four output kinds (single / sweep / trace
  / table).
* :mod:`cordis.experiments.registry` — :data:`REGISTRY` mapping
  experiment names to ``run_*`` callables; twelve experiments
  pre-registered, covering CDFs, parameter sweeps, ADMM convergence
  trajectories, and fronthaul-overhead tables.

Typical end-to-end use::

    from cordis.experiments import (
        experiment_dir, log_dir, latest_result,
        REGISTRY, ExperimentResult,
    )

    exp_dir = experiment_dir("gamma_sweep")        # results/exp_gamma_sweep/<ts>/
    log_path = log_dir(exp_dir) / "run.log"

    result = REGISTRY["gamma_sweep"](cfg, runner_cfg)
    result.save(exp_dir)

Later, the matching plot script does::

    result = ExperimentResult.load(latest_result("gamma_sweep"))
    plot_sweep(result.sweep_results, result.sweep_axis.display, "min_sinr_db")
"""
# ── Stage 8a: output path conventions ────────────────────────────────
from cordis.experiments.io import (
    experiment_dir,
    figure_dir,
    latest_result,
    log_dir,
)

# ── Stage 8b: experiment helper library ──────────────────────────────
from cordis.experiments.result import (
    ExperimentResult,
    LoadedADMMResult,
    VALID_KINDS,
)

from cordis.experiments.specs import (
    # individual spec builders
    split_spec, admm_spec, centralized_spec,
    mrt_spec, zf_spec, rzf_spec, lrmmse_spec,
    global_mrt_spec, global_zf_spec,
    # PSR-baseline variants (paper "benefit of optimal PSR" figure)
    lrmmse_p020_spec, lrmmse_p080_spec, lrmmse_split_spec,
    # spec set factories
    cordis_only, cordis_vs_centralized,
    cordis_vs_benchmarks, all_algorithms,
    psr_baselines,
)

from cordis.experiments.sweeps import (
    SweepAxis,
    sweep_config_field,
    sweep_spec_factory,
)

from cordis.experiments.registry import (
    REGISTRY,
    list_experiments,
    list_spec_sets,
    get_experiment,
    # convenience direct exports
    run_sinr_cdf, run_scnr_cdf,
    run_gamma_sweep, run_kappa_sweep, run_clutter_cnr_sweep,
    run_csi_sweep,
    run_snr_sweep, run_n_ue_sweep, run_n_ap_sweep,
    run_antennas_sweep,
    run_convergence_trace, run_fronthaul_table,
)


__all__ = [
    # io
    "experiment_dir", "figure_dir", "latest_result", "log_dir",
    # result
    "ExperimentResult", "LoadedADMMResult", "VALID_KINDS",
    # specs — builders (no DEFAULT_* exports — config is the single
    # source of truth for every parameter)
    "split_spec", "admm_spec", "centralized_spec",
    "mrt_spec", "zf_spec", "rzf_spec", "lrmmse_spec",
    "global_mrt_spec", "global_zf_spec",
    # specs — PSR baseline variants
    "lrmmse_p020_spec", "lrmmse_p080_spec", "lrmmse_split_spec",
    # specs — set factories
    "cordis_only", "cordis_vs_centralized",
    "cordis_vs_benchmarks", "all_algorithms",
    "psr_baselines",
    # sweeps
    "SweepAxis", "sweep_config_field", "sweep_spec_factory",
    # registry
    "REGISTRY", "list_experiments", "list_spec_sets", "get_experiment",
    "run_sinr_cdf", "run_scnr_cdf",
    "run_gamma_sweep", "run_kappa_sweep", "run_clutter_cnr_sweep",
    "run_csi_sweep",
    "run_snr_sweep", "run_n_ue_sweep", "run_n_ap_sweep",
    "run_antennas_sweep",
    "run_convergence_trace", "run_fronthaul_table",
]

