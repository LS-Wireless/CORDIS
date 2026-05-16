"""
Stage 8b — Parameter sweep helpers.

A *sweep* runs the Monte-Carlo runner once per value of one swept
parameter, then aggregates the per-value :class:`SimResult`\\ s into a
``{value: SimResult}`` dictionary suitable for
:func:`cordis.plotting.plot_sweep`.

Two sweep flavours
------------------
:func:`sweep_config_field`
    Vary a scalar field somewhere in the SimConfig tree (e.g.
    ``"system.snr_db"``, ``"topology.n_ue"``).  The spec list stays
    fixed; only the config changes between sweep points.

:func:`sweep_spec_factory`
    Vary a kwarg passed to a spec-factory function (e.g.
    ``gamma_u_db``, ``kappa``, ``rho_admm``).  The config stays
    fixed; specs are rebuilt for each sweep point.

Both helpers accept a :class:`SweepAxis` that bundles the sweep values
with metadata used by plot helpers and table generators.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  SweepAxis
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SweepAxis:
    """
    Metadata describing one sweep axis.

    Attributes
    ----------
    name : str
        Canonical short name used in filenames and as the manifest key.
        Should be a valid Python identifier (e.g. ``"gamma_u_db"``,
        ``"n_ue"``, ``"snr_db"``).
    values : list of float
        The sweep points.  Numeric (no strings).  Sorted on save.
    display : str, optional
        LaTeX-friendly label for plot axes, e.g. ``r"$\\gamma$ [dB]"``.
        Defaults to ``name``.
    unit : str, optional
        Physical unit if any (``"dB"``, ``""``).  Free-form, used in
        figure captions.

    Notes
    -----
    The dataclass is JSON-serialisable via ``dataclasses.asdict`` (used
    by :meth:`ExperimentResult.save`); reload with ``SweepAxis(**d)``.
    """
    name:    str
    values:  List[float]
    display: Optional[str] = None
    unit:    Optional[str] = None

    def __post_init__(self):
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"SweepAxis.name must be a Python-style identifier, "
                f"got {self.name!r}"
            )
        if not self.values:
            raise ValueError("SweepAxis.values must be non-empty")
        # Ensure values are numeric and sorted.
        self.values = sorted(float(v) for v in self.values)
        if self.display is None:
            self.display = self.name


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _set_field(obj: Any, path: str, value: Any) -> None:
    """
    Set a nested attribute by dotted path: ``obj.a.b.c = value``.

    Raises
    ------
    AttributeError
        If any intermediate attribute does not exist on ``obj``.
    """
    parts = path.split(".")
    if not parts or not all(parts):
        raise ValueError(f"Invalid field path: {path!r}")
    cursor = obj
    for p in parts[:-1]:
        cursor = getattr(cursor, p)
    setattr(cursor, parts[-1], value)


def _get_field(obj: Any, path: str) -> Any:
    """Read a nested attribute by dotted path."""
    cursor = obj
    for p in path.split("."):
        cursor = getattr(cursor, p)
    return cursor


# ─────────────────────────────────────────────────────────────────────
#  Sweep helpers
# ─────────────────────────────────────────────────────────────────────

def sweep_config_field(
    base_cfg: Any,
    specs: List[Any],            # List[AlgorithmSpec]
    field_path: str,
    axis: SweepAxis,
    runner_cfg: Any,
    *,
    runner_cls: Optional[type] = None,
    sim_result_from_run: Optional[Callable] = None,
) -> Dict[float, Any]:
    """
    Sweep a config field; the same spec list is used at every point.

    Parameters
    ----------
    base_cfg : SimConfig
        Deep-copied before each sweep point so the original is unchanged.
    specs : list of AlgorithmSpec
    field_path : str
        Dotted path to the cfg field, e.g. ``"system.snr_db"``.
    axis : SweepAxis
    runner_cfg : RunnerConfig
    runner_cls : type, optional
        Defaults to :class:`cordis.simulation.runner.MonteCarloRunner`.
        Inject your own (e.g. a mock) for testing.
    sim_result_from_run : callable, optional
        Defaults to ``SimResult.from_run``.  Inject for testing.

    Returns
    -------
    Dict[float, SimResult]
        Keyed by sweep value.
    """
    if runner_cls is None:
        from cordis.simulation.runner import MonteCarloRunner
        runner_cls = MonteCarloRunner
    if sim_result_from_run is None:
        from cordis.simulation.result import SimResult
        sim_result_from_run = SimResult.from_run

    results: Dict[float, Any] = {}
    for v in axis.values:
        cfg = copy.deepcopy(base_cfg)
        _set_field(cfg, field_path, v)
        logger.info("Sweep %s = %g (config-field %s)", axis.name, v, field_path)
        runner = runner_cls(cfg, specs, runner_cfg)
        report = runner.run()
        results[float(v)] = sim_result_from_run(report)
    return results


def sweep_spec_factory(
    base_cfg: Any,
    spec_factory: Callable[..., List[Any]],
    factory_kwarg: str,
    axis: SweepAxis,
    runner_cfg: Any,
    *,
    runner_cls: Optional[type] = None,
    sim_result_from_run: Optional[Callable] = None,
    **factory_extra,
) -> Dict[float, Any]:
    """
    Sweep a spec-factory kwarg; the config stays fixed.

    Parameters
    ----------
    base_cfg : SimConfig
    spec_factory : callable returning list[AlgorithmSpec]
        E.g. :func:`cordis.experiments.specs.cordis_only`.
    factory_kwarg : str
        Name of the kwarg whose value sweeps.  E.g. ``"gamma_u_db"``,
        ``"kappa"``, ``"rho_admm"``.
    axis : SweepAxis
    runner_cfg : RunnerConfig
    runner_cls, sim_result_from_run :
        Injection points for testing (defaults below).
    **factory_extra
        Additional kwargs passed unchanged to ``spec_factory`` at each
        point.  Use these to pin all OTHER spec-factory parameters
        (e.g. ``n_ue=4`` while sweeping ``gamma_u_db``).

    Returns
    -------
    Dict[float, SimResult]
    """
    if runner_cls is None:
        from cordis.simulation.runner import MonteCarloRunner
        runner_cls = MonteCarloRunner
    if sim_result_from_run is None:
        from cordis.simulation.result import SimResult
        sim_result_from_run = SimResult.from_run

    results: Dict[float, Any] = {}
    for v in axis.values:
        spec_kwargs = {**factory_extra, factory_kwarg: v}
        specs = spec_factory(**spec_kwargs)
        logger.info("Sweep %s = %g (factory %s, %d specs)",
                    axis.name, v, spec_factory.__name__, len(specs))
        runner = runner_cls(base_cfg, specs, runner_cfg)
        report = runner.run()
        results[float(v)] = sim_result_from_run(report)
    return results


__all__ = [
    "SweepAxis",
    "sweep_config_field",
    "sweep_spec_factory",
]

