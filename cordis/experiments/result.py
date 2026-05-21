"""
Stage 8b — :class:`ExperimentResult` dataclass.

Wraps the output of any registered experiment so that downstream code
(plot scripts, validation, batch orchestration) has a single, uniform
interface for saving, loading, and introspecting results.

Four output kinds are supported:

* ``"single"`` — one :class:`SimResult` (e.g. SINR-CDF figure)
* ``"sweep"``  — :class:`Dict[float, SimResult]` plus a :class:`SweepAxis`
                 describing the swept parameter (e.g. γ-sweep, SNR-sweep)
* ``"trace"``  — single-trial :class:`ADMMResult`-like object whose
                 ``primal_res_history`` / ``dual_res_history`` /
                 ``slack_history`` are saved as raw arrays (used by
                 the convergence-trajectory figure)
* ``"table"``  — plain Python dict (or nested dict) serialised as JSON
                 (used by the fronthaul-overhead table)

Each kind serialises into the same directory layout::

    <experiment_dir>/
    ├── manifest.json         ← always present: name, kind, metadata, axis
    ├── result.npz            ← kind="single"
    ├── result_<axis>_<v>.npz ← kind="sweep" (one per sweep value)
    ├── trace.npz             ← kind="trace"
    └── table.json            ← kind="table"

So :meth:`load` can discriminate kinds by reading the manifest and
loading the matching artefact(s) — no kind-specific logic needed in
the caller.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  JSON encoder that handles numpy + pathlib types
# ─────────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalars / arrays and pathlib.Path."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────
#  Lightweight loaded-ADMM-result for "trace" kind
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LoadedADMMResult:
    """Minimal duck-typed stand-in for ADMMResult loaded from disk.

    Mirrors the attributes that :func:`cordis.plotting.plot_admm_convergence`
    and the trace-notebook diagnostic cells read.

    Fields beyond the four convergence-plot basics
    (``primal_res_history``, ``dual_res_history``, ``slack_history``,
    ``best_iter``) are written by Stage-18-diag-v2 onwards.  Loading
    a pre-v2 trace leaves them as their dataclass defaults — None for
    optional arrays, False/0 for scalars — and the diagnostic cells
    treat that gracefully.
    """
    primal_res_history:  np.ndarray
    dual_res_history:    np.ndarray
    slack_history:       Optional[np.ndarray] = None
    best_iter:           Optional[int]        = None
    # ── extended diagnostic fields (Stage-18-diag-v2) ──
    sinr_history:        Optional[np.ndarray] = None
    sensing_obj_history: Optional[np.ndarray] = None
    z_norm_history:      Optional[np.ndarray] = None
    converged:           bool                 = False
    feasible:            bool                 = True
    n_admm_iters:        int                  = 0
    inner_failures:      int                  = 0


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _fmt_value(v: float) -> str:
    """Format a sweep value for use in a filename.

    Avoid characters that confuse filesystems while keeping the value
    human-readable.  Integers stay integers; floats render with up to
    6 significant digits and ``.`` replaced by ``p`` (e.g. ``3p14``).
    """
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return str(int(v))
    return f"{v:.6g}".replace(".", "p").replace("-", "m")


def _value_filename(axis_name: str, value: float) -> str:
    """Build ``result_<axis>_<v>.npz``."""
    return f"result_{axis_name}_{_fmt_value(value)}.npz"


# ─────────────────────────────────────────────────────────────────────
#  ExperimentResult
# ─────────────────────────────────────────────────────────────────────

VALID_KINDS = ("single", "sweep", "trace", "table")


@dataclass
class ExperimentResult:
    """
    Output of one experiment run.

    Discriminate by ``kind``: only the matching payload field is set,
    the others are ``None``.
    """
    name:           str
    kind:           str    # one of VALID_KINDS

    sim_result:     Optional[Any] = None             # kind="single"

    sweep_results:  Optional[Dict[float, Any]] = None  # kind="sweep"
    sweep_axis:     Optional[Any] = None             # kind="sweep"; SweepAxis

    admm_result:    Optional[Any] = None             # kind="trace"

    table_data:     Optional[Dict[str, Any]] = None  # kind="table"

    metadata:       Dict[str, Any] = field(default_factory=dict)

    # ── Validation ────────────────────────────────────────────────────

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"ExperimentResult.kind must be one of {VALID_KINDS}; "
                f"got {self.kind!r}"
            )
        # Sanity checks: kind agrees with which payload field is set.
        payload = {
            "single": self.sim_result,
            "sweep":  self.sweep_results,
            "trace":  self.admm_result,
            "table":  self.table_data,
        }[self.kind]
        if payload is None:
            raise ValueError(
                f"ExperimentResult(kind={self.kind!r}) requires the "
                f"matching payload field to be set."
            )
        if self.kind == "sweep" and self.sweep_axis is None:
            raise ValueError(
                "ExperimentResult(kind='sweep') requires sweep_axis."
            )

    # ── Save ──────────────────────────────────────────────────────────

    def save(self, exp_dir: Union[str, Path]) -> Path:
        """
        Write all artefacts (manifest + payload files) to ``exp_dir``.

        Returns
        -------
        Path
            The directory that was written to (already exists).
        """
        exp_dir = Path(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, Any] = {
            "name":     self.name,
            "kind":     self.kind,
            "metadata": self.metadata,
        }

        if self.kind == "single":
            self.sim_result.save(exp_dir / "result.npz")

        elif self.kind == "sweep":
            for v, sr in self.sweep_results.items():
                sr.save(exp_dir / _value_filename(self.sweep_axis.name, v))
            manifest["axis"] = dataclasses.asdict(self.sweep_axis)

        elif self.kind == "trace":
            arrs: Dict[str, Any] = {
                "primal_res_history": np.asarray(
                    self.admm_result.primal_res_history, dtype=float),
                "dual_res_history": np.asarray(
                    self.admm_result.dual_res_history, dtype=float),
            }
            slack = getattr(self.admm_result, "slack_history", None)
            if slack is not None and len(slack) > 0:
                arrs["slack_history"] = np.asarray(slack, dtype=float)
            best_iter = getattr(self.admm_result, "best_iter", None)
            if best_iter is not None:
                arrs["best_iter"] = np.asarray(int(best_iter))
            # ── Extended diagnostic fields (Stage-18-diag-v2) ──
            # Per-iter per-user SINR vectors, sensing-objective scalar,
            # consensus-vector norm — used by the per-user diagnostic
            # cells in playground_trace.ipynb to identify "stuck users"
            # when the warm-start is poor.
            for attr in ("sinr_history",
                         "sensing_obj_history",
                         "z_norm_history"):
                v = getattr(self.admm_result, attr, None)
                if v is not None and len(v) > 0:
                    arrs[attr] = np.asarray(v, dtype=float)
            # Scalar status — small, always persisted when present.
            for attr, cast in (("converged",      lambda x: int(bool(x))),
                               ("feasible",       lambda x: int(bool(x))),
                               ("n_admm_iters",   int),
                               ("inner_failures", int)):
                if hasattr(self.admm_result, attr):
                    arrs[attr] = np.asarray(cast(getattr(self.admm_result, attr)))
            np.savez_compressed(exp_dir / "trace.npz", **arrs)

        elif self.kind == "table":
            with open(exp_dir / "table.json", "w") as f:
                json.dump(self.table_data, f, indent=2, cls=_NpEncoder)

        with open(exp_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, cls=_NpEncoder)

        logger.info("Saved experiment %r (kind=%s) to %s",
                    self.name, self.kind, exp_dir)
        return exp_dir

    # ── Load ──────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        exp_dir: Union[str, Path],
        sim_result_loader: Optional[Callable] = None,
    ) -> "ExperimentResult":
        """
        Load artefacts from ``exp_dir``.

        Parameters
        ----------
        exp_dir : path
            Directory created by a previous :meth:`save`.
        sim_result_loader : callable, optional
            A function ``(path) -> SimResult`` used to deserialise
            ``result*.npz`` files.  Defaults to
            ``cordis.simulation.result.SimResult.load``.  Inject your
            own (e.g. a mock) for testing.

        Returns
        -------
        ExperimentResult
        """
        exp_dir = Path(exp_dir)
        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json in {exp_dir} — not an experiment directory."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        name = manifest["name"]
        kind = manifest["kind"]
        metadata = manifest.get("metadata", {}) or {}

        if sim_result_loader is None:
            # Late import to avoid circular dependency at module load.
            from cordis.simulation.result import SimResult
            sim_result_loader = SimResult.load

        if kind == "single":
            sr = sim_result_loader(exp_dir / "result.npz")
            return cls(name=name, kind="single",
                       sim_result=sr, metadata=metadata)

        if kind == "sweep":
            # Late import for SweepAxis to avoid circular dep.
            from cordis.experiments.sweeps import SweepAxis
            axis = SweepAxis(**manifest["axis"])
            sweep_results: Dict[float, Any] = {}
            for v in axis.values:
                p = exp_dir / _value_filename(axis.name, v)
                sweep_results[float(v)] = sim_result_loader(p)
            return cls(name=name, kind="sweep",
                       sweep_results=sweep_results, sweep_axis=axis,
                       metadata=metadata)

        if kind == "trace":
            data = np.load(exp_dir / "trace.npz")
            files = set(data.files)
            admm = LoadedADMMResult(
                primal_res_history=data["primal_res_history"],
                dual_res_history=data["dual_res_history"],
                slack_history=(data["slack_history"]
                               if "slack_history" in files else None),
                best_iter=(int(data["best_iter"])
                           if "best_iter" in files else None),
                # ── Extended fields (Stage-18-diag-v2).  Legacy traces
                # saved before this addition simply don't contain them
                # and we fall back to dataclass defaults.
                sinr_history=(data["sinr_history"]
                              if "sinr_history" in files else None),
                sensing_obj_history=(data["sensing_obj_history"]
                                     if "sensing_obj_history" in files
                                     else None),
                z_norm_history=(data["z_norm_history"]
                                if "z_norm_history" in files else None),
                converged=(bool(int(data["converged"]))
                           if "converged" in files else False),
                feasible=(bool(int(data["feasible"]))
                          if "feasible" in files else True),
                n_admm_iters=(int(data["n_admm_iters"])
                              if "n_admm_iters" in files else 0),
                inner_failures=(int(data["inner_failures"])
                                if "inner_failures" in files else 0),
            )
            return cls(name=name, kind="trace",
                       admm_result=admm, metadata=metadata)

        if kind == "table":
            with open(exp_dir / "table.json") as f:
                table = json.load(f)
            return cls(name=name, kind="table",
                       table_data=table, metadata=metadata)

        raise ValueError(f"Unknown kind {kind!r} in manifest")

