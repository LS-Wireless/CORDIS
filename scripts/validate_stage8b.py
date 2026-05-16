#!/usr/bin/env python3
"""
Stage 8b — Validation script for the experiment helper library.

Fifteen tests covering:

  1–4   :mod:`cordis.experiments.specs`   (individual builders, set
                                            factories, display names)
  5–7   :mod:`cordis.experiments.sweeps`  (SweepAxis dataclass, dotted
                                            field set/get, sweep wiring)
  8–11  :mod:`cordis.experiments.result`  (ExperimentResult save/load
                                            round-trip for single, sweep,
                                            trace, and table kinds)
  12–13 :mod:`cordis.experiments.registry`(11 names registered;
                                            run_* signatures intact)
  14–15 Integration smoke tests (run a tiny gamma-sweep + sinr-cdf via
                                  the registry IF the full CORDIS install
                                  is present, otherwise SKIP)

Tests requiring a full CORDIS install raise :class:`_SkipTest` if their
imports fail, mirroring the Stage 8a skip pattern for missing TeX/pypdf.

Run on the user's machine with::

    $ python3 scripts/validate_stage8b.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Make the in-tree cordis package importable when run from repo root ─
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────
#  Test runner infrastructure (mirrors Stage 8a)
# ─────────────────────────────────────────────────────────────────────

class _SkipTest(Exception):
    """Raised by a test when its preconditions (e.g. a heavy import) fail."""


_TESTS: List[Tuple[str, callable]] = []


def _register(label: str):
    """Decorator factory: ``@_register("label")`` registers a test."""
    def deco(fn):
        _TESTS.append((label, fn))
        return fn
    return deco


def _section(text: str) -> None:
    print(f"── {text} ──")


# ─────────────────────────────────────────────────────────────────────
#  Imports — done at module-load so syntax errors fail fast
# ─────────────────────────────────────────────────────────────────────

from cordis.experiments import (         # noqa: E402
    # specs
    split_spec, admm_spec, centralized_spec,
    mrt_spec, zf_spec, rzf_spec, lrmmse_spec,
    global_mrt_spec, global_zf_spec,
    cordis_only, cordis_vs_centralized,
    cordis_vs_benchmarks, all_algorithms,
    DEFAULT_GAMMA_DB, DEFAULT_KAPPA,
    # sweeps
    SweepAxis, sweep_config_field, sweep_spec_factory,
    # result
    ExperimentResult, LoadedADMMResult, VALID_KINDS,
    # registry
    REGISTRY, list_experiments, get_experiment,
    # io (for the integration tests)
    experiment_dir, log_dir,
)
from cordis.experiments.specs import _DISPLAY, _BENCHMARK_NAME  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
#  Tests 1–4: specs.py
# ─────────────────────────────────────────────────────────────────────

@_register("Test  1: individual spec builders construct valid specs")
def test_01_individual_specs():
    """Each spec builder returns an AlgorithmSpec with correct (name, kind),
    a length-n_ue gamma_u_db vector, and NONE of the keys that broke
    Stage 8b integration (omega, warm_start, warm_start_from_split,
    use_cvxpy)."""
    builders = {
        "split":       (split_spec,       "CORDIS-Split",  "cordis_split", None),
        "admm":        (admm_spec,        "CORDIS-ADMM",   "cordis_admm",  None),
        "centralized": (centralized_spec, "Centralized",   "centralized",  None),
        "mrt":         (mrt_spec,         "MRT-Split",     "benchmark",    "mrt_split"),
        "zf":          (zf_spec,          "ZF-Split",      "benchmark",    "zf_split"),
        "rzf":         (rzf_spec,         "RZF-Split",     "benchmark",    "rzf_split"),
        "lrmmse":      (lrmmse_spec,      "LR-MMSE-Split", "benchmark",    "lr_mmse_split"),
        "global_mrt":  (global_mrt_spec,  "Global-MRT",    "benchmark",    "global_mrt_split"),
        "global_zf":   (global_zf_spec,   "Global-ZF",     "benchmark",    "global_zf_split"),
    }
    BANNED_KEYS = {"omega", "warm_start", "warm_start_from_split", "use_cvxpy"}
    for key, (fn, want_name, want_kind, want_bn) in builders.items():
        # Build with n_ue=4 so gamma_u_db becomes a length-4 vector.
        spec = fn(n_ue=4, gamma_u_db=10.0, kappa=1.0)
        assert spec.name == want_name, f"{key}: name = {spec.name!r}, want {want_name!r}"
        assert spec.kind == want_kind, f"{key}: kind = {spec.kind!r}, want {want_kind!r}"
        if want_bn is not None:
            bn = spec.params.get("benchmark_name")
            assert bn == want_bn, f"{key}: benchmark_name = {bn!r}, want {want_bn!r}"
        # gamma_u_db must be an array of length n_ue (downstream code indexes it).
        g = spec.params.get("gamma_u_db")
        assert isinstance(g, np.ndarray) and g.shape == (4,) and float(g[0]) == 10.0, \
            f"{key}: gamma_u_db = {g!r}, want length-4 array of 10.0"
        # None of the banned keys must be present — passing any of them
        # would error in the real algorithms (the Stage 8b integration bugs).
        present_banned = BANNED_KEYS & set(spec.params)
        assert not present_banned, \
            f"{key}: spec.params still contains banned keys {present_banned}"
    # gamma_u_db propagates from the kwarg and ignored kwargs really are ignored.
    s = split_spec(n_ue=3, gamma_u_db=7.0, this_kwarg_does_not_exist=42)
    assert float(s.params["gamma_u_db"][0]) == 7.0
    assert s.params["gamma_u_db"].shape == (3,)


@_register("Test  2: ADMM spec carries kappa explicitly (Stage 7 fix)")
def test_02_admm_kappa_explicit():
    """admm_spec must put kappa in params so the dispatcher forwards it,
    and the default must be non-zero (matching configs/default.json)."""
    s = admm_spec(n_ue=4, kappa=0.25, gamma_u_db=4.0, rho_admm=12.0, n_admm_max=15)
    assert "kappa" in s.params and s.params["kappa"] == 0.25
    assert s.params["rho_admm"] == 12.0
    assert s.params["n_admm_max"] == 15
    # Default kappa is non-zero (the Stage-7 bug was κ defaulting to 0 in the
    # function signature; cfg.algorithm.admm.kappa = 1.0).
    s_default = admm_spec(n_ue=4)
    assert s_default.params["kappa"] == DEFAULT_KAPPA
    assert s_default.params["kappa"] > 0


@_register("Test  3: spec set factories return correct cardinality and names")
def test_03_spec_set_factories():
    """4 set factories must return 2/3/6/9 specs in the expected order."""
    sets = {
        "cordis_only":           (cordis_only,           2,
                                  ["CORDIS-Split", "CORDIS-ADMM"]),
        "cordis_vs_centralized": (cordis_vs_centralized, 3,
                                  ["CORDIS-Split", "CORDIS-ADMM", "Centralized"]),
        "cordis_vs_benchmarks":  (cordis_vs_benchmarks,  6,
                                  ["CORDIS-Split", "CORDIS-ADMM",
                                   "MRT-Split", "ZF-Split",
                                   "RZF-Split", "LR-MMSE-Split"]),
        "all_algorithms":        (all_algorithms,        9,
                                  ["CORDIS-Split", "CORDIS-ADMM", "Centralized",
                                   "MRT-Split", "ZF-Split",
                                   "RZF-Split", "LR-MMSE-Split",
                                   "Global-MRT", "Global-ZF"]),
    }
    for label, (fn, want_n, want_names) in sets.items():
        specs = fn(n_ue=4, gamma_u_db=3.0, kappa=0.1)
        assert len(specs) == want_n, \
            f"{label}: got {len(specs)} specs, want {want_n}"
        got = [s.name for s in specs]
        assert got == want_names, \
            f"{label}: order = {got}, want {want_names}"
        # Uniqueness within each set (required by MonteCarloRunner).
        assert len(set(got)) == len(got), \
            f"{label}: duplicate display name in {got}"


@_register("Test  4: display names match ALGORITHM_STYLE keys")
def test_04_display_matches_style():
    """Every display name in _DISPLAY must be in ALGORITHM_STYLE."""
    try:
        from cordis.plotting.style import ALGORITHM_STYLE
    except ImportError as e:
        raise _SkipTest(f"cordis.plotting.style not importable: {e}")
    for key, disp in _DISPLAY.items():
        assert disp in ALGORITHM_STYLE, (
            f"_DISPLAY[{key!r}] = {disp!r} is not a key of "
            f"ALGORITHM_STYLE. Style keys: {list(ALGORITHM_STYLE)}"
        )


# ─────────────────────────────────────────────────────────────────────
#  Tests 5–7: sweeps.py
# ─────────────────────────────────────────────────────────────────────

@_register("Test  5: SweepAxis validation + sort + JSON round-trip")
def test_05_sweep_axis():
    a = SweepAxis(name="gamma_u_db", values=[10.0, -3.0, 3.0],
                  display=r"$\gamma$ [dB]", unit="dB")
    # values are sorted on construction
    assert a.values == [-3.0, 3.0, 10.0]
    # display defaults to name when omitted
    b = SweepAxis(name="kappa", values=[0.1])
    assert b.display == "kappa"
    # JSON round-trip
    d = dataclasses.asdict(a)
    s = json.dumps(d)
    a2 = SweepAxis(**json.loads(s))
    assert a2.name == a.name and a2.values == a.values
    assert a2.display == a.display and a2.unit == a.unit
    # bad inputs
    for bad in dict(name="", values=[1.0]), dict(name="a", values=[]):
        try:
            SweepAxis(**bad)
            assert False, f"expected ValueError for {bad!r}"
        except (ValueError, TypeError):
            pass


@_register("Test  6: dotted-path field setters round-trip")
def test_06_set_field():
    from cordis.experiments.sweeps import _set_field, _get_field
    import types
    obj = types.SimpleNamespace(
        system=types.SimpleNamespace(snr_db=10.0, n_ant=8),
        topology=types.SimpleNamespace(n_ue=4, n_ap=6),
    )
    _set_field(obj, "system.snr_db", 20.0)
    assert obj.system.snr_db == 20.0
    assert _get_field(obj, "system.snr_db") == 20.0
    _set_field(obj, "topology.n_ue", 8)
    assert _get_field(obj, "topology.n_ue") == 8
    # Empty / malformed paths fail loud.
    for bad in ("", "..", "system."):
        try:
            _set_field(obj, bad, 0)
            assert False, f"expected error for path {bad!r}"
        except (ValueError, AttributeError):
            pass


@_register("Test  7: sweep_config_field + sweep_spec_factory wiring (mocked)")
def test_07_sweep_wiring():
    """Verify both sweep helpers iterate over axis.values and rebuild
    config / specs at every point, using injected mocks for the runner
    and SimResult.from_run."""
    import types

    class _MockRunner:
        instances = []
        def __init__(self, cfg, specs, runner_cfg):
            self.cfg, self.specs, self.runner_cfg = cfg, specs, runner_cfg
            _MockRunner.instances.append(self)
        def run(self):
            # Return a tag so we can verify per-point cfg/specs identity.
            return ("REPORT", self.cfg, [s.name for s in self.specs])

    def _mock_from_run(report):
        _, cfg, names = report
        return {"cfg": cfg, "names": names}

    base = types.SimpleNamespace(system=types.SimpleNamespace(snr_db=0.0))
    axis = SweepAxis(name="snr_db", values=[-5.0, 0.0, 5.0])
    specs = [split_spec()]

    _MockRunner.instances = []
    out = sweep_config_field(
        base, specs, "system.snr_db", axis, runner_cfg=None,
        runner_cls=_MockRunner, sim_result_from_run=_mock_from_run,
    )
    assert sorted(out.keys()) == [-5.0, 0.0, 5.0]
    # Confirm cfg was deep-copied (mutations don't leak back to base).
    assert base.system.snr_db == 0.0
    # And that the field was actually set on each per-point cfg.
    for v, sr in out.items():
        assert sr["cfg"].system.snr_db == v, (v, sr)

    # spec-factory sweep: vary gamma_u_db
    _MockRunner.instances = []
    out2 = sweep_spec_factory(
        base, cordis_only, "gamma_u_db", axis, runner_cfg=None,
        runner_cls=_MockRunner, sim_result_from_run=_mock_from_run,
        omega=0.5,
    )
    assert sorted(out2.keys()) == [-5.0, 0.0, 5.0]
    for v, sr in out2.items():
        assert sr["names"] == ["CORDIS-Split", "CORDIS-ADMM"]


# ─────────────────────────────────────────────────────────────────────
#  Tests 8–11: ExperimentResult save/load round-trip (all four kinds)
# ─────────────────────────────────────────────────────────────────────

class _FakeSimResult:
    """Stand-in SimResult: save/load via simple JSON, no numpy."""
    def __init__(self, payload):
        self.payload = payload

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # mirror the real SimResult.save: a .npz + a .json sidecar
        p.write_bytes(b"FAKE_NPZ")
        with open(p.with_suffix(".json"), "w") as f:
            json.dump({"payload": self.payload}, f)

    @classmethod
    def load(cls, path):
        with open(Path(path).with_suffix(".json")) as f:
            return cls(json.load(f)["payload"])


def _fake_loader(path):
    return _FakeSimResult.load(path)


@_register("Test  8: ExperimentResult single-kind save/load round-trip")
def test_08_result_single():
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "exp"
        res = ExperimentResult(
            name="sinr_cdf", kind="single",
            sim_result=_FakeSimResult("hello"),
            metadata={"n_drops": 50},
        )
        res.save(out_dir)
        assert (out_dir / "result.npz").exists()
        assert (out_dir / "manifest.json").exists()
        loaded = ExperimentResult.load(out_dir, sim_result_loader=_fake_loader)
        assert loaded.kind == "single"
        assert loaded.name == "sinr_cdf"
        assert loaded.sim_result.payload == "hello"
        assert loaded.metadata["n_drops"] == 50


@_register("Test  9: ExperimentResult sweep-kind save/load round-trip")
def test_09_result_sweep():
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "exp"
        axis = SweepAxis(name="gamma_u_db", values=[0.0, 3.0, 10.0],
                         display=r"$\gamma$ [dB]", unit="dB")
        results = {v: _FakeSimResult(f"v{v}") for v in axis.values}
        res = ExperimentResult(
            name="gamma_sweep", kind="sweep",
            sweep_results=results, sweep_axis=axis,
            metadata={"spec_set": "cordis_vs_centralized"},
        )
        res.save(out_dir)
        # one .npz per sweep value (plus its .json sidecar from _FakeSimResult)
        npz_files = sorted(p.name for p in out_dir.glob("result_*.npz"))
        assert len(npz_files) == 3, npz_files
        loaded = ExperimentResult.load(out_dir, sim_result_loader=_fake_loader)
        assert loaded.kind == "sweep"
        assert loaded.sweep_axis.name == "gamma_u_db"
        assert sorted(loaded.sweep_results.keys()) == axis.values
        for v in axis.values:
            assert loaded.sweep_results[v].payload == f"v{v}"


@_register("Test 10: ExperimentResult trace-kind save/load round-trip")
def test_10_result_trace():
    import types
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "exp"
        admm = types.SimpleNamespace(
            primal_res_history=np.array([1.0, 0.5, 0.25]),
            dual_res_history=np.array([0.8, 0.4, 0.2]),
            slack_history=np.array([0.5, 0.3, 0.1]),
            best_iter=2,
        )
        res = ExperimentResult(
            name="convergence_trace", kind="trace",
            admm_result=admm,
            metadata={"drop_seed": 42},
        )
        res.save(out_dir)
        assert (out_dir / "trace.npz").exists()
        loaded = ExperimentResult.load(out_dir, sim_result_loader=_fake_loader)
        assert loaded.kind == "trace"
        assert isinstance(loaded.admm_result, LoadedADMMResult)
        np.testing.assert_allclose(loaded.admm_result.primal_res_history,
                                   [1.0, 0.5, 0.25])
        np.testing.assert_allclose(loaded.admm_result.dual_res_history,
                                   [0.8, 0.4, 0.2])
        assert loaded.admm_result.best_iter == 2


@_register("Test 11: ExperimentResult table-kind save/load round-trip")
def test_11_result_table():
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "exp"
        table = {
            "Centralized":  {"real_scalars": 280, "scalable": False},
            "CORDIS-Split": {"real_scalars": 4,   "scalable": True},
            "CORDIS-ADMM":  {"real_scalars": 70,  "scalable": True},
        }
        res = ExperimentResult(
            name="fronthaul_table", kind="table",
            table_data=table,
            metadata={"M": 10, "n_ue": 6},
        )
        res.save(out_dir)
        assert (out_dir / "table.json").exists()
        loaded = ExperimentResult.load(out_dir, sim_result_loader=_fake_loader)
        assert loaded.kind == "table"
        assert loaded.table_data == table
        assert loaded.metadata["M"] == 10


# ─────────────────────────────────────────────────────────────────────
#  Tests 12–13: Registry
# ─────────────────────────────────────────────────────────────────────

@_register("Test 12: all 11 experiments registered + get_experiment works")
def test_12_registry_contents():
    expected = {
        "sinr_cdf", "scnr_cdf",
        "gamma_sweep", "kappa_sweep", "clutter_cnr_sweep",
        "snr_sweep", "n_ue_sweep", "n_ap_sweep", "antennas_sweep",
        "convergence_trace", "fronthaul_table",
    }
    got = set(list_experiments())
    assert got == expected, \
        f"registry mismatch: missing={expected - got}, extra={got - expected}"
    # get_experiment returns the same callable
    for name in expected:
        fn = get_experiment(name)
        assert callable(fn), f"{name} is not callable"
    # Unknown name raises with a helpful message
    try:
        get_experiment("not_a_real_experiment")
        assert False, "expected KeyError"
    except KeyError as e:
        assert "not_a_real_experiment" in str(e)


@_register("Test 13: every run_* has the (cfg, runner_cfg, **kwargs) signature")
def test_13_run_signatures():
    import inspect
    for name, fn in REGISTRY.items():
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        # Must accept (cfg, runner_cfg) as the first two positional args.
        assert len(params) >= 2, f"{name}: too few params"
        assert params[0].name == "cfg",        f"{name}: first param != 'cfg'"
        assert params[1].name == "runner_cfg", f"{name}: second param != 'runner_cfg'"
        # All remaining must be keyword-only (or VAR_KEYWORD).
        for p in params[2:]:
            assert p.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            ), f"{name}: param {p.name!r} is positional ({p.kind})"


# ─────────────────────────────────────────────────────────────────────
#  Tests 14–15: integration smoke (require full CORDIS install)
# ─────────────────────────────────────────────────────────────────────

def _try_full_import():
    """Return (cfg, RunnerConfig) tuple, or raise _SkipTest with a reason."""
    try:
        from cordis.utils.config import load_config              # noqa: F401
        from cordis.simulation.runner import (                   # noqa: F401
            MonteCarloRunner, RunnerConfig,
        )
        from cordis.simulation.scenario import build_scenario_from_seeds  # noqa: F401
        from cordis.simulation.result import SimResult           # noqa: F401
    except ImportError as e:
        raise _SkipTest(f"full CORDIS install not available: {e}")

    # Locate a config to base the tests on. Look for the smallest one.
    config_candidates = [
        REPO_ROOT / "configs" / "tiny.json",
        REPO_ROOT / "configs" / "small.json",
        REPO_ROOT / "configs" / "default.json",
    ]
    cfg_path = next((p for p in config_candidates if p.exists()), None)
    if cfg_path is None:
        raise _SkipTest(
            "no config in configs/ — expected one of tiny.json / "
            "small.json / default.json"
        )

    # Use the canonical load_config(); CORDISConfig has no .load() classmethod.
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        raise _SkipTest(f"could not load {cfg_path}: {e}")

    runner_cfg = RunnerConfig(
        n_drops=2, n_realizations_per_drop=1,
        n_workers=1, verbose=0, base_seed=42,
    )
    return cfg, runner_cfg


@_register("Test 14: integration — run a tiny gamma_sweep end-to-end")
def test_14_integration_gamma_sweep():
    cfg, runner_cfg = _try_full_import()  # may raise _SkipTest
    fn = get_experiment("gamma_sweep")
    # Smallest possible sweep: 2 points, 2 drops × 1 realization each.
    result = fn(cfg, runner_cfg,
                gamma_values=[0.0, 6.0], n_drops=2, n_realizations=1)
    assert result.kind == "sweep"
    assert result.name == "gamma_sweep"
    assert sorted(result.sweep_results.keys()) == [0.0, 6.0]

    # Data-quality check: at least one sweep point must have at least one
    # algorithm with non-NaN min_sinr_db.  This catches the Stage-8b spec
    # bugs (omega-as-float, warm_start_from_split, etc.) which would
    # silently fail every algorithm without breaking the structural
    # assertions above.
    points_with_data = 0
    for v, sr in result.sweep_results.items():
        for alg_name in ("CORDIS-Split", "CORDIS-ADMM", "Centralized"):
            try:
                val = sr.mean(alg_name, "min_sinr_db")
                if val == val:               # not NaN
                    points_with_data += 1
                    break
            except (KeyError, AttributeError, TypeError, ValueError):
                pass
    assert points_with_data > 0, (
        "No algorithm produced min_sinr_db data at any sweep point — "
        "all trials failed. Check spec.params for keys the real "
        "algorithms reject (omega-as-float, warm_start_from_split, "
        "warm_start, use_cvxpy=False)."
    )

    # Save + reload round-trip with the REAL SimResult.
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "gamma_sweep"
        result.save(out_dir)
        loaded = ExperimentResult.load(out_dir)
        assert loaded.kind == "sweep"
        assert sorted(loaded.sweep_results.keys()) == [0.0, 6.0]


@_register("Test 15: integration — sinr_cdf result plots via plot_cdf")
def test_15_integration_plot():
    cfg, runner_cfg = _try_full_import()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cordis.plotting import plot_cdf
    except ImportError as e:
        raise _SkipTest(f"matplotlib / cordis.plotting not available: {e}")

    fn = get_experiment("sinr_cdf")
    result = fn(cfg, runner_cfg, n_drops=2, n_realizations=1)
    assert result.kind == "single"
    # Should be plottable without raising
    fig, ax = plt.subplots()
    try:
        plot_cdf(result.sim_result, metric="min_sinr_db", ax=ax)
    except Exception as e:
        raise AssertionError(f"plot_cdf failed on real result: {e}") from e
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 8b validation  —  {len(_TESTS)} tests")
    print("=" * 78)
    print()

    n_pass = n_skip = n_fail = 0
    failures: List[Tuple[str, str]] = []
    for label, fn in _TESTS:
        _section(label)
        try:
            fn()
        except _SkipTest as e:
            print(f"  ⏭  SKIPPED  ({e})")
            n_skip += 1
        except Exception:
            print(f"  ✗ FAILED")
            tb = traceback.format_exc()
            for line in tb.splitlines():
                print(f"      {line}")
            failures.append((label, tb))
            n_fail += 1
        else:
            print(f"  ✓ PASSED")
            n_pass += 1
        print()

    print("=" * 78)
    if n_fail == 0:
        msg = f"✓ ALL TESTS PASSED   ({n_pass}/{len(_TESTS)} passed"
        if n_skip:
            msg += f", {n_skip} skipped"
        msg += ")"
        print(f"  {msg}")
    else:
        print(f"  ✗ {n_fail} FAILURE(S)   "
              f"({n_pass} passed, {n_skip} skipped, {n_fail} failed)")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

