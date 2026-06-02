"""
scripts/validate_stage24.py
===========================
Stage 24 validator — optional minimum pairwise separation for topology drops.

  * UE↔UE separation   : topology.ue_min_separation_m
  * UE↔target separation: topology.ue_target_min_separation_m

Both default to 0.0 (disabled).  When 0, generate_topology calls the original
area-uniform sampler `_uniform_annulus` unchanged, so the RNG draw pattern and
every previously-generated topology for a given seed are byte-for-byte
identical.  A positive value switches that entity to the rejection sampler
`_uniform_annulus_min_sep`.

Run (no pytest required; safe for `make validate` and remote/CI):
    python3 scripts/validate_stage24.py

Design: the backward-compatibility check DERIVES the expected positions on the
fly by replaying `_uniform_annulus` in the same draw order, so this validator
does NOT depend on any committed golden .npy/.npz file.  Source/ast/JSON checks
run with no dependencies; the import-based checks SKIP gracefully when cordis
is not importable (they run under `make validate`, where it is).
"""

from __future__ import annotations

import ast
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TOPOLOGY_PY = REPO_ROOT / "cordis" / "channel" / "topology.py"
CONFIG_PY = REPO_ROOT / "cordis" / "utils" / "config.py"
DEFAULT_JSON = REPO_ROOT / "configs" / "default.json"
DEFAULTS_SH = REPO_ROOT / "configs" / "recipes" / "_defaults.sh"

_FIELDS = ("ue_min_separation_m", "ue_target_min_separation_m")


# ═════════════════════════════════════════════════════════════════════════════
# Registry
# ═════════════════════════════════════════════════════════════════════════════

_TESTS: List[Tuple[str, Callable[[], None]]] = []


class _SkipTest(Exception):
    pass


def _register(label: str):
    def deco(fn):
        _TESTS.append((label, fn))
        return fn
    return deco


def _imports():
    """Return the cordis callables needed for behavioural tests, or skip."""
    try:
        from cordis.channel.topology import (
            generate_topology, _uniform_annulus, _uniform_annulus_min_sep,
        )
        from cordis.utils.config import load_config
        from cordis.utils.io_utils import make_rng
        return (generate_topology, _uniform_annulus, _uniform_annulus_min_sep,
                load_config, make_rng)
    except Exception as e:
        raise _SkipTest(f"cordis not importable here ({type(e).__name__})")


def _find_class(src: str, name: str) -> ast.ClassDef:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def _find_func(src: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


# ═════════════════════════════════════════════════════════════════════════════
# Source / config-of-truth checks (no cordis needed)
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  1: TopologyConfig has both separation fields defaulting to 0.0")
def test_01_dataclass_fields():
    cls = _find_class(CONFIG_PY.read_text(), "TopologyConfig")
    found = {}
    for node in cls.body:
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in _FIELDS):
            found[node.target.id] = node.value
    for f in _FIELDS:
        assert f in found, f"TopologyConfig missing field {f!r}"
        v = found[f]
        assert isinstance(v, ast.Constant) and float(v.value) == 0.0, (
            f"TopologyConfig.{f} must default to 0.0 (disabled / backward "
            f"compatible); got {ast.unparse(v) if v else None}"
        )


@_register("Test  2: PARAM_REGISTRY documents both separation params")
def test_02_param_registry():
    src = CONFIG_PY.read_text()
    for f in _FIELDS:
        assert f'"topology.{f}"' in src, (
            f"PARAM_REGISTRY missing entry for topology.{f} "
            f"(needed so scripts/list_config_params.py documents it)"
        )


@_register("Test  3: default.json topology has both params == 0.0")
def test_03_default_json():
    topo = json.loads(DEFAULT_JSON.read_text())["topology"]
    for f in _FIELDS:
        assert f in topo, f"default.json topology missing {f!r}"
        assert float(topo[f]) == 0.0, (
            f"default.json topology.{f} must be 0.0; got {topo[f]}"
        )


@_register("Test  4: _defaults.sh defines both vars (default 0.0) and wires "
           "them into SET_ARGS")
def test_04_defaults_sh():
    sh = DEFAULTS_SH.read_text()
    pairs = (
        ("UE_MIN_SEPARATION_M", "topology.ue_min_separation_m"),
        ("UE_TARGET_MIN_SEPARATION_M", "topology.ue_target_min_separation_m"),
    )
    for var, set_arg in pairs:
        assert re.search(rf'{var}="\$\{{{var}:-0\.0\}}"', sh), (
            f"_defaults.sh must define {var} defaulting to 0.0"
        )
        assert re.search(rf'{re.escape(set_arg)}="\${var}"', sh), (
            f"_defaults.sh SET_ARGS must include {set_arg}=\"${var}\""
        )


@_register("Test  5: topology.py defines the rejection sampler "
           "_uniform_annulus_min_sep with self/avoid params")
def test_05_sampler_defined():
    fn = _find_func(TOPOLOGY_PY.read_text(), "_uniform_annulus_min_sep")
    args = [a.arg for a in (fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs)]
    for needed in ("self_min_dist", "avoid_xy", "avoid_min_dist"):
        assert needed in args, (
            f"_uniform_annulus_min_sep must accept {needed!r}; got {args}"
        )


@_register("Test  6: generate_topology branches — off-path uses _uniform_annulus, "
           "on-path uses _uniform_annulus_min_sep, gated on the separation params")
def test_06_branch():
    src = TOPOLOGY_PY.read_text()
    fn_src = ast.get_source_segment(src, _find_func(src, "generate_topology"))
    # both samplers referenced inside generate_topology
    assert "_uniform_annulus_min_sep(" in fn_src, (
        "generate_topology must call _uniform_annulus_min_sep on the on-path"
    )
    assert "_uniform_annulus(" in fn_src, (
        "generate_topology must STILL call _uniform_annulus on the off-path "
        "(this is what preserves the RNG draw pattern / existing seeds)"
    )
    # gated on the two params being > 0
    for f in _FIELDS:
        assert re.search(rf'getattr\(\s*t,\s*"{f}"', fn_src), (
            f"generate_topology must read {f} (via getattr) to decide the branch"
        )
    # the off-path must be guarded by a `<= 0` check (so 0 => original sampler)
    assert re.search(r'<=\s*0\.0', fn_src), (
        "the branch must route separation <= 0 to the original sampler"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Behavioural checks (need cordis; SKIP if unavailable)
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  7: [needs cordis] backward compat — with separations = 0, "
           "positions equal a direct _uniform_annulus replay (RNG unchanged)")
def test_07_backward_compat():
    import numpy as np
    (generate_topology, _uniform_annulus, _uams,
     load_config, make_rng) = _imports()

    cfg = load_config(str(DEFAULT_JSON))
    # default layout draws no RNG for APs, so the first draws are UE then target
    assert cfg.topology.type == "circle", (
        "this replay assumes the circle layout (APs consume no RNG); "
        "default.json must keep type=circle for this check"
    )
    assert (cfg.topology.ue_min_separation_m == 0.0
            and cfg.topology.ue_target_min_separation_m == 0.0), (
        "default.json must keep both separations at 0.0"
    )

    seed = 20240524
    topo = generate_topology(cfg, make_rng(seed))

    rng2 = make_rng(seed)
    t = cfg.topology
    ue_exp = _uniform_annulus(rng2, t.n_ue, t.ue_min_radius_m,
                              t.ue_max_radius_m, t.h_ue_m)
    tg_exp = _uniform_annulus(rng2, t.n_targets, t.tg_min_radius_m,
                              t.tg_max_radius_m, t.h_tg_m)
    assert np.allclose(topo.ue_positions, ue_exp), (
        "UE positions diverged from the original _uniform_annulus draw — the "
        "off-path changed the RNG consumption (backward compatibility broken)"
    )
    assert np.allclose(topo.target_positions, tg_exp), (
        "target positions diverged from the original _uniform_annulus draw"
    )


@_register("Test  8: [needs cordis] with separations set, all UE↔UE and "
           "UE↔target 2-D distances clear the thresholds")
def test_08_constraints_enforced():
    import itertools
    import numpy as np
    (generate_topology, _ua, _uams, load_config, make_rng) = _imports()

    cfg = load_config(str(DEFAULT_JSON))
    d_uu, d_ut = 80.0, 100.0
    cfg.topology.ue_min_separation_m = d_uu
    cfg.topology.ue_target_min_separation_m = d_ut

    for seed in (1, 11, 123, 2024):
        topo = generate_topology(cfg, make_rng(seed))
        ue = topo.ue_positions[:, :2]
        tg = topo.target_positions[:, :2]
        for i, j in itertools.combinations(range(len(ue)), 2):
            assert np.linalg.norm(ue[i] - ue[j]) >= d_uu - 1e-6, (
                f"seed {seed}: UE {i}/{j} closer than {d_uu} m"
            )
        for u in ue:
            for g in tg:
                assert np.linalg.norm(u - g) >= d_ut - 1e-6, (
                    f"seed {seed}: a UE-target pair closer than {d_ut} m"
                )


@_register("Test  9: [needs cordis] infeasible separation raises ValueError "
           "(not an infinite loop)")
def test_09_feasibility_error():
    (_gt, _ua, _uams, _lc, make_rng) = _imports()
    raised = False
    try:
        # 4 points 200 m apart in a [35, 60] m annulus is impossible
        _uams(make_rng(1), 4, 35.0, 60.0, 1.5, self_min_dist=200.0)
    except ValueError:
        raised = True
    assert raised, "an infeasible separation must raise ValueError"


@_register("Test 10: [needs cordis] on-path is deterministic "
           "(same seed → identical positions)")
def test_10_determinism():
    import numpy as np
    (generate_topology, _ua, _uams, load_config, make_rng) = _imports()
    cfg = load_config(str(DEFAULT_JSON))
    cfg.topology.ue_min_separation_m = 80.0
    cfg.topology.ue_target_min_separation_m = 100.0
    a = generate_topology(cfg, make_rng(99))
    b = generate_topology(cfg, make_rng(99))
    assert np.allclose(a.ue_positions, b.ue_positions), "UE draw not deterministic"
    assert np.allclose(a.target_positions, b.target_positions), "tg draw not deterministic"


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    width = 78
    print("=" * width)
    print(f"  Stage 24 validator — {len(_TESTS)} tests")
    print("=" * width)

    passed = skipped = failed = 0
    failed_tests: List[str] = []

    for label, fn in _TESTS:
        print(f"\n── {label} ──")
        try:
            fn()
        except _SkipTest as e:
            print(f"  ↷ SKIPPED  ({e})")
            skipped += 1
            continue
        except AssertionError as e:
            print("  ✗ FAILED")
            for line in str(e).splitlines():
                print(f"      {line}")
            failed += 1
            failed_tests.append(label)
            continue
        except Exception:
            print("  ✗ FAILED (unexpected exception)")
            traceback.print_exc()
            failed += 1
            failed_tests.append(label)
            continue
        print("  ✓ PASSED")
        passed += 1

    print()
    print("=" * width)
    if failed:
        print(f"  ✗ {failed} FAILURE(S)   "
              f"({passed} passed, {skipped} skipped, {failed} failed)")
        for t in failed_tests:
            print(f"    failed: {t}")
    else:
        print(f"  ✓ ALL TESTS PASSED   ({passed}/{passed + skipped} "
              f"passed, {skipped} skipped)")
    print("=" * width)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

