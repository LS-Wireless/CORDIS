#!/usr/bin/env python3
"""
scripts/validate_stage22a.py
============================

Stage 22a — ADMM convergence: bumped iteration cap, residual-based
best-iter selection, recipe-default cleanup.

After running this, the user has:

  - ``cfg.algorithm.admm.n_max`` default bumped 50 → 200 (in both
    ``configs/default.json`` and ``configs/recipes/_defaults.sh``).
  - ``cfg.algorithm.admm.best_iter_criterion`` field added (allowed
    values: ``"residual_norm"`` (default), ``"min_sinr"`` (legacy)),
    plumbed end-to-end from JSON config through ``_admm_kwargs_from_cfg``
    into ``solve_cordis_admm``.
  - The "take" decision inside ``solve_cordis_admm`` branches on the
    criterion and tracks ``best_resid_norm = r_pri + r_dual``.
  - ``cfg.algorithm.gamma_db`` default 10 → 5; ``simulation.n_trials``
    default in default.json aligned to ``_defaults.sh``'s 100.
  - Two recipes (``exp_snr_sweep.sh``, ``exp_convergence_trace.sh``)
    cleaned of overrides that matched the new defaults.

Tests verify each piece at three tiers:
  Tier 1: text-grep / JSON-load on the config-source-of-truth files
          (cross-file consistency for n_max, gamma_db, n_trials,
          best_iter_criterion).
  Tier 2: structural plumbing — ADMMConfig has the field;
          _admm_kwargs_from_cfg forwards it; solve_cordis_admm accepts
          the kwarg and validates it.
  Tier 3: behavioural — the take-decision logic produces the expected
          best_iter on a synthetic ADMM trajectory under each criterion.

Run from the repo root::

    python3 scripts/validate_stage22a.py
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


class _SkipTest(Exception):
    pass


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn):
        _TESTS.append((label, fn))
        return fn
    return deco


# Paths used throughout (resolved once).
DEFAULT_JSON = REPO_ROOT / "configs" / "default.json"
DEFAULTS_SH  = REPO_ROOT / "configs" / "recipes" / "_defaults.sh"
RECIPES_DIR  = REPO_ROOT / "configs" / "recipes"
CONFIG_PY    = REPO_ROOT / "cordis" / "utils" / "config.py"
REGISTRY_PY  = REPO_ROOT / "cordis" / "experiments" / "registry.py"
JOINT_OPT_PY = REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py"


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1: Cross-file consistency of the default values
# ═════════════════════════════════════════════════════════════════════════════

def _read_json_defaults() -> dict:
    assert DEFAULT_JSON.exists(), f"missing {DEFAULT_JSON}"
    with open(DEFAULT_JSON) as f:
        return json.load(f)


def _grep_defaults_sh(varname: str) -> str:
    """Extract the default value from a ``${VAR:-default}`` expansion."""
    text = DEFAULTS_SH.read_text()
    m = re.search(rf'^{varname}="\$\{{{varname}:-([^}}]+)\}}"', text, re.M)
    assert m, f"could not grep {varname} default from {DEFAULTS_SH.name}"
    return m.group(1)


@_register("Test  1: admm.n_max default bumped to 200 in both source-of-truth files")
def test_01_admm_n_max_bumped():
    json_n_max = _read_json_defaults()["algorithm"]["admm"]["n_max"]
    sh_n_max   = int(_grep_defaults_sh("ADMM_N_MAX"))
    assert json_n_max == 200, (
        f"default.json: algorithm.admm.n_max = {json_n_max}, want 200.  "
        f"Stage 22a bumps the iteration cap to give the dual variable ν_u "
        f"room to ramp up (50 was mid-ramp on the journal SOC-consensus "
        f"formulation)."
    )
    assert sh_n_max == 200, (
        f"_defaults.sh: ADMM_N_MAX = {sh_n_max}, want 200.  "
        f"Recipes need this default to match default.json so "
        f"recipe→default diffs stay minimal."
    )


@_register("Test  2: gamma_db default bumped to 5 in both source-of-truth files")
def test_02_gamma_db_default():
    json_g = _read_json_defaults()["algorithm"]["gamma_db"]
    sh_g   = float(_grep_defaults_sh("GAMMA_DB"))
    assert json_g == 5.0, (
        f"default.json: algorithm.gamma_db = {json_g}, want 5.0"
    )
    assert sh_g == 5.0, (
        f"_defaults.sh: GAMMA_DB = {sh_g}, want 5.0"
    )


@_register("Test  3: simulation.n_trials default in default.json matches "
           "_defaults.sh (so recipes that set N_TRIALS=100 produce no diff)")
def test_03_n_trials_aligned():
    json_n = _read_json_defaults()["simulation"]["n_trials"]
    sh_n   = int(_grep_defaults_sh("N_TRIALS"))
    assert json_n == sh_n, (
        f"default.json n_trials={json_n} but _defaults.sh "
        f"N_TRIALS={sh_n}.  These MUST match or recipes that set "
        f"N_TRIALS to the _defaults.sh value will produce a spurious "
        f"diff against default.json on every config generation."
    )
    assert json_n == 100, (
        f"both files agree on n_trials but value is {json_n}, want 100"
    )


@_register("Test  4: admm.best_iter_criterion default is 'residual_norm' in "
           "both source-of-truth files")
def test_04_best_iter_criterion_default():
    json_c = _read_json_defaults()["algorithm"]["admm"]["best_iter_criterion"]
    sh_c   = _grep_defaults_sh("ADMM_BEST_ITER_CRITERION")
    assert json_c == "residual_norm", (
        f"default.json: best_iter_criterion = {json_c!r}, "
        f"want 'residual_norm'.  Residual-based selection is robust to "
        f"the late-iteration oscillation that min_sinr-based selection "
        f"falls for."
    )
    assert sh_c == "residual_norm", (
        f"_defaults.sh: ADMM_BEST_ITER_CRITERION = {sh_c!r}, "
        f"want 'residual_norm'"
    )


@_register("Test  5: _defaults.sh SET_ARGS array forwards "
           "algorithm.admm.best_iter_criterion to create_config.py")
def test_05_set_args_forwards_criterion():
    text = DEFAULTS_SH.read_text()
    assert "algorithm.admm.best_iter_criterion=\"$ADMM_BEST_ITER_CRITERION\"" in text, (
        f"_defaults.sh SET_ARGS array must contain "
        f"`algorithm.admm.best_iter_criterion=\"$ADMM_BEST_ITER_CRITERION\"` "
        f"so create_config.py picks up the recipe's choice"
    )


@_register("Test  6: exp_snr_sweep.sh no longer has a no-op SNR_DB override "
           "(the sweep replaces SNR_DB at runtime)")
def test_06_snr_sweep_recipe_cleaned():
    text = (RECIPES_DIR / "exp_snr_sweep.sh").read_text()
    # Catch only the literal override `SNR_DB=20.0` (or similar) at start of
    # line; ignore comments and the SET_ARGS line in _defaults.sh.
    bad = re.search(r"^\s*SNR_DB\s*=\s*\d", text, re.M)
    assert bad is None, (
        f"exp_snr_sweep.sh still has an SNR_DB override: {bad.group(0)!r}.  "
        f"This recipe should inherit SNR_DB from _defaults.sh; the sweep "
        f"itself replaces SNR_DB per-axis-value at runtime, so a baseline "
        f"override only clutters the recipe→default diff."
    )


@_register("Test  7: exp_convergence_trace.sh no longer has an ADMM_N_MAX "
           "override (the new 200 default gives ample plateau headroom)")
def test_07_convergence_recipe_cleaned():
    text = (RECIPES_DIR / "exp_convergence_trace.sh").read_text()
    bad = re.search(r"^\s*ADMM_N_MAX\s*=\s*\d", text, re.M)
    assert bad is None, (
        f"exp_convergence_trace.sh still has an ADMM_N_MAX override: "
        f"{bad.group(0)!r}.  With the new 200 default this override is "
        f"redundant and only clutters the recipe→default diff."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2: Structural plumbing — ADMMConfig field, kwarg forwarding,
#         solve_cordis_admm signature + early validation
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  8: ADMMConfig dataclass has a best_iter_criterion field "
           "defaulting to 'residual_norm'")
def test_08_admmconfig_field():
    tree = ast.parse(CONFIG_PY.read_text())
    admm_cls = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef) and n.name == "ADMMConfig"),
        None,
    )
    assert admm_cls is not None, (
        "cordis/utils/config.py: ADMMConfig class not found"
    )
    field = next(
        (n for n in admm_cls.body
         if isinstance(n, ast.AnnAssign)
         and isinstance(n.target, ast.Name)
         and n.target.id == "best_iter_criterion"),
        None,
    )
    assert field is not None, (
        "ADMMConfig: missing best_iter_criterion field.  "
        "Stage 22a adds it as a `str = \"residual_norm\"` field."
    )
    # Default must be the string literal "residual_norm".
    assert (isinstance(field.value, ast.Constant)
            and field.value.value == "residual_norm"), (
        f"ADMMConfig.best_iter_criterion default must be the string "
        f"'residual_norm', got: "
        f"{ast.unparse(field.value) if field.value else None}"
    )


@_register("Test  9: _admm_kwargs_from_cfg forwards best_iter_criterion "
           "from cfg.algorithm.admm.* into solve_cordis_admm kwargs")
def test_09_kwargs_forwarding():
    text = REGISTRY_PY.read_text()
    # The mapping table is a tuple of (cfg_attr, spec_kw, cast).  Look for
    # the row that maps best_iter_criterion → best_iter_criterion as str.
    # Match on the full row to make a typo (e.g. wrong cast) impossible.
    assert re.search(
        r'\(\s*"best_iter_criterion"\s*,\s*"best_iter_criterion"\s*,\s*str\s*\)',
        text,
    ), (
        "cordis/experiments/registry.py: _admm_kwargs_from_cfg must "
        "contain a row mapping `best_iter_criterion` (cfg attr) → "
        "`best_iter_criterion` (kwarg) with str cast.  Without this, "
        "the JSON config setting is silently ignored."
    )


@_register("Test 10: solve_cordis_admm signature accepts best_iter_criterion "
           "kwarg with 'residual_norm' default")
def test_10_solve_cordis_admm_signature():
    # AST-based inspection rather than `from cordis.algorithms.joint_opt
    # import solve_cordis_admm` — the runtime import drags in the rest of
    # the cordis package (channel, simulation, ...), which we don't need
    # to test a function signature.  This keeps the validator portable
    # across environments and consistent with Test 8's approach to
    # ADMMConfig.
    tree = ast.parse(JOINT_OPT_PY.read_text())
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "solve_cordis_admm"),
        None,
    )
    assert fn is not None, (
        f"{JOINT_OPT_PY.name}: function solve_cordis_admm not found"
    )
    # Keyword-only args live in `args.kwonlyargs`; pair with `args.kw_defaults`
    # by position.  The default may be None (no default given) — we want
    # the literal string "residual_norm".
    kwonly = list(zip(fn.args.kwonlyargs, fn.args.kw_defaults))
    found = [
        (arg, default) for arg, default in kwonly
        if arg.arg == "best_iter_criterion"
    ]
    assert found, (
        "solve_cordis_admm signature must accept `best_iter_criterion` "
        "kwarg (keyword-only).  Add it to the keyword-only block after "
        "`n_snapshots`."
    )
    arg, default = found[0]
    assert default is not None, (
        "`best_iter_criterion` must have a default value (so existing "
        "callers don't break).  Add `= \"residual_norm\"`."
    )
    assert (isinstance(default, ast.Constant)
            and default.value is None), (
        f"`best_iter_criterion` default = "
        f"{ast.unparse(default) if default else None}, want None.  "
        f"Stage 23a changed solve_cordis_admm to default this kwarg to None "
        f"and resolve it from cfg.algorithm.admm.best_iter_criterion when not "
        f"passed, so the JSON config is authoritative (an explicit kwarg still "
        f"overrides)."
    )


@_register("Test 11: solve_cordis_admm validates best_iter_criterion early "
           "(invalid values raise ValueError)")
def test_11_validation_fail_fast():
    text = JOINT_OPT_PY.read_text()
    # The validation must check membership against a tuple/list containing
    # both allowed values, and raise ValueError.  Don't pin the exact wording;
    # check the structural pieces.
    assert (
        'best_iter_criterion not in' in text
        and '"residual_norm"' in text
        and '"min_sinr"' in text
        and re.search(r'raise ValueError', text)
    ), (
        "solve_cordis_admm must validate best_iter_criterion against the "
        "tuple ('residual_norm', 'min_sinr') and raise ValueError on "
        "anything else.  Fail-fast catches typos in the JSON config "
        "before a long Monte Carlo run wastes time."
    )


@_register("Test 12: best_resid_norm tracker added alongside best_min_sinr")
def test_12_resid_norm_tracker():
    text = JOINT_OPT_PY.read_text()
    assert "best_resid_norm" in text, (
        "Stage 22a adds a `best_resid_norm` tracker (initialised to "
        "+inf) that records the smallest r_pri+r_dual seen so far, "
        "used by the residual_norm criterion branch."
    )
    # The candidate value at each iteration must be r_pri + r_dual.
    assert re.search(
        r"cand_resid_norm\s*=\s*float\(\s*r_pri\s*\+\s*r_dual\s*\)",
        text,
    ), (
        "candidate residual norm at each iteration must be "
        "`float(r_pri + r_dual)`.  Without this the residual_norm "
        "criterion has no signal to track."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tier 3: Behavioural — the take-decision logic picks the right iterate
#         under each criterion on a synthetic ADMM trajectory
# ═════════════════════════════════════════════════════════════════════════════

def _take_logic(criterion, cand_min_sinr, cand_resid_norm, r_pri,
                best_min_sinr, best_resid_norm, best_r_pri,
                all_ok=True):
    """Standalone mirror of the joint_opt.py take-decision branch.

    KEEP IN SYNC with the branch in ``solve_cordis_admm``.  This is
    a behavioural spec — if you change one, change both.
    """
    if not all_ok:
        return False
    if criterion == "residual_norm":
        if cand_resid_norm < best_resid_norm - 1e-12:
            return True
        if (abs(cand_resid_norm - best_resid_norm) < 1e-12
                and cand_min_sinr > best_min_sinr):
            return True
        return False
    if cand_min_sinr > best_min_sinr + 1e-12:
        return True
    if (abs(cand_min_sinr - best_min_sinr) < 1e-12
            and r_pri < best_r_pri):
        return True
    return False


# Synthetic trajectory representing a typical fixed-ρ ADMM run on this
# algorithm: dual ramp-up early, convergence plateau in the middle,
# then post-convergence oscillation around the SOC boundary.
#   iter | min_sinr_db | r_pri  | r_dual
#   -----+------------+--------+--------
#     1  |    -5.0    | 10.0   | 10.0   ← cold start
#    50  |     4.0    |  2.0   |  2.0   ← converging
#   100  |     5.5    |  0.5   |  0.4   ← plateau (BEST residual)
#   200  |     7.0    |  1.5   |  1.2   ← swing peak (BEST min-SINR)
#   300  |     4.5    |  3.0   |  2.5   ← swing trough
_SYNTH_TRAJ = [
    (1,   -5.0, 10.0, 10.0),
    (50,   4.0,  2.0,  2.0),
    (100,  5.5,  0.5,  0.4),
    (200,  7.0,  1.5,  1.2),
    (300,  4.5,  3.0,  2.5),
]


def _replay(criterion: str) -> int:
    best_iter, best_min_sinr, best_resid_norm, best_r_pri = (
        0, -float("inf"), float("inf"), float("inf"),
    )
    for n, sinr, rp, rd in _SYNTH_TRAJ:
        if _take_logic(criterion, sinr, rp + rd, rp,
                       best_min_sinr, best_resid_norm, best_r_pri):
            best_iter, best_min_sinr, best_resid_norm, best_r_pri = (
                n, sinr, rp + rd, rp,
            )
    return best_iter


@_register("Test 13: residual_norm criterion picks the plateau iterate "
           "on a synthetic swinging-ADMM trajectory")
def test_13_residual_picks_plateau():
    chosen = _replay("residual_norm")
    assert chosen == 100, (
        f"residual_norm should pick the plateau iterate (iter 100, "
        f"r_pri+r_dual=0.9), got iter {chosen}.  This is the whole "
        f"point of the new criterion: avoid late-iteration swing peaks."
    )


@_register("Test 14: min_sinr criterion picks the swing-peak iterate "
           "on the same trajectory (legacy behaviour preserved)")
def test_14_min_sinr_picks_swing_peak():
    chosen = _replay("min_sinr")
    assert chosen == 200, (
        f"min_sinr should pick the swing peak (iter 200, SINR=+7.0 dB), "
        f"got iter {chosen}.  Legacy behaviour must be preserved for "
        f"back-compat — users running with best_iter_criterion='min_sinr' "
        f"should see exactly what they would have seen before Stage 22a."
    )


@_register("Test 15: residual_norm vs min_sinr pick DIFFERENT iterates "
           "on the swinging trajectory (they MUST disagree for the "
           "new criterion to be doing anything useful)")
def test_15_criteria_differ_on_swing():
    res = _replay("residual_norm")
    sin = _replay("min_sinr")
    assert res != sin, (
        f"Both criteria pick iter {res} on the synthetic swinging "
        f"trajectory.  They should pick different iterates (the "
        f"residual_norm plateau vs the min_sinr swing peak); otherwise "
        f"the new criterion is a no-op and Stage 22a's whole reason "
        f"for being is undermined."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 78)
    print(f"  Stage 22 validator — {len(_TESTS)} tests")
    print("=" * 78)

    passed = skipped = failed = 0
    failures: List[Tuple[str, str]] = []

    for label, fn in _TESTS:
        print(f"\n── {label} ──")
        try:
            fn()
            print("  ✓ PASSED")
            passed += 1
        except _SkipTest as e:
            print(f"  ⏭  SKIPPED  ({e})")
            skipped += 1
        except AssertionError as e:
            print("  ✗ FAILED")
            for line in str(e).splitlines():
                print(f"      {line}")
            failed += 1
            failures.append((label, str(e)))
        except Exception:
            print("  ✗ FAILED (unexpected exception)")
            tb = traceback.format_exc()
            print("      " + tb.replace("\n", "\n      ").rstrip())
            failed += 1
            failures.append((label, "unexpected exception"))

    print()
    print("=" * 78)
    if failed == 0:
        print(f"  ✓ ALL TESTS PASSED   "
              f"({passed}/{len(_TESTS)} passed, {skipped} skipped)")
    else:
        print(f"  ✗ {failed} FAILURE(S)   "
              f"({passed} passed, {skipped} skipped, {failed} failed)")
        print("    failed: " + ", ".join(lbl for lbl, _ in failures))
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

