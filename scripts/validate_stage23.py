"""
scripts/validate_stage23.py
===========================
Stage 23 validator.

  Stage 23a — feasibility-aware best-iterate selection ("feasible_then_residual"
              best_iter_criterion; the natural successor to Stage 22a's
              residual_norm / min_sinr criteria).
  Stage 23b — per-AP CSI-error consensus block: the consensus vector's err
              component is gathered per transmit AP and each entry is stacked
              individually in the CPU SOC, so ‖z^err‖² = Σ_a e_au² is the EXACT
              per-user CSI-error power (the previous single-scalar form gave
              (Σ_a e_au)², an over-count of up to N_tx×).

Both changes live in cordis/algorithms/joint_opt.py.

Run (no pytest required):
    python3 scripts/validate_stage23.py

Like the other stage validators, most checks are source-level (regex / ast on
the joint_opt.py text) and the best-iterate decision rule is exercised by
ast-extracting the pure `_should_take_iterate` function and exec'ing it in a
bare namespace — so the bulk of this validator runs even when cordis / CVXPY
are not importable in the container.  The import-based helper checks
(Test 16) SKIP gracefully when cordis can't be imported.
"""

from __future__ import annotations

import ast
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_JOINT_OPT = REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py"


# ═════════════════════════════════════════════════════════════════════════════
# Test registry
# ═════════════════════════════════════════════════════════════════════════════

_TESTS: List[Tuple[str, Callable[[], None]]] = []


class _SkipTest(Exception):
    """Raised by a test to mark itself skipped (e.g. cordis not importable)."""


def _register(label: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        _TESTS.append((label, fn))
        return fn
    return deco


# ═════════════════════════════════════════════════════════════════════════════
# Source-inspection helpers (work without importing cordis)
# ═════════════════════════════════════════════════════════════════════════════

def _src() -> str:
    assert _JOINT_OPT.exists(), f"missing source file: {_JOINT_OPT}"
    return _JOINT_OPT.read_text()


def _tree() -> ast.Module:
    return ast.parse(_src())


def _find_func(name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in joint_opt.py")


def _find_class(name: str) -> ast.ClassDef:
    for node in _tree().body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in joint_opt.py")


def _func_argnames(fn: ast.FunctionDef) -> List[str]:
    a = fn.args
    return [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)]


def _class_field_annotation(cls: ast.ClassDef, field: str) -> str:
    for node in cls.body:
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == field):
            return ast.unparse(node.annotation)
    raise AssertionError(f"field {field!r} not found in class {cls.name}")


def _extract_should_take_iterate() -> Callable:
    """ast-extract _should_take_iterate and exec it in a bare namespace.

    Mirrors Stage 21's test_16 'load by path, no package import' trick so the
    decision rule can be exercised without importing cordis / CVXPY.
    """
    src = _src()
    fn_src = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_should_take_iterate":
            fn_src = ast.get_source_segment(src, node)
            break
    assert fn_src is not None, "_should_take_iterate not found in joint_opt.py"
    ns: dict = {}
    exec(fn_src, ns)
    return ns["_should_take_iterate"]


INF = float("inf")
NEG = -INF


# ═════════════════════════════════════════════════════════════════════════════
# Stage 23a — feasibility-aware best-iterate
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  1: _ALLOWED_CRITERIA lists residual_norm, min_sinr, "
           "feasible_then_residual")
def test_01_allowed_criteria():
    src = _src()
    m = re.search(r"_ALLOWED_CRITERIA\s*=\s*\(([^)]*)\)", src)
    assert m, "could not find _ALLOWED_CRITERIA tuple"
    body = m.group(1)
    for crit in ("residual_norm", "min_sinr", "feasible_then_residual"):
        assert f'"{crit}"' in body, (
            f"_ALLOWED_CRITERIA must include {crit!r} (Stage 23a adds "
            f"feasible_then_residual)"
        )


@_register("Test  2: best_iter_criterion defaults to None and is resolved "
           "from cfg.algorithm.admm (config field made authoritative)")
def test_02_criterion_resolved_from_cfg():
    src = _src()
    assert re.search(r"best_iter_criterion:\s*Optional\[str\]\s*=\s*None", src), (
        "solve_cordis_admm must default best_iter_criterion=None so the JSON "
        "config decides (Stage 23a); an explicit kwarg still overrides."
    )
    assert re.search(
        r'getattr\(\s*cfg\.algorithm\.admm,\s*"best_iter_criterion"', src, re.S
    ), (
        "When best_iter_criterion is None it must be read from "
        'cfg.algorithm.admm.best_iter_criterion (previously the kwarg default '
        '"residual_norm" shadowed the config field, making it dead)'
    )


@_register("Test  3: patience early-stop fires under residual_norm AND "
           "feasible_then_residual (not legacy min_sinr)")
def test_03_early_stop_gate():
    src = _src()
    assert re.search(
        r'best_iter_criterion\s+in\s+\(\s*"residual_norm",\s*'
        r'"feasible_then_residual"\s*\)',
        src,
    ), (
        "early-stop must gate on best_iter_criterion in "
        '("residual_norm", "feasible_then_residual"); min_sinr still runs '
        "full n_max (its swing peaks make 'no improvement' unreliable)"
    )


@_register("Test  4: residual_norm legacy behaviour unchanged "
           "(lower combined residual wins)")
def test_04_residual_norm_legacy():
    take = _extract_should_take_iterate()
    assert take("residual_norm", cand_feasible=False, cand_min_sinr=0.0,
                cand_resid_norm=5.0, cand_r_pri=5.0,
                best_feasible=False, best_min_sinr=2.0,
                best_resid_norm=9.0, best_r_pri=9.0)
    assert not take("residual_norm", cand_feasible=False, cand_min_sinr=99.0,
                    cand_resid_norm=20.0, cand_r_pri=20.0,
                    best_feasible=False, best_min_sinr=0.0,
                    best_resid_norm=9.0, best_r_pri=9.0)


@_register("Test  5: min_sinr legacy behaviour unchanged "
           "(higher worst-user SINR wins)")
def test_05_min_sinr_legacy():
    take = _extract_should_take_iterate()
    assert take("min_sinr", cand_feasible=False, cand_min_sinr=5.0,
                cand_resid_norm=99.0, cand_r_pri=99.0,
                best_feasible=False, best_min_sinr=4.0,
                best_resid_norm=1.0, best_r_pri=1.0)
    assert not take("min_sinr", cand_feasible=False, cand_min_sinr=3.0,
                    cand_resid_norm=0.1, cand_r_pri=0.1,
                    best_feasible=False, best_min_sinr=4.0,
                    best_resid_norm=9.0, best_r_pri=9.0)


@_register("Test  6: feasible_then_residual prefers feasible over infeasible "
           "incumbent regardless of residual")
def test_06_ftr_feasible_priority():
    take = _extract_should_take_iterate()
    assert take("feasible_then_residual", cand_feasible=True, cand_min_sinr=1.1,
                cand_resid_norm=99.0, cand_r_pri=99.0,
                best_feasible=False, best_min_sinr=0.9,
                best_resid_norm=0.01, best_r_pri=0.01)


@_register("Test  7: feasible_then_residual never downgrades a feasible "
           "incumbent to an infeasible candidate")
def test_07_ftr_no_downgrade():
    take = _extract_should_take_iterate()
    assert not take("feasible_then_residual", cand_feasible=False,
                    cand_min_sinr=999.0, cand_resid_norm=1e-9, cand_r_pri=1e-9,
                    best_feasible=True, best_min_sinr=1.05,
                    best_resid_norm=50.0, best_r_pri=50.0)


@_register("Test  8: feasible_then_residual ranks feasible by min-SINR and "
           "falls back to lowest residual among infeasible")
def test_08_ftr_ranking():
    take = _extract_should_take_iterate()
    # among feasible: higher min-SINR wins
    assert take("feasible_then_residual", cand_feasible=True, cand_min_sinr=2.0,
                cand_resid_norm=8.0, cand_r_pri=8.0,
                best_feasible=True, best_min_sinr=1.5,
                best_resid_norm=1.0, best_r_pri=1.0)
    # among feasible: lower min-SINR rejected even at tiny residual
    assert not take("feasible_then_residual", cand_feasible=True,
                    cand_min_sinr=1.2, cand_resid_norm=1e-3, cand_r_pri=1e-3,
                    best_feasible=True, best_min_sinr=1.5,
                    best_resid_norm=9.0, best_r_pri=9.0)
    # among infeasible: lower residual wins (fallback)
    assert take("feasible_then_residual", cand_feasible=False,
                cand_min_sinr=0.1, cand_resid_norm=3.0, cand_r_pri=3.0,
                best_feasible=False, best_min_sinr=0.9,
                best_resid_norm=9.0, best_r_pri=9.0)


@_register("Test  9: realistic sequence — feasible_then_residual keeps the "
           "transiently-feasible iterate; residual_norm discards it")
def test_09_ftr_sequence():
    take = _extract_should_take_iterate()
    # (feasible, min_sinr, resid): a feasible spike at idx 2, then the run
    # drifts back to infeasible with LOWER residual.  This is exactly the
    # failure mode Stage 23a fixes.
    seq = [(False, 0.7, 12.0), (False, 0.9, 6.0), (True, 1.1, 8.0),
           (False, 0.95, 2.0), (False, 0.99, 1.0)]

    def run(criterion):
        st = dict(best_feasible=False, best_min_sinr=NEG,
                  best_resid_norm=INF, best_r_pri=INF)
        best_idx = -1
        for i, (f, ms, rr) in enumerate(seq):
            if take(criterion, cand_feasible=f, cand_min_sinr=ms,
                    cand_resid_norm=rr, cand_r_pri=rr, **st):
                st = dict(best_feasible=f, best_min_sinr=ms,
                          best_resid_norm=rr, best_r_pri=rr)
                best_idx = i
        return best_idx, st["best_feasible"]

    idx_ftr, feas_ftr = run("feasible_then_residual")
    idx_res, feas_res = run("residual_norm")
    assert idx_ftr == 2 and feas_ftr, (
        f"feasible_then_residual should keep the feasible iterate (idx 2); "
        f"got idx={idx_ftr} feasible={feas_ftr}"
    )
    assert idx_res == 4 and not feas_res, (
        f"residual_norm should pick the lowest-residual (infeasible) iterate "
        f"(idx 4); got idx={idx_res} feasible={feas_res}.  If these two "
        f"criteria agree, Stage 23a is a no-op."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 23b — per-AP CSI-error consensus block
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test 10: ConsensusVector.err is a per-AP NDArray (LocalContribution"
           ".err stays a scalar float)")
def test_10_err_field_types():
    cv = _class_field_annotation(_find_class("ConsensusVector"), "err")
    assert "NDArray" in cv, (
        f"ConsensusVector.err must be a per-AP vector (NDArray[np.float64]); "
        f"got annotation {cv!r}"
    )
    lc = _class_field_annotation(_find_class("LocalContribution"), "err")
    assert lc == "float", (
        f"LocalContribution.err must stay a scalar float (each AP's own "
        f"amplitude); got {lc!r}"
    )


@_register("Test 11: _zeros_consensus takes an n_tx argument for the per-AP "
           "err block")
def test_11_zeros_consensus_signature():
    args = _func_argnames(_find_func("_zeros_consensus"))
    assert "n_tx" in args, (
        f"_zeros_consensus must take n_tx (err block length); got args {args}"
    )


@_register("Test 12: _sum_local_contributions GATHERS err per-AP "
           "(out.err[slot] = l.err), not a coherent sum")
def test_12_sum_gathers_err():
    src = _src()
    assert re.search(r"out\.err\[\s*slot\s*\]\s*=\s*l\.err", src), (
        "_sum_local_contributions must gather err per-AP (out.err[slot] = "
        "l.err); a coherent sum (out.err += l.err) reintroduces the "
        "(Σ_a e_au)² over-count Stage 23b removes."
    )
    assert not re.search(r"out\.err\s*\+=\s*l\.err", src), (
        "leftover coherent sum 'out.err += l.err' would over-count CSI error"
    )


@_register("Test 13: _solve_central_socp takes n_tx, declares z_err as a "
           "length-n_tx variable, and stacks it whole in the SOC")
def test_13_central_socp_per_ap():
    args = _func_argnames(_find_func("_solve_central_socp"))
    assert "n_tx" in args, f"_solve_central_socp must take n_tx; got {args}"
    src = _src()
    assert re.search(r"z_err\s*=\s*cp\.Variable\(\s*n_tx", src), (
        "z_err must be a length-n_tx cp.Variable (per-AP amplitudes)"
    )
    assert re.search(r"parts\.append\(\s*z_err\s*\)", src), (
        "the whole z_err vector must be appended to the SOC stack so "
        "‖·‖² reproduces Σ_a (z_err_a)² = exact CSI-error power"
    )
    assert not re.search(r"cp\.reshape\(\s*z_err", src), (
        "leftover scalar reshape of z_err — Stage 23b stacks the full vector"
    )


@_register("Test 14: _solve_local_qcqp takes sigma_err_au and uses it as the "
           "per-AP err residual")
def test_14_local_qcqp_sigma_err():
    args = _func_argnames(_find_func("_solve_local_qcqp"))
    assert "sigma_err_au" in args, (
        f"_solve_local_qcqp must take sigma_err_au (per-AP err residual); "
        f"got {args}"
    )
    src = _src()
    assert re.search(r"err_resid\s*=\s*err_tangent\s*\+\s*float\(\s*sigma_err_au\[u\]\s*\)", src), (
        "the local err penalty must use the per-AP residual sigma_err_au[u], "
        "not the old aggregate sigma_au_u.err"
    )


@_register("Test 15: CSI-error identity — Σ_a e_a² == Σ_a tr(Wᴴ R W) (exact "
           "power) and (Σ_a e_a)² ≥ Σ_a e_a² (old over-count)")
def test_15_csi_error_math():
    rng = np.random.default_rng(0)
    worst_ratio = 1.0
    for _ in range(200):
        n_tx = int(rng.integers(2, 7))
        Mt = int(rng.integers(4, 12))
        D = int(rng.integers(2, 8))
        e = np.empty(n_tx)
        exact_power = 0.0
        for a in range(n_tx):
            X = rng.standard_normal((Mt, Mt)) + 1j * rng.standard_normal((Mt, Mt))
            R = X @ X.conj().T                       # Hermitian PSD
            W = rng.standard_normal((Mt, D)) + 1j * rng.standard_normal((Mt, D))
            tr = float(np.real(np.trace(W.conj().T @ R @ W)))
            e[a] = np.sqrt(max(tr, 0.0))
            exact_power += tr
        new_soc = float(np.sum(e ** 2))             # per-AP stacking (Stage 23b)
        old_soc = float(np.sum(e)) ** 2             # single scalar (old)
        assert abs(new_soc - exact_power) < 1e-6 * max(exact_power, 1.0), (
            "Σ_a e_a² must equal Σ_a tr(Wᴴ R W)"
        )
        assert old_soc >= new_soc - 1e-9, "(Σ_a e_a)² must dominate Σ_a e_a²"
        worst_ratio = max(worst_ratio, old_soc / max(new_soc, 1e-30))
    # the over-count grows with n_tx — confirm it can be large (favorable
    # config has n_ap≈10 ⇒ up to ~10× inflation of the CSI-error power)
    assert worst_ratio > 2.0, (
        f"expected the old single-scalar form to over-count by >2× in some "
        f"draw; max observed ratio was {worst_ratio:.2f}"
    )


@_register("Test 16: [needs cordis] imported helpers operate on the per-AP "
           "err vector end-to-end")
def test_16_imported_helpers():
    try:
        from cordis.algorithms.joint_opt import (
            _zeros_consensus, _sum_local_contributions,
            _consensus_residual, _z_diff_norm,
            LocalContribution, ConsensusVector,
        )
    except Exception as e:                          # ImportError / ModuleNotFound / cvxpy
        raise _SkipTest(f"cordis not importable here ({type(e).__name__})")

    n_mui, n_t, n_tx = 3, 2, 5
    z0 = _zeros_consensus(n_mui, n_t, n_tx)
    assert isinstance(z0.err, np.ndarray) and z0.err.shape == (n_tx,), (
        "_zeros_consensus err must be a length-n_tx vector"
    )

    tx_idx = [10, 20, 30, 40, 50]
    rng = np.random.default_rng(1)
    amps = {a: float(abs(rng.standard_normal())) for a in tx_idx}
    l_dict = {
        (a, 0): LocalContribution(
            cds=1.0 + 0j,
            mui=np.ones(n_mui, dtype=np.complex128),
            s2ci=np.ones(n_t, dtype=np.complex128),
            err=amps[a],
        )
        for a in tx_idx
    }
    summed = _sum_local_contributions(l_dict, tx_idx, 0, n_mui, n_t)
    assert summed.err.shape == (n_tx,) and np.allclose(
        summed.err, [amps[a] for a in tx_idx]
    ), "_sum_local_contributions must gather err per-AP in tx order"
    assert abs(summed.cds - len(tx_idx)) < 1e-12, "cds must be coherently summed"
    assert abs(float(np.sum(summed.err ** 2))
               - sum(v * v for v in amps.values())) < 1e-9, (
        "‖z^err‖² must equal Σ_a e_a² (exact power)"
    )

    za = ConsensusVector(cds=0j, mui=np.zeros(n_mui, np.complex128),
                         s2ci=np.zeros(n_t, np.complex128),
                         err=np.array([1., 2., 3., 4., 5.]))
    zb = ConsensusVector(cds=0j, mui=np.zeros(n_mui, np.complex128),
                         s2ci=np.zeros(n_t, np.complex128),
                         err=np.array([1., 2., 3., 4., 6.]))
    assert abs(_consensus_residual(za, zb) - 1.0) < 1e-12, (
        "_consensus_residual must handle the vector err block"
    )
    assert abs(_z_diff_norm(za, zb) - 1.0) < 1e-12, (
        "_z_diff_norm must handle the vector err block"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 23 follow-up — feasibility-gated early stop
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test 17: patience early-stop is feasibility-gated "
           "(the gate includes `and best_feasible`)")
def test_17_early_stop_feasibility_gated():
    src = _src()
    # Isolate the early-stop `if (...)` guard and require `and best_feasible`
    # inside it.  Without the gate, patience can fire on an early low-residual
    # but still-infeasible iterate and return a sub-gamma beamformer.
    m = re.search(
        r'if\s*\(\s*\n\s*best_iter_criterion\s+in\s+\(\s*"residual_norm",\s*'
        r'"feasible_then_residual"\s*\).*?\):',
        src, re.S,
    )
    assert m is not None, (
        "could not locate the patience early-stop `if (...)` guard"
    )
    block = m.group(0)
    assert re.search(r'\band\s+best_feasible\b', block), (
        "early-stop guard must include `and best_feasible` (Stage 23 "
        "follow-up): the patience clock must not run while the incumbent is "
        "infeasible, or it returns a sub-gamma iterate (the -9 dB failure "
        "mode on the convergence trace)"
    )
    # sanity: the gate should NOT have lost the criterion restriction
    assert "min_sinr" not in block, (
        "the early-stop gate should not enable the legacy min_sinr criterion"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    width = 78
    print("=" * width)
    print(f"  Stage 23 validator — {len(_TESTS)} tests")
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

