"""
tests/test_source_hygiene.py
============================
Compile-time hygiene of the shipped package.

These are not science tests; they guard the ``pytest -W error`` gate itself.
A ``SyntaxWarning`` raised while *compiling* a module becomes a ``SyntaxError``
under ``-W error``, which means a stray escape sequence in a docstring can take
down every test that imports the package, and it does so only on a cold
``__pycache__``, which makes it maddening to reproduce.

Known offenders are listed in :data:`KNOWN_COMPILE_WARNINGS` and marked
``xfail(strict=True)``. When a finding is corrected the strict xfail turns into
a failure, which is the reminder to delete both the entry here and the matching
``filterwarnings`` line in ``pytest.ini``.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "cordis"

#: repo-relative path → finding ID that tracks the defect.
KNOWN_COMPILE_WARNINGS: dict[str, str] = {
    # Non-raw docstring containing "\%" at cordis/plotting/cdf.py:46.
    "cordis/plotting/cdf.py": "F-00-01",
}


def _package_modules() -> list[Path]:
    return sorted(
        p for p in PKG_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _param(path: Path):
    rel = path.relative_to(REPO_ROOT).as_posix()
    finding = KNOWN_COMPILE_WARNINGS.get(rel)
    marks = (
        [pytest.mark.xfail(
            strict=True,
            reason=f"known defect {finding}: compiling {rel} emits a "
                   f"SyntaxWarning; remove this entry and the matching "
                   f"pytest.ini filter once it is fixed.",
        )]
        if finding else []
    )
    return pytest.param(path, id=rel, marks=marks)


@pytest.mark.parametrize("module_path",
                         [_param(p) for p in _package_modules()])
def test_module_compiles_without_warnings(module_path: Path) -> None:
    """
    Every ``cordis`` module compiles cleanly.

    Property: ``compile()`` emits no warning of any category. Source of the
    requirement: verification plan §5.1 (``pytest -W error`` must be clean) and
    §5.6 (anticipated ``-W error`` hazards).
    """
    source = module_path.read_text(encoding="utf-8")
    rel = module_path.relative_to(REPO_ROOT).as_posix()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(source, rel, "exec")
    assert not caught, (
        f"{rel} emits {len(caught)} compile warning(s): "
        + "; ".join(f"{w.category.__name__} line {w.lineno}: {w.message}"
                    for w in caught)
    )


def test_all_package_modules_are_parseable() -> None:
    """
    Every ``cordis`` module parses as Python.

    A cheap guard against a half-saved file silently breaking collection for a
    whole stage. Parse errors are unconditional failures; unlike warnings,
    they are never expected.

    Warnings are suppressed locally: under ``pytest -W error`` a compile-time
    ``SyntaxWarning`` is promoted to ``SyntaxError``, which would make this test
    duplicate (and be masked by) the parametrized warning test above. Only
    genuine parse failures should reach the assertion.
    """
    broken = []
    for path in _package_modules():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                broken.append(f"{path.relative_to(REPO_ROOT)}: {exc}")
    assert not broken, "unparseable modules:\n  " + "\n  ".join(broken)


def test_known_compile_warning_registry_is_current() -> None:
    """
    Every path in :data:`KNOWN_COMPILE_WARNINGS` still exists.

    Stops the registry from silently accumulating entries for files that were
    renamed or deleted, which would leave a ``pytest.ini`` filter in place with
    nothing behind it.
    """
    missing = [rel for rel in KNOWN_COMPILE_WARNINGS
               if not (REPO_ROOT / rel).exists()]
    assert not missing, f"stale KNOWN_COMPILE_WARNINGS entries: {missing}"
