#!/usr/bin/env python3
"""
scripts/validate_stage13.py
============================

Stage 13 validator.

Stage 13 introduces ``scripts/sync_results_from_hpc3.sh`` — a wrapper
around rsync for pulling experiment results from HPC3 to laptop.  The
script is **safety-first** (additive by default, no upload, no
implicit delete) and **discoverable** (--help with a usage block).

Tests verify:

* **file presence + executability** — script exists and has +x bit
* **bash syntax** — passes ``bash -n``
* **--help contract** — exits 0, includes USAGE/EXAMPLES/ENVIRONMENT
  sections, mentions every flag
* **safety defaults** — script does NOT default to ``--delete`` and
  the ``--delete`` flag is opt-in only (not on by default)
* **error paths** — unknown flag, two experiments, path-like
  experiment all exit with non-zero code
* **env-var contract** — script reads ``HPC3_HOST`` and ``HPC3_REPO``
  with sensible defaults

Run from the repo root::

    python3 scripts/validate_stage13.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT    = REPO_ROOT / "scripts" / "sync_results_from_hpc3.sh"


class _SkipTest(Exception):
    pass


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        _TESTS.append((label, fn))
        return fn
    return deco


def _run(args: List[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    """Run the sync script with given args, return CompletedProcess."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(REPO_ROOT),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — file presence + executability + syntax
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: sync script exists at scripts/sync_results_from_hpc3.sh")
def test_01_exists():
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


@_register("Test  2: sync script is executable (+x bit)")
def test_02_executable():
    import stat
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, (
        f"{SCRIPT.name} is not executable (chmod +x needed)"
    )


@_register("Test  3: sync script passes 'bash -n' syntax check")
def test_03_bash_syntax():
    proc = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True, text=True, timeout=5.0,
    )
    assert proc.returncode == 0, (
        f"'bash -n {SCRIPT.name}' failed:\n{proc.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — --help contract
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  4: --help exits 0 and prints usage")
def test_04_help_exits_zero():
    proc = _run(["--help"])
    assert proc.returncode == 0, (
        f"--help exited {proc.returncode}; stderr:\n{proc.stderr}"
    )
    assert "Usage" in proc.stdout or "SYNOPSIS" in proc.stdout, (
        "--help output should include SYNOPSIS / Usage section; got: "
        f"{proc.stdout[:200]}"
    )


@_register("Test  5: -h is a synonym for --help")
def test_05_short_help_works():
    proc = _run(["-h"])
    assert proc.returncode == 0, "-h exited non-zero"
    assert "SYNOPSIS" in proc.stdout or "Usage" in proc.stdout, (
        "-h should produce the same help text as --help"
    )


@_register("Test  6: --help mentions every supported flag")
def test_06_help_mentions_flags():
    proc = _run(["--help"])
    out = proc.stdout
    for flag in ("--dry-run", "--delete", "--quiet", "--help"):
        assert flag in out, (
            f"--help output should document the {flag!r} flag"
        )


@_register("Test  7: --help mentions HPC3_HOST and HPC3_REPO env vars")
def test_07_help_mentions_envvars():
    proc = _run(["--help"])
    out = proc.stdout
    for var in ("HPC3_HOST", "HPC3_REPO"):
        assert var in out, (
            f"--help output should document the {var} env var override"
        )


@_register("Test  8: --help includes at least 3 EXAMPLES")
def test_08_help_has_examples():
    proc = _run(["--help"])
    out = proc.stdout
    assert "EXAMPLE" in out.upper(), (
        "--help output should have an EXAMPLES section"
    )
    # Count distinct `bash sync_results_from_hpc3.sh` invocations in
    # the examples block as a rough proxy for "real examples present".
    n_examples = len(re.findall(r"bash\s+\S*sync_results_from_hpc3\.sh",
                                 out, re.IGNORECASE))
    assert n_examples >= 3, (
        f"--help should show at least 3 usage examples; found {n_examples}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — safety defaults
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  9: --delete is opt-in, not default")
def test_09_delete_is_opt_in():
    """Critical safety property: laptop-only files MUST be preserved
    by default.  The script must not default to --delete behavior."""
    src = SCRIPT.read_text()
    # Variable must default to 0 (off).
    assert re.search(r"^DELETE=0\b", src, re.M), (
        "DELETE should default to 0 (opt-in only).  Letting it default "
        "to on would silently delete laptop-only results — exactly the "
        "behavior the user explicitly asked to avoid."
    )
    # rsync must conditionally add --delete only when DELETE==1.
    assert re.search(r'DELETE\s*-eq\s+1.*--delete', src), (
        "The --delete rsync flag should only be added when DELETE==1; "
        "could not find the conditional gate."
    )


@_register("Test 10: script defaults to no upload (rsync direction is "
           "remote→local)")
def test_10_no_upload_direction():
    """rsync syntax: 'rsync src dst' goes src → dst.  Source must be
    the remote (host:path) and destination must be local."""
    src = SCRIPT.read_text()
    # Look for the canonical REMOTE → LOCAL invocation.
    assert re.search(r'rsync\s+"\$\{RSYNC_FLAGS\[@\]\}"\s+"\$REMOTE"\s+"\$LOCAL"',
                     src), (
        "rsync call should be `rsync ... \"$REMOTE\" \"$LOCAL\"` — "
        "remote first (source), local second (destination).  Reversing "
        "the order would UPLOAD instead of download."
    )
    # And REMOTE must be assembled with HPC3_HOST in front.
    assert re.search(r'REMOTE=.*\$\{HPC3_HOST\}', src), (
        "REMOTE path should include $HPC3_HOST (e.g. 'hpc3:/pub/.../')"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 4 — error paths
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test 11: unknown flag exits with code 2")
def test_11_unknown_flag_rejected():
    proc = _run(["--this-flag-does-not-exist"])
    assert proc.returncode == 2, (
        f"unknown flag should exit 2; got {proc.returncode}"
    )
    assert "unknown option" in proc.stderr.lower() or \
           "unknown option" in proc.stdout.lower(), (
        "error message should mention 'unknown option'"
    )


@_register("Test 12: two EXPERIMENT args rejected")
def test_12_double_experiment_rejected():
    proc = _run(["exp_a", "exp_b"])
    assert proc.returncode == 2, (
        f"passing two experiments should exit 2; got {proc.returncode}"
    )


@_register("Test 13: path-like EXPERIMENT (with /) rejected")
def test_13_path_like_experiment_rejected():
    """Lightweight guard against accidentally passing a path that
    might break out of the results/ tree."""
    proc = _run(["../etc/passwd"])
    assert proc.returncode == 2, (
        f"path-like experiment should exit 2; got {proc.returncode}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 5 — env-var contract
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test 14: HPC3_HOST defaults to 'hpc3' (SSH alias)")
def test_14_default_host_alias():
    src = SCRIPT.read_text()
    assert re.search(r'HPC3_HOST=\"\$\{HPC3_HOST:-hpc3\}\"', src), (
        "HPC3_HOST should default to 'hpc3' (the SSH config alias) "
        "with ${HPC3_HOST:-hpc3} idiom."
    )


@_register("Test 15: HPC3_REPO defaults to /pub/mzafarid/CORDIS but is "
           "overridable")
def test_15_default_repo_path():
    src = SCRIPT.read_text()
    assert re.search(r'HPC3_REPO=\"\$\{HPC3_REPO:-/pub/', src), (
        "HPC3_REPO should default to a /pub/... path and be overridable "
        "via the ${HPC3_REPO:-...} idiom."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 13 validator — {len(_TESTS)} tests")
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
            print(f"      {e}")
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

