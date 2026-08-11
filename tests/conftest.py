"""
tests/conftest.py
=================
Shared pytest configuration and fixtures for the CORDIS test suite.

Responsibilities (verification plan §5.4):

* put the repository root on ``sys.path`` so ``import cordis`` works without a
  ``pip install -e .`` and without per-file ``sys.path.insert`` hacks;
* force the non-interactive matplotlib backend **before** anything imports
  ``pyplot``, so plotting tests never need a display;
* provide the shared fixtures every later stage builds on: a real default
  config, a deliberately tiny config, seeded ``SeedSequence`` objects, a small
  built ``Drop``/``Scenario``, and a scratch results directory;
* provide ``skip_if_no_cvxpy`` for ``solver``-marked tests.

Nothing here may import CVXPY or build a Scenario at *collection* time. The
fast gate (``pytest -m "not slow"``) must stay cheap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── repo root on sys.path (must happen before any `cordis` import) ───────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── headless matplotlib (must happen before any pyplot import) ───────────────
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)


# =============================================================================
# Paths
# =============================================================================

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_config_path(repo_root: Path) -> Path:
    """`configs/default.json`, the base every experiment merges onto."""
    return repo_root / "configs" / "default.json"


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """
    A throwaway stand-in for ``results/``.

    ``results/`` itself is cluster-authoritative and read-only for the whole
    verification effort; anything that writes result trees writes here.
    """
    d = tmp_path / "results"
    d.mkdir()
    return d


# =============================================================================
# Configs
# =============================================================================

@pytest.fixture(scope="session")
def default_cfg():
    """
    The real, validated `configs/default.json` as a :class:`CORDISConfig`.

    Session-scoped and therefore **must not be mutated** by tests. Use
    ``tiny_cfg`` (function-scoped, deep-copied) when a test needs to change
    values.
    """
    from cordis.utils.config import load_config
    return load_config("configs/default.json")


@pytest.fixture
def tiny_cfg(default_cfg):
    """
    A deep copy of the default config shrunk to ``n_ap=3, n_ant=4, n_ue=2,
    n_targets=1`` so that solver-backed tests finish in well under a second.

    Dimensions follow verification plan §5.4. ``n_rf_chains`` is pinned to
    ``n_ant`` because :meth:`CORDISConfig.validate` rejects
    ``n_rf_chains > n_ant``, and ``n_sensing_rx`` stays at 1 so that
    ``n_sensing_rx < n_ap`` holds (two transmit APs remain).
    """
    import copy
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.n_ap = 3
    cfg.topology.n_ant = 4
    cfg.topology.n_rf_chains = 4
    cfg.topology.n_ue = 2
    cfg.topology.n_targets = 1
    cfg.topology.n_sensing_rx = 1
    cfg.simulation.n_trials = 1
    cfg.validate()
    return cfg


# =============================================================================
# Seeds
# =============================================================================

#: Fixed entropy for every seeded fixture. Changing this changes every
#: reproducibility baseline in the suite, so don't, add a new constant instead.
TEST_SEED = 20260809


@pytest.fixture
def seed_seq():
    """A fresh, fixed-entropy :class:`numpy.random.SeedSequence`."""
    from numpy.random import SeedSequence
    return SeedSequence(TEST_SEED)


@pytest.fixture
def rng(seed_seq):
    """A fixed-entropy :class:`numpy.random.Generator`."""
    from numpy.random import default_rng
    return default_rng(seed_seq)


# =============================================================================
# Built pipeline objects
# =============================================================================

@pytest.fixture
def tiny_drop(tiny_cfg):
    """A :class:`Drop` (positions, LSF, sensing statistics) on ``tiny_cfg``."""
    from numpy.random import SeedSequence
    from cordis.simulation.scenario import build_drop
    return build_drop(tiny_cfg, SeedSequence(TEST_SEED), drop_idx=0)


@pytest.fixture
def tiny_scenario(tiny_drop):
    """One channel realization + estimate layered on :func:`tiny_drop`."""
    from numpy.random import SeedSequence
    from cordis.simulation.scenario import build_scenario
    return build_scenario(tiny_drop, SeedSequence(TEST_SEED + 1),
                          realization_idx=0)


# =============================================================================
# Solver availability
# =============================================================================

def skip_if_no_cvxpy() -> None:
    """
    ``pytest.skip`` the caller unless CVXPY imports and has a conic solver.

    Call this at the top of any ``solver``-marked test that would otherwise
    raise an opaque import error on a machine without CVXPY.
    """
    cp = pytest.importorskip("cvxpy", reason="CVXPY not installed")
    installed = set(cp.installed_solvers())
    if not (installed & {"CLARABEL", "SCS", "ECOS", "MOSEK"}):
        pytest.skip(f"no conic solver available; installed={sorted(installed)}")


@pytest.fixture
def require_cvxpy():
    """Fixture form of :func:`skip_if_no_cvxpy`."""
    skip_if_no_cvxpy()
