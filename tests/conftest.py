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


# =============================================================================
# Full-pipeline scenarios for the algorithm stages
# =============================================================================

def build_pipeline_scenario(cfg, seed: int = 808):
    """
    Run the whole channel pipeline once and return the tuple every algorithm
    entry point takes: ``(topo, cfg, est, sensing_stats, assoc, sigma, Pmax)``.

    This mirrors what :class:`cordis.simulation.scenario.Scenario` assembles,
    but without the runner machinery, so an algorithm test depends only on the
    channel layer.
    """
    from cordis.channel.estimation import (
        design_pilot_sequences, run_channel_estimation,
    )
    from cordis.channel.pathloss import (
        compute_large_scale_fading, noise_power_watts, snr_to_tx_power,
    )
    from cordis.channel.rician import (
        compute_channel_statistics, generate_channel_realization,
    )
    from cordis.channel.sensing_assignment import assign_sensing
    from cordis.channel.sensing_channel import compute_sensing_statistics
    from cordis.channel.topology import generate_topology
    from cordis.utils.io_utils import child_rng, make_rng

    r = make_rng(seed)
    topo = generate_topology(cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
    st = compute_channel_statistics(topo, cfg, lsf, child_rng(r))
    real = generate_channel_realization(topo, cfg, lsf, st, child_rng(r))
    Phi, cs = design_pilot_sequences(topo.n_ue, cfg.channel.tau_p, child_rng(r))
    est = run_channel_estimation(topo, cfg, lsf, st, real, child_rng(r), Phi, cs)
    ss = compute_sensing_statistics(topo, cfg, lsf, child_rng(r))
    assoc = assign_sensing(topo, cfg, lsf)
    sigma = noise_power_watts(cfg.frequency.bandwidth_hz,
                              cfg.channel.noise_figure_db,
                              cfg.channel.noise_temp_k)
    Pmax = snr_to_tx_power(cfg.channel.snr_db, sigma)
    return topo, cfg, est, ss, assoc, sigma, Pmax


def _sized_cfg(default_cfg, n_ap, n_ant, n_ue, n_targets=1, n_spatial=800):
    import copy
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.n_ap = n_ap
    cfg.topology.n_ant = cfg.topology.n_rf_chains = n_ant
    cfg.topology.n_ue = n_ue
    cfg.topology.n_targets = n_targets
    cfg.topology.n_sensing_rx = 1
    cfg.channel.n_spatial_samples = n_spatial
    cfg.validate()
    return cfg


@pytest.fixture(scope="module")
def centralized_scenario(default_cfg):
    """
    A **well-provisioned** scenario: 8 APs of 12 antennas serving 3 users and
    one target. Large enough that the centralized SOCP is comfortably
    feasible at the default ``gamma = 5`` dB, small enough to solve quickly.
    """
    skip_if_no_cvxpy()
    return build_pipeline_scenario(
        _sized_cfg(default_cfg, n_ap=8, n_ant=12, n_ue=3), seed=3000)


@pytest.fixture(scope="module")
def small_centralized_scenario(default_cfg):
    """
    An **under-provisioned** scenario: 4 APs of 6 antennas for 3 users and a
    target. The SINR floor is out of reach here, which is the regime that
    exposes F-07-02.
    """
    skip_if_no_cvxpy()
    return build_pipeline_scenario(
        _sized_cfg(default_cfg, n_ap=4, n_ant=6, n_ue=3, n_spatial=600),
        seed=909)


@pytest.fixture(scope="module")
def benchmark_scenario(default_cfg):
    """Mid-sized scenario shared by the benchmark-registry tests."""
    skip_if_no_cvxpy()
    return build_pipeline_scenario(
        _sized_cfg(default_cfg, n_ap=6, n_ant=8, n_ue=3, n_targets=2,
                   n_spatial=600),
        seed=515)


@pytest.fixture(scope="module")
def admm_scenario(default_cfg):
    """
    Scenario for the CORDIS-ADMM loop-control tests (Stage 8b).

    4 APs of 8 antennas, 3 users, one target: small enough that a 12-to-40
    iteration ADMM run finishes in a few seconds, large enough that the default
    ``gamma = 5`` dB is comfortably reachable so feasibility-dependent branches
    (best-iterate selection, the early-stop gate) are actually exercised.
    """
    skip_if_no_cvxpy()
    return build_pipeline_scenario(
        _sized_cfg(default_cfg, n_ap=4, n_ant=8, n_ue=3, n_spatial=600),
        seed=717)
