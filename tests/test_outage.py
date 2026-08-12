"""
tests/test_outage.py
====================
Tests for the outage/coverage feasibility metrics in
``cordis.metrics.outage``, the module every feasibility claim in the paper's
Section VI rests on (``P_out``, coverage, served-trial rate).

Run from the repo root::

    pytest tests/test_outage.py -v

``sys.path`` setup lives in ``tests/conftest.py``; this module does not need it.

Stage 5 of the verification plan extends this file with the missing cases
(percentile convention, the ``>= eta`` boundary, and the ``eta * gamma_db``
call made by ``fig_cdf``).
"""
import numpy as np

from cordis.metrics.outage import (
    outage_probability, likely_sinr_db, coverage_per_trial,
    served_trial_rate, strict_infeasibility_rate,
    _ar_outage_probability, _ar_likely_sinr_db,
    _ar_coverage_per_trial, _ar_served_trial_rate,
    attach_to_algorithm_result,
)

# Hand-built per-user SINR (dB), shape (4 trials, 3 users)
SINR = np.array([
    [ 2.0,  6.0,  8.0],   # trial 0: min 2 ; users>=5 -> 2/3
    [ 5.0,  5.0,  5.0],   # trial 1: min 5 ; users>=5 -> 3/3
    [10.0, 10.0, 10.0],   # trial 2: all 10
    [ 1.0,  4.0,  9.0],   # trial 3: min 1 ; users>=5 -> 1/3
])
FLAT = SINR.ravel()       # 12 values
GAMMA = 5.0


# ── pure helpers ─────────────────────────────────────────────────────

def test_outage_probability():
    assert abs(outage_probability(FLAT, GAMMA) - 3 / 12) < 1e-12
    assert outage_probability(FLAT, 0.0) == 0.0
    assert outage_probability(FLAT, 100.0) == 1.0


def test_likely_sinr_db():
    # sorted = [1,2,4,5,5,5,6,8,9,10,10,10]; pos=(12-1)*0.1=1.1 -> 2+.1*(4-2)=2.2
    assert abs(likely_sinr_db(FLAT, 0.1) - 2.2) < 1e-12
    # median of 12 -> (sorted[5]+sorted[6])/2 = (5+6)/2 = 5.5
    assert abs(likely_sinr_db(FLAT, 0.5) - 5.5) < 1e-12


def test_coverage_per_trial():
    np.testing.assert_allclose(
        coverage_per_trial(SINR, GAMMA), [2 / 3, 1.0, 1.0, 1 / 3])


def test_served_trial_rate():
    assert abs(served_trial_rate(SINR, GAMMA, 0.9) - 0.5) < 1e-12
    assert abs(served_trial_rate(SINR, GAMMA, 0.6) - 0.75) < 1e-12


def test_strict_infeasibility_rate():
    assert abs(strict_infeasibility_rate(SINR, GAMMA) - 0.5) < 1e-12


def test_consistency_outage_likely():
    # outage(likely(eps)) should recover eps for an eps on a sample
    g = likely_sinr_db(FLAT, 0.25)
    assert abs(outage_probability(FLAT, g) - 0.25) < 0.02


def test_empty_guards():
    assert np.isnan(outage_probability(np.zeros(0), 5.0))
    assert np.isnan(likely_sinr_db(np.zeros(0), 0.1))
    assert coverage_per_trial(np.zeros((0, 3)), 5.0).size == 0
    assert np.isnan(served_trial_rate(np.zeros((0, 3)), 5.0))


# ── AlgorithmResult-style adapter (duck-typed stub) ──────────────────

class _StubStats:
    def __init__(self, sinr_db_mat):
        self._db = np.asarray(sinr_db_mat, float)

    @property
    def sinr_per_trial_per_user_db(self):
        return self._db

    @property
    def all_sinr_db_flat(self):
        return self._db.ravel()


class _StubResult:
    def __init__(self, sinr_db_mat):
        self.sinr_stats = _StubStats(sinr_db_mat)

    def has_metric(self, metric):
        return self.sinr_stats is not None

    def _samples(self, metric):
        return self.sinr_stats.all_sinr_db_flat


def test_adapter_functions():
    r = _StubResult(SINR)
    assert abs(_ar_outage_probability(r, 5.0) - 3 / 12) < 1e-12
    assert abs(_ar_likely_sinr_db(r, 0.1) - 2.2) < 1e-12
    np.testing.assert_allclose(_ar_coverage_per_trial(r, 5.0),
                               [2 / 3, 1.0, 1.0, 1 / 3])
    assert abs(_ar_served_trial_rate(r, 5.0, 0.9) - 0.5) < 1e-12


def test_attach_to_algorithm_result():
    class _AR:
        pass
    attach_to_algorithm_result(_AR)
    inst = _AR()
    inst.sinr_stats = _StubStats(SINR)
    inst.has_metric = lambda m: True
    inst._samples = lambda m: SINR.ravel()
    assert abs(_AR.outage_probability(inst, 5.0) - 3 / 12) < 1e-12
    assert abs(_AR.served_trial_rate(inst, 5.0, 0.9) - 0.5) < 1e-12


def test_none_stats_guards():
    r = _StubResult(SINR)
    r.sinr_stats = None
    assert np.isnan(_ar_served_trial_rate(r, 5.0))
    assert _ar_coverage_per_trial(r, 5.0).size == 0


# =====================================================================
# Stage 5 additions: percentile convention, the eta boundary, and the
# eta * gamma_db call that build_fig_cdf.py makes.
# =====================================================================

import pytest


def test_outage_uses_a_strict_inequality():
    """
    `P_out(gamma) = Pr(SINR < gamma)`, strict. A user sitting exactly on the
    floor counts as served, which is what makes `P_out` and `coverage` exact
    complements.
    """
    on_floor = np.array([5.0, 5.0, 5.0])
    assert outage_probability(on_floor, 5.0) == 0.0
    assert outage_probability(on_floor, 5.0001) == 1.0


def test_outage_and_coverage_are_complements():
    """`P_out(gamma)` over the pool equals `1 - mean(coverage_per_trial)`."""
    p = outage_probability(FLAT, GAMMA)
    c = coverage_per_trial(SINR, GAMMA).mean()
    assert p + c == pytest.approx(1.0)


def test_served_uses_a_non_strict_eta_comparison():
    """
    `served_trial_rate` counts a trial when coverage is `>= eta`, not `> eta`.

    With 3 users the achievable coverages are 0, 1/3, 2/3, 1. At `eta = 2/3`
    the boundary trial must count as served; a strict `>` would drop it and
    silently lower every served rate in Fig. 2's right panel.
    """
    cov = coverage_per_trial(SINR, GAMMA)
    np.testing.assert_allclose(cov, [2 / 3, 1.0, 1.0, 1 / 3])
    assert served_trial_rate(SINR, GAMMA, 2 / 3) == pytest.approx(0.75)
    assert served_trial_rate(SINR, GAMMA, 2 / 3 + 1e-9) == pytest.approx(0.5)


def test_served_at_eta_one_matches_the_strict_criterion():
    """`eta = 1` means every user must clear gamma, i.e. the legacy metric."""
    assert served_trial_rate(SINR, GAMMA, 1.0) == pytest.approx(
        1.0 - strict_infeasibility_rate(SINR, GAMMA))


def test_likely_sinr_uses_the_linear_percentile_convention():
    """
    `likely_sinr_db(eps)` is `numpy.percentile(pool, 100 eps)`, i.e. linear
    interpolation between order statistics, not a nearest-rank quantile.
    """
    assert likely_sinr_db(FLAT, 0.0) == pytest.approx(FLAT.min())
    assert likely_sinr_db(FLAT, 1.0) == pytest.approx(FLAT.max())
    assert likely_sinr_db(FLAT, 0.5) == pytest.approx(np.percentile(FLAT, 50))


def test_likely_sinr_inverts_outage_probability():
    """
    `outage_probability(likely_sinr_db(eps)) ~ eps`, the identity the module
    docstring claims.

    On a finite pool the two can differ by up to one sample, because
    `likely_sinr_db` interpolates between order statistics while
    `outage_probability` counts them. The tolerance is therefore `1/n`, not
    zero, and the test uses a pool large enough for that to be meaningful.
    """
    rng = np.random.default_rng(0)
    pool = rng.normal(5.0, 6.0, size=4000)
    for eps in (0.05, 0.1, 0.25, 0.5):
        g = likely_sinr_db(pool, eps)
        assert outage_probability(pool, g) == pytest.approx(
            eps, abs=2.0 / pool.size)


def test_outage_is_monotone_in_the_threshold():
    prev = -1.0
    for g in (-10.0, 0.0, 2.0, 5.0, 9.0, 50.0):
        cur = outage_probability(FLAT, g)
        assert cur >= prev
        prev = cur


def test_served_rate_is_monotone_in_eta():
    prev = 2.0
    for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
        cur = served_trial_rate(SINR, GAMMA, eta)
        assert cur <= prev + 1e-12
        prev = cur


def test_scaling_the_db_threshold_by_eta_is_not_a_coverage_relaxation():
    """
    Pins F-05-01. `build_fig_cdf.py:334` calls
    `outage_probability(flat, eta * gamma_db)` in its `outage` mode.

    `eta` is a fraction of users. Multiplying a dB threshold by it evaluates the
    outage at a **different SINR target** (4.5 dB instead of 5 dB), which is not
    a relaxation of the coverage requirement and is not dimensionally meaningful:
    the same `eta` applied to a target of 0 dB would leave it unchanged, and
    applied to a negative target would raise it.
    """
    eta, gamma = 0.9, 5.0
    rng = np.random.default_rng(1)
    pool = rng.normal(5.0, 6.0, size=20000)
    shifted = outage_probability(pool, eta * gamma)
    correct = outage_probability(pool, gamma)
    assert correct > shifted, (shifted, correct)
    assert correct - shifted > 0.01

    # The scaling is not a coverage relaxation: it does nothing at a 0 dB
    # target and *raises* the threshold for a negative one, so its effect
    # depends on where the target happens to sit on the dB axis.
    assert eta * 0.0 == 0.0
    assert eta * -10.0 > -10.0


def test_empty_and_degenerate_inputs_are_handled():
    assert np.isnan(strict_infeasibility_rate(np.zeros((0, 3)), 5.0))
    assert coverage_per_trial(np.zeros(5), 5.0).size == 0        # not 2-D
    assert np.isnan(served_trial_rate(np.zeros((0, 2)), 5.0, 0.9))
