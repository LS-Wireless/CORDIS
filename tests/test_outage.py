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
