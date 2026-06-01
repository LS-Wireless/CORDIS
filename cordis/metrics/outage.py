"""
cordis/metrics/outage.py
========================
Outage- and coverage-based feasibility metrics for the CORDIS framework.

Motivation
----------
The original feasibility notion in :class:`AlgorithmResult` is
``infeasibility_rate(gamma_db)`` — the fraction of *trials* whose
*worst* user falls below the SINR target γ.  Equivalently a trial is
feasible iff **every** user meets γ in **every** realization.  Under a
stochastic Rician channel, clutter, and imperfect CSI this all-or-nothing
criterion is overly stringent and not the operating point cell-free /
massive-MIMO papers actually report.

This module adds the standard alternatives, all defined on the per-user
SINR pool that :class:`SINRStatistics` already stores
(``sinr_per_trial_per_user`` of shape ``(n_trials, N_ue)``):

  * **Per-user SINR outage probability** ``P_out(γ) = Pr(SINR_u < γ)``
    estimated over all (user, trial) pairs.  This is the headline metric:
    a configuration "supports γ at outage level ε" iff ``P_out(γ) ≤ ε``.
    Note ``P_out(γ) = CDF_SINR(γ)`` so it is just a reading of the SINR
    CDF you already plot — nothing is hidden.

  * **(1−ε)-likely SINR** ``likely_sinr_db(ε)`` — the (100·ε)-th
    percentile of the per-user SINR.  "95%-likely SINR ≥ γ" is the
    Björnson/cell-free convention and is equivalent to ``P_out(γ) ≤ ε``
    with ε = 0.05.

  * **Coverage per trial** and **served-trial rate** — the drop-level
    relaxation: a trial is "served" if at least a fraction η of its
    users meet γ (η = 0.9 ⇒ "90 % of users"); the served-trial rate is
    the fraction of trials that are served (⇒ "in X % of realizations
    ≥ 90 % of users are covered").

The functions are split into (a) pure, array-level helpers that take
numpy arrays (fully unit-tested, no cordis dependency) and (b) thin
:class:`AlgorithmResult` methods that pull the arrays from
``self.sinr_stats`` / ``self._samples`` and delegate to (a).  The latter
are provided here as free functions taking ``result`` as the first
argument so they can be either monkey-patched onto ``AlgorithmResult``
or copied verbatim into ``cordis/simulation/result.py`` as methods.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# (a) Pure array-level helpers  —  no cordis dependency, fully testable
# =============================================================================

def outage_probability(sinr_db_flat: NDArray[np.float64],
                        gamma_db: float) -> float:
    """Per-user SINR outage probability ``Pr(SINR_u < gamma_db)``.

    Parameters
    ----------
    sinr_db_flat : 1-D array
        Pool of per-user SINR values in dB across all (user, trial)
        pairs.  This is ``SINRStatistics.all_sinr_db_flat``.
    gamma_db : float
        SINR target γ in dB.

    Returns
    -------
    float
        Fraction of the pool strictly below γ.  ``nan`` for an empty pool.
    """
    arr = np.asarray(sinr_db_flat, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr < gamma_db))


def likely_sinr_db(sinr_db_flat: NDArray[np.float64],
                   eps: float) -> float:
    """The ``(1-eps)``-likely per-user SINR in dB.

    This is the ``(100*eps)``-th percentile of the per-user SINR pool:
    the SINR exceeded by a fraction ``(1-eps)`` of users.  A
    configuration supports γ at outage ε iff ``likely_sinr_db(eps) >= γ``,
    which is exactly the condition ``outage_probability(γ) <= eps``.

    Parameters
    ----------
    sinr_db_flat : 1-D array
        Per-user SINR pool in dB (``all_sinr_db_flat``).
    eps : float
        Target outage level in [0, 1] (e.g. 0.05 for "95 %-likely").
    """
    arr = np.asarray(sinr_db_flat, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 100.0 * float(eps)))


def coverage_per_trial(sinr_db_matrix: NDArray[np.float64],
                       gamma_db: float) -> NDArray[np.float64]:
    """Fraction of users meeting γ in each trial.

    Parameters
    ----------
    sinr_db_matrix : 2-D array, shape ``(n_trials, n_ue)``
        Per-user SINR in dB for every trial
        (``SINRStatistics.sinr_per_trial_per_user_db``).
    gamma_db : float

    Returns
    -------
    1-D array, shape ``(n_trials,)``
        ``coverage[t] = mean_u 1[SINR_{t,u} >= gamma_db]``.
        Empty array if the input has no trials.
    """
    mat = np.asarray(sinr_db_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.mean(mat >= gamma_db, axis=1)


def served_trial_rate(sinr_db_matrix: NDArray[np.float64],
                      gamma_db: float,
                      eta: float = 0.9) -> float:
    """Fraction of trials in which at least ``eta`` of users meet γ.

    The drop-level relaxation: with η = 0.9 a trial counts as "served"
    when ≥ 90 % of its users clear the SINR floor, and this returns the
    fraction of trials that are served.

    Returns ``nan`` if there are no trials.
    """
    cov = coverage_per_trial(sinr_db_matrix, gamma_db)
    if cov.size == 0:
        return float("nan")
    return float(np.mean(cov >= float(eta)))


def strict_infeasibility_rate(sinr_db_matrix: NDArray[np.float64],
                              gamma_db: float) -> float:
    """Original strict criterion: fraction of trials whose worst user < γ.

    Provided here so callers can cross-check against the existing
    :meth:`AlgorithmResult.infeasibility_rate` and report both numbers.
    Returns ``nan`` for no trials.
    """
    mat = np.asarray(sinr_db_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.size == 0:
        return float("nan")
    return float(np.mean(mat.min(axis=1) < gamma_db))


# =============================================================================
# (b) AlgorithmResult methods  —  copy into cordis/simulation/result.py
#     (shown as free functions taking `result` so they can also be
#      monkey-patched: AlgorithmResult.outage_probability = _ar_outage ...)
# =============================================================================

def _ar_outage_probability(result, gamma_db: float,
                           metric: str = "all_sinr_db_flat") -> float:
    """Per-user SINR outage ``Pr(SINR_u < gamma_db)`` for one algorithm.

    Uses the flattened per-user SINR pool registered as
    ``"all_sinr_db_flat"`` in the metric registry.  Returns ``nan`` if
    the algorithm has no SINR statistics (every trial failed).
    """
    if not result.has_metric(metric):
        return float("nan")
    return outage_probability(result._samples(metric), gamma_db)


def _ar_likely_sinr_db(result, eps: float,
                       metric: str = "all_sinr_db_flat") -> float:
    """``(1-eps)``-likely per-user SINR in dB for one algorithm."""
    if not result.has_metric(metric):
        return float("nan")
    return likely_sinr_db(result._samples(metric), eps)


def _ar_coverage_per_trial(result, gamma_db: float) -> NDArray[np.float64]:
    """Per-trial user-coverage array for one algorithm."""
    if result.sinr_stats is None:
        return np.zeros(0, dtype=np.float64)
    return coverage_per_trial(result.sinr_stats.sinr_per_trial_per_user_db,
                              gamma_db)


def _ar_served_trial_rate(result, gamma_db: float,
                          eta: float = 0.9) -> float:
    """Served-trial rate (≥ η of users meet γ) for one algorithm."""
    if result.sinr_stats is None:
        return float("nan")
    return served_trial_rate(result.sinr_stats.sinr_per_trial_per_user_db,
                             gamma_db, eta=eta)


def attach_to_algorithm_result(algorithm_result_cls) -> None:
    """Monkey-patch the four methods onto :class:`AlgorithmResult`.

    Call once at import time if you prefer not to edit result.py::

        from cordis.metrics.outage import attach_to_algorithm_result
        from cordis.simulation.result import AlgorithmResult
        attach_to_algorithm_result(AlgorithmResult)
    """
    algorithm_result_cls.outage_probability = _ar_outage_probability
    algorithm_result_cls.likely_sinr_db = _ar_likely_sinr_db
    algorithm_result_cls.coverage_per_trial = _ar_coverage_per_trial
    algorithm_result_cls.served_trial_rate = _ar_served_trial_rate

