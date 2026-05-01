"""
cordis/metrics/aggregation.py
==============================
Aggregation of per-trial metrics into ergodic statistics.

Each Monte Carlo simulation produces a list of :class:`SINRMetrics` and
:class:`SCNRMetrics`, one per channel realization.  This module collapses
those lists into compact :class:`SINRStatistics` and :class:`SCNRStatistics`
containers that hold:

  * the raw per-trial arrays (used for CDF plotting)
  * the ergodic averages (per-user / per-pair)
  * scalar summary statistics (mean, std, percentiles)

Notation
--------
"Ergodic" here follows the standard convention in the field:
  - ergodic SINR   = arithmetic mean of per-trial SINR values
  - ergodic rate   = arithmetic mean of per-trial log2(1+SINR) values
                     (the rigorous Shannon ergodic capacity)

Both are reported because the literature uses both, and they answer
slightly different questions.  Use ergodic_rate_per_user for capacity
calculations; use ergodic_sinr_per_user when comparing against an SINR
target γ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.metrics.sinr import SINRMetrics
from cordis.metrics.scnr import SCNRMetrics
from cordis.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SINRStatistics
# =============================================================================

@dataclass
class SINRStatistics:
    """
    Aggregated SINR statistics across multiple Monte Carlo trials.

    Attributes
    ----------
    n_trials : int
    n_users  : int

    sinr_per_trial_per_user : np.ndarray, shape (n_trials, N_ue)
        Linear-scale SINR for every user in every trial.
    rate_per_trial_per_user : np.ndarray, shape (n_trials, N_ue)
        Per-user rates [bits/s/Hz].

    ergodic_sinr_per_user : np.ndarray, shape (N_ue,)
        E[SINR_u] — arithmetic mean across trials.
    ergodic_rate_per_user : np.ndarray, shape (N_ue,)
        E[log2(1 + SINR_u)] — true Shannon ergodic capacity.

    min_sinr_per_trial : np.ndarray, shape (n_trials,)
        Minimum SINR across users in each trial.
    sum_rate_per_trial : np.ndarray, shape (n_trials,)
        Sum-rate Σ_u log2(1+SINR_u) in each trial.
    mean_sinr_per_trial : np.ndarray, shape (n_trials,)
        Arithmetic mean SINR across users in each trial.

    Power-decomposition diagnostics (averaged across trials, per user):
    p_cds_mean, p_mui_mean, p_s2ci_mean, p_csi_error_mean : np.ndarray
    """

    n_trials: int
    n_users:  int

    # Raw arrays
    sinr_per_trial_per_user: NDArray[np.float64]   # (n_trials, N_ue)
    rate_per_trial_per_user: NDArray[np.float64]   # (n_trials, N_ue)

    # Per-trial scalar aggregates
    min_sinr_per_trial:      NDArray[np.float64]   # (n_trials,)
    sum_rate_per_trial:      NDArray[np.float64]   # (n_trials,)
    mean_sinr_per_trial:     NDArray[np.float64]   # (n_trials,)

    # Per-user ergodic
    ergodic_sinr_per_user:   NDArray[np.float64]   # (N_ue,)
    ergodic_rate_per_user:   NDArray[np.float64]   # (N_ue,)

    # Power decomposition averages (for diagnostics)
    p_cds_mean:        NDArray[np.float64]
    p_mui_mean:        NDArray[np.float64]
    p_s2ci_mean:       NDArray[np.float64]
    p_csi_error_mean:  NDArray[np.float64]
    p_noise_mean:      float

    # ── dB versions (computed lazily as @property) ───────────────────────

    @property
    def sinr_per_trial_per_user_db(self) -> NDArray[np.float64]:
        return 10.0 * np.log10(np.maximum(self.sinr_per_trial_per_user, 1e-30))

    @property
    def min_sinr_per_trial_db(self) -> NDArray[np.float64]:
        return 10.0 * np.log10(np.maximum(self.min_sinr_per_trial, 1e-30))

    @property
    def ergodic_sinr_per_user_db(self) -> NDArray[np.float64]:
        return 10.0 * np.log10(np.maximum(self.ergodic_sinr_per_user, 1e-30))

    # ── Flat arrays useful for CDF plotting ───────────────────────────────

    @property
    def all_sinr_db_flat(self) -> NDArray[np.float64]:
        """All SINR values across trials × users, dB, flattened."""
        return self.sinr_per_trial_per_user_db.flatten()

    @property
    def all_rate_flat(self) -> NDArray[np.float64]:
        """All rate values across trials × users, flattened."""
        return self.rate_per_trial_per_user.flatten()

    # ── Summary statistics ────────────────────────────────────────────────

    @property
    def mean_min_sinr(self) -> float:
        """Average of per-trial min-SINR (linear)."""
        return float(self.min_sinr_per_trial.mean())

    @property
    def mean_min_sinr_db(self) -> float:
        return float(self.min_sinr_per_trial_db.mean())

    @property
    def mean_sum_rate(self) -> float:
        return float(self.sum_rate_per_trial.mean())

    @property
    def std_sum_rate(self) -> float:
        return float(self.sum_rate_per_trial.std())

    def percentile(self, q: float, kind: str = "min_sinr_db") -> float:
        """
        Return the q-th percentile of a chosen distribution.

        Parameters
        ----------
        q : float
            Percentile in [0, 100].
        kind : str
            Which array to compute on.  Choices:
            ``"min_sinr_db"``, ``"min_sinr"``, ``"sum_rate"``,
            ``"sinr_db"`` (all users × trials), ``"rate"`` (all).
        """
        arrays = {
            "min_sinr":    self.min_sinr_per_trial,
            "min_sinr_db": self.min_sinr_per_trial_db,
            "sum_rate":    self.sum_rate_per_trial,
            "sinr_db":     self.all_sinr_db_flat,
            "rate":        self.all_rate_flat,
        }
        if kind not in arrays:
            raise ValueError(f"Unknown 'kind': {kind}.  Choose from {list(arrays)}.")
        return float(np.percentile(arrays[kind], q))


def aggregate_sinr(metrics_list: List[SINRMetrics]) -> SINRStatistics:
    """
    Combine a list of per-trial :class:`SINRMetrics` into one
    :class:`SINRStatistics` summary.

    Parameters
    ----------
    metrics_list : list[SINRMetrics]
        Output of ``compute_sinr`` for each Monte Carlo trial.

    Returns
    -------
    SINRStatistics

    Raises
    ------
    ValueError
        If ``metrics_list`` is empty or trials have inconsistent N_ue.
    """
    if not metrics_list:
        raise ValueError("aggregate_sinr: empty metrics_list.")

    n_trials = len(metrics_list)
    n_ue     = len(metrics_list[0].sinr_per_user)

    for i, m in enumerate(metrics_list):
        if len(m.sinr_per_user) != n_ue:
            raise ValueError(
                f"Trial {i} has N_ue={len(m.sinr_per_user)}, expected {n_ue}."
            )

    sinr_arr = np.stack([m.sinr_per_user for m in metrics_list])   # (n, N_ue)
    rate_arr = np.stack([m.rate_per_user for m in metrics_list])

    p_cds  = np.stack([m.p_cds for m in metrics_list])
    p_mui  = np.stack([m.p_mui for m in metrics_list])
    p_s2ci = np.stack([m.p_s2ci for m in metrics_list])
    p_csi  = np.stack([m.p_csi_error for m in metrics_list])

    return SINRStatistics(
        n_trials=n_trials,
        n_users=n_ue,
        sinr_per_trial_per_user=sinr_arr,
        rate_per_trial_per_user=rate_arr,
        min_sinr_per_trial=sinr_arr.min(axis=1),
        sum_rate_per_trial=rate_arr.sum(axis=1),
        mean_sinr_per_trial=sinr_arr.mean(axis=1),
        ergodic_sinr_per_user=sinr_arr.mean(axis=0),
        ergodic_rate_per_user=rate_arr.mean(axis=0),
        p_cds_mean=p_cds.mean(axis=0),
        p_mui_mean=p_mui.mean(axis=0),
        p_s2ci_mean=p_s2ci.mean(axis=0),
        p_csi_error_mean=p_csi.mean(axis=0),
        p_noise_mean=float(np.mean([m.p_noise for m in metrics_list])),
    )


# =============================================================================
# SCNRStatistics
# =============================================================================

@dataclass
class SCNRStatistics:
    """
    Aggregated SCNR statistics across multiple trials.

    Attributes
    ----------
    n_trials : int
    pair_keys : list[(ar_idx, tg_idx)]
        Sorted list of (RX-AP, target) pairs that appear in at least one trial.

    scnr_per_trial_per_pair : np.ndarray, shape (n_trials, n_pairs)
        Linear-scale SCNR for every pair in every trial.  Pairs absent
        from a particular trial are filled with zeros.

    weighted_sum_per_trial : np.ndarray, shape (n_trials,)
        Σ ω_t · SCNR_{a_r, t} in each trial.
    min_scnr_per_trial : np.ndarray, shape (n_trials,)
    surrogate_per_trial : np.ndarray, shape (n_trials,)
        Value of the optimization objective in each trial.

    ergodic_scnr_per_pair : np.ndarray, shape (n_pairs,)
        E[SCNR] for each (ar, tg) pair across trials.
    """

    n_trials: int
    pair_keys: List[Tuple[int, int]]

    # Raw arrays
    scnr_per_trial_per_pair: NDArray[np.float64]   # (n_trials, n_pairs)

    # Per-trial scalar aggregates
    weighted_sum_per_trial: NDArray[np.float64]    # (n_trials,)
    min_scnr_per_trial:     NDArray[np.float64]    # (n_trials,)
    mean_scnr_per_trial:    NDArray[np.float64]    # (n_trials,)
    surrogate_per_trial:    NDArray[np.float64]    # (n_trials,)

    # Per-pair ergodic
    ergodic_scnr_per_pair:  NDArray[np.float64]    # (n_pairs,)

    # ── dB versions ───────────────────────────────────────────────────────

    @property
    def scnr_per_trial_per_pair_db(self) -> NDArray[np.float64]:
        return 10.0 * np.log10(np.maximum(self.scnr_per_trial_per_pair, 1e-30))

    @property
    def weighted_sum_per_trial_db(self) -> NDArray[np.float64]:
        return 10.0 * np.log10(np.maximum(self.weighted_sum_per_trial, 1e-30))

    @property
    def min_scnr_per_trial_db(self) -> NDArray[np.float64]:
        return 10.0 * np.log10(np.maximum(self.min_scnr_per_trial, 1e-30))

    @property
    def ergodic_scnr_per_pair_db(self) -> NDArray[np.float64]:
        return 10.0 * np.log10(np.maximum(self.ergodic_scnr_per_pair, 1e-30))

    # ── Flat arrays for CDF ───────────────────────────────────────────────

    @property
    def all_scnr_db_flat(self) -> NDArray[np.float64]:
        return self.scnr_per_trial_per_pair_db.flatten()

    # ── Summary statistics ────────────────────────────────────────────────

    @property
    def mean_weighted_sum_scnr(self) -> float:
        return float(self.weighted_sum_per_trial.mean())

    @property
    def mean_weighted_sum_scnr_db(self) -> float:
        return float(self.weighted_sum_per_trial_db.mean())

    @property
    def mean_min_scnr_db(self) -> float:
        return float(self.min_scnr_per_trial_db.mean())

    @property
    def mean_surrogate(self) -> float:
        return float(self.surrogate_per_trial.mean())

    def percentile(self, q: float, kind: str = "min_scnr_db") -> float:
        arrays = {
            "min_scnr":         self.min_scnr_per_trial,
            "min_scnr_db":      self.min_scnr_per_trial_db,
            "weighted_sum":     self.weighted_sum_per_trial,
            "weighted_sum_db":  self.weighted_sum_per_trial_db,
            "scnr_db":          self.all_scnr_db_flat,
            "surrogate":        self.surrogate_per_trial,
        }
        if kind not in arrays:
            raise ValueError(f"Unknown 'kind': {kind}.  Choose from {list(arrays)}.")
        return float(np.percentile(arrays[kind], q))


def aggregate_scnr(metrics_list: List[SCNRMetrics]) -> SCNRStatistics:
    """
    Combine per-trial :class:`SCNRMetrics` into one :class:`SCNRStatistics`.

    Parameters
    ----------
    metrics_list : list[SCNRMetrics]

    Returns
    -------
    SCNRStatistics
    """
    if not metrics_list:
        raise ValueError("aggregate_scnr: empty metrics_list.")

    n_trials = len(metrics_list)

    # Collect the union of all pair keys across trials
    all_pairs = set()
    for m in metrics_list:
        all_pairs.update(m.scnr_per_pair.keys())
    pair_keys = sorted(all_pairs)
    n_pairs   = len(pair_keys)

    # Build the (n_trials, n_pairs) matrix; missing pairs → 0
    scnr_arr = np.zeros((n_trials, n_pairs), dtype=float)
    for t, m in enumerate(metrics_list):
        for j, key in enumerate(pair_keys):
            scnr_arr[t, j] = m.scnr_per_pair.get(key, 0.0)

    weighted = np.array([m.weighted_sum_scnr for m in metrics_list])
    surrogate = np.array([m.surrogate for m in metrics_list])

    # min/mean per trial: only over the pairs that are present (non-zero)
    min_per_trial = np.zeros(n_trials)
    mean_per_trial = np.zeros(n_trials)
    for t, m in enumerate(metrics_list):
        if m.scnr_per_pair:
            vals = np.array(list(m.scnr_per_pair.values()))
            min_per_trial[t]  = vals.min()
            mean_per_trial[t] = vals.mean()

    return SCNRStatistics(
        n_trials=n_trials,
        pair_keys=pair_keys,
        scnr_per_trial_per_pair=scnr_arr,
        weighted_sum_per_trial=weighted,
        min_scnr_per_trial=min_per_trial,
        mean_scnr_per_trial=mean_per_trial,
        surrogate_per_trial=surrogate,
        ergodic_scnr_per_pair=scnr_arr.mean(axis=0),
    )

