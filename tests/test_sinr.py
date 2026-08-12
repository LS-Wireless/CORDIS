"""
tests/test_sinr.py
==================
Per-user SINR and its aggregation
(`cordis/metrics/sinr.py`, `cordis/metrics/aggregation.py`).

Paper reference: `eq:sinr`. The five-term decomposition is what every
feasibility claim in §VI rests on, so each term is checked in isolation and the
whole is checked against an analytically solvable special case.

`eq:sinr` in full:

    SINR_u = |sum_at h_hat_au^H w_au|^2
             / ( sum_{k != u} |sum_at h_hat_au^H w_ak|^2
               + sum_t |sum_at h_hat_au^H w_at|^2
               + sum_at tr(W_at^H R~_au W_at)
               + sigma_n^2 )

Note the placement of the sums: the coherent AP sum is **inside** the modulus
for CDS, MUI and S2CI (the APs combine coherently at the user), and **outside**
for the CSI-error term (the error powers add).
"""
from __future__ import annotations

import numpy as np
import pytest

from cordis.metrics.aggregation import aggregate_sinr
from cordis.metrics.sinr import SINRMetrics, compute_sinr, rate_to_sinr, sinr_to_rate


# =============================================================================
# A hand-built two-AP, two-user, one-target instance
# =============================================================================

class _Est:
    """Minimal stand-in for EstimationResult with hand-chosen values."""
    def __init__(self, h_hat, R_tilde):
        self.h_hat = h_hat
        self.R_tilde = R_tilde


class _AP:
    def __init__(self, idx):
        self.idx = idx


class _Topo:
    def __init__(self, n_ue, n_targets, tx_idx):
        self.n_ue = n_ue
        self.n_targets = n_targets
        self.tx_aps = [_AP(i) for i in tx_idx]


@pytest.fixture
def toy():
    """
    Two transmit APs, two users, one target, two antennas.

    Channels and precoders are chosen so every term can be computed by hand.
    """
    m, n_ue, n_t = 2, 2, 1
    topo = _Topo(n_ue, n_t, [0, 1])
    h = {
        (0, 0): np.array([1.0 + 0j, 0.0]),
        (0, 1): np.array([0.0, 1.0 + 0j]),
        (1, 0): np.array([1.0 + 0j, 0.0]),
        (1, 1): np.array([0.0, 1.0 + 0j]),
    }
    R_t = {k: np.zeros((m, m), dtype=complex) for k in h}
    W = {
        0: np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=complex),
        1: np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=complex),
    }
    return topo, _Est(h, R_t), W, m


# =============================================================================
# eq:sinr, term by term
# =============================================================================

def test_cds_sums_coherently_across_aps(toy) -> None:
    """
    CDS is `|sum_at h^H w|^2`, so two identical APs give a **fourfold** power
    gain, not double. This is the coherent-combining property the whole
    cell-free argument depends on.
    """
    topo, est, W, m = toy
    out = compute_sinr(W, est, topo, sigma_n_sq=1.0)
    assert out.p_cds[0] == pytest.approx(4.0)
    one_ap = compute_sinr({0: W[0]}, est, _Topo(2, 1, [0]), sigma_n_sq=1.0)
    assert one_ap.p_cds[0] == pytest.approx(1.0)


def test_mui_uses_the_other_users_beams(toy) -> None:
    """User 0's channel is orthogonal to user 1's beam here, so MUI is zero."""
    topo, est, W, m = toy
    out = compute_sinr(W, est, topo, sigma_n_sq=1.0)
    assert out.p_mui[0] == pytest.approx(0.0)
    assert out.p_mui[1] == pytest.approx(0.0)


def test_mui_appears_when_beams_are_not_orthogonal(toy) -> None:
    topo, est, W, m = toy
    W2 = {k: v.copy() for k, v in W.items()}
    W2[0][0, 1] = 1.0                      # user 1's beam now leaks into user 0
    W2[1][0, 1] = 1.0
    out = compute_sinr(W2, est, topo, sigma_n_sq=1.0)
    assert out.p_mui[0] == pytest.approx(4.0)


def test_s2ci_counts_the_sensing_columns(toy) -> None:
    """
    The sensing beam contributes `|sum_at h^H w_t|^2`. With `w_t = [0.5, 0]`
    on both APs and `h = [1, 0]`, that is `|0.5 + 0.5|^2 = 1`.
    """
    topo, est, W, m = toy
    out = compute_sinr(W, est, topo, sigma_n_sq=1.0)
    assert out.p_s2ci[0] == pytest.approx(1.0)
    assert out.p_s2ci[1] == pytest.approx(0.0)


def test_csi_error_adds_in_power_not_amplitude(toy) -> None:
    """
    The CSI-error term is `sum_at tr(W_at^H R~_au W_at)`: the AP sum is
    **outside** the quadratic form, so error powers add.

    With `R~ = c I` on both APs the term is `c ||W_0||_F^2 + c ||W_1||_F^2`.
    Getting this wrong (summing amplitudes, or collapsing to one AP) is exactly
    the bug class the ADMM consensus vector was designed around.
    """
    topo, est, W, m = toy
    c = 0.3
    est2 = _Est(est.h_hat, {k: c * np.eye(m, dtype=complex) for k in est.h_hat})
    out = compute_sinr(W, est2, topo, sigma_n_sq=1.0)
    want = c * (np.linalg.norm(W[0]) ** 2 + np.linalg.norm(W[1]) ** 2)
    assert out.p_csi_error[0] == pytest.approx(want)


def test_perfect_csi_removes_the_error_term(toy) -> None:
    topo, est, W, m = toy
    out = compute_sinr(W, est, topo, sigma_n_sq=1.0)
    assert np.allclose(out.p_csi_error, 0.0)


def test_sinr_reconstructs_from_its_five_components(toy) -> None:
    """`SINR = CDS / (MUI + S2CI + CSI + noise)` exactly, for every user."""
    topo, est, W, m = toy
    sigma = 0.7
    est2 = _Est(est.h_hat, {k: 0.2 * np.eye(m, dtype=complex) for k in est.h_hat})
    out = compute_sinr(W, est2, topo, sigma_n_sq=sigma)
    want = out.p_cds / (out.p_mui + out.p_s2ci + out.p_csi_error + out.p_noise)
    np.testing.assert_allclose(out.sinr_per_user, want, rtol=1e-12)
    assert out.p_noise == pytest.approx(sigma)


def test_single_user_no_clutter_closed_form() -> None:
    """
    One AP, one user, no sensing beam, perfect CSI: `SINR = |h^H w|^2 / sigma^2`.
    """
    m = 4
    h = np.arange(1, m + 1).astype(complex)
    w = np.ones((m, 1), dtype=complex)
    topo = _Topo(1, 0, [0])
    est = _Est({(0, 0): h}, {(0, 0): np.zeros((m, m), dtype=complex)})
    out = compute_sinr({0: w}, est, topo, sigma_n_sq=2.0)
    assert out.sinr_per_user[0] == pytest.approx(abs(h.conj() @ w[:, 0]) ** 2 / 2.0)


def test_zero_precoder_gives_zero_sinr(toy) -> None:
    topo, est, W, m = toy
    out = compute_sinr({k: np.zeros_like(v) for k, v in W.items()}, est, topo, 1.0)
    assert np.allclose(out.sinr_per_user, 0.0)


def test_missing_ap_is_treated_as_inactive(toy) -> None:
    """An AP absent from `W_tx` behaves as `W = 0`, per the docstring."""
    topo, est, W, m = toy
    partial = compute_sinr({0: W[0]}, est, topo, sigma_n_sq=1.0)
    single = compute_sinr({0: W[0]}, est, _Topo(2, 1, [0]), sigma_n_sq=1.0)
    np.testing.assert_allclose(partial.sinr_per_user, single.sinr_per_user)


def test_no_transmit_aps_raises(toy) -> None:
    topo, est, W, m = toy
    with pytest.raises(ValueError, match="no transmit APs"):
        compute_sinr({}, est, topo, sigma_n_sq=1.0)


# =============================================================================
# Rate helpers
# =============================================================================

def test_rate_round_trip() -> None:
    s = np.array([0.0, 1.0, 10.0, 100.0])
    np.testing.assert_allclose(rate_to_sinr(sinr_to_rate(s)), s, rtol=1e-12)
    np.testing.assert_allclose(sinr_to_rate(np.array([1.0])), [1.0])


# =============================================================================
# Aggregation conventions
# =============================================================================

def _metrics(sinr_lin):
    n = len(sinr_lin)
    return SINRMetrics(sinr_per_user=np.asarray(sinr_lin, float),
                       p_cds=np.zeros(n), p_mui=np.zeros(n), p_s2ci=np.zeros(n),
                       p_csi_error=np.zeros(n), p_noise=1.0)


def test_mean_sinr_is_db_of_mean_not_mean_of_db() -> None:
    """
    The figure READMEs state `mean_sinr_db` is "dB of the per-trial mean over
    users". Pinned here, because the two conventions differ by 7 dB on a
    0 dB / 20 dB pair and the choice silently changes Fig. 2's mean-metric
    variant.
    """
    st = aggregate_sinr([_metrics([1.0, 100.0])])
    assert st.mean_sinr_per_trial[0] == pytest.approx(50.5)
    assert 10 * np.log10(st.mean_sinr_per_trial[0]) == pytest.approx(
        10 * np.log10(50.5))
    assert 10 * np.log10(st.mean_sinr_per_trial[0]) != pytest.approx(10.0)


def test_min_sinr_is_the_worst_user_per_trial() -> None:
    st = aggregate_sinr([_metrics([1.0, 100.0]), _metrics([10.0, 10.0])])
    np.testing.assert_allclose(st.min_sinr_per_trial, [1.0, 10.0])
    np.testing.assert_allclose(st.min_sinr_per_trial_db, [0.0, 10.0])


def test_db_conversion_uses_the_shared_floor() -> None:
    """Zero SINR maps to the -300 dB floor rather than `-inf`."""
    st = aggregate_sinr([_metrics([0.0, 1.0])])
    assert np.isfinite(st.sinr_per_trial_per_user_db).all()
    assert st.sinr_per_trial_per_user_db.min() < -100.0


def test_flat_pool_and_matrix_are_consistent() -> None:
    st = aggregate_sinr([_metrics([1.0, 100.0]), _metrics([10.0, 10.0])])
    assert st.sinr_per_trial_per_user_db.shape == (2, 2)
    np.testing.assert_allclose(st.all_sinr_db_flat,
                               st.sinr_per_trial_per_user_db.ravel())
    assert st.n_trials == 2 and st.n_users == 2


def test_ergodic_sinr_averages_over_trials() -> None:
    st = aggregate_sinr([_metrics([1.0, 3.0]), _metrics([3.0, 5.0])])
    np.testing.assert_allclose(st.ergodic_sinr_per_user, [2.0, 4.0])


def test_sum_rate_is_the_shannon_sum() -> None:
    st = aggregate_sinr([_metrics([1.0, 3.0])])
    assert st.sum_rate_per_trial[0] == pytest.approx(np.log2(2) + np.log2(4))
