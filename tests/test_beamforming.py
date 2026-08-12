"""
tests/test_beamforming.py
=========================
Phase I beamforming primitives (`cordis/algorithms/beamforming.py`).

Paper reference: §IV-A, `eq:split-comm-bf` (LR-MMSE) and the null-space
projector that defines NS-C.

The convention that matters throughout: the received signal at user `u` is
`y_u = sum_at h_hat_au^H x_at`, so a precoder is "aligned" with user `u` when
`h_hat_u^H w_u` is real and positive, and a sensing beam is in the null space
when `h_hat_u^H w_t = 0`. Both involve the **conjugate** transpose of the
channel, which is where the printed equations and the code part company
(F-06-01).
"""
from __future__ import annotations

import numpy as np
import pytest

from cordis.algorithms.beamforming import (
    compute_alpha_weights, global_mrt, global_zf, lr_mmse, mrt, ns_c,
    random_bf, rzf, zf,
)
from cordis.utils.io_utils import make_rng

M, K = 8, 3


@pytest.fixture
def channel():
    """`H` with rows `h_u^T`, the layout `H_hat[a]` uses."""
    r = make_rng(4242)
    return (r.standard_normal((K, M)) + 1j * r.standard_normal((K, M))) / np.sqrt(2)


# =============================================================================
# LR-MMSE (eq:split-comm-bf)
# =============================================================================

def test_lr_mmse_is_unit_norm(channel) -> None:
    """`||W||_F = 1`; the power scaling comes later from `sqrt(rho Pmax)`."""
    W = lr_mmse(channel, np.zeros((M, M), complex), alpha=1.0, epsilon=1e-2)
    assert W.shape == (M, K)
    assert np.linalg.norm(W) == pytest.approx(1.0)


def test_lr_mmse_aligns_the_desired_signal(channel) -> None:
    """
    `h_u^H w_u` is real and positive for every user.

    This is what makes `beta_hat_au` real-positive, which in turn is what makes
    `S_u(rho)` a sum of concave geometric means in P-Split. If the conjugation
    were wrong the whole convexity argument of §IV-B would fail.
    """
    W = lr_mmse(channel, np.zeros((M, M), complex), alpha=1.0, epsilon=1e-2)
    for u in range(K):
        s = channel[u].conj() @ W[:, u]
        assert s.real > 0
        assert abs(s.imag) < 1e-12 * abs(s.real) + 1e-18


def test_lr_mmse_uses_the_error_covariance(channel) -> None:
    """
    `R~_H` enters the inverted Gram, so a larger CSI error changes the
    precoder. This is the "robust" in Local Robust MMSE.
    """
    W0 = lr_mmse(channel, np.zeros((M, M), complex), 1.0, 1e-2)
    W1 = lr_mmse(channel, 5.0 * np.eye(M, dtype=complex), 1.0, 1e-2)
    assert not np.allclose(W0, W1)


def test_lr_mmse_tends_to_mrt_as_the_error_grows(channel) -> None:
    """
    With `R~_H` dominating the Gram, the inverse tends to a scaled identity and
    LR-MMSE tends to matched filtering, the classical robustness limit.
    """
    scale = 1e8 * np.trace(channel.conj().T @ channel).real
    W_big = lr_mmse(channel, scale * np.eye(M, dtype=complex), 1.0, 1e-2)
    W_mrt = mrt(channel, 1.0)
    cos = abs(np.vdot(W_big.ravel(), W_mrt.ravel())) / (
        np.linalg.norm(W_big) * np.linalg.norm(W_mrt))
    assert cos > 0.99


def test_lr_mmse_regularization_is_scale_invariant(channel) -> None:
    """
    `epsilon` is applied relative to the Gram trace, so scaling the channel
    leaves the precoder direction unchanged.
    """
    Wa = lr_mmse(channel, np.zeros((M, M), complex), 1.0, 1e-2)
    Wb = lr_mmse(1e3 * channel, np.zeros((M, M), complex), 1.0, 1e-2)
    cos = abs(np.vdot(Wa.ravel(), Wb.ravel())) / (
        np.linalg.norm(Wa) * np.linalg.norm(Wb))
    assert cos > 0.999


def test_alpha_is_a_pure_scale_before_normalization(channel) -> None:
    """`alpha` cancels under the unit-norm normalization."""
    Wa = lr_mmse(channel, np.zeros((M, M), complex), 0.1, 1e-2)
    Wb = lr_mmse(channel, np.zeros((M, M), complex), 0.9, 1e-2)
    np.testing.assert_allclose(Wa, Wb, atol=1e-12)


@pytest.mark.xfail(strict=True,
                   reason="F-06-01: eq:split-comm-bf as printed uses H^H H and "
                          "H^H, which does not align h^H w for the signal model "
                          "of §III; the code uses the conjugate convention")
def test_printed_equation_aligns_the_signal(channel) -> None:
    """
    The literal `eq:split-comm-bf`, `(H^H H + R~ + eps I)^-1 H^H alpha`, with
    `H` defined in §IV as rows `h_u^T`, should also produce a real-positive
    `h_u^H w_u`. It does not: it is conjugated relative to the signal model.
    """
    gram = channel.conj().T @ channel
    eps = 1e-2 * np.trace(gram).real / M
    W = np.linalg.solve(gram + eps * np.eye(M), channel.conj().T)
    W = W / np.linalg.norm(W)
    for u in range(K):
        s = channel[u].conj() @ W[:, u]
        assert s.real > 0 and abs(s.imag) < 1e-9 * abs(s.real) + 1e-18


# =============================================================================
# NS-C (§IV-A)
# =============================================================================

def test_ns_c_nulls_the_estimated_channels(channel) -> None:
    """
    `h_u^H w_t ~ 0` for every user: the sensing beam causes no S2CI on the
    *estimated* channel, which is the whole design intent.
    """
    a_t = {0: np.ones(M, dtype=complex)}
    W_s, lam = ns_c(channel, a_t, epsilon=1e-6)
    for u in range(K):
        leak = abs(channel[u].conj() @ W_s[:, 0]) / (
            np.linalg.norm(channel[u]) * np.linalg.norm(W_s[:, 0]))
        assert leak < 1e-5


@pytest.mark.parametrize("eps,bound", [(1e-6, 1e-5), (1e-3, 1e-2), (1e-1, 0.2)])
def test_ns_c_leakage_scales_with_the_loading_factor(channel, eps, bound) -> None:
    """
    The projector is regularized, so residual leakage is of order `epsilon`.
    The shipped `epsilon_nsc = 1e-3` therefore leaves about 0.1 % leakage,
    which the CPU power-allocation stage is designed to absorb (§IV-A).
    """
    W_s, _ = ns_c(channel, {0: np.ones(M, dtype=complex)}, epsilon=eps)
    leak = max(abs(channel[u].conj() @ W_s[:, 0])
               / (np.linalg.norm(channel[u]) * np.linalg.norm(W_s[:, 0]))
               for u in range(K))
    assert leak < bound


def test_ns_c_is_unit_norm_and_allocates_by_priority(channel) -> None:
    """
    `lambda_t` is the priority-weighted projected-norm share and sums to 1;
    `||W_s||_F = 1`.
    """
    a_t = {0: np.ones(M, dtype=complex), 1: np.arange(1, M + 1).astype(complex)}
    W_s, lam = ns_c(channel, a_t, omega={0: 1.0, 1: 3.0}, epsilon=1e-3)
    assert W_s.shape == (M, 2)
    assert np.linalg.norm(W_s) == pytest.approx(1.0)
    assert sum(lam.values()) == pytest.approx(1.0)
    assert lam[1] > lam[0], "the higher-priority target should get more power"


def test_ns_c_with_no_targets_returns_an_empty_matrix(channel) -> None:
    W_s, lam = ns_c(channel, {})
    assert W_s.shape == (M, 0) and lam == {}


def test_ns_c_degenerates_when_the_null_space_is_empty() -> None:
    """
    Pins the §VI-F mechanism. Once `N_ue >= M` the estimated channel matrix
    spans the whole space, the null space is empty, and the projected beam can
    no longer suppress S2CI.

    Measured normalized leakage: about 2e-3 at `N_ue = 3, M = 8`, about 0.08 at
    `N_ue = M = 8`, and about 0.8 at `N_ue = 10, M = 3`. That last figure is
    the locally-low-rank regime of Fig. 6, and it is why CORDIS-Split's outage
    climbs there.
    """
    r = make_rng(3)
    leaks = {}
    for (k, m) in ((3, 8), (8, 8), (10, 3)):
        H = (r.standard_normal((k, m)) + 1j * r.standard_normal((k, m)))
        W_s, _ = ns_c(H, {0: np.ones(m, dtype=complex)}, epsilon=1e-3)
        leaks[(k, m)] = max(
            abs(H[u].conj() @ W_s[:, 0])
            / (np.linalg.norm(H[u]) * np.linalg.norm(W_s[:, 0]) + 1e-30)
            for u in range(k))
    assert leaks[(3, 8)] < 0.02
    assert leaks[(10, 3)] > 0.5
    assert leaks[(3, 8)] < leaks[(8, 8)] < leaks[(10, 3)]


# =============================================================================
# Classical baselines
# =============================================================================

@pytest.mark.parametrize("fn", [
    lambda H: mrt(H, 1.0),
    lambda H: zf(H, 1.0),
    lambda H: rzf(H, 1.0),
    lambda H: lr_mmse(H, np.zeros((H.shape[1], H.shape[1]), complex), 1.0, 1e-2),
], ids=["mrt", "zf", "rzf", "lr_mmse"])
def test_every_precoder_is_unit_norm(channel, fn) -> None:
    W = fn(channel)
    assert W.shape == (M, K)
    assert np.linalg.norm(W) == pytest.approx(1.0)


def test_zf_removes_multi_user_interference(channel) -> None:
    """ZF drives `h_u^H w_k` to zero for `k != u`, its defining property."""
    W = zf(channel, 1.0)
    G = channel.conj() @ W
    off = np.abs(G - np.diag(np.diag(G)))
    assert off.max() / np.abs(np.diag(G)).min() < 1e-8


def test_mrt_is_the_conjugate_channel(channel) -> None:
    W = mrt(channel, 1.0)
    for u in range(K):
        cos = abs(np.vdot(W[:, u], channel[u])) / (
            np.linalg.norm(W[:, u]) * np.linalg.norm(channel[u]))
        assert cos == pytest.approx(1.0, abs=1e-9)


def test_rzf_sits_between_mrt_and_zf(channel) -> None:
    """Large regularization tends to MRT, small tends to ZF."""
    def cos_to(W, ref):
        return abs(np.vdot(W.ravel(), ref.ravel())) / (
            np.linalg.norm(W) * np.linalg.norm(ref))
    assert cos_to(rzf(channel, 1.0, epsilon=1e-8), zf(channel, 1.0)) > 0.99
    assert cos_to(rzf(channel, 1.0, epsilon=1e8), mrt(channel, 1.0)) > 0.99


def test_random_bf_is_reproducible_and_unit_norm() -> None:
    a = random_bf(M, K, make_rng(7))
    b = random_bf(M, K, make_rng(7))
    np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)


def test_global_precoders_and_alpha_weights_need_the_full_pipeline() -> None:
    """
    `global_zf`, `global_mrt` and `compute_alpha_weights` take
    `(est, topo)` rather than a bare channel matrix, so they are exercised
    through the Phase-I integration path in `test_split_opt.py` rather than
    here. This test only pins the signature so a refactor cannot silently
    change it.
    """
    import inspect
    assert list(inspect.signature(global_zf).parameters)[:2] == ["est", "topo"]
    assert list(inspect.signature(global_mrt).parameters)[:2] == ["est", "topo"]
    assert list(inspect.signature(
        compute_alpha_weights).parameters)[:2] == ["est", "topo"]
