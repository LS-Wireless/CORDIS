"""
tests/test_rician.py
====================
Spatially correlated Rician channel (`cordis/channel/rician.py`).

Paper reference: §II-A, `eq:comm-channel-covariance`. The contract these tests
lock is the one the manuscript states explicitly:

    h_au   = sqrt(beta_au) (h^LoS + h^NLoS)
    R_au   = beta_au (eta^LoS a_au a_au^H + eta^NLoS C~_au)
    ||a_au||^2 = M_t,  tr(C~_au) = M_t,  tr(R_au) = beta_au M_t

Statistical tests use a fixed seed and a tolerance sized from the Monte Carlo
count, never an arbitrary epsilon.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from cordis.channel.pathloss import compute_large_scale_fading
from cordis.channel.rician import (
    ChannelStatistics, compute_channel_statistics, compute_spatial_correlation,
    generate_channel_realization, generate_shared_subspace, los_nlos_fractions,
)
from cordis.channel.topology import generate_topology
from cordis.utils.io_utils import child_rng, make_rng


# =============================================================================
# Helpers
# =============================================================================

@pytest.fixture
def stats_bundle(tiny_cfg):
    """(cfg, topo, lsf, stats) on the tiny config with a fixed seed."""
    cfg = copy.deepcopy(tiny_cfg)
    r = make_rng(3141)
    topo = generate_topology(cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
    stats = compute_channel_statistics(topo, cfg, lsf, child_rng(r))
    return cfg, topo, lsf, stats


# =============================================================================
# K-factor to power fractions
# =============================================================================

@pytest.mark.parametrize("k_db", [-20.0, -3.0, 0.0, 6.0, 9.0, 20.0])
def test_los_nlos_fractions_sum_to_one(k_db: float) -> None:
    """eta^LoS + eta^NLoS = 1 (paper §II-A)."""
    a, b = los_nlos_fractions(k_db)
    assert a + b == pytest.approx(1.0)
    assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0


def test_los_fraction_matches_k_over_one_plus_k() -> None:
    """eta^LoS = K/(1+K) with K in linear scale."""
    for k_db in (0.0, 9.0, 15.0):
        k_lin = 10.0 ** (k_db / 10.0)
        assert los_nlos_fractions(k_db)[0] == pytest.approx(k_lin / (1 + k_lin))


def test_very_negative_k_is_pure_rayleigh() -> None:
    """The sentinel used for NLoS links (K = -200 dB) gives eta^LoS = 0."""
    assert los_nlos_fractions(-200.0) == (0.0, 1.0)


def test_los_fraction_is_monotone_in_k() -> None:
    ks = np.linspace(-10, 25, 40)
    etas = [los_nlos_fractions(k)[0] for k in ks]
    assert all(b >= a for a, b in zip(etas, etas[1:]))


# =============================================================================
# Spatial correlation matrix C_au
# =============================================================================

@pytest.mark.parametrize("m", [4, 8])
def test_spatial_correlation_is_hermitian_psd_with_trace_m(m: int) -> None:
    """
    C_au is Hermitian, PSD, and normalized to tr(C) = M_t.

    The trace convention is what makes `tr(R_au) = beta_au M_t` come out, so it
    is load-bearing for the paper's power bookkeeping.
    """
    lam = 3e8 / 3.5e9
    C = compute_spatial_correlation("UCA", m, 30.0, 90.0, 22.5, 7.0,
                                    lam, 0.5, 2000, make_rng(11))
    assert C.shape == (m, m)
    np.testing.assert_allclose(C, C.conj().T, atol=1e-12)
    assert np.trace(C).real == pytest.approx(m, rel=1e-9)
    assert np.linalg.eigvalsh(C).min() >= -1e-9 * m


def test_narrower_spread_gives_a_more_correlated_channel() -> None:
    """
    A narrow PAS concentrates the eigenvalues; a wide PAS spreads them.

    This is the mechanism behind the locally-low-rank story of §VI-F, so the
    direction matters even though the absolute widths are a separate question
    (F-02-02).
    """
    lam, m = 3e8 / 3.5e9, 16
    def eff_rank(std):
        C = compute_spatial_correlation("UCA", m, 0.0, 90.0, std, std,
                                        lam, 0.5, 4000, make_rng(5))
        ev = np.clip(np.linalg.eigvalsh(C).real, 1e-12, None)
        p = ev / ev.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    assert eff_rank(2.0) < eff_rank(45.0)


def test_spatial_correlation_converges_with_sample_count() -> None:
    """
    `n_spatial_samples` controls a Monte Carlo integral; 2000 (the shipped
    value) is close to the 32k reference.
    """
    lam, m = 3e8 / 3.5e9, 8
    ref = compute_spatial_correlation("UCA", m, 15.0, 90.0, 22.5, 7.0,
                                      lam, 0.5, 32000, make_rng(2))
    err = {}
    for n in (250, 2000):
        C = compute_spatial_correlation("UCA", m, 15.0, 90.0, 22.5, 7.0,
                                        lam, 0.5, n, make_rng(2))
        err[n] = np.linalg.norm(C - ref) / np.linalg.norm(ref)
    assert err[2000] < err[250]
    assert err[2000] < 0.15, f"n=2000 relative error {err[2000]:.3f}"


def test_spatial_correlation_of_a_single_antenna_is_scalar_one() -> None:
    C = compute_spatial_correlation("UCA", 1, 0.0, 90.0, 10.0, 5.0,
                                    3e8 / 3.5e9, 0.5, 100, make_rng(1))
    np.testing.assert_allclose(C, np.ones((1, 1)), atol=1e-12)


# =============================================================================
# Shared scattering subspace
# =============================================================================

def test_shared_subspace_has_orthonormal_columns() -> None:
    B = generate_shared_subspace(make_rng(9), 8, 3)
    assert B.shape == (8, 3)
    np.testing.assert_allclose(B.conj().T @ B, np.eye(3), atol=1e-10)


def test_shared_subspace_rank_zero_is_empty() -> None:
    """`shared_scatter_rank = 0` disables the inter-user correlation entirely."""
    B = generate_shared_subspace(make_rng(9), 8, 0)
    assert B.shape == (8, 0)


def test_shipped_config_disables_inter_user_correlation(default_cfg) -> None:
    """
    Pins F-03-02: the shipped config sets `shared_scatter_rank = 0`, so the
    shared scattering subspace `B_a` of `eq:comm-channel-covariance` is empty
    and every cross-user covariance term is exactly zero.
    """
    assert default_cfg.channel.shared_scatter_rank == 0


# =============================================================================
# ChannelStatistics
# =============================================================================

def test_statistics_cover_every_tx_ap_ue_pair(stats_bundle) -> None:
    cfg, topo, _, st = stats_bundle
    expected = {(ap.idx, ue.idx) for ap in topo.tx_aps for ue in topo.ues}
    for d in (st.R, st.C, st.R_tilde_C, st.a, st.eta_los, st.eta_nlos):
        assert set(d) == expected
    assert set(st.B) == {ap.idx for ap in topo.tx_aps}


def test_steering_vectors_carry_the_paper_normalization(stats_bundle) -> None:
    """`||a_au||^2 = M_t`, the convention stated in §II-A."""
    cfg, _, _, st = stats_bundle
    m = cfg.topology.n_ant
    for a in st.a.values():
        assert float(np.linalg.norm(a) ** 2) == pytest.approx(m)


def test_correlation_matrices_carry_the_paper_traces(stats_bundle) -> None:
    """`tr(C_au) = tr(C~_au) = M_t` and `tr(R_au) = beta_au M_t`."""
    cfg, _, lsf, st = stats_bundle
    m = cfg.topology.n_ant
    for key, R in st.R.items():
        assert np.trace(st.C[key]).real == pytest.approx(m, rel=1e-9)
        assert np.trace(st.R_tilde_C[key]).real == pytest.approx(m, rel=1e-9)
        assert np.trace(R).real == pytest.approx(lsf.beta_lin[key] * m, rel=1e-9)


def test_R_is_hermitian_and_psd(stats_bundle) -> None:
    _, _, _, st = stats_bundle
    for R in st.R.values():
        np.testing.assert_allclose(R, R.conj().T, atol=1e-14 * np.trace(R).real)
        assert np.linalg.eigvalsh(R).min() >= -1e-9 * np.trace(R).real


def test_R_is_the_documented_combination(stats_bundle) -> None:
    """`R_au = beta_au (eta^LoS a a^H + eta^NLoS C~_au)` term for term."""
    _, _, lsf, st = stats_bundle
    for key, R in st.R.items():
        want = lsf.beta_lin[key] * (
            st.eta_los[key] * np.outer(st.a[key], st.a[key].conj())
            + st.eta_nlos[key] * st.R_tilde_C[key])
        np.testing.assert_allclose(R, want, atol=1e-12 * np.trace(R).real)


def test_nlos_links_are_pure_rayleigh(stats_bundle) -> None:
    """
    Pins F-03-01: an NLoS link is assigned K = -200 dB, i.e. `eta^LoS = 0`.

    Standard practice, but §VI-A describes the channel as Rician with a mean
    K-factor of 9 dB, and at the deployed geometry ~97 % of links take this
    branch.
    """
    _, _, lsf, st = stats_bundle
    for key in st.R:
        if not lsf.los_state[key]:
            assert st.eta_los[key] == 0.0
            assert st.eta_nlos[key] == 1.0


@pytest.mark.slow
def test_most_links_are_rayleigh_at_the_deployed_geometry(default_cfg) -> None:
    """
    Quantifies F-03-01 on the paper's own configuration: the UMi LoS
    probability at a 650 m AP circle leaves under 10 % of links Rician.
    """
    r = make_rng(4242)
    rayleigh = []
    for _ in range(12):
        topo = generate_topology(default_cfg, child_rng(r))
        lsf = compute_large_scale_fading(topo, default_cfg, child_rng(r))
        st = compute_channel_statistics(topo, default_cfg, lsf, child_rng(r))
        rayleigh += [st.eta_los[k] == 0.0 for k in st.eta_los]
    assert np.mean(rayleigh) > 0.90


# =============================================================================
# Channel realization
# =============================================================================

def test_realization_shapes_and_dtypes(stats_bundle) -> None:
    cfg, topo, lsf, st = stats_bundle
    real = generate_channel_realization(topo, cfg, lsf, st, make_rng(1))
    m = cfg.topology.n_ant
    for key, h in real.h.items():
        assert h.shape == (m,) and h.dtype == np.complex128
    for a_idx, H in real.H_mat.items():
        assert H.shape == (topo.n_ue, m)


def test_H_mat_rows_are_the_per_user_channels(stats_bundle) -> None:
    """`H_a = [h_a1; ...; h_a,Nue]`, the convention `split_opt` relies on."""
    cfg, topo, lsf, st = stats_bundle
    real = generate_channel_realization(topo, cfg, lsf, st, make_rng(1))
    for ap in topo.tx_aps:
        for i, ue in enumerate(topo.ues):
            np.testing.assert_array_equal(real.H_mat[ap.idx][i], real.h[(ap.idx, ue.idx)])


def test_realization_is_reproducible(stats_bundle) -> None:
    cfg, topo, lsf, st = stats_bundle
    a = generate_channel_realization(topo, cfg, lsf, st, make_rng(77))
    b = generate_channel_realization(topo, cfg, lsf, st, make_rng(77))
    for key in a.h:
        np.testing.assert_array_equal(a.h[key], b.h[key])


def test_realization_second_order_statistics_match_R(stats_bundle) -> None:
    """
    `E{h h^H} -> R_au` over many draws.

    This is the single most important property in the module: the estimator is
    built from `R`, so if the data do not have covariance `R` the MMSE
    estimator is mismatched and every downstream CSI-error term is wrong.
    Tolerance is set from the Monte Carlo count (n = 4000 gives ~1/sqrt(n) ~ 1.6 %).
    """
    cfg, topo, lsf, st = stats_bundle
    key = (topo.tx_aps[0].idx, topo.ues[0].idx)
    n = 4000
    r = make_rng(1234)
    acc = np.zeros((cfg.topology.n_ant,) * 2, dtype=complex)
    for _ in range(n):
        h = generate_channel_realization(topo, cfg, lsf, st, child_rng(r)).h[key]
        acc += np.outer(h, h.conj())
    emp = acc / n
    R = st.R[key]
    assert np.linalg.norm(emp - R) / np.linalg.norm(R) < 0.06


def test_realization_is_zero_mean(stats_bundle) -> None:
    """The random LoS phase makes `E{h} = 0`, as §II-A specifies."""
    cfg, topo, lsf, st = stats_bundle
    key = (topo.tx_aps[0].idx, topo.ues[0].idx)
    n = 2000
    r = make_rng(99)
    acc = np.zeros(cfg.topology.n_ant, dtype=complex)
    for _ in range(n):
        acc += generate_channel_realization(topo, cfg, lsf, st, child_rng(r)).h[key]
    scale = np.sqrt(np.trace(st.R[key]).real)
    assert np.linalg.norm(acc / n) / scale < 0.06


@pytest.mark.xfail(strict=True,
                   reason="F-03-03: with shared_scatter_rank > 0 the realization "
                          "uses raw C and an unnormalized B s term, so E{h h^H} "
                          "no longer equals R")
def test_realization_matches_R_with_shared_scattering(tiny_cfg) -> None:
    """
    The `E{h h^H} = R` contract must hold for **every** config, not only the
    shipped `shared_scatter_rank = 0`.

    `compute_channel_statistics` builds `R` from the trace-renormalized
    `C~ = C + B Sigma B^H`; `generate_channel_realization` draws from raw `C`
    plus a separately scaled `B s`, and skips the renormalization. Measured
    excess power at r = 2: about 11 %.
    """
    cfg = copy.deepcopy(tiny_cfg)
    cfg.channel.shared_scatter_rank = 2
    cfg.channel.shared_scatter_power = 0.3
    r = make_rng(77)
    topo = generate_topology(cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
    st = compute_channel_statistics(topo, cfg, lsf, child_rng(r))
    key = (topo.tx_aps[0].idx, topo.ues[0].idx)
    n = 4000
    acc = np.zeros((cfg.topology.n_ant,) * 2, dtype=complex)
    for _ in range(n):
        h = generate_channel_realization(topo, cfg, lsf, st, child_rng(r)).h[key]
        acc += np.outer(h, h.conj())
    R = st.R[key]
    assert np.linalg.norm(acc / n - R) / np.linalg.norm(R) < 0.06


def test_channels_are_independent_across_links(stats_bundle) -> None:
    """Different (AP, UE) pairs draw independent small-scale fading."""
    cfg, topo, lsf, st = stats_bundle
    k1 = (topo.tx_aps[0].idx, topo.ues[0].idx)
    k2 = (topo.tx_aps[0].idx, topo.ues[1].idx)
    r = make_rng(31)
    n = 1500
    acc = np.zeros((cfg.topology.n_ant,) * 2, dtype=complex)
    for _ in range(n):
        real = generate_channel_realization(topo, cfg, lsf, st, child_rng(r))
        acc += np.outer(real.h[k1], real.h[k2].conj())
    scale = np.sqrt(np.trace(st.R[k1]).real * np.trace(st.R[k2]).real)
    assert np.linalg.norm(acc / n) / scale < 0.10
