"""
tests/test_pathloss.py
======================
Large-scale fading assembly (`cordis/channel/pathloss.py`).

TR-38.901 conformance of the individual formulas is checked in
`test_3gpp_conformance.py`. This file tests how they are *composed*: LoS-state
sampling, shadow-fading correlation, the β bookkeeping, and the AP↔target leg,
which behaves differently from the AP↔UE leg in ways the manuscript does not
mention.
"""
from __future__ import annotations

import numpy as np
import pytest

from cordis.channel.pathloss import (
    _correlated_shadow_fading, _shadow_std, compute_large_scale_fading,
    los_probability_uma, los_probability_umi, sample_los_state,
)
from cordis.channel.topology import generate_topology
from cordis.utils.io_utils import child_rng, make_rng


# =============================================================================
# LoS-state sampling
# =============================================================================

def test_sample_los_state_reproduces_the_probability(rng) -> None:
    """Bernoulli draws converge to `p_los` and are shaped like the input."""
    p = np.full((40, 40), 0.3)
    draws = np.array([sample_los_state(p, rng).mean() for _ in range(30)])
    assert draws.mean() == pytest.approx(0.3, abs=0.02)
    assert sample_los_state(p, rng).shape == p.shape


@pytest.mark.parametrize("p,expected", [(0.0, False), (1.0, True)])
def test_sample_los_state_at_the_extremes(rng, p: float, expected: bool) -> None:
    assert bool(sample_los_state(np.full((10, 10), p), rng).all()) is expected


def test_uma_los_probability_stays_bounded() -> None:
    """
    UMa `P_LoS` remains a probability across the full range.

    The code's `C(d)` correction term is a garbled transcription of the TR
    expression (F-02-05); it is numerically negligible (< 1e-6) for every
    distance, so the result still lies in [0, 1] and matches the `h_UT ≤ 13 m`
    branch that the docstring says is assumed. This test pins that.
    """
    d = np.geomspace(1.0, 5000.0, 400)
    p = los_probability_uma(d)
    assert np.all((p >= 0.0) & (p <= 1.0))
    base = np.minimum(18.0 / d, 1.0) * (1 - np.exp(-d / 63.0)) + np.exp(-d / 63.0)
    np.testing.assert_allclose(p, base, rtol=1e-5)


# =============================================================================
# Shadow fading
# =============================================================================

def test_shadow_std_lookup_falls_back_to_umi() -> None:
    assert _shadow_std("UMi", True) == pytest.approx(4.0)
    assert _shadow_std("UMi", False) == pytest.approx(7.82)
    assert _shadow_std("NoSuchScenario", True) == _shadow_std("UMi", True)


def test_correlated_shadow_fading_has_the_requested_variance(rng) -> None:
    """Marginal variance is σ², independent of the correlation structure."""
    pos = rng.uniform(-500, 500, size=(6, 2))
    draws = np.array([_correlated_shadow_fading(pos, 7.82, 50.0, rng)
                      for _ in range(4000)])
    assert draws.std(axis=0).mean() == pytest.approx(7.82, rel=0.06)
    assert draws.mean() == pytest.approx(0.0, abs=0.4)


def test_correlated_shadow_fading_decays_with_distance(rng) -> None:
    """
    Cov(ψ_j, ψ_k) = σ² exp(−d_jk / D_corr): a close pair correlates more
    strongly than a distant pair.
    """
    near = np.array([[0.0, 0.0], [5.0, 0.0]])
    far = np.array([[0.0, 0.0], [500.0, 0.0]])
    n = 4000
    cn = np.corrcoef(np.array([_correlated_shadow_fading(near, 8.0, 50.0, rng)
                               for _ in range(n)]).T)[0, 1]
    cf = np.corrcoef(np.array([_correlated_shadow_fading(far, 8.0, 50.0, rng)
                               for _ in range(n)]).T)[0, 1]
    assert cn > 0.8
    assert abs(cf) < 0.1


def test_correlated_shadow_fading_single_ue(rng) -> None:
    out = _correlated_shadow_fading(np.zeros((1, 2)), 4.0, 50.0, rng)
    assert out.shape == (1,)


# =============================================================================
# Composition: compute_large_scale_fading
# =============================================================================

@pytest.fixture
def lsf_and_topo(tiny_cfg):
    r = make_rng(4242)
    topo = generate_topology(tiny_cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, tiny_cfg, child_rng(r),
                                     include_targets=True)
    return lsf, topo


def test_beta_shapes_and_dtypes(lsf_and_topo, tiny_cfg) -> None:
    lsf, topo = lsf_and_topo
    shape = (topo.n_ap, topo.n_ue)
    for arr in (lsf.beta_db, lsf.beta_lin, lsf.path_loss_db, lsf.shadow_db):
        assert arr.shape == shape
    assert lsf.los_state.shape == shape and lsf.los_state.dtype == np.bool_
    assert lsf.beta_tg_db.shape == (topo.n_ap, topo.n_targets)


def test_beta_lin_is_the_linear_form_of_beta_db(lsf_and_topo) -> None:
    """`β_lin = 10^(−β_db/10)`, note the sign: larger β_db is *more* loss."""
    lsf, _ = lsf_and_topo
    np.testing.assert_allclose(lsf.beta_lin, 10.0 ** (-lsf.beta_db / 10.0))
    assert np.all(lsf.beta_lin > 0.0)
    assert np.all(lsf.beta_lin < 1.0)


def test_beta_db_is_path_loss_plus_shadowing(lsf_and_topo) -> None:
    lsf, _ = lsf_and_topo
    np.testing.assert_allclose(lsf.beta_db, lsf.path_loss_db + lsf.shadow_db)


def test_path_loss_follows_the_sampled_los_state(lsf_and_topo, tiny_cfg) -> None:
    """
    Each link's deterministic path loss is the LoS or NLoS branch according to
    its own sampled state, not a per-AP majority vote.
    """
    from cordis.channel.pathloss import path_loss_umi_los, path_loss_umi_nlos
    lsf, topo = lsf_and_topo
    d2, d3 = topo.ap_ue_distances_2d(), topo.ap_ue_distances_3d()
    fc = tiny_cfg.frequency.carrier_freq_hz
    h_ap, h_ue = tiny_cfg.topology.h_ap_m, tiny_cfg.topology.h_ue_m
    want = np.where(lsf.los_state,
                    path_loss_umi_los(d2, d3, fc, h_ap, h_ue),
                    path_loss_umi_nlos(d2, d3, fc, h_ap, h_ue))
    np.testing.assert_allclose(lsf.path_loss_db, want)


def test_large_scale_fading_is_reproducible(tiny_cfg) -> None:
    def run():
        r = make_rng(99)
        topo = generate_topology(tiny_cfg, child_rng(r))
        return compute_large_scale_fading(topo, tiny_cfg, child_rng(r))
    a, b = run(), run()
    np.testing.assert_array_equal(a.beta_db, b.beta_db)
    np.testing.assert_array_equal(a.los_state, b.los_state)


def test_targets_can_be_skipped(tiny_cfg) -> None:
    r = make_rng(5)
    topo = generate_topology(tiny_cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, tiny_cfg, child_rng(r),
                                     include_targets=False)
    assert lsf.beta_tg_db is None and lsf.beta_tg_lin is None


# =============================================================================
# The AP↔target leg: undisclosed asymmetries (F-02-03, F-02-04)
# =============================================================================

def test_target_shadow_fading_always_uses_the_los_sigma(tiny_cfg) -> None:
    """
    Pins F-02-04: the AP→target leg samples a LoS/NLoS state and uses it for
    *path loss*, but draws shadow fading with the LoS σ (4 dB) regardless.

    The UE leg is consistent (σ follows the state); the target leg is not.
    Measured against the empirical spread over many drops: if the NLoS σ were
    used for the ~97 % NLoS target links the std would sit near 7.82 dB.
    """
    r = make_rng(11)
    sf = []
    for _ in range(300):
        topo = generate_topology(tiny_cfg, child_rng(r))
        lsf = compute_large_scale_fading(topo, tiny_cfg, child_rng(r),
                                         include_targets=True)
        sf.append((lsf.beta_tg_db - _target_path_loss(topo, tiny_cfg, lsf)).ravel())
    spread = np.concatenate(sf).std()
    assert spread == pytest.approx(4.0, rel=0.15), (
        f"target shadow-fading std is {spread:.2f} dB; LoS σ is 4.0, NLoS σ is 7.82")


def _target_path_loss(topo, cfg, lsf):
    from cordis.channel.pathloss import path_loss_umi_los, path_loss_umi_nlos
    n_ap, n_tg = topo.n_ap, topo.n_targets
    d2 = np.zeros((n_ap, n_tg))
    d3 = np.zeros((n_ap, n_tg))
    for i, ap in enumerate(topo.aps):
        for j, tg in enumerate(topo.targets):
            d2[i, j] = ap.distance_2d_to(tg.pos)
            d3[i, j] = ap.distance_to(tg.pos)
    fc = cfg.frequency.carrier_freq_hz
    h_ap, h_tg = cfg.topology.h_ap_m, cfg.topology.h_tg_m
    return np.where(lsf.los_state_tg,
                    path_loss_umi_los(d2, d3, fc, h_ap, h_tg),
                    path_loss_umi_nlos(d2, d3, fc, h_ap, h_tg))


def test_target_links_are_mostly_nlos_at_the_deployment_radius(default_cfg) -> None:
    """
    Pins F-02-03: at the paper's 650 m / 1 km geometry the UMi LoS probability
    makes ~97 % of AP→target links NLoS, so `β^tgt` takes the NLoS branch,
    while `sensing.los_model = "always"` forces the sensing indicator `s_t = 1`.
    The two halves of the target model disagree about whether a LoS path exists.
    """
    r = make_rng(2026)
    frac = []
    for _ in range(40):
        topo = generate_topology(default_cfg, child_rng(r))
        lsf = compute_large_scale_fading(topo, default_cfg, child_rng(r),
                                         include_targets=True)
        frac.append(lsf.los_state_tg.mean())
    assert np.mean(frac) < 0.10, f"AP→target LoS fraction {np.mean(frac):.3f}"


def test_comm_links_are_not_forced_los_by_the_sensing_switch(default_cfg) -> None:
    """
    `sensing.los_model = "always"` must not leak into the communication links.

    It is read only in `sensing_channel.py`; the comm LoS state stays
    stochastic. Verified here because a leak would silently turn the comm
    channel fully Rician.
    """
    assert default_cfg.sensing.los_model == "always"
    r = make_rng(31)
    frac = []
    for _ in range(40):
        topo = generate_topology(default_cfg, child_rng(r))
        lsf = compute_large_scale_fading(topo, default_cfg, child_rng(r))
        frac.append(lsf.los_state.mean())
    assert 0.0 < np.mean(frac) < 0.20, (
        f"comm LoS fraction {np.mean(frac):.3f}, expected the stochastic UMi value")
