"""
tests/test_sensing_channel.py
=============================
Sensing channel, clutter model, and the STAP metric
(`cordis/channel/sensing_channel.py`, `cordis/metrics/scnr.py`).

Paper reference: §II-B (Proposition 1), §III-B (Lemma 1, Proposition 3),
`eq:clutter-channel`, `eq:expedted-scnr`, `eq:admm-linear-objective`, and
Appendices A, C, D.

Proposition 3 is the headline sensing result, so it is tested factor by factor:
each of `sigma_RCS^2`, `T`, the per-transmit-AP sum, the bistatic path gain, the
transmit beam gain, and the receive whitening term is exercised separately.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from cordis.channel.pathloss import compute_large_scale_fading, noise_power_watts
from cordis.channel.sensing_assignment import assign_sensing
from cordis.channel.sensing_channel import (
    compute_clutter_noise_covariance, compute_clutter_spatial_correlation,
    compute_sensing_statistics,
)
from cordis.channel.topology import generate_topology
from cordis.metrics.scnr import compute_scnr, compute_sensing_surrogate
from cordis.utils.io_utils import child_rng, make_rng


# =============================================================================
# Fixtures
# =============================================================================

def _build(cfg, seed: int = 606):
    """(topo, lsf, sensing_stats, association, W_tx, sigma_n_sq, Pmax)."""
    r = make_rng(seed)
    topo = generate_topology(cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
    ss = compute_sensing_statistics(topo, cfg, lsf, child_rng(r))
    assoc = assign_sensing(topo, cfg, lsf)
    sigma_n = noise_power_watts(cfg.frequency.bandwidth_hz,
                                cfg.channel.noise_figure_db,
                                cfg.channel.noise_temp_k)
    pmax = 10.0 ** (cfg.channel.snr_db / 10.0) * sigma_n
    m = cfg.topology.n_ant
    d = cfg.topology.n_ue + cfg.topology.n_targets
    rw = make_rng(99)
    W = {}
    for ap in topo.tx_aps:
        X = (rw.standard_normal((m, d)) + 1j * rw.standard_normal((m, d))) / np.sqrt(2)
        W[ap.idx] = X * np.sqrt(pmax) / np.linalg.norm(X)
    return topo, lsf, ss, assoc, W, sigma_n, pmax


@pytest.fixture
def sensing(tiny_cfg):
    return (tiny_cfg,) + _build(tiny_cfg)


# =============================================================================
# Scalar parameters (Table III)
# =============================================================================

def test_sigma_rcs_matches_table_iii(default_cfg) -> None:
    """`sigma_RCS^2 = 10^(-3/10) = 0.501`, which Table III rounds to 0.5."""
    *_, ss, _, _, _, _ = _build(default_cfg)
    assert ss.sigma_rcs_sq == pytest.approx(0.5, rel=0.01)


def test_clutter_power_follows_the_cnr(default_cfg) -> None:
    """`sigma_clt^2 = 10^(CNR/10) sigma_n^2` when `sigma_clt` is left null."""
    cfg = copy.deepcopy(default_cfg)
    assert cfg.sensing.sigma_clt is None
    sigma_n = noise_power_watts(cfg.frequency.bandwidth_hz,
                                cfg.channel.noise_figure_db,
                                cfg.channel.noise_temp_k)
    for cnr in (-20.0, -10.0, 0.0, 10.0):
        cfg.sensing.clutter_cnr_db = cnr
        *_, ss, _, _, _, _ = _build(cfg)
        assert ss.sigma_clt_sq == pytest.approx(10.0 ** (cnr / 10.0) * sigma_n)


def test_explicit_sigma_clt_overrides_the_cnr(tiny_cfg) -> None:
    """A positive `sigma_clt` bypasses the CNR formula, as documented."""
    cfg = copy.deepcopy(tiny_cfg)
    cfg.sensing.sigma_clt = 0.25
    *_, ss, _, _, _, _ = _build(cfg)
    assert ss.sigma_clt_sq == pytest.approx(0.25 ** 2)


# =============================================================================
# Conventions: Proposition 1 quantities
# =============================================================================

def test_sensing_steering_vectors_use_the_paper_normalization(sensing) -> None:
    """`||a_at||^2 = M_t` and `||a_ar||^2 = M_r`, as §II states."""
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    m = cfg.topology.n_ant
    for v in list(ss.a_tx.values()) + list(ss.a_rx.values()):
        assert float(np.linalg.norm(v) ** 2) == pytest.approx(m)


def test_clutter_covariances_are_hermitian_psd_with_trace_m(sensing) -> None:
    """`tr(C_at) = M_t`, `tr(C_ar) = M_r` (Prop. 1 normalization)."""
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    m = cfg.topology.n_ant
    for C in list(ss.C_tx.values()) + list(ss.C_rx.values()):
        np.testing.assert_allclose(C, C.conj().T, atol=1e-12)
        assert np.trace(C).real == pytest.approx(m, rel=1e-9)
        assert np.linalg.eigvalsh(C).min() >= -1e-9 * m


def test_proposition_1_trace_identity(sensing) -> None:
    """
    Appendix A: `tr(R^sens) = (s_t beta^tgt sigma_RCS^2 + sigma_clt^2) M_r M_t`.

    The code never forms `R^sens` explicitly, so the identity is reconstructed
    from the pieces it does store, using
    `tr(A_at^T kron A_ar) = ||a_at||^2 ||a_ar||^2 = M_t M_r`
    and `tr(C_at^T kron C_ar) = M_t M_r`.
    """
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    m = cfg.topology.n_ant
    at, ar, tg = next(iter(ss.beta_bistatic))
    a_at, a_ar = ss.a_tx[(at, tg)], ss.a_rx[(ar, tg)]
    s_t = float(ss.los_state_tg[(at, tg)])
    beta = ss.beta_bistatic[(at, ar, tg)]
    tr_tgt = s_t * beta * ss.sigma_rcs_sq * (
        np.linalg.norm(a_at) ** 2 * np.linalg.norm(a_ar) ** 2)
    tr_clt = ss.sigma_clt_sq * (np.trace(ss.C_tx[at]).real
                                * np.trace(ss.C_rx[ar]).real)
    want = (s_t * beta * ss.sigma_rcs_sq + ss.sigma_clt_sq) * m * m
    assert tr_tgt + tr_clt == pytest.approx(want, rel=1e-9)


def test_los_model_always_forces_s_t_to_one(default_cfg) -> None:
    """`sensing.los_model = "always"` sets `s_t = 1` on every bistatic link."""
    assert default_cfg.sensing.los_model == "always"
    *_, ss, _, _, _, _ = _build(default_cfg)
    assert all(ss.los_state_tg.values())


# =============================================================================
# Clutter placement
# =============================================================================

def test_clutter_pas_is_hermitian_psd_with_trace_m() -> None:
    C = compute_clutter_spatial_correlation(
        "UCA", 8, 30.0, 90.0, 15.0, 3e8 / 3.5e9, 0.5, 2000, make_rng(3))
    np.testing.assert_allclose(C, C.conj().T, atol=1e-12)
    assert np.trace(C).real == pytest.approx(8, rel=1e-9)
    assert np.linalg.eigvalsh(C).min() >= -1e-9 * 8


@pytest.mark.parametrize("strategy", ["target_centroid", "offset", "uniform", "random"])
def test_every_clutter_strategy_produces_a_valid_covariance(tiny_cfg, strategy) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    cfg.sensing.clutter_center_strategy = strategy
    *_, ss, _, _, _, _ = _build(cfg)
    m = cfg.topology.n_ant
    for C in ss.C_tx.values():
        assert np.trace(C).real == pytest.approx(m, rel=1e-9)
        assert np.linalg.eigvalsh(C).min() >= -1e-9 * m


def test_unknown_clutter_strategy_raises(tiny_cfg) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    cfg.sensing.clutter_center_strategy = "sideways"
    with pytest.raises(ValueError, match="clutter_center_strategy"):
        _build(cfg)


def test_offset_moves_the_clutter_away_from_the_target(tiny_cfg) -> None:
    """
    `offset` steers the clutter PAS `clutter_offset_az_deg` away from the
    target direction; `target_centroid` leaves it on the target.

    Measured through the covariance itself: the quadratic form
    `a_at^H C_at a_at` is the clutter power radiated toward the target, and it
    must be larger under `target_centroid`.
    """
    powers = {}
    for strategy in ("target_centroid", "offset"):
        cfg = copy.deepcopy(tiny_cfg)
        cfg.sensing.clutter_center_strategy = strategy
        topo, lsf, ss, assoc, W, sigma_n, pmax = _build(cfg)
        vals = []
        for ap in topo.tx_aps:
            a_at = ss.a_tx[(ap.idx, 0)]
            vals.append(float(np.real(a_at.conj() @ ss.C_tx[ap.idx] @ a_at)))
        powers[strategy] = float(np.mean(vals))
    assert powers["target_centroid"] > powers["offset"], powers


# =============================================================================
# Lemma 1 / R_g  (clutter-plus-noise covariance)
# =============================================================================

def test_R_g_is_hermitian_positive_definite(sensing) -> None:
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    m = cfg.topology.n_ant
    for ap_r in topo.rx_aps:
        R_g = compute_clutter_noise_covariance(ap_r, W, ss, sigma_n, m)
        np.testing.assert_allclose(R_g, R_g.conj().T,
                                   atol=1e-12 * np.trace(R_g).real)
        assert np.linalg.eigvalsh(R_g).min() > 0


def test_R_g_reduces_to_the_noise_floor_with_no_transmission(sensing) -> None:
    """`W = 0` gives `R_g = sigma_n^2 I`, the noise-only limit of Lemma 1."""
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    m = cfg.topology.n_ant
    zero = {k: np.zeros_like(v) for k, v in W.items()}
    R_g = compute_clutter_noise_covariance(topo.rx_aps[0], zero, ss, sigma_n, m)
    np.testing.assert_allclose(R_g, sigma_n * np.eye(m), atol=1e-30)


def test_R_g_matches_the_proposition_3_expression(sensing) -> None:
    """
    `R_g = sigma_clt^2 sum_at tr(W_at^H C_at W_at) C_ar + sigma_n^2 I`,
    recomputed independently from the stored pieces.
    """
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    m = cfg.topology.n_ant
    ap_r = topo.rx_aps[0]
    total = sum(float(np.real(np.trace(W[a].conj().T @ ss.C_tx[a] @ W[a])))
                for a in W)
    want = ss.sigma_clt_sq * total * ss.C_rx[ap_r.idx] + sigma_n * np.eye(m)
    got = compute_clutter_noise_covariance(ap_r, W, ss, sigma_n, m)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_clutter_raises_R_g_above_the_noise_floor(sensing) -> None:
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    m = cfg.topology.n_ant
    R_g = compute_clutter_noise_covariance(topo.rx_aps[0], W, ss, sigma_n, m)
    assert np.trace(R_g).real > m * sigma_n


# =============================================================================
# Proposition 3, factor by factor
# =============================================================================

def test_scnr_matches_an_independent_recomputation(sensing) -> None:
    """
    Recompute `eq:expedted-scnr` from scratch and compare:

        SCNR = sigma_RCS^2 T sum_at beta^tgt ||a_at^H W_at||^2
               (a_ar^H R_g^-1 a_ar)
    """
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    m = cfg.topology.n_ant
    got = compute_scnr(topo, cfg, W, ss, assoc, sigma_n)
    for (ar, tg), value in got.scnr_per_pair.items():
        R_g = compute_clutter_noise_covariance(topo.aps[ar], W, ss, sigma_n, m)
        a_ar = ss.a_rx[(ar, tg)]
        spatial = float(np.real(a_ar.conj() @ np.linalg.solve(R_g, a_ar)))
        tx = 0.0
        for at in assoc.tx_aps_for(tg):
            if at not in W or not ss.los_state_tg.get((at, tg), False):
                continue
            a_at = ss.a_tx[(at, tg)]
            tx += (ss.beta_bistatic[(at, ar, tg)]
                   * float(np.linalg.norm(a_at.conj() @ W[at]) ** 2))
        want = ss.sigma_rcs_sq * cfg.sensing.n_snapshots * tx * spatial
        assert value == pytest.approx(want, rel=1e-9)


def test_scnr_is_linear_in_the_snapshot_count(sensing) -> None:
    """The `T` factor of Prop. 3: doubling the snapshots doubles the SCNR."""
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    base = compute_scnr(topo, cfg, W, ss, assoc, sigma_n)
    c2 = copy.deepcopy(cfg)
    c2.sensing.n_snapshots = 2 * cfg.sensing.n_snapshots
    doubled = compute_scnr(topo, c2, W, ss, assoc, sigma_n)
    for key, v in base.scnr_per_pair.items():
        assert doubled.scnr_per_pair[key] == pytest.approx(2 * v, rel=1e-12)


def test_scnr_is_linear_in_the_rcs_variance(sensing) -> None:
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    base = compute_scnr(topo, cfg, W, ss, assoc, sigma_n)
    ss2 = copy.deepcopy(ss)
    ss2.sigma_rcs_sq = 2 * ss.sigma_rcs_sq
    doubled = compute_scnr(topo, cfg, W, ss2, assoc, sigma_n)
    for key, v in base.scnr_per_pair.items():
        assert doubled.scnr_per_pair[key] == pytest.approx(2 * v, rel=1e-12)


def test_removing_clutter_improves_scnr(sensing) -> None:
    """Clutter-free is the best case: `R_g` shrinks to the noise floor."""
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    base = compute_scnr(topo, cfg, W, ss, assoc, sigma_n)
    ss0 = copy.deepcopy(ss)
    ss0.sigma_clt_sq = 0.0
    clean = compute_scnr(topo, cfg, W, ss0, assoc, sigma_n)
    for key, v in base.scnr_per_pair.items():
        assert clean.scnr_per_pair[key] > v


def test_scnr_vanishes_without_a_los_path(sensing) -> None:
    """`s_t = 0` removes the target echo entirely (§II-B)."""
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    ss0 = copy.deepcopy(ss)
    ss0.los_state_tg = {k: False for k in ss.los_state_tg}
    blocked = compute_scnr(topo, cfg, W, ss0, assoc, sigma_n)
    assert all(v == 0.0 for v in blocked.scnr_per_pair.values())


def test_scnr_is_computed_only_for_associated_pairs(sensing) -> None:
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    got = compute_scnr(topo, cfg, W, ss, assoc, sigma_n)
    for (ar, tg) in got.scnr_per_pair:
        assert ar in assoc.rx_aps_for(tg)


@pytest.mark.parametrize("model,bw", [("constant", 0.1), ("jakes", 0.5),
                                      ("gaussian", 0.9)])
def test_doppler_correlation_is_inert(default_cfg, model: str, bw: float) -> None:
    """
    Proposition 3 claims the expectation over the data symbols block-diagonalizes
    `R_gST` and "completely nullifies the clutter's slow-time Doppler
    correlation". If that reduction is right, `rho_clt_model` and
    `rho_clt_bandwidth` cannot affect any reported number, and they do not:
    the SCNR is bit-identical across all three models.

    This confirms the derivation and simultaneously explains why the two config
    fields are dead (F-01-05).
    """
    cfg = copy.deepcopy(default_cfg)
    ref = compute_scnr(*_ref_args(cfg))
    cfg2 = copy.deepcopy(default_cfg)
    cfg2.sensing.rho_clt_model = model
    cfg2.sensing.rho_clt_bandwidth = bw
    got = compute_scnr(*_ref_args(cfg2))
    assert set(got.scnr_per_pair) == set(ref.scnr_per_pair)
    for key, v in ref.scnr_per_pair.items():
        assert got.scnr_per_pair[key] == pytest.approx(v, rel=1e-12)


def _ref_args(cfg):
    topo, lsf, ss, assoc, W, sigma_n, pmax = _build(cfg)
    return topo, cfg, W, ss, assoc, sigma_n


# =============================================================================
# Bistatic path loss (F-04-01)
# =============================================================================

def test_bistatic_path_loss_is_the_geometric_mean_not_the_round_trip(sensing) -> None:
    """
    Pins F-04-01: `beta^tgt_{a_t a_r}` is computed as
    `sqrt(beta_{a_t->tgt} beta_{tgt->a_r})`, the **geometric mean** of the two
    one-way gains, while the comment directly above it says the **product**.

    A round-trip bistatic gain is the product. The geometric mean is
    dimensionally a one-way gain, and at the deployed geometry it is about
    130 dB larger than the product, which is the difference between the
    reported SCNR distribution and one with no detections at all.
    """
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    gaps = []
    for (at, ar, tg), beta in ss.beta_bistatic.items():
        one_way_t = lsf.beta_tg_lin[at, tg]
        one_way_r = lsf.beta_tg_lin[ar, tg]
        assert beta == pytest.approx(np.sqrt(one_way_t * one_way_r), rel=1e-12)
        gaps.append(10 * np.log10(beta / (one_way_t * one_way_r)))
    assert np.mean(gaps) > 100.0, f"mean gap {np.mean(gaps):.1f} dB"


def test_bistatic_path_loss_is_symmetric_in_the_two_aps(sensing) -> None:
    """The geometric mean is symmetric, so swapping the legs changes nothing."""
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    for (at, ar, tg), beta in ss.beta_bistatic.items():
        swapped = np.sqrt(lsf.beta_tg_lin[ar, tg] * lsf.beta_tg_lin[at, tg])
        assert beta == pytest.approx(swapped, rel=1e-12)


# =============================================================================
# Clutter-aware surrogate (eq:admm-linear-objective)
# =============================================================================

def test_surrogate_matches_the_linear_objective(sensing) -> None:
    """
    `U = sum_at [ sum_t omega_bar_t beta_bar_at^t ||a_at^H W_at||^2
                  - kappa tr(W_at^H C_at W_at) ]`
    recomputed independently, with `omega_bar_t = omega_t T sigma_RCS^2`.
    """
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    kappa = cfg.algorithm.admm.kappa
    want = 0.0
    for at, W_at in W.items():
        want -= kappa * float(np.real(np.trace(W_at.conj().T @ ss.C_tx[at] @ W_at)))
        for tg in range(topo.n_targets):
            if at not in assoc.tx_aps_for(tg):
                continue
            if not ss.los_state_tg.get((at, tg), False):
                continue
            beta_bar = sum(ss.beta_bistatic.get((at, ar, tg), 0.0)
                           for ar in assoc.rx_aps_for(tg))
            omega_bar = 1.0 * cfg.sensing.n_snapshots * ss.sigma_rcs_sq
            a_at = ss.a_tx[(at, tg)]
            want += omega_bar * beta_bar * float(
                np.linalg.norm(a_at.conj() @ W_at) ** 2)
    got = compute_sensing_surrogate(topo, cfg, W, ss, assoc)
    assert got == pytest.approx(want, rel=1e-9)


def test_surrogate_is_decreasing_in_kappa(sensing) -> None:
    """Raising the clutter penalty lowers the surrogate for a fixed `W`."""
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    vals = []
    for k in (0.0, 0.08, 1.0):
        c = copy.deepcopy(cfg)
        c.algorithm.admm.kappa = k
        vals.append(compute_sensing_surrogate(topo, c, W, ss, assoc))
    assert vals[0] > vals[1] > vals[2]


def test_surrogate_reads_the_admm_kappa_not_the_split_one(sensing) -> None:
    """
    Pins a scoring asymmetry: `compute_sensing_surrogate` always uses
    `cfg.algorithm.admm.kappa`, so a CORDIS-Split solution (optimized with
    `algorithm.split.kappa = 1.0`) is scored with the ADMM `kappa = 0.08`.
    """
    cfg, topo, lsf, ss, assoc, W, sigma_n, pmax = sensing
    assert cfg.algorithm.admm.kappa != cfg.algorithm.split.kappa
    c = copy.deepcopy(cfg)
    c.algorithm.split.kappa = 99.0
    assert compute_sensing_surrogate(topo, c, W, ss, assoc) == pytest.approx(
        compute_sensing_surrogate(topo, cfg, W, ss, assoc))


@pytest.mark.xfail(strict=True,
                   reason="F-04-02: at the deployed parameters the clutter "
                          "penalty dominates the echo term by ~10 orders of "
                          "magnitude, so the surrogate is not a trade-off")
def test_surrogate_terms_are_of_comparable_magnitude(default_cfg) -> None:
    """
    §V presents `eq:admm-linear-objective` as "the aggregate priority-weighted
    target echo power **minus** a regularized penalty for the total transmit
    clutter illumination", i.e. a balance between two terms.

    Measured at the deployed parameters the echo term is about 3e-10 and the
    clutter penalty about 15, a ratio of 2e-11. Whatever the optimizer is
    maximizing, it is numerically pure clutter avoidance.
    """
    topo, lsf, ss, assoc, W, sigma_n, pmax = _build(default_cfg)
    echo_only = copy.deepcopy(default_cfg)
    echo_only.algorithm.admm.kappa = 0.0
    echo = compute_sensing_surrogate(topo, echo_only, W, ss, assoc)
    full = compute_sensing_surrogate(topo, default_cfg, W, ss, assoc)
    penalty = echo - full
    assert 1e-3 < abs(echo) / abs(penalty) < 1e3, (
        f"echo {echo:.3e} vs penalty {penalty:.3e}")
