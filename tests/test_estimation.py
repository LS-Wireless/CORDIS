"""
tests/test_estimation.py
========================
Uplink pilot transmission and linear MMSE channel estimation
(`cordis/channel/estimation.py`).

Paper reference: §II-C, Proposition 2, `eq:pilot-observation`,
`eq:channel-estimate`, and Appendix B. The estimator's error covariance
`R~_{a_t u}` propagates into the SINR of `eq:sinr` and into the ADMM consensus
vector, so its correctness is load-bearing well beyond this module.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from cordis.channel.estimation import (
    compute_mmse_matrices, design_pilot_sequences, run_channel_estimation,
)
from cordis.channel.pathloss import compute_large_scale_fading, noise_power_watts
from cordis.channel.rician import (
    compute_channel_statistics, generate_channel_realization,
)
from cordis.channel.topology import generate_topology
from cordis.utils.io_utils import child_rng, make_rng


# =============================================================================
# Helpers
# =============================================================================

def _pipeline(cfg, seed: int = 707):
    r = make_rng(seed)
    topo = generate_topology(cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
    st = compute_channel_statistics(topo, cfg, lsf, child_rng(r))
    Phi, cs = design_pilot_sequences(topo.n_ue, cfg.channel.tau_p, child_rng(r))
    real = generate_channel_realization(topo, cfg, lsf, st, child_rng(r))
    est = run_channel_estimation(topo, cfg, lsf, st, real, child_rng(r), Phi, cs)
    return topo, lsf, st, Phi, cs, real, est


@pytest.fixture
def bundle(tiny_cfg):
    return (tiny_cfg,) + _pipeline(tiny_cfg)


# =============================================================================
# Pilot design (paper §II-C)
# =============================================================================

@pytest.mark.parametrize("n_ue,tau_p", [(1, 4), (3, 10), (10, 10)])
def test_orthogonal_pilots_when_tau_p_is_large_enough(n_ue: int, tau_p: int) -> None:
    """
    `phi_k^H phi_l = tau_p` for `l` sharing `k`'s pilot and 0 otherwise (§II-C).

    With `tau_p >= N_ue` every user gets its own sequence, so each
    contamination set is the singleton `{u}`.
    """
    Phi, sets = design_pilot_sequences(n_ue, tau_p, make_rng(5))
    assert Phi.shape == (n_ue, tau_p)
    G = Phi.conj() @ Phi.T
    np.testing.assert_allclose(np.diag(G).real, tau_p, rtol=1e-9)
    off = G - np.diag(np.diag(G))
    assert np.abs(off).max() < 1e-9
    assert [sorted(s) for s in sets] == [[u] for u in range(n_ue)]


def test_pilot_contamination_appears_when_tau_p_is_too_small() -> None:
    """
    `N_ue > tau_p` forces pilot reuse, and the contamination sets record it.

    §II-C: "in practical scenarios it is often the case that N_ue >= tau_p ...
    some users need to share the same pilot sequence".
    """
    n_ue, tau_p = 12, 10
    Phi, sets = design_pilot_sequences(n_ue, tau_p, make_rng(5))
    sizes = [len(s) for s in sets]
    assert max(sizes) >= 2
    # exactly `n_ue - tau_p` pilots get reused, each by one extra user, so
    # 2*(n_ue - tau_p) users sit in a set of size 2 and the rest are singletons
    assert sum(sizes) == n_ue + 2 * (n_ue - tau_p)
    for u, s in enumerate(sets):
        assert u in s
        for k in s:
            assert np.abs(Phi[u].conj() @ Phi[k]) == pytest.approx(tau_p, rel=1e-9)


def test_contamination_sets_are_symmetric() -> None:
    """`k in P_u` implies `u in P_k`: pilot sharing is a mutual relation."""
    _, sets = design_pilot_sequences(12, 10, make_rng(5))
    for u, s in enumerate(sets):
        for k in s:
            assert u in sets[k]


def test_shipped_config_has_no_pilot_contamination(default_cfg) -> None:
    """`tau_p = 10 >= N_ue = 4`, so every shipped campaign is contamination-free."""
    assert default_cfg.channel.tau_p >= default_cfg.topology.n_ue
    _, sets = design_pilot_sequences(default_cfg.topology.n_ue,
                                     default_cfg.channel.tau_p, make_rng(1))
    assert all(len(s) == 1 for s in sets)


# =============================================================================
# Proposition 2 matrices
# =============================================================================

def test_mmse_matrices_match_proposition_2(bundle) -> None:
    """
    Without contamination Prop. 2 reduces to
    `D = sqrt(P_p) tau_p R` and `Psi = P_p tau_p^2 R + sigma_n^2 tau_p I`.
    """
    cfg, topo, lsf, st, Phi, cs, real, est = bundle
    tau_p = cfg.channel.tau_p
    sigma2 = noise_power_watts(cfg.frequency.bandwidth_hz,
                               cfg.channel.noise_figure_db,
                               cfg.channel.noise_temp_k)
    pp = 10.0 ** (cfg.channel.pilot_power_db / 10.0) * sigma2
    m = cfg.topology.n_ant
    a_idx, u_idx = topo.tx_aps[0].idx, topo.ues[0].idx
    D, Psi = compute_mmse_matrices(u_idx, a_idx, st, cs, tau_p, pp, sigma2, m)
    R = st.R[(a_idx, u_idx)]
    np.testing.assert_allclose(D, np.sqrt(pp) * tau_p * R, rtol=1e-10)
    np.testing.assert_allclose(
        Psi, pp * tau_p ** 2 * R + sigma2 * tau_p * np.eye(m), rtol=1e-10)


def test_psi_is_hermitian_and_invertible(bundle) -> None:
    """`Psi` is a covariance: Hermitian, and the noise floor keeps it full rank."""
    cfg, topo, lsf, st, Phi, cs, real, est = bundle
    sigma2 = noise_power_watts(cfg.frequency.bandwidth_hz,
                               cfg.channel.noise_figure_db,
                               cfg.channel.noise_temp_k)
    pp = 10.0 ** (cfg.channel.pilot_power_db / 10.0) * sigma2
    m = cfg.topology.n_ant
    for ap in topo.tx_aps:
        for ue in topo.ues:
            _, Psi = compute_mmse_matrices(ue.idx, ap.idx, st, cs,
                                           cfg.channel.tau_p, pp, sigma2, m)
            np.testing.assert_allclose(Psi, Psi.conj().T,
                                       atol=1e-12 * np.trace(Psi).real)
            assert np.linalg.eigvalsh(Psi).min() > 0


def test_cross_user_terms_vanish_without_a_shared_subspace(bundle) -> None:
    """
    With `shared_scatter_rank = 0` the `xi B Gamma B^H` corrections of Prop. 2
    are identically zero, so `D` and `Psi` collapse to the single-user forms
    even when pilots are shared. Pins the scope of F-03-02.
    """
    cfg, topo, lsf, st, Phi, cs, real, est = bundle
    assert cfg.channel.shared_scatter_rank == 0
    for B in st.B.values():
        assert B.shape[1] == 0


# =============================================================================
# Estimator properties (Prop. 2 and Appendix B)
# =============================================================================

def test_estimate_shapes(bundle) -> None:
    cfg, topo, lsf, st, Phi, cs, real, est = bundle
    m = cfg.topology.n_ant
    for key, h in est.h_hat.items():
        assert h.shape == (m,) and h.dtype == np.complex128
        assert est.R_hat[key].shape == (m, m)
        assert est.R_tilde[key].shape == (m, m)
    for a_idx, H in est.H_hat.items():
        assert H.shape == (topo.n_ue, m)
    for a_idx, RH in est.R_tilde_H.items():
        assert RH.shape == (m, m)


def test_error_covariance_decomposition(bundle) -> None:
    """
    `R = R_hat + R_tilde` exactly, the identity §II-C states right after
    Prop. 2 and that the SINR model in `eq:sinr` depends on.
    """
    cfg, topo, lsf, st, Phi, cs, real, est = bundle
    for key, R in st.R.items():
        recon = est.R_hat[key] + est.R_tilde[key]
        np.testing.assert_allclose(recon, R, atol=1e-10 * np.trace(R).real)


def test_error_covariance_is_psd_and_bounded_by_R(bundle) -> None:
    """`0 <= R_tilde <= R` in the PSD order: estimation cannot add energy."""
    cfg, topo, lsf, st, Phi, cs, real, est = bundle
    for key, R in st.R.items():
        Rt = est.R_tilde[key]
        scale = np.trace(R).real
        assert np.linalg.eigvalsh(Rt).min() >= -1e-9 * scale
        assert np.linalg.eigvalsh(R - Rt).min() >= -1e-9 * scale


def test_R_tilde_H_is_the_sum_over_users(bundle) -> None:
    """
    `R~_{H_a} = sum_u R~_{a u}`, the quantity `eq:split-comm-bf` feeds into the
    LR-MMSE precoder.
    """
    cfg, topo, lsf, st, Phi, cs, real, est = bundle
    for ap in topo.tx_aps:
        want = sum(est.R_tilde[(ap.idx, ue.idx)] for ue in topo.ues)
        np.testing.assert_allclose(est.R_tilde_H[ap.idx], want,
                                   atol=1e-9 * np.trace(want).real)


def test_nmse_is_the_trace_ratio(bundle) -> None:
    cfg, topo, lsf, st, Phi, cs, real, est = bundle
    for key, R in st.R.items():
        want = np.trace(est.R_tilde[key]).real / np.trace(R).real
        assert est.nmse[key] == pytest.approx(want, rel=1e-9)
        assert 0.0 <= est.nmse[key] <= 1.0 + 1e-9


@pytest.mark.slow
def test_mmse_orthogonality(tiny_cfg) -> None:
    """
    `E{h_hat (h - h_hat)^H} = 0`: the orthogonality principle that makes the
    estimator MMSE and makes `R = R_hat + R_tilde` true (Appendix B).
    """
    cfg = copy.deepcopy(tiny_cfg)
    r = make_rng(23)
    topo = generate_topology(cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
    st = compute_channel_statistics(topo, cfg, lsf, child_rng(r))
    Phi, cs = design_pilot_sequences(topo.n_ue, cfg.channel.tau_p, child_rng(r))
    key = (topo.tx_aps[0].idx, topo.ues[0].idx)
    n, m = 2500, cfg.topology.n_ant
    cross = np.zeros((m, m), dtype=complex)
    for _ in range(n):
        real = generate_channel_realization(topo, cfg, lsf, st, child_rng(r))
        est = run_channel_estimation(topo, cfg, lsf, st, real, child_rng(r), Phi, cs)
        e = real.h[key] - est.h_hat[key]
        cross += np.outer(est.h_hat[key], e.conj())
    assert np.linalg.norm(cross / n) / np.trace(st.R[key]).real < 0.03


# =============================================================================
# Pilot-power limits
# =============================================================================

@pytest.mark.parametrize("pp_db,lo,hi", [
    (40.0, 0.98, 1.001),      # no information: R_tilde -> R
    (220.0, 0.0, 1e-6),       # near-perfect: R_tilde -> 0
])
def test_pilot_power_limits(tiny_cfg, pp_db: float, lo: float, hi: float) -> None:
    """`P_p -> 0` gives NMSE -> 1; `P_p -> inf` gives NMSE -> 0."""
    cfg = copy.deepcopy(tiny_cfg)
    cfg.channel.pilot_power_db = pp_db
    *_, est = _pipeline(cfg, seed=31)
    mean_nmse = float(np.mean(list(est.nmse.values())))
    assert lo <= mean_nmse <= hi


def test_nmse_decreases_monotonically_with_pilot_power(tiny_cfg) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    prev = np.inf
    for pp in (80.0, 110.0, 140.0, 170.0):
        cfg.channel.pilot_power_db = pp
        *_, est = _pipeline(cfg, seed=17)
        cur = float(np.mean(list(est.nmse.values())))
        assert cur <= prev + 1e-12
        prev = cur


@pytest.mark.slow
def test_deployed_pilot_power_is_not_near_perfect_csi(default_cfg) -> None:
    """
    Pins F-03-04: at the shipped `P_p* = 128 dB` the mean NMSE is about 0.3.

    §VI-E describes the sweep as running "up to near-perfect CSI at the
    operating point" and says the network "sits well inside the high-quality
    plateau". Measured on the paper's own configuration the operating point is
    on the steep part of the curve, not the plateau: NMSE only falls below 0.1
    beyond roughly 140 dB.
    """
    cfg = copy.deepcopy(default_cfg)
    assert cfg.channel.pilot_power_db == 128.0
    r = make_rng(20260811)
    vals = []
    for _ in range(3):
        topo = generate_topology(cfg, child_rng(r))
        lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
        st = compute_channel_statistics(topo, cfg, lsf, child_rng(r))
        Phi, cs = design_pilot_sequences(topo.n_ue, cfg.channel.tau_p, child_rng(r))
        real = generate_channel_realization(topo, cfg, lsf, st, child_rng(r))
        est = run_channel_estimation(topo, cfg, lsf, st, real, child_rng(r), Phi, cs)
        vals += list(est.nmse.values())
    mean_nmse = float(np.mean(vals))
    assert 0.15 < mean_nmse < 0.55, f"mean NMSE at P_p* is {mean_nmse:.3f}"


# =============================================================================
# estimation_method variants
# =============================================================================

def test_perfect_csi_returns_the_true_channel(tiny_cfg) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    cfg.channel.estimation_method = "perfect"
    _, _, st, _, _, real, est = _pipeline(cfg, seed=9)
    for key, h in real.h.items():
        np.testing.assert_allclose(est.h_hat[key], h)
        assert est.nmse[key] == pytest.approx(0.0)
        assert np.linalg.norm(est.R_tilde[key]) == pytest.approx(0.0)


@pytest.mark.xfail(strict=True,
                   reason="F-03-05: the LS branch reports R_hat = R - R_tilde, "
                          "so its stated NMSE understates the true LS error")
def test_ls_reported_nmse_matches_the_realized_error(tiny_cfg) -> None:
    """
    For any estimator the reported NMSE should track the realized error.

    The LS estimate itself is textbook (`h_hat = y / (sqrt(P_p) tau_p)`), but
    its covariance bookkeeping treats LS as if it were MMSE: `R_hat` is set to
    `R - sigma^2/(P_p tau_p) I`, whereas an unbiased LS estimate has
    `R_hat = R + R_tilde`. The realized error is far larger than the reported
    figure. LS is unreachable in every shipped campaign
    (`estimation_method = "MMSE"`), so nothing published depends on it.
    """
    cfg = copy.deepcopy(tiny_cfg)
    cfg.channel.estimation_method = "LS"
    _, _, st, _, _, real, est = _pipeline(cfg, seed=9)
    key = next(iter(est.nmse))
    realized = (np.linalg.norm(real.h[key] - est.h_hat[key]) ** 2
                / np.linalg.norm(real.h[key]) ** 2)
    assert realized == pytest.approx(est.nmse[key], rel=1.0)


def test_estimation_is_reproducible(tiny_cfg) -> None:
    """
    Same seed gives the same estimate.

    Note the tolerance: the MMSE solve is reproducible to ~1e-13 relative, not
    bitwise, because `np.linalg.solve` runs through a multithreaded BLAS whose
    summation order is not fixed. The channel realization itself *is* bitwise
    reproducible. Recorded as a lead for Stage 9, which owns the
    "same seed gives bit-identical TrialOutput" requirement.
    """
    a = _pipeline(tiny_cfg, seed=404)[-1]
    b = _pipeline(tiny_cfg, seed=404)[-1]
    for key in a.h_hat:
        np.testing.assert_allclose(a.h_hat[key], b.h_hat[key], rtol=1e-11)
    for key in a.nmse:
        assert a.nmse[key] == pytest.approx(b.nmse[key], rel=1e-11)
