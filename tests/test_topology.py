"""
tests/test_topology.py
======================
Network geometry (`cordis/channel/topology.py`) and the sensing-RX split.

The manuscript's §VI-A describes the deployment precisely. APs on a 650 m
circle, UEs and targets uniform in a 1 km disc, "for each target, the
geographically closest AP is designated as the sensing receive AP", so these
tests read as a check of the text as much as of the code.
"""
from __future__ import annotations

import numpy as np
import pytest

from cordis.channel.pathloss import compute_large_scale_fading
from cordis.channel.sensing_assignment import assign_sensing
from cordis.channel.topology import generate_topology
from cordis.utils.io_utils import child_rng, make_rng

N_DROPS = 40


@pytest.fixture
def drops(default_cfg):
    """A handful of drops at the paper's default geometry."""
    r = make_rng(20260810)
    return [generate_topology(default_cfg, child_rng(r)) for _ in range(N_DROPS)]


# =============================================================================
# Placement
# =============================================================================

def test_aps_lie_on_the_configured_circle(drops, default_cfg) -> None:
    """§VI-A: "$\\Nap$ ISAC transmit APs are uniformly deployed on a circle of
    radius 650 m"."""
    for topo in drops:
        r = np.linalg.norm(topo.ap_positions[:, :2], axis=1)
        np.testing.assert_allclose(r, default_cfg.topology.ap_radius_m, rtol=1e-9)


def test_aps_are_equally_spaced(drops) -> None:
    for topo in drops:
        ang = np.sort(np.arctan2(topo.ap_positions[:, 1], topo.ap_positions[:, 0]))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
        np.testing.assert_allclose(gaps, 2 * np.pi / topo.n_ap, atol=1e-9)


def test_ues_and_targets_lie_inside_the_configured_annulus(drops, default_cfg) -> None:
    t = default_cfg.topology
    for topo in drops:
        r_ue = np.linalg.norm(topo.ue_positions[:, :2], axis=1)
        r_tg = np.linalg.norm(topo.target_positions[:, :2], axis=1)
        assert np.all(r_ue >= t.ue_min_radius_m - 1e-9)
        assert np.all(r_ue <= t.ue_max_radius_m + 1e-9)
        assert np.all(r_tg >= t.tg_min_radius_m - 1e-9)
        assert np.all(r_tg <= t.tg_max_radius_m + 1e-9)


def test_heights_match_the_config(drops, default_cfg) -> None:
    t = default_cfg.topology
    for topo in drops:
        np.testing.assert_allclose(topo.ap_positions[:, 2], t.h_ap_m)
        np.testing.assert_allclose(topo.ue_positions[:, 2], t.h_ue_m)
        np.testing.assert_allclose(topo.target_positions[:, 2], t.h_tg_m)


def test_ue_placement_is_area_uniform(default_cfg) -> None:
    """
    "dropped uniformly at random within a concentric disc" means uniform in
    *area*, so `(r² − r_min²)/(r_max² − r_min²)` must be U(0,1), mean 0.5.

    Naive uniform-in-radius sampling would cluster toward the center and give
    a mean well below 0.5.
    """
    r = make_rng(7)
    radii = np.concatenate([
        np.linalg.norm(generate_topology(default_cfg, child_rng(r)).ue_positions[:, :2],
                       axis=1)
        for _ in range(300)
    ])
    t = default_cfg.topology
    u = (radii ** 2 - t.ue_min_radius_m ** 2) / (
        t.ue_max_radius_m ** 2 - t.ue_min_radius_m ** 2)
    assert u.mean() == pytest.approx(0.5, abs=0.02)
    assert np.all((u >= -1e-12) & (u <= 1 + 1e-12))


def test_topology_is_reproducible(default_cfg) -> None:
    a = generate_topology(default_cfg, make_rng(123))
    b = generate_topology(default_cfg, make_rng(123))
    np.testing.assert_array_equal(a.ue_positions, b.ue_positions)
    np.testing.assert_array_equal(a.target_positions, b.target_positions)


# =============================================================================
# Minimum-separation rejection sampler
# =============================================================================

def test_min_separation_is_enforced_when_requested(default_cfg) -> None:
    import copy
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.ue_min_separation_m = 150.0
    cfg.topology.ue_target_min_separation_m = 120.0
    r = make_rng(3)
    for _ in range(15):
        topo = generate_topology(cfg, child_rng(r))
        ue = topo.ue_positions[:, :2]
        d = np.linalg.norm(ue[:, None, :] - ue[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert d.min() >= 150.0 - 1e-9
        tg = topo.target_positions[:, :2]
        dt = np.linalg.norm(ue[:, None, :] - tg[None, :, :], axis=-1)
        assert dt.min() >= 120.0 - 1e-9


def test_zero_separation_keeps_the_legacy_sampler(default_cfg) -> None:
    """
    `0.0` must take the original code path so previously generated seeds still
    reproduce their topologies bit-for-bit.
    """
    import copy
    assert default_cfg.topology.ue_min_separation_m == 0.0
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.ue_min_separation_m = 0.0
    a = generate_topology(default_cfg, make_rng(55))
    b = generate_topology(cfg, make_rng(55))
    np.testing.assert_array_equal(a.ue_positions, b.ue_positions)


def test_impossible_separation_raises_rather_than_hanging(default_cfg) -> None:
    """The packing pre-check refuses instead of spinning in rejection sampling."""
    import copy
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.n_ue = 8
    cfg.topology.ue_min_separation_m = 5000.0
    with pytest.raises(ValueError, match="too large|Could not place"):
        generate_topology(cfg, make_rng(1))


# =============================================================================
# Array geometry
# =============================================================================

def test_every_ap_carries_the_configured_array(drops, default_cfg) -> None:
    for topo in drops[:5]:
        for ap in topo.aps:
            assert ap.n_ant == default_cfg.topology.n_ant


def test_ap_ue_distances_are_consistent(drops) -> None:
    """`d_3D² = d_2D² + Δh²` for every AP–UE pair."""
    for topo in drops[:5]:
        d2, d3 = topo.ap_ue_distances_2d(), topo.ap_ue_distances_3d()
        dh = topo.ap_positions[:, 2][:, None] - topo.ue_positions[:, 2][None, :]
        np.testing.assert_allclose(d3 ** 2, d2 ** 2 + dh ** 2, rtol=1e-9)
        assert np.all(d3 >= d2 - 1e-9)


# =============================================================================
# Transmit / receive AP split
# =============================================================================

def test_tx_and_rx_ap_sets_partition_the_network(drops) -> None:
    """`A_t ∩ A_r = ∅` and `A_t ∪ A_r = A` (§II)."""
    for topo in drops:
        tx = {a.idx for a in topo.tx_aps}
        rx = {a.idx for a in topo.rx_aps}
        assert not (tx & rx)
        assert tx | rx == set(range(topo.n_ap))


def test_receive_ap_count_follows_n_sensing_rx(drops, default_cfg) -> None:
    """
    Pins F-02-06: `|A_r| = n_sensing_rx`. **one** receive AP by default,
    independent of `N_t`.

    §VI-A says "for each target, the geographically closest AP is designated
    as the sensing receive AP", which reads as `|A_r| = N_t = 2`.
    """
    for topo in drops:
        assert len(topo.rx_aps) == default_cfg.topology.n_sensing_rx
        assert len(topo.tx_aps) == topo.n_ap - default_cfg.topology.n_sensing_rx
    assert default_cfg.topology.n_sensing_rx == 1
    assert default_cfg.topology.n_targets == 2


def test_transmit_antenna_budget_of_the_default_config(default_cfg, drops) -> None:
    """
    The deployed transmit-antenna budget is `|A_t| · M = 9 × 16 = 144`.

    Recorded because §VI-F's `fig_lowrank` claims the budget is "held constant"
    against this baseline; it is the denominator of that comparison.
    """
    topo = drops[0]
    assert len(topo.tx_aps) * default_cfg.topology.n_ant == 144


def test_sensing_rx_is_the_ap_closest_to_the_target_centroid(default_cfg) -> None:
    """
    Pins the `single_closest_centroid` strategy: the receive AP minimizes the
    distance to the *centroid of all targets*, not to any individual target.
    """
    r = make_rng(88)
    for _ in range(N_DROPS):
        topo = generate_topology(default_cfg, child_rng(r))
        lsf = compute_large_scale_fading(topo, default_cfg, child_rng(r))
        assign_sensing(topo, default_cfg, lsf)
        centroid = topo.target_positions.mean(axis=0)
        d = np.linalg.norm(topo.ap_positions[:, :2] - centroid[:2], axis=1)
        assert int(np.argmin(d)) in {a.idx for a in topo.rx_aps}


def test_default_strategy_assigns_every_tx_ap_to_every_target(default_cfg) -> None:
    """
    Pins F-02-07: `single_closest_centroid` sets `T_{a_t} = T` for every AP.

    §II says each transmit AP illuminates "a CPU-selected subset of targets
    T_{a_t} ⊆ T". With the shipped strategy that subset is always the whole
    set, so the selection is vacuous.
    """
    r = make_rng(9)
    topo = generate_topology(default_cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, default_cfg, child_rng(r))
    assoc = assign_sensing(topo, default_cfg, lsf)
    all_tx = sorted(a.idx for a in topo.tx_aps)
    for t_idx in range(topo.n_targets):
        assert sorted(assoc.tx_aps_per_target[t_idx]) == all_tx
        assert len(assoc.rx_aps_per_target[t_idx]) == 1


def test_sensing_association_is_half_duplex(default_cfg) -> None:
    """No AP both illuminates and receives for the same target."""
    r = make_rng(9)
    topo = generate_topology(default_cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, default_cfg, child_rng(r))
    assoc = assign_sensing(topo, default_cfg, lsf)
    for t_idx in range(topo.n_targets):
        assert not (set(assoc.tx_aps_per_target[t_idx])
                    & set(assoc.rx_aps_per_target[t_idx]))
    assoc.validate(topo)


@pytest.mark.xfail(strict=True,
                   reason="F-02-07: max_tx_aps_per_target / max_rx_aps_per_target "
                          "are never read by any assignment strategy")
def test_sensing_assignment_respects_the_caps(default_cfg) -> None:
    """
    `K_tx = 5` and `K_rx = 3` should bound the per-target association.

    `_single_global` assigns all 9 transmit APs to every target and never
    consults either cap.
    """
    r = make_rng(9)
    topo = generate_topology(default_cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, default_cfg, child_rng(r))
    assoc = assign_sensing(topo, default_cfg, lsf)
    s = default_cfg.sensing
    for t_idx in range(topo.n_targets):
        assert len(assoc.tx_aps_per_target[t_idx]) <= s.max_tx_aps_per_target
        assert len(assoc.rx_aps_per_target[t_idx]) <= s.max_rx_aps_per_target
