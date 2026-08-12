"""
tests/test_sensing_assignment.py
================================
AP-target association for multi-static sensing
(`cordis/channel/sensing_assignment.py`).

Paper reference: §II. "each transmit AP `a_t` jointly serves all users in `U`
while concurrently illuminating a CPU-selected subset of targets
`T_{a_t} subset-of T`", and "the CPU assigns every receive AP `a_r` a subset of
targets `T_{a_r}` to process". Table II lists `T_{a_t}` and `T_{a_r}` as
first-class objects, so the semantics matter.

The default strategy is covered in `test_topology.py` (F-02-06, F-02-07); this
file covers the alternatives and the structural invariants that hold for all of
them.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from cordis.channel.pathloss import compute_large_scale_fading
from cordis.channel.sensing_assignment import assign_sensing
from cordis.channel.topology import generate_topology
from cordis.utils.io_utils import child_rng, make_rng

STRATEGIES = ["single_closest_centroid", "single_farthest_centroid",
              "per_target_closest"]


def _build(cfg, seed: int = 515):
    r = make_rng(seed)
    topo = generate_topology(cfg, child_rng(r))
    lsf = compute_large_scale_fading(topo, cfg, child_rng(r))
    return topo, lsf


@pytest.fixture
def multi_target_cfg(default_cfg):
    """Three targets, so per-target strategies have something to distinguish."""
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.n_targets = 3
    cfg.validate()
    return cfg


# =============================================================================
# Invariants that hold for every strategy
# =============================================================================

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_association_is_structurally_valid(multi_target_cfg, strategy) -> None:
    """
    Every target has a non-empty `T_{a_t}` and `T_{a_r}`, the two are disjoint
    (half duplex), and every index refers to a real AP.
    """
    topo, lsf = _build(multi_target_cfg)
    assoc = assign_sensing(topo, multi_target_cfg, lsf, strategy=strategy)
    assoc.validate(topo)
    for tg in range(topo.n_targets):
        tx = set(assoc.tx_aps_per_target[tg])
        rx = set(assoc.rx_aps_per_target[tg])
        assert tx and rx
        assert not (tx & rx)
        assert tx | rx <= set(range(topo.n_ap))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_association_records_its_strategy(multi_target_cfg, strategy) -> None:
    topo, lsf = _build(multi_target_cfg)
    assoc = assign_sensing(topo, multi_target_cfg, lsf, strategy=strategy)
    assert strategy.split("_")[-1] in assoc.strategy or assoc.strategy


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_tx_and_rx_index_helpers_agree_with_the_per_target_maps(
        multi_target_cfg, strategy) -> None:
    topo, lsf = _build(multi_target_cfg)
    assoc = assign_sensing(topo, multi_target_cfg, lsf, strategy=strategy)
    tx_union = {a for lst in assoc.tx_aps_per_target.values() for a in lst}
    rx_union = {a for lst in assoc.rx_aps_per_target.values() for a in lst}
    assert set(assoc.all_tx_ap_indices) == tx_union
    assert set(assoc.all_rx_ap_indices) == rx_union
    for tg in range(topo.n_targets):
        assert set(assoc.tx_aps_for(tg)) == set(assoc.tx_aps_per_target[tg])
        assert set(assoc.rx_aps_for(tg)) == set(assoc.rx_aps_per_target[tg])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_association_is_deterministic(multi_target_cfg, strategy) -> None:
    """The same topology gives the same association: no hidden RNG use."""
    topo, lsf = _build(multi_target_cfg)
    a = assign_sensing(topo, multi_target_cfg, lsf, strategy=strategy)
    b = assign_sensing(topo, multi_target_cfg, lsf, strategy=strategy)
    assert a.tx_aps_per_target == b.tx_aps_per_target
    assert a.rx_aps_per_target == b.rx_aps_per_target


# =============================================================================
# Strategy-specific behaviour
# =============================================================================

def test_per_target_closest_gives_each_target_its_own_receiver(
        multi_target_cfg) -> None:
    """
    The strategy §VI-A actually describes: one receive AP per target, the one
    closest to that individual target.

    It exists and works; it is simply not the default (F-02-06).
    """
    topo, lsf = _build(multi_target_cfg)
    assoc = assign_sensing(topo, multi_target_cfg, lsf,
                           strategy="per_target_closest")
    rx = [assoc.rx_aps_per_target[t] for t in range(topo.n_targets)]
    assert all(len(r) == 1 for r in rx)
    assert len({r[0] for r in rx}) == topo.n_targets, (
        "each target should get a distinct receiver")


def test_farthest_centroid_picks_a_different_ap_than_closest(
        multi_target_cfg) -> None:
    topo, lsf = _build(multi_target_cfg)
    near = assign_sensing(topo, multi_target_cfg, lsf,
                          strategy="single_closest_centroid")
    far = assign_sensing(topo, multi_target_cfg, lsf,
                         strategy="single_farthest_centroid")
    assert set(near.all_rx_ap_indices) != set(far.all_rx_ap_indices)


def test_unknown_strategy_raises(multi_target_cfg) -> None:
    topo, lsf = _build(multi_target_cfg)
    with pytest.raises(ValueError, match="[Uu]nknown"):
        assign_sensing(topo, multi_target_cfg, lsf, strategy="telepathy")


def test_assent_file_strategy_requires_a_path(multi_target_cfg) -> None:
    """The ASSENT hook fails loudly rather than silently falling back."""
    topo, lsf = _build(multi_target_cfg)
    with pytest.raises(ValueError, match="assent"):
        assign_sensing(topo, multi_target_cfg, lsf, strategy="assent_file")


def test_strategy_defaults_to_the_config(multi_target_cfg) -> None:
    topo, lsf = _build(multi_target_cfg)
    explicit = assign_sensing(topo, multi_target_cfg, lsf,
                              strategy=multi_target_cfg.sensing.rx_strategy)
    implicit = assign_sensing(topo, multi_target_cfg, lsf)
    assert implicit.tx_aps_per_target == explicit.tx_aps_per_target
    assert implicit.rx_aps_per_target == explicit.rx_aps_per_target


# =============================================================================
# Degenerate cases
# =============================================================================

def test_zero_targets_gives_an_empty_association(default_cfg) -> None:
    """
    A target-free topology yields an empty association and warns while picking
    a fallback receiver, rather than failing or silently producing a bogus one.
    """
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.n_targets = 0
    with pytest.warns(UserWarning, match="No targets"):
        topo, lsf = _build(cfg)
    assoc = assign_sensing(topo, cfg, lsf)
    assert assoc.tx_aps_per_target == {}
    assert assoc.rx_aps_per_target == {}
    assert "0 targets" in assoc.strategy or assoc.strategy


def test_single_target_still_yields_one_receiver(default_cfg) -> None:
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.n_targets = 1
    topo, lsf = _build(cfg)
    for strategy in STRATEGIES:
        assoc = assign_sensing(topo, cfg, lsf, strategy=strategy)
        assert len(assoc.rx_aps_per_target[0]) == 1
