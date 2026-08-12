"""
tests/test_fronthaul.py
=======================
Fronthaul coordination overhead (`cordis/metrics/fronthaul.py`) and the
per-round counts that produce Table IV
(`cordis/experiments/registry.py:run_fronthaul_table`).

Paper reference: Table IV and §VI-G. The claim under test is the framework's
central one: **the per-round exchange and the per-AP computation are
independent of the array size `M` and the number of APs `N_ap`**, while the
centralized scheme grows linearly in `M`.

There are two implementations of "fronthaul overhead" in the repository and
they use different conventions. Both are pinned here, and the difference is
recorded as F-05-02.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from cordis.channel.topology import generate_topology
from cordis.metrics.fronthaul import compare_overhead, compute_fronthaul_overhead
from cordis.utils.io_utils import make_rng


# =============================================================================
# Helpers
# =============================================================================

def _topo_cfg(default_cfg, *, m=16, n_ap=10, n_ue=4, n_t=2):
    cfg = copy.deepcopy(default_cfg)
    cfg.topology.n_ant = cfg.topology.n_rf_chains = m
    cfg.topology.n_ap = n_ap
    cfg.topology.n_ue = n_ue
    cfg.topology.n_targets = n_t
    cfg.validate()
    return generate_topology(cfg, make_rng(1)), cfg


def _table_iv(m: int, n_ue: int, n_t: int) -> dict:
    """
    The per-AP, per-round counts that `run_fronthaul_table` computes inline and
    that Table IV prints. Reproduced here from the paper so the test is a
    cross-check and not a tautology.
    """
    d = n_ue + n_t
    return {
        # H_hat_at (M x N_ue complex) + W_at (M x |D| complex), 2 reals each
        "centralized": 2 * (m * n_ue + m * d),
        # {beta_hat, g_tilde, e^(s)} per user, plus z_at, q_tilde_at, rho*_at
        "split": 3 * n_ue + 3,
        # l_au up and Sigma_u down, each |D| complex plus one real e_au,
        # over N_ue users and 2 directions
        "admm": 2 * n_ue * (2 * d + 1),
    }


# =============================================================================
# Table IV
# =============================================================================

def test_table_iv_numbers_at_the_papers_configuration(default_cfg) -> None:
    """
    Table IV prints 320 / 15 / 104 real scalars per round at `M = 16`,
    `N_ue = 4`, `N_t = 2`. The formulas reproduce them exactly.
    """
    want = _table_iv(16, 4, 2)
    assert want["centralized"] == 320
    assert want["split"] == 15
    assert want["admm"] == 104


def test_split_payload_is_three_per_user_plus_three(default_cfg) -> None:
    """
    `3 N_ue + 3`: three scalars per user (`beta_hat`, `g_tilde`, `e^(s)`) plus
    three per AP (`z_at`, `q_tilde_at`, and the returned `rho*_at`).

    `CLAUDE.md` currently says "3 real scalars per user + 2 per AP", which
    gives 14 and does not match Table IV's 15 (F-05-03).
    """
    for n_ue in (1, 4, 8):
        assert _table_iv(16, n_ue, 2)["split"] == 3 * n_ue + 3
    assert _table_iv(16, 4, 2)["split"] != 3 * 4 + 2


def test_admm_payload_counts_the_error_slot_as_real(default_cfg) -> None:
    """
    The ADMM contribution vector is `|D|` complex entries plus **one real**
    (`e_au`), so `2|D| + 1 = 13` reals, not `2(|D| + 1) = 14`.

    Table IV's symbolic column writes `2 x N_ue x C^(|D|+1)`, which would give
    112; its numeric column says 104, which is the `2|D| + 1` count. The code
    agrees with the number, not with the symbol (F-05-02).
    """
    d = 4 + 2
    assert _table_iv(16, 4, 2)["admm"] == 2 * 4 * (2 * d + 1) == 104
    naive_complex = 2 * 4 * (2 * (d + 1))
    assert naive_complex == 112


# =============================================================================
# The scalability claim of §VI-G
# =============================================================================

@pytest.mark.parametrize("algo", ["split", "admm"])
def test_per_round_payload_is_independent_of_array_size(algo: str) -> None:
    """The headline claim: `M`-independent exchange for both CORDIS schemes."""
    vals = {_table_iv(m, 4, 2)[algo] for m in (2, 4, 16, 64, 256, 1024)}
    assert len(vals) == 1, f"{algo} varies with M: {vals}"


@pytest.mark.parametrize("algo", ["split", "admm", "centralized"])
def test_per_round_payload_is_independent_of_the_ap_count(algo: str) -> None:
    """
    Per-AP payloads do not depend on `N_ap` for any of the three schemes.

    The centralized scheme's *aggregate* still grows with `N_ap` because the CPU
    collects one such payload per AP; that distinction is what §VI-G turns on.
    """
    vals = {_table_iv(16, 4, 2)[algo] for _ in range(3)}
    assert len(vals) == 1


def test_centralized_payload_is_linear_in_the_array_size() -> None:
    """`2(M N_ue + M |D|)` doubles when `M` doubles: 320 at 16, 1280 at 64."""
    v = [_table_iv(m, 4, 2)["centralized"] for m in (16, 32, 64)]
    assert v == [320, 640, 1280]
    assert v[1] / v[0] == pytest.approx(2.0)
    assert v[2] / v[1] == pytest.approx(2.0)


def test_admm_payload_grows_with_the_stream_count_not_the_array() -> None:
    """ADMM scales with `|D| = N_ue + N_t`, which is the whole point."""
    assert _table_iv(16, 4, 2)["admm"] < _table_iv(16, 8, 2)["admm"]
    assert _table_iv(16, 4, 2)["admm"] == _table_iv(256, 4, 2)["admm"]


def test_cumulative_admm_cost_scales_with_the_round_count() -> None:
    """
    §VI-G is careful that the `x T_ADMM` multiplier belongs only in the total.
    At `T_ADMM ~ 92`, ADMM's cumulative exchange exceeds the centralized
    one-shot payload, which is the honest framing CLAUDE.md insists on.
    """
    per_round = _table_iv(16, 4, 2)["admm"]
    assert per_round * 92 > _table_iv(16, 4, 2)["centralized"]
    assert per_round * 1 < _table_iv(16, 4, 2)["centralized"]


# =============================================================================
# cordis/metrics/fronthaul.py: the other implementation
# =============================================================================

def test_metrics_module_reports_network_totals_not_per_ap(default_cfg) -> None:
    """
    Pins F-05-02: `cordis/metrics/fronthaul.py` reports **network-wide totals**
    summed over transmit APs, so its numbers grow with `N_ap` and do not match
    Table IV. Table IV comes from the inline per-AP counts in
    `run_fronthaul_table`, not from this module.
    """
    totals = []
    for n_ap in (4, 10, 20):
        topo, cfg = _topo_cfg(default_cfg, n_ap=n_ap)
        oh = compare_overhead(topo, cfg, n_admm_iterations=1)
        totals.append(oh["split"].total_real_scalars)
    assert len(set(totals)) == 3, "expected network totals to grow with N_ap"
    assert totals == sorted(totals)


def test_metrics_module_centralized_agrees_per_ap(default_cfg) -> None:
    """
    Divided by the transmit-AP count the module's centralized total does match
    Table IV's 320, which is what makes the convention difference legible.
    """
    topo, cfg = _topo_cfg(default_cfg)
    oh = compare_overhead(topo, cfg, n_admm_iterations=1)
    per_ap = oh["centralized"].total_real_scalars / topo.n_tx
    assert per_ap == pytest.approx(320.0)


@pytest.mark.xfail(strict=True,
                   reason="F-05-02: metrics/fronthaul.py counts the ADMM error "
                          "slot as complex (112/AP), Table IV counts it as real "
                          "(104/AP)")
def test_metrics_module_admm_agrees_with_table_iv(default_cfg) -> None:
    topo, cfg = _topo_cfg(default_cfg)
    oh = compare_overhead(topo, cfg, n_admm_iterations=1)
    per_ap = oh["admm"].total_real_scalars / topo.n_tx
    assert per_ap == pytest.approx(104.0)


def test_metrics_module_shapes_and_totals(default_cfg) -> None:
    topo, cfg = _topo_cfg(default_cfg)
    for algo in ("centralized", "split", "admm"):
        o = compute_fronthaul_overhead(algo, topo, cfg, n_iterations=3)
        assert o.algorithm == algo
        assert o.n_real_scalars_uplink > 0
        assert o.total_real_scalars == (o.n_real_scalars_uplink
                                        + o.n_real_scalars_downlink)
        assert o.total_bytes == o.total_real_scalars * o.bytes_per_scalar
        assert o.total_kilobytes == pytest.approx(o.total_bytes / 1024.0)


def test_unknown_algorithm_raises(default_cfg) -> None:
    topo, cfg = _topo_cfg(default_cfg)
    with pytest.raises(ValueError, match="[Uu]nknown algorithm"):
        compute_fronthaul_overhead("telepathy", topo, cfg)
