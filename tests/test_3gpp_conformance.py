"""
tests/test_3gpp_conformance.py
==============================
Conformance of the CORDIS channel constants and formulas to 3GPP TR 38.901.

Reference values live in `tests/data/3gpp_umi_reference.json`, a self-contained
snapshot extracted from a verified TR 38.901 v19.4.0 transcription (provenance
header inside the file). **The tests never read the source repository**, so they
must run on any machine.

The manuscript's §VI-A claims the simulation uses "the path-loss, LoS
probability, shadow-fading, and angular-spread parameters of 3GPP TR 38.901".
These tests check that claim clause by clause. Three of the four hold; the
angular-spread clause does not, and the known deviations are marked
`xfail(strict=True)` against the findings that track them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cordis.channel.pathloss import (
    _SHADOW_STD, _breakpoint_distance_umi, los_probability_umi,
    noise_power_watts, path_loss_umi_los, path_loss_umi_nlos, snr_to_tx_power,
)

REF_PATH = Path(__file__).resolve().parent / "data" / "3gpp_umi_reference.json"
REF = json.loads(REF_PATH.read_text())

FC_GHZ = REF["carrier_freq_ghz"]
FC_HZ = FC_GHZ * 1e9
H_BS = REF["geometry"]["h_bs_m"]
H_UT = REF["geometry"]["h_ut_m"]


# =============================================================================
# Provenance
# =============================================================================

def test_reference_fixture_declares_its_provenance() -> None:
    """
    The fixture records which document, version, and tables it came from.

    Without this the conformance suite is just a set of magic numbers.
    """
    p = REF["_provenance"]
    assert p["document"] == "3GPP TR 38.901"
    assert p["version"] == "V19.4.0"
    assert p["tables"] and p["extracted_on"]


# =============================================================================
# Path loss: Table 7.4.1-1, UMi-StreetCanyon
# =============================================================================

def _pl_los_ref(d_2d: float, d_3d: float) -> float:
    c = REF["path_loss"]["los"]
    d_bp = _breakpoint_distance_umi(FC_HZ, H_BS, H_UT)
    if d_2d <= d_bp:
        k = c["pl1"]
        return (k["const"] + k["log10_d3d_coeff"] * math.log10(d_3d)
                + k["log10_fc_coeff"] * math.log10(FC_GHZ))
    k = c["pl2"]
    return (k["const"] + k["log10_d3d_coeff"] * math.log10(d_3d)
            + k["log10_fc_coeff"] * math.log10(FC_GHZ)
            + k["breakpoint_term_coeff"] * math.log10(d_bp ** 2 + (H_BS - H_UT) ** 2))


def _pl_nlos_prime_ref(d_3d: float) -> float:
    k = REF["path_loss"]["nlos_prime"]
    return (k["log10_d3d_coeff"] * math.log10(d_3d) + k["const"]
            + k["log10_fc_coeff"] * math.log10(FC_GHZ)
            + k["h_ut_coeff"] * (H_UT - k["h_ut_ref_m"]))


@pytest.mark.parametrize("d_2d", [10.0, 35.0, 100.0, 209.0, 211.0, 650.0,
                                  1000.0, 1414.0, 5000.0])
def test_umi_los_path_loss_matches_the_tr_formula(d_2d: float) -> None:
    """
    `path_loss_umi_los` reproduces Table 7.4.1-1 exactly, on both sides of the
    breakpoint. Distances 209/211 m straddle `d'_BP` at 3.5 GHz.
    """
    d_3d = math.sqrt(d_2d ** 2 + (H_BS - H_UT) ** 2)
    got = path_loss_umi_los(np.array([d_2d]), np.array([d_3d]), FC_HZ,
                            H_BS, H_UT).item()
    assert got == pytest.approx(_pl_los_ref(d_2d, d_3d), abs=1e-9)


@pytest.mark.parametrize("d_2d", [10.0, 35.0, 100.0, 650.0, 1000.0, 5000.0])
def test_umi_nlos_path_loss_matches_the_tr_formula(d_2d: float) -> None:
    """`PL_NLOS = max(PL_LOS, PL'_NLOS)` with the Table 7.4.1-1 coefficients."""
    d_3d = math.sqrt(d_2d ** 2 + (H_BS - H_UT) ** 2)
    got = path_loss_umi_nlos(np.array([d_2d]), np.array([d_3d]), FC_HZ,
                             H_BS, H_UT).item()
    want = max(_pl_los_ref(d_2d, d_3d), _pl_nlos_prime_ref(d_3d))
    assert got == pytest.approx(want, abs=1e-9)


def test_nlos_is_never_below_los() -> None:
    """The `max(·)` clamp of Table 7.4.1-1 is applied, not just the primed form."""
    d_2d = np.array([10.0, 20.0, 50.0, 200.0, 1000.0])
    d_3d = np.sqrt(d_2d ** 2 + (H_BS - H_UT) ** 2)
    los = path_loss_umi_los(d_2d, d_3d, FC_HZ, H_BS, H_UT)
    nlos = path_loss_umi_nlos(d_2d, d_3d, FC_HZ, H_BS, H_UT)
    assert np.all(nlos >= los - 1e-12)


def test_breakpoint_distance_uses_the_umi_effective_height() -> None:
    """`d'_BP = 4 (h_BS − h_E)(h_UT − h_E) f_c / c` with `h_E = 1 m` (TR note 1)."""
    h_e = REF["geometry"]["h_e_m"]
    want = 4.0 * (H_BS - h_e) * (H_UT - h_e) * FC_HZ / 3.0e8
    assert _breakpoint_distance_umi(FC_HZ, H_BS, H_UT) == pytest.approx(want)
    assert want == pytest.approx(210.0, abs=1.0)


def test_path_loss_is_monotone_in_distance() -> None:
    d_2d = np.geomspace(10.0, 5000.0, 200)
    d_3d = np.sqrt(d_2d ** 2 + (H_BS - H_UT) ** 2)
    for fn in (path_loss_umi_los, path_loss_umi_nlos):
        pl = fn(d_2d, d_3d, FC_HZ, H_BS, H_UT)
        assert np.all(np.diff(pl) > -1e-9), f"{fn.__name__} is not monotone"


def test_path_loss_is_continuous_at_the_breakpoint() -> None:
    """
    The two-slope LoS model is continuous at `d'_BP`.

    A discontinuity here would show up as a ring artifact in the SINR maps.
    """
    d_bp = _breakpoint_distance_umi(FC_HZ, H_BS, H_UT)
    lo, hi = d_bp - 1e-3, d_bp + 1e-3
    d3 = lambda d: math.sqrt(d ** 2 + (H_BS - H_UT) ** 2)
    a = path_loss_umi_los(np.array([lo]), np.array([d3(lo)]), FC_HZ, H_BS, H_UT).item()
    b = path_loss_umi_los(np.array([hi]), np.array([d3(hi)]), FC_HZ, H_BS, H_UT).item()
    assert abs(a - b) < 0.05


def test_d2d_is_clamped_to_the_tr_minimum() -> None:
    """Below 10 m the model is out of range; the code clamps rather than extrapolates."""
    d3 = math.sqrt(10.0 ** 2 + (H_BS - H_UT) ** 2)
    at_min = path_loss_umi_los(np.array([10.0]), np.array([d3]), FC_HZ, H_BS, H_UT)
    below = path_loss_umi_los(np.array([1.0]), np.array([d3]), FC_HZ, H_BS, H_UT)
    assert below.item() == pytest.approx(at_min.item())


# =============================================================================
# LoS probability. Table 7.4.2-1
# =============================================================================

def _p_los_ref(d: float) -> float:
    t = REF["los_probability"]["d_threshold_m"]
    k = REF["los_probability"]["decay_m"]
    if d <= t:
        return 1.0
    return t / d + math.exp(-d / k) * (1.0 - t / d)


@pytest.mark.parametrize("d", [1.0, 10.0, 18.0, 18.001, 25.0, 50.0, 100.0,
                               650.0, 1000.0, 5000.0])
def test_umi_los_probability_matches_the_tr_formula(d: float) -> None:
    """
    Table 7.4.2-1: `P_LoS = 1` for `d ≤ 18 m`, else `18/d + e^{−d/36}(1 − 18/d)`.

    The code writes it as `min(18/d,1)(1 − e^{−d/36}) + e^{−d/36}`, which is the
    same expression rearranged, this test is what proves the rearrangement.
    """
    assert los_probability_umi(np.array([d])).item() == pytest.approx(_p_los_ref(d))


def test_los_probability_is_bounded_and_decreasing() -> None:
    d = np.geomspace(1.0, 5000.0, 500)
    p = los_probability_umi(d)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p) <= 1e-12)


def test_los_probability_is_one_inside_18_m() -> None:
    assert np.allclose(los_probability_umi(np.array([1.0, 10.0, 18.0])), 1.0)


# =============================================================================
# Shadow fading: Table 7.4.1-1 σ_SF, Table 7.5-6 decorrelation distance
# =============================================================================

def test_umi_shadow_fading_std_matches_the_tr_table() -> None:
    """σ_SF = 4 dB (LoS) and 7.82 dB (NLoS), the values §VI-A quotes."""
    assert _SHADOW_STD["UMi"]["LoS"] == pytest.approx(REF["shadow_fading_std_db"]["los"])
    assert _SHADOW_STD["UMi"]["NLoS"] == pytest.approx(REF["shadow_fading_std_db"]["nlos"])


def test_uma_shadow_fading_std_matches_the_tr_table() -> None:
    ref = REF["other_scenarios_shadow_fading_std_db"]["UMa"]
    assert _SHADOW_STD["UMa"]["LoS"] == pytest.approx(ref["los"])
    assert _SHADOW_STD["UMa"]["NLoS"] == pytest.approx(ref["nlos"])


@pytest.mark.xfail(strict=True,
                   reason="F-02-01: shadow_corr_distance_m = 50 m is the UMa-NLOS "
                          "value; UMi-Street Canyon is 10 m (LoS) / 13 m (NLoS)")
def test_shadow_decorrelation_distance_matches_umi(default_cfg) -> None:
    """
    Table 7.5-6 gives `corr_distance_SF_m` = 10 m (LOS) / 13 m (NLOS) for
    UMi-Street Canyon. The config uses a single 50 m, which is the UMa-NLOS
    entry. A longer decorrelation distance makes shadowing more common-mode
    across UEs, which compresses the spread of the per-user SINR.
    """
    got = default_cfg.channel.shadow_corr_distance_m
    assert got in (REF["corr_distance_sf_m"]["los"], REF["corr_distance_sf_m"]["nlos"])


# =============================================================================
# Rician K-factor. Table 7.5-6
# =============================================================================

def test_rician_k_factor_matches_umi_street_canyon_los(default_cfg) -> None:
    """
    μ_K = 9 dB, σ_K = 5 dB. σ_K = 5 is specific to UMi-SC (UMa LoS is 3.5),
    so this also confirms the right scenario row was used.
    """
    assert default_cfg.channel.rician_k_db_mean == pytest.approx(REF["rician_k_db"]["mu"])
    assert default_cfg.channel.rician_k_db_std == pytest.approx(REF["rician_k_db"]["sigma"])


# =============================================================================
# Angular spread. Table 7.5-6 / 7.5-7
# =============================================================================

_XFAIL_AS = pytest.mark.xfail(
    strict=True,
    reason="F-02-02: fixed stand-in, not the frequency/distance-dependent "
           "TR 38.901 value",
)

#: (config field, TR condition, TR key, deviates?). Only the NLoS azimuth is
#: close to its TR counterpart (22.5° vs ASD 24.17°); the other three are not.
_AS_CASES = [
    pytest.param("as_azimuth_deg_std_los", "los", "asd",
                 id="as_azimuth_deg_std_los", marks=[_XFAIL_AS]),
    pytest.param("as_azimuth_deg_std_nlos", "nlos", "asd",
                 id="as_azimuth_deg_std_nlos"),
    pytest.param("as_elevation_deg_std_los", "los", "zsd_at_650m",
                 id="as_elevation_deg_std_los", marks=[_XFAIL_AS]),
    pytest.param("as_elevation_deg_std_nlos", "nlos", "zsd_at_650m",
                 id="as_elevation_deg_std_nlos", marks=[_XFAIL_AS]),
]


@pytest.mark.parametrize("field,cond,ref_key", _AS_CASES)
def test_angular_spread_matches_tr(default_cfg, field: str, cond: str,
                                   ref_key: str) -> None:
    """
    §VI-A claims TR 38.901 "angular-spread parameters".

    The AP is the transmitter and `C_{au}` is the departure-side correlation, so
    the governing quantities are ASD (azimuth) and ZSD (zenith), both of which
    TR 38.901 specifies as lognormals in `f_c` (and, for ZSD, `d_2D`). The code
    uses four constants instead. Tolerance is deliberately loose (±25 %), this
    is not about rounding.
    """
    want = REF["angular_spread_deg_at_3p5ghz"][cond][ref_key]
    got = getattr(default_cfg.channel, field)
    assert got == pytest.approx(want, rel=0.25)


def test_angular_spread_deviation_is_the_recorded_one(default_cfg) -> None:
    """
    Pins the *size* of the F-02-02 deviation so the correction phase can judge
    the blast radius, and so a silent drift in either direction fails here.
    """
    ch = default_cfg.channel
    los, nlos = (REF["angular_spread_deg_at_3p5ghz"]["los"],
                 REF["angular_spread_deg_at_3p5ghz"]["nlos"])
    assert ch.as_azimuth_deg_std_los == pytest.approx(5.0)      # TR ASD  15.04
    assert ch.as_azimuth_deg_std_nlos == pytest.approx(22.5)    # TR ASD  24.17
    assert ch.as_elevation_deg_std_los == pytest.approx(3.0)    # TR ZSD   0.32
    assert ch.as_elevation_deg_std_nlos == pytest.approx(7.0)   # TR ZSD   0.34
    # The NLoS azimuth is the only one that is even close.
    assert abs(ch.as_azimuth_deg_std_nlos - nlos["asd"]) / nlos["asd"] < 0.10
    assert abs(ch.as_azimuth_deg_std_los - los["asd"]) / los["asd"] > 0.50


# =============================================================================
# Link budget: §VI-A of the manuscript
# =============================================================================

def test_noise_floor_is_minus_94_dbm(default_cfg) -> None:
    """`σ_n² = k_B T_0 B · NF ≈ 4.0e-13 W (−94 dBm)` at B = 20 MHz, NF = 7 dB."""
    s = noise_power_watts(default_cfg.frequency.bandwidth_hz,
                          default_cfg.channel.noise_figure_db,
                          default_cfg.channel.noise_temp_k)
    assert s == pytest.approx(4.0e-13, rel=0.02)
    assert 10 * math.log10(s * 1e3) == pytest.approx(-94.0, abs=0.1)


def test_snr_db_137_gives_20_watts(default_cfg) -> None:
    """`P_max/σ_n² = 137 dB ⇒ P_max ≈ 20 W (43 dBm)`, the paper's Table III."""
    s = noise_power_watts(default_cfg.frequency.bandwidth_hz,
                          default_cfg.channel.noise_figure_db,
                          default_cfg.channel.noise_temp_k)
    assert snr_to_tx_power(137.0, s) == pytest.approx(20.0, rel=0.02)


def test_pilot_power_128_db_gives_2p5_watts(default_cfg) -> None:
    """`P_p/σ_n² = 128 dB ⇒ P_p ≈ 2.5 W (34 dBm)`."""
    s = noise_power_watts(default_cfg.frequency.bandwidth_hz,
                          default_cfg.channel.noise_figure_db,
                          default_cfg.channel.noise_temp_k)
    assert snr_to_tx_power(128.0, s) == pytest.approx(2.5, rel=0.03)


def test_snr_to_tx_power_inverts_exactly() -> None:
    s = noise_power_watts(20e6, 7.0, 290.0)
    for snr_db in (0.0, 20.0, 128.0, 137.0):
        assert 10 * math.log10(snr_to_tx_power(snr_db, s) / s) == pytest.approx(snr_db)


def test_noise_power_scales_with_bandwidth_and_noise_figure() -> None:
    base = noise_power_watts(20e6, 7.0, 290.0)
    assert noise_power_watts(40e6, 7.0, 290.0) == pytest.approx(2 * base)
    assert noise_power_watts(20e6, 10.0, 290.0) == pytest.approx(base * 10 ** 0.3)
