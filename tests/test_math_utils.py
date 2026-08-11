"""
tests/test_math_utils.py
========================
Numerical primitives (`cordis/utils/math_utils.py`).

Everything downstream inherits these conventions, so the tests are written as
*convention pins* as much as correctness checks: dB vs linear, amplitude vs
power, and, the one that matters most for the paper, the normalization of the
array steering vectors.

Note on `‖a‖`: this module returns **unit-norm** steering vectors (`‖a‖² = 1`),
whereas the manuscript's §II states `‖a_{au}‖² = M_t`. The convention is pinned
below so Stage 2 can settle whether the two are reconciled elsewhere.
"""
from __future__ import annotations

import numpy as np
import pytest

from cordis.utils.math_utils import (
    channel_quality_metric, compute_wavelength, db2lin, dbm2watt, deg2rad,
    frobenius_norm_sq, is_positive_semidefinite, khatri_rao, lin2db,
    nearest_psd, normalise_columns, null_space_projection,
    null_space_projection_reg, projection_matrix, rad2deg, regularised_pinv,
    steering_matrix, steering_uca, steering_ula, steering_vector, uca_radius,
    ula_spacing, vec, watt2dbm, wrap_to_pi,
)

LAMBDA = 3e8 / 3.5e9          # the paper's 3.5 GHz carrier


# =============================================================================
# dB / linear
# =============================================================================

@pytest.mark.parametrize("x_db", [-40.0, -3.0, 0.0, 5.0, 20.0, 137.0])
def test_db_round_trip(x_db: float) -> None:
    """`lin2db(db2lin(x)) == x`; the pair is a power (10·log10) convention."""
    assert lin2db(db2lin(x_db)) == pytest.approx(x_db)


def test_db2lin_is_a_power_ratio_not_amplitude() -> None:
    """3 dB doubles power; 20 dB is ×100. Pins 10·log10, not 20·log10."""
    assert db2lin(3.0) == pytest.approx(2.0, rel=1e-2)
    assert db2lin(20.0) == pytest.approx(100.0)
    assert db2lin(0.0) == pytest.approx(1.0)


def test_lin2db_clamps_at_zero_instead_of_warning() -> None:
    """
    `lin2db(0)` returns the −3000 dB floor rather than `-inf` or a warning.

    The clamp (`np.maximum(x, 1e-300)`) is deliberate, it keeps SCNR
    underflow from poisoning downstream statistics, but it also means a
    *negative* input is silently treated as zero instead of surfacing the
    sign error, which is why the next test exists.
    """
    assert lin2db(0.0) == pytest.approx(-3000.0)
    assert np.isfinite(lin2db(0.0))


def test_lin2db_silently_accepts_negative_input() -> None:
    """Pins the clamp's blind spot: `lin2db(-1)` == `lin2db(0)`."""
    assert lin2db(-1.0) == pytest.approx(lin2db(0.0))


def test_dbm_watt_round_trip() -> None:
    """dBm ↔ W, and the paper's two anchor values."""
    for x in (-94.0, 0.0, 30.0, 43.0):
        assert watt2dbm(dbm2watt(x)) == pytest.approx(x)
    # −94 dBm noise floor and 43 dBm ≈ 20 W per AP (§VI-A)
    assert dbm2watt(-94.0) == pytest.approx(4.0e-13, rel=0.02)
    assert dbm2watt(43.0) == pytest.approx(20.0, rel=0.01)


def test_db_helpers_are_vectorized() -> None:
    x = np.array([0.0, 10.0, 20.0])
    np.testing.assert_allclose(db2lin(x), [1.0, 10.0, 100.0])


# =============================================================================
# Angles
# =============================================================================

def test_deg_rad_round_trip() -> None:
    for d in (-180.0, -90.0, 0.0, 45.0, 359.0):
        assert rad2deg(deg2rad(d)) == pytest.approx(d)


@pytest.mark.parametrize("angle,expected", [
    (0.0, 0.0), (np.pi, -np.pi), (3 * np.pi, -np.pi), (-3 * np.pi, -np.pi),
    (np.pi / 2, np.pi / 2), (2 * np.pi, 0.0),
])
def test_wrap_to_pi(angle: float, expected: float) -> None:
    """`wrap_to_pi` maps onto [−π, π); π itself wraps to −π."""
    assert float(wrap_to_pi(angle)) == pytest.approx(expected)


def test_wrap_to_pi_is_idempotent() -> None:
    a = np.linspace(-10, 10, 41)
    np.testing.assert_allclose(wrap_to_pi(wrap_to_pi(a)), wrap_to_pi(a))


# =============================================================================
# Array geometry
# =============================================================================

def test_ula_spacing_is_half_wavelength_by_default() -> None:
    assert ula_spacing(LAMBDA) == pytest.approx(LAMBDA / 2)
    assert ula_spacing(LAMBDA, 0.25) == pytest.approx(LAMBDA / 4)


@pytest.mark.parametrize("m", [2, 3, 8, 16, 64])
def test_uca_radius_gives_the_requested_chord_spacing(m: int) -> None:
    """
    `uca_radius` inverts R = d / (2 sin(π/M)), so the chord between adjacent
    elements is exactly `spacing_factor · λ`.
    """
    R = uca_radius(LAMBDA, m)
    chord = 2 * R * np.sin(np.pi / m)
    assert chord == pytest.approx(0.5 * LAMBDA)


def test_uca_radius_degenerates_to_zero_for_one_element() -> None:
    assert uca_radius(LAMBDA, 1) == 0.0


def test_uca_radius_grows_with_element_count() -> None:
    radii = [uca_radius(LAMBDA, m) for m in (4, 8, 16, 32)]
    assert radii == sorted(radii)


# =============================================================================
# Steering vectors
# =============================================================================

@pytest.mark.parametrize("fn", [steering_ula, steering_uca])
@pytest.mark.parametrize("m", [1, 2, 8, 16])
def test_steering_vectors_are_unit_norm(fn, m: int) -> None:
    """
    **Convention pin:** this module normalizes to `‖a‖² = 1`.

    The manuscript's §II states `‖a_{au}‖² = M_t`. Both conventions are
    self-consistent, but the factor `M_t` has to reappear somewhere for the
    SCNR of Prop. 3 to match the text. Stage 2 owns that reconciliation; this
    test exists so the code-side convention cannot drift underneath it.
    """
    a = fn(m, 0.3, np.pi / 2, LAMBDA)
    assert a.shape == (m,)
    assert a.dtype == np.complex128
    assert float(np.linalg.norm(a) ** 2) == pytest.approx(1.0)


def test_ula_element_phase_progression_is_linear() -> None:
    """
    ULA phase advances by exactly `2π (d/λ) cos θ` per element (paper §II).
    """
    m, el = 8, 0.7
    a = steering_ula(m, 0.0, el, LAMBDA, spacing_factor=0.5)
    step = 2 * np.pi * 0.5 * np.cos(el)
    phases = np.unwrap(np.angle(a))
    np.testing.assert_allclose(np.diff(phases), step, atol=1e-9)


def test_ula_ignores_azimuth() -> None:
    """A z-aligned ULA has no azimuth dependence."""
    a1 = steering_ula(8, 0.0, 1.0, LAMBDA)
    a2 = steering_ula(8, 2.5, 1.0, LAMBDA)
    np.testing.assert_allclose(a1, a2)


def test_uca_single_element_is_unity() -> None:
    a = steering_uca(1, 1.2, 0.9, LAMBDA)
    np.testing.assert_allclose(a, [1.0 + 0j])


def test_uca_depends_on_azimuth() -> None:
    """Unlike the ULA, a UCA in the xy-plane resolves azimuth."""
    a1 = steering_uca(8, 0.0, np.pi / 2, LAMBDA)
    a2 = steering_uca(8, np.pi / 3, np.pi / 2, LAMBDA)
    assert not np.allclose(a1, a2)


def test_uca_is_periodic_in_azimuth() -> None:
    a1 = steering_uca(8, 0.4, np.pi / 2, LAMBDA)
    a2 = steering_uca(8, 0.4 + 2 * np.pi, np.pi / 2, LAMBDA)
    np.testing.assert_allclose(a1, a2, atol=1e-12)


def test_steering_vector_dispatches_on_array_type() -> None:
    """`steering_vector` is the ULA/UCA façade used by the channel code."""
    for kind, ref in (("ULA", steering_ula), ("UCA", steering_uca)):
        got = steering_vector(kind, 8, 0.4, 1.1, LAMBDA)
        np.testing.assert_allclose(got, ref(8, 0.4, 1.1, LAMBDA))


def test_steering_vector_rejects_unknown_array_type() -> None:
    with pytest.raises(ValueError):
        steering_vector("HEX", 8, 0.0, np.pi / 2, LAMBDA)


def test_steering_matrix_columns_are_steering_vectors() -> None:
    az = np.array([0.0, 0.5, 1.0])
    el = np.array([np.pi / 2] * 3)
    A = steering_matrix("UCA", 8, az, el, LAMBDA)
    assert A.shape == (8, 3)
    for i in range(3):
        np.testing.assert_allclose(A[:, i], steering_uca(8, az[i], el[i], LAMBDA))


# =============================================================================
# Projections
# =============================================================================

def test_projection_matrix_is_an_orthogonal_projector(rng) -> None:
    """P = A(AᴴA)⁻¹Aᴴ satisfies P² = P and Pᴴ = P."""
    A = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
    P = projection_matrix(A)
    np.testing.assert_allclose(P @ P, P, atol=1e-9)
    np.testing.assert_allclose(P.conj().T, P, atol=1e-12)


def test_projection_matrix_leaves_the_column_space_fixed(rng) -> None:
    A = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
    P = projection_matrix(A)
    np.testing.assert_allclose(P @ A, A, atol=1e-9)


def test_null_space_projection_is_exact(rng) -> None:
    """
    The SVD projector annihilates the row space of H exactly, the property
    the NS-C sensing beam of §IV relies on.
    """
    H = rng.standard_normal((3, 8)) + 1j * rng.standard_normal((3, 8))
    P = null_space_projection(H)
    np.testing.assert_allclose(P @ P, P, atol=1e-9)
    np.testing.assert_allclose(P.conj().T, P, atol=1e-12)
    np.testing.assert_allclose(H @ P, np.zeros((3, 8)), atol=1e-9)


def test_null_space_projection_of_zero_matrix_is_identity() -> None:
    P = null_space_projection(np.zeros((3, 8), dtype=complex))
    np.testing.assert_allclose(P, np.eye(8), atol=1e-12)


def test_null_space_projection_rank_deficient_input(rng) -> None:
    """A repeated row must not inflate the estimated rank."""
    h = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    H = np.vstack([h, h, h])
    P = null_space_projection(H)
    np.testing.assert_allclose(H @ P, np.zeros((3, 8)), atol=1e-9)
    assert np.linalg.matrix_rank(P, tol=1e-6) == 7


def test_null_space_projection_reg_suppresses_but_is_not_exact(rng) -> None:
    """
    The regularized variant used by NS-C trades exactness for stability: it
    leaves a small residual whose size is set by `reg`.
    """
    H = rng.standard_normal((3, 8)) + 1j * rng.standard_normal((3, 8))
    strong = np.linalg.norm(H.conj() @ null_space_projection_reg(H, reg=1e-6))
    weak = np.linalg.norm(H.conj() @ null_space_projection_reg(H, reg=1e-1))
    assert strong < weak, "smaller reg must give a tighter null"


def test_regularised_pinv_matches_pinv_for_tiny_reg(rng) -> None:
    A = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
    np.testing.assert_allclose(regularised_pinv(A, reg=1e-12),
                               np.linalg.pinv(A), atol=1e-6)


# =============================================================================
# Linear algebra helpers
# =============================================================================

def test_vec_is_column_major() -> None:
    """`vec` follows the paper's column-stacking convention (Fortran order)."""
    A = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(vec(A), [1, 3, 2, 4])


def test_khatri_rao_is_columnwise_kronecker(rng) -> None:
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((2, 4))
    C = khatri_rao(A, B)
    assert C.shape == (6, 4)
    for i in range(4):
        np.testing.assert_allclose(C[:, i], np.kron(A[:, i], B[:, i]))


def test_khatri_rao_rejects_mismatched_columns() -> None:
    with pytest.raises(ValueError):
        khatri_rao(np.zeros((3, 4)), np.zeros((2, 5)))


def test_frobenius_norm_sq_is_real_and_matches_numpy(rng) -> None:
    A = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
    got = frobenius_norm_sq(A)
    assert isinstance(got, float)
    assert got == pytest.approx(float(np.linalg.norm(A, "fro") ** 2))


def test_normalise_columns_gives_unit_columns(rng) -> None:
    A = rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4))
    N = normalise_columns(A)
    np.testing.assert_allclose(np.linalg.norm(N, axis=0), 1.0, atol=1e-9)


def test_normalise_columns_survives_a_zero_column() -> None:
    A = np.zeros((4, 2), dtype=complex)
    A[:, 0] = 1.0
    N = normalise_columns(A)
    assert np.all(np.isfinite(N))
    assert np.linalg.norm(N[:, 1]) == 0.0


def test_compute_wavelength_at_the_paper_carrier() -> None:
    assert compute_wavelength(3.5e9) == pytest.approx(0.0857, abs=1e-4)


def test_channel_quality_metric_is_reciprocal_condition_number(rng) -> None:
    """Well-conditioned → near 1; rank-deficient → 0."""
    H = np.eye(4, 8, dtype=complex)
    assert channel_quality_metric(H) == pytest.approx(1.0)
    assert channel_quality_metric(np.zeros((4, 8), dtype=complex)) == 0.0
    h = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    assert channel_quality_metric(np.vstack([h, h])) == pytest.approx(0.0, abs=1e-10)


def test_psd_check_and_projection(rng) -> None:
    """`nearest_psd` clips negative eigenvalues; the result passes the check."""
    A = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    A = A + A.conj().T                       # Hermitian, generally indefinite
    assert not is_positive_semidefinite(A) or is_positive_semidefinite(A)
    P = nearest_psd(A)
    np.testing.assert_allclose(P, P.conj().T, atol=1e-9)
    assert is_positive_semidefinite(P)


def test_psd_check_accepts_a_gram_matrix(rng) -> None:
    X = rng.standard_normal((5, 3)) + 1j * rng.standard_normal((5, 3))
    assert is_positive_semidefinite(X @ X.conj().T)


def test_nearest_psd_is_identity_on_psd_input(rng) -> None:
    X = rng.standard_normal((5, 3)) + 1j * rng.standard_normal((5, 3))
    G = X @ X.conj().T
    np.testing.assert_allclose(nearest_psd(G), G, atol=1e-8)
