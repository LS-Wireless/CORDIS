"""
cordis/utils/math_utils.py
==========================
Mathematical utility functions for the CORDIS simulation framework.

Contents
--------
- Steering vectors for ULA and UCA antenna arrays (3-D angles)
- Array geometry helpers (inter-element spacing, radius)
- Projection matrix construction (orthogonal, null-space)
- Regularized pseudo-inverse
- dB / linear conversion utilities
- Miscellaneous linear-algebra helpers used across the codebase

Conventions
-----------
Angles
    Azimuth  φ  ∈ (-π, π]   radians  (or degrees, converted internally)
    Elevation θ  ∈ [0, π]    radians  (3GPP zenith-angle convention)
                             θ = π/2 → broadside (horizontal)
                             θ = 0   → top,  θ = π → bottom

Array orientation
    ULA : elements placed along the z-axis with spacing ``d``.
    UCA : elements in the xy-plane; element 0 on the positive x-axis.

All returned steering vectors are column vectors of shape (M,) with
dtype ``np.complex128``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cordis.utils.logger import get_logger

logger = get_logger(__name__)

# ── Physical constants ─────────────────────────────────────────────────────────
SPEED_OF_LIGHT: float = 3.0e8   # m/s


# =============================================================================
# Angle conversion helpers
# =============================================================================

def deg2rad(deg: ArrayLike) -> NDArray[np.float64]:
    """Convert degrees to radians (thin wrapper kept for API consistency)."""
    return np.deg2rad(deg)


def rad2deg(rad: ArrayLike) -> NDArray[np.float64]:
    """Convert radians to degrees."""
    return np.rad2deg(rad)


def wrap_to_pi(angle_rad: ArrayLike) -> NDArray[np.float64]:
    """Wrap angle(s) to (-π, π]."""
    return (np.asarray(angle_rad) + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# dB / linear conversion
# =============================================================================

def db2lin(x_db: ArrayLike) -> NDArray[np.float64]:
    """Convert dB (power ratio) to linear scale: 10^(x/10)."""
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def lin2db(x_lin: ArrayLike) -> NDArray[np.float64]:
    """Convert linear power ratio to dB: 10 log10(x)."""
    x = np.asarray(x_lin, dtype=float)
    return 10.0 * np.log10(np.maximum(x, 1e-300))


def dbm2watt(x_dbm: ArrayLike) -> NDArray[np.float64]:
    """Convert dBm to Watts."""
    return db2lin(np.asarray(x_dbm, dtype=float) - 30.0)


def watt2dbm(x_watt: ArrayLike) -> NDArray[np.float64]:
    """Convert Watts to dBm."""
    return lin2db(x_watt) + 30.0


# =============================================================================
# Array geometry helpers
# =============================================================================

def ula_spacing(wavelength: float, spacing_factor: float = 0.5) -> float:
    """
    Return ULA inter-element spacing in metres.

    Parameters
    ----------
    wavelength : float
        Carrier wavelength in metres.
    spacing_factor : float
        Normalized spacing d/λ.  Default ``0.5`` (half-wavelength).

    Returns
    -------
    float
        Spacing d in metres.
    """
    return spacing_factor * wavelength


def uca_radius(wavelength: float, n_elements: int,
               spacing_factor: float = 0.5) -> float:
    """
    Return the UCA radius R such that the arc length between adjacent
    elements equals ``spacing_factor * wavelength``.

    For a UCA with M elements the arc-chord relationship gives

        R = d / (2 sin(π / M))

    Parameters
    ----------
    wavelength : float
        Carrier wavelength in metres.
    n_elements : int
        Number of antenna elements M.
    spacing_factor : float
        Normalised inter-element spacing d/λ.  Default ``0.5``.

    Returns
    -------
    float
        Radius R in metres.
    """
    if n_elements == 1:
        return 0.0
    d = spacing_factor * wavelength
    return d / (2.0 * np.sin(np.pi / n_elements))


# =============================================================================
# Steering vectors
# =============================================================================

def steering_ula(
    n_elements: int,
    az_rad: float,
    el_rad: float,
    wavelength: float,
    spacing_factor: float = 0.5,
) -> NDArray[np.complex128]:
    """
    Steering vector for a **Uniform Linear Array (ULA)** aligned with the
    z-axis.

    The phase progression along the array is determined by the projection of
    the unit direction vector onto the z-axis:

        a[m] = exp( j 2π m (d/λ) cos(θ) ),   m = 0, …, M−1

    where θ is the zenith (elevation) angle measured from the positive z-axis.
    The azimuth angle φ does not affect the response of a z-aligned ULA.

    Parameters
    ----------
    n_elements : int
        Number of antenna elements M.
    az_rad : float
        Azimuth angle φ in radians.  **Not used** for a z-aligned ULA but
        kept in the signature for API consistency with ``steering_uca``.
    el_rad : float
        Zenith (elevation) angle θ ∈ [0, π] in radians.
        θ = π/2 corresponds to the broadside direction.
    wavelength : float
        Carrier wavelength in metres (λ = c / f_c).
    spacing_factor : float
        Normalised inter-element spacing d/λ.  Default ``0.5``.

    Returns
    -------
    np.ndarray, shape (M,), dtype complex128
        Unit-norm steering vector.

    Examples
    --------
    >>> import numpy as np
    >>> from cordis.utils.math_utils import steering_ula
    >>> lam = 3e8 / 3.5e9          # 3.5 GHz carrier
    >>> a = steering_ula(8, 0.0, np.pi/2, lam)   # broadside
    >>> np.allclose(np.abs(a), 1.0 / np.sqrt(8))
    True
    """
    m = np.arange(n_elements, dtype=float)
    d_over_lam = spacing_factor  # d/λ
    # Phase shift per element: 2π (d/λ) cos(θ)
    phase = 2.0 * np.pi * d_over_lam * np.cos(el_rad) * m
    a = np.exp(1j * phase) / np.sqrt(n_elements)
    return a.astype(np.complex128)


def steering_uca(
    n_elements: int,
    az_rad: float,
    el_rad: float,
    wavelength: float,
    spacing_factor: float = 0.5,
) -> NDArray[np.complex128]:
    """
    Steering vector for a **Uniform Circular Array (UCA)** in the xy-plane.

    Element m is placed at azimuth angle φ_m = 2π m / M from the positive
    x-axis.  The phase of element m for an incoming plane wave from direction
    (φ, θ) is

        a[m] = exp( j 2π (R/λ) sin(θ) cos(φ − 2πm/M) ),  m = 0, …, M−1

    where R is the array radius determined by ``spacing_factor`` and M.

    Parameters
    ----------
    n_elements : int
        Number of antenna elements M.
    az_rad : float
        Azimuth angle φ ∈ (−π, π] in radians.
    el_rad : float
        Zenith angle θ ∈ [0, π] in radians.  θ = π/2 is the horizon.
    wavelength : float
        Carrier wavelength λ in metres.
    spacing_factor : float
        Normalized arc-chord inter-element spacing d/λ.  Default ``0.5``.

    Returns
    -------
    np.ndarray, shape (M,), dtype complex128
        Unit-norm steering vector.

    Notes
    -----
    For M = 1 the radius is 0 and the steering vector is identically 1.

    Examples
    --------
    >>> import numpy as np
    >>> from cordis.utils.math_utils import steering_uca
    >>> lam = 3e8 / 3.5e9
    >>> a = steering_uca(8, 0.0, np.pi/2, lam)
    >>> np.allclose(np.abs(a), 1.0 / np.sqrt(8))
    True
    """
    if n_elements == 1:
        return np.array([1.0 + 0j], dtype=np.complex128)

    m = np.arange(n_elements, dtype=float)
    phi_m = 2.0 * np.pi * m / n_elements              # element azimuth angles

    R = uca_radius(wavelength, n_elements, spacing_factor)
    R_over_lam = R / wavelength

    # sin(θ) projects the array onto the horizontal plane
    phase = 2.0 * np.pi * R_over_lam * np.sin(el_rad) * np.cos(az_rad - phi_m)
    a = np.exp(1j * phase) / np.sqrt(n_elements)
    return a.astype(np.complex128)


def steering_vector(
    array_type: str,
    n_elements: int,
    az_rad: float,
    el_rad: float,
    wavelength: float,
    spacing_factor: float = 0.5,
) -> NDArray[np.complex128]:
    """
    Unified dispatcher for array steering vectors.

    Parameters
    ----------
    array_type : {"ULA", "UCA"}
        Array geometry.
    n_elements : int
        Number of antenna elements M.
    az_rad : float
        Azimuth angle in radians.
    el_rad : float
        Zenith (elevation) angle in radians.
    wavelength : float
        Carrier wavelength in metres.
    spacing_factor : float
        Normalized inter-element spacing d/λ.

    Returns
    -------
    np.ndarray, shape (M,), dtype complex128
        Unit-norm steering vector.

    Raises
    ------
    ValueError
        If ``array_type`` is not recognized.
    """
    atype = array_type.upper()
    if atype == "ULA":
        return steering_ula(n_elements, az_rad, el_rad, wavelength, spacing_factor)
    if atype == "UCA":
        return steering_uca(n_elements, az_rad, el_rad, wavelength, spacing_factor)
    raise ValueError(
        f"Unknown array_type '{array_type}'.  Choose 'ULA' or 'UCA'."
    )


def steering_matrix(
    array_type: str,
    n_elements: int,
    az_rad_list: ArrayLike,
    el_rad_list: ArrayLike,
    wavelength: float,
    spacing_factor: float = 0.5,
) -> NDArray[np.complex128]:
    """
    Build a steering matrix A ∈ ℂ^{M × K} from K direction pairs.

    Parameters
    ----------
    array_type : {"ULA", "UCA"}
    n_elements : int
    az_rad_list : array-like, shape (K,)
        Azimuth angles in radians.
    el_rad_list : array-like, shape (K,)
        Zenith angles in radians.
    wavelength : float
    spacing_factor : float

    Returns
    -------
    np.ndarray, shape (M, K), dtype complex128
        Columns are individual steering vectors.
    """
    az = np.atleast_1d(az_rad_list)
    el = np.atleast_1d(el_rad_list)
    if az.shape != el.shape:
        raise ValueError(
            f"az_rad_list and el_rad_list must have the same length; "
            f"got {az.shape} and {el.shape}."
        )
    cols = [
        steering_vector(array_type, n_elements, a, e, wavelength, spacing_factor)
        for a, e in zip(az, el)
    ]
    return np.column_stack(cols).astype(np.complex128)


# =============================================================================
# Projection matrix utilities
# =============================================================================

def projection_matrix(
    A: NDArray[np.complex128],
    reg: float = 0.0,
) -> NDArray[np.complex128]:
    """
    Orthogonal projection matrix onto the **column space** of A.

        P = A (A^H A + reg I)^{-1} A^H

    Parameters
    ----------
    A : np.ndarray, shape (M, K)
        Basis matrix whose columns span the subspace.
    reg : float
        Tikhonov regularization coefficient (default 0 — no regularization).
        Should be > 0 if A is rank-deficient or poorly conditioned.

    Returns
    -------
    np.ndarray, shape (M, M), dtype complex128
    """
    M, K = A.shape
    AhA = A.conj().T @ A + reg * np.eye(K, dtype=complex)
    return (A @ np.linalg.solve(AhA, A.conj().T)).astype(np.complex128)


def null_space_projection(
    H: NDArray[np.complex128],
    rank_tol: float = 1e-6,
) -> NDArray[np.complex128]:
    """
    **Exact** orthogonal projection onto the null space of H via SVD.

        P_perp = I_M − V_r V_r^H

    where V_r contains the right singular vectors of H corresponding to
    non-negligible singular values (i.e. the columns that span the row
    space of H).

    Parameters
    ----------
    H : np.ndarray, shape (N_ue, M)
        Channel matrix at one AP: rows are UE channels, columns are antennas.
    rank_tol : float
        Singular values below ``rank_tol × σ_max`` are treated as zero
        when determining the numerical rank of H.

    Returns
    -------
    np.ndarray, shape (M, M), dtype complex128
        True orthogonal projector; satisfies P^2 = P and P^H = P.

    Notes
    -----
    For the **regularized** approximation used in the NS-C beamformer
    (eq. 12 of the CORDIS paper), see :func:`null_space_projection_reg`.
    That version is intentionally *not* an exact projector but avoids
    an explicit SVD and is more numerically stable when H is rank-deficient.
    """
    _, S, Vh = np.linalg.svd(H, full_matrices=False)   # Vh: (min(N,M), M)
    M = H.shape[1]
    if S[0] < 1e-12:
        # H is all zeros — null space is the full space
        return np.eye(M, dtype=complex)
    rank = int(np.sum(S > rank_tol * S[0]))
    # V_r: columns spanning row space of H, shape (M, rank)
    V_r = Vh[:rank, :].conj().T
    P_row = V_r @ V_r.conj().T           # (M, M)  projection onto row space
    return (np.eye(M, dtype=complex) - P_row).astype(np.complex128)


def null_space_projection_reg(
    H: NDArray[np.complex128],
    reg: float = 1e-3,
) -> NDArray[np.complex128]:
    """
    **Regularized** null-space projection used in the NS-C beamformer
    (CORDIS paper, eq. 12):

        P_perp ≈ I_M − H^H (H H^H + ε I)^{-1} H

    This is the formula used at each AP in practice.  It is NOT an exact
    projector (P^2 ≠ P in general) but avoids an explicit SVD, handles
    rank-deficient channels gracefully through the regularization, and
    converges to the exact null-space projector as ε → 0.

    Parameters
    ----------
    H : np.ndarray, shape (N_ue, M)
    reg : float
        Regularization coefficient ε (default 1e-3).

    Returns
    -------
    np.ndarray, shape (M, M), dtype complex128
    """
    N_ue, M = H.shape
    # H H^H + ε I :  (N_ue, N_ue)
    HHh = H @ H.conj().T + reg * np.eye(N_ue, dtype=complex)
    # H^H (H H^H + ε I)^{-1} H : (M, M)
    P_approx = H.conj().T @ np.linalg.solve(HHh, H)
    return (np.eye(M, dtype=complex) - P_approx).astype(np.complex128)


def regularised_pinv(
    A: NDArray[np.complex128],
    reg: float = 1e-6,
) -> NDArray[np.complex128]:
    """
    Regularised pseudo-inverse (A^H A + reg I)^{-1} A^H.

    Parameters
    ----------
    A : np.ndarray, shape (M, K)
    reg : float
        Regularization coefficient λ.  Should scale with noise variance
        when used as an MMSE-type estimator.

    Returns
    -------
    np.ndarray, shape (K, M), dtype complex128
    """
    _, K = A.shape
    return np.linalg.solve(
        A.conj().T @ A + reg * np.eye(K, dtype=complex),
        A.conj().T,
    ).astype(np.complex128)


# =============================================================================
# Condition-number-based channel quality metric M(a)
# =============================================================================

def channel_quality_metric(H: NDArray[np.complex128]) -> float:
    """
    Compute the channel quality metric M(a) = 1 / κ(H_a) used in SplitOpt
    for the LM-RZF beamformer weight α_a.

    A well-conditioned channel (small condition number κ) yields a large
    M(a), giving AP a more responsibility in the BF design.

    Parameters
    ----------
    H : np.ndarray, shape (N_ue, M)
        Channel matrix H_a at AP a.

    Returns
    -------
    float
        Reciprocal condition number ∈ (0, 1].  Returns 0.0 for a zero matrix.
    """
    if np.allclose(H, 0):
        return 0.0
    singular_values = np.linalg.svd(H, compute_uv=False)
    if singular_values[0] < 1e-12:
        return 0.0
    return float(singular_values[-1] / singular_values[0])   # σ_min / σ_max


# =============================================================================
# General linear-algebra utilities
# =============================================================================

def vec(A: NDArray) -> NDArray:
    """Vectorise matrix A column-by-column (Fortran order)."""
    return A.flatten(order="F")


def khatri_rao(A: NDArray, B: NDArray) -> NDArray:
    """
    Khatri-Rao (column-wise Kronecker) product of A ∈ ℂ^{m×k} and
    B ∈ ℂ^{n×k}.

    Returns C ∈ ℂ^{mn×k} where C[:, i] = kron(A[:, i], B[:, i]).
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"A and B must have the same number of columns; "
            f"got {A.shape[1]} and {B.shape[1]}."
        )
    return np.vstack([
        np.kron(A[:, i], B[:, i]) for i in range(A.shape[1])
    ]).T


def frobenius_norm_sq(A: NDArray) -> float:
    """Return ‖A‖_F² (squared Frobenius norm)."""
    return float(np.real(np.sum(A * A.conj())))


def normalise_columns(A: NDArray, eps: float = 1e-12) -> NDArray:
    """
    Normalize each column of A to unit ℓ2-norm.

    Parameters
    ----------
    A : np.ndarray, shape (M, K)
    eps : float
        Small value added to the norm to avoid division by zero.

    Returns
    -------
    np.ndarray, shape (M, K)
    """
    norms = np.linalg.norm(A, axis=0, keepdims=True) + eps
    return A / norms


def compute_wavelength(freq_hz: float) -> float:
    """
    Compute carrier wavelength λ = c / f.

    Parameters
    ----------
    freq_hz : float
        Carrier frequency in Hz.

    Returns
    -------
    float
        Wavelength in metres.
    """
    return SPEED_OF_LIGHT / freq_hz


def is_positive_semidefinite(A: NDArray, tol: float = 1e-8) -> bool:
    """
    Check whether a Hermitian matrix is positive semidefinite.

    Uses the minimum eigenvalue as the test criterion.
    """
    eigvals = np.linalg.eigvalsh(A)
    return bool(eigvals.min() >= -tol)


def nearest_psd(A: NDArray) -> NDArray:
    """
    Project a Hermitian matrix onto the cone of positive semidefinite
    matrices by zeroing out negative eigenvalues.

    Useful for covariance matrices that become slightly indefinite due to
    floating-point accumulation.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)

    Returns
    -------
    np.ndarray, shape (N, N), dtype complex128
    """
    A_sym = (A + A.conj().T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_clipped = np.maximum(eigvals, 0.0)
    return (eigvecs * eigvals_clipped) @ eigvecs.conj().T

