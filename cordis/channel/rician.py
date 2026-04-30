"""
cordis/channel/rician.py
========================
Spatially correlated Rician fading channel generation for the CORDIS
simulation framework.

Implements the channel model from Section II-A of the CORDIS journal paper:

    h_{au} = sqrt(β_{au}) (h_{au}^LoS + h_{au}^NLoS)

    h_{au}^LoS  = sqrt(η_{au}^LoS)  e^{jφ_{au}} a_{au}
    h_{au}^NLoS = sqrt(η_{au}^NLoS) (C_{au}^{1/2} q_{au} + B_a s_{au})

where:
    η_{au}^LoS  = K_{au} / (1 + K_{au})   (LoS power fraction)
    η_{au}^NLoS = 1       / (1 + K_{au})   (NLoS power fraction)
    φ_{au}  ~ Uniform[-π, π]              (random LoS phase)
    q_{au}  ~ CN(0, I_{Mt})               (independent Rayleigh component)
    B_a     ∈ C^{Mt × r}                  (shared scattering subspace, r ≪ Mt)
    s_{au}  ~ CN(0, Σ_{au})               (user-specific scattering coefficients)

The per-AP spatial correlation matrix is:
    R_{au} = β_{au} (η^LoS a_{au} a_{au}^H + η^NLoS C̃_{au})
where C̃_{au} = C_{au} + B_a Σ_{au} B_a^H,  tr(C̃_{au}) = Mt.

The NLoS spatial correlation matrix C_{au} is computed via Monte Carlo
integration over a truncated Laplacian Power Angular Spectrum (PAS):
    P(φ | μ_φ, σ_φ) ∝ exp(-√2 |φ - μ_φ|_wrap / σ_φ)

Key outputs
-----------
ChannelStatistics
    Second-order statistics: R_{au}, C_{au}, η^LoS, η^NLoS, K for every
    AP-UE pair.  These are computed once per topology realization and
    reused across Monte Carlo trials with the same large-scale fading.

ChannelRealization
    Actual channel vectors h_{au} ∈ C^{Mt} for all AP-UE pairs in a
    single coherence block.  New realizations are drawn for each trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.utils.config import CORDISConfig
from cordis.utils.logger import get_logger
from cordis.utils.math_utils import (
    steering_vector, compute_wavelength, nearest_psd
)
from cordis.channel.topology import NetworkTopology
from cordis.channel.pathloss import LargeScaleFading

logger = get_logger(__name__)


# =============================================================================
# Truncated Laplacian PAS sampler
# =============================================================================

def _sample_laplacian_angles(
    rng: np.random.Generator,
    n: int,
    mean_deg: float,
    std_deg: float,
    clip_range_deg: float = 180.0,
) -> NDArray[np.float64]:
    """
    Draw n angle samples from a truncated Laplacian distribution.

    The Laplacian scale parameter b = std / √2  (so that Var = 2b² = std²).
    Samples are drawn via the closed-form inverse CDF and then clipped to
    [mean - clip_range, mean + clip_range] before wrapping to (-π, π].

    Parameters
    ----------
    rng : np.random.Generator
    n : int
        Number of samples.
    mean_deg, std_deg : float
        Mean and standard deviation in degrees.
    clip_range_deg : float
        Hard clip around the mean in degrees.

    Returns
    -------
    np.ndarray, shape (n,), in radians
    """
    if std_deg < 1e-6:
        return np.full(n, np.deg2rad(mean_deg))

    b = std_deg / np.sqrt(2.0)    # Laplacian scale parameter [degrees]

    # Inverse CDF sampling: u ~ Uniform(0,1)
    # X = -b sign(u-0.5) ln(1 - 2|u-0.5|)
    u = rng.uniform(0.0, 1.0, size=n)
    sign = np.sign(u - 0.5)
    sign[sign == 0] = 1.0
    x_deg = -b * sign * np.log(1.0 - 2.0 * np.abs(u - 0.5) + 1e-12)

    # Shift by mean and clip
    x_deg = np.clip(x_deg, -clip_range_deg, clip_range_deg) + mean_deg

    # Wrap to (-180, 180] degrees, then convert to radians
    x_deg = ((x_deg + 180.0) % 360.0) - 180.0
    return np.deg2rad(x_deg)


# =============================================================================
# Spatial correlation matrix  C_{au}
# =============================================================================

def compute_spatial_correlation(
    array_type: str,
    n_ant: int,
    az_mean_deg: float,
    el_mean_deg: float,
    az_std_deg: float,
    el_std_deg: float,
    wavelength: float,
    spacing_factor: float,
    n_samples: int,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """
    Compute the NLoS spatial correlation matrix C_{au} ∈ C^{Mt × Mt} via
    Monte Carlo integration over the truncated Laplacian PAS.

        [C_{au}]_{m,n} = E_{φ,θ}[ a_m(φ,θ) a_n^*(φ,θ) ]

    where the expectation is taken over both azimuth and elevation angles
    drawn independently from truncated Laplacian distributions.

    The result is normalized so that tr(C_{au}) = Mt, matching the
    CORDIS paper convention.

    Parameters
    ----------
    array_type : {"ULA", "UCA"}
    n_ant : int
        Number of antenna elements Mt.
    az_mean_deg, el_mean_deg : float
        Mean azimuth and elevation (zenith) angles in degrees.
        Convention: azimuth ∈ (-180, 180], elevation ∈ [0, 180]
        (θ = 90° is broadside / horizontal).
    az_std_deg, el_std_deg : float
        Angular spread standard deviations in degrees.
    wavelength : float
        Carrier wavelength λ in metres.
    spacing_factor : float
        Normalised inter-element spacing d/λ.
    n_samples : int
        Number of Monte Carlo integration samples.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (Mt, Mt), dtype complex128
        Positive semidefinite spatial correlation matrix, tr(C) = Mt.
    """
    # Sample azimuth and elevation angles from truncated Laplacian
    az_rad = _sample_laplacian_angles(rng, n_samples, az_mean_deg, az_std_deg)
    el_rad = _sample_laplacian_angles(
        rng, n_samples, el_mean_deg, el_std_deg,
        clip_range_deg=89.0    # keep elevation in (1°, 179°)
    )
    # Clip elevation to valid zenith range [1°, 179°]
    el_rad = np.clip(el_rad, np.deg2rad(1.0), np.deg2rad(179.0))

    # Build steering matrix — shape (Mt, n_samples)
    A = np.column_stack([
        np.sqrt(n_ant) *    # undo unit-norm normalisation in steering_vector
        steering_vector(array_type, n_ant, az_rad[i], el_rad[i],
                        wavelength, spacing_factor)
        for i in range(n_samples)
    ])  # shape (Mt, n_samples),  columns are un-normalised steering vectors

    # Monte Carlo estimate:  C ≈ (1/N) A A^H
    C = (A @ A.conj().T) / n_samples    # (Mt, Mt)

    # Ensure exact Hermitian symmetry (suppress floating-point asymmetry)
    C = (C + C.conj().T) / 2.0

    # Normalise so tr(C) = Mt
    tr = np.real(np.trace(C))
    if tr > 1e-12:
        C *= n_ant / tr

    # Project onto PSD cone to suppress numerical negativity
    C = nearest_psd(C)
    return C.astype(np.complex128)


# =============================================================================
# LoS / NLoS power fractions from K-factor
# =============================================================================

def los_nlos_fractions(k_factor_db: float) -> Tuple[float, float]:
    """
    Convert Rician K-factor (dB) to LoS / NLoS power fractions.

        η^LoS  = K / (1 + K)
        η^NLoS = 1 / (1 + K)

    Parameters
    ----------
    k_factor_db : float
        Rician K-factor in dB.  Set to -inf (or very negative) for
        Rayleigh fading (η^LoS = 0).

    Returns
    -------
    eta_los, eta_nlos : float
        Power fractions satisfying η^LoS + η^NLoS = 1.
    """
    if k_factor_db < -100:
        return 0.0, 1.0
    k_lin = 10.0 ** (k_factor_db / 10.0)
    eta_los  = k_lin / (1.0 + k_lin)
    eta_nlos = 1.0   / (1.0 + k_lin)
    return eta_los, eta_nlos


# =============================================================================
# Shared scattering subspace  B_a
# =============================================================================

def generate_shared_subspace(
    rng: np.random.Generator,
    n_ant: int,
    rank: int,
) -> NDArray[np.complex128]:
    """
    Generate the shared scattering subspace matrix B_a ∈ C^{Mt × r}.

    B_a is drawn as a random partial isometry (orthonormal columns) to
    represent a common scattering cluster visible from all UEs served by AP a.
    The columns are orthonormal: B_a^H B_a = I_r.

    Parameters
    ----------
    rng : np.random.Generator
    n_ant : int
        Number of antenna elements Mt.
    rank : int
        Subspace rank r.  Use r = 0 to disable shared scattering.

    Returns
    -------
    np.ndarray, shape (Mt, r), dtype complex128.
        Returns an empty (Mt, 0) matrix if rank == 0.
    """
    if rank <= 0:
        return np.zeros((n_ant, 0), dtype=complex)
    G = (rng.standard_normal((n_ant, rank))
         + 1j * rng.standard_normal((n_ant, rank))) / np.sqrt(2)
    Q, _ = np.linalg.qr(G)   # orthonormal columns
    return Q[:, :rank].astype(np.complex128)


# =============================================================================
# Per-AP channel statistics (second-order, computed once per topology)
# =============================================================================

@dataclass
class ChannelStatistics:
    """
    Second-order channel statistics for all AP-UE pairs in one topology.

    These quantities depend only on the topology and large-scale fading
    (which are fixed for a given coherence block set) and are therefore
    computed once and reused across many channel realizations.

    Attributes
    ----------
    R : dict[(ap_idx, ue_idx), np.ndarray shape (Mt, Mt)]
        Full spatial correlation matrix R_{au} = β_{au}(η^LoS a a^H + η^NLoS C̃).
        Satisfies tr(R_{au}) = β_{au} Mt.
    C : dict[(ap_idx, ue_idx), np.ndarray shape (Mt, Mt)]
        NLoS spatial correlation matrix C_{au}.  tr(C_{au}) = Mt.
    R_tilde_C : dict[(ap_idx, ue_idx), np.ndarray shape (Mt, Mt)]
        C̃_{au} = C_{au} + B_a Σ_{au} B_a^H.
    a : dict[(ap_idx, ue_idx), np.ndarray shape (Mt,)]
        LoS steering vector a_{au} (unit-norm × √Mt).
    eta_los : dict[(ap_idx, ue_idx), float]
        LoS power fraction η^LoS_{au}.
    eta_nlos : dict[(ap_idx, ue_idx), float]
        NLoS power fraction η^NLoS_{au}.
    B : dict[ap_idx, np.ndarray shape (Mt, r)]
        Shared scattering subspace B_a at each AP.
    """

    R:          Dict[Tuple[int, int], NDArray[np.complex128]]
    C:          Dict[Tuple[int, int], NDArray[np.complex128]]
    R_tilde_C:  Dict[Tuple[int, int], NDArray[np.complex128]]
    a:          Dict[Tuple[int, int], NDArray[np.complex128]]
    eta_los:    Dict[Tuple[int, int], float]
    eta_nlos:   Dict[Tuple[int, int], float]
    B:          Dict[int, NDArray[np.complex128]]


def compute_channel_statistics(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    lsf: LargeScaleFading,
    rng: np.random.Generator,
) -> ChannelStatistics:
    """
    Pre-compute second-order channel statistics for all transmit AP-UE pairs.

    This function is called once per topology realisation.  The outputs
    (R_{au}, C_{au}, etc.) are passed to the channel estimator and
    the beamforming algorithms.

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    lsf  : LargeScaleFading   (large-scale fading β_{au}, LoS state)
    rng  : np.random.Generator

    Returns
    -------
    ChannelStatistics
    """
    ch  = cfg.channel
    t   = cfg.topology
    frq = cfg.frequency

    lam  = frq.wavelength_m
    n_ant_tx = t.n_ant              # Mt (transmit APs)
    array_type = t.array_type
    spacing_factor = t.antenna_spacing_factor
    n_samples = ch.n_spatial_samples
    r_shared  = ch.shared_scatter_rank

    # Lookup tables keyed by (ap_idx, ue_idx)
    R_dict:         Dict[Tuple[int, int], NDArray] = {}
    C_dict:         Dict[Tuple[int, int], NDArray] = {}
    R_tilde_C_dict: Dict[Tuple[int, int], NDArray] = {}
    a_dict:         Dict[Tuple[int, int], NDArray] = {}
    eta_los_dict:   Dict[Tuple[int, int], float]   = {}
    eta_nlos_dict:  Dict[Tuple[int, int], float]   = {}
    B_dict:         Dict[int, NDArray] = {}

    for ap in topo.tx_aps:
        a_idx = ap.idx

        # ── Shared scattering subspace B_a ────────────────────────────────
        B_a = generate_shared_subspace(rng, n_ant_tx, r_shared)
        B_dict[a_idx] = B_a

        for ue in topo.ues:
            u_idx = ue.idx
            key   = (a_idx, u_idx)

            beta  = lsf.beta_lin[a_idx, u_idx]     # large-scale fading
            is_los = lsf.los_state[a_idx, u_idx]   # LoS / NLoS state

            # ── K-factor and power fractions ──────────────────────────────
            if is_los:
                k_db = rng.normal(ch.rician_k_db_mean, ch.rician_k_db_std)
            else:
                k_db = -200.0       # Rayleigh fading for NLoS links
            eta_l, eta_n = los_nlos_fractions(k_db)
            eta_los_dict[key]  = eta_l
            eta_nlos_dict[key] = eta_n

            # ── AoD/AoA from AP to UE ─────────────────────────────────────
            diff_3d = ue.pos - ap.pos            # (3,)
            dist_2d = np.linalg.norm(diff_3d[:2]) + 1e-9
            dist_3d = np.linalg.norm(diff_3d)    + 1e-9

            az_rad = np.arctan2(diff_3d[1], diff_3d[0])   # azimuth φ
            el_rad = np.arccos(np.clip(diff_3d[2] / dist_3d, -1, 1))  # zenith θ

            # ── LoS steering vector a_{au} (un-normalised, ‖a‖² = Mt) ──────
            a_unit = steering_vector(array_type, n_ant_tx,
                                     az_rad, el_rad, lam, spacing_factor)
            a_au = np.sqrt(n_ant_tx) * a_unit     # ‖a_{au}‖² = Mt
            a_dict[key] = a_au

            # ── NLoS spatial correlation matrix C_{au} ────────────────────
            az_std  = (ch.as_azimuth_deg_std_los   if is_los
                       else ch.as_azimuth_deg_std_nlos)
            el_std  = (ch.as_elevation_deg_std_los  if is_los
                       else ch.as_elevation_deg_std_nlos)

            C_au = compute_spatial_correlation(
                array_type, n_ant_tx,
                np.rad2deg(az_rad), np.rad2deg(el_rad),
                az_std, el_std,
                lam, spacing_factor, n_samples, rng,
            )
            C_dict[key] = C_au

            # ── Shared-subspace contribution (r > 0) ─────────────────────
            # Σ_{au} = (power fraction) × I_r  (isotropic per-user variance)
            if r_shared > 0:
                sigma_au = ch.shared_scatter_power * np.eye(r_shared)
                B_contrib = B_a @ sigma_au @ B_a.conj().T     # (Mt, Mt)
                # Ensure tr(C̃_{au}) = Mt
                C_tilde = C_au + B_contrib
                C_tilde *= n_ant_tx / np.real(np.trace(C_tilde))
            else:
                C_tilde = C_au

            R_tilde_C_dict[key] = C_tilde

            # ── Full spatial correlation matrix R_{au} ────────────────────
            # R_{au} = β_{au} (η^LoS a a^H + η^NLoS C̃_{au})
            R_au = beta * (eta_l * np.outer(a_au, a_au.conj())
                           + eta_n * C_tilde)
            # Enforce tr(R_{au}) = β_{au} Mt  (numerical cleanup)
            R_au = nearest_psd(R_au)
            R_dict[key] = R_au

    logger.debug(
        "Channel statistics computed: %d AP-UE pairs, "
        "shared_scatter_rank=%d",
        len(R_dict), r_shared,
    )
    return ChannelStatistics(
        R=R_dict, C=C_dict, R_tilde_C=R_tilde_C_dict,
        a=a_dict, eta_los=eta_los_dict, eta_nlos=eta_nlos_dict,
        B=B_dict,
    )


# =============================================================================
# Channel realisation
# =============================================================================

@dataclass
class ChannelRealization:
    """
    One coherence-block realisation of all AP-UE channel vectors.

    Attributes
    ----------
    h : dict[(ap_idx, ue_idx), np.ndarray shape (Mt,)]
        True channel vectors h_{au} = sqrt(β_{au})(h^LoS + h^NLoS).
    H_mat : dict[ap_idx, np.ndarray shape (N_ue, Mt)]
        Channel matrix H_{a} = [h_{a1}; …; h_{a,N_ue}] at each TX AP.
    stats : ChannelStatistics
        Reference to the second-order statistics used to generate this
        realisation (needed by the estimator).
    """

    h:     Dict[Tuple[int, int], NDArray[np.complex128]]
    H_mat: Dict[int, NDArray[np.complex128]]
    stats: ChannelStatistics


def generate_channel_realization(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    lsf: LargeScaleFading,
    stats: ChannelStatistics,
    rng: np.random.Generator,
) -> ChannelRealization:
    """
    Draw one coherence-block realisation of all AP-UE channel vectors.

    For each AP-UE pair (a, u):
        h^LoS_{au}  = sqrt(η^LoS) e^{jφ} a_{au}
        h^NLoS_{au} = sqrt(η^NLoS) (C_{au}^{1/2} q_{au} + B_a s_{au})
        h_{au}      = sqrt(β_{au}) (h^LoS + h^NLoS)

    where φ ~ Uniform[-π, π], q_{au} ~ CN(0, I), s_{au} ~ CN(0, Σ_{au}).

    Parameters
    ----------
    topo  : NetworkTopology
    cfg   : CORDISConfig
    lsf   : LargeScaleFading
    stats : ChannelStatistics  (pre-computed second-order statistics)
    rng   : np.random.Generator

    Returns
    -------
    ChannelRealization
    """
    ch    = cfg.channel
    t     = cfg.topology
    n_ant = t.n_ant
    r_shared = ch.shared_scatter_rank

    h_dict:    Dict[Tuple[int, int], NDArray] = {}
    H_mat_dict: Dict[int, NDArray]            = {}

    for ap in topo.tx_aps:
        a_idx = ap.idx
        B_a   = stats.B[a_idx]          # (Mt, r)  or (Mt, 0)
        h_rows = []

        for ue in topo.ues:
            u_idx = ue.idx
            key   = (a_idx, u_idx)

            beta   = lsf.beta_lin[a_idx, u_idx]
            eta_l  = stats.eta_los[key]
            eta_n  = stats.eta_nlos[key]
            a_au   = stats.a[key]           # (Mt,), ‖a‖² = Mt
            C_au   = stats.C[key]           # (Mt, Mt), tr = Mt

            # ── LoS component ─────────────────────────────────────────────
            phi = rng.uniform(-np.pi, np.pi)
            h_los = np.sqrt(eta_l) * np.exp(1j * phi) * a_au

            # ── NLoS component: C_{au}^{1/2} q_{au} ──────────────────────
            # Use eigendecomposition for the matrix square root
            eigvals, eigvecs = np.linalg.eigh(C_au)
            eigvals_sqrt = np.sqrt(np.maximum(eigvals, 0.0))
            C_sqrt = eigvecs * eigvals_sqrt     # C^{1/2} (Mt, Mt)

            q_au = (rng.standard_normal(n_ant)
                    + 1j * rng.standard_normal(n_ant)) / np.sqrt(2.0)
            h_nlos = np.sqrt(eta_n) * C_sqrt @ q_au

            # ── Shared scattering contribution B_a s_{au} ─────────────────
            if r_shared > 0:
                s_au = (rng.standard_normal(r_shared)
                        + 1j * rng.standard_normal(r_shared)) / np.sqrt(2.0)
                # Scale: shared_scatter_power fraction of NLoS power
                s_au *= np.sqrt(ch.shared_scatter_power)
                h_nlos = h_nlos + np.sqrt(eta_n) * (B_a @ s_au)

            # ── Full channel vector ───────────────────────────────────────
            h_au = np.sqrt(beta) * (h_los + h_nlos)
            h_dict[key] = h_au.astype(np.complex128)
            h_rows.append(h_au)

        # ── Channel matrix at AP a:  H_a ∈ C^{N_ue × Mt} ─────────────────
        H_mat_dict[a_idx] = np.vstack(h_rows).astype(np.complex128)

    logger.debug(
        "Channel realisation generated: %d TX APs × %d UEs",
        len(H_mat_dict), topo.n_ue,
    )
    return ChannelRealization(h=h_dict, H_mat=H_mat_dict, stats=stats)
