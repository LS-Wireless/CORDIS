"""
cordis/channel/estimation.py
============================
Linear MMSE uplink channel estimation for the CORDIS simulation framework.

Implements Proposition 2 (Section II-C) of the CORDIS journal paper.

TDD frame structure
-------------------
Each frame has τ_f samples:
    - τ_p  : uplink pilot phase  (channel estimation)
    - τ_d  : downlink ISAC phase (optimized by CORDIS algorithms)
    - τ_u  : uplink payload      (not analyzed in this work)

Pilot model
-----------
Users transmit uplink pilot sequences φ_u ∈ C^{τ_p}.  When τ_p ≥ N_ue,
all pilots are mutually orthogonal (no pilot contamination).  When
τ_p < N_ue, pilots must be reused, introducing contamination.

The received pilot signal at AP a is:
    Y_a^p = sqrt(P_p) Σ_k h_{ak} φ_k^T + N_a^p  ∈ C^{Mt × τ_p}

Projecting onto φ_u^*:
    y_{au}^p = sqrt(P_p) τ_p Σ_{k ∈ P_u} h_{ak} + n_{au}^p

MMSE estimate (Proposition 2)
------------------------------
Cross-covariance:
    D_{au} = sqrt(P_p) τ_p (R_{au} + Σ_{k ∈ P_u without u} ξ_{a,uk} B_a Γ_{a,uk} B_a^H)

Pilot observation covariance:
    Ψ_{au} = P_p τ_p^2 Σ_{l ∈ P_u} (R_{al} + cross-user terms)
             + σ_n^2 τ_p I_{Mt}

MMSE estimate:
    ĥ_{au} = D_{au} Ψ_{au}^{-1} y_{au}^p

Estimate covariance:
    R̂_{au} = D_{au} Ψ_{au}^{-1} D_{au}^H

Error covariance (used in SINR and beamformer design):
    R̃_{au} = R_{au} - R̂_{au}

Key outputs consumed by algorithms
-----------------------------------
EstimationResult.H_hat[ap_idx]     : np.ndarray (N_ue, Mt)  — H̃_a
EstimationResult.R_hat[(ap,ue)]    : np.ndarray (Mt, Mt)    — R̂_{au}
EstimationResult.R_tilde[(ap,ue)]  : np.ndarray (Mt, Mt)    — R̃_{au}
EstimationResult.R_tilde_H[ap_idx] : np.ndarray (Mt, Mt)    — Σ_u R̃_{au}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.utils.config import CORDISConfig
from cordis.utils.logger import get_logger
from cordis.utils.math_utils import nearest_psd
from cordis.channel.topology import NetworkTopology
from cordis.channel.pathloss import LargeScaleFading
from cordis.channel.rician import ChannelStatistics, ChannelRealization

logger = get_logger(__name__)


# =============================================================================
# Pilot sequence design
# =============================================================================

def design_pilot_sequences(
    n_ue: int,
    tau_p: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[NDArray[np.complex128], List[List[int]]]:
    """
    Design uplink pilot sequences and compute pilot contamination sets.

    Strategy
    --------
    - If τ_p ≥ N_ue:  assign mutually orthogonal pilots (DFT columns),
      no contamination, P_u = {u} for all u.
    - If τ_p < N_ue:  assign pilots cyclically (users u and u + τ_p share
      the same sequence), P_u = {u, u + τ_p, …}.

    Parameters
    ----------
    n_ue : int
        Number of users N_ue.
    tau_p : int
        Pilot length τ_p.
    rng : np.random.Generator or None
        Used only for randomizing pilot assignment when contamination occurs.
        If None, a default deterministic assignment is used.

    Returns
    -------
    Phi : np.ndarray, shape (n_ue, tau_p), dtype complex128
        Pilot matrix.  Row u is φ_u^T, so Phi[u] = φ_u.
        Satisfies Phi[u] @ Phi[k].conj() = τ_p if u, k share a pilot,
        else 0.
    contamination_sets : list[list[int]]
        contamination_sets[u] = P_u = list of user indices sharing a pilot
        with user u, including u itself.
    """
    # Build full DFT matrix of size τ_p for orthogonal pilots
    idx = np.arange(tau_p)
    dft = np.exp(-2j * np.pi * np.outer(idx, idx) / tau_p) / np.sqrt(tau_p)
    # Rows of dft are orthogonal; scale by sqrt(τ_p) for correct power
    pilots_full = dft * np.sqrt(tau_p)   # shape (tau_p, tau_p)

    # Assign pilots to users
    pilot_idx = np.arange(n_ue) % tau_p  # cyclic assignment
    if tau_p < n_ue and rng is not None:
        # Random permutation of base pilots
        perm = rng.permutation(tau_p)
        pilot_idx = perm[np.arange(n_ue) % tau_p]

    Phi = pilots_full[pilot_idx]   # (n_ue, tau_p)

    # Contamination sets
    contamination_sets: List[List[int]] = []
    for u in range(n_ue):
        P_u = [k for k in range(n_ue) if pilot_idx[k] == pilot_idx[u]]
        contamination_sets.append(P_u)

    n_contaminated = sum(len(P) > 1 for P in contamination_sets)
    if n_contaminated > 0:
        logger.debug(
            "Pilot contamination: τ_p=%d < N_ue=%d → %d users contaminated",
            tau_p, n_ue, n_contaminated,
        )
    return Phi.astype(np.complex128), contamination_sets


# =============================================================================
# MMSE estimation matrices  (D_{au}, Ψ_{au})
# =============================================================================

def _cross_user_term(
    B_a: NDArray[np.complex128],
    Gamma: NDArray[np.complex128],
    xi: float,
) -> NDArray[np.complex128]:
    """
    Compute cross-user covariance term: ξ_{a,uk} B_a Γ_{a,uk} B_a^H.

    Returns a zero matrix if B_a is empty (r = 0).
    """
    if B_a.shape[1] == 0:
        return np.zeros((B_a.shape[0], B_a.shape[0]), dtype=complex)
    return xi * (B_a @ Gamma @ B_a.conj().T)


def compute_mmse_matrices(
    u_idx: int,
    a_idx: int,
    stats: ChannelStatistics,
    contamination_sets: List[List[int]],
    tau_p: float,
    pilot_power_lin: float,
    sigma_n_sq: float,
    n_ant: int,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Compute the cross-covariance D_{au} and observation covariance Ψ_{au}
    for the MMSE estimator (Proposition 2).

    Parameters
    ----------
    u_idx : int
        User index.
    a_idx : int
        AP index.
    stats : ChannelStatistics
    contamination_sets : list[list[int]]
        P_u = contamination_sets[u].
    tau_p : float
        Pilot length τ_p.
    pilot_power_lin : float
        Pilot power P_p in linear scale.
    sigma_n_sq : float
        Noise variance σ_n^2 at the AP.
    n_ant : int
        Number of antennas Mt.

    Returns
    -------
    D_au : np.ndarray, shape (Mt, Mt)
    Psi_au : np.ndarray, shape (Mt, Mt)
    """
    key_uu = (a_idx, u_idx)
    R_au   = stats.R[key_uu]           # (Mt, Mt)
    B_a    = stats.B[a_idx]            # (Mt, r)
    P_u    = contamination_sets[u_idx]

    sqrt_Pp = np.sqrt(pilot_power_lin)

    # ── D_{au} (cross-covariance) ─────────────────────────────────────────
    # D_{au} = sqrt(P_p) τ_p (R_{au} + Σ_{k ∈ P_u without u} ξ_{a,uk} B_a Γ_{a,uk} B_a^H)
    D_au = R_au.copy()
    for k in P_u:
        if k == u_idx:
            continue
        # ξ_{a,uk}: inter-user cross-covariance scaling
        # Approximation: ξ = sqrt(β_{au} η^NLoS_{au} · β_{ak} η^NLoS_{ak})
        key_ak = (a_idx, k)
        xi_auk = np.sqrt(
            stats.eta_nlos[key_uu] * stats.eta_nlos.get(key_ak, 0.0)
            * np.real(np.trace(stats.R[key_uu]))
            * np.real(np.trace(stats.R.get(key_ak, np.zeros((n_ant, n_ant)))))
        ) / n_ant
        # Use identity for Γ_{a,uk} (simplification when cross-user
        # scattering structure is not fully specified)
        Gamma_auk = np.eye(B_a.shape[1]) if B_a.shape[1] > 0 else np.zeros((0, 0))
        D_au = D_au + _cross_user_term(B_a, Gamma_auk, xi_auk)
    D_au = sqrt_Pp * tau_p * D_au

    # ── Ψ_{au} (observation covariance) ───────────────────────────────────
    # Ψ_{au} = P_p τ_p^2 Σ_{l ∈ P_u} R_{al} + σ_n^2 τ_p I
    Psi_au = np.zeros((n_ant, n_ant), dtype=complex)
    for l in P_u:
        key_al = (a_idx, l)
        R_al = stats.R.get(key_al, np.zeros((n_ant, n_ant), dtype=complex))
        Psi_au = Psi_au + R_al
        # Add cross-user contamination terms from other pilots sharing l's pilot
        for k_prime in P_u:
            if k_prime == l:
                continue
            key_akp = (a_idx, k_prime)
            xi_lkp = np.sqrt(
                stats.eta_nlos.get(key_al, 0.0)
                * stats.eta_nlos.get(key_akp, 0.0)
                * np.real(np.trace(R_al))
                * np.real(np.trace(stats.R.get(key_akp, np.zeros((n_ant, n_ant)))))
            ) / n_ant
            Gamma = (np.eye(B_a.shape[1]) if B_a.shape[1] > 0
                     else np.zeros((0, 0)))
            Psi_au = Psi_au + _cross_user_term(B_a, Gamma, xi_lkp)

    Psi_au = pilot_power_lin * tau_p**2 * Psi_au + sigma_n_sq * tau_p * np.eye(n_ant)

    return D_au.astype(np.complex128), Psi_au.astype(np.complex128)


# =============================================================================
# EstimationResult container
# =============================================================================

@dataclass
class EstimationResult:
    """
    MMSE channel estimates and error covariances for all AP-UE pairs.

    Attributes
    ----------
    H_hat : dict[ap_idx, np.ndarray (N_ue, Mt)]
        Estimated channel matrix Ĥ_{a} at each transmit AP.
        Row u is ĥ_{au}^T.
    h_hat : dict[(ap_idx, ue_idx), np.ndarray (Mt,)]
        Individual estimated channel vectors ĥ_{au}.
    R_hat : dict[(ap_idx, ue_idx), np.ndarray (Mt, Mt)]
        Estimate covariance R̂_{au} = D_{au} Ψ_{au}^{-1} D_{au}^H.
    R_tilde : dict[(ap_idx, ue_idx), np.ndarray (Mt, Mt)]
        Error covariance R̃_{au} = R_{au} - R̂_{au}.
    R_tilde_H : dict[ap_idx, np.ndarray (Mt, Mt)]
        Aggregate error covariance at AP a: R̃_{H_a} = Σ_u R̃_{au}.
        Used in LR-MMSE precoder design.
    nmse : dict[(ap_idx, ue_idx), float]
        Normalised MSE: tr(R̃_{au}) / tr(R_{au}).
    """

    H_hat:    Dict[int, NDArray[np.complex128]]
    h_hat:    Dict[Tuple[int, int], NDArray[np.complex128]]
    R_hat:    Dict[Tuple[int, int], NDArray[np.complex128]]
    R_tilde:  Dict[Tuple[int, int], NDArray[np.complex128]]
    R_tilde_H: Dict[int, NDArray[np.complex128]]
    nmse:     Dict[Tuple[int, int], float]


# =============================================================================
# Main estimation function
# =============================================================================

def run_channel_estimation(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    lsf: LargeScaleFading,
    stats: ChannelStatistics,
    realization: ChannelRealization,
    rng: np.random.Generator,
    Phi: Optional[NDArray[np.complex128]] = None,
    contamination_sets: Optional[List[List[int]]] = None,
) -> EstimationResult:
    """
    Run linear MMSE channel estimation at all transmit APs.

    Parameters
    ----------
    topo        : NetworkTopology
    cfg         : CORDISConfig
    lsf         : LargeScaleFading
    stats       : ChannelStatistics   (second-order statistics)
    realization : ChannelRealization  (true channel vectors h_{au})
    rng         : np.random.Generator
    Phi         : pilot matrix (N_ue, tau_p).  Generated if None.
    contamination_sets : list of lists.  Generated from Phi if None.

    Returns
    -------
    EstimationResult
    """
    ch  = cfg.channel
    t   = cfg.topology
    frq = cfg.frequency

    n_ue  = topo.n_ue
    n_ant = t.n_ant
    tau_p = ch.tau_p

    # ── Noise variance and pilot power ────────────────────────────────────
    from cordis.channel.pathloss import noise_power_watts
    sigma_n_sq = noise_power_watts(frq.bandwidth_hz, ch.noise_figure_db,
                                   ch.noise_temp_k)
    # P_p relative to noise: pilot_power_db = 10 log10(P_p / σ_n^2)
    pilot_power_lin = (10.0 ** (ch.pilot_power_db / 10.0)) * sigma_n_sq

    # ── Pilot sequences ───────────────────────────────────────────────────
    if Phi is None or contamination_sets is None:
        Phi, contamination_sets = design_pilot_sequences(n_ue, tau_p, rng)

    # Shortcut for perfect CSI
    if ch.estimation_method == "perfect":
        return _perfect_csi_result(topo, stats, realization)

    # ── MMSE estimation at each transmit AP ───────────────────────────────
    H_hat_dict:    Dict[int, NDArray]            = {}
    h_hat_dict:    Dict[Tuple[int, int], NDArray] = {}
    R_hat_dict:    Dict[Tuple[int, int], NDArray] = {}
    R_tilde_dict:  Dict[Tuple[int, int], NDArray] = {}
    R_tilde_H_dict: Dict[int, NDArray]           = {}
    nmse_dict:     Dict[Tuple[int, int], float]  = {}

    for ap in topo.tx_aps:
        a_idx = ap.idx

        # ── Simulate pilot reception at this AP ───────────────────────────
        # Y_a^p = sqrt(P_p) Σ_k h_{ak} φ_k^T + N_a^p  ∈ C^{Mt × τ_p}
        Y_pilot = np.zeros((n_ant, tau_p), dtype=complex)
        sqrt_Pp = np.sqrt(pilot_power_lin)
        for ue in topo.ues:
            h_au = realization.h[(a_idx, ue.idx)]
            phi_u = Phi[ue.idx]                        # (tau_p,)
            Y_pilot += sqrt_Pp * np.outer(h_au, phi_u)

        # Add noise  N_a^p ~ CN(0, σ_n^2 I ⊗ I)
        noise_std = np.sqrt(sigma_n_sq / 2.0)
        N_pilot = (noise_std * rng.standard_normal((n_ant, tau_p))
                   + 1j * noise_std * rng.standard_normal((n_ant, tau_p)))
        Y_pilot += N_pilot

        h_hat_rows = []
        R_tilde_H  = np.zeros((n_ant, n_ant), dtype=complex)

        for ue in topo.ues:
            u_idx = ue.idx
            key   = (a_idx, u_idx)

            R_au = stats.R[key]

            if ch.estimation_method == "LS":
                # ── LS estimate: ĥ_{au} = Y_a^p φ_u^* / (sqrt(P_p) τ_p) ──
                phi_u = Phi[u_idx]
                y_pilot_u = Y_pilot @ phi_u.conj()     # (Mt,)
                h_hat_au  = y_pilot_u / (sqrt_Pp * tau_p)
                # LS error covariance  ≈ σ_n^2 / (P_p τ_p) I  (no pilot contam.)
                R_hat_au  = R_au - sigma_n_sq / (pilot_power_lin * tau_p) * np.eye(n_ant)
                R_hat_au  = nearest_psd(R_hat_au)
                R_tilde_au = nearest_psd(R_au - R_hat_au)

            else:
                # ── MMSE estimate (Proposition 2) ──────────────────────────
                D_au, Psi_au = compute_mmse_matrices(
                    u_idx, a_idx, stats, contamination_sets,
                    tau_p, pilot_power_lin, sigma_n_sq, n_ant,
                )

                # Sufficient statistic: y_{au}^p = Y_a^p φ_u^*  (Mt,)
                phi_u     = Phi[u_idx]
                y_pilot_u = Y_pilot @ phi_u.conj()    # (Mt,)

                # ĥ_{au} = D_{au} Ψ_{au}^{-1} y_{au}^p
                Psi_reg = Psi_au + 1e-10 * sigma_n_sq * np.eye(n_ant)
                L = np.linalg.solve(Psi_reg.conj().T, D_au.conj().T).conj().T
                # L = D_{au} Ψ_{au}^{-1}  (Mt, Mt)
                h_hat_au = L @ y_pilot_u

                # R̂_{au} = D_{au} Ψ_{au}^{-1} D_{au}^H
                R_hat_au = nearest_psd(L @ D_au.conj().T)

                # R̃_{au} = R_{au} − R̂_{au}
                R_tilde_au = nearest_psd(R_au - R_hat_au)

            # Store results
            h_hat_dict[key]   = h_hat_au.astype(np.complex128)
            R_hat_dict[key]   = R_hat_au.astype(np.complex128)
            R_tilde_dict[key] = R_tilde_au.astype(np.complex128)
            R_tilde_H        += R_tilde_au

            # NMSE
            tr_R = np.real(np.trace(R_au))
            tr_Rt = np.real(np.trace(R_tilde_au))
            nmse_dict[key] = float(tr_Rt / max(tr_R, 1e-20))

            h_hat_rows.append(h_hat_au)

        H_hat_dict[a_idx]    = np.vstack(h_hat_rows).astype(np.complex128)
        R_tilde_H_dict[a_idx] = nearest_psd(R_tilde_H).astype(np.complex128)

    mean_nmse = float(np.mean(list(nmse_dict.values())))
    logger.debug(
        "MMSE estimation complete: mean NMSE = %.2e  (method=%s)",
        mean_nmse, ch.estimation_method,
    )
    return EstimationResult(
        H_hat=H_hat_dict,
        h_hat=h_hat_dict,
        R_hat=R_hat_dict,
        R_tilde=R_tilde_dict,
        R_tilde_H=R_tilde_H_dict,
        nmse=nmse_dict,
    )


def _perfect_csi_result(
    topo: NetworkTopology,
    stats: ChannelStatistics,
    realization: ChannelRealization,
) -> EstimationResult:
    """
    Return an EstimationResult where ĥ_{au} = h_{au} (perfect CSI).
    R̃_{au} = 0, R̂_{au} = R_{au}.
    """
    H_hat_dict:    Dict[int, NDArray]            = {}
    h_hat_dict:    Dict[Tuple[int, int], NDArray] = {}
    R_hat_dict:    Dict[Tuple[int, int], NDArray] = {}
    R_tilde_dict:  Dict[Tuple[int, int], NDArray] = {}
    R_tilde_H_dict: Dict[int, NDArray]           = {}
    nmse_dict:     Dict[Tuple[int, int], float]  = {}

    n_ant = next(iter(stats.R.values())).shape[0]

    for ap in topo.tx_aps:
        a_idx  = ap.idx
        h_rows = []
        for ue in topo.ues:
            u_idx = ue.idx
            key   = (a_idx, u_idx)
            h_au  = realization.h[key]
            h_hat_dict[key]   = h_au.copy()
            R_hat_dict[key]   = stats.R[key].copy()
            R_tilde_dict[key] = np.zeros((n_ant, n_ant), dtype=complex)
            nmse_dict[key]    = 0.0
            h_rows.append(h_au)

        H_hat_dict[a_idx]     = np.vstack(h_rows).astype(np.complex128)
        R_tilde_H_dict[a_idx] = np.zeros((n_ant, n_ant), dtype=complex)

    logger.debug("Perfect CSI mode: R̃ = 0 for all links.")
    return EstimationResult(
        H_hat=H_hat_dict, h_hat=h_hat_dict,
        R_hat=R_hat_dict, R_tilde=R_tilde_dict,
        R_tilde_H=R_tilde_H_dict, nmse=nmse_dict,
    )

# =============================================================================
# Pilot diagnostics
# =============================================================================

def compute_pilot_diagnostics(
    topo,
    cfg,
    lsf,
) -> dict:
    """
    Compute the effective received pilot SNR and predicted NMSE for every
    transmit AP-UE link, without running the full estimation.

    This is the single most important sanity-check tool when setting up a
    new scenario: if ``snr_rx_db`` is below ~0 dB for any link, expect
    NMSE ≈ 1 and essentially random channel estimates.

    The effective received pilot SNR is:

        SNR_rx_{au} = (P_p / σ_n²) × τ_p × β_{au}

    and the predicted NMSE (scalar-channel approximation) is:

        NMSE ≈ 1 / (1 + SNR_rx)

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    lsf  : LargeScaleFading

    Returns
    -------
    dict with keys:
        "snr_rx_db"       : np.ndarray (N_tx, N_ue)  — received pilot SNR per link [dB]
        "nmse_predicted"  : np.ndarray (N_tx, N_ue)  — predicted NMSE per link
        "pilot_power_db"  : float      — P_p/σ_n² from config
        "sigma_n_sq"      : float      — absolute noise power [W]
        "pilot_power_lin" : float      — P_p in watts
        "tau_p"           : int
        "summary"         : str        — human-readable summary
    """
    from cordis.channel.pathloss import noise_power_watts

    ch    = cfg.channel
    frq   = cfg.frequency
    tau_p = ch.tau_p

    sigma_n_sq      = noise_power_watts(frq.bandwidth_hz, ch.noise_figure_db,
                                        ch.noise_temp_k)
    pilot_power_lin = (10.0 ** (ch.pilot_power_db / 10.0)) * sigma_n_sq

    n_tx = topo.n_tx
    n_ue = topo.n_ue

    snr_rx     = np.zeros((n_tx, n_ue))
    nmse_pred  = np.zeros((n_tx, n_ue))

    for row, ap in enumerate(topo.tx_aps):
        for col, ue in enumerate(topo.ues):
            beta = lsf.beta_lin[ap.idx, ue.idx]
            snr  = pilot_power_lin * tau_p * beta / sigma_n_sq
            snr_rx[row, col]    = snr
            nmse_pred[row, col] = 1.0 / (1.0 + snr)

    snr_rx_db = 10.0 * np.log10(np.maximum(snr_rx, 1e-30))

    # Build a human-readable summary
    lines = [
        "=== Pilot Diagnostics ===",
        f"  pilot_power_db  = {ch.pilot_power_db:.1f} dB  (P_p/σ_n²)",
        f"  sigma_n_sq      = {sigma_n_sq:.3e} W",
        f"  pilot_power_lin = {pilot_power_lin:.3e} W  "
        f"({10*np.log10(pilot_power_lin*1e3):.1f} dBm)",
        f"  tau_p           = {tau_p}",
        f"  SNR_rx range    = [{snr_rx_db.min():.1f}, {snr_rx_db.max():.1f}] dB",
        f"  NMSE predicted  = [{nmse_pred.min():.4f}, {nmse_pred.max():.4f}]",
    ]

    # Warn if estimation will be poor
    n_poor = int((nmse_pred > 0.5).sum())
    n_total = n_tx * n_ue
    if n_poor > 0:
        lines.append(
            f"  ⚠ WARNING: {n_poor}/{n_total} links have NMSE > 0.5 "
            f"(SNR_rx < 1 dB).  Increase pilot_power_db by at least "
            f"{max(0, -snr_rx_db.min()):.0f} dB."
        )
    else:
        lines.append(f"  ✓ All {n_total} links have SNR_rx > 0 dB.")

    return {
        "snr_rx_db":       snr_rx_db,
        "nmse_predicted":  nmse_pred,
        "pilot_power_db":  ch.pilot_power_db,
        "sigma_n_sq":      sigma_n_sq,
        "pilot_power_lin": pilot_power_lin,
        "tau_p":           tau_p,
        "summary":         "\n".join(lines),
    }

