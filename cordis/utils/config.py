"""
cordis/utils/config.py
======================
Configuration management for the CORDIS simulation framework.

Design
------
All simulation parameters are stored in JSON files under ``configs/``.
A *base* config (``configs/default.json``) defines every parameter with
sensible defaults.  An *experiment* config (e.g.
``configs/exp_pareto.json``) may override **only** the fields that differ
from the base — the loader performs a deep merge so that every key is
always populated.

The merged JSON is validated and converted to a tree of Python
*dataclasses*, giving full IDE auto-completion, type checking, and clear
error messages when a field is missing or has the wrong type.

Usage
-----
    from cordis.utils.config import load_config
    cfg = load_config("configs/default.json", "configs/exp_pareto.json")
    print(cfg.topology.n_ap)           # 6
    print(cfg.algorithm.admm.rho)      # 1.0

Argument-parser integration
---------------------------
    cfg = load_config(args.config, args.exp_config)
    # Override individual fields from CLI flags:
    if args.n_ap is not None:
        cfg.topology.n_ap = args.n_ap
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cordis.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Dataclass definitions — one per JSON block
# =============================================================================

@dataclass
class TopologyConfig:
    """Spatial layout of APs, UEs, and targets."""

    # Layout
    type: str = "circle"            # "circle" | "random" | "grid"
    ap_radius_m: float = 650.0      # AP placement circle radius [m]
    ue_max_radius_m: float = 1000.0 # Max UE placement radius [m]
    ue_min_radius_m: float = 35.0   # Min UE placement radius [m]
    tg_max_radius_m: float = 1000.0 # Max target placement radius [m]
    tg_min_radius_m: float = 35.0   # Min target placement radius [m]

    # Minimum pairwise separation (Stage 24).  0.0 disables (default),
    # preserving the original area-uniform draw and every existing seed's
    # topology; a positive value switches that entity to a rejection sampler.
    ue_min_separation_m: float = 0.0  # min UE-UE 2-D distance [m]
    ue_target_min_separation_m: float = 0.0  # min UE-target 2-D distance [m]

    # Network dimensions
    n_ap: int = 6                   # Total number of APs
    n_ue: int = 4                   # Number of communication UEs
    n_targets: int = 1              # Number of sensing targets

    # AP hardware
    n_ant: int = 10                 # Antennas per AP (M)
    n_rf_chains: int = 10           # RF chains per AP (≤ n_ant)
    array_type: str = "UCA"         # "ULA" | "UCA"
    antenna_spacing_factor: float = 0.5   # d/λ

    # Multi-static sensing
    n_sensing_rx: int = 1           # Number of dedicated receive APs

    # Heights [m] — used for 3-D path loss
    h_ap_m: float = 10.0           # AP antenna height
    h_ue_m: float = 1.5            # UE antenna height
    h_tg_m: float = 1.5            # Target height

    # Grid layout (used only when type == "grid")
    grid_n_rows: int = 3
    grid_n_cols: int = 2
    grid_spacing_m: float = 433.0


@dataclass
class FrequencyConfig:
    """Carrier and waveform parameters."""

    carrier_freq_ghz: float = 3.5   # Carrier frequency [GHz]
    bandwidth_mhz: float = 20.0     # Total bandwidth [MHz]
    n_subcarriers: int = 64         # OFDM subcarriers (N_sc)
    rb_n_subcarriers: int = 12      # Subcarriers per RB (Q)
    rb_n_symbols: int = 14          # OFDM symbols per RB (K)
    subcarrier_spacing_khz: float = 15.0   # Δf [kHz]
    cp_fraction: float = 0.0714     # Cyclic prefix length / OFDM symbol

    @property
    def carrier_freq_hz(self) -> float:
        return self.carrier_freq_ghz * 1e9

    @property
    def bandwidth_hz(self) -> float:
        return self.bandwidth_mhz * 1e6

    @property
    def wavelength_m(self) -> float:
        return 3e8 / self.carrier_freq_hz

    @property
    def subcarrier_spacing_hz(self) -> float:
        return self.subcarrier_spacing_khz * 1e3


@dataclass
class ChannelConfig:
    """Channel model and propagation parameters (journal paper, Section II-A)."""

    model: str = "3gpp_umi_rician"  # channel model identifier
    scenario: str = "UMi"           # "UMi" | "UMa" | "RMa"
    environment: str = "StreetCanyon"

    # ── Power budget ──────────────────────────────────────────────────────
    snr_db: float = 137.0           # P_max / σ²_n [dB] — ≈ 20 W per AP
                                    # at B=20 MHz, NF=7 dB, T=290 K
                                    # (σ²_n ≈ -94 dBm noise floor)
                                    # Matches upper end of 5G mMIMO mid-band
                                    # micro/small-cell conducted power (2-20 W)
    noise_figure_db: float = 7.0    # Receiver noise figure [dB]
    noise_temp_k: float = 290.0     # Thermal noise temperature [K]

    # ── TDD frame structure ───────────────────────────────────────────────
    tau_f: int = 200                # Frame size τ_f [samples]
    tau_p: int = 10                 # Pilot length τ_p (≥ N_ue for no contamination)
    tau_d: int = 100                # Downlink ISAC symbols τ_d
    # τ_u = τ_f - τ_p - τ_d (uplink payload, not optimised here)

    # ── Pilot power ───────────────────────────────────────────────────────
    # IMPORTANT — scaling: pilot_power_db = 10 log10(P_p / σ_n²)
    # where P_p is the uplink pilot transmit power in watts and σ_n² is the
    # absolute thermal noise power (computed from bandwidth and noise figure).
    #
    # The RECEIVED pilot SNR at the AP — which actually drives estimation
    # quality — is:
    #   SNR_rx_{au} = (P_p / σ_n²) × τ_p × β_{au}
    #
    # With 3GPP UMi path loss, β_{au} << 1, so pilot_power_db must be set
    # high enough to compensate:
    #
    #   ap_radius 50 m  → β ≈ 1e-8  → need pilot_power_db ≥ 80 dB
    #   ap_radius 200 m → β ≈ 3e-9  → need pilot_power_db ≥ 95 dB
    #   ap_radius 650 m → β ≈ 7e-11 → need pilot_power_db ≥ 110 dB
    #
    # A value of 120 dB corresponds to P_p ≈ 200 mW (23 dBm) — physically
    # realistic for a UE uplink pilot.  Values of 10–30 dB are only
    # meaningful for toy (no-path-loss) channel models.
    pilot_power_db: float = 120.0   # P_p / σ_n²  [dB]

    # ── Shadow fading ─────────────────────────────────────────────────────
    shadow_fading_los_std_db: float = 4.0    # σ_SF LoS [dB]
    shadow_fading_nlos_std_db: float = 7.82  # σ_SF NLoS [dB]
    shadow_corr_distance_m: float = 50.0     # D_corr [m]

    # ── Rician K-factor (3GPP TR 38.901 Table 7.5-6) ─────────────────────
    rician_k_db_mean: float = 9.0   # Mean K [dB]
    rician_k_db_std: float = 5.0    # STD of K [dB]

    # ── NLoS spatial correlation — truncated Laplacian PAS ───────────────
    # η^LoS = K/(1+K),  η^NLoS = 1/(1+K)
    as_azimuth_deg_std_los: float = 5.0      # ASD σ_φ LoS [°]
    as_azimuth_deg_std_nlos: float = 22.5    # ASD σ_φ NLoS [°]
    as_elevation_deg_std_los: float = 3.0    # ESD σ_θ LoS [°]
    as_elevation_deg_std_nlos: float = 7.0   # ESD σ_θ NLoS [°]
    n_spatial_samples: int = 2000            # Monte Carlo samples for C_{au}

    # ── Inter-user spatial correlation (shared scattering subspace B_a) ──
    # Captured by B_a ∈ C^{Mt × r} with r ≪ Mt (eq. comm-channel-covariance)
    shared_scatter_rank: int = 0    # r = 0 disables inter-user correlation
    shared_scatter_power: float = 0.1  # Fraction of NLoS power in shared subspace

    # ── Estimation method ─────────────────────────────────────────────────
    estimation_method: str = "MMSE" # "MMSE" | "LS" | "perfect"

    # ── mmWave extensions (active when carrier_freq_ghz > 6) ─────────────
    mmwave_n_clusters: int = 2
    mmwave_n_rays: int = 10
    mmwave_cluster_as_deg: float = 10.0


@dataclass
class SensingConfig:
    """
    Sensing channel and target parameters (journal paper, Section II-B).

    Target channel:
        H_{a_t a_r}^tgt = s_t sqrt(β_{a_t a_r}^tgt) ζ_t a_{a_r} a_{a_t}^H
    Clutter channel (separable space-time):
        H_{a_t a_r}^clt[τ] = σ_clt C_{a_r}^{1/2} Q[τ] C_{a_t}^{1/2}
    SCNR (expected STAP output, Proposition 3):
        SCNR_{a_r} = σ_RCS^2 T Σ_{a_t} β_{a_t a_r}^tgt ||a_{a_t}^H W_{a_t}||^2
                     × (a_{a_r}^H R_{g_{a_r}}^{-1} a_{a_r})
    where:
        R_{g_{a_r}} = σ_clt^2 Σ_{a_t} tr(W_{a_t}^H C_{a_t} W_{a_t}) C_{a_r}
                      + σ_{n,a_r}^2 I
    """

    # ── Target RCS (Swerling-I: ζ_t ~ CN(0, σ_RCS^2)) ────────────────────
    sigma_rcs_sq_db: float = -3.0   # σ_RCS^2 in dB (RCS power variance)
    swerling_model: int = 1         # 0 = deterministic RCS, 1 = Swerling-I

    # ── Receive AP assignment strategy ────────────────────────────────────
    rx_strategy: str = "single_closest_centroid"
    # Options:
    #   "single_closest_centroid"  – one Rx AP, closest to target centroid
    #   "single_farthest_centroid" – one Rx AP, farthest from AP centroid
    #   "per_target_closest"       – one dedicated Rx AP per target
    #   "assent_file"              – load pre-computed ASSENT decision
    #   "assent_live"              – call ASSENT model at runtime

    # ── Stochastic LoS availability (binary s_t) ─────────────────────────
    los_model: str = "always"       # "3gpp_umi" | "always" | "deterministic"
    # Default is "always" because the paper's target channel model
    # (eq. sensing-channel-target) is rank-one LoS-only: s_t ∈ {0,1}
    # with no NLoS contribution.  Under stochastic UMi LoS at urban
    # ranges, most (AP, target) links are NLoS → s_t = 0 → zero target
    # echo → SCNR underflows to the -300 dB sentinel.  "always" forces
    # s_t = 1 for all bistatic links and matches the assumption made
    # in the paper's SCNR derivation (Proposition 3).
    los_probability_override: Optional[float] = None  # Fixed P_LoS if not None

    # ── Multi-static sensing ──────────────────────────────────────────────
    multi_static: bool = True
    max_tx_aps_per_target: int = 5  # K_tx cap (|A_t| association limit)
    max_rx_aps_per_target: int = 3  # K_rx cap

    # ── STAP parameters (Proposition 3) ──────────────────────────────────
    n_snapshots: int = 20           # T = number of slow-time sensing symbols

    # ── Clutter model (eq. clutter-channel) ──────────────────────────────
    # Two ways to set σ_clt²:
    #   • ``clutter_cnr_db``  → σ_clt² = 10^(CNR/10) × σ_n²  (default, recommended)
    #   • ``sigma_clt``       → σ_clt² = sigma_clt²          (explicit override)
    # If ``sigma_clt`` is None (default) the CNR-based formula is used.
    # Set sigma_clt to a positive float to bypass CNR and pin σ_clt to a
    # noise-independent value (useful for unit tests and power-fixed studies).
    sigma_clt: Optional[float] = None     # Direct σ_clt override; None ⇒ use CNR
    clutter_cnr_db: float = -10.0         # Clutter-to-noise ratio [dB]
    # Temporal correlation of clutter (ρ_clt(Δτ)):
    rho_clt_model: str = "constant" # "constant" (ρ=1) | "jakes" | "gaussian"
    rho_clt_bandwidth: float = 0.1  # Normalised clutter Doppler bandwidth

    # ── Spatial correlation at clutter direction ──────────────────────────
    clutter_as_deg: float = 15.0    # Angular spread of clutter returns [°]

    # ── Clutter direction placement ───────────────────────────────────────
    # "target_centroid": clutter PAS centred on AP-to-target-centroid
    #     direction.  Clutter and target are spatially aligned, so the
    #     clutter penalty term ‖C^{1/2} W‖² and the sensing gradient
    #     |a_t^H W|² have negatively correlated gradients — high κ pushes
    #     the beam away from the target, collapsing both SCNR and SINR
    #     for any UE the target shadows.  Default for backward compat.
    # "offset":  clutter PAS centred at  (target azimuth + clutter_offset_az_deg)
    #     so it is decoupled from the target direction.  Use this when
    #     you want κ to trade off clutter avoidance against sensing gain
    #     without geometric coupling to comm coverage.
    # "uniform": clutter PAS spans the full azimuth (clutter_as_deg
    #     effectively → 180°), giving an isotropic C ≈ (P_clt/M)·I.
    #     The clutter penalty becomes a global power regulariser; κ no
    #     longer steers the beam.
    # "random":  per-AP random azimuth drawn from U[-π, π] using the
    #     supplied RNG.  Useful for Monte-Carlo over clutter geometries.
    clutter_center_strategy: str = "target_centroid"
    clutter_offset_az_deg:   float = 60.0       # used by "offset" strategy


@dataclass
class ADMMConfig:
    """
    Hyper-parameters for CORDIS-ADMM (Algorithm 2, journal paper).

    The ADMM iterates between:
      - Parallel local AP updates  → P-Local  (QCQP per AP via CVXPY)
      - Central CPU update         → P-Central (SOC projection per user)

    The objective is the clutter-aware linear sensing surrogate
    (eq. admm-linear-objective), subject to per-user SINR SOC constraints.
    There is NO λ trade-off parameter — communication is a hard constraint.
    """

    # ── Sensing objective ─────────────────────────────────────────────────
    kappa: float = 0.1              # Clutter penalty κ ≥ 0 (eq. admm-linear-objective)

    # ── ADMM penalty ──────────────────────────────────────────────────────
    rho: float = 1.0                # ADMM penalty parameter ρ

    # ── Termination ───────────────────────────────────────────────────────
    n_max: int = 250                 # Maximum ADMM iterations N_max
    eps_pri: float = 1.0            # Primal residual tolerance ε_pri
    eps_dual: float = 1.0           # Dual residual tolerance ε_dual
    # With the auto-balanced ρ (rho=1), the primal/dual residuals settle
    # near 1 in SNR-amplitude units, so tolerances around 0.3–1.0 match
    # the algorithm's natural equilibrium.  Setting these below ~0.1
    # typically requires rho > 1, which can destabilise SCA.

    # ── P-Central slack (eq. admm-p-central) ─────────────────────────────
    xi_slack: float = 1e4           # Slack penalty ξ ≫ 0 for P-Central

    # ── Target priority weights ───────────────────────────────────────────
    target_priority_equal: bool = True   # Use ω_t = 1 for all targets
    # If False, weights must be supplied at runtime as a list of length N_targets

    # ── SCA linearisation ─────────────────────────────────────────────────
    # The sensing utility ||a^H W||^2 is convexified via first-order Taylor
    # around W^(n); the approximation is re-linearised at every iteration.

    # ── Initialization ────────────────────────────────────────────────────
    warm_start_from_split: bool = True  # Initialise W^(0) from CORDIS-Split Phase I

    # ── Best-iterate selection (Stage 22a) ────────────────────────────────
    # When the algorithm runs to ``n_max`` without satisfying the
    # primal/dual residual tolerances, it returns the best W seen so far.
    # Two criteria are supported for selecting "best":
    #
    #   "feasible_then_residual" (default):
    #    Pick the feasible iterate with the smallest residual norm.
    #
    #   "residual_norm" (previous default):
    #       Pick the iterate with the smallest combined primal-plus-dual
    #       consensus residual.  Mathematically the most natural choice
    #       since these residuals measure how close the iterate is to
    #       satisfying the ADMM optimality conditions.  Robust to the
    #       late-iteration oscillation around the SOC boundary that
    #       fixed-ρ ADMM exhibits: the residual minimum lives in the
    #       converged plateau, not in a swing-peak iteration that
    #       happens to have high min-SINR by phase luck.
    #
    #   "min_sinr" (legacy):
    #       Pick the iterate with the highest worst-user SINR, ties
    #       broken by lower primal residual.  Maximises a quantity the
    #       user can interpret directly, but on this algorithm it tends
    #       to select swinging iterates from the post-convergence
    #       oscillation region whose consensus is not actually tight,
    #       so the reported W can fail to deliver its claimed SINR in
    #       downstream evaluation.
    best_iter_criterion: str = "feasible_then_residual"

    # ── Adaptive ρ (Stage 22b) ────────────────────────────────────────────
    # Boyd-Parikh-Chu (2011) §3.4.1 adaptive penalty parameter scheme.
    # Rebalances primal vs dual residual by scaling rho_admm within
    # [rho_min_factor, rho_max_factor].  Bounds and step size are
    # deliberately conservative because the SCA-linearisation of the
    # err penalty in P-Local loses validity outside a trust region
    # around W^(n) — empirically, scaling rho_admm by ≥5× the input
    # value destabilises the algorithm.  We cap at 3× and use τ=1.5
    # to creep up slowly.
    adaptive_rho: bool = False
    rho_mu_balance: float = 10.0
    rho_tau: float = 1.5
    rho_max_factor: float = 3.0
    rho_min_factor: float = 0.5
    rho_adapt_warmup: int = 5
    rho_adapt_interval: int = 3

    # ── Patience-based early stopping (Stage 22b) ─────────────────────────
    # Active only under best_iter_criterion="residual_norm".  Bails
    # out when no new best iterate has been recorded for
    # `early_stop_patience` consecutive iterations, after a warmup
    # of `early_stop_min_iters`.  Set patience=0 to disable.
    early_stop_patience: int = 30
    early_stop_min_iters: int = 30

    # ── Solver ────────────────────────────────────────────────────────────
    solver: str = "CLARABEL"        # CVXPY solver: "CLARABEL" | "GUROBI" | "MOSEK"


@dataclass
class SplitOptConfig:
    """
    Hyper-parameters for CORDIS-Split (Algorithm 1, journal paper).

    Phase I  — Distributed BF at each AP:
        Communication : LR-MMSE (Local Robust MMSE, eq. split-comm-bf)
        Sensing       : NS-C with priority-aware projection allocation

    Phase II — Centralized PA at CPU:
        Solves P-Split (eq. split-pa-opt) via CVXPY interior-point.
        Objective : clutter-aware linear sensing surrogate
        Constraint: per-user SINR constraints SINR_u ≥ γ_u

    Note
    ----
    γ_u (the per-user SINR target) lives at
    ``cfg.algorithm.gamma_db`` (umbrella level) so it is shared with
    ADMM, Centralized, and benchmarks.  This dataclass keeps
    Split-specific knobs only.
    """

    # ── LR-MMSE (communication BF, eq. split-comm-bf) ────────────────────
    epsilon_reg: float = 1e-2       # Regularisation ε in LR-MMSE precoder

    # ── NS-C (sensing BF, null-space projection) ─────────────────────────
    epsilon_nsc: float = 1e-3       # Loading factor ε for null-space projector P̂_perp

    # ── Priority-aware projection allocation ─────────────────────────────
    target_priority_equal: bool = True   # Use ω_t = 1 for all targets

    # ── P-Split objective (eq. split-pa-opt) ─────────────────────────────
    kappa: float = 1.0              # Clutter regularisation κ ≥ 0
    xi_penalty: float = 1e4         # Slack penalty ξ ≫ 0 for QoS violations

    # ── Solver ────────────────────────────────────────────────────────────
    solver: str = "CLARABEL"        # CVXPY solver for P-Split


@dataclass
class AlgorithmConfig:
    """
    Container for all algorithm-specific configs.

    ``gamma_db`` is the per-user minimum SINR target γ_u applied
    uniformly across users.  It is shared by every algorithm in the
    framework — Split, ADMM, Centralized, and the local benchmarks
    — because the point of comparing algorithms is to evaluate them
    against the *same* QoS requirement.  For sweep studies that want
    per-spec heterogeneity, the spec builders in
    :mod:`cordis.experiments.specs` accept a ``gamma_u_db`` kwarg that
    overrides this default.
    """

    gamma_db: float = 10.0          # Minimum SINR requirement γ_u [dB]

    admm: ADMMConfig = field(default_factory=ADMMConfig)
    split: SplitOptConfig = field(default_factory=SplitOptConfig)


@dataclass
class SimulationConfig:
    """Monte Carlo simulation control."""

    n_trials: int = 500             # Number of independent realisations
    seed: int = 42                  # Master random seed
    n_jobs: int = -1                # Parallel workers (-1 = all cores)
    save_dir: str = "results/"      # Output directory
    tag: str = ""                   # Label appended to filenames

    # Which metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "sinr_cdf", "scnr_cdf", "fronthaul", "convergence"
    ])

    # Benchmarks to run alongside the primary algorithm
    benchmarks: List[str] = field(default_factory=lambda: [
        "centralized", "mrt_centpa", "zf_centpa", "random_bf"
    ])

    # Verbosity
    log_level: str = "INFO"
    log_file: Optional[str] = None
    progress_bar: bool = True


@dataclass
class CORDISConfig:
    """
    Top-level configuration object.  Every field maps to a JSON block.

    Attributes
    ----------
    topology   : TopologyConfig
    frequency  : FrequencyConfig
    channel    : ChannelConfig
    sensing    : SensingConfig
    algorithm  : AlgorithmConfig
    simulation : SimulationConfig
    """

    topology: TopologyConfig = field(default_factory=TopologyConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the config back to a plain dictionary."""
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        """Write the config to a JSON file."""
        from cordis.utils.paths import get_project_root
        p = Path(path)
        if not p.is_absolute():
            p = get_project_root() / p
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug("Config saved to %s", p)

    def validate(self) -> None:
        """
        Run basic sanity checks and raise ``ValueError`` for invalid
        combinations of parameters.
        """
        t = self.topology

        if t.n_rf_chains > t.n_ant:
            raise ValueError(
                f"n_rf_chains ({t.n_rf_chains}) cannot exceed "
                f"n_ant ({t.n_ant})."
            )
        if t.n_sensing_rx >= t.n_ap:
            raise ValueError(
                f"n_sensing_rx ({t.n_sensing_rx}) must be < n_ap ({t.n_ap}) "
                f"to leave at least one transmit AP."
            )
        if t.array_type.upper() not in {"ULA", "UCA"}:
            raise ValueError(
                f"Unknown array_type '{t.array_type}'.  Use 'ULA' or 'UCA'."
            )
        if not (0.0 < t.antenna_spacing_factor <= 1.0):
            raise ValueError(
                f"antenna_spacing_factor must be in (0, 1]; "
                f"got {t.antenna_spacing_factor}."
            )
        if t.ue_min_separation_m < 0:
            raise ValueError("topology.ue_min_separation_m must be >= 0.")
        if t.ue_target_min_separation_m < 0:
            raise ValueError("topology.ue_target_min_separation_m must be >= 0.")
        if self.algorithm.admm.kappa < 0:
            raise ValueError("admm.kappa (clutter penalty) must be ≥ 0.")
        if self.algorithm.split.kappa < 0:
            raise ValueError("split.kappa (clutter penalty) must be ≥ 0.")
        if self.channel.estimation_method not in {"MMSE", "LS", "perfect"}:
            raise ValueError(
                f"Unknown estimation_method "
                f"'{self.channel.estimation_method}'."
            )
        if self.simulation.n_trials < 1:
            raise ValueError("n_trials must be ≥ 1.")

        logger.debug("Config validation passed.")


# =============================================================================
# Deep-merge helper
# =============================================================================

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge *override* into *base*.

    - For nested dicts, recurse.
    - For all other types, the override value replaces the base value.
    - Keys present only in *base* are kept unchanged.
    - Keys present only in *override* are added.

    Parameters
    ----------
    base : dict
        The default configuration dictionary.
    override : dict
        The experiment-specific overrides.

    Returns
    -------
    dict
        Merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# =============================================================================
# Dataclass population from dict
# =============================================================================

def _populate_dataclass(cls, d: Dict[str, Any]):
    """
    Recursively instantiate a dataclass ``cls`` from a plain dictionary
    ``d``.  Nested dataclass fields are handled automatically.

    Unknown keys in ``d`` are silently ignored (forward compatibility).
    Missing keys fall back to the dataclass field default.
    """
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs: Dict[str, Any] = {}

    # Build a mapping from field name to its actual type object
    import typing, sys
    type_map = {}
    for f in fields(cls):
        # Resolve string annotations
        if isinstance(f.type, str):
            # Evaluate the annotation in the module's global namespace
            ns = {**vars(sys.modules[cls.__module__])}
            try:
                resolved = eval(f.type, ns)
            except Exception:
                resolved = f.type
        else:
            resolved = f.type
        type_map[f.name] = resolved

    for f in fields(cls):
        name = f.name
        if name not in d:
            # Use default — will be applied by the dataclass __init__
            continue
        val = d[name]
        resolved_type = type_map[name]

        # Check if the resolved type is itself a dataclass
        import dataclasses as _dc
        if _dc.is_dataclass(resolved_type) and isinstance(val, dict):
            kwargs[name] = _populate_dataclass(resolved_type, val)
        else:
            kwargs[name] = val

    return cls(**kwargs)


# =============================================================================
# Public API
# =============================================================================

def load_config(
    base_path: Union[str, Path],
    exp_path: Optional[Union[str, Path]] = None,
) -> CORDISConfig:
    """
    Load and merge configuration files, returning a validated
    :class:`CORDISConfig` instance.

    Parameters
    ----------
    base_path : str or Path
        Path to the base JSON config (e.g. ``"configs/default.json"``).
    exp_path : str or Path or None
        Optional path to an experiment-specific JSON that **overrides**
        selected fields in the base config.

    Returns
    -------
    CORDISConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If either config file is not found.
    json.JSONDecodeError
        If either file contains invalid JSON.
    ValueError
        If the merged config fails validation.

    Examples
    --------
    >>> cfg = load_config("configs/default.json")
    >>> cfg = load_config("configs/default.json", "configs/exp_pareto.json")
    >>> cfg.topology.n_ap
    6
    """
    # ── Resolve paths against the project root when relative ─────────────
    # This makes the code work regardless of which directory PyCharm (or any
    # other runner) uses as the working directory at runtime.
    from cordis.utils.paths import get_project_root

    def _resolve(p: Union[str, Path]) -> Path:
        """Return an absolute path, resolving relative paths against the
        project root (not the current working directory)."""
        p = Path(p)
        if p.is_absolute():
            return p
        # Try the path as-is first (in case cwd happens to be correct)
        if p.exists():
            return p.resolve()
        # Fall back to project-root-relative resolution
        return get_project_root() / p

    # ── Load base config ──────────────────────────────────────────────────
    base_p = _resolve(base_path)
    if not base_p.exists():
        raise FileNotFoundError(f"Base config not found: {base_p}")
    with open(base_p, encoding="utf-8") as f:
        merged: Dict[str, Any] = json.load(f)
    logger.info("Loaded base config: %s", base_p)

    # ── Load and merge experiment config ──────────────────────────────────
    if exp_path is not None:
        exp_p = _resolve(exp_path)
        if not exp_p.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_p}")
        with open(exp_p, encoding="utf-8") as f:
            exp_dict: Dict[str, Any] = json.load(f)
        merged = _deep_merge(merged, exp_dict)
        logger.info("Merged experiment config: %s", exp_p)

    # ── Populate dataclasses ──────────────────────────────────────────────
    cfg = _populate_dataclass(CORDISConfig, merged)

    # ── Validate ──────────────────────────────────────────────────────────
    cfg.validate()

    return cfg


def config_from_dict(d: Dict[str, Any]) -> CORDISConfig:
    """
    Build a :class:`CORDISConfig` directly from a Python dictionary.

    Useful for programmatic construction in tests or scripts that
    generate config values at runtime.

    Parameters
    ----------
    d : dict
        Configuration dictionary (same structure as the JSON files).

    Returns
    -------
    CORDISConfig
    """
    cfg = _populate_dataclass(CORDISConfig, d)
    cfg.validate()
    return cfg


# =============================================================================
# Parameter registry — single source of truth for documentation
# =============================================================================
# Maps "Section.field_name" -> {help, unit, range}
# Introspected by scripts/list_config_params.py to produce the reference.
# "range" uses the convention:
#   "> 0"          numeric bound
#   "[0, 1]"       closed interval
#   "{A|B|C}"      discrete choices
#   "positive int" plain English when no simple bound applies

PARAM_REGISTRY: Dict[str, Dict[str, str]] = {

    # ── TopologyConfig ────────────────────────────────────────────────────
    "topology.type": {
        "help":  "AP placement geometry",
        "unit":  "-",
        "range": "{circle|random|grid}",
    },
    "topology.ap_radius_m": {
        "help":  "Radius of the circle on which APs are placed (circle layout)",
        "unit":  "m",
        "range": "> 0",
    },
    "topology.ue_max_radius_m": {
        "help":  "Outer radius of the annulus in which UEs are dropped",
        "unit":  "m",
        "range": "> ue_min_radius_m",
    },
    "topology.ue_min_radius_m": {
        "help":  "Inner radius of the UE placement annulus (minimum UE-origin distance)",
        "unit":  "m",
        "range": ">= 0",
    },
    "topology.tg_max_radius_m": {
        "help":  "Outer radius of the annulus in which targets are dropped",
        "unit":  "m",
        "range": "> tg_min_radius_m",
    },
    "topology.tg_min_radius_m": {
        "help":  "Inner radius of the target placement annulus",
        "unit":  "m",
        "range": ">= 0",
    },
    "topology.ue_min_separation_m": {
        "help":  ("Minimum 2-D distance between any two UEs (Stage 24). 0 disables "
                  "(default), preserving the original area-uniform draw and existing "
                  "seeds; a positive value enables a rejection sampler that re-draws "
                  "UEs until all pairwise distances clear the threshold."),
        "unit":  "m",
        "range": ">= 0 (0 disables)",
    },
    "topology.ue_target_min_separation_m": {
        "help":  ("Minimum 2-D distance between any UE and any target (Stage 24). 0 "
                  "disables (default); a positive value places targets via a rejection "
                  "sampler that avoids the already-placed UEs."),
        "unit":  "m",
        "range": ">= 0 (0 disables)",
    },
    "topology.n_ap": {
        "help":  "Total number of APs (|A| = |A_t| + |A_r|)",
        "unit":  "-",
        "range": ">= 2",
    },
    "topology.n_ue": {
        "help":  "Number of single-antenna communication users (N_ue)",
        "unit":  "-",
        "range": ">= 1",
    },
    "topology.n_targets": {
        "help":  "Number of sensing targets (N_t)",
        "unit":  "-",
        "range": ">= 1",
    },
    "topology.n_ant": {
        "help":  "Number of antenna elements per AP (M_t = M_r = M)",
        "unit":  "-",
        "range": ">= 1",
    },
    "topology.n_rf_chains": {
        "help":  "Number of RF chains per AP; limits simultaneous streams (N_RF <= M)",
        "unit":  "-",
        "range": "[1, n_ant]",
    },
    "topology.array_type": {
        "help":  "Antenna array geometry at every AP",
        "unit":  "-",
        "range": "{ULA|UCA}",
    },
    "topology.antenna_spacing_factor": {
        "help":  "Normalised inter-element spacing d/lambda",
        "unit":  "lambda",
        "range": "(0, 1]",
    },
    "topology.n_sensing_rx": {
        "help":  "Number of APs dedicated to sensing reception (|A_r|); remaining APs transmit",
        "unit":  "-",
        "range": "[1, n_ap - 1]",
    },
    "topology.h_ap_m": {
        "help":  "AP antenna height above ground (used in 3GPP path loss)",
        "unit":  "m",
        "range": "> h_ue_m",
    },
    "topology.h_ue_m": {
        "help":  "UE antenna height above ground",
        "unit":  "m",
        "range": "> 0",
    },
    "topology.h_tg_m": {
        "help":  "Target height above ground",
        "unit":  "m",
        "range": ">= 0",
    },
    "topology.grid_n_rows": {
        "help":  "Number of rows in the AP grid (grid layout only); total APs = rows x cols",
        "unit":  "-",
        "range": ">= 1",
    },
    "topology.grid_n_cols": {
        "help":  "Number of columns in the AP grid (grid layout only)",
        "unit":  "-",
        "range": ">= 1",
    },
    "topology.grid_spacing_m": {
        "help":  "Distance between adjacent APs in the grid",
        "unit":  "m",
        "range": "> 0",
    },

    # ── FrequencyConfig ───────────────────────────────────────────────────
    "frequency.carrier_freq_ghz": {
        "help":  "Carrier frequency f_c; determines wavelength and path loss exponents",
        "unit":  "GHz",
        "range": "[0.5, 100]  (3GPP TR 38.901 validity range)",
    },
    "frequency.bandwidth_mhz": {
        "help":  "Total system bandwidth B",
        "unit":  "MHz",
        "range": "> 0",
    },
    "frequency.n_subcarriers": {
        "help":  "Total number of OFDM subcarriers N_sc",
        "unit":  "-",
        "range": "power of 2, >= rb_n_subcarriers",
    },
    "frequency.rb_n_subcarriers": {
        "help":  "Number of subcarriers per resource block Q",
        "unit":  "-",
        "range": ">= 1",
    },
    "frequency.rb_n_symbols": {
        "help":  "Number of OFDM symbols per resource block K",
        "unit":  "-",
        "range": ">= 1",
    },
    "frequency.subcarrier_spacing_khz": {
        "help":  "OFDM subcarrier spacing Delta_f (15 kHz = 5G NR numerology 0)",
        "unit":  "kHz",
        "range": "> 0",
    },
    "frequency.cp_fraction": {
        "help":  "Cyclic prefix duration as a fraction of the OFDM symbol period",
        "unit":  "-",
        "range": "(0, 0.25]",
    },

    # ── ChannelConfig ─────────────────────────────────────────────────────
    "channel.model": {
        "help":  "Channel model identifier string (reserved for future model selection)",
        "unit":  "-",
        "range": "{3gpp_umi_rician}",
    },
    "channel.scenario": {
        "help":  "3GPP propagation scenario; controls path loss and LoS probability formulas",
        "unit":  "-",
        "range": "{UMi|UMa|RMa}",
    },
    "channel.snr_db": {
        "help":  ("Transmit SNR = P_max / sigma_n^2; sets per-AP power "
                  "budget.  At B=20 MHz, NF=7 dB, T=290 K, the mapping is "
                  "P_max[W] = 10^((SNR_dB - 124)/10).  Examples: "
                  "124 dB = 1 W, 130 dB = 4 W, 134 dB = 10 W, "
                  "137 dB = 20 W (default), 140 dB = 40 W, 144 dB = 100 W. "
                  "The default 137 dB matches the upper end of 5G mMIMO "
                  "mid-band micro/small-cell conducted power (2-20 W).  "
                  "Note: realistic urban-micro pathloss is ~120-130 dB at "
                  "500 m / 3 GHz, so SNR_dB << 120 leaves no link budget"),
        "unit":  "dB",
        "range": "any real (cell-free typical: 124 to 144; default: 137)",
    },
    "channel.noise_figure_db": {
        "help":  "Receiver noise figure NF added to thermal noise floor",
        "unit":  "dB",
        "range": ">= 0",
    },
    "channel.noise_temp_k": {
        "help":  "Thermal noise reference temperature T_0",
        "unit":  "K",
        "range": "> 0 (standard: 290)",
    },
    "channel.tau_f": {
        "help":  "TDD frame size tau_f in channel uses; must satisfy tau_f >= tau_p + tau_d",
        "unit":  "samples",
        "range": "> tau_p + tau_d",
    },
    "channel.tau_p": {
        "help":  "Pilot sequence length tau_p; set >= n_ue to avoid pilot contamination",
        "unit":  "samples",
        "range": ">= 1",
    },
    "channel.tau_d": {
        "help":  "Number of downlink ISAC symbols per TDD frame (tau_d)",
        "unit":  "samples",
        "range": ">= 1",
    },
    "channel.pilot_power_db": {
        "help":  (
            "Uplink pilot transmit SNR: 10 log10(P_p / sigma_n^2). "
            "The RECEIVED pilot SNR at the AP is SNR_rx = (P_p/sigma_n^2) x tau_p x beta_{au}. "
            "With 3GPP UMi path loss, beta << 1, so this must be set much higher than the "
            "data SNR to compensate. Rule of thumb: pilot_power_db = snr_db + |PL_dB|. "
            "120 dB corresponds to P_p ~ 200 mW (23 dBm), which is physically realistic."
        ),
        "unit":  "dB",
        "range": "any real; 80-130 dB for 3GPP UMi at 50-650 m; 10-30 dB only for no-path-loss toy models",
    },
    "channel.shadow_fading_los_std_db": {
        "help":  "Shadow fading standard deviation sigma_SF for LoS links (3GPP UMi: 4 dB)",
        "unit":  "dB",
        "range": ">= 0",
    },
    "channel.shadow_fading_nlos_std_db": {
        "help":  "Shadow fading standard deviation sigma_SF for NLoS links (3GPP UMi: 7.82 dB)",
        "unit":  "dB",
        "range": ">= 0",
    },
    "channel.shadow_corr_distance_m": {
        "help":  "Shadow fading spatial decorrelation distance D_corr between UEs at a common AP",
        "unit":  "m",
        "range": "> 0 (3GPP default: 50 m)",
    },
    "channel.rician_k_db_mean": {
        "help":  "Mean Rician K-factor for LoS links (drawn from Gaussian; 3GPP UMi LoS: 9 dB)",
        "unit":  "dB",
        "range": "any real",
    },
    "channel.rician_k_db_std": {
        "help":  "Standard deviation of the Rician K-factor (3GPP UMi: 5 dB)",
        "unit":  "dB",
        "range": ">= 0",
    },
    "channel.as_azimuth_deg_std_los": {
        "help":  "Azimuth angular spread (ASD) sigma_phi for LoS links; used to build C_{au}",
        "unit":  "deg",
        "range": ">= 0",
    },
    "channel.as_azimuth_deg_std_nlos": {
        "help":  "Azimuth angular spread (ASD) sigma_phi for NLoS links",
        "unit":  "deg",
        "range": ">= 0",
    },
    "channel.as_elevation_deg_std_los": {
        "help":  "Elevation angular spread (ESD) sigma_theta for LoS links",
        "unit":  "deg",
        "range": ">= 0",
    },
    "channel.as_elevation_deg_std_nlos": {
        "help":  "Elevation angular spread (ESD) sigma_theta for NLoS links",
        "unit":  "deg",
        "range": ">= 0",
    },
    "channel.n_spatial_samples": {
        "help":  "Monte Carlo samples for integrating the spatial correlation matrix C_{au}; higher = more accurate but slower",
        "unit":  "-",
        "range": ">= 100 (typical: 1000-5000)",
    },
    "channel.shared_scatter_rank": {
        "help":  "Rank r of shared scattering subspace B_a (inter-user correlation); 0 = disabled",
        "unit":  "-",
        "range": "[0, n_ant)",
    },
    "channel.shared_scatter_power": {
        "help":  "Fraction of NLoS power allocated to the shared scattering subspace B_a",
        "unit":  "-",
        "range": "[0, 1)",
    },
    "channel.estimation_method": {
        "help":  "Channel estimation algorithm at transmit APs",
        "unit":  "-",
        "range": "{MMSE|LS|perfect}  (perfect = genie-aided, zero estimation error)",
    },
    "channel.mmwave_n_clusters": {
        "help":  "Number of mmWave scattering clusters (active when carrier_freq_ghz > 6)",
        "unit":  "-",
        "range": ">= 1",
    },
    "channel.mmwave_n_rays": {
        "help":  "Number of rays per mmWave cluster",
        "unit":  "-",
        "range": ">= 1",
    },
    "channel.mmwave_cluster_as_deg": {
        "help":  "Per-cluster angular spread for mmWave channel",
        "unit":  "deg",
        "range": "> 0",
    },

    # ── SensingConfig ─────────────────────────────────────────────────────
    "sensing.rx_strategy": {
        "help":  "Receive AP selection strategy; determines which APs collect target echoes",
        "unit":  "-",
        "range": "{single_closest_centroid|single_farthest_centroid|per_target_closest|assent_file|assent_live}",
    },
    "sensing.sigma_rcs_sq_db": {
        "help":  "RCS power variance sigma_RCS^2 = E[|zeta_t|^2]; Swerling-I model",
        "unit":  "dB",
        "range": "any real (typical: -10 to 10)",
    },
    "sensing.swerling_model": {
        "help":  "Target RCS fluctuation model: 0 = deterministic, 1 = Swerling-I (CN)",
        "unit":  "-",
        "range": "{0|1}",
    },
    "sensing.los_model": {
        "help":  ("LoS availability model for target s_t in (eq. sensing-"
                  "channel-target).  'always' forces s_t=1 for all bistatic "
                  "links — required by the paper's rank-one LoS-only target "
                  "channel model (Proposition 3) because s_t=0 zeros out the "
                  "entire target echo and SCNR drops to the -300 dB sentinel.  "
                  "'3gpp_umi' uses distance-dependent stochastic LoS probability "
                  "(realistic but most urban targets become undetectable).  "
                  "'deterministic' uses a fixed P_LoS via los_probability_override"),
        "unit":  "-",
        "range": "{always|3gpp_umi|deterministic} (default: always)",
    },
    "sensing.los_probability_override": {
        "help":  "Fixed P_LoS value for all targets when los_model='deterministic'; null = use model",
        "unit":  "-",
        "range": "[0, 1] or null",
    },
    "sensing.multi_static": {
        "help":  "Enable multi-static sensing (multiple TX/RX AP pairs per target)",
        "unit":  "-",
        "range": "{true|false}",
    },
    "sensing.max_tx_aps_per_target": {
        "help":  "Maximum number of transmit APs illuminating a single target K_tx",
        "unit":  "-",
        "range": "[1, n_ap - 1]",
    },
    "sensing.max_rx_aps_per_target": {
        "help":  "Maximum number of receive APs processing echoes from a single target K_rx",
        "unit":  "-",
        "range": "[1, n_sensing_rx]",
    },
    "sensing.n_snapshots": {
        "help":  "Number of slow-time STAP snapshots T; multiplicative temporal gain in SCNR",
        "unit":  "symbols",
        "range": ">= 1",
    },
    "sensing.sigma_clt": {
        "help":  "OPTIONAL explicit override for sigma_clt.  If null (default), "
                 "sigma_clt is derived from clutter_cnr_db; if set, this value "
                 "is used directly (sigma_clt^2 = sigma_clt^2, ignoring CNR).",
        "unit":  "-",
        "range": "null OR >= 0",
    },
    "sensing.clutter_cnr_db": {
        "help":  "Clutter-to-noise ratio (CNR); sets sigma_clt^2 = 10^(CNR/10) * "
                 "sigma_n^2.  Used only when sensing.sigma_clt is null (default).",
        "unit":  "dB",
        "range": "any real (typical: -20 to 0)",
    },
    "sensing.rho_clt_model": {
        "help":  "Temporal autocorrelation model for the clutter Doppler process rho_clt(Delta_tau)",
        "unit":  "-",
        "range": "{constant|jakes|gaussian}  (constant => rho=1 for all lags)",
    },
    "sensing.rho_clt_bandwidth": {
        "help":  "Normalised clutter Doppler bandwidth (fraction of PRF); used by jakes/gaussian models",
        "unit":  "-",
        "range": "(0, 0.5]",
    },
    "sensing.clutter_as_deg": {
        "help":  "Angular spread of clutter returns; controls spatial correlation C_a of clutter",
        "unit":  "deg",
        "range": "> 0",
    },
    "sensing.clutter_center_strategy": {
        "help":  (
            "Spatial placement strategy for the clutter Power Angular Spectrum "
            "(PAS) at each transmit AP. Controls how clutter geometry couples "
            "to beam steering and therefore how kappa trades off clutter "
            "avoidance against beam quality. "
            "'target_centroid': PAS centred on the AP->target-centroid direction "
            "(clutter and target spatially aligned; high kappa pushes the beam "
            "away from the target, collapsing both SCNR and SINR for any UE the "
            "target shadows). "
            "'offset': PAS centred at (target azimuth + clutter_offset_az_deg), "
            "decoupling clutter avoidance from comm coverage. "
            "'uniform': PAS spans the full azimuth (effectively C ~ (P_clt/M)*I); "
            "the clutter penalty becomes a global power regulariser and kappa "
            "no longer steers the beam. "
            "'random': per-AP random azimuth drawn from U[-pi, pi]; useful for "
            "Monte-Carlo over clutter geometries."
        ),
        "unit":  "-",
        "range": "{target_centroid|offset|uniform|random}",
    },
    "sensing.clutter_offset_az_deg": {
        "help":  (
            "Azimuth offset added to the target azimuth when "
            "clutter_center_strategy='offset'; the clutter PAS is centred at "
            "(target azimuth + clutter_offset_az_deg). Larger values move the "
            "clutter further from the target direction, reducing the geometric "
            "coupling between sensing gain and clutter penalty. Ignored by "
            "other strategies."
        ),
        "unit":  "deg",
        "range": "any real (typical: 30 to 90)",
    },

    # ── ADMMConfig ────────────────────────────────────────────────────────
    "algorithm.admm.kappa": {
        "help":  "Clutter penalty weight kappa in the linear sensing surrogate U_cpu^sens",
        "unit":  "-",
        "range": ">= 0 (0 = ignore clutter in objective)",
    },
    "algorithm.admm.rho": {
        "help":  "ADMM penalty parameter rho controlling consensus convergence speed",
        "unit":  "-",
        "range": "> 0 (typical: 0.1 to 10)",
    },
    "algorithm.admm.n_max": {
        "help":  "Maximum number of ADMM iterations N_max before forced termination",
        "unit":  "-",
        "range": ">= 1",
    },
    "algorithm.admm.eps_pri": {
        "help":  ("Primal residual convergence threshold ε_pri.  With the "
                  "auto-balanced ρ (rho=1), residuals settle near 1 in SNR-"
                  "amplitude units, so tolerances of 0.3-1.0 match the "
                  "algorithm's natural equilibrium.  Values << 0.1 require "
                  "rho > 1 (risks SCA destabilisation) and often cause ADMM "
                  "to run to n_max without ever triggering the stop criterion"),
        "unit":  "-",
        "range": "> 0 (typical: 0.3 to 1.0; default: 1.0)",
    },
    "algorithm.admm.eps_dual": {
        "help":  ("Dual residual convergence threshold ε_dual.  Same scaling "
                  "considerations as eps_pri — keep at O(1) under default "
                  "rho=1 auto-balance"),
        "unit":  "-",
        "range": "> 0 (typical: 0.3 to 1.0; default: 1.0)",
    },
    "algorithm.admm.xi_slack": {
        "help":  "Slack variable penalty xi >> 0 in P-Central; ensures feasibility of SOC constraint",
        "unit":  "-",
        "range": ">> 1 (typical: 1e3 to 1e5)",
    },
    "algorithm.admm.target_priority_equal": {
        "help":  "Use equal priority weights omega_t = 1 for all targets; if false supply weights at runtime",
        "unit":  "-",
        "range": "{true|false}",
    },
    "algorithm.admm.warm_start_from_split": {
        "help":  "Initialise ADMM beamformers W^(0) from CORDIS-Split Phase I output",
        "unit":  "-",
        "range": "{true|false}",
    },
    "algorithm.admm.solver": {
        "help":  "CVXPY backend solver for local QCQP subproblems (P-Local)",
        "unit":  "-",
        "range": "{CLARABEL|GUROBI|MOSEK|SCS}",
    },
    "algorithm.admm.best_iter_criterion": {
        "help":  ("'feasible_then_residual' (default) returns the best feasible iterate "
                  "(highest min-SINR among iterates meeting gamma for every user), falling back "
                  "to the lowest r_pri+r_dual iterate when none are feasible, and never "
                  "downgrades a feasible incumbent. 'residual_norm' picks the iterate "
                  "minimising r_pri+r_dual (the converged plateau). 'min_sinr' picks the "
                  "highest min-user SINR (legacy; may pick a swing peak in non-converged "
                  "trajectories). See Stage 22a / 23a."),
        "unit":  "-",
        "range": "{residual_norm|min_sinr|feasible_then_residual}",
    },

    "algorithm.admm.adaptive_rho": {
        "help":  ("Enable the Boyd-Parikh-Chu (2011) Sec.3.4.1 adaptive rho scheme (rebalances "
                  "primal vs dual residual within [rho_min_factor, rho_max_factor]). DEFAULT "
                  "OFF (Stage 23 follow-up): in this SCA-in-the-loop ADMM the rho changes "
                  "perturb the err-tangent trust region and were empirically destabilising "
                  "(residuals swing, slower/no convergence within n_max), while the always-on "
                  "auto-rho scaling already balances the problem. Set true to re-enable. Stage 22b."),
        "unit":  "-",
        "range": "{true|false}",
    },
    "algorithm.admm.rho_mu_balance": {
        "help":  ("Boyd's μ threshold for the relative-residual balance "
                  "test in adaptive ρ.  Adapt only when "
                  "(r_pri/ε_pri) > μ·(r_dual/ε_dual) (or vice-versa)."),
        "unit":  "-",
        "range": "> 1 (typical: 5–20; Boyd's default: 10)",
    },
    "algorithm.admm.rho_tau": {
        "help":  ("Symmetric multiplicative step for adaptive ρ. "
                  "τ=2.0 is Boyd's default; we use 1.5 because the "
                  "SCA-linearised err penalty loses validity at large ρ "
                  "and a gentler step preserves stability."),
        "unit":  "-",
        "range": "> 1 (typical: 1.25–2.0)",
    },
    "algorithm.admm.rho_max_factor": {
        "help":  ("Upper bound on the adaptive ρ multiplier (factor × "
                  "input rho_admm).  Default 3.0 — half the empirical "
                  "instability threshold (~5×) of this SCA-based ADMM."),
        "unit":  "-",
        "range": "> 1.0 (typical: 2.0–5.0)",
    },
    "algorithm.admm.rho_min_factor": {
        "help":  ("Lower bound on the adaptive ρ multiplier.  Default "
                  "0.5 — protects against over-shrinking which would "
                  "produce a near-zero penalty and lose all consensus."),
        "unit":  "-",
        "range": "(0, 1.0]  (typical: 0.25–0.75)",
    },
    "algorithm.admm.rho_adapt_warmup": {
        "help":  ("Number of warmup iterations before adaptive ρ engages. "
                  "Protects against adapting on transient residuals while "
                  "the SCA gradient is still settling."),
        "unit":  "iters",
        "range": ">= 0 (typical: 3–10)",
    },
    "algorithm.admm.rho_adapt_interval": {
        "help":  ("Minimum iterations between consecutive ρ changes — "
                  "gives the algorithm time to respond before adjusting "
                  "again.  Acts as cooldown / hysteresis."),
        "unit":  "iters",
        "range": ">= 1 (typical: 2–5)",
    },
    "algorithm.admm.early_stop_patience": {
        "help":  ("Patience for early stop (iterations). Bails out when no new best iterate has "
                  "been recorded for this many consecutive iters. Active under the "
                  "residual-based criteria (residual_norm, feasible_then_residual); the patience "
                  "clock starts only once a FEASIBLE incumbent exists (Stage 23 follow-up "
                  "feasibility gate), so it never stops on a sub-gamma iterate. Set 0 to disable "
                  "(e.g. the convergence-trace experiment)."),
        "unit":  "iters",
        "range": ">= 0 (0 disables; typical: 10–30)",
    },
    "algorithm.admm.early_stop_min_iters": {
        "help":  ("Warmup before patience-stop can fire.  Protects "
                  "against stopping prematurely on transient improvement "
                  "in the early iterations."),
        "unit":  "iters",
        "range": ">= 0 (typical: 20–50)",
    },

    # ── AlgorithmConfig (umbrella — shared by every algorithm) ───────────
    "algorithm.gamma_db": {
        "help":  "Per-user minimum SINR requirement gamma_u; shared by "
                 "Split / ADMM / Centralized / benchmarks",
        "unit":  "dB",
        "range": "any real (typical: 0 to 20)",
    },

    # ── SplitOptConfig ────────────────────────────────────────────────────
    "algorithm.split.epsilon_reg": {
        "help":  "Regularisation parameter epsilon in LR-MMSE precoder (eq. split-comm-bf)",
        "unit":  "-",
        "range": "> 0 (typical: 1e-3 to 1e-1)",
    },
    "algorithm.split.epsilon_nsc": {
        "help":  "Loading factor epsilon for null-space projection P_perp in NS-C beamformer",
        "unit":  "-",
        "range": "> 0 (typical: 1e-4 to 1e-2)",
    },
    "algorithm.split.target_priority_equal": {
        "help":  "Use equal priority weights omega_t = 1 for all targets in NS-C allocation",
        "unit":  "-",
        "range": "{true|false}",
    },
    "algorithm.split.kappa": {
        "help":  "Clutter penalty weight kappa in the P-Split sensing utility U_cpu^sens",
        "unit":  "-",
        "range": ">= 0",
    },
    "algorithm.split.xi_penalty": {
        "help":  "Slack penalty xi >> 0 in P-Split; penalises violation of SINR QoS constraint",
        "unit":  "-",
        "range": ">> 1 (typical: 1e3 to 1e5)",
    },
    "algorithm.split.solver": {
        "help":  "CVXPY backend solver for P-Split convex program",
        "unit":  "-",
        "range": "{CLARABEL|GUROBI|MOSEK|SCS}",
    },

    # ── SimulationConfig ──────────────────────────────────────────────────
    "simulation.n_trials": {
        "help":  "Number of independent Monte Carlo realisations",
        "unit":  "-",
        "range": ">= 1",
    },
    "simulation.seed": {
        "help":  "Master random seed; fully determines all random draws when fixed",
        "unit":  "-",
        "range": ">= 0",
    },
    "simulation.n_jobs": {
        "help":  "Number of parallel workers for Monte Carlo (-1 = all CPU cores)",
        "unit":  "-",
        "range": "-1 or >= 1",
    },
    "simulation.save_dir": {
        "help":  "Root directory for saved result files (.npz + _meta.json pairs)",
        "unit":  "-",
        "range": "any valid path",
    },
    "simulation.tag": {
        "help":  "Short label appended to output filenames for easy identification",
        "unit":  "-",
        "range": "any string",
    },
    "simulation.log_level": {
        "help":  "Python logging level for the cordis.* logger hierarchy",
        "unit":  "-",
        "range": "{DEBUG|INFO|WARNING|ERROR}",
    },
}

