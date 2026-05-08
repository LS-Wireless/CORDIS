"""
scripts/run_sanity_checks.py
==============================
Sanity-check scenarios to validate the CORDIS simulation framework
up to Stage 5 (beamforming).

Produces three figures:

  Scenario 1 — SINR vs P_max/σ²  (all-LoS, no targets)
      Validates beamformer correctness and CSI-error impact.
      Expected:  global_zf >> lr_mmse ≈ rzf > mrt at high SNR.
                 Perfect CSI ≥ imperfect CSI.
                 MRT plateaus (MUI-limited); ZF keeps climbing.

  Scenario 2 — SINR vs pilot SNR  (3GPP LoS prob, no targets)
      Validates channel estimation quality impact.
      Expected:  all methods converge to their perfect-CSI ceiling
                 as pilot SNR increases.  LR-MMSE degrades more
                 gracefully than RZF at low pilot SNR.

  Scenario 3 — SINR & SCNR vs pilot SNR  (3GPP LoS, with targets)
      Validates ISAC interaction (ρ < 1, sensing beams active).
      Expected:  SINR lower than Scenario 2 (S2CI from sensing beams).
                 SCNR improves with pilot quality and is largely
                 BF-method-independent (NS-C regardless of comm BF).

Reasonable parameter ranges (50 m cell, β_avg ≈ 1e-8, τ_p = 10)
-----------------------------------------------------------------
  SNR sweep   :  70–120 dB   noise-limited → interference-limited
  Pilot sweep :  60–120 dB   NMSE ≈ 0.9 (60 dB) → ≈ 0 (120 dB)
  Fixed SNR   :  90 dB       clear method separation

Usage
-----
  # Quick check (5 trials)
  python scripts/run_sanity_checks.py --n-trials 5

  # Full validation (100 trials, parallel)
  python scripts/run_sanity_checks.py --n-trials 100 --n-jobs 8

  # Custom sweep ranges
  python scripts/run_sanity_checks.py --n-trials 50 \\
      --pilot-snr-min 60 --pilot-snr-max 120 --pilot-snr-step 5

  # Single scenario with custom output
  python scripts/run_sanity_checks.py --n-trials 50 --scenario 1 \\
      --out-dir results/figures/validation
"""

from __future__ import annotations

SCRIPT_DESCRIPTION = (
    "Sanity-check scenarios for the CORDIS simulation framework (Stage 5). "
    "Produces validation figures for SINR vs SNR, SINR vs pilot SNR, "
    "and ISAC performance."
)

import argparse
import copy
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cordis.utils.logger import setup_logging, get_logger
from cordis.utils.config import load_config, CORDISConfig
from cordis.utils.io_utils import make_rng, child_rng
from cordis.channel.topology import generate_topology
from cordis.channel.pathloss import (
    compute_large_scale_fading, noise_power_watts, snr_to_tx_power,
)
from cordis.channel.rician import (
    compute_channel_statistics, generate_channel_realization,
)
from cordis.channel.estimation import (
    run_channel_estimation, design_pilot_sequences, compute_pilot_diagnostics,
)
from cordis.channel.sensing_channel import compute_sensing_statistics
from cordis.channel.sensing_assignment import assign_sensing
from cordis.algorithms.beamforming import design_phase_i
from cordis.metrics import compute_sinr, compute_scnr

logger = get_logger("sanity_checks")

try:
    from joblib import Parallel, delayed as jdelayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# =============================================================================
# Plot style
# =============================================================================

COLORS = {
    "lr_mmse":   "#2563EB",
    "rzf":       "#DC2626",
    "mrt":       "#16A34A",
    "global_zf": "#9333EA",
}
LABELS = {
    "lr_mmse":   "LR-MMSE",
    "rzf":       "Local RZF",
    "mrt":       "Local MRT",
    "global_zf": "Global ZF",
}
MARKERS = {
    "lr_mmse": "o", "rzf": "s", "mrt": "^", "global_zf": "D",
}


def _style_axis(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=9)


def _mean_db(values: list) -> float:
    """Mean of linear-scale values in dB."""
    if not values:
        return -np.inf
    return float(10.0 * np.log10(max(float(np.mean(values)), 1e-30)))


# =============================================================================
# Logging: silence all cordis.* sub-module INFO messages
# =============================================================================

def _silence_submodules() -> None:
    """
    Suppress INFO-level logs from all cordis.* sub-modules.

    Without this, every trial fires dozens of repeated messages from
    topology, pathloss, estimation, sensing_assignment, beamforming etc.
    The script's own sanity_checks logger stays at INFO.
    """
    logging.getLogger("cordis").setLevel(logging.WARNING)
    logging.getLogger("sanity_checks").setLevel(logging.INFO)


# =============================================================================
# Dual-output helper: terminal + log file
# =============================================================================

_log_fh = None   # file handle; set in main() when --no-save-log is not passed


def say(msg: str = "") -> None:
    """
    Print ``msg`` to the terminal AND write it to the log file (if open).

    Leading ``\\r`` characters are stripped when writing to the file so
    that the in-place trial-progress lines are saved as clean single lines
    rather than as stacked overwrites.
    """
    print(msg)
    if _log_fh is not None:
        _log_fh.write(msg.lstrip("\r").rstrip() + "\n")
        _log_fh.flush()


# =============================================================================
# Config factory (load base config ONCE; deep-copy per scenario)
# =============================================================================

_BASE_CFG: Optional[CORDISConfig] = None   # set in main()


def _base_cell() -> CORDISConfig:
    """
    Deep copy of the base config with 50 m cell geometry.

    Using n_ue = 6 (closer to n_ant = 8) makes MUI more significant
    so that BF method differences are clearly visible.
    """
    cfg = copy.deepcopy(_BASE_CFG)
    cfg.topology.type            = "circle"
    cfg.topology.ap_radius_m     = 50.0
    cfg.topology.ue_min_radius_m = 10.0
    cfg.topology.ue_max_radius_m = 60.0
    cfg.topology.n_ap            = 6
    cfg.topology.n_ue            = 8      # 6/8 loading → visible MUI
    cfg.topology.n_ant           = 8
    cfg.topology.n_rf_chains     = 8
    cfg.topology.n_targets       = 0
    cfg.topology.n_sensing_rx    = 0
    cfg.channel.tau_p            = 10
    return cfg


def make_config_s1(snr_db: float, csi_mode: str,
                   fixed_pilot_snr: Optional[float] = None) -> CORDISConfig:
    """
    All-LoS, no targets.

    Pilot SNR: if ``fixed_pilot_snr`` is given it is used directly;
    otherwise defaults to ``snr_db + 70 dB`` so estimation stays excellent
    throughout the SNR sweep without manual tuning.
    """
    cfg = _base_cell()
    cfg.channel.rician_k_db_mean  = 100.0
    cfg.channel.rician_k_db_std   = 0.0
    cfg.channel.snr_db            = float(snr_db)
    cfg.channel.pilot_power_db    = (float(fixed_pilot_snr)
                                     if fixed_pilot_snr is not None
                                     else float(snr_db) + 70.0)
    cfg.channel.estimation_method = csi_mode
    return cfg


def make_config_s2(snr_db: float, pilot_db: float) -> CORDISConfig:
    """3GPP LoS probability, no targets. pilot_db is the sweep variable."""
    cfg = _base_cell()
    cfg.channel.rician_k_db_mean  = 9.0
    cfg.channel.rician_k_db_std   = 5.0
    cfg.channel.snr_db            = float(snr_db)
    cfg.channel.pilot_power_db    = float(pilot_db)
    cfg.channel.estimation_method = "MMSE"
    return cfg


def make_config_s3(snr_db: float, pilot_db: float) -> CORDISConfig:
    """3GPP LoS, 2 sensing targets."""
    cfg = make_config_s2(snr_db, pilot_db)
    cfg.topology.n_targets       = 2
    cfg.topology.tg_min_radius_m = 10.0
    cfg.topology.tg_max_radius_m = 60.0
    cfg.topology.n_sensing_rx    = 1
    cfg.sensing.los_model        = "always"
    return cfg


# =============================================================================
# Single-trial worker
# =============================================================================

def _run_one_trial(
    cfg: CORDISConfig,
    seed: int,
    methods: List[str],
    psr: float,
    with_scnr: bool = False,
) -> Dict:
    """Run one full pipeline trial. Returns per-method linear SINR/SCNR."""
    rng = make_rng(seed)
    topo    = generate_topology(cfg, child_rng(rng))
    lsf     = compute_large_scale_fading(
        topo, cfg, child_rng(rng),
        include_targets=(cfg.topology.n_targets > 0),
    )
    stats   = compute_channel_statistics(topo, cfg, lsf, child_rng(rng))
    real    = generate_channel_realization(topo, cfg, lsf, stats, child_rng(rng))
    Phi, ct = design_pilot_sequences(cfg.topology.n_ue, cfg.channel.tau_p,
                                      child_rng(rng))
    est     = run_channel_estimation(topo, cfg, lsf, stats, real,
                                      child_rng(rng), Phi, ct)
    s_stats = compute_sensing_statistics(topo, cfg, lsf, child_rng(rng))
    assoc   = assign_sensing(topo, cfg, lsf)

    sigma_n_sq = noise_power_watts(cfg.frequency.bandwidth_hz,
                                    cfg.channel.noise_figure_db,
                                    cfg.channel.noise_temp_k)
    Pmax = snr_to_tx_power(cfg.channel.snr_db, sigma_n_sq)

    out: Dict = {}
    for method in methods:
        p1     = design_phase_i(topo, cfg, est, s_stats, assoc,
                                 comm_bf_method=method)
        W_tx   = p1.build_W_tx_equal_psr(psr, Pmax)
        sinr_m = compute_sinr(W_tx, est, topo, sigma_n_sq)
        entry  = {"min_sinr": sinr_m.min_sinr}
        if with_scnr and cfg.topology.n_targets > 0:
            scnr_m = compute_scnr(topo, cfg, W_tx, s_stats, assoc, sigma_n_sq)
            entry["weighted_sum_scnr"] = scnr_m.weighted_sum_scnr
        out[method] = entry
    return out


# =============================================================================
# Sweep-point runner with clean progress display
# =============================================================================

def _run_sweep_point(
    cfg: CORDISConfig,
    sweep_idx: int,
    n_sweep: int,
    sweep_label: str,
    sweep_val: float,
    n_trials: int,
    methods: List[str],
    psr: float,
    scenario_id: int,
    with_scnr: bool,
    n_jobs: int,
    seed_base: int,
) -> Dict:
    """
    Run all trials for one sweep point, print a single summary line,
    and return the per-method averaged results.

    Progress format (in-place \r overwrite during sequential runs):
      [S2] P_p/σ²= 80 dB ( 5/13) | trial  47/100
    Then replaced with a summary line when the sweep point finishes:
      [S2] P_p/σ²= 80 dB ( 5/13) | LR-MMSE= +8.3  Local RZF= +7.1  …  dB
    """
    seeds = [seed_base + sweep_idx * 10000 + t for t in range(n_trials)]
    prefix = f"  [S{scenario_id}] {sweep_label}={sweep_val:6.0f} dB ({sweep_idx+1:2d}/{n_sweep})"

    if _HAS_JOBLIB and n_jobs != 1:
        print(f"{prefix} | running {n_trials} trials (parallel) …", flush=True)
        trial_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            jdelayed(_run_one_trial)(cfg, s, methods, psr, with_scnr)
            for s in seeds
        )
    else:
        trial_results = []
        for t_idx, seed in enumerate(seeds):
            print(f"\r{prefix} | trial {t_idx+1:3d}/{n_trials}  ",
                  end="", flush=True)
            trial_results.append(_run_one_trial(cfg, seed, methods, psr, with_scnr))

    # Aggregate
    agg: Dict = {m: {"min_sinr": [], "weighted_sum_scnr": []} for m in methods}
    for r in trial_results:
        for m in methods:
            agg[m]["min_sinr"].append(r[m]["min_sinr"])
            if "weighted_sum_scnr" in r[m]:
                agg[m]["weighted_sum_scnr"].append(r[m]["weighted_sum_scnr"])

    # Summary line (overwrites the trial progress line)
    parts = [f"{LABELS[m]}={_mean_db(agg[m]['min_sinr']):+6.1f}"
             for m in methods]
    say(f"\r{prefix} | " + "  ".join(parts) + "  dB" + " " * 8)

    return {m: {k: float(np.mean(v)) if v else 0.0
                for k, v in vals.items()}
            for m, vals in agg.items()}


# =============================================================================
# Scenario 1
# =============================================================================

def scenario_1(args: argparse.Namespace, save_dir: Path, ts: str) -> None:
    say(f"\n{'─'*72}")
    say("  Scenario 1: All-LoS, No Targets — SINR vs P_max/σ²")
    if args.fixed_pilot_snr is not None:
        say(f"  Pilot SNR fixed at {args.fixed_pilot_snr:.0f} dB")
        fixed_pilot_snr_flag = rf"$P_p/\sigma_n^2$ = {args.fixed_pilot_snr:.0f}"
    else:
        say("  Pilot SNR = data SNR + 70 dB  (auto, always excellent MMSE)")
        fixed_pilot_snr_flag = "data SNR + 70"
    say(f"{'─'*72}")

    snr_range  = np.arange(args.snr_min, args.snr_max + 0.1,
                            args.snr_step, dtype=float)
    bf_methods = [m for m in ["lr_mmse", "rzf", "mrt", "global_zf"]
                  if m in args.methods]
    csi_modes  = ["perfect", "MMSE"]
    n_sweep    = len(snr_range)

    # Build ALL configs once (outside trial loop)
    cfgs = {
        (si, csi): make_config_s1(snr_db, csi, args.fixed_pilot_snr)
        for si, snr_db in enumerate(snr_range)
        for csi in csi_modes
    }

    results: Dict = {csi: {} for csi in csi_modes}
    for csi in csi_modes:
        say(f"\n  CSI mode: {'perfect' if csi == 'perfect' else 'imperfect (MMSE)'}")
        seed_offset = 0 if csi == "perfect" else 500_000
        for si, snr_db in enumerate(snr_range):
            results[csi][si] = _run_sweep_point(
                cfgs[(si, csi)], si, n_sweep,
                "SNR", snr_db, args.n_trials, bf_methods,
                psr=1.0, scenario_id=1, with_scnr=False,
                n_jobs=args.n_jobs,
                seed_base=1_000_000 + seed_offset,
            )

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in bf_methods:
        for csi, ls in [("perfect", "-"), ("MMSE", "--")]:
            y_db = [_mean_db([results[csi][si][method]["min_sinr"]])
                    for si in range(n_sweep)]
            lbl  = f"{LABELS[method]} ({'perfect' if csi == 'perfect' else 'imperfect'})"
            ax.plot(snr_range, y_db, ls, color=COLORS[method],
                    marker=MARKERS[method], markersize=4, linewidth=2, label=lbl)

    _style_axis(ax,
                xlabel=r"$P_{\max}/\sigma_n^2$ [dB]",
                ylabel="Average min-user SINR [dB]",
                title=f"Scenario 1: All LoS, No Targets — SINR vs SNR\n"
                       rf"Pilot SNR: {fixed_pilot_snr_flag} [dB]")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    path = save_dir / f"{ts}_scenario1_sinr_vs_snr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    say(f"\n  ✓  Figure → {path}")


# =============================================================================
# Scenario 2
# =============================================================================

def scenario_2(args: argparse.Namespace, save_dir: Path, ts: str) -> None:
    say(f"\n{'─'*72}")
    say("  Scenario 2: 3GPP LoS, No Targets — SINR vs Pilot SNR")
    say(f"{'─'*72}")

    pilot_range = np.arange(args.pilot_snr_min, args.pilot_snr_max + 0.1,
                             args.pilot_snr_step, dtype=float)
    bf_methods  = [m for m in ["lr_mmse", "rzf", "mrt", "global_zf"]
                   if m in args.methods]
    n_sweep     = len(pilot_range)

    cfgs = {pi: make_config_s2(args.fixed_snr, pilot_db)
            for pi, pilot_db in enumerate(pilot_range)}

    results: Dict = {}
    for pi, pilot_db in enumerate(pilot_range):
        results[pi] = _run_sweep_point(
            cfgs[pi], pi, n_sweep,
            "P_p/σ²", pilot_db, args.n_trials, bf_methods,
            psr=1.0, scenario_id=2, with_scnr=False,
            n_jobs=args.n_jobs, seed_base=2_000_000,
        )

    # Perfect-CSI ceiling
    say("\n  Perfect-CSI ceiling …")
    cfg_perf = make_config_s2(args.fixed_snr, args.fixed_snr + 70.0)
    cfg_perf.channel.estimation_method = "perfect"
    perf = _run_sweep_point(
        cfg_perf, 0, 1,
        "perf", args.fixed_snr, args.n_trials, bf_methods,
        psr=1.0, scenario_id=2, with_scnr=False,
        n_jobs=args.n_jobs, seed_base=2_900_000,
    )

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in bf_methods:
        y_db = [_mean_db([results[pi][method]["min_sinr"]]) for pi in range(n_sweep)]
        ax.plot(pilot_range, y_db, "-", color=COLORS[method],
                marker=MARKERS[method], markersize=4, linewidth=2, label=LABELS[method])
        perf_db = _mean_db([perf[method]["min_sinr"]])
        ax.axhline(perf_db, color=COLORS[method], linestyle=":", alpha=0.55, linewidth=1.5)

    _style_axis(ax,
                xlabel=r"$P_p/\sigma_n^2$ [dB]",
                ylabel="Average min-user SINR [dB]",
                title=(f"Scenario 2: 3GPP LoS, No Targets — SINR vs Pilot SNR\n"
                       rf"$P_{{\max}}/\sigma_n^2 = {args.fixed_snr:.0f}$ dB"
                       r" · dotted = perfect-CSI ceiling"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = save_dir / f"{ts}_scenario2_sinr_vs_pilot_snr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    say(f"\n  ✓  Figure → {path}")


# =============================================================================
# Scenario 3
# =============================================================================

def scenario_3(args: argparse.Namespace, save_dir: Path, ts: str) -> None:
    say(f"\n{'─'*72}")
    say("  Scenario 3: 3GPP LoS, 2 Targets — ISAC vs Pilot SNR")
    say(f"{'─'*72}")

    pilot_range = np.arange(args.pilot_snr_min, args.pilot_snr_max + 0.1,
                             args.pilot_snr_step, dtype=float)
    # global_zf excluded by default for S3 (computationally heavier with targets)
    bf_methods  = [m for m in ["lr_mmse", "rzf", "mrt"]
                   if m in args.methods]
    n_sweep     = len(pilot_range)

    cfgs = {pi: make_config_s3(args.fixed_snr, pilot_db)
            for pi, pilot_db in enumerate(pilot_range)}

    results: Dict = {}
    for pi, pilot_db in enumerate(pilot_range):
        results[pi] = _run_sweep_point(
            cfgs[pi], pi, n_sweep,
            "P_p/σ²", pilot_db, args.n_trials, bf_methods,
            psr=args.psr, scenario_id=3, with_scnr=True,
            n_jobs=args.n_jobs, seed_base=3_000_000,
        )

    # Figure: 1×2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for method in bf_methods:
        y_sinr = [_mean_db([results[pi][method]["min_sinr"]])
                  for pi in range(n_sweep)]
        y_scnr = [_mean_db([results[pi][method].get("weighted_sum_scnr", 1e-30)])
                  for pi in range(n_sweep)]
        ax1.plot(pilot_range, y_sinr, "-", color=COLORS[method],
                 marker=MARKERS[method], markersize=4, linewidth=2, label=LABELS[method])
        ax2.plot(pilot_range, y_scnr, "-", color=COLORS[method],
                 marker=MARKERS[method], markersize=4, linewidth=2, label=LABELS[method])

    _style_axis(ax1, r"$P_p/\sigma_n^2$ [dB]", "Average min-user SINR [dB]",
                rf"SINR  ($\rho={args.psr}$, $P_{{\max}}/\sigma_n^2={args.fixed_snr:.0f}$ dB)")
    ax1.legend(fontsize=9)
    _style_axis(ax2, r"$P_p/\sigma_n^2$ [dB]", "Average weighted-sum SCNR [dB]",
                rf"SCNR  ($\rho={args.psr}$)")
    ax2.legend(fontsize=9)

    fig.suptitle("Scenario 3: 3GPP LoS, 2 Targets — ISAC Performance vs Pilot SNR",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = save_dir / f"{ts}_scenario3_isac_vs_pilot_snr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    say(f"\n  ✓  Figure → {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_sanity_checks.py",
        description=SCRIPT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
default sweep ranges (50 m cell, β ≈ 1e-8 at 50 m, τ_p = 10):
  --snr-min 70 --snr-max 120 --snr-step 5           (11 points, Scenario 1)
  --pilot-snr-min 60 --pilot-snr-max 120 --pilot-snr-step 5  (13 points, S2/S3)
  --fixed-snr 90                                    (data SNR for S2/S3)

examples:
  python scripts/run_sanity_checks.py --n-trials 100
  python scripts/run_sanity_checks.py --n-trials 100 --n-jobs 8
  python scripts/run_sanity_checks.py --n-trials 50 --scenario 2 3 \\
      --pilot-snr-min 60 --pilot-snr-max 120
""",
    )
    p.add_argument("--n-trials",        type=int,   default=20)
    p.add_argument("--scenario",        type=int,   nargs="+", default=[1, 2, 3],
                   choices=[1, 2, 3])
    p.add_argument("--n-jobs",          type=int,   default=1,
                   help="Parallel workers via joblib (-1=all CPUs). Default: 1.")
    p.add_argument("--methods",         type=str,   nargs="+",
                   default=["lr_mmse", "rzf", "mrt", "global_zf"],
                   help="BF methods to compare.")

    # Sweep ranges
    p.add_argument("--snr-min",         type=float, default=70.0)
    p.add_argument("--snr-max",         type=float, default=120.0)
    p.add_argument("--snr-step",        type=float, default=5.0)
    p.add_argument("--pilot-snr-min",   type=float, default=60.0)
    p.add_argument("--pilot-snr-max",   type=float, default=120.0)
    p.add_argument("--pilot-snr-step",  type=float, default=5.0)
    p.add_argument("--fixed-snr",       type=float, default=90.0,
                   help="Data SNR for scenarios 2 & 3 [dB]. Default: 90.")
    p.add_argument("--fixed-pilot-snr", type=float, default=None,
                   help="Fixed pilot SNR P_p/σ² for scenario 1 [dB]. "
                        "If not set, pilot SNR = data SNR + 70 dB at each "
                        "sweep point (keeps estimation excellent throughout).")
    p.add_argument("--psr",             type=float, default=0.7,
                   help="Power splitting ratio ρ for scenario 3. Default: 0.7.")

    # Output
    p.add_argument("--out-dir",  default="eval/figures/sanity",
                   help="Directory for figures. Default: eval/figures/sanity.")
    p.add_argument("--log-dir",  default="",
                   help="Directory for the run log file. "
                        "Default: mirrors --out-dir under eval/logs/ "
                        "(e.g. eval/figures/sanity → eval/logs/sanity). "
                        "Pass an explicit path to override.")
    p.add_argument("--show",    action="store_true",
                   help="Display figures interactively after saving.")
    p.add_argument("--no-save-log", action="store_true",
                   help="Do not save a log file (default: save {ts}_run.log "
                        "alongside the figures).")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")
    _silence_submodules()

    global _BASE_CFG, _log_fh
    _BASE_CFG = load_config("configs/default.json")

    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve log directory ─────────────────────────────────────────────
    # If --log-dir was not given (empty string or None), mirror the figures
    # path: eval/figures/sanity → eval/logs/sanity.
    # If explicitly provided, use it as-is.
    if not args.log_dir:
        # eval/figures/sanity → .parent.parent = eval, then /logs/<leaf>
        log_dir = save_dir.parent.parent / "logs" / save_dir.name
    else:
        log_dir = Path(args.log_dir)

    # ── Timestamp (shared by figures and log file) ─────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Open log file (unless --no-save-log was passed) ───────────────────
    if not args.no_save_log:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{ts}_run.log"
        _log_fh  = open(log_path, "w", encoding="utf-8")

    n_s1  = int((args.snr_max       - args.snr_min)       / args.snr_step)       + 1
    n_s23 = int((args.pilot_snr_max - args.pilot_snr_min) / args.pilot_snr_step) + 1
    parallel = (f"joblib n_jobs={args.n_jobs}"
                if _HAS_JOBLIB and args.n_jobs != 1 else "sequential")

    say()
    say("╔══════════════════════════════════════════════════════════════╗")
    say("║  CORDIS Sanity Checks — Stage 5 (Beamforming)                ║")
    say("╠══════════════════════════════════════════════════════════════╣")
    say(f"║  Timestamp            : {ts:<37}║")
    say(f"║  Trials / sweep point : {args.n_trials:<37}║")
    say(f"║  Scenarios            : {str(args.scenario):<37}║")
    say(f"║  BF methods           : {', '.join(args.methods):<37}║")
    say(f"║  SNR sweep (S1)       : {args.snr_min:.0f}–{args.snr_max:.0f} dB  "
        f"step={args.snr_step:.0f} dB  ({n_s1} pts){'':<7}║")
    say(f"║  Pilot sweep (S2/S3)  : {args.pilot_snr_min:.0f}–{args.pilot_snr_max:.0f} dB  "
        f"step={args.pilot_snr_step:.0f} dB  ({n_s23} pts){'':<7}║")
    say(f"║  Fixed SNR (S2/S3)    : {args.fixed_snr:.0f} dB{'':<32}║")
    say(f"║  PSR (S3)             : {args.psr:<37}║")
    say(f"║  Execution            : {parallel:<37}║")
    say(f"║  Output               : {str(save_dir):<37}║")
    if _log_fh is not None:
        say(f"║  Log dir              : {str(log_dir):<37}║")
    say("╚══════════════════════════════════════════════════════════════╝")

    say("\n  Pilot diagnostics (scenario 2, pilot_snr = fixed_snr):")
    _diag_cfg  = make_config_s2(args.fixed_snr, args.fixed_snr)
    _diag_rng  = make_rng(0)
    _diag_topo = generate_topology(_diag_cfg, child_rng(_diag_rng))
    _diag_lsf  = compute_large_scale_fading(_diag_topo, _diag_cfg, child_rng(_diag_rng))
    diag = compute_pilot_diagnostics(_diag_topo, _diag_cfg, _diag_lsf)
    for line in diag["summary"].split("\n"):
        say("  " + line)

    t0 = time.time()

    if 1 in args.scenario:
        scenario_1(args, save_dir, ts)
    if 2 in args.scenario:
        scenario_2(args, save_dir, ts)
    if 3 in args.scenario:
        scenario_3(args, save_dir, ts)

    elapsed = time.time() - t0
    h, r    = divmod(int(elapsed), 3600)
    m, s    = divmod(r, 60)
    say(f"\n  ✓  All done in {h:02d}h {m:02d}m {s:02d}s")
    say(f"  Figures saved to: {save_dir.resolve()}")
    if _log_fh is not None:
        say(f"  Log saved to    : {log_path.resolve()}")
        _log_fh.close()
        _log_fh = None
    say()

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()

