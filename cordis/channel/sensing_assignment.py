"""
cordis/channel/sensing_assignment.py
=====================================
Sensing AP assignment strategies for the CORDIS simulation framework.

This module is responsible for determining which APs transmit to which
targets and which APs receive from which targets, producing a
:class:`SensingAssociation` object that is then consumed by the sensing
metrics (SCNR) and the CORDIS algorithms.

Available strategies
--------------------

``"single_closest_centroid"``  (default, matches original code)
    One receive AP overall, chosen as the AP closest to the centroid of
    all targets.  All TX APs illuminate all targets.  Simplest baseline.

``"single_farthest_centroid"``
    One receive AP overall, chosen as the AP farthest from the AP
    centroid (maximizes the effective aperture of the bistatic geometry).

``"per_target_closest"``
    One dedicated receive AP per target — the AP closest to that
    individual target.  Different targets may have different receive APs.
    TX APs illuminate all targets.  Produces a richer multi-static
    geometry for benchmarking.

``"assent"``
    Full joint AP clustering, user/target scheduling, and mode selection
    from the ASSENT framework (arXiv:2511.09992).  ASSENT takes the
    lightweight link statistics (average channel gains, spatial
    correlations, bistatic sensing gains) as input and returns the
    complete binary association matrices.  Two sub-modes:
      - ``assent_file`` : load a pre-computed ASSENT decision from a
        JSON file (offline / reproducible benchmark mode).
      - ``assent_live`` : call the ASSENT model directly at runtime
        via the ``assent_cellfree_isac`` Python package (requires the
        ASSENT repo to be installed).

Usage
-----
    from cordis.channel.sensing_assignment import assign_sensing

    assoc = assign_sensing(topo, cfg, lsf)          # uses cfg strategy
    assoc = assign_sensing(topo, cfg, lsf,
                           strategy="per_target_closest")
    print(assoc.summary())
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from cordis.utils.config import CORDISConfig
from cordis.utils.logger import get_logger
from cordis.channel.topology import NetworkTopology, SensingAssociation
from cordis.channel.pathloss import LargeScaleFading

logger = get_logger(__name__)


# =============================================================================
# Public dispatcher
# =============================================================================

def assign_sensing(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    lsf: LargeScaleFading,
    strategy: Optional[str] = None,
    assent_decision_path: Optional[Union[str, Path]] = None,
) -> SensingAssociation:
    """
    Produce a :class:`SensingAssociation` using the requested strategy.

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    lsf  : LargeScaleFading
        Used by the ASSENT adapter to build link statistics.
    strategy : str or None
        Which assignment strategy to use.  If None, reads
        ``cfg.sensing.rx_strategy`` from the config.  Choices:

        ``"single_closest_centroid"``
            One Rx AP, closest to the centroid of all targets.
        ``"single_farthest_centroid"``
            One Rx AP, farthest from the AP centroid.
        ``"per_target_closest"``
            One dedicated Rx AP per target (closest to that target).
        ``"assent_file"``
            Load ASSENT decision from ``assent_decision_path``.
        ``"assent_live"``
            Call ASSENT model at runtime (requires installation).

    assent_decision_path : str or Path or None
        Path to a JSON file containing a pre-computed ASSENT decision.
        Required when strategy == ``"assent_file"``.

    Returns
    -------
    SensingAssociation
        Validated association object.

    Raises
    ------
    ValueError
        If the strategy is unknown or required arguments are missing.
    """
    if strategy is None:
        strategy = getattr(cfg.sensing, "rx_strategy", "single_closest_centroid")

    strategy = strategy.lower().strip()

    # ── No targets → all APs transmit, empty association ──────────────────
    if topo.n_targets == 0:
        logger.info("No targets — all APs set to transmit mode.")
        for ap in topo.aps:
            ap.is_transmit = True
            ap.is_receive  = False
        return SensingAssociation(
            tx_aps_per_target={},
            rx_aps_per_target={},
            strategy="none (0 targets)",
        )

    logger.info("Sensing assignment strategy: '%s'", strategy)

    if strategy == "single_closest_centroid":
        assoc = _single_global(topo, cfg, mode="closest_centroid")

    elif strategy == "single_farthest_centroid":
        assoc = _single_global(topo, cfg, mode="farthest_centroid")

    elif strategy == "per_target_closest":
        assoc = _per_target_closest(topo, cfg)

    elif strategy == "assent_file":
        if assent_decision_path is None:
            raise ValueError(
                "strategy='assent_file' requires assent_decision_path to be set."
            )
        assoc = _assent_from_file(topo, cfg, Path(assent_decision_path))

    elif strategy == "assent_live":
        assoc = _assent_live(topo, cfg, lsf)

    else:
        raise ValueError(
            f"Unknown sensing assignment strategy '{strategy}'. "
            f"Choose from: single_closest_centroid, single_farthest_centroid, "
            f"per_target_closest, assent_file, assent_live."
        )

    # Update the AP flags on the topology to be consistent with the association
    _sync_ap_flags(topo, assoc)

    assoc.validate(topo)
    logger.debug("\n%s", assoc.summary())
    return assoc


# =============================================================================
# Strategy 1 & 2 — single global receive AP
# =============================================================================

def _single_global(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    mode: str = "closest_centroid",
) -> SensingAssociation:
    """
    Select one receive AP for the entire network; all other APs transmit.
    All TX APs illuminate all targets; the single RX AP receives all.

    Parameters
    ----------
    mode : {"closest_centroid", "farthest_centroid"}
        ``closest_centroid``  : RX AP is closest to the mean target position.
        ``farthest_centroid`` : RX AP is farthest from the mean AP position
                                (maximises bistatic aperture).
    """
    ap_positions_2d = topo.ap_positions[:, :2]   # (N_ap, 2)

    if mode == "closest_centroid":
        if topo.n_targets == 0:
            rx_idx = 0
        else:
            tg_centroid = topo.target_positions[:, :2].mean(axis=0)
            dists = np.linalg.norm(ap_positions_2d - tg_centroid, axis=1)
            rx_idx = int(np.argmin(dists))

    elif mode == "farthest_centroid":
        ap_centroid = ap_positions_2d.mean(axis=0)
        dists = np.linalg.norm(ap_positions_2d - ap_centroid, axis=1)
        rx_idx = int(np.argmax(dists))

    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    tx_indices = [ap.idx for ap in topo.aps if ap.idx != rx_idx]
    rx_indices  = [rx_idx]

    tx_per_tg = {tg.idx: tx_indices for tg in topo.targets}
    rx_per_tg = {tg.idx: rx_indices for tg in topo.targets}

    logger.debug(
        "_single_global (%s): RX AP=%d, TX APs=%s",
        mode, rx_idx, tx_indices,
    )
    return SensingAssociation(
        tx_aps_per_target=tx_per_tg,
        rx_aps_per_target=rx_per_tg,
        strategy=f"single_{mode}",
    )


# =============================================================================
# Strategy 3 — one dedicated receive AP per target
# =============================================================================

def _per_target_closest(
    topo: NetworkTopology,
    cfg: CORDISConfig,
) -> SensingAssociation:
    """
    Assign one receive AP per target — the AP that is closest (in 3-D
    Euclidean distance) to that individual target.

    If two targets share the same closest AP, a secondary candidate is
    chosen for one of them to avoid assigning the same AP as the sole
    receiver for multiple targets (which would collapse to the single
    global case).  TX APs are all APs that are not assigned as RX for
    any target.

    Notes
    -----
    When N_targets > N_ap / 2, it becomes impossible to give every
    target a unique dedicated receiver.  In that case we fall back to
    allowing shared receivers and log a warning.
    """
    n_tg = topo.n_targets
    n_ap = topo.n_ap

    # Compute 3-D distance from every AP to every target: (N_ap, N_tg)
    ap_pos = topo.ap_positions          # (N_ap, 3)
    tg_pos = topo.target_positions      # (N_tg, 3)
    dist   = np.linalg.norm(
        ap_pos[:, np.newaxis, :] - tg_pos[np.newaxis, :, :], axis=-1
    )  # (N_ap, N_tg)

    # Greedy assignment: for each target (sorted by min-distance, so the
    # most "certain" assignment goes first), pick the closest available AP.
    assigned_rx: Dict[int, int] = {}   # tg_idx -> rx_ap_idx
    available   = set(ap.idx for ap in topo.aps)

    # Sort targets by their minimum AP distance (easiest assignment first)
    min_dist_per_tg = dist.min(axis=0)             # (N_tg,)
    target_order    = np.argsort(min_dist_per_tg)  # process nearest first

    for tg_col in target_order:
        tg_idx = topo.targets[tg_col].idx
        # Find the closest available AP for this target
        sorted_aps = np.argsort(dist[:, tg_col])
        chosen = None
        for ap_row in sorted_aps:
            ap_idx = topo.aps[ap_row].idx
            if ap_idx in available:
                chosen = ap_idx
                break

        if chosen is None:
            # No available AP left — allow sharing with a warning
            chosen = int(np.argmin(dist[:, tg_col]))
            logger.warning(
                "Target %d: no unique Rx AP available; sharing AP %d "
                "(n_targets=%d may be too large for n_ap=%d).",
                tg_idx, chosen, n_tg, n_ap,
            )
        else:
            available.discard(chosen)

        assigned_rx[tg_idx] = chosen
        logger.debug("Target %d → Rx AP %d", tg_idx, chosen)

    # TX APs: all APs not used as a receive AP for any target
    all_rx = set(assigned_rx.values())
    all_tx = [ap.idx for ap in topo.aps if ap.idx not in all_rx]

    if not all_tx:
        # Edge case: more targets than APs — every AP is an RX AP.
        # Fall back: let all APs transmit and receive (not physically valid
        # for half-duplex, but at least we don't crash).
        logger.warning(
            "per_target_closest: all APs assigned as receivers. "
            "Falling back to first AP as the sole transmitter."
        )
        all_tx = [topo.aps[0].idx]

    tx_per_tg = {tg.idx: all_tx for tg in topo.targets}
    rx_per_tg = {tg_idx: [rx_ap] for tg_idx, rx_ap in assigned_rx.items()}

    return SensingAssociation(
        tx_aps_per_target=tx_per_tg,
        rx_aps_per_target=rx_per_tg,
        strategy="per_target_closest",
    )


# =============================================================================
# Strategy 4a — ASSENT from file (offline / reproducible mode)
# =============================================================================

def _assent_from_file(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    path: Path,
) -> SensingAssociation:
    """
    Load a pre-computed ASSENT association decision from a JSON file.

    Expected JSON format (output of ASSENT inference script)::

        {
          "tau":  [1, 1, 0, 1, 1, 0],        // AP modes: 1=Tx, 0=Rx
          "ytx":  [[1,0],[1,0],[0,0],[1,1],[1,0],[0,0]],  // (N_ap, N_tg)
          "yrx":  [[0,0],[0,0],[0,1],[0,0],[0,0],[0,1]],  // (N_ap, N_tg)
          "strategy": "assent_file",
          "source": "path/to/decision.json"
        }

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    path : Path
        Path to the ASSENT decision JSON file.

    Returns
    -------
    SensingAssociation
    """
    import json

    if not path.exists():
        raise FileNotFoundError(
            f"ASSENT decision file not found: {path}\n"
            f"Generate it by running the ASSENT inference script:\n"
            f"  python assent_inference.py --output {path}"
        )

    with open(path, encoding="utf-8") as f:
        decision = json.load(f)

    logger.info("Loaded ASSENT decision from: %s", path)

    ytx = np.array(decision["ytx"], dtype=int)   # (N_ap, N_tg)
    yrx = np.array(decision["yrx"], dtype=int)   # (N_ap, N_tg)

    n_ap_file, n_tg_file = ytx.shape
    if n_ap_file != topo.n_ap:
        raise ValueError(
            f"ASSENT file has {n_ap_file} APs but topology has {topo.n_ap}."
        )
    if n_tg_file != topo.n_targets:
        raise ValueError(
            f"ASSENT file has {n_tg_file} targets but topology has "
            f"{topo.n_targets}."
        )

    tx_per_tg: Dict[int, List[int]] = {}
    rx_per_tg: Dict[int, List[int]] = {}

    for tg_col, tg in enumerate(topo.targets):
        tx_per_tg[tg.idx] = [
            topo.aps[ap_row].idx
            for ap_row in range(topo.n_ap)
            if ytx[ap_row, tg_col] == 1
        ]
        rx_per_tg[tg.idx] = [
            topo.aps[ap_row].idx
            for ap_row in range(topo.n_ap)
            if yrx[ap_row, tg_col] == 1
        ]

    return SensingAssociation(
        tx_aps_per_target=tx_per_tg,
        rx_aps_per_target=rx_per_tg,
        strategy="assent_file",
    )


# =============================================================================
# Strategy 4b — ASSENT live (calls model at runtime)
# =============================================================================

def _assent_live(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    lsf: LargeScaleFading,
) -> SensingAssociation:
    """
    Call the ASSENT GNN model at runtime to obtain association decisions.

    Requires the ASSENT package to be installed:
        pip install git+https://github.com/LS-Wireless/ASSENT-CellFree-ISAC

    The function builds the lightweight link statistics (average channel
    gains, spatial correlations, bistatic sensing gains) in the format
    expected by ASSENT, runs inference, and converts the output binary
    matrices back into a :class:`SensingAssociation`.

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    lsf  : LargeScaleFading

    Returns
    -------
    SensingAssociation
    """
    try:
        import assent  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "ASSENT package not found.  Install it with:\n"
            "  pip install git+https://github.com/LS-Wireless/ASSENT-CellFree-ISAC\n"
            "Or use strategy='assent_file' with a pre-computed decision."
        )

    logger.info("Running ASSENT live inference …")

    # ── Build ASSENT input: average channel gains G_comm (N_ap, N_ue) ────
    G_comm = lsf.beta_lin                       # (N_ap, N_ue)  linear scale

    # ── Build ASSENT input: bistatic sensing gains G_sens (N_ap, N_ap, N_tg)
    n_ap = topo.n_ap
    n_tg = topo.n_targets
    G_sens = np.zeros((n_ap, n_ap, n_tg), dtype=float)

    if lsf.beta_tg_lin is not None:
        for at_idx in range(n_ap):
            for ar_idx in range(n_ap):
                if at_idx == ar_idx:
                    continue
                for tg_idx in range(n_tg):
                    beta_at = lsf.beta_tg_lin[at_idx, tg_idx]
                    beta_ar = lsf.beta_tg_lin[ar_idx, tg_idx]
                    G_sens[at_idx, ar_idx, tg_idx] = np.sqrt(beta_at * beta_ar)

    # ── Call ASSENT inference ─────────────────────────────────────────────
    decision = assent.infer(
        G_comm=G_comm,
        G_sens=G_sens,
        n_rf_chains=cfg.topology.n_rf_chains,
        alpha=0.5,                              # comm-sensing trade-off
    )

    tau = np.array(decision["tau"], dtype=int)   # (N_ap,)
    ytx = np.array(decision["ytx"], dtype=int)   # (N_ap, N_tg)
    yrx = np.array(decision["yrx"], dtype=int)   # (N_ap, N_tg)

    tx_per_tg: Dict[int, List[int]] = {}
    rx_per_tg: Dict[int, List[int]] = {}

    for tg_col, tg in enumerate(topo.targets):
        tx_per_tg[tg.idx] = [
            topo.aps[r].idx for r in range(n_ap) if ytx[r, tg_col] == 1
        ]
        rx_per_tg[tg.idx] = [
            topo.aps[r].idx for r in range(n_ap) if yrx[r, tg_col] == 1
        ]

    logger.info("ASSENT inference complete.")
    return SensingAssociation(
        tx_aps_per_target=tx_per_tg,
        rx_aps_per_target=rx_per_tg,
        strategy="assent_live",
    )


# =============================================================================
# Helper: sync AP-level flags with association
# =============================================================================

def _sync_ap_flags(topo: NetworkTopology, assoc: SensingAssociation) -> None:
    """
    Update ``is_transmit`` / ``is_receive`` flags on all APs in the
    topology so they are consistent with the :class:`SensingAssociation`.

    An AP is marked as transmit if it appears as a TX AP for *any* target.
    An AP is marked as receive if it appears as a RX AP for *any* target.
    An AP may appear as neither (unused AP) but not as both (half-duplex).
    """
    tx_set = set(assoc.all_tx_ap_indices)
    rx_set = set(assoc.all_rx_ap_indices)

    for ap in topo.aps:
        ap.is_transmit = ap.idx in tx_set
        ap.is_receive  = ap.idx in rx_set

