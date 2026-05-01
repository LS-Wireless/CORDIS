"""
cordis/metrics/fronthaul.py
============================
Fronthaul communication overhead computation for the CORDIS framework.

Counts the number of real-valued scalars that must be exchanged between
APs and the central CPU for each algorithm, broken down by direction
(uplink / downlink) and per ADMM iteration where applicable.

The counts follow the algorithm descriptions in the CORDIS journal paper
(Section IV for CORDIS-Split, Section V for CORDIS-ADMM) and assume:
  - one complex scalar = 2 real scalars (real + imaginary parts)
  - the centralized benchmark sends the full Ĥ matrices uplink and the
    full W matrices downlink in a single round trip
  - CORDIS-Split exchanges only compressed per-AP scalars in one round
  - CORDIS-ADMM exchanges per-iteration consensus vectors

These counts give a fair, scenario-independent comparison of fronthaul
load across algorithms — a key contribution metric for the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cordis.utils.config import CORDISConfig
from cordis.utils.logger import get_logger
from cordis.channel.topology import NetworkTopology

logger = get_logger(__name__)


# =============================================================================
# Container
# =============================================================================

@dataclass
class FronthaulOverhead:
    """
    Fronthaul overhead for a single algorithm execution.

    Attributes
    ----------
    algorithm : str
        Algorithm identifier (``"centralized"``, ``"split"``, ``"admm"``).
    n_real_scalars_uplink : int
        Total real scalars sent from APs to CPU.
    n_real_scalars_downlink : int
        Total real scalars sent from CPU to APs (broadcast counted once
        per recipient AP).
    n_iterations : int
        Number of round trips (1 for non-iterative algorithms).
    bytes_per_scalar : int
        Bytes used to encode each real scalar (default 4 = float32).

    Properties:
    -----------
    total_real_scalars : int
    total_bytes : int
    """

    algorithm:               str
    n_real_scalars_uplink:   int
    n_real_scalars_downlink: int
    n_iterations:            int = 1
    bytes_per_scalar:        int = 4   # float32

    @property
    def total_real_scalars(self) -> int:
        return self.n_real_scalars_uplink + self.n_real_scalars_downlink

    @property
    def total_bytes(self) -> int:
        return self.total_real_scalars * self.bytes_per_scalar

    @property
    def total_kilobytes(self) -> float:
        return self.total_bytes / 1024.0

    @property
    def total_megabytes(self) -> float:
        return self.total_bytes / (1024.0 ** 2)


# =============================================================================
# Main computation
# =============================================================================

def compute_fronthaul_overhead(
    algorithm: str,
    topo: NetworkTopology,
    cfg: CORDISConfig,
    n_iterations: int = 1,
) -> FronthaulOverhead:
    """
    Compute the fronthaul overhead in number of real scalars exchanged.

    Parameters
    ----------
    algorithm : str
        One of ``"centralized"``, ``"split"``, ``"admm"``.
    topo : NetworkTopology
    cfg  : CORDISConfig
    n_iterations : int
        Number of ADMM iterations actually executed (used only when
        ``algorithm == "admm"``; ignored for the other two).

    Returns
    -------
    FronthaulOverhead

    Raises
    ------
    ValueError
        If ``algorithm`` is not recognized.
    """
    n_tx = topo.n_tx
    n_ue = topo.n_ue
    n_t  = topo.n_targets
    M    = cfg.topology.n_ant
    D    = n_ue + n_t   # |D| — total downlink streams

    algo = algorithm.lower().strip()

    if algo == "centralized":
        return _centralized(n_tx, n_ue, n_t, M, D)
    if algo == "split":
        return _split(n_tx, n_ue, n_t)
    if algo == "admm":
        return _admm(n_tx, n_ue, n_t, D, n_iterations)

    raise ValueError(
        f"Unknown algorithm '{algorithm}'.  "
        f"Choose from: centralized, split, admm."
    )


# =============================================================================
# Per-algorithm counters
# =============================================================================

def _centralized(n_tx: int, n_ue: int, n_t: int,
                 M: int, D: int) -> FronthaulOverhead:
    """
    Centralized benchmark.

    Uplink   : each TX AP sends Ĥ_{a_t} ∈ ℂ^{N_ue × M}  → 2·N_ue·M reals/AP
    Downlink : CPU sends W_{a_t} ∈ ℂ^{M × D}            → 2·M·D     reals/AP

    Single round trip.
    """
    per_ap_uplink   = 2 * n_ue * M
    per_ap_downlink = 2 * M * D
    return FronthaulOverhead(
        algorithm="centralized",
        n_real_scalars_uplink   = n_tx * per_ap_uplink,
        n_real_scalars_downlink = n_tx * per_ap_downlink,
        n_iterations            = 1,
    )


def _split(n_tx: int, n_ue: int, n_t: int) -> FronthaulOverhead:
    """
    CORDIS-Split fronthaul (Algorithm 1, Section IV).

    Uplink (per TX AP):
      - 1 scalar  η(a_t)                       (channel quality metric)
      - 3·N_ue   {β̂_{a_t u}, g̃_{a_t u}, e_{a_t u}^{(s)}}
      - 2 scalars z_{a_t}, q̃_{a_t}             (sensing scalars)

    Downlink (per TX AP):
      - 1 scalar  Σ η(a_t')                    (broadcast normalizer)
      - 1 scalar  ρ_{a_t}^*                    (optimal PSR)

    Single round trip.

    Note: β̂_{a_t u} = ĥ^H ŵ is generally complex.  We follow the paper's
    Table I convention which counts it as a real scalar — this is exact
    when phase alignment is applied (Section V-B of the journal), making
    β̂ real-valued by construction.
    """
    per_ap_uplink   = 1 + 3 * n_ue + 2
    per_ap_downlink = 1 + 1
    return FronthaulOverhead(
        algorithm="split",
        n_real_scalars_uplink   = n_tx * per_ap_uplink,
        n_real_scalars_downlink = n_tx * per_ap_downlink,
        n_iterations            = 1,
    )


def _admm(n_tx: int, n_ue: int, n_t: int,
          D: int, n_iterations: int) -> FronthaulOverhead:
    """
    CORDIS-ADMM fronthaul (Algorithm 2, Section V).

    Per iteration:
      Uplink (per TX AP):
        - For each user u: local contribution vector l_{a_t u} of size
          (|D| + 1) complex scalars = 2(|D| + 1) reals
        - Total per AP: 2 · N_ue · (|D| + 1)

      Downlink (broadcast to every TX AP):
        - For each user u: aggregate residual Σ̃_u of the same size
        - Total per AP per iteration: 2 · N_ue · (|D| + 1)

    Multiplied by the number of ADMM iterations actually executed.
    """
    per_ap_per_iter_uplink   = 2 * n_ue * (D + 1)
    per_ap_per_iter_downlink = 2 * n_ue * (D + 1)

    return FronthaulOverhead(
        algorithm="admm",
        n_real_scalars_uplink   = n_tx * per_ap_per_iter_uplink   * n_iterations,
        n_real_scalars_downlink = n_tx * per_ap_per_iter_downlink * n_iterations,
        n_iterations            = n_iterations,
    )


# =============================================================================
# Comparison helper (for tables / plots)
# =============================================================================

def compare_overhead(
    topo: NetworkTopology,
    cfg: CORDISConfig,
    n_admm_iterations: int = 10,
) -> dict:
    """
    Compute fronthaul overhead for all three algorithms in one call,
    returning a dict suitable for a results table.

    Parameters
    ----------
    topo : NetworkTopology
    cfg  : CORDISConfig
    n_admm_iterations : int
        Number of ADMM iterations to assume (typical: 5–10).

    Returns
    -------
    dict[algorithm_name, FronthaulOverhead]
    """
    return {
        "centralized": compute_fronthaul_overhead("centralized", topo, cfg),
        "split":       compute_fronthaul_overhead("split",       topo, cfg),
        "admm":        compute_fronthaul_overhead("admm",        topo, cfg,
                                                   n_iterations=n_admm_iterations),
    }

