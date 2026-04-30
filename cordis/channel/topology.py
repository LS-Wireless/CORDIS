"""
cordis/channel/topology.py
==========================
Network topology generation for the CORDIS simulation framework.

This module creates and manages the 3-D spatial layout of all network
entities: Access Points (APs), User Equipment (UEs), and sensing targets.
The output of this module is a :class:`NetworkTopology` object consumed by
every downstream module (path loss, channel generation, beamforming, …).

Supported AP placement layouts
-------------------------------
``"circle"``
    APs are equally spaced on a circle of radius ``ap_radius_m`` in the
    horizontal plane at height ``h_ap_m``.  This is the layout used in the
    conference paper experiments.

``"random"``
    APs are placed uniformly at random within an annulus
    [``ap_min_radius_m``, ``ap_radius_m``] to avoid near-zero inter-AP
    distances.

``"grid"``
    APs are placed on a regular rectangular grid with ``grid_n_rows`` rows
    and ``grid_n_cols`` columns and spacing ``grid_spacing_m``.  The grid
    is centred at the origin.

UE and target placement
-----------------------
Both UEs and targets are drawn uniformly at random inside an annulus
[``min_radius_m``, ``max_radius_m``] to enforce a minimum separation from
the origin (which often coincides with a cluster of APs).

AP mode assignment
------------------
Transmit APs (``At``) and receive APs (``Ar``) are selected by the
:meth:`NetworkTopology.assign_sensing_rx` method, which picks the AP(s)
closest to the target centroid as receive AP(s).  This heuristic minimizes
the sensing path loss on the receive side and matches the strategy used in
the paper.  The CS can override this assignment at any time.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from cordis.utils.config import CORDISConfig, TopologyConfig
from cordis.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Entity dataclasses
# =============================================================================

@dataclass
class AccessPoint:
    """
    Single Access Point (AP).

    Attributes
    ----------
    idx : int
        Zero-based index within the network.
    pos : np.ndarray, shape (3,)
        3-D position [x, y, z] in metres.
    n_ant : int
        Number of antenna elements M.
    n_rf : int
        Number of RF chains (≤ n_ant).
    array_type : str
        ``"ULA"`` or ``"UCA"``.
    is_transmit : bool
        ``True`` if this AP is in the transmit set A_t.
    is_receive : bool
        ``True`` if this AP is in the receive set A_r.
    """

    idx: int
    pos: NDArray[np.float64]           # shape (3,)
    n_ant: int
    n_rf: int
    array_type: str
    is_transmit: bool = True
    is_receive: bool = False

    @property
    def xy(self) -> NDArray[np.float64]:
        return self.pos[:2]

    @property
    def height(self) -> float:
        return float(self.pos[2])

    def distance_to(self, other_pos: NDArray[np.float64]) -> float:
        """3-D Euclidean distance to another position."""
        return float(np.linalg.norm(self.pos - other_pos))

    def distance_2d_to(self, other_pos: NDArray[np.float64]) -> float:
        """2-D (horizontal) Euclidean distance to another position."""
        return float(np.linalg.norm(self.pos[:2] - other_pos[:2]))


@dataclass
class UserEquipment:
    """
    Single-antenna communication user (UE).

    Attributes
    ----------
    idx : int
        Zero-based index.
    pos : np.ndarray, shape (3,)
        3-D position [x, y, z] in metres.
    """

    idx: int
    pos: NDArray[np.float64]           # shape (3,)

    @property
    def xy(self) -> NDArray[np.float64]:
        return self.pos[:2]

    @property
    def height(self) -> float:
        return float(self.pos[2])

    def distance_to(self, other_pos: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(self.pos - other_pos))

    def distance_2d_to(self, other_pos: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(self.pos[:2] - other_pos[:2]))


@dataclass
class SensingTarget:
    """
    Point-reflector sensing target.

    Attributes
    ----------
    idx : int
        Zero-based index.
    pos : np.ndarray, shape (3,)
        3-D position [x, y, z] in metres.
    velocity : np.ndarray, shape (3,)
        Velocity vector [vx, vy, vz] in m/s.  Zero for stationary targets.
    """

    idx: int
    pos: NDArray[np.float64]           # shape (3,)
    velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def xy(self) -> NDArray[np.float64]:
        return self.pos[:2]

    def distance_to(self, other_pos: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(self.pos - other_pos))

    def distance_2d_to(self, other_pos: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(self.pos[:2] - other_pos[:2]))

    def radial_velocity(
        self,
        ap_pos: NDArray[np.float64],
    ) -> float:
        """
        Radial (range-rate) component of the target velocity as seen from
        an AP at position ``ap_pos``.

        f_D = (v_radial / c) * f_c   →   provided by the caller.
        """
        diff = self.pos - ap_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return 0.0
        unit = diff / dist
        return float(np.dot(self.velocity, unit))


# =============================================================================
# NetworkTopology
# =============================================================================

@dataclass
class NetworkTopology:
    """
    Complete description of a single network realization.

    This is the primary output of the topology generator and the primary
    input to every downstream module.

    Attributes
    ----------
    aps : list[AccessPoint]
        All APs (transmit + receive) in index order.
    ues : list[UserEquipment]
        All communication UEs.
    targets : list[SensingTarget]
        All sensing targets.
    """

    aps: List[AccessPoint]
    ues: List[UserEquipment]
    targets: List[SensingTarget]

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def n_ap(self) -> int:
        return len(self.aps)

    @property
    def n_ue(self) -> int:
        return len(self.ues)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def tx_aps(self) -> List[AccessPoint]:
        """Subset A_t: transmit APs."""
        return [ap for ap in self.aps if ap.is_transmit]

    @property
    def rx_aps(self) -> List[AccessPoint]:
        """Subset A_r: receive (sensing) APs."""
        return [ap for ap in self.aps if ap.is_receive]

    @property
    def n_tx(self) -> int:
        return len(self.tx_aps)

    @property
    def n_rx(self) -> int:
        return len(self.rx_aps)

    # ── Position arrays ───────────────────────────────────────────────────

    @property
    def ap_positions(self) -> NDArray[np.float64]:
        """All AP positions, shape (N_ap, 3)."""
        return np.array([ap.pos for ap in self.aps], dtype=float)

    @property
    def ue_positions(self) -> NDArray[np.float64]:
        """All UE positions, shape (N_ue, 3)."""
        return np.array([ue.pos for ue in self.ues], dtype=float)

    @property
    def target_positions(self) -> NDArray[np.float64]:
        """All target positions, shape (N_tg, 3)."""
        return np.array([tg.pos for tg in self.targets], dtype=float)

    # ── Distance matrices ─────────────────────────────────────────────────

    def ap_ue_distances_2d(self) -> NDArray[np.float64]:
        """
        2-D distances between every AP and every UE.

        Returns
        -------
        np.ndarray, shape (N_ap, N_ue)
            ``D[a, u]`` = 2-D distance from AP a to UE u in metres.
        """
        ap_pos = self.ap_positions[:, :2]   # (N_ap, 2)
        ue_pos = self.ue_positions[:, :2]   # (N_ue, 2)
        diff = ap_pos[:, np.newaxis, :] - ue_pos[np.newaxis, :, :]  # (N_ap, N_ue, 2)
        return np.linalg.norm(diff, axis=-1)   # (N_ap, N_ue)

    def ap_ue_distances_3d(self) -> NDArray[np.float64]:
        """3-D distances (N_ap, N_ue)."""
        ap_pos = self.ap_positions
        ue_pos = self.ue_positions
        diff = ap_pos[:, np.newaxis, :] - ue_pos[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=-1)

    def ap_target_distances_3d(self) -> NDArray[np.float64]:
        """3-D distances (N_ap, N_tg)."""
        ap_pos = self.ap_positions
        tg_pos = self.target_positions
        diff = ap_pos[:, np.newaxis, :] - tg_pos[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=-1)

    # ── Mode assignment ───────────────────────────────────────────────────

    def assign_sensing_rx(
        self,
        n_rx: int = 1,
        strategy: str = "closest_to_target_centroid",
    ) -> None:
        """
        Designate ``n_rx`` APs as receive-only (A_r) and the remainder as
        transmit (A_t), then update the ``is_transmit`` / ``is_receive``
        flags on all :class:`AccessPoint` objects.

        Parameters
        ----------
        n_rx : int
            Number of receive APs.
        strategy : str
            ``"closest_to_target_centroid"`` : pick the AP(s) whose 2-D
            position is closest to the centroid of all targets.  Minimizes
            the sensing receive path loss.

            ``"farthest_from_ap_centroid"`` : pick the AP(s) farthest from
            the centroid of all APs, maximizing the geometric aperture of
            the multi-static receiver array.

            ``"fixed"`` : do not change existing flags (useful when the
            caller sets modes externally).

        Notes
        -----
        The constraint At ∩ Ar = ∅ (half-duplex) is enforced by clearing
        ``is_transmit`` for any AP assigned to A_r.
        """
        if n_rx <= 0:
            for ap in self.aps:
                ap.is_transmit = True
                ap.is_receive  = False
            return

        if n_rx >= self.n_ap:
            raise ValueError(
                f"n_rx ({n_rx}) must be < n_ap ({self.n_ap}) to leave at "
                f"least one transmit AP."
            )

        if strategy == "fixed":
            return

        # Compute selection scores
        if strategy == "closest_to_target_centroid":
            if self.n_targets == 0:
                warnings.warn(
                    "No targets in topology; defaulting to last AP as Rx.",
                    stacklevel=2,
                )
                rx_indices = [self.n_ap - 1]
            else:
                centroid = self.target_positions[:, :2].mean(axis=0)  # (2,)
                ap_2d = self.ap_positions[:, :2]                       # (N_ap, 2)
                dists = np.linalg.norm(ap_2d - centroid, axis=1)      # (N_ap,)
                rx_indices = list(np.argsort(dists)[:n_rx])

        elif strategy == "farthest_from_ap_centroid":
            centroid = self.ap_positions[:, :2].mean(axis=0)
            ap_2d = self.ap_positions[:, :2]
            dists = np.linalg.norm(ap_2d - centroid, axis=1)
            rx_indices = list(np.argsort(dists)[-n_rx:])

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'.  Choose "
                f"'closest_to_target_centroid', 'farthest_from_ap_centroid', "
                f"or 'fixed'."
            )

        for ap in self.aps:
            if ap.idx in rx_indices:
                ap.is_transmit = False
                ap.is_receive  = True
            else:
                ap.is_transmit = True
                ap.is_receive  = False

        logger.debug(
            "Mode assignment: Tx APs = %s, Rx APs = %s",
            [ap.idx for ap in self.tx_aps],
            [ap.idx for ap in self.rx_aps],
        )

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"NetworkTopology",
            f"  APs      : {self.n_ap}  (Tx={self.n_tx}, Rx={self.n_rx})",
            f"  UEs      : {self.n_ue}",
            f"  Targets  : {self.n_targets}",
        ]
        if self.n_ap > 0:
            d_ap_ue = self.ap_ue_distances_2d()
            lines.append(
                f"  d_AP-UE  : min={d_ap_ue.min():.0f} m, "
                f"mean={d_ap_ue.mean():.0f} m, max={d_ap_ue.max():.0f} m"
            )
        return "\n".join(lines)


# =============================================================================
# Placement helpers
# =============================================================================

def _uniform_annulus(
    rng: np.random.Generator,
    n: int,
    r_min: float,
    r_max: float,
    height: float,
) -> NDArray[np.float64]:
    """
    Draw ``n`` points uniformly inside an annulus [r_min, r_max] at
    the given ``height``.

    Sampling is performed in polar coordinates with the correct Jacobian
    (r drawn from the CDF of the uniform-area distribution) to avoid the
    central clustering artifact that arises from naive uniform-angle +
    uniform-radius sampling.

    Parameters
    ----------
    rng : np.random.Generator
    n : int
    r_min, r_max : float
        Inner and outer radii in metres.
    height : float
        z-coordinate in metres.

    Returns
    -------
    np.ndarray, shape (n, 3)
    """
    if r_max <= r_min:
        raise ValueError(f"r_max ({r_max}) must be > r_min ({r_min}).")

    # r ~ Uniform on [r_min², r_max²] then take sqrt → uniform area density
    r_sq = rng.uniform(r_min**2, r_max**2, size=n)
    r    = np.sqrt(r_sq)
    phi  = rng.uniform(0.0, 2 * np.pi, size=n)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.full(n, height, dtype=float)
    return np.column_stack([x, y, z])   # (n, 3)


def _circle_layout(
    n: int,
    radius: float,
    height: float,
    start_angle: float = 0.0,
) -> NDArray[np.float64]:
    """
    Place ``n`` points equally spaced on a circle of the given radius.

    Parameters
    ----------
    n : int
    radius : float
    height : float
    start_angle : float
        Angle of the first element in radians.  Default 0.

    Returns
    -------
    np.ndarray, shape (n, 3)
    """
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, n,
                         endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.full(n, height, dtype=float)
    return np.column_stack([x, y, z])


def _grid_layout(
    n_rows: int,
    n_cols: int,
    spacing: float,
    height: float,
) -> NDArray[np.float64]:
    """
    Place APs on a rectangular grid centred at the origin.

    Parameters
    ----------
    n_rows, n_cols : int
    spacing : float
        Distance between adjacent grid points in metres.
    height : float

    Returns
    -------
    np.ndarray, shape (n_rows * n_cols, 3)
    """
    xs = (np.arange(n_cols) - (n_cols - 1) / 2.0) * spacing
    ys = (np.arange(n_rows) - (n_rows - 1) / 2.0) * spacing
    grid_x, grid_y = np.meshgrid(xs, ys)
    x = grid_x.flatten()
    y = grid_y.flatten()
    z = np.full(len(x), height, dtype=float)
    return np.column_stack([x, y, z])


# =============================================================================
# Main generator
# =============================================================================

def generate_topology(
    cfg: CORDISConfig,
    rng: np.random.Generator,
    rx_strategy: str = "closest_to_target_centroid",
) -> NetworkTopology:
    """
    Generate a single network topology realization from a CORDIS config.

    Parameters
    ----------
    cfg : CORDISConfig
        Simulation configuration.
    rng : np.random.Generator
        Random number generator (from ``cordis.utils.io_utils.make_rng``).
        The caller is responsible for seeding; this function is stateless.
    rx_strategy : str
        Strategy for assigning receive APs.  Passed to
        :meth:`NetworkTopology.assign_sensing_rx`.

    Returns
    -------
    NetworkTopology

    Examples
    --------
    >>> from cordis.utils.config import load_config
    >>> from cordis.utils.io_utils import make_rng
    >>> from cordis.channel.topology import generate_topology
    >>> cfg = load_config("configs/default.json")
    >>> topo = generate_topology(cfg, make_rng(42))
    >>> print(topo.summary())
    """
    t = cfg.topology

    # ── AP positions ──────────────────────────────────────────────────────
    layout = t.type.lower()
    if layout == "circle":
        ap_pos = _circle_layout(t.n_ap, t.ap_radius_m, t.h_ap_m)

    elif layout == "random":
        r_min = getattr(t, "ap_min_radius_m", 50.0)
        ap_pos = _uniform_annulus(rng, t.n_ap, r_min, t.ap_radius_m, t.h_ap_m)

    elif layout == "grid":
        ap_pos = _grid_layout(t.grid_n_rows, t.grid_n_cols,
                              t.grid_spacing_m, t.h_ap_m)
        grid_count = len(ap_pos)
        if grid_count != t.n_ap:
            # Silently correct n_ap to match the actual grid count.
            # The script sets grid_n_rows/grid_n_cols from --n-ap before
            # calling here, so this branch is only reached when the JSON
            # has n_ap inconsistent with grid_n_rows * grid_n_cols.
            logger.debug(
                "Grid layout: %d x %d grid produces %d APs; "
                "setting n_ap=%d to match.",
                t.grid_n_rows, t.grid_n_cols, grid_count, grid_count,
            )
            t.n_ap = grid_count

    else:
        raise ValueError(
            f"Unknown topology type '{t.type}'.  "
            f"Choose 'circle', 'random', or 'grid'."
        )

    n_ap_actual = len(ap_pos)
    aps = [
        AccessPoint(
            idx=i,
            pos=ap_pos[i],
            n_ant=t.n_ant,
            n_rf=t.n_rf_chains,
            array_type=t.array_type,
            is_transmit=True,   # updated by assign_sensing_rx below
            is_receive=False,
        )
        for i in range(n_ap_actual)
    ]

    # ── UE positions ──────────────────────────────────────────────────────
    ue_pos_arr = _uniform_annulus(
        rng, t.n_ue, t.ue_min_radius_m, t.ue_max_radius_m, t.h_ue_m
    )
    ues = [
        UserEquipment(idx=u, pos=ue_pos_arr[u])
        for u in range(t.n_ue)
    ]

    # ── Target positions ──────────────────────────────────────────────────
    tg_pos_arr = _uniform_annulus(
        rng, t.n_targets, t.tg_min_radius_m, t.tg_max_radius_m, t.h_tg_m
    )
    targets = [
        SensingTarget(idx=j, pos=tg_pos_arr[j])
        for j in range(t.n_targets)
    ]

    # ── Assemble topology ─────────────────────────────────────────────────
    topo = NetworkTopology(aps=aps, ues=ues, targets=targets)
    topo.assign_sensing_rx(n_rx=t.n_sensing_rx, strategy=rx_strategy)

    logger.debug("Topology generated: %s", topo.summary().replace("\n", " | "))
    return topo


def generate_topology_batch(
    cfg: CORDISConfig,
    rng: np.random.Generator,
    n_trials: int,
    rx_strategy: str = "closest_to_target_centroid",
) -> List[NetworkTopology]:
    """
    Generate multiple independent topology realizations.

    Each trial uses a child RNG spawned from ``rng`` so that the
    realizations are statistically independent and the calling seed
    fully determines the batch.

    Parameters
    ----------
    cfg : CORDISConfig
    rng : np.random.Generator
        Parent RNG.
    n_trials : int
    rx_strategy : str

    Returns
    -------
    list[NetworkTopology], length n_trials
    """
    from cordis.utils.io_utils import child_rng
    topos = []
    for _ in range(n_trials):
        trial_rng = child_rng(rng)
        topos.append(generate_topology(cfg, trial_rng, rx_strategy))
    logger.info("Generated %d topology realisations.", n_trials)
    return topos

