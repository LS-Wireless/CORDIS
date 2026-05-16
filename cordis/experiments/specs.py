"""
Stage 8b — Canonical algorithm spec sets.

Single source of truth for which algorithms appear in each figure, what
hyperparameters they use, and how they're displayed.  Each public
spec-set factory returns a list of :class:`AlgorithmSpec` suitable for
passing to :class:`MonteCarloRunner`.

Display names match :data:`cordis.plotting.style.ALGORITHM_STYLE` so
all spec sets get consistent colors and markers across the paper.

Spec builders (one per algorithm)
---------------------------------
:func:`split_spec`       — CORDIS-Split            ``kind="cordis_split"``
:func:`admm_spec`        — CORDIS-ADMM             ``kind="cordis_admm"``
:func:`centralized_spec` — Centralized joint BF+PA ``kind="centralized"``
:func:`mrt_spec`         — Local MRT + P-Split PA  ``benchmark_name="mrt_split"``
:func:`zf_spec`          — Local ZF + P-Split PA   ``benchmark_name="zf_split"``
:func:`rzf_spec`         — Local RZF + P-Split PA  ``benchmark_name="rzf_split"``
:func:`lrmmse_spec`      — Local LR-MMSE + PA      ``benchmark_name="lr_mmse_split"``
:func:`global_mrt_spec`  — Global MRT + P-Split PA ``benchmark_name="global_mrt_split"``
:func:`global_zf_spec`   — Global ZF  + P-Split PA ``benchmark_name="global_zf_split"``

Spec set factories (composition)
--------------------------------
:func:`cordis_only`           — 2 specs
:func:`cordis_vs_centralized` — 3 specs
:func:`cordis_vs_benchmarks`  — 6 specs (+ MRT/ZF/RZF/LR-MMSE)
:func:`all_algorithms`        — 9 specs (+ Global-MRT/Global-ZF)

API match with Stage 7 / Stage 6d
---------------------------------
* ``AlgorithmSpec(name, kind, params)`` from
  ``cordis.simulation.scenario``.  ``kind`` is one of the four entries
  in :data:`ALGORITHM_KINDS`; benchmarks use ``kind="benchmark"`` with
  the actual baseline keyed by ``params["benchmark_name"]``.
* Common ``params`` keys (forwarded by the dispatcher):

  - ``cordis_split``  : ``comm_bf_method``, ``gamma_u_db``, ``omega``,
                        ``use_cvxpy``
  - ``cordis_admm``   : ``gamma_u_db``, ``omega``, ``rho_admm``,
                        ``kappa``, ``xi_slack`` / ``slack_tol``,
                        ``n_admm_max``, ``eps_pri``, ``eps_dual``,
                        ``warm_start_from_split``
  - ``centralized``   : ``gamma_u_db``, ``omega``, ``use_cvxpy``,
                        ``warm_start``
  - ``benchmark``     : ``benchmark_name`` (required), ``gamma_u_db``,
                        ``omega``, ``use_cvxpy``

Note that ``gamma_u_db`` may be passed as a scalar — the dispatcher
broadcasts it across all UEs — so spec builders here don't need to
know ``n_ue`` at spec-build time.

Spec builders silently absorb unrecognised keyword arguments
(``**ignored``) so spec-set factories can pass a single uniform kwarg
bundle (γ, κ, ω, …) and each builder picks up only the parameters it
consumes.  This makes parameter sweeps trivial: vary one kwarg, every
relevant spec responds, irrelevant ones drop it.
"""
from __future__ import annotations

from typing import Any, Dict, List

from cordis.simulation.scenario import AlgorithmSpec


# ─────────────────────────────────────────────────────────────────────
#  Benchmark names (must match cordis.algorithms.benchmarks.BENCHMARK_REGISTRY)
# ─────────────────────────────────────────────────────────────────────

_BENCHMARK_NAME: Dict[str, str] = {
    "mrt":          "mrt_split",
    "zf":           "zf_split",
    "rzf":          "rzf_split",
    "lrmmse":       "lr_mmse_split",
    "global_mrt":   "global_mrt_split",
    "global_zf":    "global_zf_split",
    # PA-fixed variants — not exposed by default but available for
    # specialised comparisons (e.g. an "LR-MMSE no PA" line).
    "mrt_fixed":    "mrt_fixed",
    "rzf_fixed":    "rzf_fixed",
    "lrmmse_fixed": "lr_mmse_fixed",
}

# ── Display names matching cordis.plotting.style.ALGORITHM_STYLE keys ─

_DISPLAY: Dict[str, str] = {
    "split":         "CORDIS-Split",
    "admm":          "CORDIS-ADMM",
    "centralized":   "Centralized",
    "mrt":           "MRT-Split",        # local MRT + P-Split PA
    "zf":            "ZF-Split",         # local ZF  + P-Split PA
    "rzf":           "RZF-Split",        # local RZF + P-Split PA
    "lrmmse":        "LR-MMSE-Split",    # local LR-MMSE + P-Split PA
    "global_mrt":    "Global-MRT",
    "global_zf":     "Global-ZF",
}


# ─────────────────────────────────────────────────────────────────────
#  Sensible defaults (consumed by the spec builders)
# ─────────────────────────────────────────────────────────────────────

DEFAULT_GAMMA_DB    = 3.0     # per-UE SINR target [dB]
DEFAULT_KAPPA       = 0.1     # prox regularisation weight
DEFAULT_OMEGA       = 0.5     # comm/sensing trade-off in [0, 1]
DEFAULT_RHO_ADMM    = 10.0    # ADMM penalty parameter
DEFAULT_N_ADMM_MAX  = 20
DEFAULT_XI_SLACK    = 1e-3    # ADMM slack tolerance (a.k.a. slack_tol)

# Backwards-compatible aliases (some Stage 8c scripts may use the
# paper's λ, ρ, slack_tol naming).
DEFAULT_LAMBDA      = DEFAULT_OMEGA
DEFAULT_RHO         = DEFAULT_RHO_ADMM
DEFAULT_SLACK_TOL   = DEFAULT_XI_SLACK


# ─────────────────────────────────────────────────────────────────────
#  Individual spec builders
#
#  Each builder takes only the parameters it understands as explicit
#  keyword arguments, then absorbs everything else into ``**ignored``.
#  This pattern lets spec-set factories pass a uniform kwarg bundle
#  and each builder silently drops what it doesn't use — exactly the
#  behaviour we want for parameter sweeps.
# ─────────────────────────────────────────────────────────────────────

def split_spec(
    *,
    gamma_u_db: float = DEFAULT_GAMMA_DB,
    omega:      float = DEFAULT_OMEGA,
    comm_bf_method: str = "lr_mmse",
    use_cvxpy:  bool  = False,
    **ignored,
) -> AlgorithmSpec:
    """CORDIS-Split (Algorithm 1): local-BF + centralised P-Split PA."""
    return AlgorithmSpec(
        name=_DISPLAY["split"],
        kind="cordis_split",
        params={
            "gamma_u_db":     float(gamma_u_db),
            "omega":          float(omega),
            "comm_bf_method": str(comm_bf_method),
            "use_cvxpy":      bool(use_cvxpy),
        },
    )


def admm_spec(
    *,
    gamma_u_db: float = DEFAULT_GAMMA_DB,
    omega:      float = DEFAULT_OMEGA,
    kappa:      float = DEFAULT_KAPPA,
    rho_admm:   float = DEFAULT_RHO_ADMM,
    n_admm_max: int   = DEFAULT_N_ADMM_MAX,
    xi_slack:   float = DEFAULT_XI_SLACK,
    warm_start_from_split: bool = True,
    **ignored,
) -> AlgorithmSpec:
    """CORDIS-ADMM (Algorithm 2): decentralised consensus-ADMM.

    Note
    ----
    ``kappa`` is explicitly placed in ``spec.params`` to work around the
    Stage 7 dispatcher gap (cfg.algorithm.admm.kappa was not being
    forwarded to the solver).  Keeping it in spec.params guarantees the
    value reaches the solver regardless of where cfg defaults live.
    """
    return AlgorithmSpec(
        name=_DISPLAY["admm"],
        kind="cordis_admm",
        params={
            "gamma_u_db":            float(gamma_u_db),
            "omega":                 float(omega),
            "rho_admm":              float(rho_admm),
            "kappa":                 float(kappa),
            "xi_slack":              float(xi_slack),
            "n_admm_max":            int(n_admm_max),
            "warm_start_from_split": bool(warm_start_from_split),
        },
    )


def centralized_spec(
    *,
    gamma_u_db: float = DEFAULT_GAMMA_DB,
    omega:      float = DEFAULT_OMEGA,
    use_cvxpy:  bool  = False,
    warm_start: bool  = True,
    **ignored,
) -> AlgorithmSpec:
    """Centralized joint BF + PA (upper bound)."""
    return AlgorithmSpec(
        name=_DISPLAY["centralized"],
        kind="centralized",
        params={
            "gamma_u_db": float(gamma_u_db),
            "omega":      float(omega),
            "use_cvxpy":  bool(use_cvxpy),
            "warm_start": bool(warm_start),
        },
    )


def _benchmark_spec(
    kind: str,
    *,
    gamma_u_db: float = DEFAULT_GAMMA_DB,
    omega:      float = DEFAULT_OMEGA,
    use_cvxpy:  bool  = False,
    **ignored,
) -> AlgorithmSpec:
    """Generic builder for any registered benchmark."""
    return AlgorithmSpec(
        name=_DISPLAY[kind],
        kind="benchmark",
        params={
            "benchmark_name": _BENCHMARK_NAME[kind],
            "gamma_u_db":     float(gamma_u_db),
            "omega":          float(omega),
            "use_cvxpy":      bool(use_cvxpy),
        },
    )


def mrt_spec(**kw)        -> AlgorithmSpec: return _benchmark_spec("mrt",        **kw)
def zf_spec(**kw)         -> AlgorithmSpec: return _benchmark_spec("zf",         **kw)
def rzf_spec(**kw)        -> AlgorithmSpec: return _benchmark_spec("rzf",        **kw)
def lrmmse_spec(**kw)     -> AlgorithmSpec: return _benchmark_spec("lrmmse",     **kw)
def global_mrt_spec(**kw) -> AlgorithmSpec: return _benchmark_spec("global_mrt", **kw)
def global_zf_spec(**kw)  -> AlgorithmSpec: return _benchmark_spec("global_zf",  **kw)


# ─────────────────────────────────────────────────────────────────────
#  Spec set factories
# ─────────────────────────────────────────────────────────────────────

def cordis_only(**kw) -> List[AlgorithmSpec]:
    """2 specs: CORDIS-Split + CORDIS-ADMM."""
    return [split_spec(**kw), admm_spec(**kw)]


def cordis_vs_centralized(**kw) -> List[AlgorithmSpec]:
    """3 specs: CORDIS-Split + CORDIS-ADMM + Centralized.

    Useful for figures where you want to show that the distributed
    methods approach the centralised upper bound.
    """
    return [split_spec(**kw), admm_spec(**kw), centralized_spec(**kw)]


def cordis_vs_benchmarks(**kw) -> List[AlgorithmSpec]:
    """6 specs: + 4 local-BF + optimised-PA benchmarks (MRT/ZF/RZF/LR-MMSE).

    Useful for figures showing that CORDIS outperforms classical
    local-BF + PA approaches at the same operating point.
    """
    return [
        split_spec(**kw),
        admm_spec(**kw),
        mrt_spec(**kw),
        zf_spec(**kw),
        rzf_spec(**kw),
        lrmmse_spec(**kw),
    ]


def all_algorithms(**kw) -> List[AlgorithmSpec]:
    """9 specs: everything in the algorithm style table.

    Used by CDF figures and broad sweeps where many curves are wanted.
    """
    return [
        split_spec(**kw),
        admm_spec(**kw),
        centralized_spec(**kw),
        mrt_spec(**kw),
        zf_spec(**kw),
        rzf_spec(**kw),
        lrmmse_spec(**kw),
        global_mrt_spec(**kw),
        global_zf_spec(**kw),
    ]


__all__ = [
    # Display-name + benchmark-name lookup tables
    "_DISPLAY", "_BENCHMARK_NAME",
    # Defaults
    "DEFAULT_GAMMA_DB", "DEFAULT_KAPPA", "DEFAULT_OMEGA",
    "DEFAULT_RHO_ADMM", "DEFAULT_N_ADMM_MAX", "DEFAULT_XI_SLACK",
    # Backwards-compat aliases
    "DEFAULT_LAMBDA", "DEFAULT_RHO", "DEFAULT_SLACK_TOL",
    # Individual spec builders
    "split_spec", "admm_spec", "centralized_spec",
    "mrt_spec", "zf_spec", "rzf_spec", "lrmmse_spec",
    "global_mrt_spec", "global_zf_spec",
    # Spec set factories
    "cordis_only", "cordis_vs_centralized",
    "cordis_vs_benchmarks", "all_algorithms",
]

