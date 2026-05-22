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

API match with the real algorithm stack
---------------------------------------
* ``AlgorithmSpec(name, kind, params)`` from
  ``cordis.simulation.scenario``.  ``kind`` is one of the four entries
  in ``ALGORITHM_KINDS``; benchmarks use ``kind="benchmark"`` with the
  actual baseline keyed by ``params["benchmark_name"]``.

* The journal-paper formulation (which Stage 6c/6d implement) has
  **no λ-style comm/sensing trade-off**.  SINR is a hard constraint,
  the sensing utility is maximised subject to it, and κ alone weights
  the clutter penalty.  These specs therefore expose only the knobs
  the real solvers actually consume; ``omega`` (per-target priority
  Dict[int, float]) is left at its default (uniform 1.0) and is NOT
  a tunable in the spec API — pass it directly via
  ``AlgorithmSpec.params`` if you need non-uniform priorities.

* ``gamma_u_db`` is required to be an array of length n_ue
  (downstream code does ``gamma_lin[u]``); spec builders therefore
  take ``n_ue`` and broadcast a scalar to a vector via
  :func:`numpy.full`.

* Stage-7 dispatcher gaps closed by these specs:
    - ``admm_spec`` explicitly forwards ``kappa`` (function default
      0.0 overrides the cfg-level 1.0 if not passed; the dispatcher
      never reads cfg.algorithm.admm.kappa).
    - ``admm_spec`` also forwards ``rho_admm`` and ``n_admm_max``
      so cfg values propagate.

* The following knobs are deliberately NOT placed in spec.params
  even though the dispatcher would forward them:
    - ``warm_start_from_split`` — not a parameter of
      ``solve_cordis_admm``; Phase-I warm start is always done.
    - ``warm_start`` — not a parameter of ``solve_centralized``
      (the real name is ``warm_start_psr``; dispatcher fwd_keys
      has the typo, so passing it would TypeError).
    - ``use_cvxpy`` — defaults to None (auto-detect), which is what
      we want; passing False forces the scipy fallback.

Spec builders silently absorb unrecognised keyword arguments
(``**ignored``) so spec-set factories can pass a single uniform kwarg
bundle (γ, κ, ρ, …) and each builder picks up only the parameters it
consumes.  This is the property parameter sweeps rely on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

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
    # PA-fixed variants — available but not in the default spec sets.
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
#  Sensible defaults — aligned with configs/default.json so a spec
#  built with no overrides matches what cfg promises.
# ─────────────────────────────────────────────────────────────────────

DEFAULT_KAPPA       = 1.0     # matches cfg.algorithm.admm.kappa
DEFAULT_RHO_ADMM    = 1.0     # matches cfg.algorithm.admm.rho
DEFAULT_N_ADMM_MAX  = 50      # matches cfg.algorithm.admm.n_max
DEFAULT_XI_SLACK    = 1e4     # matches cfg.algorithm.admm.xi_slack
DEFAULT_EPS_PRI     = 1.0     # matches cfg.algorithm.admm.eps_pri  (was 1e-3 in Stage-19b)
DEFAULT_EPS_DUAL    = 1.0     # matches cfg.algorithm.admm.eps_dual (was 1e-3 in Stage-19b)

# Sentinel used by spec builders.  ``gamma_u_db=None`` means "use the
# value from cfg.algorithm.gamma_db at solver invocation time" — i.e.
# don't put a gamma_u_db entry in spec.params.  Pass a number to
# override.  See cordis/algorithms/{joint_opt,centralized}.py for the
# fallback branch.


def _gamma_vec(gamma_u_db: float, n_ue: int) -> np.ndarray:
    """Per-UE SINR target vector (uniform across UEs).

    Required because the solvers index ``gamma_lin[u]`` per user — a
    scalar becomes a 0-d array which is unindexable.
    """
    return np.full(int(n_ue), float(gamma_u_db), dtype=float)


def _gamma_params(gamma_u_db: Optional[float], n_ue: int) -> dict:
    """Return either ``{"gamma_u_db": vec}`` or ``{}`` for spec.params.

    ``None`` means "let the algorithm read from cfg.algorithm.gamma_db".
    """
    if gamma_u_db is None:
        return {}
    return {"gamma_u_db": _gamma_vec(gamma_u_db, n_ue)}


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
    n_ue: int = 1,
    gamma_u_db: Optional[float] = None,
    **ignored,
) -> AlgorithmSpec:
    """CORDIS-Split (Algorithm 1): local-BF + centralised P-Split PA."""
    return AlgorithmSpec(
        name=_DISPLAY["split"],
        kind="cordis_split",
        params={
            **_gamma_params(gamma_u_db, n_ue),
            # comm_bf_method defaults to "lr_mmse" in the dispatcher.
            # omega / use_cvxpy left out → solver-side defaults apply.
        },
    )


def admm_spec(
    *,
    n_ue: int = 1,
    gamma_u_db: Optional[float] = None,
    kappa:      float = DEFAULT_KAPPA,
    rho_admm:   float = DEFAULT_RHO_ADMM,
    n_admm_max: int   = DEFAULT_N_ADMM_MAX,
    eps_pri:    float = DEFAULT_EPS_PRI,
    eps_dual:   float = DEFAULT_EPS_DUAL,
    xi_slack:   float = DEFAULT_XI_SLACK,
    **ignored,
) -> AlgorithmSpec:
    """CORDIS-ADMM (Algorithm 2): decentralised consensus-ADMM.

    Note
    ----
    ``kappa``, ``rho_admm``, ``n_admm_max``, ``eps_pri``, ``eps_dual``
    and ``xi_slack`` are placed explicitly in spec.params because the
    Stage-7 dispatcher does NOT read these from ``cfg.algorithm.admm.*``.
    Without them, ``solve_cordis_admm`` would silently fall back to its
    function-level defaults (Stage 19 originally caught this for
    kappa/rho_admm/n_admm_max; Stage 19b extends to eps_pri/eps_dual/
    xi_slack on the same audit).

    The ``_admm_kwargs_from_cfg(cfg)`` helper in
    ``cordis/experiments/registry.py`` extracts these from
    ``cfg.algorithm.admm.*`` for every ``run_*`` function so user
    config takes effect.
    """
    return AlgorithmSpec(
        name=_DISPLAY["admm"],
        kind="cordis_admm",
        params={
            **_gamma_params(gamma_u_db, n_ue),
            "kappa":       float(kappa),
            "rho_admm":    float(rho_admm),
            "n_admm_max":  int(n_admm_max),
            "eps_pri":     float(eps_pri),
            "eps_dual":    float(eps_dual),
            "xi_slack":    float(xi_slack),
            # slack_tol / warm_start_from_split deliberately NOT included
            # — slack_tol has no cfg counterpart; warm_start_from_split is
            # a dead config field (algorithm always warm-starts internally).
        },
    )


def centralized_spec(
    *,
    n_ue: int = 1,
    gamma_u_db: Optional[float] = None,
    **ignored,
) -> AlgorithmSpec:
    """Centralized joint BF + PA (upper bound).

    Note: ``solve_centralized`` reads κ from ``cfg.algorithm.admm.kappa``
    directly (the dispatcher does NOT forward κ for kind="centralized"),
    so the spec does not carry κ.  For a κ sweep, override
    ``cfg.algorithm.admm.kappa`` per sweep point — see
    :func:`cordis.experiments.registry.run_kappa_sweep`.
    """
    return AlgorithmSpec(
        name=_DISPLAY["centralized"],
        kind="centralized",
        params={
            **_gamma_params(gamma_u_db, n_ue),
            # warm_start / use_cvxpy intentionally omitted.
        },
    )


def _benchmark_spec(
    kind: str,
    *,
    n_ue: int = 1,
    gamma_u_db: Optional[float] = None,
    **ignored,
) -> AlgorithmSpec:
    """Generic builder for any registered benchmark."""
    return AlgorithmSpec(
        name=_DISPLAY[kind],
        kind="benchmark",
        params={
            "benchmark_name": _BENCHMARK_NAME[kind],
            **_gamma_params(gamma_u_db, n_ue),
            # omega / use_cvxpy left out → solver-side defaults apply.
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
    # Defaults (aligned with configs/default.json)
    "DEFAULT_KAPPA",
    "DEFAULT_RHO_ADMM", "DEFAULT_N_ADMM_MAX", "DEFAULT_XI_SLACK",
    # Individual spec builders
    "split_spec", "admm_spec", "centralized_spec",
    "mrt_spec", "zf_spec", "rzf_spec", "lrmmse_spec",
    "global_mrt_spec", "global_zf_spec",
    # Spec set factories
    "cordis_only", "cordis_vs_centralized",
    "cordis_vs_benchmarks", "all_algorithms",
]

