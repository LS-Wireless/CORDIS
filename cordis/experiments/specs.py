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
:func:`split_spec`       — CORDIS-Split             ``kind="cordis_split"``
:func:`admm_spec`        — CORDIS-ADMM              ``kind="cordis_admm"``
:func:`centralized_spec` — Centralized joint BF+PA  ``kind="centralized"``
:func:`mrt_spec`         — Local MRT + P-Split PA   ``benchmark_name="mrt_split"``
:func:`zf_spec`          — Local ZF + P-Split PA    ``benchmark_name="zf_split"``
:func:`rzf_spec`         — Local RZF + P-Split PA   ``benchmark_name="rzf_split"``
:func:`lrmmse_spec`      — Local LR-MMSE + fixed ρ=0.5  ``benchmark_name="lr_mmse_fixed"``
:func:`global_mrt_spec`  — Global MRT + P-Split PA  ``benchmark_name="global_mrt_split"``
:func:`global_zf_spec`   — Global ZF  + P-Split PA  ``benchmark_name="global_zf_split"``

PSR-baseline variants (paper "benefit of optimal PSR" figure)
-------------------------------------------------------------
:func:`lrmmse_p020_spec` — LR-MMSE + ρ=0.2  ``benchmark_name="lr_mmse_fixed_p020"``
:func:`lrmmse_p080_spec` — LR-MMSE + ρ=0.8  ``benchmark_name="lr_mmse_fixed_p080"``
:func:`lrmmse_split_spec`— LR-MMSE + P-Split PA (≡ CORDIS-Split; legacy alias)

Spec set factories (composition)
--------------------------------
:func:`cordis_only`           — 2 specs
:func:`cordis_vs_centralized` — 3 specs
:func:`cordis_vs_benchmarks`  — 6 specs (+ MRT/ZF/RZF/LR-MMSE@ρ=0.5)
:func:`all_algorithms`        — 9 specs (+ Global-MRT/Global-ZF)
:func:`psr_baselines`         — 4 specs (CORDIS-Split + LR-MMSE@{ρ=0.2,0.5,0.8})

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

* Config as single source of truth (paper revision).  Tuning knobs
  with a counterpart in ``cfg.algorithm.*`` default to ``None`` here:
  ``gamma_u_db``, ``kappa``, ``rho_admm``, ``n_admm_max``,
  ``eps_pri``, ``eps_dual``, ``xi_slack``.  The corresponding
  spec.params entry is only emitted if the caller explicitly passes
  a value.  Every ``run_*`` function in
  ``cordis.experiments.registry`` calls ``_admm_kwargs_from_cfg(cfg)``
  to extract these from cfg and forward them, so user config takes
  effect — but the JSON config remains the only place the value lives.
  No shadow defaults; no DEFAULT_* constants in this module.

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
    # ``lrmmse`` is the default benchmark exposed in cordis_vs_benchmarks /
    # all_algorithms.  Before the paper revision it pointed at
    # ``lr_mmse_split`` (LR-MMSE Phase-I + optimal P-Split Phase-II PA),
    # which is bit-for-bit identical to CORDIS-Split — so including it
    # alongside CORDIS-Split produced overlapping curves.  It now points
    # at the fixed-ρ=0.5 variant, which is a meaningful baseline (no PA
    # optimisation) for the LR-MMSE + fixed-PSR family.
    "lrmmse":       "lr_mmse_fixed",
    "global_mrt":   "global_mrt_split",
    "global_zf":    "global_zf_split",
    # PA-fixed variants (for the psr_baselines spec set used by the
    # paper's "benefit of optimal PSR" figure).
    "mrt_fixed":      "mrt_fixed",
    "rzf_fixed":      "rzf_fixed",
    "lrmmse_p020":    "lr_mmse_fixed_p020",   # sensing-biased
    "lrmmse_p080":    "lr_mmse_fixed_p080",   # comm-biased
    # Legacy alias: explicit access to the LR-MMSE + optimal P-Split PA
    # path (= CORDIS-Split numerically).  Not in any default spec set;
    # exposed so older notebooks / debugging scripts can reach it.
    "lrmmse_split":   "lr_mmse_split",
}

# ── Display names matching cordis.plotting.style.ALGORITHM_STYLE keys ─
# LaTeX-mode-safe: ρ is rendered via mathmode "$\\rho$" so the same
# string works under usetex=True (paper figures), usetex=False
# (mathtext fallback), AND in plain-terminal output via list_benchmarks
# (which just prints the literal string).

_DISPLAY: Dict[str, str] = {
    "split":         "CORDIS-Split",
    "admm":          "CORDIS-ADMM",
    "centralized":   "Centralized",
    "mrt":           "MRT-Split",                       # local MRT + P-Split PA
    "zf":            "ZF-Split",                        # local ZF  + P-Split PA
    "rzf":           "RZF-Split",                       # local RZF + P-Split PA
    "lrmmse":        r"LR-MMSE ($\rho$=0.5)",           # fixed PSR (new default)
    "lrmmse_p020":   r"LR-MMSE ($\rho$=0.2)",           # sensing-biased
    "lrmmse_p080":   r"LR-MMSE ($\rho$=0.8)",           # comm-biased
    "lrmmse_split":  "LR-MMSE (P-Split PA)",            # legacy alias (≡ CORDIS-Split)
    "global_mrt":    "Global-MRT",
    "global_zf":     "Global-ZF",
}


# ─────────────────────────────────────────────────────────────────────
#  Sensible defaults — aligned with configs/default.json so a spec
#  built with no overrides matches what cfg promises.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
#  Sentinel-default policy (paper revision)
#
#  Every ADMM tuning knob that has a counterpart in
#  ``cfg.algorithm.admm.*`` defaults to None here.  ``None`` means
#  "let the algorithm read the value from cfg" — the spec.params dict
#  is built conditionally via :func:`_admm_params`, so omitted knobs
#  aren't injected.  This keeps the JSON config the single source of
#  truth and prevents the out-of-sync-defaults pitfall where bumping
#  cfg.algorithm.admm.n_max silently has no effect (the old
#  DEFAULT_N_ADMM_MAX=50 would override it).
#
#  All six knobs follow the same pattern: ``kappa``, ``rho_admm``,
#  ``n_admm_max``, ``eps_pri``, ``eps_dual``, ``xi_slack``.  Under
#  normal use, :func:`_admm_kwargs_from_cfg` in registry.py extracts
#  them from cfg and forwards them as kwargs — so the cfg values DO
#  end up in spec.params and the algorithm receives them.
# ─────────────────────────────────────────────────────────────────────


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


def _admm_params(
    *,
    kappa:      Optional[float] = None,
    rho_admm:   Optional[float] = None,
    n_admm_max: Optional[int]   = None,
    eps_pri:    Optional[float] = None,
    eps_dual:   Optional[float] = None,
    xi_slack:   Optional[float] = None,
) -> dict:
    """Conditionally pack ADMM tuning knobs into spec.params.

    Each knob is included only if the caller explicitly passed a value
    (i.e. not ``None``).  ``run_*`` functions in ``registry.py`` call
    ``_admm_kwargs_from_cfg(cfg)`` to extract these from
    ``cfg.algorithm.admm.*`` and then forward them as kwargs — so under
    normal use the cfg values DO end up in spec.params and the
    algorithm receives them.  When called directly with no kwargs
    (e.g. ``admm_spec(n_ue=4)``), the spec carries no ADMM tuning
    keys and the algorithm reads cfg in its dispatcher.
    """
    out: dict = {}
    if kappa is not None:      out["kappa"]      = float(kappa)
    if rho_admm is not None:   out["rho_admm"]   = float(rho_admm)
    if n_admm_max is not None: out["n_admm_max"] = int(n_admm_max)
    if eps_pri is not None:    out["eps_pri"]    = float(eps_pri)
    if eps_dual is not None:   out["eps_dual"]   = float(eps_dual)
    if xi_slack is not None:   out["xi_slack"]   = float(xi_slack)
    return out


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
    n_ue:       int             = 1,
    gamma_u_db: Optional[float] = None,
    kappa:      Optional[float] = None,
    rho_admm:   Optional[float] = None,
    n_admm_max: Optional[int]   = None,
    eps_pri:    Optional[float] = None,
    eps_dual:   Optional[float] = None,
    xi_slack:   Optional[float] = None,
    **ignored,
) -> AlgorithmSpec:
    """CORDIS-ADMM (Algorithm 2): decentralised consensus-ADMM.

    All six ADMM tuning knobs (``kappa``, ``rho_admm``, ``n_admm_max``,
    ``eps_pri``, ``eps_dual``, ``xi_slack``) default to ``None`` and are
    only added to spec.params if the caller passes an explicit value
    (the ``_admm_params`` sentinel pattern).  Under normal use,
    :func:`_admm_kwargs_from_cfg` in ``cordis/experiments/registry.py``
    extracts them from ``cfg.algorithm.admm.*`` for every ``run_*``
    function and forwards them, so user config takes effect — but the
    JSON config remains the single source of truth (no shadow defaults
    here).
    """
    return AlgorithmSpec(
        name=_DISPLAY["admm"],
        kind="cordis_admm",
        params={
            **_gamma_params(gamma_u_db, n_ue),
            **_admm_params(
                kappa=kappa, rho_admm=rho_admm,
                n_admm_max=n_admm_max,
                eps_pri=eps_pri, eps_dual=eps_dual,
                xi_slack=xi_slack,
            ),
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

# ── PSR-baseline variants ────────────────────────────────────────────
# These three lr_mmse_* specs share the same Phase-I (LR-MMSE BF) but
# differ only in the (uniform) fixed power-splitting ratio ρ.  Used
# together with split_spec to demonstrate the benefit of CORDIS-Split's
# Phase-II PSR optimisation: any fixed ρ choice is dominated by the
# optimised allocation.
def lrmmse_p020_spec(**kw)  -> AlgorithmSpec: return _benchmark_spec("lrmmse_p020", **kw)
def lrmmse_p080_spec(**kw)  -> AlgorithmSpec: return _benchmark_spec("lrmmse_p080", **kw)
# Legacy alias — LR-MMSE + optimal P-Split PA (numerically ≡ CORDIS-Split).
# Kept for explicit access; not in any default spec set.
def lrmmse_split_spec(**kw) -> AlgorithmSpec: return _benchmark_spec("lrmmse_split", **kw)


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
    """6 specs: + 4 local-BF benchmarks (MRT/ZF/RZF/LR-MMSE).

    Useful for figures showing that CORDIS outperforms classical
    local-BF approaches at the same operating point.

    Note on the LR-MMSE entry: this is LR-MMSE Phase-I BF + fixed
    ρ=0.5 (no PA optimisation), not the legacy LR-MMSE + P-Split PA
    which is numerically identical to CORDIS-Split.  See
    ``cordis.algorithms.benchmarks.BENCHMARK_REGISTRY``.
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


def psr_baselines(**kw) -> List[AlgorithmSpec]:
    """4 specs: CORDIS-Split + 3 LR-MMSE fixed-ρ variants.

    Designed for the paper's "benefit of optimal PSR" figure.
    Compares CORDIS-Split's Phase-II optimal power-splitting ratio
    against three fixed-ρ baselines (sharing the same LR-MMSE Phase-I
    BF):

        - LR-MMSE (ρ=0.2): sensing-biased
        - LR-MMSE (ρ=0.5): balanced (same as default lrmmse benchmark)
        - LR-MMSE (ρ=0.8): comm-biased

    The expected story: CORDIS-Split dominates every fixed-ρ point,
    showing that the optimal ρ depends on the operating regime and
    that Phase-II is non-trivially better than any single fixed choice.
    """
    return [
        split_spec(**kw),
        lrmmse_p020_spec(**kw),
        lrmmse_spec(**kw),         # ρ=0.5 (default lr_mmse benchmark)
        lrmmse_p080_spec(**kw),
    ]


__all__ = [
    # Display-name + benchmark-name lookup tables
    "_DISPLAY", "_BENCHMARK_NAME",
    # Individual spec builders
    "split_spec", "admm_spec", "centralized_spec",
    "mrt_spec", "zf_spec", "rzf_spec", "lrmmse_spec",
    "global_mrt_spec", "global_zf_spec",
    # PSR-baseline variants (paper "benefit of optimal PSR" figure)
    "lrmmse_p020_spec", "lrmmse_p080_spec", "lrmmse_split_spec",
    # Spec set factories
    "cordis_only", "cordis_vs_centralized",
    "cordis_vs_benchmarks", "all_algorithms",
    "psr_baselines",
]

