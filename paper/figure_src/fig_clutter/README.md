# Fig. — Clutter suppression (κ sweep)

Stacked two-panel, single-column figure for the Simulation Results section,
showing how the clutter-penalty weight **κ** trades communication against
sensing in a `kappa_sweep` campaign.

- **(top) min-SINR vs κ** — the communication cost. As κ grows the beamformer
  spends degrees of freedom steering a null toward the clutter, so the
  worst-user SINR drifts down.
- **(bottom) sum-SCNR vs κ** — the sensing benefit (the "suppression"). As κ
  grows clutter leakage into the receive filter falls and SCNR climbs,
  saturating once the clutter is essentially nulled.
- A dashed line marks the **deployed operating point κ\*** (= `ADMM_KAPPA` in
  `_defaults.sh`, 0.08 by default), so the reader sees the shipped system sits
  in the gentle-cost / near-saturated-gain region rather than at an extreme.

Filtered to the proposed pair vs the centralized ceiling
(`Centralized`, `CORDIS-ADMM`, `CORDIS-Split`) and styled through
`style_for` / `ALGORITHM_STYLE`, so curves match `fig_cs_tradeoff` / `fig_cdf`.

## ⚠️ Geometry requirement: clutter must be *offset* from the target

With `clutter_center_strategy='target_centroid'` the clutter sits on top of the
target, so penalising clutter also penalises the target echo and **SCNR barely
moves with κ** — the figure goes flat. This is geometric, not a bug (see
`cordis/algorithms/joint_opt.py`). The `kappa_sweep` recipe inherits the
**`offset`** strategy from `configs/recipes/_defaults.sh`
(`CLUTTER_OFFSET_AZ_DEG=60`), which decouples clutter avoidance from comm
coverage and gives the rising-SCNR story. Keep it that way for this figure.

## Files

| File | Purpose |
|---|---|
| `build_fig_clutter.py` | Headless builder **and** importable module (loader + helpers + `build_figure`). Writes `../../figures/fig_clutter.pdf`. |
| `fig_clutter.ipynb` | Interactive front-end. Loader/summary helpers are imported from the module; **`build_figure` is inlined as an editable cell** so you can override style/layout without touching the module. |
| `README.md` | This file. |

No runner — point the builder at a `kappa_sweep` campaign you copied from the
cluster (or let it auto-discover the newest one).

## How to build

```bash
# Newest aggregated run, κ* marked at 0.08:
python3 paper/figure_src/fig_clutter/build_fig_clutter.py

# On a node without pdflatex:
python3 paper/figure_src/fig_clutter/build_fig_clutter.py --no-tex

# Pin the operating point / sensing metric / run:
python3 paper/figure_src/fig_clutter/build_fig_clutter.py \
    --kappa-star 0.08 --scnr-metric sum_scnr_db \
    --result-dir results/exp_kappa_sweep/array_12345_aggregated
```

The builder prints a per-algorithm summary — SCNR (median) at κ_min / κ\* /
κ_max, the **suppression gain** SCNR(κ_max)−SCNR(κ_min), and the **comm cost**
min-SINR(κ\*)−min-SINR(κ_min) — ready for the caption.

Discovery prefers `array_<jobid>_aggregated/` (highest job ID) and never the
`latest` symlink, same as the sibling builders.

## Recommended sweep grid and campaign config

See the chat notes, but in short: sweep κ over a grid that is dense at low κ
(to bracket κ\*=0.08 and capture the onset) and reaches the useful-range ceiling
κ=2, keep `clutter_center_strategy=offset`, set the clutter strong enough that
suppression is visible (`clutter_cnr_db ≈ 0`), hold the SINR floor at γ\*=5 dB,
and keep the drop count high (geometry variance dominates).

## Research-integrity note

As with the CDF/feasibility figures, report the **stress scenario**
(`configs/scenarios/stress.json`) alongside the favorable-config version: re-run
the κ-sweep under the stress override and point the builder at that run with
`--result-dir`.

