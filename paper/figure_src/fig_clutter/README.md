# Fig. — Clutter suppression (κ sweep)

Stacked two-panel, single-column figure for the Simulation Results section,
from a `kappa_sweep` campaign.

- **(top) SINR vs κ** — the communication cost of clutter avoidance. As κ grows
  the beamformer spends degrees of freedom steering a null toward the clutter
  PAS, so the SINR drifts down.
- **(bottom) SCNR vs κ** — the sensing benefit ("suppression"). As κ grows
  clutter leakage into the receive filter falls and SCNR climbs, saturating
  once the clutter is essentially nulled.

Each panel's central line is the across-trial mean; an optional shaded band
shows dispersion. A vertical dashed line marks the deployed operating point
**κ\*** (= `ADMM_KAPPA`, 0.08), so the reader sees the shipped system sits in
the gentle-cost / near-saturated-gain region.

Filtered to the proposed pair vs the centralized ceiling (`Centralized`,
`CORDIS-ADMM`, `CORDIS-Split`), styled via `style_for` / `ALGORITHM_STYLE`.

## Selectable metrics and band (defaults match the paper)

**SINR metric — `--sinr-metric {min,mean}`** (default `min`):

| key | registry metric | axis label |
|---|---|---|
| `min` | `min_sinr_db` | min-user SINR |
| `mean` | `mean_sinr_db` | mean-user SINR |

**SCNR metric — `--scnr-metric {min,mean,wsum}`** (default `wsum`):

| key | registry metric | axis label | legend short |
|---|---|---|---|
| `min` | `min_scnr_db` | min-target SCNR | min-target SCNR |
| `mean` | `mean_scnr_db` | mean-target SCNR | mean SCNR |
| `wsum` | `weighted_sum_scnr_db` | weighted-sum SCNR | w-sum SCNR |

`wsum` is the weighted target-SCNR sum Sigma_t omega_t*SCNR_{a_r,t} (there is no
plain `sum_scnr_db` in the registry).

**Dispersion band — `--band {std,iqr,none}`** (default `std`): the shaded band
around each mean curve is the standard deviation (`std`), the inter-quartile
range (`iqr`), or removed entirely (`none`). Passed straight to `plot_sweep`'s
`error` argument.

> This figure plots *achieved* SINR/SCNR vs κ, so it has no feasibility/served
> panel — the moderate-outage metrics don't apply here.

## Needs the *offset* clutter geometry

With `clutter_center_strategy='target_centroid'` the clutter PAS sits on top of
the target, so penalising clutter also penalises the target echo and SCNR
barely moves with κ (geometric, not a bug). The `kappa_sweep` recipe inherits
the `offset` strategy from `_defaults.sh` — keep it for this figure.

## Files

| File | Purpose |
|---|---|
| `build_fig_clutter.py` | Headless builder **and** importable module (loader + helpers + `build_figure`). Writes `../../figures/fig_clutter.pdf`. |
| `fig_clutter.ipynb` | Interactive front-end. Loader/summary helpers imported from the module; **`build_figure` inlined as an editable cell** with `SINR_METRIC` / `SCNR_METRIC` / `BAND` / `KAPPA_STAR` knobs. |
| `README.md` | This file. |

No runner — point the builder at a `kappa_sweep` run (or let it auto-discover
the newest, preferring `array_<jobid>_aggregated/`, never `latest`).

## How to build

```bash
# Newest aggregated run; min-user SINR, weighted-sum SCNR, std band, kappa*=0.08:
python3 paper/figure_src/fig_clutter/build_fig_clutter.py

# Mean-user SINR + min-target SCNR, band removed; no pdflatex:
python3 paper/figure_src/fig_clutter/build_fig_clutter.py \
    --sinr-metric mean --scnr-metric min --band none --no-tex

# Pin run / operating point:
python3 paper/figure_src/fig_clutter/build_fig_clutter.py \
    --kappa-star 0.08 --result-dir results/exp_kappa_sweep/array_12345_aggregated
```

The builder prints a per-algorithm summary (SCNR at kappa_min / kappa* /
kappa_max, the suppression gain, and the SINR cost at kappa*) for the caption.

## Campaign

`kappa_sweep` at the favorable config, `spec_set=cordis_vs_centralized`, with a
kappa grid that brackets kappa*=0.08 and reaches the saturated regime, e.g.
`kappa in {0, 0.02, 0.04, 0.08, 0.15, 0.3, 0.5, 1.0, 1.5, 2.0}` with
`CLUTTER_CNR_DB=0` (a louder clutter than the -10 dB default makes the
suppression curve legible). `N_TRIALS_TOTAL=400`, `N_ARRAY_TASKS=10`, aggregate
with `scripts/aggregate_array_batch.py kappa_sweep <jobid>`.

