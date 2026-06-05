# Fig. — Communication–sensing tradeoff / served region

Two-panel, double-column figure for the Simulation Results section, from a
`gamma_sweep` campaign.

- **(a) Tradeoff vs γ** — achieved SINR (left axis, solid) and the weighted
  sum-SCNR (right axis, dashed, `weighted_sum_scnr_db` = Σ_t ω_t·SCNR_{a_r,t},
  conditioned on the served/feasible trials). A grey `y = x` guide marks the
  "target met" line on the SINR axis.
- **(b) Served / feasible region vs γ** — where the operating region is read
  off. The definition is selectable (see below); with the favorable config
  CORDIS-ADMM tracks Centralized at low–mid γ and falls off at high γ.

The SINR curve and the region curve are over all successful trials; the SCNR
is conditional on the served/feasible trials (`--scnr-all-trials` to disable).
Centralized is the one-shot ceiling; CORDIS-ADMM/Split return their best
feasible iterate. Per-algorithm style comes from `style_for` / `ALGORITHM_STYLE`.

## Two selectable definitions (defaults match the paper)

**SINR metric — `--sinr-metric {min,mean}`** (default `min`). Panel (a)'s SINR
curve is the per-trial worst-user SINR (`min_sinr_db`) or the across-user mean
(`mean_sinr_db` = dB of the per-trial mean over users). The y-axis label and the
legend's solid-line key update accordingly.

**Feasibility / region — `--feasibility {served,outage,strict}`** (default
`served`), following `cordis/metrics/outage.py`:

- `served` (**moderate, default**) — served-trial rate: fraction of trials in
  which at least η of users meet γ (`--eta`, default 0.9). Panel (b) and the
  conditional-SCNR mask both use it.
- `outage` (moderate) — per-user coverage `1 − Pr(SINR_u < γ)` over the
  (user, trial) pool. SCNR is then conditioned over all trials (the per-user
  view has no per-trial served mask).
- `strict` (legacy) — `1 − infeasibility_rate(γ, sinr_metric)` (all-or-nothing,
  the old behaviour); SCNR conditioned over strictly-feasible trials.

Panel (b)'s y-label and title adapt to the mode ("served region" / "coverage" /
"feasible region"). The moderate definitions are the operating point cell-free /
massive-MIMO papers actually report.

## Files

| File | Purpose |
|---|---|
| `build_fig_cs_tradeoff.py` | Headless builder **and** importable module (loader, `collect_curves`, `build_figure`). Writes `../../figures/fig_cs_tradeoff.pdf`. |
| `fig_cs_tradeoff.ipynb` | Interactive front-end. The **committed PDF is built from the module** (`B.build_figure`) for reproducibility; a verbatim, IEEE-styled `build_figure_editable` cell is provided for live tweaking (carries the same `sinr_metric` / `feasibility` / `eta` options). |
| `README.md` | This file. |

No runner — point the builder at a `gamma_sweep` run (or let it auto-discover
the newest under `results/exp_gamma_sweep/`).

## How to build

```bash
# Newest run; moderate 'served' region (η=0.9), worst-user SINR:
python3 paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py

# Mean-SINR axis + per-user coverage; no pdflatex:
python3 paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py \
    --sinr-metric mean --feasibility outage --no-tex

# Legacy strict definition, pinned run:
python3 paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py \
    --feasibility strict --result-dir results/exp_gamma_sweep/<ts>
```

Other flags: `--eta`, `--scnr-all-trials`, `--no-sinr-band`, `--feas-percent`,
`--only`, `--out`. The builder prints the region rate by γ for a sanity check.

## γ grid

`[0, 2.5, 5, 7.5, 10, 12.5, 15]` (clean 2.5 dB spacing; includes γ\*=5 for
cross-figure consistency; reaches into the infeasible regime so panel (b) shows
the fall-off). The builder reads the γ values from the result keys, so changing
the grid needs no code edit.

## Note on the region definition vs the SINR metric

The moderate metrics (`served`, `outage`) are computed on the per-user SINR pool
/ per-user coverage and are therefore **independent of `--sinr-metric`**, which
only changes the plotted SINR curve in panel (a). Only `strict` uses the chosen
SINR metric to classify feasibility.

