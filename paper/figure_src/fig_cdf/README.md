# Fig. — SINR CDF + SCNR CDF at γ\*

Two-panel empirical-CDF figure for the Simulation Results section, evaluated at
the chosen operating point **γ\*** (the per-user SINR floor used across the
paper's headline figures).

- **(a) SINR CDF at γ\*** — empirical CDF of the per-trial SINR for each
  algorithm, with a vertical dashed line at γ\*. The SINR metric is selectable
  via `--sinr-metric`: `min` (worst-user, default) or `mean` (across users,
  = `mean_sinr_db`). Each legend entry is annotated with the **moderate-outage**
  feasibility scalar at γ\* (see below).
- **(b) SCNR CDF at γ\*** — empirical CDF of the per-trial sensing SCNR at the
  **same** operating point. No γ line: γ\* is an SINR floor, not an SCNR target.

### Feasibility / outage definition (moderate by default)

The legend annotation and the summary's feasibility column use a selectable
definition (`--feasibility`), following `cordis/metrics/outage.py`:

- `served` (**default, moderate**) — served-trial rate: fraction of trials in
  which at least η of users meet γ\* (`--eta`, default 0.9). Higher is better.
- `outage` (moderate) — per-user outage probability Pr(SINR_u < γ\*) over the
  (user, trial) pool (= the per-user SINR CDF read at γ\*). Lower is better.
- `strict` (legacy) — fraction of trials whose chosen SINR metric < γ\*
  (all-or-nothing; the old behaviour). Lower is better.

The summary also reports the (1−ε)-likely per-user SINR (ε=0.05 → 95%-likely),
the Björnson/cell-free convention. **Note:** the moderate metrics (`served`,
`outage`) are computed on the per-user SINR pool and are therefore independent
of the `--sinr-metric` choice, which only affects the plotted SINR curve.

Both panels use the project-wide per-algorithm style (`style_for` /
`ALGORITHM_STYLE`), so the same algorithm reads identically here and in
`fig_cs_tradeoff`. The algorithm set defaults to the proposed pair against the
centralized ceiling: `Centralized`, `CORDIS-ADMM`, `CORDIS-Split` (the CDF
experiments run `all_algorithms`, so the builder filters down).

## Files

| File | Purpose |
|---|---|
| `build_fig_cdf.py` | Headless builder **and** importable module (the notebook reuses its functions). Writes `../../figures/fig_cdf.pdf`. |
| `fig_cdf.ipynb` | Interactive front-end; imports `build_fig_cdf` so plotting logic never forks. |
| `README.md` | This file. |

There is **no runner** — point the builder at a CDF campaign you copied from the
cluster (or let it auto-discover the newest one).

## Data sources

The figure index pairs this figure with the `sinr_cdf` (panel a) and `scnr_cdf`
(panel b) experiments. Both experiments compute the full per-trial SINR *and*
SCNR statistics, so:

- if both campaigns have finished, panel (a) is drawn from `sinr_cdf` and panel
  (b) from `scnr_cdf`;
- if only one has finished, the builder **reuses that single run for both
  panels** and prints a note — handy while the second campaign is still queued.

Discovery is HPC-aware: it prefers the merged `array_<jobid>_aggregated/`
directory (highest job ID) and **never** trusts the `latest` symlink, which
`sync_results_from_hpc3.sh` can leave pointing at a per-task dir.

## How to build

From the repo root (or anywhere — paths auto-resolve):

```bash
# Newest aggregated runs, γ* auto-detected from metadata:
python3 paper/figure_src/fig_cdf/build_fig_cdf.py

# On a node without pdflatex:
python3 paper/figure_src/fig_cdf/build_fig_cdf.py --no-tex

# Pin the operating point and the sensing metric for a paper-grade figure:
python3 paper/figure_src/fig_cdf/build_fig_cdf.py --gamma-db 5 --scnr-metric min_scnr_db

# Pin specific runs:
python3 paper/figure_src/fig_cdf/build_fig_cdf.py \
    --sinr-dir results/exp_sinr_cdf/array_12345_aggregated \
    --scnr-dir results/exp_scnr_cdf/array_12346_aggregated
```

The builder prints a per-algorithm summary (median & 5th-pct min-SINR,
infeasibility at γ\*, median SCNR) — copy those numbers straight into the
caption / running text.

## Metric notes

- **min-SINR** uses `min_sinr_db` (worst-user per trial), the same metric the
  feasibility definition in `fig_cs_tradeoff` is built on.
- **SCNR** defaults to the first available of `min_scnr_db` →
  `weighted_sum_scnr_db` → `mean_scnr_db`. For the draft's single-target
  scenario, `min_scnr_db` (worst-target SCNR) is the natural dual of the
  worst-user min-SINR — both panels then read as worst-case CDFs. Override with
  `--scnr-metric` if you want the sum-SCNR view to match a sweep figure.
- **γ\*** is auto-detected from run metadata (any `*gamma*` key); override with
  `--gamma-db`. The fallback when metadata lacks it is the cross-figure default.

## Research-integrity note

As agreed for the CDF and feasibility figures, report the **stress scenario**
(`configs/scenarios/stress.json`) alongside the favorable-config version when
this figure goes in the paper — disclosing the favorable config and also showing
the harder regime. Re-run the campaign under the stress override and point the
builder at that run with `--sinr-dir` / `--scnr-dir`.

