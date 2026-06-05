# Fig. — Robustness to imperfect CSI (pilot-power sweep)

Stacked two-panel, single-column figure for the Simulation Results section,
from a `csi_sweep` campaign (a sweep over the uplink pilot power P_p/sigma_n^2,
which sets the MMSE channel-estimation quality).

- **(top) communication metric vs pilot power** — selectable:
  - `min` : worst-user SINR [dB] (default)
  - `mean`: across-user SINR [dB]
  - `outage`: per-user outage probability Pr(SINR_u < gamma*) — the reliability
    cliff. gamma* is auto-detected from run metadata.
- **(bottom) SCNR vs pilot power** — selectable: `wsum` (weighted target-SCNR
  sum, default), `min` (worst target), `mean`.

Lower pilot power -> noisier estimates -> every algorithm degrades. The point
is the *shape*: the centralized bound and the CORDIS variants bend down
gracefully (or their outage rises gently) while interference-naive baselines
(e.g. MRT) cliff-edge sooner. So this figure plots **all** algorithms by
default (the trio leading, benchmarks appended). A dashed line marks the
operating point P_p* (= `PILOT_POWER_DB`, 128 dB).

## Selectable metrics and band (defaults match the paper)

**Communication metric — `--comm-metric {min,mean,outage}`** (default `min`):

| key | what | axis |
|---|---|---|
| `min` | `min_sinr_db` (min-user SINR) | dB |
| `mean` | `mean_sinr_db` (mean-user SINR) | dB |
| `outage` | per-user Pr(SINR_u < gamma*) | probability [0,1] |

For `outage`, set gamma* with `--gamma-db` (default: auto-detected from
metadata, else 5 dB). The outage line is the per-user pool fraction; its band
is the std / IQR of the per-trial outage fraction.

**SCNR metric — `--scnr-metric {min,mean,wsum}`** (default `wsum`):
`min_scnr_db` (min-target SCNR) / `mean_scnr_db` (mean-target SCNR) /
`weighted_sum_scnr_db` (weighted-sum SCNR, legend short "w-sum SCNR").

**Dispersion band — `--band {std,iqr,none}`** (default `std`): std (mean+/-std),
iqr (25-75 pct), or none. For SINR/SCNR it is `plot_sweep`'s `error` argument;
for the outage curve it is computed from the per-trial outage fraction.

## Files

| File | Purpose |
|---|---|
| `build_fig_csi.py` | Headless builder **and** importable module (loader + helpers + `build_figure`). Writes `../../figures/fig_csi.pdf`. |
| `fig_csi.ipynb` | Interactive front-end. Loader/summary helpers imported from the module; **`build_figure` inlined as an editable cell** with `COMM_METRIC` / `SCNR_METRIC` / `BAND` / `PILOT_STAR` / `GAMMA_DB` knobs. |
| `README.md` | This file. |

No runner — point the builder at a `csi_sweep` run (or let it auto-discover the
newest, preferring `array_<jobid>_aggregated/`, never `latest`).

## How to build

```bash
# Newest run; min-user SINR (top), weighted-sum SCNR (bottom), std band:
python3 paper/figure_src/fig_csi/build_fig_csi.py

# Outage-probability view of robustness (the cliff), min-target SCNR:
python3 paper/figure_src/fig_csi/build_fig_csi.py --comm-metric outage --scnr-metric min

# Mean-user SINR, band removed; no pdflatex:
python3 paper/figure_src/fig_csi/build_fig_csi.py --comm-metric mean --band none --no-tex
```

The builder prints the comm metric at worst-CSI / operating / best-CSI pilot
power, the best-worst change (small = graceful, large = cliff), and the SCNR
change.

## Campaign

`csi_sweep` (MMSE, `all_algorithms`) at the favorable config; pilot grid that
brackets the knee (~110 dB at ap_radius=650 m) and P_p*=128 dB, e.g.
`pilot in {80, 90, 100, 108, 114, 120, 128}`. `N_TRIALS_TOTAL=400`,
`N_ARRAY_TASKS=10`, aggregate with
`scripts/aggregate_array_batch.py csi_sweep <jobid>`. If `topology.ap_radius_m`
changes, shift the grid to keep the knee in view.

