# Fig. — Scalability in the locally low-rank regime (ADMM headline)

Stacked two-panel, single-column figure for the Simulation Results section,
from an `n_ue_sweep` campaign run in the few-antennas-per-AP regime — the
figure that shows CORDIS-ADMM's headline advantage.

- **(top) communication metric vs N_ue** — selectable:
  - `min` : worst-user SINR [dB] (default)
  - `mean`: across-user SINR [dB]
  - `outage`: per-user outage probability Pr(SINR_u < gamma*) — CORDIS-Split's
    outage spikes toward 1 in the low-rank region. gamma* auto-detected.
- **(bottom) SCNR vs N_ue** — selectable: `wsum` (weighted target-SCNR sum,
  default), `min` (worst target), `mean`.

The campaign fixes a small per-AP array (M = n_ant antennas/AP, total Tx-antenna
budget held constant) and raises the user load N_ue. The per-AP spatial DoF are
M, so once **N_ue > M** the local channels are locally low-rank: the
fixed-local-beamformer Algorithm 1 (CORDIS-Split) can no longer null all
co-users from one AP and diverges (SINR drops / outage spikes), while the
consensus-ADMM Algorithm 2 (CORDIS-ADMM) keeps meeting the floor by coordinating
across APs — tracking the centralized ceiling. That divergence in the shaded
**N_ue > M** region is the money shot.

Each curve's central line is the across-trial mean (the outage line is the
per-user pool fraction); an optional band shows dispersion.

## Selectable metrics and band (defaults match the paper)

**Communication metric — `--comm-metric {min,mean,outage}`** (default `min`):

| key | what | axis |
|---|---|---|
| `min` | `min_sinr_db` (min-user SINR) | dB |
| `mean` | `mean_sinr_db` (mean-user SINR) | dB |
| `outage` | per-user Pr(SINR_u < gamma*) | probability [0,1] |

For `outage`, set gamma* with `--gamma-db` (default: auto-detected, else 5 dB).
The outage band is the std / IQR of the per-trial outage fraction.

**SCNR metric — `--scnr-metric {min,mean,wsum}`** (default `wsum`):
`min_scnr_db` (min-target SCNR) / `mean_scnr_db` (mean-target SCNR) /
`weighted_sum_scnr_db` (weighted-sum SCNR, legend short "w-sum SCNR").

**Dispersion band — `--band {std,iqr,none}`** (default `std`).

## Regime boundary M and shading

`--rank-threshold M` sets the per-AP DoF; `N_ue > M` is shaded on both panels.
Default: auto-detected from metadata (`n_ant`), else 3. `--no-shade` turns the
shading off. N_ue ticks are forced to integers.

## Files

| File | Purpose |
|---|---|
| `build_fig_lowrank.py` | Headless builder **and** importable module (loader + helpers + `build_figure`). Writes `../../figures/fig_lowrank.pdf`. |
| `fig_lowrank.ipynb` | Interactive front-end. Loader/summary helpers imported from the module; **`build_figure` inlined as an editable cell** with `COMM_METRIC` / `SCNR_METRIC` / `BAND` / `RANK_THRESHOLD` / `GAMMA_DB` knobs. |
| `README.md` | This file. |

No runner — point the builder at the low-rank `n_ue_sweep` run (or let it
auto-discover the newest, preferring `array_<jobid>_aggregated/`, never `latest`).

## How to build

```bash
# Newest run; min-user SINR (top), weighted-sum SCNR (bottom), std band:
python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py

# Outage-probability view (Split's collapse is starkest here):
python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py --comm-metric outage --scnr-metric min

# Mean-user SINR, band off, pin M=3:
python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py --comm-metric mean --band none --rank-threshold 3
```

The builder prints the comm metric + SCNR at the full-rank reference (N_ue=M)
and max load, and the headline CORDIS-ADMM advantage over CORDIS-Split at max
N_ue (the divergence).

## Campaign (LOW-RANK config — differs from the favorable default)

`n_ue_sweep` with the low-rank overrides: `n_ap=11`, `n_sensing_rx=1`,
`n_ant=3`, `n_rf_chains=3`, `n_targets=1`, so the total Tx-antenna budget is
30 (matched to the other scenarios), per-AP DoF M=3, and a single target. N_ue
grid `{3, 4, 5, 6}` straddles M, `spec_set=cordis_vs_centralized`, `gamma=5` dB.
`N_TRIALS_TOTAL=400`, `N_ARRAY_TASKS=10`, aggregate with
`scripts/aggregate_array_batch.py n_ue_sweep <jobid>`. Thread the low-rank
`--values` overrides through the array `.sub`, not just the recipe.

