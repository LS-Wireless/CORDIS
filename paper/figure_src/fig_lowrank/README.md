# Fig. — Scalability in the locally low-rank regime (ADMM headline)

Stacked two-panel, single-column figure for the Simulation Results section —
CORDIS-ADMM's headline advantage — from an `n_ue_sweep` campaign run in the
few-antennas-per-AP regime.

- **(top) min-SINR vs N_ue** and **(bottom) sum-SCNR vs N_ue.**
- The campaign fixes a small per-AP array (**M = n_ant** antennas per AP) and
  increases the user load N_ue, holding the total Tx-antenna budget constant.
  Per-AP spatial DoF = M, so once **N_ue > M** the local channels are *locally
  low-rank*: the fixed-local-beamformer Algorithm 1 (CORDIS-Split) can't null
  all co-users from a single AP and its QoS diverges, while the consensus-ADMM
  Algorithm 2 (CORDIS-ADMM) keeps meeting the SINR floor via cross-AP
  coordination and tracks the centralized ceiling. **That divergence in the
  shaded N_ue > M region is the figure's money shot.**
- N_ue = M is the full-rank reference; the N_ue > M region is shaded.

This is the journal version of the conference Fig. 2(right): correlated-Rician
+ clutter model, with the proposed pair vs the centralized ceiling
(`Centralized`, `CORDIS-ADMM`, `CORDIS-Split`), styled via `style_for` /
`ALGORITHM_STYLE`.

## Files

| File | Purpose |
|---|---|
| `build_fig_lowrank.py` | Headless builder **and** importable module (loader + helpers + `build_figure`). Writes `../../figures/fig_lowrank.pdf`. |
| `fig_lowrank.ipynb` | Interactive front-end. Loader/summary helpers imported from the module; **`build_figure` inlined as an editable cell** for style/layout overrides. |
| `README.md` | This file. |

No runner — point the builder at a low-rank `n_ue_sweep` campaign (or let it
auto-discover the newest, preferring `array_<jobid>_aggregated/`, never
`latest`).

## How to build

```bash
# Newest aggregated run; M auto-detected from metadata (shades N_ue > M):
python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py

# On a node without pdflatex:
python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py --no-tex

# Pin M / run; or drop the shading:
python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py \
    --rank-threshold 3 \
    --result-dir results/exp_n_ue_sweep/array_12345_aggregated
```

The builder prints a per-algorithm summary (min-SINR & SCNR at the full-rank
reference and at max load, the min-SINR drop), plus the headline
**CORDIS-ADMM − CORDIS-Split** min-SINR gap at the most-loaded point — the
divergence number for the caption.

## ⚠️ This figure needs the LOW-RANK campaign, not the favorable config

The other figures run at the favorable config (n_ap=10, n_ant=16). This one
**deliberately overrides** to the locally-low-rank regime. Point the builder at
that run; `--rank-threshold` auto-detects from metadata (= n_ant) but pass it
explicitly if the metadata doesn't carry it. See the chat notes for the full
campaign config and N_ue grid; in brief:

- `n_ap=11`, `n_sensing_rx=1` (→ 10 Tx APs), `n_ant=3`, `n_rf_chains=3`,
  `n_targets=1` (total Tx antennas = 30, matched across points).
- `n_ue ∈ {3,4,5,6}` (3 = full-rank reference, since M=3; 4,5,6 = low-rank).
- `gamma_db = γ*` (= 5; drop to 3 dB if 5 proves infeasible at low rank).
- `spec_set = cordis_vs_centralized` (the trio).
- keep the drop count high.

## Research-integrity note

As with the other figures, you can report the **stress scenario**
(`configs/scenarios/stress.json`) alongside the favorable low-rank version:
re-run the low-rank N_ue sweep under the stress override and point the builder
at that run with `--result-dir`.

