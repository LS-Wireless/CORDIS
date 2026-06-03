# Fig. — Robustness to imperfect CSI (pilot-power sweep)

Stacked two-panel, single-column figure for the Simulation Results section,
showing how each algorithm holds up as the channel-state information degrades,
from a `csi_sweep` campaign (a sweep over the uplink pilot power P_p/σ_n²,
which sets the MMSE channel-estimation quality).

- **(top) min-SINR vs pilot power** and **(bottom) sum-SCNR vs pilot power.**
- Lower pilot power → noisier estimates ĥ_{a,u} → every algorithm degrades.
  The figure's point is the **shape** of that degradation: the centralized
  bound and the CORDIS variants bend down gracefully, while interference-naïve
  baselines (e.g. MRT) cliff-edge sooner.
- A dashed line marks the deployed operating point **P_p\*** (= `PILOT_POWER_DB`
  in `_defaults.sh`, 128 dB ≈ near-perfect CSI), so the reader sees the rest of
  the paper runs in the good-CSI plateau and this figure explores what happens
  as CSI degrades.

Because the graceful-vs-cliff contrast *is* the figure, this one plots **all**
algorithms in the run by default — the proposed trio
(`Centralized`, `CORDIS-ADMM`, `CORDIS-Split`) leads (taking the consistent
`style_for` / `ALGORITHM_STYLE` styles and legend order) and any benchmarks are
appended. Pass `--only` to filter to a subset.

## Where the knee lives (and why ap_radius matters)

Received pilot SNR = (P_p/σ_n²)·τ_p·β, and the predicted NMSE ≈ 1/(1+SNR_rx).
For the default `topology.ap_radius_m = 650 m`, β ≈ 7e-11, so NMSE crosses ~0.1
around **110 dB** — that's where the curves turn. If the campaign changes
`ap_radius_m`, the informative sweep range shifts (closer APs → larger β → the
knee moves to lower pilot power). Keep the sweep grid centred on the knee.

## Files

| File | Purpose |
|---|---|
| `build_fig_csi.py` | Headless builder **and** importable module (loader + helpers + `build_figure`). Writes `../../figures/fig_csi.pdf`. |
| `fig_csi.ipynb` | Interactive front-end. Loader/summary helpers imported from the module; **`build_figure` inlined as an editable cell** for style/layout overrides. |
| `README.md` | This file. |

No runner — point the builder at a `csi_sweep` campaign (or let it auto-discover
the newest, preferring `array_<jobid>_aggregated/`, never `latest`).

## How to build

```bash
# Newest aggregated run, P_p* marked at 128 dB, all algorithms:
python3 paper/figure_src/fig_csi/build_fig_csi.py

# On a node without pdflatex:
python3 paper/figure_src/fig_csi/build_fig_csi.py --no-tex

# Just the proposed trio (drop benchmarks), pinned run:
python3 paper/figure_src/fig_csi/build_fig_csi.py \
    --only "Centralized,CORDIS-ADMM,CORDIS-Split" \
    --result-dir results/exp_csi_sweep/array_12345_aggregated
```

The builder prints a per-algorithm summary — min-SINR at the worst / operating /
best pilot, the SINR dynamic range (best−worst), and the SCNR range — ready for
the caption.

## Recommended sweep grid and campaign config

In short (see the chat notes): sweep pilot power with points anchoring the
estimation-error floor (~80–90 dB), sampling the NMSE knee densely
(~100–120 dB at 650 m), and reaching the deployed operating point 128 dB at the
near-perfect plateau; keep `estimation_method=MMSE` (not `perfect`), run
`spec_set=all_algorithms` so the benchmarks are present for the contrast, hold
the rest of the favorable config fixed, and keep the drop count high.

## Research-integrity note

As with the CDF/feasibility figures, report the **stress scenario**
(`configs/scenarios/stress.json`) alongside the favorable-config version: re-run
the CSI sweep under the stress override and point the builder at that run with
`--result-dir`.

