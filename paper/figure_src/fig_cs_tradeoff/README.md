# Fig. — Communication–sensing tradeoff / feasible region

Built from a single `gamma_sweep` result (no runner here — you fetch the result from the
cluster into `results/exp_gamma_sweep/` and this just loads + plots it).

Two panels, double-column:

- **(a)** achieved worst-user **min-SINR** (left axis, solid + marker) and **sum-SCNR**
  (right axis, dashed + marker, conditioned on feasibility) vs the SINR target γ. The grey
  `y = x` line marks "target met"; where an algorithm's min-SINR peels below it, it's
  starting to miss the constraint, and the SCNR you see there is the sensing it buys on the
  trials it still solved.
- **(b)** **feasibility rate** vs γ (fraction of trials whose worst user meets γ,
  `= 1 − infeasibility_rate`). This is the operating-region panel: with the Stage-23 fixes
  and the favorable config, CORDIS-ADMM tracks Centralized at low–mid γ and falls off at
  high γ.

Feasibility uses the served/feasible definition (min-SINR ≥ γ per trial). SCNR is reported
conditional on feasibility; min-SINR and feasibility are over all successful trials.
Centralized is the one-shot ceiling; CORDIS-ADMM/Split return their best feasible iterate.

## Files

| File | Purpose |
|---|---|
| `build_fig_cs_tradeoff.py` | Headless builder **and** importable module (`load_sweep`, `collect_curves`, `build_figure`). Writes `../../figures/fig_cs_tradeoff.pdf`. |
| `fig_cs_tradeoff.ipynb` | Interactive front-end. Imports the module for the canonical build **and** carries an editable copy of `build_figure` for live tweaking + a style-override cell. |

## Recommended γ grid

`[0, 2.5, 5, 7.5, 10, 12.5, 15]` dB — a clean 2.5 dB grid that includes the operating point
γ\*=5 (consistent with the CDF figures) and reaches into the infeasible regime so panel (b)
shows the fall-off. If Centralized is still near-feasible at 15 dB, extend to 17.5/20; if
ADMM/Split are already near-zero feasibility by 12.5–15, the conditional-SCNR points there
get noisy or drop out (visible gaps), which is expected.

## How to build

```bash
# After copying a gamma_sweep result into results/exp_gamma_sweep/:
python3 paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py
#   --no-tex                 on a node without pdflatex
#   --result-dir <dir>       pin a specific run (e.g. an array_*_aggregated dir)
#   --scnr-all-trials        SCNR over all trials instead of feasible-only
#   --no-sinr-band           hide the SINR IQR band in panel (a)
#   --feas-percent           feasibility as %  instead of [0,1]
#   --only "Centralized,CORDIS-ADMM,CORDIS-Split"   which algorithms to plot
```

Or open `fig_cs_tradeoff.ipynb` and Run-All.

## Config

`gamma_sweep` is run with the favorable defaults (n_ap=10, n_ant=16, pilot 128 dB, κ=0.08,
clutter offset) and the all_algorithms spec set; the builder filters to
Centralized / CORDIS-ADMM / CORDIS-Split via `--only`. The γ values come straight from the
sweep result's keys, so the figure adapts to whatever grid you ran.

