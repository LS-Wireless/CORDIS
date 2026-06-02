# Fig. — CORDIS-ADMM convergence

Two-panel convergence figure for the Simulation Results section.

- **(a)** primal `r_pri` and dual `r_dual` consensus residuals vs ADMM iteration
  (log-y), overlaying the **fixed-ρ** run (the shipped default) and the
  **adaptive-ρ** run (the ablation). Fixed-ρ settles monotonically; adaptive-ρ
  perturbs the in-loop SCA and swings — this is the evidence for `adaptive_rho=false`
  being the default.
- **(b)** for the default run: the SOC slack `max_u ε_u` collapsing toward zero (the
  augmented-Lagrangian driving feasibility) and the worst-user **min-SINR** climbing
  across the γ requirement line.

The figure deliberately tells the convergence story with **slack + min-SINR** rather
than a ρ-trajectory panel, because the `convergence_trace` result persists the
residual / slack / SINR histories but not `rho_history`.

## Files

| File | Purpose |
|---|---|
| `run_traces.sh` | Generates the two traces (fixed-ρ, adaptive-ρ) and stashes them in `data/`. |
| `build_fig_convergence.py` | Headless builder **and** importable module (the notebook reuses its functions). Writes `../../figures/fig_convergence.pdf`. |
| `fig_convergence.ipynb` | Interactive front-end for tweaking — identical plotting logic. |
| `data/trace_fixed/`, `data/trace_adaptive/` | The exact `manifest.json` + `trace.npz` that produced the committed PDF (the reproducibility anchor). |

## How to regenerate

From the repo root (or anywhere — paths auto-resolve):

```bash
# 1. Produce the two traces. Edit RUN_CMD at the top of the script to match
#    how you run a single experiment in this repo (Makefile target / SLURM /
#    python runner). Both runs use the same seed and disable patience early-stop.
bash paper/figure_src/fig_convergence/run_traces.sh

# 2. Build the PDF (+ a PNG preview) into paper/figures/.
python3 paper/figure_src/fig_convergence/build_fig_convergence.py
#   add --no-tex on a compute node without pdflatex
```

Or open `fig_convergence.ipynb` and Run-All.

### Generating the traces by hand

If you'd rather not use `run_traces.sh`, the two runs are just the
`convergence_trace` experiment with these overrides, same seed for both:

```bash
# fixed-ρ (default)
ADMM_ADAPTIVE_RHO=false ADMM_EARLY_STOP_PATIENCE=0 bash configs/recipes/exp_convergence_trace.sh && <your run cmd>
# adaptive-ρ (ablation)
ADMM_ADAPTIVE_RHO=true  ADMM_EARLY_STOP_PATIENCE=0 bash configs/recipes/exp_convergence_trace.sh && <your run cmd>
```

then point the builder at the two result dirs:

```bash
python3 build_fig_convergence.py \
    --fixed-dir    results/exp_convergence_trace/<ts_fixed> \
    --adaptive-dir results/exp_convergence_trace/<ts_adaptive>
```

If you only have a fixed-ρ run, the builder still works — panel (a) just shows the
default curve and panel (b) is unchanged.

## Config

Uses the favorable defaults (n_ap=10, n_ant=16, pilot 128 dB, γ=5, κ=0.08, clutter
offset) with `n_max` from the config. The γ guide line in panel (b) is read from the
trace metadata; `--gamma-db` only supplies a fallback if it's absent.

