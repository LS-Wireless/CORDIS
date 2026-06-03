# Tab. — Fronthaul coordination overhead (+ bar figure)

Two deliverables from a single `fronthaul_table` run (`kind == 'table'`):

1. **The table** — per-AP fronthaul payload for `Centralized` vs `CORDIS-Split`
   vs `CORDIS-ADMM`, written as LaTeX (`figures/tab_fronthaul.tex`, for
   `\input{}`) and Markdown (`figures/tab_fronthaul.md`, preview). Columns: data
   shared, size, **real scalars / round**, iterations (= T_ADMM for ADMM, 1
   otherwise), total / solve, what it scales with, and scalable (✓/✗).
2. **The bar figure** — `figures/fig_fronthaul.pdf`: per-AP, per-coordination-
   round real-scalar payload as **grouped bars at two array sizes (M_lo, M_hi)**.
   The headline reads straight off the bars: Centralized's payload grows with
   the per-AP array size M, while CORDIS-Split and CORDIS-ADMM are **independent
   of M** (their two bars are the same height). T_ADMM is annotated on the ADMM
   bars.

Per-algorithm bar colors come from `style_for` / `ALGORITHM_STYLE`, so they
match the line figures.

## Two modeling facts that shape these deliverables

- **Per-round, apples-to-apples.** `real_scalars` is reported *per coordination
  round* for every algorithm (a Stage-17 fix forbids the ×T_ADMM total here,
  which would otherwise make ADMM look larger than Centralized). The ×T_ADMM
  total lives in the table's "total / solve" column; ADMM's bar is annotated
  with T_ADMM.
- **M_hi is exact from one run.** The payloads are deterministic formulas, not
  Monte-Carlo estimates — only T_ADMM is measured. Centralized is exactly linear
  in M (`2·M·(N_ue+|D|)`), so its M_hi bar is the loaded value × (M_hi / M_lo);
  the M-independent rows are unchanged. So one run suffices for the
  M-independence bars. Pass `--result-dir-hi` to use a *measured* second run
  (e.g. at n_ant=32) instead.

## Files

| File | Purpose |
|---|---|
| `build_tab_fronthaul.py` | Headless builder **and** importable module (loader, `render_markdown`/`render_latex`, `build_bar_figure`). Writes `tab_fronthaul.{tex,md}` and `fig_fronthaul.pdf` into `../../figures/`. |
| `tab_fronthaul.ipynb` | Interactive front-end. Table renderers imported from the module; **`build_bar_figure` inlined as an editable cell** for style overrides. |
| `README.md` | This file. |

No runner — point the builder at a `fronthaul_table` run (or let it
auto-discover the newest).

## How to build

```bash
# Newest run; bars at the run's M and M=32 (extrapolated):
python3 paper/figure_src/tab_fronthaul/build_tab_fronthaul.py

# Figure text without pdflatex (the .tex table is unaffected):
python3 paper/figure_src/tab_fronthaul/build_tab_fronthaul.py --no-tex

# Single-M bars (no extrapolation); or a measured second run:
python3 paper/figure_src/tab_fronthaul/build_tab_fronthaul.py --m-hi 16
python3 paper/figure_src/tab_fronthaul/build_tab_fronthaul.py \
    --result-dir-hi results/exp_fronthaul_table/<run_at_n_ant_32>
```

The builder prints a per-algorithm summary (per-round payload at M_lo and M_hi,
iterations, M-independence, scalable) for the caption.

The LaTeX table is row-per-algorithm; if you prefer the conference Table-I
layout (metrics as rows, algorithms as columns) just transpose it — the numbers
are identical.

## Campaign

`fronthaul_table` is cheap — the payloads are analytical; the run only measures
**T_ADMM** (average ADMM iterations). Run it once at the favorable config so
T_ADMM matches the deployed operating point; a single (non-array) job suffices.
See the chat notes for exact settings.

