# Paper figures

Build code and final PDFs for the journal paper's Simulation Results section.

```
paper/                         (can be renamed — nothing hard-codes the name)
├── figures/                   final PDFs that \includegraphics reads  (committed)
│   └── fig_convergence.pdf
└── figure_src/                build code, one folder per figure       (committed)
    ├── README.md              (this file)
    └── fig_convergence/
        ├── README.md
        ├── run_traces.sh            generate the experiment data
        ├── build_fig_convergence.py headless builder + shared module
        ├── fig_convergence.ipynb    interactive front-end
        └── data/                    the exact traces behind the PDF
```

Each `fig_*/` builds itself from the simulation framework: a small script/notebook
loads an `ExperimentResult` (or its stashed `data/`) and emits a single PDF into
`figures/`. The builder is also an importable module so the notebook and the headless
build never drift.

## Conventions

- **One subdirectory per figure**, named `fig_<topic>` (no numbers — figure ordering
  in the paper can change; the topic name is stable).
- **Final PDFs live only in `figures/`** so zipping `paper/` for IEEE/arXiv is one line.
- **Repo-root auto-detection**: builders walk up to the directory containing `cordis/`,
  so they run from anywhere and survive renaming `paper/`.
- **Headless-safe**: every builder takes `--no-tex` for compute nodes without `pdflatex`
  (falls back to mathtext via `apply_paper_style(use_latex=False)`).

## Style contract

All figures share `cordis/plotting/style.py`: call `apply_paper_style()` for the IEEE
two-column rcParams and `figsize(width=..., aspect=...)` for sizing (`"single"`=3.5″,
`"double"`=7.16″). For multi-algorithm figures, pull per-algorithm color/marker from
`style_for(name)` / `ALGORITHM_STYLE` so the same algorithm reads identically across
every figure. (The convergence figure is a single-algorithm diagnostic, so it uses
figure-local colors for fixed-ρ vs adaptive-ρ instead.)

## What to commit vs ignore

- **Commit**: builders, notebooks (outputs cleared), the small `data/*/trace.npz`
  anchors, and the final `figures/*.pdf`.
- **Ignore** (see `paper/.gitignore`): notebook checkpoints, `__pycache__`, and PNG
  previews / scratch.

## Figure index (Simulation Results section)

| Folder | Figure | Experiment(s) |
|---|---|---|
| `fig_convergence` | CORDIS-ADMM convergence (residuals; slack→0 + min-SINR vs γ) | `convergence_trace` |
| `fig_cs_tradeoff` | Communication–sensing tradeoff / feasibility vs γ | `gamma_sweep` |
| `fig_cdf` | min-SINR CDF + SCNR CDF at γ\* | `sinr_cdf`, `scnr_cdf` |
| `fig_clutter` | Clutter suppression (κ sweep) | `kappa_sweep` |
| `fig_csi` | Robustness to imperfect CSI | `csi_sweep` |
| `fig_lowrank` | Locally low-rank scalability (ADMM headline) | `sinr_cdf`/`scnr_cdf` at n_ap=11, n_ant=3 |
| (table) `tab_fronthaul` | Fronthaul overhead | `fronthaul_table` |
| `fig_snr` (optional) | SNR/power sweep (appendix) | `snr_sweep` |


