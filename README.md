<div align="center">

# CORDIS

### A Scalable Coordinated Resource Allocation Framework for Distributed Cell-Free ISAC

*Decentralized beamforming and power allocation for cell-free integrated sensing and communication,
coordinated through consensus ADMM, with fronthaul and per-AP computation independent of the array size and the number of APs.*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Solver: CVXPY](https://img.shields.io/badge/solver-CVXPY%20%2F%20CLARABEL-orange.svg)](https://www.cvxpy.org/)
[![Paper](https://img.shields.io/badge/paper-IEEE%20TWC%20(submission%20process)-b31b1b.svg)](#citation)

</div>

<p align="center">
  <img src="paper/figures/fig_lowrank_outage.png" alt="CORDIS-ADMM tracks the centralized bound in the locally low-rank regime" width="620">
  <br>
  <sub><b>CORDIS-ADMM</b> coordinates beamformers across APs and tracks the centralized bound on both
  communication outage and target SCNR, even in the locally low-rank regime (<code>N_UE &gt; M</code>) where
  the lightweight CORDIS-Split saturates.</sub>
</p>

---

## Overview

**CORDIS** (**CO**ordinated **R**esource allocation for **D**istributed **I**SAC **S**ystems) is a
research framework for joint **beamforming (BF)** and **power allocation (PA)** in **distributed
cell-free ISAC** networks. It makes the communication–sensing trade-off explicit and solves it under
realistic fronthaul and edge-computation limits, over a 3GPP urban-microcell (UMi) spatially correlated
Rician channel with environmental clutter and imperfect CSI.

The framework ships **two complementary algorithms** plus a centralized ceiling and a family of classical
baselines, a fully reproducible Monte-Carlo simulation stack, a publication figure pipeline, and
interactive notebooks.

| Algorithm | What it does | Coordination cost |
|---|---|---|
| **CORDIS-Split** | Fixed local robust beamformers + a centralized convex power allocation | A few real scalars per user (one round) |
| **CORDIS-ADMM** | Joint BF + PA via consensus ADMM (ALM/MM), per-AP QCQP + a low-dimensional central SOC projection | Low-dimensional consensus exchange, **independent of array size `M` and AP count `N_ap`** |
| **Centralized** | One-shot joint optimum (performance ceiling / benchmark) | Aggregates full CSI and precoders at the CPU |

> CORDIS-ADMM approaches the centralized sensing bound to within ~1 dB, degrades gracefully under
> imperfect CSI, and preserves both per-user QoS and target SCNR where the lightweight CORDIS-Split saturates.

---

## Highlights

- **Two scalable solvers** — a low-overhead split scheme and a fully decentralized consensus-ADMM scheme.
- **Realistic modeling** — 3GPP UMi correlated Rician fading, MMSE channel estimation with explicit
  error covariance, STAP-based target SCNR, and environmental clutter.
- **Honest scalability** — per-round fronthaul and per-AP computation that do not grow with the antenna
  count or the number of APs.
- **Reproducible by design** — every figure in the paper is regenerated from committed scripts, seeds,
  and configs; results carry provenance metadata (Git SHA, date).
- **Batteries included** — a dozen experiment types, five algorithm spec-sets, a CLI config system,
  cluster (SLURM) launchers, and four interactive plot playgrounds.

---

## Repository layout

```text
CORDIS/
├── cordis/                  # Core Python package
│   ├── algorithms/          #   CORDIS-Split, CORDIS-ADMM, centralized, benchmark registry
│   ├── channel/             #   topology, correlated-Rician channel, sensing AP–target association
│   ├── experiments/         #   experiment specs, sweep axes, registry, ExperimentResult I/O
│   ├── metrics/             #   SINR / SCNR / outage metrics
│   ├── plotting/            #   IEEE styling, plot_cdf / plot_sweep, save_figure, style tables
│   ├── simulation/          #   parallel Monte-Carlo runner
│   └── utils/               #   config (CORDISConfig + PARAM_REGISTRY), paths, I/O
├── configs/                 # default.json + experiment overrides + shell recipes
├── scripts/                 # exp_*/plot_* launchers, config tools, SLURM trees, validators
├── notebooks/               # interactive plot playgrounds (cdf / sweep / trace / table)
├── paper/
│   ├── figure_src/          #   publication figure builders (+ a notebook per figure)
│   └── figures/             #   rendered PDFs / PNGs
├── docs/                    # quickstart + auto-generated config & benchmark references
├── results/                 # experiment outputs (git-ignored; synced from the cluster)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

Requires **Python 3.10+**. The core solver path uses CVXPY with the bundled CLARABEL solver.

```bash
git clone https://github.com/LS-Wireless/CORDIS
cd CORDIS

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                  # makes `cordis` importable from any directory
```

Optional extras:

```bash
pip install -e ".[notebooks]"     # Jupyter playgrounds
pip install -e ".[assent]"        # learning-based sensing association (see below)
```

Smoke-test the install:

```bash
python -c "import cordis, numpy, cvxpy; print('CORDIS ready —', 'cvxpy', cvxpy.__version__)"
```

---

## Quickstart

Run an experiment, then plot it. Results are written to `results/exp_<name>/<timestamp>/`
and figures to `figures/`.

```bash
# 1) Run a gamma (QoS-target) sweep across all algorithms
bash scripts/exp_gamma_sweep.sh           # or: python scripts/exp_gamma_sweep.py

# 2) Plot the most recent run
python scripts/plot_gamma_sweep.py        # or: make plot-gamma_sweep
```

Restrict the algorithm set, change the trial count, or override any config field on the fly:

```bash
SPECS=cordis_vs_centralized N_TRIALS=400 bash scripts/exp_gamma_sweep.sh
python scripts/create_config.py --set algorithm.admm.kappa=0.08 --set topology.n_ap=10
```

See [`docs/quickstart.md`](docs/quickstart.md) for the full walkthrough, including cluster runs and the
result-sync workflow.

---

## Experiments

Each experiment has a launcher (`scripts/exp_<name>.py` / `.sh`) and a plotter
(`scripts/plot_<name>.py`). The runnable types:

| Kind | Experiments |
|---|---|
| Distributions | `sinr_cdf`, `scnr_cdf` |
| Sweeps | `snr_sweep`, `gamma_sweep`, `kappa_sweep`, `clutter_cnr_sweep`, `csi_sweep`, `n_ue_sweep`, `n_ap_sweep`, `antennas_sweep` |
| Trace | `convergence_trace` |
| Table | `fronthaul_table` |

Algorithms are chosen per run via a **spec-set**:

| Spec-set | Contents |
|---|---|
| `cordis_only` | CORDIS-Split + CORDIS-ADMM |
| `cordis_vs_centralized` | + the centralized ceiling |
| `cordis_vs_benchmarks` | + classical BF/PA baselines |
| `all_algorithms` | everything (default for most experiments) |
| `psr_baselines` | fixed power-split-ratio baselines |

---

## Configuration

All physical and algorithmic parameters live in `configs/default.json`, validated by the dataclasses in
`cordis/utils/config.py`. The single source of truth for documentation is `PARAM_REGISTRY`, introspected
by the helper tools:

```bash
python scripts/list_config_params.py                 # full reference (terminal)
python scripts/list_config_params.py --search pilot  # filter by keyword
python scripts/list_config_params.py --section admm sensing
./scripts/list_config_params.sh                      # regenerate docs/config_reference.{md,csv}
```

Representative defaults: `N_ap = 10` APs, `M = 16`-element UCAs, `N_UE = 4` users, `N_t = 2` targets,
QoS floor `γ = 5` dB, clutter penalty `κ = 0.08`, carrier `3.5` GHz, bandwidth `20` MHz. The complete,
always-in-sync table is in the documentation below.

> **Reference:** [`docs/config_reference.md`](docs/config_reference.md) ·
> [`docs/config_reference.csv`](docs/config_reference.csv)

---

## Baselines

The classical BF/PA baselines (Global MRT, Global ZF, LR-MMSE, fixed power-split ratios, ...) are declared
in `cordis/algorithms/benchmarks.py:BENCHMARK_REGISTRY` and documented by:

```bash
python scripts/list_benchmarks.py                    # terminal listing
./scripts/list_benchmarks.sh                         # regenerate docs/benchmarks_reference.{md,csv}
```

> **Reference:** [`docs/benchmarks_reference.md`](docs/benchmarks_reference.md) ·
> [`docs/benchmarks_reference.csv`](docs/benchmarks_reference.csv)

---

## Reproducing the paper figures

Every figure in the paper is built by a self-contained script under `paper/figure_src/<figure>/`, which is
**both** a headless CLI and an importable module reused by a sibling notebook. Outputs go to `paper/figures/`.

```bash
# Headless rebuild (add --no-tex on a node without pdflatex)
python3 paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py
python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py --comm-metric outage
```

Most sweep/distribution builders expose selectable metrics, e.g. `--comm-metric {min,mean,outage}`,
`--scnr-metric {min,mean,wsum}`, and `--band {std,iqr,none}`.

---

## Interactive notebooks

Four playgrounds under `notebooks/` cover every experiment of a given result *kind*. Switch experiments by
editing a single line and re-running:

| Notebook | Covers |
|---|---|
| [`notebooks/playground_cdf.ipynb`](notebooks/playground_cdf.ipynb) | `sinr_cdf`, `scnr_cdf` |
| [`notebooks/playground_sweep.ipynb`](notebooks/playground_sweep.ipynb) | all `*_sweep` experiments |
| [`notebooks/playground_trace.ipynb`](notebooks/playground_trace.ipynb) | `convergence_trace` |
| [`notebooks/playground_table.ipynb`](notebooks/playground_table.ipynb) | `fronthaul_table` |

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/playground_sweep.ipynb
```

The four notebooks are generated from `notebooks/_build_playgrounds.py`; clear outputs before committing.

---

## Running on a cluster (SLURM)

The repo ships generic SLURM templates (`scripts/slurm/exp_*.sbatch`) and a ready-to-submit, site-specific
tree for **UCI HPC3** (`scripts/slurm/uci-hpc3/exp_*.sub`):

```bash
sbatch scripts/slurm/uci-hpc3/exp_gamma_sweep.sub
```

Submission settings (account, partition, repo path, Python module, venv name) live in
`scripts/slurm/uci-hpc3/_config.sh` and are overridable per submit. Full cluster + result-sync instructions
are in [`docs/quickstart.md`](docs/quickstart.md).

---

## Learning-based association (optional)

CORDIS can consume AP–target sensing associations from a pre-computed decision file, or call the
[ASSENT](https://github.com/LS-Wireless/ASSENT-CellFree-ISAC) GNN model at runtime via
`cordis.channel.sensing_assignment`:

```bash
pip install -e ".[assent]"
```

When ASSENT is unavailable, use the `assent_file` strategy with a pre-computed decision, or one of the
built-in heuristic association strategies.

---

## Documentation

| Document | Description |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Install, run, plot, sync results, and submit cluster jobs |
| [`docs/config_reference.md`](docs/config_reference.md) · [`.csv`](docs/config_reference.csv) | Every config parameter: type, default, unit, range, description |
| [`docs/benchmarks_reference.md`](docs/benchmarks_reference.md) · [`.csv`](docs/benchmarks_reference.csv) | Every registered BF/PA baseline |

The Markdown and CSV references are **auto-generated** — regenerate them with
`./scripts/list_config_params.sh` and `./scripts/list_benchmarks.sh` whenever the registries change.

---

## Citation

If you use CORDIS in your research, please cite the software and the paper:

```bibtex
@misc{Zafari2026CORDIScode,
  author       = {Mehdi Zafari},
  title        = {{CORDIS}: Coordinated Resource Allocation for Distributed {ISAC} Systems},
  howpublished = {GitHub repository},
  year         = {2026},
  url          = {https://github.com/LS-Wireless/CORDIS}
}

@article{Zafari2026CORDIS,
  author  = {Mehdi Zafari and Bjorn Ottersten and A. Lee Swindlehurst},
  title   = {{CORDIS}: A Scalable Coordinated Resource Allocation Framework
             for Distributed Cell-Free {ISAC}},
  journal = {IEEE Transactions on Wireless Communications (submission process)},
  year    = {2026}
}
```

The framework builds on our preliminary ADMM-based design presented at the Asilomar Conference on Signals,
Systems, and Computers (2024).

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Contact

Mehdi Zafari · LS-Wireless · [github.com/LS-Wireless/CORDIS](https://github.com/LS-Wireless/CORDIS)

