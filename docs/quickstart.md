# CORDIS Quickstart

Reproducing the paper figures from a fresh clone.

---

## Prerequisites

- Python ≥ 3.10
- GNU Make (already installed on macOS and most Linux distros)
- The Python dependencies listed in your environment file (NumPy, SciPy,
  CVXPY, joblib, matplotlib, …)
- For cluster runs: SLURM with a partition you can submit to

You don't need `pip install -e .` — the Makefile and `.sh` launchers
prepend the repo root to `PYTHONPATH` automatically.

---

## 30-second tour

```bash
git clone <repo-url> CORDIS
cd CORDIS

# (Activate your venv if you keep one — .venv/ or venv/ at repo root is
#  auto-sourced by the launchers.)

make help            # see all targets
make validate        # ~30s; runs every Stage 8 sub-validator
make sinr_cdf        # run one experiment with paper-sized defaults
make plot-sinr_cdf   # render the figure from the latest run
```

The figure lands in `figures/exp_sinr_cdf/` as both `.pdf` and `.pgf`.

---

## Running experiments

### One experiment

```bash
make sinr_cdf        # equivalent to: bash scripts/exp_sinr_cdf.sh
```

The launcher reads paper-sized defaults from `configs/recipes/exp_sinr_cdf.sh`
(via `_defaults.sh`) and runs `scripts/exp_sinr_cdf.py`.  Results land
under `results/exp_sinr_cdf/<timestamp>/`, with a `latest` symlink
updated to point at the newest run.

### Override knobs from the command line

Every launcher honours environment-variable overrides:

```bash
make sinr_cdf N_DROPS=10 N_REAL=2 N_WORKERS=4
```

| Variable      | Default                                 | Meaning                  |
|---------------|-----------------------------------------|--------------------------|
| `BASE_CONFIG` | `configs/default.json`                  | base config              |
| `EXP_CONFIG`  | `configs/exp_<name>.json` (if present)  | experiment overrides     |
| `N_DROPS`     | per-experiment (see recipe)             | scenario realizations    |
| `N_REAL`      | per-experiment                          | channel realizations     |
| `N_WORKERS`   | `-1` (all cores)                        | joblib worker count      |
| `SEED`        | `42`                                    | master RNG seed          |
| `OUTPUT_ROOT` | `results`                               | output tree root         |

### All experiments

```bash
make all             # sequential: ~hours at paper-sized defaults
```

For a smoke run, drop the trial counts:

```bash
make all N_DROPS=4 N_REAL=2 N_WORKERS=4
```

---

## Plotting

```bash
make plot-sinr_cdf                                       # latest run
make plot-sinr_cdf EXP_DIR=results/exp_sinr_cdf/<ts>     # specific run
make figures                                             # plot all 11
```

Figures are written to `figures/exp_<name>/` in `.pdf` and `.pgf`
formats by default.  Override formats:

```bash
python3 scripts/plot_sinr_cdf.py --formats pdf,png
```

---

## Cluster submission (SLURM)

```bash
make submit-sinr_cdf      # sbatch scripts/slurm/exp_sinr_cdf.sbatch
make submit-all           # submit every experiment as 11 separate jobs
```

Before the first submission, edit the SLURM scripts to set partition,
account, and any modules your cluster needs.  The defaults are
intentionally generic:

```bash
# scripts/slurm/exp_sinr_cdf.sbatch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
# Uncomment and customise for your cluster:
# #SBATCH --partition=standard
# #SBATCH --account=your-account
```

`N_WORKERS` inside the launcher automatically tracks
`$SLURM_CPUS_PER_TASK`, so worker counts scale with the allocation.

---

## Validation

Run every Stage 8 sub-validator:

```bash
make validate
# equivalent to:
python3 scripts/validate_stage8.py
```

Run one stage:

```bash
python3 scripts/validate_stage8.py --only 8c
```

Halt at the first failure:

```bash
python3 scripts/validate_stage8.py --stop-on-fail
```

Each sub-validator's exit code feeds the umbrella's aggregate exit
code (0 iff every sub-validator passes).

---

## Common tasks

### Add a new experiment

1. Implement `run_<name>(...)` in `cordis/experiments/registry.py`.
2. Add a metadata entry to the `EXPERIMENTS` dict in
   `scripts/regenerate_experiment_scripts.py`.
3. Add the name to the `EXPERIMENTS :=` line in the `Makefile`.
4. Add the name to `EXPERIMENTS = [...]` in `scripts/validate_stage8c.py`
   and `scripts/validate_stage8d.py`.
5. Run `make regenerate-scripts` (or `python3 scripts/regenerate_experiment_scripts.py`).
6. Run `make validate` — Tests 10 + similar will tell you if anything is out of sync.

### Trial-count knobs (Stage 9)

There are two equivalent ways to control how many trials each experiment runs:

```bash
# Option A — set the total, let the framework decompose into drops × real:
make sinr_cdf N_TRIALS=400         # → 20 drops × 20 real (closest factor pair)
make sinr_cdf N_TRIALS=100         # → 10 × 10

# Option B — set drops and realizations explicitly:
make sinr_cdf N_DROPS=10 N_REAL=40

# Partial mix is OK too — explicit dim wins, the other is computed:
make sinr_cdf N_TRIALS=200 N_DROPS=8        # → 8 drops × 25 real
```

The precedence chain (highest to lowest):

1. Explicit `--n-drops` AND `--n-realizations` from CLI / env
2. `--n-trials` CLI flag (decompose; if one dim is set, computes the other)
3. `cfg.simulation.n_trials` from the experiment's JSON config (decompose)
4. Per-experiment fallback baked into `scripts/exp_<name>.py`

The chosen path is logged on every run (`Trial counts: n_drops=…, n_realizations=…  (<source>)`).

### Reset between runs

```bash
make clean-figures        # rm -rf figures/
make clean-logs           # rm -rf logs/
make clean-results        # rm -rf results/  (5-second warning, then deletes)
```

### After pulling new code

```bash
make regenerate-scripts   # refresh exp_*.sh/.py, plot_*.py, slurm/*.sbatch
make validate             # confirm everything still hangs together
```

---

## Stage 9 migration notes

Stage 9 consolidated three vestigial config patterns.  If you're updating
from an older checkout, edit `configs/default.json` (and any `configs/exp_*.json`)
accordingly:

| Old key | New key | What changed |
|---|---|---|
| `algorithm.split.gamma_db` | `algorithm.gamma_db` | Hoisted to umbrella level; shared by every algorithm.  Old key is silently ignored by the loader. |
| `simulation.n_trials` | *same key, now actively used* | Previously declared-but-unread; now decomposed into `(n_drops, n_realizations)` when CLI doesn't provide them. |
| `sensing.sigma_clt: 0.1` | `sensing.sigma_clt: null` | Field is now `Optional[float]`.  `null` (default) means "derive σ_clt² from CNR".  Any explicit value is treated as an **override** — `σ_clt² = sigma_clt²`, ignoring `clutter_cnr_db`. |

**The σ_clt change is the only one that can silently alter behavior**: an
existing `default.json` with `"sigma_clt": 0.1` now pins σ_clt² to 0.01
instead of deriving from `clutter_cnr_db`.  Set it to `null` (or remove
the key) to keep the previous CNR-based semantics.

To verify the migration: `python3 scripts/validate_stage9.py` should
report 21/21 passed.

---

## Reference

- `Makefile` — every target documented in `make help`
- `scripts/regenerate_experiment_scripts.py` — single source of truth for
  per-experiment files
- `scripts/validate_stage8.py` — umbrella validator
- `docs/config_reference.md` — every config field
- `docs/scripts_reference.md` — every script and its role

