# Running large Monte Carlo campaigns via SLURM job arrays

(Stage 21 — UCI HPC3)

For final-paper figures where you need 500-2000+ trials per
experiment, single-job submission becomes the wall-time bottleneck.
SLURM job arrays let you split the trials across N parallel tasks,
each running on its own node with 40 CPUs.  Total wall time drops by
~N×, at the cost of a final aggregation step.

## Quick start

```bash
# Submit a sinr_cdf array job (default 10 tasks × 50 trials = 500 trials)
sbatch scripts/slurm/uci-hpc3/array/exp_sinr_cdf.array.sub

# After all tasks finish, merge per-task results
python3 scripts/aggregate_array_batch.py sinr_cdf <jobid>

# Then plot using the aggregated result:
python3 scripts/plot_sinr_cdf.py \
    --result results/exp_sinr_cdf/array_<jobid>_aggregated/
```

## Directory layout (experiment-first)

```
results/
├── exp_sinr_cdf/
│   ├── 20260522_183000/                  # normal single-job run
│   ├── array_12345_task_0/               # array task 0
│   ├── array_12345_task_1/               # array task 1
│   ├── ...
│   └── array_12345_aggregated/           # merged result
├── exp_gamma_sweep/
│   ├── array_12346_task_0/, ...
│   └── array_12346_aggregated/
└── ...
```

All runs of a given experiment live under a single `exp_<name>/`
subtree — easy to find, easy to sync per-experiment, easy to delete
in bulk.

## What's in a per-task or aggregated directory

For a **single-kind** experiment (CDF):

```
results/exp_sinr_cdf/array_12345_task_0/
├── manifest.json            # experiment metadata
├── result.npz               # per-trial arrays
├── result.json              # SimResult scalars + sidecar
└── logs/run.log
```

For a **sweep-kind** experiment:

```
results/exp_gamma_sweep/array_12346_task_0/
├── manifest.json
├── result_gamma_db_5.0.npz  # one per sweep value
├── result_gamma_db_5.0.json
├── result_gamma_db_10.0.npz
├── result_gamma_db_10.0.json
├── ...
└── logs/run.log
```

After aggregation, the merged dir has the same structure with arrays
concatenated across tasks.

## How seed-stream isolation works

`scripts/slurm/uci-hpc3/array/_array_common.sh` computes per-task:

```bash
SEED=$((BASE_SEED + ARRAY_TASK_ID))     # different per task
N_TRIALS=$(( (N_TRIALS_TOTAL + N_ARRAY_TASKS - 1) / N_ARRAY_TASKS ))
RUN_ID="array_${ARRAY_JOB_ID}_task_${ARRAY_TASK_ID}"
OUTPUT_ROOT="results"                    # runner appends exp_<name>/
```

The Python runner uses `numpy.random.SeedSequence(seed).spawn(...)`
to derive per-trial seeds.  `SeedSequence` is **prefix-stable**:
giving each task a different base seed avoids the prefix collision
and gives each task an independent trial stream.

The `RUN_ID` env var is plumbed through `scripts/exp_*.sh` to the
`--run-id` CLI flag, which the runner uses in place of the auto-
timestamp.  Result: each task's output lands at
`results/exp_<name>/array_<jobid>_task_<id>/` — predictable, unique,
trivially globbed by the aggregator.

## Override knobs at submit time

```bash
# Bigger campaign: 1000 trials split across 20 tasks
N_TRIALS_TOTAL=1000 N_ARRAY_TASKS=20 \
    sbatch --array=0-19 \
    scripts/slurm/uci-hpc3/array/exp_sinr_cdf.array.sub

# Different base seed (avoid collision with earlier runs)
BASE_SEED=1000 \
    sbatch scripts/slurm/uci-hpc3/array/exp_gamma_sweep.array.sub

# Free partition + your own account
sbatch --partition=free --account=your_lab \
    scripts/slurm/uci-hpc3/array/exp_kappa_sweep.array.sub
```

Note: `--array=0-N` must match `N_ARRAY_TASKS` to keep trial accounting
correct.  If you bump one, bump the other.

## Aggregation

```bash
# Most common
python3 scripts/aggregate_array_batch.py sinr_cdf 12345
# → reads results/exp_sinr_cdf/array_12345_task_*/
# → writes results/exp_sinr_cdf/array_12345_aggregated/

# 'array_' prefix is accepted too:
python3 scripts/aggregate_array_batch.py sinr_cdf array_12345

# 'exp_' prefix on name is accepted too:
python3 scripts/aggregate_array_batch.py exp_sinr_cdf 12345

# Dry-run discovery (no merging or writing)
python3 scripts/aggregate_array_batch.py sinr_cdf 12345 --dry-run

# Override results root
python3 scripts/aggregate_array_batch.py sinr_cdf 12345 \
    --results-root /custom/results
```

The aggregator dispatches on experiment **kind**:
- `single` (CDF): concatenates per-trial SINR/SCNR samples across tasks
- `sweep`: concatenates per-trial samples *per sweep axis value* across tasks
- `trace`: not applicable (excluded from `ARRAY_ENABLED_EXPERIMENTS`)

### Missing tasks

If some tasks failed (SLURM out-of-time, node failure, etc.), the
aggregator emits a warning per missing task but proceeds with the
remaining ones.  Re-submit failed indices:

```bash
# Re-submit only tasks 3 and 7
sbatch --array=3,7 scripts/slurm/uci-hpc3/array/exp_sinr_cdf.array.sub
```

then re-run the aggregator — it will overwrite the previous
`*_aggregated/` dir with the new merge.

## Syncing results from HPC3

The existing sync script works seamlessly with the new layout:

```bash
# Pull everything (all experiments, all single + array runs)
bash scripts/sync_results_from_hpc3.sh

# Pull one experiment (all its single + array runs)
bash scripts/sync_results_from_hpc3.sh exp_sinr_cdf

# Pull only one array's tasks + aggregated result
bash scripts/sync_results_from_hpc3.sh --array-id 12345 exp_sinr_cdf

# Dry-run any of the above
bash scripts/sync_results_from_hpc3.sh --dry-run exp_sinr_cdf
```

The `--array-id` filter scopes rsync to subdirs named
`array_<id>_task_*` and `array_<id>_aggregated`.  Useful when you have
many old array runs locally and only want the latest one from the
cluster.

## When to use single-job vs array

| Scenario | Use |
|---|---|
| N_TRIALS ≤ 100, quick paper-figure pilot | Single .sub (8 h wall, fits) |
| N_TRIALS 100-300, sweep w/ ~5 points | Single .sub still works |
| **N_TRIALS ≥ 500, paper-grade figures** | **Array (10-20 tasks)** |
| Need different seed batches for variance | Array (each task = batch) |
| Hot-fix on a failed batch | Array (re-submit specific indices) |
| `convergence_trace`, `fronthaul_table` | Always single .sub |

## Resource accounting

| Submission | Nodes | CPUs/node | Wall time | Total core-h |
|---|---|---|---|---|
| Single .sub (8 h) | 1 | 40 | up to 8 h | up to 320 |
| Array, 10 tasks, ~50 min real | 10 | 40 | ~50 min | ~333 |
| Array, 20 tasks, ~25 min real | 20 | 40 | ~25 min | ~333 |

Same total CPU-hours, shorter wall time for arrays — good for paper
deadlines.

