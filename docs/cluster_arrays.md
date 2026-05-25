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

# After all tasks finish, merge per-task pickles
python3 scripts/aggregate_array_batch.py results/array_<jobid>/

# Then plot using the aggregated result:
python3 scripts/plot_sinr_cdf.py \
    --result results/array_<jobid>/aggregated/result.pkl
```

## What changed in Stage 21

### Existing single-job `.sub` scripts (unchanged interface, bumped resources)

- 40 CPUs per task (was 16) — uses full HPC3 standard partition node
- 8h wall time (was 1-4h) — uniform envelope for all experiments
- No `--mem` directive — proportional default at 40 CPUs gives ~192 GB

Still submit the same way:
```bash
sbatch scripts/slurm/uci-hpc3/exp_sinr_cdf.sub
```

### New array `.array.sub` scripts (9 experiments)

Located in `scripts/slurm/uci-hpc3/array/`:
- `exp_sinr_cdf.array.sub`, `exp_scnr_cdf.array.sub`
- `exp_gamma_sweep.array.sub`, `exp_kappa_sweep.array.sub`,
  `exp_snr_sweep.array.sub`, `exp_n_ap_sweep.array.sub`,
  `exp_n_ue_sweep.array.sub`, `exp_antennas_sweep.array.sub`,
  `exp_clutter_cnr_sweep.array.sub`

**Not** included as arrays (no benefit):
- `convergence_trace` — single trial, not parallelizable across trials
- `fronthaul_table` — deterministic, fast, single job is fine

Each array script:
- Spawns 10 tasks by default (`--array=0-9`)
- 4h per task — plenty of headroom even at 500+ trials
- 40 CPUs per task
- Each task uses a unique `SEED = BASE_SEED + SLURM_ARRAY_TASK_ID`
- Each task writes to `results/array_<jobid>/task_<task_id>/`

### How seed-stream isolation works

`scripts/slurm/uci-hpc3/array/_array_common.sh` computes:

```bash
SEED=$((BASE_SEED + ARRAY_TASK_ID))     # different per task
N_TRIALS=$(( (N_TRIALS_TOTAL + N_ARRAY_TASKS - 1) / N_ARRAY_TASKS ))
OUTPUT_ROOT="results/array_${ARRAY_JOB_ID}/task_${ARRAY_TASK_ID}"
```

The Python runner uses `numpy.random.SeedSequence(seed).spawn(...)`
to derive per-trial seeds.  `SeedSequence` is **prefix-stable**:
`SeedSequence(42).spawn(10)` and `SeedSequence(42).spawn(20)` share
the first 10 spawns.  By giving each task a **different** base seed,
we avoid the prefix collision and each task gets an independent
trial stream.

No changes to the Python runner were needed — `scripts/exp_*.sh`
already accepts `SEED` and `N_TRIALS` env vars.

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

After all tasks complete, run:

```bash
python3 scripts/aggregate_array_batch.py results/array_<jobid>/
```

The aggregator:

1. Globs `results/array_<jobid>/task_*/exp_*/*/result.pkl`
2. Loads each per-task SimResult
3. Concatenates per-trial samples across tasks (SINR, SCNR, runtime,
   convergence, etc.)
4. Recomputes aggregate scalars (`iters_mean`, `convergence_rate`, ...)
5. Writes `results/array_<jobid>/aggregated/result.pkl`

### Missing tasks

If some tasks failed (SLURM out-of-time, node failure, etc.), the
aggregator emits a warning per missing task but proceeds with the
remaining ones.  The aggregate trial count reflects only the
successful tasks — you can re-submit the failed indices:

```bash
# Re-submit tasks 3 and 7 only
sbatch --array=3,7 scripts/slurm/uci-hpc3/array/exp_sinr_cdf.array.sub
```

then re-run the aggregator on the same array dir to merge in the new
data.

### Dry-run discovery

To check what the aggregator will pick up before merging:

```bash
python3 scripts/aggregate_array_batch.py results/array_<jobid>/ --dry-run
```

## When to use single-job vs array

| Scenario | Use |
|---|---|
| N_TRIALS ≤ 100, quick paper-figure pilot | Single .sub (8 h wall, fits comfortably) |
| N_TRIALS 100-300, sweep w/ ~5 points | Single .sub still works |
| N_TRIALS ≥ 500, paper-grade figures | Array (10-20 tasks) |
| Need different seed batches for variance | Array (each task = batch) |
| Hot-fix on a failed batch | Array (re-submit specific indices) |
| `convergence_trace`, `fronthaul_table` | Always single .sub |

## Resource accounting

Single-job .sub:
- 1 node × 40 CPUs × 8 h = 320 core-hours
- Wall time: up to 8 h

Array .sub (10 tasks, default):
- 10 nodes × 40 CPUs × ~50 min real = ~333 core-hours (similar total)
- Wall time: ~50 min (10× faster real time)

Array .sub (20 tasks):
- 20 × 40 × ~25 min = ~333 core-hours
- Wall time: ~25 min

You're trading the same total CPU-hours for shorter wall time.  Good
for paper deadlines.

