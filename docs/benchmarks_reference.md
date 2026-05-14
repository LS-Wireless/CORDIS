# CORDIS Benchmark Registry Reference

Generated automatically from `cordis.algorithms.benchmarks.BENCHMARK_REGISTRY`.
Regenerate with: `./scripts/list_benchmarks.sh`

Each benchmark pairs a communication beamformer (varied across rows) with a power-allocation strategy (Phase II of CORDIS-Split, or a uniform fixed ratio ρ). All benchmarks share the same interface and return a `BenchmarkResult` for the simulation runner.

## Quick usage

```python
from cordis.algorithms.benchmarks import (
    BENCHMARK_REGISTRY, run_benchmark, run_all_benchmarks,
)

# Single benchmark
res = run_benchmark(
    "mrt_split", topo, cfg, est, sensing_stats,
    association, sigma_n_sq, Pmax,
)
# res.W_tx, res.phase_i, res.split_res, res.psr

# Whole sweep on one scenario
all_res = run_all_benchmarks(
    topo, cfg, est, sensing_stats, association,
    sigma_n_sq, Pmax, skip_failures=True,
)
```

## PA optimised via P-Split (Phase II of CORDIS-Split)

Each AP's power-splitting ratio ρ_a is chosen by the projected-gradient / CVXPY Phase II solver to satisfy the per-user SINR target γ while maximising the sensing utility. These variants isolate the *choice of communication beamformer*.

| Name | Comm BF | ρ | Description |
|------|---------|---|-------------|
| `lr_mmse_split` | `lr_mmse` | — | LR-MMSE + P-Split PA  (≡ CORDIS-Split) |
| `mrt_split` | `mrt` | — | Local MRT + P-Split PA |
| `rzf_split` | `rzf` | — | Local RZF + P-Split PA |
| `zf_split` | `zf` | — | Local ZF + P-Split PA |
| `global_zf_split` | `global_zf` | — | Global ZF + P-Split PA |
| `global_mrt_split` | `global_mrt` | — | Global MRT + P-Split PA |

## PA fixed (no Phase II optimisation)

All APs use a uniform ρ (no PA optimisation). These variants isolate the *value of the PA optimisation itself* by holding the beamformer fixed and skipping Phase II.

| Name | Comm BF | ρ | Description |
|------|---------|---|-------------|
| `lr_mmse_fixed` | `lr_mmse` | 0.50 | LR-MMSE + equal ρ=0.5  (no PA opt) |
| `mrt_fixed` | `mrt` | 0.50 | Local MRT + equal ρ=0.5 |
| `rzf_fixed` | `rzf` | 0.50 | Local RZF + equal ρ=0.5 |

## Adding a benchmark

Add a new entry to `BENCHMARK_REGISTRY` in `cordis/algorithms/benchmarks.py`. The tuple is `(description, comm_bf_method, pa_optimized, fixed_psr)`. Then run `./scripts/list_benchmarks.sh` to refresh this file.

