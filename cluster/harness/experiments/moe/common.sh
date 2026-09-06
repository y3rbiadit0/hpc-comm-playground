#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=moe
GPU_BENCH_N_LABEL="tokens per rank"

source "$GPU_BENCH_PROJECT_ROOT/cluster/harness/experiments/common.sh"

# Unmeasured. This inherited the alltoall rationale, which the topology sweep
# then disproved for alltoall itself -- but MoE exchanges are alltoallv, a
# different Open MPI path, so the value stands until MoE has its own A/B rather
# than being flipped on alltoall's evidence.
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}

gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-16384}
  GPU_BENCH_HIDDEN=${GPU_BENCH_HIDDEN:-256}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-100}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-20}
  # MoE binaries accept: <tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]
  GPU_BENCH_EXTRA_ARGS="$GPU_BENCH_HIDDEN $GPU_BENCH_ITERS $GPU_BENCH_WARMUP"
  if [[ -n "${GPU_BENCH_ROUTINGS:-}" ]]; then
    GPU_BENCH_EXTRA_ARGS="$GPU_BENCH_EXTRA_ARGS $GPU_BENCH_ROUTINGS"
  fi
}

gpu_bench_experiment_extra_summary() {
  echo "hidden size: $GPU_BENCH_HIDDEN"
  echo "routing cases: ${GPU_BENCH_ROUTINGS:-uniform,locality80,hotspot80}"
}

gpu_bench_moe_main() { gpu_bench_experiment_main; }
