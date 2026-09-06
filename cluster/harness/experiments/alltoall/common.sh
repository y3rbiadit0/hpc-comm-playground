#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=alltoall
GPU_BENCH_N_LABEL="count per peer"

source "$GPU_BENCH_PROJECT_ROOT/cluster/harness/experiments/common.sh"

# UCC avoids Open MPI's pathological small-message alltoall at 16+ ranks: at
# 8n4g with 256 B per peer the tuned path takes 974 us against UCC's 31 us, and
# the collapse persists to 512 B per peer. UCC does regress large messages --
# above ~16 KB per peer on multi-node topologies, by at most 1.8x -- so this is
# a trade, not a free win, and the measured crossover is in the README.
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-1}

# alltoall sends GPU_BENCH_N elements to every peer (send/recv buffers are ranks*GPU_BENCH_N).
gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-65536}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-100}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-20}
  # alltoall binaries accept: <max_count_per_peer> [iterations] [warmup] [sizes]
  GPU_BENCH_EXTRA_ARGS=${GPU_BENCH_EXTRA_ARGS:-"$GPU_BENCH_ITERS $GPU_BENCH_WARMUP"}
  if [[ -n "${GPU_BENCH_MSG_SIZES:-}" ]]; then
    GPU_BENCH_EXTRA_ARGS="$GPU_BENCH_EXTRA_ARGS $GPU_BENCH_MSG_SIZES"
  fi
}

gpu_bench_alltoall_main() { gpu_bench_experiment_main; }

gpu_bench_experiment_extra_summary() {
  echo "message sizes: ${GPU_BENCH_MSG_SIZES:-powers-of-two}"
}
