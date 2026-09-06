#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=cg_step
GPU_BENCH_N_LABEL="grid side"

source "$GPU_BENCH_PROJECT_ROOT/cluster/harness/experiments/common.sh"

# UCC adds overhead to cg_step's two 8-byte reductions -- confirmed by the
# allreduce sweep at multi-node topologies (+25% to +49% at 2n4g/4n4g/8n4g), but
# reversed on one node, where UCC is 30-50% faster at these sizes. A scalar
# default cannot express that; this one is correct for the multi-node cells.
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}

# cg_step works on a square GPU_BENCH_N x GPU_BENCH_N grid; GPU_BENCH_N is the side length.
# Kept small so the halo exchange + two reductions dominate over stencil compute.
#
# Unlike the microbenchmarks, a bare run measures the single grid GPU_BENCH_N
# names rather than sweeping: the side sets each rank's column count, so a
# default sweep would change what "the cg_step result" means. GPU_BENCH_MSG_SIZES
# asks for a sweep explicitly, and its values are grid sides, not message sizes.
gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-512}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-50}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-10}
  # cg_step binaries accept: <max_side> [iterations] [warmup] [comma-separated sides]
  GPU_BENCH_EXTRA_ARGS=${GPU_BENCH_EXTRA_ARGS:-"$GPU_BENCH_ITERS $GPU_BENCH_WARMUP"}
  if [[ -n "${GPU_BENCH_MSG_SIZES:-}" ]]; then
    GPU_BENCH_EXTRA_ARGS="$GPU_BENCH_EXTRA_ARGS $GPU_BENCH_MSG_SIZES"
  fi
}

gpu_bench_experiment_extra_summary() {
  echo "grid sides: ${GPU_BENCH_MSG_SIZES:-$GPU_BENCH_N}"
  # The phase breakdown is a second, more synchronized pass; recorded for every
  # run so a job log says whether the reported record carries phase fields.
  echo "phase breakdown: ${GPU_BENCH_CG_PHASES:-off}"
}

gpu_bench_cg_step_main() { gpu_bench_experiment_main; }
