#!/bin/bash -l
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --cpus-per-task=8
#
# The single submitted script. Everything that varies between cells arrives as
# environment:
#
#   GPU_BENCH_BENCHMARK   pingpong | halo_1d | allreduce | alltoall | moe | cg_step
#   GPU_BENCH_BACKEND     see cluster/<cluster>/backends.sh
#   GPU_BENCH_TOPOLOGY    <nodes>n<gpus_per_node>g, e.g. 2n4g
#
# Only the invariant #SBATCH directives stay above. --nodes, --ntasks-per-node,
# --gres, --time and --job-name are passed on the sbatch command line, which
# takes precedence over #SBATCH directives -- so one file covers every shape of
# job.
#
# Submit through cluster/harness/launch.sh -- one cell, or --all for the matrix. Running `sbatch job.sh` directly without the three variables set is an
# error, not a default, because a silent default would produce results filed
# under the wrong name.

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
HARNESS="$GPU_BENCH_PROJECT_ROOT/cluster/harness"
EXP="$HARNESS/experiments"
GPU_BENCH_CLUSTER=${GPU_BENCH_CLUSTER:-leonardo}
CLUSTER_DIR="$GPU_BENCH_PROJECT_ROOT/cluster/$GPU_BENCH_CLUSTER"

for required in GPU_BENCH_BENCHMARK GPU_BENCH_BACKEND GPU_BENCH_TOPOLOGY; do
  if [[ -z "${!required:-}" ]]; then
    echo "error: $required is not set." >&2
    echo "  submit with: cluster/harness/launch.sh <benchmark> <backend> <topology>" >&2
    exit 2
  fi
done

# Everything machine-specific arrives through this one file.
source "$CLUSTER_DIR/cluster.sh"
source "$HARNESS/matrix.sh"   # topology parsing
gpu_bench_backend_fields "$GPU_BENCH_BACKEND"
gpu_bench_topology_fields "$GPU_BENCH_TOPOLOGY"

# Sanity check against what Slurm actually granted. The allocation comes from the
# sbatch command line; if it disagrees with the topology label, every result
# would be filed under a topology it was not measured on.
if [[ -n "${SLURM_JOB_NUM_NODES:-}" && "$SLURM_JOB_NUM_NODES" != "$GPU_BENCH_NODES" ]]; then
  echo "error: topology $GPU_BENCH_TOPOLOGY wants $GPU_BENCH_NODES nodes," \
       "allocation has $SLURM_JOB_NUM_NODES" >&2
  exit 3
fi

# Result names keep the historical shape: underscores in backend names become
# hyphens, so allreduce/sycl_oneccl_oshmpi/2n4g stays
# allreduce-sycl-oneccl-oshmpi-2n4g and existing results/ directories still match.
_backend_slug="${GPU_BENCH_BACKEND//_/-}"
# GPU_BENCH_VARIANT_TAG is empty unless the cluster selected a non-default build
# of a communication library, in which case it suffixes both build/ and results/
# below -- one rule, applied to both trees. They must move together: a binary
# built against one library version and filed under another is a wrong result
# that still validates.
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build${GPU_BENCH_VARIANT_TAG:-}/$GPU_BENCH_PRESET/$GPU_BENCH_BINDIR/${GPU_BENCH_BINARY_PREFIX}_${GPU_BENCH_BENCHMARK}}
# An A/B sweep must land in its own results *tree*, not a renamed directory
# inside the shared one. benchscribe derives the topology by regex-searching the
# whole path for <n>n<n>g and the backend from the report line, so a directory
# suffix is invisible to it: allreduce-sycl-mpi-1n2g-hpcx and
# allreduce-sycl-mpi-1n2g both resolve to the same cell and get averaged
# together. Use GPU_BENCH_RESULTS_ROOT and point benchscribe at that tree:
#
#   GPU_BENCH_RESULTS_ROOT=results-hpcx cluster/harness/launch.sh --all allreduce
#   python3 tools/benchscribe results-hpcx --benchmark allreduce
#
# A library-variant sweep is exactly that situation, and it is the one case the
# harness can recognise on its own -- so it separates the tree by default rather
# than relying on the operator to remember. results-nvshmem-3.7.2 alongside
# results, both still overridable.
GPU_BENCH_RESULTS_ROOT=${GPU_BENCH_RESULTS_ROOT:-results${GPU_BENCH_VARIANT_TAG:-}}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-${GPU_BENCH_BENCHMARK//_/-}-$_backend_slug-$GPU_BENCH_TOPOLOGY}

export GPU_BENCH_CLUSTER GPU_BENCH_RESULTS_ROOT
export GPU_BENCH_PROJECT_ROOT GPU_BENCH_STACK GPU_BENCH_RUNTIME GPU_BENCH_LAUNCHER
export GPU_BENCH_NODES GPU_BENCH_TASKS_PER_NODE GPU_BENCH_BINARY GPU_BENCH_RESULT_NAME

source "$EXP/$GPU_BENCH_BENCHMARK/common.sh"
gpu_bench_experiment_main
