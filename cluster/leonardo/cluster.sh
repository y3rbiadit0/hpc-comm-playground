#!/usr/bin/env bash
# Leonardo: everything this machine does differently.
#
# This file is the interface every cluster implements. The harness under
# cluster/harness/ is machine-independent and reaches the machine only through
# what is defined here -- so adding a second cluster means writing one of these
# plus an env/ and runtime/ directory, and changing nothing in the harness.
#
# Selected by GPU_BENCH_CLUSTER (default: leonardo).
#
# Required of every implementation:
#
#   SBATCH_PARTITION, optional SBATCH_ACCOUNT
#   gpu_bench_backend_fields <backend>    -> stack, runtime, launcher, preset,
#                                            bindir, binary prefix
#   gpu_bench_backend_names               -> the backends this cluster can run
#   gpu_bench_walltime_for <nodes>        -> queue policy
#   gpu_bench_cluster_environment <stack> -> modules, compilers, library paths
#   gpu_bench_cluster_runtime <runtime>   -> comm-library settings for a run
#   gpu_bench_cluster_env_files <stack> <runtime>
#                                         -> the files above, in execution
#                                            order, for `launch.sh --explain`

GPU_BENCH_CLUSTER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Optional account and default partition.
source "$GPU_BENCH_CLUSTER_DIR/slurm.sh"

# Install prefixes and the library-variant selection. Sourced here rather than
# only from environment.sh because the harness needs GPU_BENCH_VARIANT_TAG
# before any module is loaded: it keys the build directory a job resolves its
# binary from, and the results tree that job writes to.
source "$GPU_BENCH_CLUSTER_DIR/layout.sh"

# Which backends exist here, and where their binaries land.
source "$GPU_BENCH_CLUSTER_DIR/backends.sh"

# Leonardo's queue policy: more nodes means more collective work per sample, and
# a job that times out wastes the whole allocation. Short limits also backfill
# better, so this stays as tight as it safely can.
gpu_bench_walltime_for() {
  local nodes="$1"
  if   (( nodes >= 8 )); then echo "00:20:00"
  elif (( nodes >= 4 )); then echo "00:15:00"
  else                        echo "00:10:00"
  fi
}

gpu_bench_cluster_environment() {
  source "$GPU_BENCH_CLUSTER_DIR/environment.sh" "$1"
}

gpu_bench_cluster_runtime() {
  source "$GPU_BENCH_CLUSTER_DIR/runtime/$1.sh"
}

# The files the two functions above touch, earliest first. `launch.sh --explain`
# reads these statically to report which file sets each variable and which wins.
gpu_bench_cluster_env_files() {
  local stack="$1" runtime="$2"
  echo "$GPU_BENCH_CLUSTER_DIR/slurm.sh"
  echo "$GPU_BENCH_CLUSTER_DIR/layout.sh"
  echo "$GPU_BENCH_CLUSTER_DIR/env/$stack.sh"
  # The MPI-backed runtimes source the shared baseline before their own settings.
  grep -q '_openmpi.sh' "$GPU_BENCH_CLUSTER_DIR/runtime/$runtime.sh" 2>/dev/null \
    && echo "$GPU_BENCH_CLUSTER_DIR/runtime/_openmpi.sh"
  grep -q '_ucx-gpu.sh' "$GPU_BENCH_CLUSTER_DIR/runtime/$runtime.sh" 2>/dev/null \
    && echo "$GPU_BENCH_CLUSTER_DIR/runtime/_ucx-gpu.sh"
  echo "$GPU_BENCH_CLUSTER_DIR/runtime/$runtime.sh"
}

# Leonardo ships ncu with nvhpc; 24.5 cannot attach, so 25.3 is the default.
export GPU_BENCH_NCU_BIN=${GPU_BENCH_NCU_BIN:-/leonardo/prod/opt/compilers/nvhpc/25.3/binary/Linux_x86_64/25.3/compilers/bin/ncu}
