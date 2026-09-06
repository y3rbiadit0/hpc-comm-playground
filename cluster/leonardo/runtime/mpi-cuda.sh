#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_openmpi.sh"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_ucx-gpu.sh"

# NCCL's own service level, for the NCCL-backed benchmarks built on this stack.
# Scoped to multi-node like the UCX equivalent: it selects an InfiniBand service
# level, and there is no InfiniBand inside a node.
if [[ ${GPU_BENCH_JOB_NODES:-1} -gt 1 ]]; then
  export NCCL_IB_SL=${NCCL_IB_SL:-1}
fi
