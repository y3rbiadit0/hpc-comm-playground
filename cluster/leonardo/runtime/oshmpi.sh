#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_openmpi.sh"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_ucx-gpu.sh"

export SHMEM_INFO=${SHMEM_INFO:-0}
export SHMEM_VERSION=${SHMEM_VERSION:-0}

export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}
export OSHMPI_MPI_GPU_FEATURES=${OSHMPI_MPI_GPU_FEATURES:-all}
export OSHMPI_VERBOSE=${OSHMPI_VERBOSE:-0}

# OSHMPI implements OpenSHMEM on MPI-3 one-sided, so the osc component is this
# backend's RMA data plane; ucx is the one that takes device pointers.
#
# Reductions are a separate component (coll) and a separate decision: they need
# UCC to stay on device, which the experiment shims set per benchmark. See the
# GPU_BENCH_OSHMPI_*_MEM handling in src/shmem/oshmpi.
export OMPI_MCA_osc=${OMPI_MCA_osc:-ucx}
export OMPI_MCA_osc_base_verbose=${OMPI_MCA_osc_base_verbose:-0}
