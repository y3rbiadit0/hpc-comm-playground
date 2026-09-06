#!/usr/bin/env bash
set -euo pipefail

# _openmpi.sh only: NVSHMEM drives its own transport, so _ucx-gpu.sh's
# rendezvous tuning would not apply. See the TLS choice at the end of this file.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_openmpi.sh"

# Aligned with the other MPI-using runtimes. NVSHMEM uses MPI only to bootstrap
# and for the harness's own validation/statistics reductions, both outside every
# timed region, so this does not affect the measurement - it is set so the only
# difference between runtime files is the transport actually under test.
# Open MPI 4.x renamed this; the old name still works but prints a
# deprecation banner into every rank's stderr on every run.

export NVSHMEM_BOOTSTRAP=${NVSHMEM_BOOTSTRAP:-MPI}

# Transport defaults follow the library the run selected (layout.sh).
#
# The nvhpc module ships NVSHMEM 2.11, which has only the IBRC proxy path. A
# bootstrapped 3.x has CPU-assisted IBGDA, which is the reason for bootstrapping
# one at all: Leonardo cannot run classic IBGDA -- it needs nvidia.ko loaded with
# PeerMappingOverride=1, absent from /proc/driver/nvidia/params, and
# NVreg_EnableStreamMemOPs=1, which is 0. Both live in
# /etc/modprobe.d/nvidia.conf behind a driver reload, so both are a CINECA
# request rather than a job-script change. CPU-assisted IBGDA is the user-space
# route around that: the GPU still generates the work requests and a CPU handler
# rings the NIC doorbell, so the mapping the driver will not grant is never
# needed. It arrived in 3.0 and stopped requiring GDRCopy in 3.4.5.
#
# The version decides only the default transport. Everything else follows from
# the transport actually in effect, so NVSHMEM_REMOTE_TRANSPORT is a single
# coherent knob: setting it to ibrc for a 3.x run holds the transport constant
# and reads the version against 2.11 on the same proxy path -- which separates
# "newer NVSHMEM" from "IBGDA instead of IBRC", two claims that are otherwise
# measured as one, and it does so without also having to remember to turn the
# two IBGDA settings off by hand.
if [[ ${GPU_BENCH_NVSHMEM_VERSION:-module} == module ]]; then
  export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-ibrc}
else
  export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-ibgda}
fi

if [[ ${NVSHMEM_REMOTE_TRANSPORT} == ibgda ]]; then
  export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-1}
  export NVSHMEM_IBGDA_NIC_HANDLER=${NVSHMEM_IBGDA_NIC_HANDLER:-cpu}
else
  export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-0}
fi
if [[ ${NVSHMEM_IB_ENABLE_IBGDA} == 1 && ${NVSHMEM_REMOTE_TRANSPORT} == ibrc ]]; then
  printf '%s\n' \
    'warning: NVSHMEM_IB_ENABLE_IBGDA=1 does not override NVSHMEM_REMOTE_TRANSPORT=ibrc; this run will still use IBRC' \
    >&2
fi
# NVSHMEM dispatches its collectives to NCCL when it is available. Disabling
# that leaves its own reduction path, which does not scale:
#
#   allreduce 1n4g, 16 MiB   DISABLE_NCCL=1   0.63 GB/s
#                            DISABLE_NCCL=0  85.13 GB/s   (135x)
#
# With NCCL enabled the result is within 0.2% of cuda_nccl at every size, which
# is the point: NVSHMEM's allreduce *is* NCCL's. That equivalence is a result to
# report, not something to hide by turning NCCL off -- and off is not a
# defensible alternative, since 0.63 GB/s is a non-scaling fallback rather than
# a measurement of anything users would run.
#
# NVSHMEM's native strength is RMA, which halo_1d and pingpong measure and which
# this flag does not affect.
export NVSHMEM_DISABLE_NCCL=${NVSHMEM_DISABLE_NCCL:-0}
export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}

if [[ ${GPU_BENCH_JOB_NODES:-1} -gt 1 ]]; then
  export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}
  export UCX_TLS=${GPU_BENCH_NVSHMEM_UCX_TLS:-sm,cuda_copy,cuda_ipc,rc,self}
  # Service level 1 enables adaptive routing on Leonardo's Dragonfly+ fabric.
  # UCX covers the MPI bootstrap; the ibrc data plane needs NVSHMEM's own knob.
  export UCX_IB_SL=${UCX_IB_SL:-1}
  export NVSHMEM_IB_SL=${NVSHMEM_IB_SL:-1}
else
  export UCX_TLS=${GPU_BENCH_NVSHMEM_UCX_TLS:-sm,cuda_copy,cuda_ipc,self}
fi
