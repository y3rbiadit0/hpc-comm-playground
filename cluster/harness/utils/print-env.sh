#!/usr/bin/env bash
# Pretty-print the communication-relevant environment as an ENV block.
# Sourced by the cluster's environment.sh; call gpu_bench_print_env from
# experiment summary functions so every stdout log is self-describing.

gpu_bench_print_env() {
  local vars=(
    # UCX (Open MPI / OSHMPI data plane)
    UCX_NET_DEVICES
    UCX_IB_GPU_DIRECT_RDMA
    UCX_IB_SL
    UCX_MAX_RNDV_RAILS
    UCX_TLS
    UCX_RNDV_SCHEME
    UCX_RNDV_THRESH
    # Open MPI MCA
    OMPI_MCA_pml
    OMPI_MCA_btl
    OMPI_MCA_osc
    OMPI_MCA_coll_hcoll_enable
    OMPI_MCA_coll_ucc_enable
    OMPI_MCA_opal_cuda_support
    # NCCL
    NCCL_IB_SL
    NCCL_ALGO
    NCCL_PROTO
    NCCL_SOCKET_IFNAME
    NCCL_DEBUG
    # NVSHMEM / SHMEM
    # Which NVSHMEM this run resolved, and from where. "module" is the one the
    # nvhpc module ships; anything else is a bootstrapped release. Without these
    # two a log cannot say which library produced its numbers, and that is the
    # whole point of running more than one.
    GPU_BENCH_NVSHMEM_VERSION
    NVSHMEM_HOME
    NVSHMEM_IB_SL
    NVSHMEM_BOOTSTRAP
    NVSHMEM_REMOTE_TRANSPORT
    NVSHMEM_IB_ENABLE_IBGDA
    # Which side rings the NIC doorbell under IBGDA. Only meaningful once the
    # transport is ibgda, and load-bearing there: cpu is the handler that makes
    # IBGDA usable without PeerMappingOverride=1.
    NVSHMEM_IBGDA_NIC_HANDLER
    SHMEM_SYMMETRIC_SIZE
    OSHMPI_MPI_GPU_FEATURES
    # oneCCL / Intel MPI / libfabric
    CCL_BACKEND
    CCL_ATL_TRANSPORT
    CCL_MPI_LIBRARY_PATH
    CCL_WORKER_COUNT
    I_MPI_FABRICS
    I_MPI_OFI_PROVIDER
    FI_PROVIDER
    FI_PROVIDER_PATH
    FI_LOG_LEVEL
    # Host-side execution
    OMP_NUM_THREADS
    SLURM_CPU_BIND
    # Names the MPI install in use, so a job log says which one produced it.
    OPAL_PREFIX
  )

  echo "${GPU_BENCH_CLUSTER:-cluster} ENV : {"
  local var
  for var in "${vars[@]}"; do
    echo "  ${var}: ${!var:-unset}"
  done
  echo "}"
}
