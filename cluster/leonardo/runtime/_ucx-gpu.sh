#!/usr/bin/env bash
# UCX transport tuning for the runtimes that move GPU buffers through Open MPI's
# UCX point-to-point layer: mpi-cuda, sycl-mpi and oshmpi.
#
# One copy, one set of override names (GPU_BENCH_UCX_*): this tuning decides an
# inter-node comparison, so a per-runtime copy is a chance for the comparison to
# measure a settings difference instead.
#
# nvshmem.sh deliberately does NOT source this: NVSHMEM drives its own transport
# and only wants the intra/inter TLS choice, not the rendezvous tuning.

# UCX owns InfiniBand here; see _openmpi.sh for why btl is ^openib.
export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}

# Make device selection and GPUDirect requirements explicit so every run uses
# the same policy rather than relying on UCX build defaults.
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
export UCX_IB_GPU_DIRECT_RDMA=${UCX_IB_GPU_DIRECT_RDMA:-yes}
# get_zcopy pulls from the target rather than pushing, which keeps the sender's
# GPU out of the transfer once the rendezvous is agreed.
if [[ -z ${UCX_RNDV_SCHEME:-} ]]; then
  export UCX_RNDV_SCHEME=${GPU_BENCH_UCX_RNDV_SCHEME:-get_zcopy}
fi
if [[ -z ${UCX_RNDV_THRESH:-} ]]; then
  export UCX_RNDV_THRESH=${GPU_BENCH_UCX_RNDV_THRESH:-1024}
fi

if [[ ${GPU_BENCH_JOB_NODES:-1} -gt 1 ]]; then
  # The cuda alias expands to cuda_copy,cuda_ipc,gdr_copy.
  if [[ -z ${UCX_TLS:-} ]]; then
    export UCX_TLS=${GPU_BENCH_UCX_TLS:-sm,rc,cuda,self}
  fi
  # Service level 1 enables adaptive routing on Leonardo's Dragonfly+ fabric.
  export UCX_IB_SL=${UCX_IB_SL:-1}
  # Two rails, though the node has four mlx5 devices.
  export UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-2}
else
  # Single node: no InfiniBand transport, so leave rc out entirely.
  if [[ -z ${UCX_TLS:-} ]]; then
    export UCX_TLS=${GPU_BENCH_UCX_TLS:-sm,cuda,self}
  fi
fi
