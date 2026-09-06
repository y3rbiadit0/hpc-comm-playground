#!/usr/bin/env bash
# Backend registry: everything that is constant about a backend, in one table.
#
# Fields, colon-separated: stack:runtime:launcher:preset:bindir:binary_prefix
#
#   stack           cuda | sycl -- selects the toolchain module set
#   runtime         cluster/leonardo/runtime/<runtime>.sh, the env for this backend
#   launcher        srun | mpirun -- oneCCL needs mpirun; see runtime/oneccl-*.sh
#   preset          CMake preset, so build/<preset>/... locates the binary
#   bindir          where CMake puts that preset's binaries, mirroring src/
#   binary_prefix   target name prefix; the binary is <prefix>_<benchmark>
#
# sycl_oneccl and sycl_oneccl_oshmpi build the *same* sources into the same
# target names from two different presets -- the difference is which transport
# oneCCL was configured against -- so they share a binary_prefix and differ in
# preset and runtime.
#
# A library *version* is deliberately not a backend row. cuda_nvshmem names the
# backend; which NVSHMEM it was built against is a property of the run, selected
# by GPU_BENCH_NVSHMEM_VERSION (see layout.sh) and recorded in the build and
# results paths. A row per version would multiply this table by every version
# anyone wanted to try, and would need a matching preset, benchscribe enum entry
# and plot colour for each.
GPU_BENCH_BACKENDS=(
  "cuda_mpi:cuda:mpi-cuda:srun:leonardo-cuda-mpi:src/mpi/cuda:cuda_mpi"
  "cuda_nccl:cuda:mpi-cuda:srun:leonardo-cuda-nccl:src/xccl/cuda:cuda_nccl"
  "cuda_nvshmem:cuda:nvshmem:srun:leonardo-cuda-nvshmem:src/shmem/nvshmem:cuda_nvshmem"
  "oshmpi:cuda:oshmpi:srun:leonardo-oshmpi:src/shmem/oshmpi:oshmpi"
  "sycl_mpi:sycl:sycl-mpi:srun:leonardo-sycl-mpi:src/mpi/sycl:sycl_mpi"
  "sycl_oneccl:sycl:oneccl-nccl:mpirun:leonardo-sycl-oneccl:src/xccl/sycl:sycl_oneccl"
  "sycl_oneccl_oshmpi:sycl:oneccl-oshmpi:mpirun:leonardo-sycl-oneccl-oshmpi:src/xccl/sycl:sycl_oneccl"
)

# gpu_bench_backend_fields <backend> -> sets GPU_BENCH_STACK, _RUNTIME,
# _LAUNCHER, _PRESET, _BINARY_PREFIX. Fails loudly on an unknown backend rather
# than submitting a job that would die after the allocation is granted.
gpu_bench_backend_fields() {
  local want="$1" entry name
  for entry in "${GPU_BENCH_BACKENDS[@]}"; do
    name="${entry%%:*}"
    if [[ "$name" == "$want" ]]; then
      IFS=: read -r _ GPU_BENCH_STACK GPU_BENCH_RUNTIME GPU_BENCH_LAUNCHER \
                     GPU_BENCH_PRESET GPU_BENCH_BINDIR GPU_BENCH_BINARY_PREFIX <<<"$entry"
      return 0
    fi
  done
  echo "error: unknown backend '$want'; known: $(gpu_bench_backend_names)" >&2
  return 1
}

gpu_bench_backend_names() {
  local entry
  for entry in "${GPU_BENCH_BACKENDS[@]}"; do printf '%s ' "${entry%%:*}"; done
}
