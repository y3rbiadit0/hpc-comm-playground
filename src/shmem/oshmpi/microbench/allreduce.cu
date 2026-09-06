#include <cuda_runtime.h>

#include <cstddef>

#ifndef USE_CUDA
#define USE_CUDA 1
#endif
#include <shmem.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "stats/collective_shmem.hpp"
#include "oshmpi_space.h"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

// Float32 sum allreduce, OSHMPI, over a message-size sweep.
//
// The data lives on the GPU, as it does for every other allreduce backend
// (cuda_mpi and cuda_nccl use cudaMalloc, cuda_nvshmem uses nvshmem_malloc).
// That matters for comparability: a host-memory allreduce measured beside
// device-memory ones is a different workload, and Benchscribe normalizes all of
// them against cuda_mpi without knowing the difference.
//
// OSHMPI's CUDA memory space serves RMA on device buffers - halo_1d, alltoall,
// pingpong and moe all use it - but the reduction routines are a separate
// question, so which path applies is selectable:
//
//   GPU_BENCH_OSHMPI_ALLREDUCE_MEM=device   (default)
//       source, result and pWrk come from the OSHMPI CUDA space and the
//       reduction is issued straight on device pointers, exactly like the
//       NVSHMEM implementation. The timed region is the reduction alone.
//
//       This REQUIRES Open MPI's UCC collectives (OMPI_MCA_coll_ucc_enable=1,
//       the default in cluster/leonardo/runtime/oshmpi.sh). With UCC disabled,
//       OSHMPI's reduction falls through to the host `ompi_op`
//       (ompi_op_avx_2buff_add_float_avx2), which cannot read device memory and
//       segfaults -- reproduced on Leonardo as jobs 53261883 and 53263113
//       before UCC was enabled. The check below refuses to run rather than
//       crash. With UCC on (job 53263792) all 23 sizes validate, and the result
//       lands within 1-3% of cuda_mpi, because both dispatch to the same UCC
//       collective: OSHMPI's reduction *is* Open MPI's.
//
//   GPU_BENCH_OSHMPI_ALLREDUCE_MEM=staged
//       Buffers are cudaMalloc'd and the timed region stages device->host,
//       reduces into host-symmetric memory, and stages the result back. This is
//       the fallback for a build whose reduction cannot take device pointers,
//       and it is what measures the staging cost itself. It is 34x slower than
//       the device path at 16 MiB (12160 us vs 354 us, 1n4g), most of which is
//       pageable-memory PCIe traffic at roughly 7 GB/s.
//
// The chosen path is reported in `memory=`, so a results line always says which
// of the two produced it.

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

bool validate_result(const float* values, std::size_t count, float expected) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!gpu_bench::nearly_equal(values[i], expected)) {
      return false;
    }
  }
  return true;
}

enum class reduce_memory { staged, device };

reduce_memory parse_reduce_memory() {
  const char* value = std::getenv("GPU_BENCH_OSHMPI_ALLREDUCE_MEM");
  if (value == nullptr || std::strcmp(value, "device") == 0) {
    return reduce_memory::device;
  }
  if (std::strcmp(value, "staged") == 0) {
    return reduce_memory::staged;
  }
  throw std::runtime_error(
      "GPU_BENCH_OSHMPI_ALLREDUCE_MEM must be 'staged' or 'device', got: " + std::string(value));
}

}  // namespace

int main(int argc, char** argv) {
  shmem_init();

  const int pe = shmem_my_pe();
  const int pes = shmem_n_pes();
  void* space = nullptr;

  try {
    const auto mode = parse_reduce_memory();
    if (mode == reduce_memory::device) {
      const char* ucc = std::getenv("OMPI_MCA_coll_ucc_enable");
      if (ucc != nullptr && std::strcmp(ucc, "0") == 0) {
        throw std::runtime_error(
            "GPU_BENCH_OSHMPI_ALLREDUCE_MEM=device needs OMPI_MCA_coll_ucc_enable=1; with UCC "
            "disabled OSHMPI's reduction runs as a host Open MPI op and segfaults on device "
            "memory. Enable UCC, or use GPU_BENCH_OSHMPI_ALLREDUCE_MEM=staged.");
      }
    }
    const auto max_elements = gpu_bench::parse_size_arg(argc, argv, 4194304U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elements);
    const auto max_bytes =
        gpu_bench::checked_size_multiply(max_elements, sizeof(float), "allreduce allocation");
    const auto pwrk_elements = std::max<std::size_t>(max_elements / 2U + 1U,
                                                      SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    const auto pwrk_bytes =
        gpu_bench::checked_size_multiply(pwrk_elements, sizeof(float), "allreduce work buffer");
    const auto psync_bytes = gpu_bench::checked_size_multiply(
        static_cast<std::size_t>(SHMEM_REDUCE_SYNC_SIZE), sizeof(long), "allreduce sync buffer");
    const auto ok_bytes = gpu_bench::checked_size_multiply(
        static_cast<std::size_t>(pes), sizeof(double), "allreduce validation buffer");
    const auto gather_bytes =
        gpu_bench::checked_size_multiply(gpu_bench::collective_gather_elements(pes, iterations),
                                         sizeof(double), "allreduce sample buffer");

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    // pSync and the reporting scratch are control data and stay in the default
    // symmetric heap in both modes.
    auto* psync = static_cast<long*>(shmem_malloc(psync_bytes));
    auto* ok_by_pe = static_cast<double*>(shmem_malloc(ok_bytes));
    auto* sample_gather = static_cast<double*>(shmem_malloc(gather_bytes));
    if (psync == nullptr || ok_by_pe == nullptr || sample_gather == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric scratch");
    }

    float* device_source = nullptr;   // the application's GPU buffers
    float* device_result = nullptr;
    float* sym_source = nullptr;      // what the reduction actually reads/writes
    float* sym_result = nullptr;
    float* pwrk = nullptr;

    if (mode == reduce_memory::device) {
      const auto space_bytes = 2U * max_bytes + pwrk_bytes + (1U << 20U);
      space = gpu_bench_oshmpi_space_create(space_bytes);
      if (space == nullptr) {
        throw std::runtime_error("failed to create OSHMPI CUDA memory space");
      }
      sym_source = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, max_bytes));
      sym_result = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, max_bytes));
      pwrk = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, pwrk_bytes));
      if (sym_source == nullptr || sym_result == nullptr || pwrk == nullptr) {
        throw std::runtime_error("failed to allocate OSHMPI CUDA symmetric memory");
      }
      // The symmetric device buffers are the application buffers, as they are
      // for NVSHMEM - no separate allocation and no staging.
      device_source = sym_source;
      device_result = sym_result;
    } else {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_source), max_bytes),
                 "cudaMalloc(source)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_result), max_bytes),
                 "cudaMalloc(result)");
      sym_source = static_cast<float*>(shmem_malloc(max_bytes));
      sym_result = static_cast<float*>(shmem_malloc(max_bytes));
      pwrk = static_cast<float*>(shmem_malloc(pwrk_bytes));
      if (sym_source == nullptr || sym_result == nullptr || pwrk == nullptr) {
        throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
      }
    }

    // Every PE contributes its rank+1, so the sum is pes*(pes+1)/2 everywhere.
    {
      const std::vector<float> seed(max_elements, static_cast<float>(pe + 1));
      check_cuda(cudaMemcpy(device_source, seed.data(), max_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(seed source)");
      check_cuda(cudaMemset(device_result, 0, max_bytes), "cudaMemset(result)");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(seed)");
    }
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) {
      psync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();

    const auto expected = static_cast<float>(static_cast<double>(pes) * (pes + 1) / 2.0);
    std::vector<float> host_result;
    int all_sizes_ok = 1;

    for (const auto count : message_sizes) {
      const auto bytes = gpu_bench::checked_size_multiply(count, sizeof(float), "allreduce message");
      shmem_barrier_all();

      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        if (mode == reduce_memory::staged) {
          // The staging is part of the cost: this is what it takes to reduce
          // GPU-resident data through a host-only reduction path. cudaMemcpy on
          // pageable memory is synchronous, so no extra device sync is needed.
          check_cuda(cudaMemcpy(sym_source, device_source, bytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy(stage in)");
          shmem_float_sum_to_all(sym_result, sym_source, count, 0, 0, pes, pwrk, psync);
          check_cuda(cudaMemcpy(device_result, sym_result, bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(stage out)");
        } else {
          shmem_float_sum_to_all(sym_result, sym_source, count, 0, 0, pes, pwrk, psync);
        }
      });

      const auto global = gpu_bench::collective_stats(stats, sample_gather, pe, pes);

      // Validation reads the device result, so it checks the buffer the caller
      // would actually go on to use - staging included.
      host_result.assign(count, 0.0F);
      check_cuda(cudaMemcpy(host_result.data(), device_result, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(validate)");
      const int local_ok = validate_result(host_result.data(), count, expected) ? 1 : 0;
      const double local_value = static_cast<double>(local_ok);
      shmem_putmem(ok_by_pe + pe, &local_value, sizeof(double), 0);
      shmem_quiet();
      shmem_barrier_all();

      int global_ok = local_ok;
      if (pe == 0) {
        global_ok = 1;
        for (int source_pe = 0; source_pe < pes; ++source_pe) {
          if (ok_by_pe[source_pe] < 0.5) {
            global_ok = 0;
          }
        }
        // Hand the verdict back to everyone so all PEs agree on the exit code.
        // Slot 0 is PE 0's own flag, already consumed and rewritten next size.
        const double global_value = static_cast<double>(global_ok);
        for (int target_pe = 0; target_pe < pes; ++target_pe) {
          shmem_putmem(ok_by_pe, &global_value, sizeof(global_value), target_pe);
        }
        shmem_quiet();
      }
      shmem_barrier_all();
      global_ok = ok_by_pe[0] >= 0.5 ? 1 : 0;
      all_sizes_ok = std::min(all_sizes_ok, global_ok);

      if (pe == 0) {
        const double algorithm_gbytes_per_s =
            global.avg_s > 0.0 ? static_cast<double>(bytes) / global.avg_s / 1.0e9 : 0.0;
        const double bus_gbytes_per_s =
            algorithm_gbytes_per_s * 2.0 * static_cast<double>(pes - 1) / static_cast<double>(pes);

        gpu_bench::bench_report report;
        report.name = "oshmpi_allreduce";
        report.n = count;
        report.ranks = pes;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = "datatype=float32 reduction=sum bus_gbytes_per_s=" +
                       std::to_string(bus_gbytes_per_s) + " memory=" +
                       (mode == reduce_memory::device ? "device_symmetric" : "device_staged");
        gpu_bench::print_report(report);
      }
    }

    shmem_free(sample_gather);
    shmem_free(ok_by_pe);
    shmem_free(psync);
    if (mode == reduce_memory::device) {
      // Space allocations go back before the space they came from, as OSHMPI's
      // own CUDA-space test does.
      shmem_free(pwrk);
      shmem_free(sym_result);
      shmem_free(sym_source);
      gpu_bench_oshmpi_space_destroy(space);
    } else {
      shmem_free(pwrk);
      shmem_free(sym_result);
      shmem_free(sym_source);
      check_cuda(cudaFree(device_result), "cudaFree(result)");
      check_cuda(cudaFree(device_source), "cudaFree(source)");
    }

    shmem_finalize();
    return all_sizes_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    // Space cleanup is collective and is unsafe when another PE may still be
    // inside the operation that failed locally.
    shmem_global_exit(1);
  }
}
