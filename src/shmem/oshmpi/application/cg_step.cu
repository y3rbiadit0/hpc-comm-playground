#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>

#ifndef USE_CUDA
#define USE_CUDA 1
#endif
#include <shmem.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "stats/collective_shmem.hpp"
#include "oshmpi_space.h"
#include "partition.hpp"
#include "report.hpp"
#include "benchmarks/cg_phases.hpp"
#include "benchmarks/cg_step.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

// CG iteration communication skeleton (see src/mpi/cuda/application/cg_step.cu). The halo
// columns use the OSHMPI CUDA symmetric space; the two reductions run on the host
// symmetric heap (local dot computed on the GPU then reduced), the OSHMPI model.

__global__ void init_p_kernel(float* p, std::size_t side, std::size_t local_cols, std::size_t width) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    p[i * width + (jj + 1U)] = 1.0F;
  }
}

__global__ void pack_column_kernel(const float* padded, float* contiguous, std::size_t side, std::size_t width,
                                   std::size_t column) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < side) {
    contiguous[i] = padded[i * width + column];
  }
}

__global__ void unpack_column_kernel(float* padded, const float* contiguous, std::size_t side, std::size_t width,
                                     std::size_t column) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < side) {
    padded[i * width + column] = contiguous[i];
  }
}

__global__ void spmv_kernel(const float* p, float* q, std::size_t side, std::size_t local_cols,
                            std::size_t width) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    const auto j = jj + 1U;
    const float north = i > 0 ? p[(i - 1U) * width + j] : 0.0F;
    const float south = i + 1U < side ? p[(i + 1U) * width + j] : 0.0F;
    const float west = p[i * width + (j - 1U)];
    const float east = p[i * width + (j + 1U)];
    q[i * width + j] = 0.25F * (north + south + west + east);
  }
}

// Grid-stride dot over the interior with a per-block reduction, so each block
// issues a single atomicAdd instead of one per element (avoids 16M-way
// contention on a single scalar). Launch with a 1D block of 256 threads.
__global__ void cg_dot_kernel(const float* p, const float* q, double* partial_pq, double* partial_qq,
                              std::size_t side, std::size_t local_cols, std::size_t width) {
  __shared__ double shared_pq[256];
  __shared__ double shared_qq[256];
  const std::size_t total = side * local_cols;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double thread_pq = 0.0;
  double thread_qq = 0.0;
  for (auto idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const auto off = (idx / local_cols) * width + (idx % local_cols + 1U);
    const double pv = p[off];
    const double qv = q[off];
    thread_pq += pv * qv;
    thread_qq += qv * qv;
  }
  shared_pq[threadIdx.x] = thread_pq;
  shared_qq[threadIdx.x] = thread_qq;
  __syncthreads();
  for (unsigned s = blockDim.x / 2U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) {
      shared_pq[threadIdx.x] += shared_pq[threadIdx.x + s];
      shared_qq[threadIdx.x] += shared_qq[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    atomicAdd(partial_pq, shared_pq[0]);
    atomicAdd(partial_qq, shared_qq[0]);
  }
}

}  // namespace

namespace {

enum class reduce_memory { device, staged };

/* Where the two scalar CG reductions happen.
 *
 * `device` keeps them on the GPU, as the other five backends do:
 * partial, result and pWrk all come from the OSHMPI CUDA space, so
 * shmem_double_sum_to_all is issued on device pointers. This needs UCC, which
 * cluster/harness/experiments/cg_step/common.sh turns OFF by default -- so
 * `device` refuses to start unless OMPI_MCA_coll_ucc_enable=1 is set with it.
 * Without UCC the reduction reaches a host Open MPI op and segfaults.
 *
 * `staged` (default) copies the two doubles device->host and reduces in the
 * default symmetric heap. At 8 bytes that is actually faster, but it is not
 * what the other backends do, so the difference must be reported rather than
 * presented as a like-for-like result. Measured on Leonardo 2n4g, job 54065216:
 * staged 81.4 us, device 221.9 us -- 2.73x, of which ~116 us is UCC's
 * small-message penalty, which the device path cannot avoid.
 */
reduce_memory parse_reduce_memory() {
  const char* value = std::getenv("GPU_BENCH_OSHMPI_CG_REDUCE_MEM");
  if (value == nullptr || std::strcmp(value, "staged") == 0) {
    return reduce_memory::staged;
  }
  if (std::strcmp(value, "device") == 0) {
    return reduce_memory::device;
  }
  throw std::runtime_error(
      "GPU_BENCH_OSHMPI_CG_REDUCE_MEM must be 'device' or 'staged', got: " + std::string(value));
}

}  // namespace

int main(int argc, char** argv) {
  shmem_init();

  const int pe = shmem_my_pe();
  const int pes = shmem_n_pes();
  void* space = nullptr;
  double* source = nullptr;
  double* result = nullptr;
  double* pwrk_pq = nullptr;
  double* pwrk_qq = nullptr;
  long* psync_pq = nullptr;
  long* psync_qq = nullptr;
  double* ok_by_pe = nullptr;
  double* sample_gather = nullptr;

  try {
    const auto reduce_mode = parse_reduce_memory();
    if (reduce_mode == reduce_memory::device) {
      const char* ucc = std::getenv("OMPI_MCA_coll_ucc_enable");
      if (ucc != nullptr && std::strcmp(ucc, "0") == 0) {
        throw std::runtime_error(
            "GPU_BENCH_OSHMPI_CG_REDUCE_MEM=device needs OMPI_MCA_coll_ucc_enable=1; with UCC "
            "disabled OSHMPI's reduction runs as a host Open MPI op and segfaults on device "
            "memory. Use GPU_BENCH_OSHMPI_CG_REDUCE_MEM=staged instead.");
      }
    }

    const auto max_side = gpu_bench::parse_size_arg(argc, argv, 1U << 9U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 50);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 10);
    const auto sides = gpu_bench::parse_size_list_or_single(argc, argv, 4, max_side);
    const bool phase_pass = gpu_bench::cg_phases_requested();
    const int left = pe == 0 ? -1 : pe - 1;
    const int right = pe + 1 == pes ? -1 : pe + 1;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    // One allocation for the largest side in the sweep; smaller sides use a
    // prefix of it. Symmetric allocation is collective, so every PE must size
    // from max_side rather than from its own slab.
    const auto max_field_elems = max_side * (gpu_bench::local_count(max_side, pe, pes) + 2U);

    const auto symmetric_bytes = std::max<std::size_t>(8U * max_side * sizeof(float), 1U << 20U);
    space = gpu_bench_oshmpi_space_create(symmetric_bytes);
    if (space == nullptr) {
      throw std::runtime_error("failed to create OSHMPI CUDA memory space");
    }
    float* p_field = nullptr;
    float* q_field = nullptr;
    double* partial_pq = nullptr;
    double* partial_qq = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p_field), max_field_elems * sizeof(float)), "cudaMalloc(p)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&q_field), max_field_elems * sizeof(float)), "cudaMalloc(q)");
    auto* send_west = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, max_side * sizeof(float)));
    auto* send_east = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, max_side * sizeof(float)));
    auto* recv_west = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, max_side * sizeof(float)));
    auto* recv_east = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, max_side * sizeof(float)));

    // In device mode the reduction operands must be symmetric *device* memory,
    // so partial/result/pWrk come from the CUDA space and the dot kernel writes
    // straight into the reduction's source. pSync stays in the default heap: it
    // is control data, and allreduce.cu uses the same split.
    if (reduce_mode == reduce_memory::device) {
      // Allocated from the space directly. An earlier version cudaMalloc'd them
      // and freed them here, which fails with "invalid argument" once the CUDA
      // memory space is attached -- the space owns the device allocator.
      partial_pq = static_cast<double*>(gpu_bench_oshmpi_space_malloc(space, sizeof(double)));
      partial_qq = static_cast<double*>(gpu_bench_oshmpi_space_malloc(space, sizeof(double)));
      result = static_cast<double*>(gpu_bench_oshmpi_space_malloc(space, 2U * sizeof(double)));
      pwrk_pq = static_cast<double*>(
          gpu_bench_oshmpi_space_malloc(space, SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)));
      pwrk_qq = static_cast<double*>(
          gpu_bench_oshmpi_space_malloc(space, SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)));
      source = result;  // unused in device mode; keeps the null-check uniform
    } else {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&partial_pq), sizeof(double)),
                 "cudaMalloc(partial_pq)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&partial_qq), sizeof(double)),
                 "cudaMalloc(partial_qq)");
      source = static_cast<double*>(shmem_malloc(2U * sizeof(double)));
      result = static_cast<double*>(shmem_malloc(2U * sizeof(double)));
      pwrk_pq = static_cast<double*>(shmem_malloc(SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)));
      pwrk_qq = static_cast<double*>(shmem_malloc(SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)));
    }
    psync_pq = static_cast<long*>(shmem_malloc(SHMEM_REDUCE_SYNC_SIZE * sizeof(long)));
    psync_qq = static_cast<long*>(shmem_malloc(SHMEM_REDUCE_SYNC_SIZE * sizeof(long)));
    ok_by_pe = static_cast<double*>(shmem_malloc(static_cast<std::size_t>(pes) * sizeof(double)));
    sample_gather = static_cast<double*>(
        shmem_malloc(gpu_bench::collective_gather_elements(pes, iterations) * sizeof(double)));
    if (send_west == nullptr || send_east == nullptr || recv_west == nullptr || recv_east == nullptr ||
        source == nullptr || result == nullptr || pwrk_pq == nullptr || pwrk_qq == nullptr ||
        psync_pq == nullptr || psync_qq == nullptr || ok_by_pe == nullptr ||
        sample_gather == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
    }
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) {
      psync_pq[i] = SHMEM_SYNC_VALUE;
      psync_qq[i] = SHMEM_SYNC_VALUE;
    }

    const auto sync = [&]() { check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(phase)"); };

    int all_sides_ok = 1;
    for (const std::size_t side : sides) {
      const auto local_cols = gpu_bench::local_count(side, pe, pes);
      const auto col_offset = gpu_bench::local_offset(side, pe, pes);
      const auto width = local_cols + 2U;
      const auto field_elems = side * width;

      check_cuda(cudaMemset(p_field, 0, field_elems * sizeof(float)), "cudaMemset(p)");
      check_cuda(cudaMemset(q_field, 0, field_elems * sizeof(float)), "cudaMemset(q)");
      check_cuda(cudaMemset(recv_west, 0, side * sizeof(float)), "cudaMemset(recv_west)");
      check_cuda(cudaMemset(recv_east, 0, side * sizeof(float)), "cudaMemset(recv_east)");

      const dim3 block2d(16, 16);
      const dim3 grid2d(static_cast<unsigned>((local_cols + block2d.x - 1U) / block2d.x),
                        static_cast<unsigned>((side + block2d.y - 1U) / block2d.y));
      constexpr int block1d = 256;
      const auto grid1d = static_cast<int>((side + block1d - 1) / block1d);
      const auto dot_grid =
          static_cast<int>(std::min<std::size_t>((side * local_cols + 255U) / 256U, 4096U));

      if (local_cols > 0) {
        init_p_kernel<<<grid2d, block2d>>>(p_field, side, local_cols, width);
        check_cuda(cudaGetLastError(), "init_p_kernel");
      }
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
      shmem_barrier_all();

      /* The step, split into the four phases the analysis decomposes it into.
       * Composing them is exactly the step this benchmark has always timed: the
       * pack and compute phases already synchronized before the puts and the
       * reduction could read their output, so the split adds no synchronization
       * to the headline loop - only the phase pass adds any. */
      const auto pack = [&]() {
        if (local_cols > 0) {
          pack_column_kernel<<<grid1d, block1d>>>(p_field, send_west, side, width, 1U);
          pack_column_kernel<<<grid1d, block1d>>>(p_field, send_east, side, width, local_cols);
          check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pack)");
        }
      };
      const auto halo = [&]() {
        if (left >= 0) {
          shmem_putmem(recv_east, send_west, side * sizeof(float), left);
        }
        if (right >= 0) {
          shmem_putmem(recv_west, send_east, side * sizeof(float), right);
        }
        shmem_quiet();
        shmem_barrier_all();
      };
      const auto compute = [&]() {
        check_cuda(cudaMemset(partial_pq, 0, sizeof(double)), "cudaMemset(partial_pq)");
        check_cuda(cudaMemset(partial_qq, 0, sizeof(double)), "cudaMemset(partial_qq)");
        if (local_cols > 0) {
          unpack_column_kernel<<<grid1d, block1d>>>(p_field, recv_west, side, width, 0U);
          unpack_column_kernel<<<grid1d, block1d>>>(p_field, recv_east, side, width, local_cols + 1U);
          spmv_kernel<<<grid2d, block2d>>>(p_field, q_field, side, local_cols, width);
          cg_dot_kernel<<<dot_grid, block1d>>>(p_field, q_field, partial_pq, partial_qq, side, local_cols, width);
        }
        // The reduction reads the dot kernel's output, so the partials must have
        // landed. In staged mode the device-to-host copy below would serialize
        // anyway; making it explicit keeps the phase boundary honest in both.
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(dot)");
      };
      // Two separate global reductions, as a real CG iteration performs.
      const auto reduce = [&]() {
        if (reduce_mode == reduce_memory::device) {
          // Operands stay on the GPU, matching the other five backends.
          shmem_double_sum_to_all(&result[0], partial_pq, 1, 0, 0, pes, pwrk_pq, psync_pq);
          shmem_double_sum_to_all(&result[1], partial_qq, 1, 0, 0, pes, pwrk_qq, psync_qq);
        } else {
          check_cuda(cudaMemcpy(&source[0], partial_pq, sizeof(double), cudaMemcpyDeviceToHost),
                     "cudaMemcpy(pq)");
          check_cuda(cudaMemcpy(&source[1], partial_qq, sizeof(double), cudaMemcpyDeviceToHost),
                     "cudaMemcpy(qq)");
          shmem_double_sum_to_all(&result[0], &source[0], 1, 0, 0, pes, pwrk_pq, psync_pq);
          shmem_double_sum_to_all(&result[1], &source[1], 1, 0, 0, pes, pwrk_qq, psync_qq);
        }
      };

      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        pack();
        halo();
        compute();
        reduce();
      });
      const auto global = gpu_bench::collective_stats(stats, sample_gather, pe, pes);

      gpu_bench::cg_phase_stats phase_global;
      if (phase_pass) {
        shmem_barrier_all();
        const auto phase_samples =
            gpu_bench::measure_cg_phases(warmup, iterations, sync, pack, halo, compute, reduce);
        for (int phase = 0; phase < gpu_bench::cg_phase_count; ++phase) {
          phase_global[phase] = gpu_bench::collective_stats(gpu_bench::summarize(phase_samples[phase]),
                                                            sample_gather, pe, pes);
        }
      }

      const auto ones = [](std::size_t, std::size_t) { return 1.0F; };
      const auto qval = [&](std::size_t i, std::size_t jg) { return gpu_bench::stencil5(i, jg, side, ones); };
      double ref_pq = 0.0;
      double ref_qq = 0.0;
      for (std::size_t i = 0; i < side; ++i) {
        for (std::size_t jg = 0; jg < side; ++jg) {
          const double q = qval(i, jg);
          ref_pq += q;
          ref_qq += q * q;
        }
      }
      double final_pq = 0.0;
      double final_qq = 0.0;
      if (reduce_mode == reduce_memory::device) {
        double host_result[2] = {0.0, 0.0};
        check_cuda(cudaMemcpy(host_result, result, 2U * sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(result)");
        final_pq = host_result[0];
        final_qq = host_result[1];
      } else {
        final_pq = result[0];
        final_qq = result[1];
      }
      int local_ok =
          gpu_bench::nearly_equal(final_pq, ref_pq) && gpu_bench::nearly_equal(final_qq, ref_qq) ? 1 : 0;
      if (local_cols > 0) {
        std::vector<float> host_q(field_elems);
        check_cuda(cudaMemcpy(host_q.data(), q_field, field_elems * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(q)");
        if (!gpu_bench::validate_columns(host_q.data(), side, local_cols, width, col_offset, qval)) {
          local_ok = 0;
        }
      }

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
      }
      all_sides_ok = all_sides_ok && global_ok;

      if (pe == 0) {
        gpu_bench::bench_report report;
        report.name = "oshmpi_cg_step";
        report.n = side;
        report.ranks = pes;
        report.bytes_per_iter = 2U * side * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = std::string("reduce_mem=") +
                       (reduce_mode == reduce_memory::device ? "device_symmetric" : "host_staged");
        if (phase_pass) {
          report.extra += ' ' + gpu_bench::cg_phase_fields(phase_global);
        }
        gpu_bench::print_report(report);
      }
      shmem_barrier_all();
    }

    check_cuda(cudaFree(q_field), "cudaFree(q)");
    check_cuda(cudaFree(p_field), "cudaFree(p)");
    shmem_free(sample_gather);
    shmem_free(ok_by_pe);
    shmem_free(psync_qq);
    shmem_free(psync_pq);
    shmem_free(pwrk_qq);
    shmem_free(pwrk_pq);
    if (reduce_mode == reduce_memory::device) {
      // Space allocations go back with shmem_free, before the space itself.
      shmem_free(result);
      shmem_free(partial_qq);
      shmem_free(partial_pq);
    } else {
      shmem_free(result);
      shmem_free(source);
      check_cuda(cudaFree(partial_qq), "cudaFree(partial_qq)");
      check_cuda(cudaFree(partial_pq), "cudaFree(partial_pq)");
    }
    // The space allocations go back before the space they came from does, as
    // OSHMPI's own CUDA-space test does.
    shmem_free(recv_east);
    shmem_free(recv_west);
    shmem_free(send_east);
    shmem_free(send_west);
    gpu_bench_oshmpi_space_destroy(space);

    shmem_finalize();
    return all_sides_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    // Space cleanup is collective and is unsafe when another PE may still be
    // inside the operation that failed locally.
    shmem_global_exit(1);
  }
}
