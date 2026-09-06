#include <mpi.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "stats/collective_mpi.hpp"
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

// Communication skeleton of a conjugate-gradient iteration on a 2D 5-point
// Laplacian: an SpMV (column-slab halo exchange of p + stencil -> q) followed by
// two global reductions dot(p,q) and dot(q,q). p == 1 everywhere, so the
// reductions have exact references and q validates the halo (see reference()).
// Combines the column-slab halo exchange and global reductions in one step.

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

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  try {
    const auto max_side = gpu_bench::parse_size_arg(argc, argv, 1U << 9U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 50);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 10);
    const auto sides = gpu_bench::parse_size_list_or_single(argc, argv, 4, max_side);
    const bool phase_pass = gpu_bench::cg_phases_requested();
    const int left = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int right = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    // One allocation for the largest side in the sweep; smaller sides use a
    // prefix of it. side * (local_cols + 2) grows with side, so the largest
    // side needs the most room.
    const auto max_field_elems = max_side * (gpu_bench::local_count(max_side, rank, ranks) + 2U);

    float* p_field = nullptr;
    float* q_field = nullptr;
    float* send_west = nullptr;
    float* send_east = nullptr;
    float* recv_west = nullptr;
    float* recv_east = nullptr;
    double* partial_pq = nullptr;
    double* partial_qq = nullptr;
    double* result_pq = nullptr;
    double* result_qq = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p_field), max_field_elems * sizeof(float)), "cudaMalloc(p)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&q_field), max_field_elems * sizeof(float)), "cudaMalloc(q)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&send_west), max_side * sizeof(float)), "cudaMalloc(send_west)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&send_east), max_side * sizeof(float)), "cudaMalloc(send_east)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&recv_west), max_side * sizeof(float)), "cudaMalloc(recv_west)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&recv_east), max_side * sizeof(float)), "cudaMalloc(recv_east)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&partial_pq), sizeof(double)), "cudaMalloc(partial_pq)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&partial_qq), sizeof(double)), "cudaMalloc(partial_qq)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&result_pq), sizeof(double)), "cudaMalloc(result_pq)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&result_qq), sizeof(double)), "cudaMalloc(result_qq)");

    int all_sides_ok = 1;
    for (const std::size_t side : sides) {
      const auto local_cols = gpu_bench::local_count(side, rank, ranks);
      const auto col_offset = gpu_bench::local_offset(side, rank, ranks);
      const auto width = local_cols + 2U;
      const auto field_elems = side * width;
      const int column_count = static_cast<int>(side);

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

      /* The step, split into the four phases the analysis decomposes it into.
       * Composing them is exactly the step this benchmark has always timed:
       * the pack and compute phases already had to synchronize before MPI could
       * read their output, so the split adds no synchronization to the
       * headline loop - only the phase pass adds any. */
      const auto pack = [&]() {
        if (local_cols > 0) {
          pack_column_kernel<<<grid1d, block1d>>>(p_field, send_west, side, width, 1U);
          pack_column_kernel<<<grid1d, block1d>>>(p_field, send_east, side, width, local_cols);
          check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pack)");
        }
      };
      const auto halo = [&]() {
        MPI_Sendrecv(send_west, column_count, MPI_FLOAT, left, 0, recv_east, column_count, MPI_FLOAT, right, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_east, column_count, MPI_FLOAT, right, 1, recv_west, column_count, MPI_FLOAT, left, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      };
      const auto compute = [&]() {
        check_cuda(cudaMemset(partial_pq, 0, sizeof(double)), "cudaMemset(partial_pq)");
        check_cuda(cudaMemset(partial_qq, 0, sizeof(double)), "cudaMemset(partial_qq)");
        if (local_cols > 0) {
          unpack_column_kernel<<<grid1d, block1d>>>(p_field, recv_west, side, width, 0U);
          unpack_column_kernel<<<grid1d, block1d>>>(p_field, recv_east, side, width, local_cols + 1U);
          spmv_kernel<<<grid2d, block2d>>>(p_field, q_field, side, local_cols, width);
          cg_dot_kernel<<<dot_grid, block1d>>>(p_field, q_field, partial_pq, partial_qq, side, local_cols, width);
          check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(spmv+dot)");
        }
      };
      const auto reduce = [&]() {
        MPI_Allreduce(partial_pq, result_pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(partial_qq, result_qq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      };

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        pack();
        halo();
        compute();
        reduce();
      });
      const auto global = gpu_bench::collective_stats(stats);

      gpu_bench::cg_phase_stats phase_global;
      if (phase_pass) {
        MPI_Barrier(MPI_COMM_WORLD);
        const auto phase_samples = gpu_bench::measure_cg_phases(
            warmup, iterations, [&]() { check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(phase)"); },
            pack, halo, compute, reduce);
        for (int phase = 0; phase < gpu_bench::cg_phase_count; ++phase) {
          phase_global[phase] = gpu_bench::collective_stats(gpu_bench::summarize(phase_samples[phase]));
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
      double host_pq = 0.0;
      double host_qq = 0.0;
      check_cuda(cudaMemcpy(&host_pq, result_pq, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(pq)");
      check_cuda(cudaMemcpy(&host_qq, result_qq, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(qq)");
      int local_ok = gpu_bench::nearly_equal(host_pq, ref_pq) && gpu_bench::nearly_equal(host_qq, ref_qq)
                         ? 1
                         : 0;
      if (local_cols > 0) {
        std::vector<float> host_q(field_elems);
        check_cuda(cudaMemcpy(host_q.data(), q_field, field_elems * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(q)");
        if (!gpu_bench::validate_columns(host_q.data(), side, local_cols, width, col_offset, qval)) {
          local_ok = 0;
        }
      }
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_sides_ok = all_sides_ok && global_ok;

      if (rank == 0) {
        gpu_bench::bench_report report;
        report.name = "cuda_mpi_cg_step";
        report.n = side;
        report.ranks = ranks;
        report.bytes_per_iter = 2U * side * sizeof(float);  // halo volume (the two reductions are scalars)
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        if (phase_pass) {
          report.extra = gpu_bench::cg_phase_fields(phase_global);
        }
        report.valid = global_ok != 0;
        gpu_bench::print_report(report);
      }
    }

    check_cuda(cudaFree(p_field), "cudaFree(p)");
    check_cuda(cudaFree(q_field), "cudaFree(q)");
    check_cuda(cudaFree(send_west), "cudaFree(send_west)");
    check_cuda(cudaFree(send_east), "cudaFree(send_east)");
    check_cuda(cudaFree(recv_west), "cudaFree(recv_west)");
    check_cuda(cudaFree(recv_east), "cudaFree(recv_east)");
    check_cuda(cudaFree(partial_pq), "cudaFree(partial_pq)");
    check_cuda(cudaFree(partial_qq), "cudaFree(partial_qq)");
    check_cuda(cudaFree(result_pq), "cudaFree(result_pq)");
    check_cuda(cudaFree(result_qq), "cudaFree(result_qq)");

    MPI_Finalize();
    return all_sides_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
