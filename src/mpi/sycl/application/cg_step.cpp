#include <mpi.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cli.hpp"
#include "stats/collective_mpi.hpp"
#include "partition.hpp"
#include "report.hpp"
#include "benchmarks/cg_phases.hpp"
#include "benchmarks/cg_step.hpp"
#include "timing.hpp"
#include "validation.hpp"

// CG iteration communication skeleton (see src/mpi/cuda/application/cg_step.cu): SpMV
// (column-slab halo exchange of p + 5-point stencil -> q) then two global
// reductions dot(p,q) and dot(q,q). p == 1 so references are exact.

namespace {

sycl::device device_for_rank(int rank) {
  const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (devices.empty()) {
    throw std::runtime_error("no SYCL GPU devices available");
  }
  return devices[static_cast<std::size_t>(rank) % devices.size()];
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

    sycl::queue queue(device_for_rank(rank), sycl::property::queue::in_order());

    // One allocation for the largest side in the sweep; smaller sides use a
    // prefix of it. side * (local_cols + 2) grows with side, so the largest
    // side needs the most room.
    const auto max_field_elems = max_side * (gpu_bench::local_count(max_side, rank, ranks) + 2U);

    float* p_field = sycl::malloc_device<float>(max_field_elems, queue);
    float* q_field = sycl::malloc_device<float>(max_field_elems, queue);
    float* send_west = sycl::malloc_device<float>(max_side, queue);
    float* send_east = sycl::malloc_device<float>(max_side, queue);
    float* recv_west = sycl::malloc_device<float>(max_side, queue);
    float* recv_east = sycl::malloc_device<float>(max_side, queue);
    double* partial_pq = sycl::malloc_device<double>(1, queue);
    double* partial_qq = sycl::malloc_device<double>(1, queue);
    double* result_pq = sycl::malloc_device<double>(1, queue);
    double* result_qq = sycl::malloc_device<double>(1, queue);
    if (p_field == nullptr || q_field == nullptr || send_west == nullptr || send_east == nullptr ||
        recv_west == nullptr || recv_east == nullptr || partial_pq == nullptr || partial_qq == nullptr ||
        result_pq == nullptr || result_qq == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }

    const auto sync = [&]() { queue.wait(); };

    int all_sides_ok = 1;
    for (const std::size_t side : sides) {
      const auto local_cols = gpu_bench::local_count(side, rank, ranks);
      const auto col_offset = gpu_bench::local_offset(side, rank, ranks);
      const auto width = local_cols + 2U;
      const auto field_elems = side * width;
      const int column_count = static_cast<int>(side);

      queue.memset(p_field, 0, field_elems * sizeof(float)).wait();
      queue.memset(q_field, 0, field_elems * sizeof(float)).wait();
      queue.memset(recv_west, 0, side * sizeof(float)).wait();
      queue.memset(recv_east, 0, side * sizeof(float)).wait();
      if (local_cols > 0) {
        queue.parallel_for(sycl::range<2>{side, local_cols}, [=](sycl::id<2> id) {
          p_field[id[0] * width + (id[1] + 1U)] = 1.0F;
        }).wait();
      }

      /* The step, split into the four phases the analysis decomposes it into.
       * The queue is in-order and every submission is waited on, so composing
       * them is exactly the step this benchmark has always timed; only the
       * phase pass adds synchronization. */
      const auto pack = [&]() {
        if (local_cols > 0) {
          const auto last = local_cols;
          queue.parallel_for(sycl::range<1>{side}, [=](sycl::id<1> id) {
            const auto i = id[0];
            send_west[i] = p_field[i * width + 1U];
            send_east[i] = p_field[i * width + last];
          });
        }
        queue.wait();
      };
      const auto halo = [&]() {
        MPI_Sendrecv(send_west, column_count, MPI_FLOAT, left, 0, recv_east, column_count, MPI_FLOAT, right, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_east, column_count, MPI_FLOAT, right, 1, recv_west, column_count, MPI_FLOAT, left, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      };
      /* One wait, at the end, not one per submission.
       *
       * The queue is in-order, so every submission below already runs after the
       * one before it; the only ordering this phase has to establish is against
       * the host, which reads partial_pq/partial_qq in the reduction that follows.
       * Waiting on each submission separately cost about 68 us per iteration at
       * 4n4g -- 90% of this backend's gap over cuda_mpi, which runs the same MPI
       * and issues its kernels asynchronously with a single synchronization. That
       * gap was an artifact of this loop, not of SYCL. */
      const auto compute = [&]() {
        queue.memset(partial_pq, 0, sizeof(double));
        queue.memset(partial_qq, 0, sizeof(double));
        if (local_cols > 0) {
          const auto east_ghost = local_cols + 1U;
          queue.parallel_for(sycl::range<1>{side}, [=](sycl::id<1> id) {
            const auto i = id[0];
            p_field[i * width + 0U] = recv_west[i];
            p_field[i * width + east_ghost] = recv_east[i];
          });
          queue.parallel_for(sycl::range<2>{side, local_cols}, [=](sycl::id<2> id) {
            const auto i = id[0];
            const auto j = id[1] + 1U;
            const float north = i > 0 ? p_field[(i - 1U) * width + j] : 0.0F;
            const float south = i + 1U < side ? p_field[(i + 1U) * width + j] : 0.0F;
            const float west = p_field[i * width + (j - 1U)];
            const float east = p_field[i * width + (j + 1U)];
            q_field[i * width + j] = 0.25F * (north + south + west + east);
          });
          queue.submit([&](sycl::handler& handler) {
            auto sum_pq = sycl::reduction(partial_pq, sycl::plus<double>());
            auto sum_qq = sycl::reduction(partial_qq, sycl::plus<double>());
            handler.parallel_for(sycl::range<2>{side, local_cols}, sum_pq, sum_qq,
                                 [=](sycl::id<2> id, auto& acc_pq, auto& acc_qq) {
                                   const auto i = id[0];
                                   const auto j = id[1] + 1U;
                                   const double pv = p_field[i * width + j];
                                   const double qv = q_field[i * width + j];
                                   acc_pq += pv * qv;
                                   acc_qq += qv * qv;
                                 });
          });
        }
        queue.wait();
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
        const auto phase_samples =
            gpu_bench::measure_cg_phases(warmup, iterations, sync, pack, halo, compute, reduce);
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
      queue.copy(result_pq, &host_pq, 1).wait();
      queue.copy(result_qq, &host_qq, 1).wait();
      int local_ok = gpu_bench::nearly_equal(host_pq, ref_pq) && gpu_bench::nearly_equal(host_qq, ref_qq)
                         ? 1
                         : 0;
      if (local_cols > 0) {
        std::vector<float> host_q(field_elems);
        queue.copy(q_field, host_q.data(), field_elems).wait();
        if (!gpu_bench::validate_columns(host_q.data(), side, local_cols, width, col_offset, qval)) {
          local_ok = 0;
        }
      }
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_sides_ok = all_sides_ok && global_ok;

      if (rank == 0) {
        gpu_bench::bench_report report;
        report.name = "sycl_mpi_cg_step";
        report.n = side;
        report.ranks = ranks;
        report.bytes_per_iter = 2U * side * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = "device=\"" + queue.get_device().get_info<sycl::info::device::name>() + "\"";
        if (phase_pass) {
          report.extra += ' ' + gpu_bench::cg_phase_fields(phase_global);
        }
        gpu_bench::print_report(report);
      }
    }

    sycl::free(p_field, queue);
    sycl::free(q_field, queue);
    sycl::free(send_west, queue);
    sycl::free(send_east, queue);
    sycl::free(recv_west, queue);
    sycl::free(recv_east, queue);
    sycl::free(partial_pq, queue);
    sycl::free(partial_qq, queue);
    sycl::free(result_pq, queue);
    sycl::free(result_qq, queue);

    MPI_Finalize();
    return all_sides_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
