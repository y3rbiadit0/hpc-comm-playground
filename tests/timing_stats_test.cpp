#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "benchmarks/cg_phases.hpp"
#include "report.hpp"
#include "stats/collective.hpp"
#include "stats/summary.hpp"
#include "timing.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double lhs, double rhs, double tolerance = 1.0e-12) {
  return std::abs(lhs - rhs) <= tolerance;
}

void test_average_of_iteration_maxima() {
  const auto local = gpu_bench::summarize({10.0, 1.0, 8.0});
  const std::vector<double> peer = {1.0, 11.0, 2.0};

  const auto global = gpu_bench::collective_stats(
      local, [&](const double* input, double* output, std::size_t count) {
        require(count == peer.size(), "collective received the wrong sample count");
        for (std::size_t i = 0; i < count; ++i) {
          output[i] = std::max(input[i], peer[i]);
        }
      });

  const std::vector<double> expected = {10.0, 11.0, 8.0};
  require(global.samples == expected, "collective did not preserve iteration-wise maxima");
  require(near(global.total_s, 29.0), "global total is not based on iteration-wise maxima");
  require(near(global.avg_s, 29.0 / 3.0), "headline is not AVG(MAX per iteration)");
  require(near(global.min_s, 8.0), "minimum is not from the global series");
  require(near(global.max_s, 11.0), "maximum is not from the global series");
  require(near(global.median_s, 10.0), "median is not from the global series");
  require(near(global.p25_s, 8.0), "p25 is not from the global series");
  require(near(global.p75_s, 11.0), "p75 is not from the global series");
  require(near(global.stddev_s, std::sqrt(7.0 / 3.0)),
          "standard deviation is not from the global series");

  const double max_of_rank_averages = std::max(local.avg_s, gpu_bench::summarize(peer).avg_s);
  require(!near(global.avg_s, max_of_rank_averages),
          "test data does not distinguish AVG(MAX) from MAX(AVG)");
}

void test_wall_timing_samples_completed_bodies() {
  int calls = 0;
  const auto stats = gpu_bench::run_benchmark(2, 3, [&]() {
    ++calls;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  });

  require(calls == 5, "benchmark did not execute every warmup and timed body");
  require(stats.iterations == 3, "benchmark reported the wrong timed iteration count");
  require(stats.samples.size() == 3U, "benchmark did not retain one sample per timed body");
  require(std::all_of(stats.samples.begin(), stats.samples.end(),
                      [](double sample) { return sample > 0.0; }),
          "benchmark recorded a non-positive elapsed time");
  require(near(stats.total_s, stats.samples[0] + stats.samples[1] + stats.samples[2]),
          "benchmark total does not match its raw samples");
}

void test_batched_timing_amortizes_completed_operations() {
  int before_calls = 0;
  int after_calls = 0;
  int completed_operations = 0;
  const auto stats = gpu_bench::run_batched_benchmark(
      2, 3, 4, [&]() { ++before_calls; }, [&](int count) {
        completed_operations += count;
        std::this_thread::sleep_for(std::chrono::milliseconds(count));
      }, [&]() { ++after_calls; });

  require(before_calls == 5, "batch setup did not run before warmup and timed batches");
  require(after_calls == 4, "batch validation did not run after every timed batch");
  require(completed_operations == 14, "batched benchmark executed the wrong operation count");
  require(stats.iterations == 4, "batched benchmark reported the wrong sample count");
  require(stats.samples.size() == 4U, "batched benchmark did not retain every batch sample");
  require(std::all_of(stats.samples.begin(), stats.samples.end(),
                      [](double sample) { return sample > 0.0; }),
          "batched benchmark recorded a non-positive amortized time");
}

void test_isolated_case_gets_its_own_sample_count() {
  require(gpu_bench::batch_samples_for(1, 10, 100) == 100,
          "an isolated batch did not take the isolated sample count");
  require(gpu_bench::batch_samples_for(100, 10, 100) == 10,
          "a steady batch did not take the steady sample count");
  require(gpu_bench::batch_samples_for(2, 10, 100) == 10,
          "a two-operation batch is not a steady batch");

  bool rejected = false;
  try {
    gpu_bench::batch_samples_for(1, 0, 100);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "batch_samples_for accepted a non-positive sample count");
}

void test_report_contains_build_provenance() {
  gpu_bench::bench_report report;
  report.name = "cuda_mpi_test";
  report.valid = true;

  std::ostringstream output;
  auto* original_buffer = std::cout.rdbuf(output.rdbuf());
  gpu_bench::print_report(report);
  std::cout.rdbuf(original_buffer);

  require(output.str().find(std::string("suite_version=") + gpu_bench::suite_version) !=
              std::string::npos,
          "report omitted the suite version");
  require(output.str().find(std::string("source_revision=") + gpu_bench::source_revision) !=
              std::string::npos,
          "report omitted the source revision");
}

void test_cg_phases_run_every_phase_in_order() {
  std::vector<std::string> order;
  int syncs = 0;
  const auto samples = gpu_bench::measure_cg_phases(
      1, 2, [&]() { ++syncs; }, [&]() { order.push_back("pack"); },
      [&]() { order.push_back("halo"); }, [&]() { order.push_back("compute"); },
      [&]() { order.push_back("reduce"); });

  // One warmup step plus two timed ones, each running all four phases in order.
  const std::vector<std::string> expected = {"pack", "halo", "compute", "reduce",
                                             "pack", "halo", "compute", "reduce",
                                             "pack", "halo", "compute", "reduce"};
  require(order == expected, "phases did not run once each, in order, per step");
  // Warmup is untimed, so it must not synchronize: 2 steps x 4 phases.
  require(syncs == 8, "phase timing synchronized outside the timed phases");
  for (int phase = 0; phase < gpu_bench::cg_phase_count; ++phase) {
    require(samples[phase].size() == 2, "a phase recorded the wrong sample count");
  }

  bool rejected = false;
  try {
    gpu_bench::measure_cg_phases(
        0, 0, []() {}, []() {}, []() {}, []() {}, []() {});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "measure_cg_phases accepted a non-positive iteration count");
}

void test_cg_phase_fields_report_each_phase_and_their_sum() {
  gpu_bench::cg_phase_stats phases;
  phases[gpu_bench::cg_phase_pack] = gpu_bench::summarize({1.0e-6});
  phases[gpu_bench::cg_phase_halo] = gpu_bench::summarize({2.0e-6});
  phases[gpu_bench::cg_phase_compute] = gpu_bench::summarize({3.0e-6});
  phases[gpu_bench::cg_phase_reduce] = gpu_bench::summarize({4.0e-6});

  const auto fields = gpu_bench::cg_phase_fields(phases);
  require(fields.find("phase_pack_usec=1") != std::string::npos, "pack phase missing");
  require(fields.find("phase_halo_usec=2") != std::string::npos, "halo phase missing");
  require(fields.find("phase_compute_usec=3") != std::string::npos, "compute phase missing");
  require(fields.find("phase_reduce_usec=4") != std::string::npos, "reduce phase missing");
  // The sum is reported so a reader compares against the headline without
  // adding the four fields by hand.
  require(fields.find("phase_sum_usec=10") != std::string::npos, "phase sum missing or wrong");
}

}  // namespace

int main() {
  test_average_of_iteration_maxima();
  test_wall_timing_samples_completed_bodies();
  test_batched_timing_amortizes_completed_operations();
  test_isolated_case_gets_its_own_sample_count();
  test_report_contains_build_provenance();
  test_cg_phases_run_every_phase_in_order();
  test_cg_phase_fields_report_each_phase_and_their_sum();
  return 0;
}
