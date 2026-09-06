#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stats/summary.hpp"
#include "timing.hpp"

namespace gpu_bench {

/* Per-phase timing for the CG-step benchmarks.
 *
 * `cg_step` reports one number per iteration, which cannot be decomposed from
 * outside: a model that predicts it from the halo_1d and allreduce
 * microbenchmarks has to assume how much of the step each accounts for, and
 * those assumptions are what the model is supposed to be testing. Timing the
 * phases directly turns the decomposition from an inference into a measurement.
 *
 * It is a SECOND, OPT-IN PASS rather than instrumentation of the headline loop.
 * Separating the phases means synchronizing the device at every boundary, which
 * both adds cost and removes overlap the unsplit step is free to exploit. Doing
 * that inside the reported loop would change the headline number and break
 * comparability with every result measured before this existed. So the reported
 * `usec` still comes from the unsplit loop, and the phase figures are published
 * beside it as their own, explicitly more-synchronized measurement.
 *
 * Read the phase figures as an upper bound on each phase and a lower bound on
 * the overlap: their sum exceeds the unsplit step by however much the phases
 * overlapped, and that difference is itself the useful quantity. */

enum cg_phase {
  cg_phase_pack,     // gather the boundary columns into contiguous send buffers
  cg_phase_halo,     // exchange them with the west and east neighbours
  cg_phase_compute,  // unpack, stencil, and the two local dot products
  cg_phase_reduce,   // the two global sums
  cg_phase_count
};

inline const char* cg_phase_name(int phase) {
  switch (phase) {
    case cg_phase_pack:
      return "pack";
    case cg_phase_halo:
      return "halo";
    case cg_phase_compute:
      return "compute";
    case cg_phase_reduce:
      return "reduce";
    default:
      throw std::invalid_argument("unknown CG phase");
  }
}

// GPU_BENCH_CG_PHASES=1 runs the extra pass. Off by default: it roughly doubles
// the benchmark's run time and is only wanted when decomposing a result.
inline bool cg_phases_requested() {
  const char* setting = std::getenv("GPU_BENCH_CG_PHASES");
  return setting != nullptr && *setting != '\0' && std::string(setting) != "0";
}

// One local timing series per phase, in `cg_phase` order.
using cg_phase_samples = std::array<std::vector<double>, cg_phase_count>;
// The same series after being reduced across ranks.
using cg_phase_stats = std::array<bench_stats, cg_phase_count>;

/* Times each phase of `iterations` complete steps.
 *
 * The four phase callables must together perform exactly the step the headline
 * loop performs, in this order, so that the two passes measure the same work.
 * `sync` must make the calling rank's outstanding device work observable; it
 * runs inside the timed region of every phase, which is what makes a phase
 * boundary meaningful and what makes this pass more synchronized than the
 * headline loop. */
template <typename Sync, typename Pack, typename Halo, typename Compute, typename Reduce>
inline cg_phase_samples measure_cg_phases(int warmup, int iterations, Sync&& sync, Pack&& pack,
                                          Halo&& halo, Compute&& compute, Reduce&& reduce) {
  if (warmup < 0 || iterations <= 0) {
    throw std::invalid_argument("invalid CG phase iteration count");
  }

  for (int i = 0; i < warmup; ++i) {
    pack();
    halo();
    compute();
    reduce();
  }

  cg_phase_samples samples;
  for (auto& series : samples) {
    series.reserve(static_cast<std::size_t>(iterations));
  }

  const auto record = [&sync](auto&& phase, std::vector<double>& series) {
    wall_timer timer;
    phase();
    sync();
    series.push_back(timer.seconds());
  };

  for (int i = 0; i < iterations; ++i) {
    record(pack, samples[cg_phase_pack]);
    record(halo, samples[cg_phase_halo]);
    record(compute, samples[cg_phase_compute]);
    record(reduce, samples[cg_phase_reduce]);
  }
  return samples;
}

/* The `extra` report fields carrying the breakdown.
 *
 * Emitted only when the pass ran, so records without them are results measured
 * without phase timing rather than results with a zero phase. benchscribe keeps
 * unrecognized key=value pairs, so adding these does not change the schema. */
inline std::string cg_phase_fields(const cg_phase_stats& phases) {
  std::ostringstream fields;
  double total_s = 0.0;
  for (int phase = 0; phase < cg_phase_count; ++phase) {
    fields << "phase_" << cg_phase_name(phase) << "_usec=" << (phases[phase].avg_s * 1.0e6) << ' ';
    total_s += phases[phase].avg_s;
  }
  // The sum is what the step would cost with no overlap between phases. Report
  // it so a reader comparing against the headline `usec` is comparing against
  // the right quantity rather than adding the four fields themselves.
  fields << "phase_sum_usec=" << (total_s * 1.0e6);
  return fields.str();
}

}  // namespace gpu_bench
