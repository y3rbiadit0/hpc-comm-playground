#!/usr/bin/env bash
set -euo pipefail

# Shared engine for all Leonardo experiments. GPU_BENCH_ is this project's variable
# namespace throughout: harness inputs, runtime settings and job metadata.
#
# Each experiment's common.sh is a thin shim that sources this file and provides:
#
#   GPU_BENCH_EXPERIMENT              results subdirectory name (e.g. allreduce)
#   GPU_BENCH_N_LABEL                 summary label for GPU_BENCH_N (e.g. "problem size")
#   gpu_bench_experiment_defaults     required; sets GPU_BENCH_N/GPU_BENCH_ITERS/GPU_BENCH_EXTRA_ARGS defaults
#   gpu_bench_experiment_extra_summary  optional; extra summary lines
#   gpu_bench_experiment_launch_prefix  optional override; echoes a per-rank command
#                                prefix spliced in front of the binary (e.g. nsys)
#   gpu_bench_<name>_main      wrapper around gpu_bench_experiment_main, called
#                              by cluster/harness/job.sh

# Opt-in per-rank profiling, shared by all experiments. GPU_BENCH_PROFILE=nsys wraps
# each rank in Nsight Systems (timeline: kernels, comm calls, overlap);
# GPU_BENCH_PROFILE=ncu wraps each rank in Nsight Compute collecting the roofline
# section for every kernel launch. One report per rank+trial lands under
# $GPU_BENCH_RUN_DIR/profiles. Profiling perturbs timing: use a dedicated
# GPU_BENCH_NTRIALS=1 run and never report numbers from a profiled run.
#
# The GPU rank wrapper normalizes launcher-specific rank variables into
# GPU_BENCH_GLOBAL_RANK. The profilers substitute %q{VAR} at runtime, so
# the token must stay unquoted.
#
# ncu knobs: GPU_BENCH_NCU_BIN (default: nvhpc/25.3's ncu -- the nvhpc/24.5 ncu
# 2024.1.1 in PATH fails to attach with "Failed to connect to process";
# NVIDIA fixed the launch path in later releases), GPU_BENCH_NCU_SET (default
# roofline), GPU_BENCH_NCU_LAUNCH_COUNT (default 3,
# total profiled launches -- GLOBAL across all kernels in launch order, not
# per kernel, counted after the GPU_BENCH_NCU_KERNELS filter; multi-kernel binaries
# need a filter or a larger count to reach later kernels), GPU_BENCH_NCU_KERNELS
# (regex, no spaces; limits which kernels are profiled). ncu replays each
# profiled launch several times to collect its metric set, so a
# kernel that waits on another rank (device-initiated comm) can hang under
# replay -- exclude such kernels via GPU_BENCH_NCU_KERNELS or profile a 1n1g run.
gpu_bench_experiment_launch_prefix() {
  [[ -n "${GPU_BENCH_PROFILE:-}" ]] || return 0

  local rank_token='%q{GPU_BENCH_GLOBAL_RANK}'

  mkdir -p "$GPU_BENCH_RUN_DIR/profiles"
  local out="$GPU_BENCH_RUN_DIR/profiles/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-manual}-${trial}-rank${rank_token}"

  case "$GPU_BENCH_PROFILE" in
    nsys)
      printf '%s ' nsys profile --force-overwrite=true --trace="${GPU_BENCH_NSYS_TRACE:-cuda,nvtx,mpi}" \
        --sample=none --cpuctxsw=none --output="$out"
      ;;
    ncu)
      # MPI_Init fails when ncu sits between srun and the binary (the rank
      # loses its PMI bootstrap); profiled ncu runs need GPU_BENCH_LAUNCHER=mpirun.
      printf '%s ' "${GPU_BENCH_NCU_BIN:?cluster.sh must define GPU_BENCH_NCU_BIN}" \
        --set "${GPU_BENCH_NCU_SET:-roofline}" \
        --launch-count "${GPU_BENCH_NCU_LAUNCH_COUNT:-3}" \
        ${GPU_BENCH_NCU_KERNELS:+--kernel-name regex:${GPU_BENCH_NCU_KERNELS}} \
        --force-overwrite --export "$out"
      ;;
    *)
      echo "unknown GPU_BENCH_PROFILE: $GPU_BENCH_PROFILE (expected nsys or ncu)" >&2
      return 1
      ;;
  esac
}

gpu_bench_experiment_setup() {
  : "${GPU_BENCH_EXPERIMENT:?missing GPU_BENCH_EXPERIMENT}"
  : "${GPU_BENCH_N_LABEL:?missing GPU_BENCH_N_LABEL}"
  : "${GPU_BENCH_STACK:?missing GPU_BENCH_STACK}"
  : "${GPU_BENCH_RUNTIME:?missing GPU_BENCH_RUNTIME}"
  : "${GPU_BENCH_BINARY:?missing GPU_BENCH_BINARY}"
  : "${GPU_BENCH_RESULT_NAME:?missing GPU_BENCH_RESULT_NAME}"
  : "${GPU_BENCH_NODES:?missing GPU_BENCH_NODES}"
  : "${GPU_BENCH_TASKS_PER_NODE:?missing GPU_BENCH_TASKS_PER_NODE}"

  mkdir -p ./logs

  export LC_ALL=C
  export GPU_BENCH_JOB_NODES="$GPU_BENCH_NODES"

  gpu_bench_cluster_environment "$GPU_BENCH_STACK"
  gpu_bench_cluster_runtime "$GPU_BENCH_RUNTIME"

  GPU_BENCH_NTRIALS=${GPU_BENCH_NTRIALS:-3}
  GPU_BENCH_LAUNCHER=${GPU_BENCH_LAUNCHER:-srun}
  GPU_BENCH_TRIAL_TIMEOUT=${GPU_BENCH_TRIAL_TIMEOUT:-2m}
  GPU_BENCH_RUN_DIR=${GPU_BENCH_RUN_DIR:-${GPU_BENCH_RESULTS_ROOT:-results}/$GPU_BENCH_RESULT_NAME/$GPU_BENCH_EXPERIMENT}

  gpu_bench_experiment_defaults

  case "${GPU_BENCH_PROFILE:-}" in
    '' | nsys | ncu) ;;
    *) echo "unknown GPU_BENCH_PROFILE: $GPU_BENCH_PROFILE (expected nsys or ncu)" >&2; exit 1 ;;
  esac

  [ -x "$GPU_BENCH_BINARY" ] || { echo "no executable: $GPU_BENCH_BINARY" >&2; exit 1; }
  command -v timeout >/dev/null 2>&1 || { echo "timeout command is unavailable" >&2; exit 1; }
}

gpu_bench_experiment_print_summary() {
  echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
  echo "nodes: ${SLURM_NODELIST:-manual}"
  echo "node: $(hostname)"
  echo "project_root: $GPU_BENCH_PROJECT_ROOT"
  echo "binary: $GPU_BENCH_BINARY"
  echo "result name: $GPU_BENCH_RESULT_NAME"
  echo "$GPU_BENCH_N_LABEL: $GPU_BENCH_N"
  [[ -n "${GPU_BENCH_ITERS:-}" ]] && echo "iterations: $GPU_BENCH_ITERS"
  [[ -n "${GPU_BENCH_WARMUP:-}" ]] && echo "warmup: $GPU_BENCH_WARMUP"
  [[ -n "${GPU_BENCH_EXTRA_ARGS:-}" ]] && echo "extra args: $GPU_BENCH_EXTRA_ARGS"
  echo "trials: $GPU_BENCH_NTRIALS"
  echo "nodes requested: $GPU_BENCH_NODES"
  echo "tasks per node: $GPU_BENCH_TASKS_PER_NODE"
  echo "launcher: $GPU_BENCH_LAUNCHER"
  echo "launcher path: $(command -v "$GPU_BENCH_LAUNCHER" 2>/dev/null || true)"
  echo "trial timeout: $GPU_BENCH_TRIAL_TIMEOUT"
  echo "stack: $GPU_BENCH_STACK"
  echo "runtime: $GPU_BENCH_RUNTIME"
  echo "profile: ${GPU_BENCH_PROFILE:-off}"
  if declare -F gpu_bench_experiment_extra_summary >/dev/null; then
    gpu_bench_experiment_extra_summary
  fi
  gpu_bench_print_env
  nvidia-smi || true
}

gpu_bench_experiment_run_trials() {
  for trial in $(seq "$GPU_BENCH_NTRIALS"); do
    local outfile="$GPU_BENCH_RUN_DIR/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-manual}-${trial}-stdout.txt"
    local errfile="$GPU_BENCH_RUN_DIR/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-manual}-${trial}-stderr.txt"

    mkdir -p "$(dirname "$outfile")" "$(dirname "$errfile")"

    echo "$GPU_BENCH_RESULT_NAME - Trial $trial of $GPU_BENCH_NTRIALS"
    echo "stdout: ${outfile}.tmp"
    echo "stderr: ${errfile}.tmp"

    local trial_status=0
    set +e
    if [[ "$GPU_BENCH_LAUNCHER" == "mpirun" ]]; then
      /usr/bin/time -p --verbose \
        timeout --signal=TERM --kill-after=30s "$GPU_BENCH_TRIAL_TIMEOUT" \
        mpirun -np "$((GPU_BENCH_NODES * GPU_BENCH_TASKS_PER_NODE))" \
        "$GPU_BENCH_PROJECT_ROOT/cluster/harness/utils/gpu-rank-wrapper.sh" \
        $(gpu_bench_experiment_launch_prefix) \
        "$GPU_BENCH_BINARY" \
        "$GPU_BENCH_N" \
        $GPU_BENCH_EXTRA_ARGS \
        >"${outfile}.tmp" 2>"${errfile}.tmp"
      trial_status=$?
    else
      /usr/bin/time -p --verbose \
        timeout --signal=TERM --kill-after=30s "$GPU_BENCH_TRIAL_TIMEOUT" \
        srun --cpu-freq=high \
        -N "$GPU_BENCH_NODES" \
        --ntasks-per-node="$GPU_BENCH_TASKS_PER_NODE" \
        "$GPU_BENCH_PROJECT_ROOT/cluster/harness/utils/gpu-rank-wrapper.sh" \
        $(gpu_bench_experiment_launch_prefix) \
        "$GPU_BENCH_BINARY" \
        "$GPU_BENCH_N" \
        $GPU_BENCH_EXTRA_ARGS \
        >"${outfile}.tmp" 2>"${errfile}.tmp"
      trial_status=$?
    fi
    set -e

    if [[ "$trial_status" -ne 0 ]]; then
      if [[ "$trial_status" -eq 124 || "$trial_status" -eq 137 ]]; then
        printf 'gpu-comm-bench trial exceeded timeout %s (status %s)\n' \
          "$GPU_BENCH_TRIAL_TIMEOUT" "$trial_status" | tee -a "${errfile}.tmp" >&2
      else
        printf 'gpu-comm-bench trial failed with status %s\n' "$trial_status" \
          | tee -a "${errfile}.tmp" >&2
      fi
      echo "failed stdout retained at ${outfile}.tmp" >&2
      echo "failed stderr retained at ${errfile}.tmp" >&2
      return "$trial_status"
    fi

    mv --verbose "${outfile}.tmp" "$outfile"
    mv --verbose "${errfile}.tmp" "$errfile"

    # Back-to-back job steps can stall on Leonardo while the previous step
    # finalizes; give it a moment before launching the next srun.
    if [[ "$trial" -lt "$GPU_BENCH_NTRIALS" ]]; then
      sleep 1
    fi
  done
}

gpu_bench_experiment_main() {
  gpu_bench_experiment_setup
  gpu_bench_experiment_print_summary
  gpu_bench_experiment_run_trials
}
