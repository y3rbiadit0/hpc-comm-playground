#!/usr/bin/env bash
# Launch benchmark jobs through the selected cluster integration.
#
# One cell:
#   cluster/harness/launch.sh halo_1d cuda_mpi 1n2g
#   cluster/harness/launch.sh cg_step cuda_mpi 2n4g --qos=boost_qos_dbg   # extra sbatch args
#
# Many cells:
#   cluster/harness/launch.sh --all                        every benchmark, backend, topology
#   cluster/harness/launch.sh --all halo_1d moe            only these benchmarks
#
# Inspect without submitting:
#   cluster/harness/launch.sh --dry-run --all              what would be submitted
#   cluster/harness/launch.sh --explain halo_1d cuda_mpi 1n2g
#                                                how every value is resolved, and
#                                                which file writes each env var
#
# Filters (env, space-separated globs; --all only):
#   GPU_BENCH_ONLY_BACKENDS="cuda_mpi cuda_nccl"
#   GPU_BENCH_ONLY_TOPOS="1n4g 2n1g"
#   GPU_BENCH_REPEATS=5            submit each cell as N independent jobs
#
# Repetition that matters is at the job level, not inside one. Trials within a
# job share an allocation, so they measure the same nodes and GPUs; the spread
# between allocations is several times larger. Each repeat is a separate job, and
# results key on $SLURM_JOB_ID so they accumulate side by side.
#
# Pass-through overrides (GPU_BENCH_N, GPU_BENCH_ITERS, GPU_BENCH_WARMUP,
# GPU_BENCH_NTRIALS, GPU_BENCH_MSG_SIZES, ...) are exported here and inherited by
# every job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$ROOT/cluster/harness"
EXP="$HARNESS/experiments"

# Which machine we are on. The harness is machine-independent; everything
# specific arrives through cluster/<name>/cluster.sh.
GPU_BENCH_CLUSTER=${GPU_BENCH_CLUSTER:-leonardo}
CLUSTER_DIR="$ROOT/cluster/$GPU_BENCH_CLUSTER"
[[ -f "$CLUSTER_DIR/cluster.sh" ]] || {
  echo "error: no such cluster: $GPU_BENCH_CLUSTER" >&2
  echo "known: $(cd "$ROOT/cluster" && ls -d */cluster.sh 2>/dev/null | cut -d/ -f1 | tr '\n' ' ')" >&2
  exit 2
}
export GPU_BENCH_CLUSTER
export GPU_BENCH_PROJECT_ROOT="$ROOT"

source "$CLUSTER_DIR/cluster.sh"
source "$HARNESS/matrix.sh"
source "$HARNESS/utils/where-set.sh"

mode=single
dry_run=${GPU_BENCH_DRYRUN:-0}
explain=0
args=()
for arg in "$@"; do
  case "$arg" in
    --all)     mode=all ;;
    --dry-run) dry_run=1 ;;
    --explain) explain=1; dry_run=1 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         args+=("$arg") ;;
  esac
done

require_sbatch() {
  if [[ "$dry_run" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch not found -- run this on Leonardo (a login node)." >&2
    exit 1
  fi
}

# submit <benchmark> <backend> <topology> [extra sbatch args...]
submit() {
  local bench="$1" backend="$2" topo="$3"; shift 3
  gpu_bench_matrix_validate_cell "$bench" "$backend" "$topo"
  gpu_bench_backend_fields "$backend"
  gpu_bench_topology_fields "$topo"

  if [[ "$explain" == "1" ]]; then
    explain_cell "$bench" "$backend" "$topo"
    return 0
  fi
  if [[ "$dry_run" == "1" ]]; then
    echo "would submit: $bench/$backend/$topo"
    return 0
  fi

  # The allocation shape is passed here, not baked into a script: sbatch
  # command-line options override #SBATCH directives, so one job.sh serves every
  # cell. The three GPU_BENCH_* variables reach the job through the environment,
  # which sbatch propagates by default.
  GPU_BENCH_BENCHMARK="$bench" \
  GPU_BENCH_BACKEND="$backend" \
  GPU_BENCH_TOPOLOGY="$topo" \
  GPU_BENCH_CLUSTER="$GPU_BENCH_CLUSTER" \
  sbatch \
    --job-name="${bench}_${backend}_${topo}" \
    --nodes="$GPU_BENCH_NODES" \
    --ntasks-per-node="$GPU_BENCH_TASKS_PER_NODE" \
    --gres=gpu:"$GPU_BENCH_TASKS_PER_NODE" \
    --time="$(gpu_bench_walltime_for "$GPU_BENCH_NODES")" \
    "$@" \
    "$HARNESS/job.sh"
}

# Print how one cell resolves: which files decide it, and for every environment
# variable, which file sets it and which definition wins.
#
# Read statically from the files rather than by running the chain. The layering
# is fixed -- benchmark shim, then environment, then runtime, every layer using
# ${VAR:-default} -- so the first definition in that order wins, and that can be
# determined without a cluster, without modules, and without $SCRATCH.
explain_cell() {
  local bench="$1" backend="$2" topo="$3"
  # Kept in step with job.sh, which computes these two paths for real. A
  # non-default library version suffixes both; see cluster/<cluster>/layout.sh.
  local tag="${GPU_BENCH_VARIANT_TAG:-}"
  cat <<EOF
cell        : $bench / $backend / $topo
run entry   : cluster/harness/launch.sh
job script  : cluster/harness/job.sh
matrix      : cluster/harness/matrix.sh
backend row : cluster/$GPU_BENCH_CLUSTER/backends.sh
  stack     : $GPU_BENCH_STACK        (cluster/$GPU_BENCH_CLUSTER/env/$GPU_BENCH_STACK.sh)
  runtime   : cluster/$GPU_BENCH_CLUSTER/runtime/$GPU_BENCH_RUNTIME.sh
  launcher  : $GPU_BENCH_LAUNCHER
  preset    : $GPU_BENCH_PRESET
  binary    : build${tag}/$GPU_BENCH_PRESET/$GPU_BENCH_BINDIR/${GPU_BENCH_BINARY_PREFIX}_${bench}
benchmark   : cluster/harness/experiments/$bench/common.sh
topology    : $GPU_BENCH_NODES node(s) x $GPU_BENCH_TASKS_PER_NODE GPU(s) = $((GPU_BENCH_NODES * GPU_BENCH_TASKS_PER_NODE)) ranks
sbatch      : --nodes=$GPU_BENCH_NODES --ntasks-per-node=$GPU_BENCH_TASKS_PER_NODE --gres=gpu:$GPU_BENCH_TASKS_PER_NODE --time=$(gpu_bench_walltime_for "$GPU_BENCH_NODES")
results     : ${GPU_BENCH_RESULTS_ROOT:-results$tag}/${bench//_/-}-${backend//_/-}-$topo/$bench
account     : ${SBATCH_ACCOUNT:-Slurm default}   partition: $SBATCH_PARTITION
libraries   : NVSHMEM $GPU_BENCH_NVSHMEM_VERSION${tag:+   prefix $NVSHMEM_HOME}

environment (definitions in execution order; the first one wins):
EOF
  gpu_bench_where_set "$bench" "$GPU_BENCH_STACK" "$GPU_BENCH_RUNTIME"
}

if [[ "$mode" == "single" ]]; then
  if [[ ${#args[@]} -lt 3 ]]; then
    # A bare benchmark name is the common slip: it reads like "run allreduce",
    # but one cell needs a backend and a topology too. Name the likely intent
    # rather than only restating the grammar.
    if [[ ${#args[@]} -eq 1 && -f "$EXP/${args[0]}/common.sh" ]]; then
      echo "error: '${args[0]}' is a benchmark, not a cell." >&2
      echo "  every backend and topology:  cluster/harness/launch.sh --all ${args[0]}" >&2
      echo "  one cell:                    cluster/harness/launch.sh ${args[0]} <backend> <topology>" >&2
      echo "  backends:   $(gpu_bench_backend_names)" >&2
      echo "  topologies: $(gpu_bench_matrix_topologies "${args[0]}")" >&2
      exit 2
    fi
    echo "usage: cluster/harness/launch.sh <benchmark> <backend> <topology> [sbatch args...]" >&2
    echo "       cluster/harness/launch.sh --all [benchmark...]" >&2
    echo "       cluster/harness/launch.sh --explain <benchmark> <backend> <topology>" >&2
    exit 2
  fi
  [[ -f "$EXP/${args[0]}/common.sh" ]] \
    || { echo "error: no such benchmark: ${args[0]}" >&2; exit 2; }
  require_sbatch
  submit "${args[@]}"
  exit 0
fi

benchmarks=(${args[@]+"${args[@]}"})
if [[ ${#benchmarks[@]} -eq 0 ]]; then
  read -r -a benchmarks <<<"$GPU_BENCH_ALL_BENCHMARKS"
fi

matches() {  # matches <value> <space-separated-globs-or-empty>
  local value="$1" filters="$2" f
  [[ -z "$filters" ]] && return 0
  for f in $filters; do [[ "$value" == $f ]] && return 0; done
  return 1
}

contains() {  # contains <value> <space-separated-values>
  case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

require_sbatch

repeats=${GPU_BENCH_REPEATS:-1}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_BENCH_REPEATS must be a positive integer" >&2; exit 2; }

submitted=0
skipped=0
for bench in "${benchmarks[@]}"; do
  if [[ ! -f "$EXP/$bench/common.sh" ]]; then
    echo "warning: no such benchmark: $bench" >&2
    continue
  fi
  for backend in $(gpu_bench_matrix_backends "$bench"); do
    contains "$backend" "$(gpu_bench_backend_names)" || { skipped=$((skipped+1)); continue; }
    matches "$backend" "${GPU_BENCH_ONLY_BACKENDS:-}" || { skipped=$((skipped+1)); continue; }
    for topo in $(gpu_bench_matrix_topologies "$bench"); do
      matches "$topo" "${GPU_BENCH_ONLY_TOPOS:-}" || { skipped=$((skipped+1)); continue; }
      for repeat in $(seq "$repeats"); do
        if [[ "$dry_run" != "1" ]]; then
          label="$bench/$backend/$topo"
          [[ "$repeats" -gt 1 ]] && label="$label (repeat $repeat/$repeats)"
          echo "submitting: $label"
        fi
        submit "$bench" "$backend" "$topo"
        submitted=$((submitted+1))
      done
    done
  done
done

echo "---"
echo "submitted: $submitted   skipped (filtered): $skipped"
echo "watch queue: squeue -u \$USER"
echo "when drained, parse: python3 tools/benchscribe > RESULTS.md"
