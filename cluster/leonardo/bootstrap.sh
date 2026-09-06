#!/usr/bin/env bash
set -euo pipefail

# Builds the dependencies a preset needs, in order, then the benchmarks.
#
#   ./cluster/leonardo/bootstrap.sh                 # default target set
#   ./cluster/leonardo/bootstrap.sh oneccl-oshmpi   # one target and its requires
#   ./cluster/leonardo/bootstrap.sh --list
#   GPU_BENCH_FORCE=1 ./cluster/leonardo/bootstrap.sh ...  # rebuild even if installed
#
# Each target is cluster/leonardo/deps/<name>.sh and declares the stack it needs,
# what it requires, and a path that proves it is already built. A backend only
# needs one here if it depends on a library this project installs; the rest of a
# backend is a row in backends.sh, a file in runtime/, and a CMake preset.
#
# The prerequisites this cannot install are a CUDA-capable DPC++ compiler and
# hwloc; point DPCPP_HOME and HWLOC_ROOT at them (see the Leonardo README).

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="$script_dir/deps"

# Both oneCCL libraries: the NCCL-backed one for sycl_oneccl and the OSHMPI-backed
# one for sycl_oneccl_oshmpi. Building only one leaves a backend without a
# library, which surfaces later as a preset that will not configure.
default_targets=(oneccl-nccl oneccl-oshmpi benchmarks)

list_targets() {
    local f
    for f in "$build_dir"/*.sh; do
        [[ "$(basename "$f")" == _* ]] && continue
        basename "$f" .sh
    done
}

if [[ ${1:-} == --list ]]; then
    list_targets
    exit 0
fi

requested=("$@")
[[ ${#requested[@]} -eq 0 ]] && requested=("${default_targets[@]}")

# Depth-first over GPU_BENCH_BUILD_REQUIRES, so each target appears once, after its
# dependencies. Plain strings rather than associative arrays: the target count is
# small and this runs anywhere, including bash 3.2.
seen=" "
ordered=()
stack=""

resolve() {
    local target=$1 file="$build_dir/$1.sh"
    case "$seen" in *" $target "*) return 0 ;; esac
    if [[ ! -f "$file" ]]; then
        printf 'error: no such build target: %s\n' "$target" >&2
        printf 'available: %s\n' "$(list_targets | tr '\n' ' ')" >&2
        exit 2
    fi
    seen="$seen$target "

    # Sourced for metadata only; the guard in each target stops it running here.
    local GPU_BENCH_BUILD_REQUIRES="" GPU_BENCH_BUILD_STACK=""
    # shellcheck disable=SC1090
    source "$file"

    # environment.sh mutates the shell, so a run that mixed stacks would build
    # later targets against the wrong toolchain. Targets are split across both
    # (nvshmem is cuda, oneccl and benchmarks are sycl), so refuse rather than
    # do that silently.
    if [[ -n "$stack" && -n "$GPU_BENCH_BUILD_STACK" && "$GPU_BENCH_BUILD_STACK" != "$stack" ]]; then
        printf 'error: %s needs the %s stack, but %s is already selected\n' \
            "$target" "$GPU_BENCH_BUILD_STACK" "$stack" >&2
        printf 'build them in separate invocations\n' >&2
        exit 2
    fi
    [[ -n "$GPU_BENCH_BUILD_STACK" ]] && stack=$GPU_BENCH_BUILD_STACK

    local dep
    for dep in $GPU_BENCH_BUILD_REQUIRES; do resolve "$dep"; done
    ordered+=("$target")
}

for target in "${requested[@]}"; do resolve "$target"; done

printf 'building: %s\n' "${ordered[*]}"

# shellcheck disable=SC1090
source "$script_dir/environment.sh" "${stack:-sycl}"

for target in "${ordered[@]}"; do
    GPU_BENCH_BUILD_RUN=1 bash "$build_dir/$target.sh"
done

printf '\nbootstrap complete\n'
printf 'build the complete benchmark suite with:\n'
printf '  make leonardo\n'
printf 'then submit an experiment with:\n'
printf '  cluster/harness/launch.sh --all allreduce\n'
