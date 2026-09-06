#!/usr/bin/env bash

# Shared helpers for cluster/leonardo/deps/<target>.sh.
#
# Each build target is a standalone script that can be run directly, or through
# cluster/leonardo/bootstrap.sh which resolves ordering. A target declares:
#
#   GPU_BENCH_BUILD_STACK      env/<stack>.sh it needs (cuda or sycl)
#   GPU_BENCH_BUILD_REQUIRES   other targets to build first, space separated
#
# Whether to skip work already done is each target's own call: an install prefix
# existing is enough for OSHMPI, while oneCCL always re-enters cmake so a source
# change is picked up.
#
# Everything is built under $SCRATCH, including clones. Only install prefixes
# land in $HOME.

# The layout owns every path a target produces; targets never invent their own.
# shellcheck disable=SC1090
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/layout.sh"

# layout.sh is sourced by the harness too, including off-cluster where $SCRATCH
# does not exist, so it cannot demand one. The build targets are the only
# consumers of GPU_BENCH_WORK_ROOT, so the demand belongs here: a clone or a
# build tree written to /gpu-comm-bench would fail later and less clearly.
: "${SCRATCH:?set SCRATCH or GPU_BENCH_WORK_ROOT to a build filesystem}"

gpu_bench_build_log() { printf '\n== %s\n' "$*"; }

# gpu_bench_clone_at <url> <ref> <dir>
#
# Leaves <dir> checked out at <ref> in a detached head, whether it was just
# cloned or already present. Fetches only when the ref is missing, so re-running
# offline works once the clone exists.
gpu_bench_clone_at() {
    local url=$1 ref=$2 dir=$3

    if [[ ! -d "$dir/.git" ]]; then
        gpu_bench_build_log "cloning $url -> $dir"
        mkdir -p "$(dirname "$dir")"
        git clone --quiet "$url" "$dir"
    fi

    if ! git -C "$dir" rev-parse --verify --quiet "$ref^{commit}" >/dev/null; then
        git -C "$dir" fetch --quiet --tags origin "$ref" 2>/dev/null ||
            git -C "$dir" fetch --quiet origin
    fi

    local resolved
    resolved=$(git -C "$dir" rev-parse --verify --quiet "origin/$ref^{commit}" ||
               git -C "$dir" rev-parse --verify --quiet "$ref^{commit}") || {
        printf 'error: ref %s not found in %s\n' "$ref" "$url" >&2
        return 2
    }

    git -C "$dir" checkout --quiet --detach "$resolved"
    printf '   %s at %s (%s)\n' "$(basename "$dir")" "$ref" "${resolved:0:8}"
}

# gpu_bench_build_done <path> - true when the target is already built.
gpu_bench_build_done() {
    [[ -n "${GPU_BENCH_FORCE:-}" ]] && return 1
    [[ -e "$1" ]]
}

# gpu_bench_reset_cmake_dir <dir> <expected-cxx-compiler>
#
# A CMake build directory records the compiler it was configured with, and
# refuses to be reconfigured with another one. That is the right default, but
# here it strands the build: oneCCL is configured with DPC++ while the same
# scratch tree may hold a directory configured by an earlier attempt (or a
# half-written cache, which reports an empty compiler and matches nothing).
#
# Removing the directory is safe -- it holds only build output -- so this wipes
# it whenever the recorded compiler is missing or different, and leaves it alone
# when it matches so incremental builds still work.
gpu_bench_reset_cmake_dir() {
    local dir="$1" want="$2" cache="$1/CMakeCache.txt" have=""

    if [[ -n "${GPU_BENCH_FORCE:-}" && -d "$dir" ]]; then
        gpu_bench_build_log "rebuilding from scratch: $dir"
        rm -rf "$dir"
        return 0
    fi

    [[ -f "$cache" ]] || return 0
    have=$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$cache" | head -1)
    if [[ "$have" != "$want" ]]; then
        gpu_bench_build_log "stale build directory (compiler was '${have:-unset}', want '$want'): $dir"
        rm -rf "$dir"
    fi
}

# gpu_bench_strip_debug <file>...
#
# OSHMPI is built under the sycl stack (gcc 12.2, binutils 2.42) but linked into
# the CUDA benchmarks by nvc++, which drives /usr/bin/ld -- RHEL8 binutils 2.30.
# The newer assembler writes compressed .debug_* sections that the older linker
# cannot read, and it fails with
#
#   unable to initialize decompress status for section .debug_aranges
#   file not recognized: File format not recognized
#
# on a library that is otherwise perfectly good. Debug info in a dependency buys
# us nothing, so drop it and the two toolchains stop disagreeing. Stripping after
# install rather than passing -gz=none during the build keeps this independent of
# whether the upstream build script honours CFLAGS.
gpu_bench_strip_debug() {
    local f stripped=0
    for f in "$@"; do
        # Skip symlinks: liboshmpi.so and .so.0 point at .so.0.0.0.
        [[ -f "$f" && ! -L "$f" ]] || continue
        strip --strip-debug "$f" && stripped=$((stripped + 1))
    done
    gpu_bench_build_log "stripped debug sections from $stripped librar$( ((stripped == 1)) && echo y || echo ies )"
}
