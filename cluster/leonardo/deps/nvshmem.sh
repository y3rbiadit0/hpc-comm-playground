#!/usr/bin/env bash
set -euo pipefail

# A specific NVSHMEM release, from NVIDIA's prebuilt redistributable archive.
#
# This target is only reached when GPU_BENCH_NVSHMEM_VERSION names a release
# (layout.sh); the default, module, is the 2.11 the nvhpc module already ships
# and needs no install. Each version lands in its own prefix, so several are
# installed side by side and a version can be A/B'd rather than swapped under
# the results.
#
# This target unpacks a binary release rather than building one, which is not the
# usual preference here but is the right call for this library:
#
#   - NVSHMEM's CMake refuses every compiler the cuda stack uses. Upstream
#     CMakeLists.txt fails with "NVSHMEM only supports compilation with GNU (gcc)
#     or Clang compilers", so a source build means introducing a third toolchain
#     into a stack that is otherwise entirely nvc/nvc++.
#   - The redistributable is already configured the way this project would have
#     configured it. Read back from the unpacked
#     include/non_abi/nvshmem_build_options.h:
#
#         NVSHMEM_USE_NCCL       defined
#         NVSHMEM_IBGDA_SUPPORT  defined
#         NVSHMEM_USE_GDRCOPY    defined
#         NVSHMEM_USE_MLX5DV     defined
#
#     NVSHMEM_USE_NCCL is the one that matters most: it is OFF by default in a
#     source build, and without it NVSHMEM's collectives fall off the NCCL
#     dispatch path that runtime/nvshmem.sh documents (0.63 GB/s against
#     85.13 GB/s at 1n4g/16 MiB). A source build that forgot it would make a
#     3.x-vs-2.11 comparison a comparison of build flags.
#
# Three things were checked on Leonardo before adopting the binary, because a
# prebuilt library is only usable if it matches the machine it lands on:
#
#   - Device linking. The archive is built with CUDA 12.9 while this project
#     device-links with nvhpc 24.5's CUDA 12.4 nvcc, and the benchmarks compile
#     -rdc=true. nvlink accepts it: an nvshmem_init/nvshmem_n_pes program links
#     against lib/libnvshmem_device.a and runs to completion. The probe below
#     re-runs that check at bootstrap time so a future version bump cannot fail
#     silently later, in the middle of a preset build.
#   - MPI bootstrap. nvshmem_bootstrap_mpi.so needs libmpi.so.40 and resolves
#     against Leonardo's own hpcx-mpi/2.19, so NVSHMEM_BOOTSTRAP=MPI works with
#     the shipped plugin and nothing has to be rebuilt against HPC-X.
#   - libstdc++. libnvshmem_host.so requires no more than GLIBCXX_3.4.21, well
#     under RHEL8's 3.4.25. Unlike OSHMPI this install therefore needs no gcc
#     module at run time and no gpu_bench_strip_debug pass.
#
# The IBGDA transport plugin is the reason this target exists: Leonardo has no
# PeerMappingOverride=1, which rules out classic IBGDA, and CPU-assisted IBGDA
# (NVSHMEM_IBGDA_NIC_HANDLER=cpu, see runtime/nvshmem.sh) needs a 3.x library.
# The nvhpc module ships 2.11.

GPU_BENCH_BUILD_STACK=cuda
GPU_BENCH_BUILD_REQUIRES=""

# Only run standalone; bootstrap.sh sources this file for its metadata first.
[[ "${BASH_SOURCE[0]}" == "$0" || -n "${GPU_BENCH_BUILD_RUN:-}" ]] || return 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/_lib.sh"

version=${GPU_BENCH_NVSHMEM_VERSION:-module}
if [[ "$version" == module ]]; then
    printf 'error: GPU_BENCH_NVSHMEM_VERSION is module -- the nvhpc module ships\n' >&2
    printf 'NVSHMEM 2.11 and there is nothing to install. Name a release instead:\n' >&2
    printf '  GPU_BENCH_NVSHMEM_VERSION=3.7.2 %s nvshmem\n' \
        "$(dirname "$script_dir")/bootstrap.sh" >&2
    exit 2
fi

# layout.sh maps the selected version to this prefix, and the same mapping is
# what the preset, LD_LIBRARY_PATH and the CMake hints resolve later. Reading it
# back here rather than recomputing it keeps one definition of where a version
# lives.
prefix=${NVSHMEM_HOME:?layout.sh must map GPU_BENCH_NVSHMEM_VERSION to NVSHMEM_HOME}

# Checksums come from NVIDIA's own manifest at
# https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_<version>.json.
# A version with no entry here must supply GPU_BENCH_NVSHMEM_SHA256 rather than
# silently install an unverified archive.
case "$version" in
    3.7.2) known_sha256=996690215898856695521846292b221bc68a34755788ea0b41893be766128f43 ;;
    3.4.5) known_sha256=058cbaddc4ff8646b8d1bd9322e93c90eae54c86e1ac8922f20d8a55a7fa8b7e ;;
    *)     known_sha256="" ;;
esac
sha256=${GPU_BENCH_NVSHMEM_SHA256:-$known_sha256}
if [[ -z "$sha256" ]]; then
    printf 'error: no known checksum for NVSHMEM %s\n' "$version" >&2
    printf 'set GPU_BENCH_NVSHMEM_SHA256 from redistrib_%s.json\n' "$version" >&2
    exit 2
fi

archive="libnvshmem-linux-x86_64-${version}_cuda12-archive"
url=${GPU_BENCH_NVSHMEM_URL:-https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/${archive}.tar.xz}
tarball="$GPU_BENCH_SRC_DIR/${archive}.tar.xz"

if gpu_bench_build_done "$prefix/include/nvshmem.h"; then
    gpu_bench_build_log "NVSHMEM already at $prefix"
else
    gpu_bench_build_log "NVSHMEM $version redistributable"

    mkdir -p "$GPU_BENCH_SRC_DIR"
    if [[ ! -f "$tarball" ]]; then
        printf '   fetching %s\n' "$url"
        # .part so an interrupted transfer is never mistaken for a complete one
        # on the next run.
        if command -v curl >/dev/null 2>&1; then
            curl -fL --progress-bar -o "$tarball.part" "$url"
        else
            wget -q --show-progress -O "$tarball.part" "$url"
        fi
        mv "$tarball.part" "$tarball"
    fi

    printf '   verifying sha256\n'
    printf '%s  %s\n' "$sha256" "$tarball" | sha256sum --check --status || {
        printf 'error: checksum mismatch for %s\n' "$tarball" >&2
        printf 'delete it and re-run, or set GPU_BENCH_NVSHMEM_SHA256\n' >&2
        exit 2
    }

    # Replace rather than merge: unpacking a new version over an old one leaves
    # the previous release's transport and bootstrap plugins behind, and NVSHMEM
    # dlopens those by name at run time.
    rm -rf "$prefix"
    mkdir -p "$prefix"
    tar -xf "$tarball" -C "$prefix" --strip-components=1
fi

for required in include/nvshmem.h lib/libnvshmem_host.so lib/libnvshmem_device.a \
                lib/nvshmem_transport_ibgda.so lib/nvshmem_bootstrap_mpi.so; do
    [[ -e "$prefix/$required" ]] || {
        printf 'error: %s is missing from the unpacked archive\n' "$required" >&2
        exit 2
    }
done

# The build options the library was compiled with decide whether a comparison
# against the nvhpc 2.11 is measuring the version or the configuration, so print
# them where the bootstrap log will keep them.
gpu_bench_build_log "NVSHMEM build options"
grep -E '^#define NVSHMEM_(USE_NCCL|USE_GDRCOPY|USE_MLX5DV|IBGDA_SUPPORT|IBRC_SUPPORT|MPI_SUPPORT)$' \
    "$prefix/include/non_abi/nvshmem_build_options.h" | sed 's/^/   /' || true

# Device-link probe. A prebuilt device library is only useful if the nvcc that
# builds the benchmarks can merge it, and an nvlink refusal here is far easier to
# read than the same refusal inside a preset build.
gpu_bench_build_log "device-link probe against ${CUDACXX:?env/cuda.sh must define CUDACXX}"
probe_dir=$(mktemp -d)
trap 'rm -rf "$probe_dir"' EXIT
cat >"$probe_dir/probe.cu" <<'PROBE'
#include <nvshmem.h>
#include <nvshmemx.h>
int main() { nvshmem_init(); int n = nvshmem_n_pes(); nvshmem_finalize(); return n; }
PROBE
# CPATH is scrubbed deliberately: the nvhpc module puts its own NVSHMEM 2.11
# headers on the default include path, and picking those up while linking the 3.x
# device library fails with a size mismatch on nvshmemi_device_state_d that reads
# like a toolkit incompatibility and is not one.
if env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
    "$CUDACXX" -std=c++17 -rdc=true -arch="sm_${GPU_BENCH_CUDA_ARCH:-80}" \
    -I "$prefix/include" "$probe_dir/probe.cu" \
    -L "$prefix/lib" -lnvshmem_host \
    "$prefix/lib/libnvshmem_device.a" \
    -o "$probe_dir/probe" 2>"$probe_dir/probe.log"; then
    printf '   device link ok\n'
else
    printf 'error: %s cannot device-link this NVSHMEM build\n' "$CUDACXX" >&2
    sed 's/^/   /' "$probe_dir/probe.log" >&2
    printf 'the archive is built for a newer CUDA than this stack provides;\n' >&2
    printf 'try an older GPU_BENCH_NVSHMEM_VERSION (3.4.5 targets CUDA 12.2)\n' >&2
    exit 2
fi

printf '\nNVSHMEM %s: %s\n' "$version" "$prefix"
