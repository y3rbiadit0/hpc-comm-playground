#!/usr/bin/env bash

# Where everything this project builds lives. One definition, read by the
# bootstrap targets that produce these paths and by the runtime scripts that
# consume them, so the two cannot drift and nothing has to be overridden by hand.
#
# Two roots, because they have different lifetimes:
#
#   GPU_BENCH_WORK_ROOT    clones and build trees. Large, disposable, on scratch.
#                   Safe to purge; bootstrap recreates it.
#   GPU_BENCH_PREFIX_ROOT  install prefixes. Small, persistent, on $HOME, because jobs
#                   resolve libraries from here at run time.
#
# Relocate everything by setting one of those. Individual paths can still be
# overridden for one-off experiments, and are respected if already set.
#
# This file is sourced from cluster.sh as well as environment.sh, so it must work
# on a login node with no modules loaded and no $SCRATCH -- `launch.sh --explain`
# runs off-cluster. deps/_lib.sh asserts $SCRATCH for the targets that need it.

GPU_BENCH_WORK_ROOT=${GPU_BENCH_WORK_ROOT:-${SCRATCH:-}/gpu-comm-bench}
GPU_BENCH_PREFIX_ROOT=${GPU_BENCH_PREFIX_ROOT:-$HOME/opt/gpu-comm-bench}

# Sources and build trees - scratch.
export GPU_BENCH_SRC_DIR=${GPU_BENCH_SRC_DIR:-$GPU_BENCH_WORK_ROOT/src}
export GPU_BENCH_BUILD_DIR=${GPU_BENCH_BUILD_DIR:-$GPU_BENCH_WORK_ROOT/build}

# Install prefixes - persistent. One target under deps/ produces each of these, and
# runtime/*.sh resolves them by these names. Nothing else may define them: two
# definitions with different defaults resolve by source order, which is how a build
# silently links one install's headers against another's libraries.
#
# Libraries that arrive with a module are not listed here; they belong to
# env/<stack>.sh, and are selected by the variant mechanism below.
export OSHMPI_HOME=${OSHMPI_HOME:-$GPU_BENCH_PREFIX_ROOT/oshmpi}
export ONECCL_OSHMPI_ROOT=${ONECCL_OSHMPI_ROOT:-$GPU_BENCH_PREFIX_ROOT/oneccl-oshmpi}
export ONECCL_NCCL_ROOT=${ONECCL_NCCL_ROOT:-$GPU_BENCH_PREFIX_ROOT/oneccl-nccl}

export GPU_BENCH_WORK_ROOT GPU_BENCH_PREFIX_ROOT

# --- Library variants -------------------------------------------------------
#
# A communication library can come from two places: the module that ships it, or
# an install this project bootstraps. Which one a run used has to key three
# things -- the install prefix, the build directory, and the results tree -- or
# the two meet somewhere they should not. Both failure modes are silent: a stale
# binary left in a shared build directory still runs, and two versions filed
# under one backend name are averaged into one cell by benchscribe.
#
# One selector per library keys all three. GPU_BENCH_<LIB>_VERSION=module, the
# default, leaves the library wherever env/<stack>.sh found it and changes no
# path at all, so the common case is byte-identical to having no variants. Any
# other value names a bootstrapped install: it repoints <LIB>_HOME -- the same
# variable env/<stack>.sh sets with :-, and the one the presets, LD_LIBRARY_PATH
# and the CMake find_* hints already read -- and appends to
# GPU_BENCH_VARIANT_TAG, which suffixes the build/ and results/ trees.
#
#   GPU_BENCH_NVSHMEM_VERSION=3.7.2 ./cluster/leonardo/bootstrap.sh nvshmem
#   GPU_BENCH_NVSHMEM_VERSION=3.7.2 make leonardo-cuda
#   GPU_BENCH_NVSHMEM_VERSION=3.7.2 cluster/harness/launch.sh --all halo_1d
#
#     prefix   $HOME/opt/gpu-comm-bench/nvshmem-3.7.2
#     build    build-nvshmem-3.7.2/leonardo-cuda-nvshmem
#     results  results-nvshmem-3.7.2
#
# Nothing has to be cleaned between two selections, and pointing benchscribe at
# one tree is the whole of the A/B protocol.

# gpu_bench_select_variant <library>
#
# <library> is lowercase and names both the prefix directory and the <LIB>_HOME
# variable, which is why this is one function rather than a block per library:
# NVSHMEM_HOME and NCCL_HOME already follow that shape, so adding NCCL here is
# one more call and no new mechanism.
gpu_bench_select_variant() {
  local lower=$1 upper version
  upper=$(printf '%s' "$lower" | tr '[:lower:]' '[:upper:]')

  local version_var="GPU_BENCH_${upper}_VERSION"
  version=${!version_var:-module}
  printf -v "$version_var" '%s' "$version"
  export "${version_var?}"

  [[ "$version" == module ]] && return 0

  # The version string becomes a path component in three trees, so a stray space
  # or slash from a copy-paste would silently produce a prefix nobody meant and
  # a results directory nobody would find again.
  if [[ ! "$version" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'error: %s=%q is not a usable version\n' "$version_var" "$version" >&2
    printf 'expected "module" or a release such as 3.7.2\n' >&2
    return 2
  fi

  GPU_BENCH_VARIANT_TAG="${GPU_BENCH_VARIANT_TAG}-${lower}-${version}"
  printf -v "${upper}_HOME" '%s' "$GPU_BENCH_PREFIX_ROOT/${lower}-${version}"
  export "${upper}_HOME"
}

# Recomputed from scratch on every source: this file is sourced more than once
# per shell (cluster.sh, then environment.sh), and an accumulating tag would
# append the same suffix twice.
GPU_BENCH_VARIANT_TAG=""
if ! gpu_bench_select_variant nvshmem; then
  # Must not fall through. An unusable version leaves the tag empty and
  # <LIB>_HOME at the module's, so the run would quietly measure the default
  # library and file the results in the default tree -- the exact confusion the
  # tag exists to prevent. Exit in a script; in an interactive shell only stop
  # sourcing, so a typo does not close the terminal.
  case $- in
    *i*) return 2 ;;
    *)   exit 2 ;;
  esac
fi
export GPU_BENCH_VARIANT_TAG
