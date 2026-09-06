include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)

find_path(NVSHMEM_INCLUDE_DIR nvshmem.h
  HINTS ${NVSHMEM_INCLUDE_DIR} $ENV{NVSHMEM_HOME}
  PATH_SUFFIXES include)
find_library(NVSHMEM_HOST_LIBRARY nvshmem_host
  HINTS ${NVSHMEM_LIBRARY_PATH} $ENV{NVSHMEM_HOME}
  PATH_SUFFIXES lib lib64)
find_library(NVSHMEM_DEVICE_LIBRARY nvshmem_device
  HINTS ${NVSHMEM_LIBRARY_PATH} $ENV{NVSHMEM_HOME}
  PATH_SUFFIXES lib lib64)

if(NOT NVSHMEM_INCLUDE_DIR OR NOT NVSHMEM_HOST_LIBRARY OR NOT NVSHMEM_DEVICE_LIBRARY)
  message(FATAL_ERROR "NVSHMEM not found. Set NVSHMEM_INCLUDE_DIR/NVSHMEM_LIBRARY_PATH or NVSHMEM_HOME")
endif()

# The headers and the libraries must come from one install.
#
# Selecting an NVSHMEM version gives it its own build directory
# (GPU_BENCH_VARIANT_TAG, see cluster/leonardo/layout.sh), so in normal use two
# versions never reach the same CMake cache. This is the backstop for when they
# do -- a hand-run cmake, or a directory reconfigured by hand.
#
# The three find_* calls each cache their answer, so a reconfigure that repoints
# the include directory leaves a previously cached NVSHMEM_DEVICE_LIBRARY
# untouched. Two NVSHMEM versions then meet in one link, and the device linker
# rejects it with
#
#   nvlink error : Size doesn't match for 'nvshmemi_device_state_d'
#
# because the device state struct differs between versions. That message reads
# like a CUDA toolkit incompatibility and is not one, so catch it here, where
# both paths are still in hand, rather than in a ninja log.
get_filename_component(_nvshmem_hdr_prefix "${NVSHMEM_INCLUDE_DIR}" DIRECTORY)
get_filename_component(_nvshmem_lib_dir    "${NVSHMEM_DEVICE_LIBRARY}" DIRECTORY)
get_filename_component(_nvshmem_lib_prefix "${_nvshmem_lib_dir}" DIRECTORY)
if(NOT _nvshmem_hdr_prefix STREQUAL _nvshmem_lib_prefix)
  message(FATAL_ERROR
    "NVSHMEM headers and libraries come from different installs:\n"
    "  headers:   ${NVSHMEM_INCLUDE_DIR}\n"
    "  libraries: ${NVSHMEM_DEVICE_LIBRARY}\n"
    "Remove ${CMAKE_BINARY_DIR} to clear the cached find_library result, "
    "then reconfigure.")
endif()

# Binaries land here, not in the microbench/ and application/ subdirectories:
# cluster/harness/job.sh resolves one path per backend from the registry in
# cluster/<cluster>/backends.sh. The source layout is a reading aid; the flat
# output directory is the contract.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Two tiers, see README "Benchmarks".
add_subdirectory(microbench)    # one operation, swept over message size
add_subdirectory(application)   # one problem size, several operations combined
