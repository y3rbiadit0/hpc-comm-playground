include(${PROJECT_SOURCE_DIR}/cmake/sycl.cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Which MPI the executables call directly. They must use the same one oneCCL loads:
# an Open MPI executable driving a oneCCL that dlopens Intel MPI fails in transport
# setup, and the reverse fails at link.
option(GPU_BENCH_ONECCL_USE_BUNDLED_MPI "Link against oneCCL's bundled Intel MPI" OFF)
# The OSHMPI-backed oneCCL leaves libccl.so with a DT_NEEDED on liboshmpi, which the
# link has to resolve even though nothing here calls shmem_* directly.
option(GPU_BENCH_ONECCL_USE_OSHMPI "The oneCCL install uses the OSHMPI backend" OFF)
set(ONECCL_BUNDLED_MPI_ROOT "" CACHE PATH "oneCCL bundled MPI installation root")

# Two tiers, see README "Benchmarks".
set(GPU_BENCH_ONECCL_MICROBENCH halo_1d pingpong alltoall allreduce)
set(GPU_BENCH_ONECCL_APPLICATION cg_step moe)
set(GPU_BENCH_ONECCL_BENCHMARKS
  ${GPU_BENCH_ONECCL_MICROBENCH} ${GPU_BENCH_ONECCL_APPLICATION})

# find_path/find_library cache their results, so pointing oneCCL_ROOT at a different
# install would otherwise keep the previous one's libraries and link them against the
# new one's headers. Drop the cached answers whenever the root changes.
set(_oneccl_fingerprint "${oneCCL_ROOT}|$ENV{ONECCL_ROOT}|$ENV{CCL_ROOT}")
if(NOT "${_oneccl_fingerprint}" STREQUAL "${ONECCL_CACHED_FINGERPRINT}")
  unset(ONECCL_INCLUDE_DIR CACHE)
  unset(ONECCL_LIBRARY CACHE)
  unset(ONECCL_BUNDLED_MPI_INCLUDE_DIR CACHE)
  unset(ONECCL_BUNDLED_MPI_LIBRARY CACHE)
  set(ONECCL_CACHED_FINGERPRINT "${_oneccl_fingerprint}" CACHE INTERNAL "")
endif()

find_path(ONECCL_INCLUDE_DIR oneapi/ccl.hpp
  HINTS ${oneCCL_ROOT} $ENV{ONECCL_ROOT} $ENV{CCL_ROOT}
  PATH_SUFFIXES include)
find_library(ONECCL_LIBRARY ccl
  HINTS ${oneCCL_ROOT} $ENV{ONECCL_ROOT} $ENV{CCL_ROOT}
  PATH_SUFFIXES lib lib64)

if(NOT ONECCL_INCLUDE_DIR OR NOT ONECCL_LIBRARY)
  message(FATAL_ERROR "oneCCL not found. Set oneCCL_ROOT, ONECCL_ROOT, or CCL_ROOT")
endif()

get_filename_component(ONECCL_LIBRARY_DIR "${ONECCL_LIBRARY}" DIRECTORY)
get_filename_component(_oneccl_install_root "${ONECCL_LIBRARY_DIR}" DIRECTORY)

# Everything the executables need on the loader path. oneCCL is built with
# CMAKE_SKIP_RPATH, so nothing records these for us.
set(ONECCL_RUNTIME_RPATH "${ONECCL_LIBRARY_DIR};$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}")
set(_oneccl_link_libraries gpu_bench_common ${ONECCL_LIBRARY})

if(GPU_BENCH_ONECCL_USE_BUNDLED_MPI)
  # The bundled MPI must come from the same install as the libccl.so being linked.
  # Discovering it independently is what let a stale ONECCL_LIBRARY be linked against
  # another install's headers.
  set(_oneccl_bundled_mpi_roots "${_oneccl_install_root}/opt/mpi")
  if(ONECCL_BUNDLED_MPI_ROOT)
    list(INSERT _oneccl_bundled_mpi_roots 0 "${ONECCL_BUNDLED_MPI_ROOT}")
  endif()

  find_path(ONECCL_BUNDLED_MPI_INCLUDE_DIR mpi.h
    HINTS ${_oneccl_bundled_mpi_roots} PATH_SUFFIXES include NO_DEFAULT_PATH)
  find_library(ONECCL_BUNDLED_MPI_LIBRARY mpi
    HINTS ${_oneccl_bundled_mpi_roots} PATH_SUFFIXES lib/release lib NO_DEFAULT_PATH)

  # Intel MPI ships libmpi.so.12 without the unversioned symlink find_library needs.
  if(NOT ONECCL_BUNDLED_MPI_LIBRARY)
    foreach(_root IN LISTS _oneccl_bundled_mpi_roots)
      file(GLOB _candidates "${_root}/lib/release/libmpi.so.*" "${_root}/lib/libmpi.so.*")
      if(_candidates)
        list(SORT _candidates)
        list(GET _candidates 0 ONECCL_BUNDLED_MPI_LIBRARY)
        break()
      endif()
    endforeach()
  endif()

  if(NOT ONECCL_BUNDLED_MPI_INCLUDE_DIR OR NOT ONECCL_BUNDLED_MPI_LIBRARY)
    message(FATAL_ERROR
      "oneCCL bundled MPI not found under: ${_oneccl_bundled_mpi_roots}\n"
      "That oneCCL install has no opt/mpi. Point oneCCL_ROOT at one that does, set "
      "ONECCL_BUNDLED_MPI_ROOT, or configure with GPU_BENCH_ONECCL_USE_BUNDLED_MPI=OFF.")
  endif()

  # A -I on a missing directory is silently ignored, so the build would otherwise fall
  # through to whichever mpi.h the loaded modules put on CPATH and link those objects
  # against Intel MPI.
  if(NOT EXISTS "${ONECCL_BUNDLED_MPI_INCLUDE_DIR}/mpi.h")
    message(FATAL_ERROR "ONECCL_BUNDLED_MPI_INCLUDE_DIR=${ONECCL_BUNDLED_MPI_INCLUDE_DIR} has no mpi.h")
  endif()

  find_package(Threads REQUIRED)
  get_filename_component(ONECCL_BUNDLED_MPI_LIBRARY_DIR "${ONECCL_BUNDLED_MPI_LIBRARY}" DIRECTORY)
  list(APPEND ONECCL_RUNTIME_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR}")
  list(APPEND _oneccl_link_libraries
    ${ONECCL_BUNDLED_MPI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
else()
  include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
  list(APPEND _oneccl_link_libraries MPI::MPI_CXX)
endif()

if(GPU_BENCH_ONECCL_USE_OSHMPI)
  # Located rather than assumed: OSHMPI installs to lib or lib64 depending on how it
  # was configured, and naming the wrong one surfaces only as fifty undefined shmem_*
  # symbols at link time.
  find_package(OSHMPI REQUIRED)
  list(APPEND ONECCL_RUNTIME_RPATH "${OSHMPI_LIBRARY_DIRS}")
  list(APPEND _oneccl_link_libraries ${OSHMPI_LIBRARIES})
endif()

# Binaries land here, not in the microbench/ and application/ subdirectories:
# cluster/harness/job.sh resolves one path per backend from the registry in
# cluster/<cluster>/backends.sh. The source layout is a reading aid; the flat
# output directory is the contract.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(microbench)    # one operation, swept over message size
add_subdirectory(application)   # one problem size, several operations combined
