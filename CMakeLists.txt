cmake_minimum_required(VERSION 3.24)

project(gpu_bench VERSION 0.0.3 LANGUAGES CXX)

include(cmake/version.cmake)

include(CTest)

option(ENABLE_CUDA_MPI "Build CUDA + MPI examples" OFF)
option(ENABLE_SYCL_MPI "Build SYCL + MPI examples" OFF)
option(ENABLE_SYCL_ONECCL "Build SYCL + oneCCL examples" OFF)
option(ENABLE_CUDA_NCCL "Build CUDA + NCCL examples" OFF)
option(ENABLE_CUDA_NVSHMEM "Build CUDA + NVSHMEM examples" OFF)
option(ENABLE_OSHMPI "Build OSHMPI examples" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

configure_file(cmake/version.hpp.in generated/gpu_bench/version.hpp @ONLY)

add_library(gpu_bench_common INTERFACE)
target_include_directories(gpu_bench_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)

if(BUILD_TESTING)
  find_package(Threads REQUIRED)
  add_executable(timing_stats_test tests/timing_stats_test.cpp)
  target_link_libraries(timing_stats_test PRIVATE gpu_bench_common Threads::Threads)
  add_test(NAME timing_stats COMMAND timing_stats_test)
endif()

if(ENABLE_CUDA_MPI)
  add_subdirectory(src/mpi/cuda)
endif()

if(ENABLE_SYCL_MPI)
  add_subdirectory(src/mpi/sycl)
endif()

if(ENABLE_SYCL_ONECCL)
  add_subdirectory(src/xccl/sycl)
endif()

if(ENABLE_CUDA_NCCL)
  add_subdirectory(src/xccl/cuda)
endif()

if(ENABLE_CUDA_NVSHMEM)
  add_subdirectory(src/shmem/nvshmem)
endif()

if(ENABLE_OSHMPI)
  add_subdirectory(src/shmem/oshmpi)
endif()
