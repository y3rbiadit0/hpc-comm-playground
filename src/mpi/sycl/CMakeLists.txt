include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/sycl.cmake)

# Binaries land here, not in the microbench/ and application/ subdirectories:
# cluster/harness/job.sh resolves one path per backend from the registry in
# cluster/<cluster>/backends.sh. The source layout is a reading aid; the flat
# output directory is the contract.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Two tiers, see README "Benchmarks".
add_subdirectory(microbench)    # one operation, swept over message size
add_subdirectory(application)   # one problem size, several operations combined
