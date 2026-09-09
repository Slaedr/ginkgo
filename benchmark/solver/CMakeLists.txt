ginkgo_add_typed_benchmark_executables(solver "YES" solver.cpp)
ginkgo_add_single_benchmark_executable(
    solver_compare
    "YES"
    "GKO_BENCHMARK_USE_DOUBLE_PRECISION"
    "d"
    solver_compare.cpp
)
if(GINKGO_BUILD_MPI)
    add_subdirectory(distributed)
endif()
