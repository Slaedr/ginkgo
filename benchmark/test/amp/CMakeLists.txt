if(GINKGO_BUILD_MPI)
    add_executable(benchmark_test_amp_matrix_generation matrix_generation.cpp)
    target_link_libraries(
        benchmark_test_amp_matrix_generation
        PRIVATE ginkgo GTest::GTest ginkgo_gtest_main_mpi
    )
    target_include_directories(
        benchmark_test_amp_matrix_generation
        PRIVATE ${Ginkgo_BINARY_DIR}
    )
    add_test(
        NAME benchmark/test/amp/matrix_generation
        COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            $<TARGET_FILE:benchmark_test_amp_matrix_generation>
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>"
    )

    add_executable(
        benchmark_test_amp_dist_matrix_generation
        distributed_matrix_generation.cpp
    )
    target_link_libraries(
        benchmark_test_amp_dist_matrix_generation
        PRIVATE ginkgo GTest::GTest ginkgo_gtest_main_mpi
    )
    target_include_directories(
        benchmark_test_amp_dist_matrix_generation
        PRIVATE ${Ginkgo_BINARY_DIR}
    )
    add_test(
        NAME benchmark/test/amp/distributed_matrix_generation
        COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            $<TARGET_FILE:benchmark_test_amp_dist_matrix_generation>
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>"
    )
endif()
