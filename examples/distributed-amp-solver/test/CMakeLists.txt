if(GINKGO_BUILD_MPI AND GINKGO_BUILD_TESTS)
    add_executable(
        example_dist_amp_matrix_generation
        matrix_generation_test.cpp
    )
    target_link_libraries(
        example_dist_amp_matrix_generation
        PRIVATE problem_generation ginkgo GTest::GTest ginkgo_gtest_main_mpi
    )
    target_include_directories(
        example_dist_amp_matrix_generation
        PRIVATE ${Ginkgo_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    add_test(
        NAME examples/distributed-amp-solver/test/matrix_generation
        COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            $<TARGET_FILE:example_dist_amp_matrix_generation>
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>"
    )

    add_executable(
        example_dist_amp_distributed_matrix_generation
        distributed_matrix_generation_test.cpp
    )
    target_link_libraries(
        example_dist_amp_distributed_matrix_generation
        PRIVATE problem_generation ginkgo GTest::GTest ginkgo_gtest_main_mpi
    )
    target_include_directories(
        example_dist_amp_distributed_matrix_generation
        PRIVATE ${Ginkgo_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    add_test(
        NAME examples/distributed-amp-solver/test/distributed_matrix_generation
        COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            $<TARGET_FILE:example_dist_amp_distributed_matrix_generation>
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>"
    )
endif()
