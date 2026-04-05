cmake_minimum_required(VERSION 3.16)
project(distributed-amp-solver)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_library(problem_generation generate_problem.cpp)
target_link_libraries(problem_generation PUBLIC Ginkgo::ginkgo)

add_executable(solve_poisson_amp solve_poisson_amp.cpp utils.cpp)
target_link_libraries(
    solve_poisson_amp
    PRIVATE problem_generation nlohmann_json::nlohmann_json
)

add_subdirectory(test)
