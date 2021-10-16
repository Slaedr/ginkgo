cmake_minimum_required(VERSION 3.13)
project(batched-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

add_executable(batched-picard-solver-from-files batched-picard-solver-from-files.cpp)
target_link_libraries(batched-picard-solver-from-files Ginkgo::ginkgo)

foreach(ibatch RANGE 0 1)
    configure_file("data/pores_1/${ibatch}/A.mtx" "data/pores_1/p_0/${ibatch}/A.mtx" COPYONLY)
    configure_file("data/pores_1/${ibatch}/b.mtx" "data/pores_1/p_0/${ibatch}/b.mtx" COPYONLY)
    configure_file("data/pores_1/${ibatch}/b.mtx" "data/pores_1/p_0/${ibatch}/x_init.mtx" COPYONLY)
    configure_file("data/pores_1/${ibatch}/A.mtx" "data/pores_1/p_1/${ibatch}/A.mtx" COPYONLY)
    configure_file("data/pores_1/${ibatch}/b.mtx" "data/pores_1/p_1/${ibatch}/b.mtx" COPYONLY)
    configure_file("data/pores_1/${ibatch}/b.mtx" "data/pores_1/p_1/${ibatch}/x_init.mtx" COPYONLY)
endforeach()
