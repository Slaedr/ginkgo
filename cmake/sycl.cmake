# IntelSYCL for dpcpp and icpx if the config is existed and cmake reaches the requirement
if(CMAKE_CXX_COMPILER MATCHES "dpcpp|icpx")
    if(CMAKE_HOST_WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
        find_package(IntelSYCL QUIET)
    elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20.5)
        find_package(IntelSYCL QUIET)
    endif()
endif()
# If we do not have the config from compiler, try to set components to make it work.
if(NOT COMMAND add_sycl_to_target)
    if(NOT DEFINED SYCL_FLAGS)
        set(SYCL_FLAGS "-fsycl" CACHE STRING "SYCL flags for compiler")
    endif()
endif()

# oneAPI 2026.0 removed -fsycl-device-lib, where linking all device libraries
# became the default, so only pass it to compilers that still accept it.
# GINKGO_SYCL_DEVICE_LIB_FLAGS is used by every target that links SYCL device
# code, both the ginkgo_dpcpp library and the tests in create_test.cmake.
include(CheckCXXCompilerFlag)
set(gko_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "-fsycl")
check_cxx_compiler_flag(-fsycl-device-lib=all GKO_HAS_SYCL_DEVICE_LIB_ALL)
set(CMAKE_REQUIRED_FLAGS "${gko_saved_required_flags}")
unset(gko_saved_required_flags)
if(GKO_HAS_SYCL_DEVICE_LIB_ALL)
    set(GINKGO_SYCL_DEVICE_LIB_FLAGS -fsycl-device-lib=all)
else()
    set(GINKGO_SYCL_DEVICE_LIB_FLAGS)
endif()

# Provide a uniform way for those package without add_sycl_to_target
function(gko_add_sycl_to_target)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(
        SYCL
        ""
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )
    if(COMMAND add_sycl_to_target)
        add_sycl_to_target(${ARGN})
        return()
    endif()
    # We handle them by adding SYCL_FLAGS to compile and link to the target
    target_compile_options(${SYCL_TARGET} PRIVATE "${SYCL_FLAGS}")
    target_link_options(${SYCL_TARGET} PRIVATE "${SYCL_FLAGS}")
endfunction()
