foreach(bench spmv fgs gmres)
    add_executable(amp_benchmark_${bench} ${bench}.cpp)
    target_link_libraries(
        amp_benchmark_${bench}
        ginkgo
        nlohmann_json::nlohmann_json
    )
endforeach()
