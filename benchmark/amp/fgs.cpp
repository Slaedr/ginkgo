// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * AMPLify FGS (Forward Gauss-Seidel) benchmark
 *
 * Times a single multicolor forward Gauss-Seidel sweep for ELL<double>,
 * ELL<float>, and AMP<double> on a 3D 27-point stencil.
 *
 * The sweep is applied to b = all-ones, starting from x = 0.
 * After timing, the output of each format is compared against ELL<double>
 * (the reference) to quantify accuracy loss.
 *
 * Usage: benchmark_fgs [config.json]
 *
 * Config JSON keys (all optional, defaults shown):
 *   nx, ny, nz        : grid dimensions (64)
 *   executor          : "cuda" | "hip" | "omp" | "reference"  ("cuda")
 *   warmup_reps       : 5
 *   bench_reps        : 20
 *   amp_tolerance     : 0.01
 *   amp_base_format       : "ell" | "csr" ("ell")
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/amp/common.hpp"
#include "benchmark/amp/matrix_generation.hpp"


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};
    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    if (comm.size() != 1) {
        if (comm.rank() == 0) {
            std::cerr << "Error: benchmark_fgs must be run with exactly 1 "
                         "MPI rank.\n";
        }
        return 1;
    }

    const Config cfg = (argc >= 2) ? load_config(argv[1]) : Config();

    std::cout << "AMPLify FGS Benchmark (single sweep)\n";
    print_config(cfg);

    auto exec = make_executor(cfg.executor);

    std::cout << "\nBuilding 3D 27-pt stencil...";
    std::cout.flush();
    OffdiagFn fn(42, cfg);
    const std::array<int, 3> local_grid_dims{cfg.nx, cfg.ny, cfg.nz};
    const auto problem =
        generate_problem_data<double, int>(comm, local_grid_dims, fn);
    const auto& data = problem.mat_data;
    const auto& color_ptrs = problem.color_ptrs;
    const int64_t n = static_cast<int64_t>(data.size[0]);
    const int64_t nnz = static_cast<int64_t>(data.nonzeros.size());
    std::cout << " done.\n";

    // FGS flops per sweep ≈ 2*nnz (SpMV-equivalent work + division per row)
    const double flops = 2.0 * static_cast<double>(nnz);

    print_perf_header("FGS (single sweep)", n, nnz, comm.size());

    json results;
    results["config"] = {{"nx", cfg.nx},
                         {"ny", cfg.ny},
                         {"nz", cfg.nz},
                         {"n", n},
                         {"nnz", nnz},
                         {"executor", cfg.executor},
                         {"amp_tolerance", cfg.amp_tolerance},
                         {"amp_base_format", cfg.amp_base_format}};
    json rows = json::array();

    double baseline_ms = 1.0;
    std::shared_ptr<gko::matrix::Dense<double>> ref_out;

    const std::string fmt_upper =
        (cfg.amp_base_format == "csr") ? "CSR" : "ELL";

    // ---- Base<double> (reference) ----
    {
        using Vec = gko::matrix::Dense<double>;
        using Solver = gko::solver::FwdGaussSeidel<double, int>;

        exec->synchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        auto mat = gko::share(create_local_matrix<double, int>(exec, cfg));
        dynamic_cast<gko::ReadableFromMatrixData<double, int>*>(mat.get())
            ->read(data);
        auto solver =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_color_ptrs(color_ptrs)
                .on(exec)
                ->generate(mat);
        exec->synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double setup_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0);

        const double ms = time_ms(comm, exec, cfg.warmup_reps, cfg.bench_reps,
                                  [&] { solver->apply(b, x); });
        baseline_ms = ms;
        ref_out = gko::clone(exec, x);

        const std::string label = fmt_upper + "<double>";
        const double gflops = flops / (ms * 1e6);
        print_perf_row(label, setup_ms, ms, gflops, baseline_ms, 0.0);
        rows.push_back({{"format", label},
                        {"setup_ms", setup_ms},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", 1.0},
                        {"rel_error_vs_ell_double", 0.0}});
    }

    // ---- Base<float> ----
    {
        using Vec = gko::matrix::Dense<float>;
        using VecD = gko::matrix::Dense<double>;
        using Solver = gko::solver::FwdGaussSeidel<float, int>;

        exec->synchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        gko::matrix_data<float, int> fdata;
        fdata.size = data.size;
        fdata.nonzeros.reserve(data.nonzeros.size());
        for (auto& nz : data.nonzeros) {
            fdata.nonzeros.emplace_back(nz.row, nz.column,
                                        static_cast<float>(nz.value));
        }
        auto mat = gko::share(create_local_matrix<float, int>(exec, cfg));
        dynamic_cast<gko::ReadableFromMatrixData<float, int>*>(mat.get())->read(
            fdata);
        exec->synchronize();
        auto solver =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_color_ptrs(color_ptrs)
                .on(exec)
                ->generate(mat);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double setup_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0f);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0f);

        const double ms = time_ms(comm, exec, cfg.warmup_reps, cfg.bench_reps,
                                  [&] { solver->apply(b, x); });

        // Convert float result to double for error comparison
        gko::matrix_data<float, int> xf_data;
        gko::clone(exec->get_master(), x)->write(xf_data);
        gko::matrix_data<double, int> xd_data;
        xd_data.size = xf_data.size;
        xd_data.nonzeros.reserve(xf_data.nonzeros.size());
        for (auto& nz : xf_data.nonzeros)
            xd_data.nonzeros.emplace_back(nz.row, nz.column,
                                          static_cast<double>(nz.value));
        auto x_d = VecD::create(exec);
        x_d->read(xd_data);

        const std::string label = fmt_upper + "<float>";
        const double gflops = flops / (ms * 1e6);
        const double err = relative_error(exec, x_d.get(), ref_out.get());
        print_perf_row(label, setup_ms, ms, gflops, baseline_ms, err);
        rows.push_back({{"format", label},
                        {"setup_ms", setup_ms},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", baseline_ms / ms},
                        {"rel_error_vs_ell_double", err}});
    }

    // ---- AMP<double> ----
    std::string amp_details;
    {
        using Amp = gko::matrix::AMP<double, int>;
        using Vec = gko::matrix::Dense<double>;
        using Solver = gko::solver::FwdGaussSeidel<double, int>;

        exec->synchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        auto base_mat = gko::share(create_local_matrix<double, int>(exec, cfg));
        dynamic_cast<gko::ReadableFromMatrixData<double, int>*>(base_mat.get())
            ->read(data);
        exec->synchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();
        auto mat =
            gko::share(Amp::build()
                           .with_tolerance(cfg.amp_tolerance)
                           .with_strategy(Amp::tolerance_type::componentwise)
                           .on(exec)
                           ->generate(base_mat));
        exec->synchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto solver =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_color_ptrs(color_ptrs)
                .on(exec)
                ->generate(mat);
        exec->synchronize();
        auto t3 = std::chrono::high_resolution_clock::now();
        const double setup_ms =
            std::chrono::duration<double, std::milli>(t3 - t0).count();
        const double amp_setup_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0);

        const double ms = time_ms(comm, exec, cfg.warmup_reps, cfg.bench_reps,
                                  [&] { solver->apply(b, x); });

        const double gflops = flops / (ms * 1e6);
        const double err = relative_error(exec, x.get(), ref_out.get());
        print_perf_row("AMP<double>", setup_ms, ms, gflops, baseline_ms, err);
        rows.push_back({{"format", "AMP<double>"},
                        {"setup_ms", setup_ms},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", baseline_ms / ms},
                        {"rel_error_vs_ell_double", err}});
        amp_details = compute_amp_details(mat.get(), amp_setup_ms, rows);
    }

    results["fgs"] = rows;

    const std::string out = cfg.output_file_prefix + "fgs_results.json";
    std::ofstream of(out);
    of << std::setw(2) << results << "\n";
    std::cout << amp_details << std::endl;
    std::cout << "Results written to " << out << "\n";
    return 0;
}
