// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * AMPLify SpMV benchmark
 *
 * Compares SpMV performance for ELL<double>, ELL<float>, ELL<half> (if
 * enabled at build time), and AMP<double> on a 3D 27-point stencil.
 *
 * The input vector is all-ones.  After timing, the output of each format
 * is compared against ELL<double> (the reference) to quantify accuracy loss.
 *
 * Usage: benchmark_spmv [config.json]
 *
 * Config JSON keys (all optional, defaults shown):
 *   nx, ny, nz        : grid dimensions (64)
 *   executor          : "cuda" | "hip" | "omp" | "reference"  ("cuda")
 *   warmup_reps       : 5
 *   bench_reps        : 20
 *   amp_tolerance     : 0.01
 *   matrix_values_type: "laplace" | "diagonal_dominant" | "general" ("laplace")
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/amp/amp_benchmark_common.hpp"
#include "benchmark/amp/matrix_generation.hpp"
#include "benchmark/utils/general.hpp"


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    const auto do_print = rank == 0;
    const Config cfg = (argc >= 2) ? load_config(argv[1]) : Config();

    if (do_print) {
        std::cout << "AMPLify SpMV Benchmark\n";
        print_config(cfg);
    }

    auto exec = executor_factory_mpi.at(cfg.executor)(comm.get());

    if (do_print) {
        std::cout << "\nBuilding 3D 27-pt stencil...";
        std::cout.flush();
    }
    OffdiagFn fn(42, cfg);
    const std::array<int, 3> local_grid_dims{cfg.nx, cfg.ny, cfg.nz};
    // std::vector<int32> color_ptrs;
    const auto data = generate_problem_data(comm, local_grid_dims, fn);
    const int64_t n = data.mat_data.size[0];
    const int64_t nnz = data.mat_data.nonzeros.size();
    if (do_print) {
        std::cout << " done.\n";
    }

    // SpMV flops = 2 * nnz (one multiply + one add per nonzero)
    const double flops = 2.0 * static_cast<double>(nnz) * comm.size();

    if (do_print) {
        print_perf_header("SpMV", n, nnz, comm.size());
    }

    json results;
    results["config"] = {{"nx", cfg.nx},
                         {"ny", cfg.ny},
                         {"nz", cfg.nz},
                         {"n", n},
                         {"nnz", nnz},
                         {"executor", cfg.executor},
                         {"amp_tolerance", cfg.amp_tolerance}};
    json rows = json::array();

    double baseline_ms = 1.0;
    // Reference output (ELL<double> result) for error comparison.
    std::shared_ptr<dist_vec_t<scalar_t>> ref_out;

    // Convenience lambda: time, compute error, print, record.
    // x_dev is the output vector after one (post-warmup) apply.
    auto record = [&](const std::string& label, const double ms,
                      std::shared_ptr<dist_vec_t<scalar_t>> x_dev) {
        double gflops = flops / (ms * 1e6);
        double err = relative_error(exec, x_dev.get(), ref_out.get());
        print_perf_row(label, ms, gflops, baseline_ms, err);
        rows.push_back({{"format", label},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", baseline_ms / ms},
                        {"rel_error_vs_ell_double", err}});
    };

    // ---- ELL<double> (reference) ----
    {
        using Ell = gko::matrix::Ell<double, int32>;
        using Vec = gko::matrix::Dense<double>;
        auto ell_ref = Ell::create(exec->get_master());
        ell_ref->read(data);
        auto mat = gko::share(gko::clone(exec, ell_ref));
        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0);

        double ms = time_ms(exec, cfg.warmup_reps, cfg.bench_reps,
                            [&] { mat->apply(b, x); });
        baseline_ms = ms;

        // Capture the reference output (one clean apply after warmup)
        ref_out = gko::clone(exec, x);

        // ELL<double> error vs itself is always 0; record explicitly.
        double gflops = flops / (ms * 1e6);
        print_perf_row("ELL<double>", ms, gflops, baseline_ms, 0.0);
        rows.push_back({{"format", "ELL<double>"},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", 1.0},
                        {"rel_error_vs_ell_double", 0.0}});
    }

    // ---- ELL<float> ----
    {
        using Ell = gko::matrix::Ell<float, int32>;
        using Vec = gko::matrix::Dense<float>;
        using VecD = gko::matrix::Dense<double>;
        gko::matrix_data<float, int32> fdata;
        fdata.size = data.size;
        fdata.nonzeros.reserve(data.nonzeros.size());
        for (auto& nz : data.nonzeros)
            fdata.nonzeros.emplace_back(nz.row, nz.column,
                                        static_cast<float>(nz.value));
        auto ell_ref = Ell::create(exec->get_master());
        ell_ref->read(fdata);
        auto mat = gko::share(gko::clone(exec, ell_ref));
        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0f);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0f);

        double ms = time_ms(exec, cfg.warmup_reps, cfg.bench_reps,
                            [&] { mat->apply(b, x); });

        // Convert float result to double for error comparison
        gko::matrix_data<double, int32> xd_data;
        {
            gko::matrix_data<float, int32> xf_data;
            gko::clone(exec->get_master(), x)->write(xf_data);
            xd_data.size = xf_data.size;
            xd_data.nonzeros.reserve(xf_data.nonzeros.size());
            for (auto& nz : xf_data.nonzeros)
                xd_data.nonzeros.emplace_back(nz.row, nz.column,
                                              static_cast<double>(nz.value));
        }
        auto x_d = VecD::create(exec);
        x_d->read(xd_data);

        record("ELL<float>", ms, gko::share(std::move(x_d)));
    }

#ifdef GINKGO_HAVE_AMP_HALF
    // ---- ELL<half> ----
    {
        using Half = gko::amp::half;
        using Ell = gko::matrix::Ell<Half, int32>;
        using Vec = gko::matrix::Dense<Half>;
        using VecD = gko::matrix::Dense<double>;
        gko::matrix_data<Half, int32> hdata;
        hdata.size = data.size;
        hdata.nonzeros.reserve(data.nonzeros.size());
        for (auto& nz : data.nonzeros)
            hdata.nonzeros.emplace_back(nz.row, nz.column,
                                        static_cast<Half>(nz.value));
        auto ell_ref = Ell::create(exec->get_master());
        ell_ref->read(hdata);
        auto mat = gko::share(gko::clone(exec, ell_ref));
        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(Half{1.0f});
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(Half{0.0f});

        double ms = time_ms(exec, cfg.warmup_reps, cfg.bench_reps,
                            [&] { mat->apply(b, x); });

        // Convert half result to double for error comparison
        gko::matrix_data<double, int32> xd_data;
        {
            gko::matrix_data<Half, int32> xh_data;
            gko::clone(exec->get_master(), x)->write(xh_data);
            xd_data.size = xh_data.size;
            xd_data.nonzeros.reserve(xh_data.nonzeros.size());
            for (auto& nz : xh_data.nonzeros)
                xd_data.nonzeros.emplace_back(nz.row, nz.column,
                                              static_cast<double>(nz.value));
        }
        auto x_d = VecD::create(exec);
        x_d->read(xd_data);

        record("ELL<half>", ms, gko::share(std::move(x_d)));
    }
#endif

    std::string amp_details;
    // ---- AMP<double> ----
    {
        using Ell = gko::matrix::Ell<double, int32>;
        using Amp = gko::matrix::AMP<double, int32>;
        using Vec = gko::matrix::Dense<double>;
        auto ell_ref = Ell::create(exec->get_master());
        ell_ref->read(data);
        auto ell_dev = gko::share(gko::clone(exec, ell_ref));
        auto mat =
            gko::share(Amp::build()
                           .with_tolerance(cfg.amp_tolerance)
                           .with_strategy(Amp::tolerance_type::componentwise)
                           .on(exec)
                           ->generate(ell_dev));
        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0);

        double ms = time_ms(exec, cfg.warmup_reps, cfg.bench_reps,
                            [&] { mat->apply(b, x); });

        record("AMP<double>", ms, gko::share(std::move(x)));
        amp_details = compute_amp_details(mat.get(), rows);
    }

    results["spmv"] = rows;

    std::string out = "spmv_results.json";
    std::ofstream of(out);
    of << std::setw(2) << results << "\n";
    std::cout << amp_details << std::endl;
    std::cout << "Results written to " << out << "\n";
    return 0;
}
