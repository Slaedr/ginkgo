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
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/amp/amp_benchmark_common.hpp"


int main(int argc, char* argv[])
{
    const Config cfg = (argc >= 2) ? load_config(argv[1]) : Config();

    std::cout << "AMPLify FGS Benchmark (single sweep)\n";
    print_config(cfg);

    auto exec = make_executor(cfg.executor);

    std::cout << "\nBuilding 3D 27-pt stencil...";
    std::cout.flush();
    std::vector<int32> color_ptrs;
    OffdiagFn fn(42, cfg);
    auto data = generate_stencil_data(cfg.nx, cfg.ny, cfg.nz, fn, color_ptrs);
    const int64_t n = data.size[0];
    const int64_t nnz = data.nonzeros.size();
    std::cout << " done.\n";

    // FGS flops per sweep ≈ 2*nnz (SpMV-equivalent work + division per row)
    const double flops = 2.0 * static_cast<double>(nnz);

    print_perf_header("FGS (single sweep)", n, nnz);

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
    std::shared_ptr<gko::matrix::Dense<double>> ref_out;

    // ---- ELL<double> (reference) ----
    {
        using Ell = gko::matrix::Ell<double, int32>;
        using Vec = gko::matrix::Dense<double>;
        using Solver = gko::solver::FwdGaussSeidel<double, int32>;

        auto ell_ref = Ell::create(exec->get_master());
        ell_ref->read(data);
        auto mat = gko::share(gko::clone(exec, ell_ref));
        auto solver =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_color_ptrs(color_ptrs)
                .on(exec)
                ->generate(mat);

        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0);

        double ms = time_ms(exec, cfg.warmup_reps, cfg.bench_reps, [&] {
            // x->fill(0.0);
            solver->apply(b, x);
        });
        baseline_ms = ms;
        ref_out = gko::clone(exec, x);

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
        using Solver = gko::solver::FwdGaussSeidel<float, int32>;

        gko::matrix_data<float, int32> fdata;
        fdata.size = data.size;
        fdata.nonzeros.reserve(data.nonzeros.size());
        for (auto& nz : data.nonzeros)
            fdata.nonzeros.emplace_back(nz.row, nz.column,
                                        static_cast<float>(nz.value));
        auto ell_ref = Ell::create(exec->get_master());
        ell_ref->read(fdata);
        auto mat = gko::share(gko::clone(exec, ell_ref));
        auto solver =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_color_ptrs(color_ptrs)
                .on(exec)
                ->generate(mat);

        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0f);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0f);

        double ms = time_ms(exec, cfg.warmup_reps, cfg.bench_reps, [&] {
            // x->fill(0.0f);
            solver->apply(b, x);
        });

        // Convert float result to double for error comparison
        gko::matrix_data<float, int32> xf_data;
        gko::clone(exec->get_master(), x)->write(xf_data);
        gko::matrix_data<double, int32> xd_data;
        xd_data.size = xf_data.size;
        xd_data.nonzeros.reserve(xf_data.nonzeros.size());
        for (auto& nz : xf_data.nonzeros)
            xd_data.nonzeros.emplace_back(nz.row, nz.column,
                                          static_cast<double>(nz.value));
        auto x_d = VecD::create(exec);
        x_d->read(xd_data);

        double gflops = flops / (ms * 1e6);
        double err = relative_error(exec, x_d.get(), ref_out.get());
        print_perf_row("ELL<float>", ms, gflops, baseline_ms, err);
        rows.push_back({{"format", "ELL<float>"},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", baseline_ms / ms},
                        {"rel_error_vs_ell_double", err}});
    }

    // ---- AMP<double> ----
    std::string amp_details;
    {
        using Ell = gko::matrix::Ell<double, int32>;
        using Amp = gko::matrix::AMP<double, int32>;
        using Vec = gko::matrix::Dense<double>;
        using Solver = gko::solver::FwdGaussSeidel<double, int32>;

        auto ell_ref = Ell::create(exec->get_master());
        ell_ref->read(data);
        auto ell_dev = gko::share(gko::clone(exec, ell_ref));
        auto mat =
            gko::share(Amp::build()
                           .with_tolerance(cfg.amp_tolerance)
                           .with_strategy(Amp::tolerance_type::componentwise)
                           .on(exec)
                           ->generate(ell_dev));
        auto solver =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .with_color_ptrs(color_ptrs)
                .on(exec)
                ->generate(mat);

        auto b =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        b->fill(1.0);
        auto x =
            Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        x->fill(0.0);

        double ms = time_ms(exec, cfg.warmup_reps, cfg.bench_reps, [&] {
            // x->fill(0.0);
            solver->apply(b, x);
        });

        double gflops = flops / (ms * 1e6);
        double err = relative_error(exec, x.get(), ref_out.get());
        print_perf_row("AMP<double>", ms, gflops, baseline_ms, err);
        rows.push_back({{"format", "AMP<double>"},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", baseline_ms / ms},
                        {"rel_error_vs_ell_double", err}});
        amp_details = compute_amp_details(mat.get(), rows);
    }

    results["fgs"] = rows;

    std::string out = "fgs_results.json";
    std::ofstream of(out);
    of << std::setw(2) << results << "\n";
    std::cout << amp_details << std::endl;
    std::cout << "Results written to " << out << "\n";
    return 0;
}
