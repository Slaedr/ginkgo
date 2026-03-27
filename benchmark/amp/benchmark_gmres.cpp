// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * AMPLify GMRES + GS preconditioner benchmark
 *
 * Solves a 3D 27-point stencil system using GMRES preconditioned with a
 * single-sweep Forward Gauss-Seidel, comparing ELL<double> vs AMP<double>
 * for the system matrix (and preconditioner matrix).
 *
 * The RHS is b = A * ones, so the analytical solution is x* = ones.
 * The relative solution error ||x - ones||_2 / ||ones||_2 is reported.
 *
 * Usage: benchmark_gmres [config.json]
 *
 * Config JSON keys (all optional, defaults shown):
 *   nx, ny, nz        : grid dimensions (64)
 *   executor          : "cuda" | "hip" | "omp" | "reference"  ("cuda")
 *   amp_tolerance     : 0.01
 *   gmres_tol         : 1e-8
 *   gmres_max_iters   : 1000
 *   gmres_krylov_dim  : 50
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/amp/amp_benchmark_common.hpp"
#include "benchmark/amp/matrix_generation.hpp"


struct GmresStats {
    double setup_ms;
    double solve_ms;
    int iters;
    double rel_error;
    bool converged;
    double final_res_norm;
};

/**
 * Solve A*x = b using GMRES preconditioned by a single FGS sweep.
 * Returns timing, iteration count, and solution error vs x* = ones.
 */
GmresStats run_gmres(std::shared_ptr<const gko::Executor> exec,
                     std::shared_ptr<const gko::LinOp> system_mat,
                     std::shared_ptr<const gko::matrix::Dense<double>> b,
                     std::shared_ptr<const gko::matrix::Dense<double>> x_ref,
                     const std::vector<int32>& color_ptrs, const Config& cfg)
{
    using Vec = gko::matrix::Dense<double>;
    using RVec = gko::matrix::Dense<gko::remove_complex<double>>;
    using Gmres = gko::solver::Gmres<double>;
    using FGS = gko::solver::FwdGaussSeidel<double, int32>;

    const auto n = system_mat->get_size()[0];

    // --- Setup ---
    exec->synchronize();
    auto t_setup0 = std::chrono::high_resolution_clock::now();

    auto solver =
        Gmres::build()
            .with_krylov_dim(static_cast<gko::size_type>(cfg.gmres_krylov_dim))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(
                    static_cast<unsigned>(cfg.gmres_max_iters)),
                gko::stop::ResidualNorm<double>::build().with_reduction_factor(
                    cfg.gmres_tol))
            .with_preconditioner(
                FGS::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .with_color_ptrs(color_ptrs))
            .on(exec)
            ->generate(system_mat);

    exec->synchronize();
    auto t_setup1 = std::chrono::high_resolution_clock::now();
    const double setup_ms =
        std::chrono::duration<double, std::milli>(t_setup1 - t_setup0).count();

    // --- Solve ---
    auto logger = gko::share(gko::log::Convergence<double>::create());
    solver->add_logger(logger);

    auto x = Vec::create(exec, gko::dim<2>{n, 1});

    exec->synchronize();
    auto t_solve0 = std::chrono::high_resolution_clock::now();
    for (int irep = 0; irep < cfg.solver_reps; irep++) {
        x->fill(0.0);
        solver->apply(b, x);
    }
    exec->synchronize();
    auto t_solve1 = std::chrono::high_resolution_clock::now();

    const double solve_ms =
        std::chrono::duration<double, std::milli>(t_solve1 - t_solve0).count();

    // --- Final residual norm (untimed) ---
    auto rnorm = RVec::create(exec, gko::dim<2>{1, 1});
    auto residual = gko::clone(exec, b);
    const auto one = gko::initialize<Vec>({1.0}, exec);
    const auto neg_one = gko::initialize<Vec>({-1.0}, exec);
    system_mat->apply(neg_one, x, one, residual);  // residual = b - A*x
    residual->compute_norm2(rnorm);
    const double final_res_norm =
        gko::clone(exec->get_master(), rnorm)->at(0, 0);

    // --- Solution error vs reference solution ---
    const double err = relative_error(exec, x.get(), x_ref.get());

    return {setup_ms,
            solve_ms,
            static_cast<int>(logger->get_num_iterations()),
            err,
            logger->has_converged(),
            final_res_norm};
}

// Build RHS b = A_ell_double * ones  (so x* = ones)
std::shared_ptr<gko::matrix::Dense<double>> generate_easy_rhs(
    std::shared_ptr<const gko::Executor> exec,
    const gko::matrix_data<double>& data)
{
    using Ell = gko::matrix::Ell<double, int32>;
    using Vec = gko::matrix::Dense<double>;
    const auto n = data.size[0];
    auto ell_ref = Ell::create(exec->get_master());
    ell_ref->read(data);
    auto mat = gko::share(gko::clone(exec, ell_ref));
    auto ones =
        Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
    ones->fill(1.0);
    auto rhs =
        Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
    rhs->fill(0.0);
    mat->apply(ones, rhs);
    return gko::share(std::move(rhs));
}

std::shared_ptr<const gko::matrix::Dense<double>> generate_rhs(
    std::shared_ptr<const gko::Executor> exec, const size_t n)
{
    using Vec = gko::matrix::Dense<double>;
    auto hrhs = Vec::create(exec->get_master(),
                            gko::dim<2>{static_cast<gko::size_type>(n), 1});
    auto harr = hrhs->get_values();
    for (uint32_t i = 0; i < n; i++) {
        const auto x = static_cast<double>(i);
        harr[i] = 2.0 * std::sin(4 * 3.1415 * x / n);
    }
    auto rhs =
        Vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
    rhs->copy_from(hrhs);
    return gko::share(std::move(rhs));
}

std::shared_ptr<const gko::matrix::Dense<double>> compute_reference_solution(
    std::shared_ptr<const gko::matrix::Ell<double, int32>> mat,
    const std::vector<int32>& color_ptrs,
    std::shared_ptr<const gko::matrix::Dense<double>> rhs)
{
    using Vec = gko::matrix::Dense<double>;
    using RVec = gko::matrix::Dense<gko::remove_complex<double>>;
    using Gmres = gko::solver::Gmres<double>;
    using FGS = gko::solver::FwdGaussSeidel<double, int32>;

    const auto n = mat->get_size()[0];
    auto exec = mat->get_executor();
    const double ref_tol = 1e-14;

    // --- Setup ---
    exec->synchronize();
    auto t_setup0 = std::chrono::high_resolution_clock::now();

    auto solver =
        Gmres::build()
            .with_krylov_dim(60u)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2000),
                gko::stop::ResidualNorm<double>::build().with_reduction_factor(
                    ref_tol))
            .with_preconditioner(
                FGS::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .with_color_ptrs(color_ptrs))
            .on(exec)
            ->generate(mat);

    // --- Solve ---
    auto logger = gko::share(gko::log::Convergence<double>::create());
    solver->add_logger(logger);

    auto x = gko::share(Vec::create(exec, gko::dim<2>{n, 1}));

    solver->apply(rhs, x);

    if (!logger->has_converged()) {
        const std::string err = "Reference solve did not converge to " +
                                std::to_string(ref_tol) + "!\n";
        std::cout << err;
        throw std::runtime_error(err);
    }

    return x;
}

int main(int argc, char* argv[])
{
    using Vec = gko::matrix::Dense<double>;
    std::cout << "Num args = " << argc << std::endl;
    const Config cfg = (argc >= 2) ? load_config(argv[1]) : Config();

    std::cout << "AMPLify GMRES + GS Preconditioner Benchmark\n";
    print_config(cfg);
    std::cout << "  GMRES tol: " << std::scientific << cfg.gmres_tol
              << "  max_iters: " << cfg.gmres_max_iters
              << "  krylov_dim: " << cfg.gmres_krylov_dim << "\n";

    auto exec = make_executor(cfg.executor);

    std::cout << "\nBuilding 3D 27-pt stencil...";
    std::cout.flush();
    std::vector<int32> color_ptrs;
    OffdiagFn fn(42, cfg);
    auto data = generate_stencil_data(cfg.nx, cfg.ny, cfg.nz, fn, color_ptrs);
    const int64_t n = data.size[0];
    const int64_t nnz = data.nonzeros.size();
    std::cout << "\n  Constructing Ell matrix...";
    using EllD = gko::matrix::Ell<double, int32>;
    auto ell_ref = EllD::create(exec->get_master());
    ell_ref->read(data);
    std::shared_ptr<const EllD> ellmat = gko::share(gko::clone(exec, ell_ref));
    std::cout << " done.\n";

    // Build RHS b = A_ell_double * ones  (so x* = ones)
    std::cout << "Generating RHS and reference solution...\n";
    const std::shared_ptr<const Vec> b = generate_rhs(exec, n);
    const std::shared_ptr<const Vec> x_ref =
        compute_reference_solution(ellmat, color_ptrs, b);
    std::cout << "Generated reference solution.\n";

    // Initial residual norm ||b||_2 (x_0 = 0, so r_0 = b)
    json results;
    {
        using RVec = gko::matrix::Dense<double>;
        auto init_rnorm = RVec::create(exec, gko::dim<2>{1, 1});
        b->compute_norm2(init_rnorm);
        const double init_res_norm =
            gko::clone(exec->get_master(), init_rnorm)->at(0, 0);
        results["init_res_norm"] = init_res_norm;
        std::cout << "\n  Initial residual norm ||b||_2 = " << std::scientific
                  << std::setprecision(4) << init_res_norm << "\n";
    }

    std::cout << "\n=== GMRES + GS Preconditioner ===\n";
    std::cout << "  n = " << n << "  nnz = " << nnz << "\n";
    std::cout << std::left << std::setw(12) << "Format" << std::setw(13)
              << "Setup (ms)" << std::setw(13) << "Solve (ms)" << std::setw(13)
              << "Speedup" << std::setw(8) << "Iters" << std::setw(11)
              << "Converged" << std::setw(13) << "Final |r|" << std::setw(14)
              << "Rel. error"
              << "\n"
              << std::string(83, '-') << "\n";

    results["config"] = {{"nx", cfg.nx},
                         {"ny", cfg.ny},
                         {"nz", cfg.nz},
                         {"n", n},
                         {"nnz", nnz},
                         {"executor", cfg.executor},
                         {"amp_tolerance", cfg.amp_tolerance},
                         {"gmres_tol", cfg.gmres_tol},
                         {"gmres_max_iters", cfg.gmres_max_iters},
                         {"gmres_krylov_dim", cfg.gmres_krylov_dim}};
    json rows = json::array();

    auto print_row = [&](const std::string& label, const GmresStats& s,
                         const GmresStats& ref_s) {
        std::cout << std::left << std::setw(12) << label << std::setw(13)
                  << std::fixed << std::setprecision(2) << s.setup_ms
                  << std::setw(13) << std::fixed << std::setprecision(2)
                  << s.solve_ms << std::setw(13) << std::fixed
                  << std::setprecision(2) << ref_s.solve_ms / s.solve_ms
                  << std::setw(8) << s.iters << std::setw(11)
                  << (s.converged ? "yes" : "no") << std::setw(13)
                  << std::scientific << std::setprecision(2) << s.final_res_norm
                  << std::setw(14) << std::scientific << std::setprecision(2)
                  << s.rel_error << "\n";
        rows.push_back({{"format", label},
                        {"setup_ms", s.setup_ms},
                        {"solve_ms", s.solve_ms},
                        {"speedup", ref_s.solve_ms / s.solve_ms},
                        {"iters", s.iters},
                        {"converged", s.converged},
                        {"final_res_norm", s.final_res_norm},
                        {"rel_error_vs_exact", s.rel_error}});
    };

    GmresStats ref_s;
    // ---- ELL<double> system ----
    {
        auto s = run_gmres(exec, ellmat, b, x_ref, color_ptrs, cfg);
        print_row("ELL<double>", s, s);
        ref_s = s;
    }

    // ---- AMP<double> system ----
    std::string amp_details;
    {
        using Amp = gko::matrix::AMP<double, int32>;
        auto mat =
            gko::share(Amp::build()
                           .with_tolerance(cfg.amp_tolerance)
                           .with_strategy(Amp::tolerance_type::componentwise)
                           .on(exec)
                           ->generate(ellmat));
        auto s = run_gmres(exec, mat, b, x_ref, color_ptrs, cfg);
        print_row("AMP<double>", s, ref_s);
        amp_details = compute_amp_details(mat.get(), rows);
    }

    results["gmres"] = rows;

    std::string out = "gmres_results.json";
    std::ofstream of(out);
    of << std::setw(2) << results << "\n";
    std::cout << amp_details << std::endl;
    std::cout << "Results written to " << out << "\n";
    return 0;
}
