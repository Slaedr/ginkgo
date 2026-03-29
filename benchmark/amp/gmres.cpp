// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * AMPLify distributed GMRES + Schwarz(FGS) preconditioner benchmark
 *
 * Solves a distributed 3D 27-point stencil system using GMRES preconditioned
 * with a Schwarz preconditioner whose local solver is a single-sweep Forward
 * Gauss-Seidel.  Compares ELL<double> vs AMP<double> for the system matrix.
 *
 * The RHS is a sine-wave vector.  A high-accuracy reference solution is
 * computed first, and the relative solution error is reported.
 *
 * Usage: mpirun -np <P> benchmark_gmres [config.json]
 *
 * Config JSON keys (all optional, defaults shown):
 *   nx, ny, nz        : local grid dimensions per rank (64)
 *   executor          : "cuda" | "hip" | "omp" | "reference"  ("cuda")
 *   amp_tolerance     : 0.01
 *   gmres_tol         : 1e-8
 *   gmres_max_iters   : 1000
 *   gmres_krylov_dim  : 50
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/amp/common.hpp"
#include "benchmark/amp/matrix_generation.hpp"

struct GmresStats {
    double setup_ms;
    double solve_ms;
    int iters;
    double rel_error;
    bool converged;
    double final_res_norm;
};

using local_idx_t = int;
using global_idx_t = long;
template <typename scalar_t>
using dist_vec_t = gko::experimental::distributed::Vector<scalar_t>;
template <typename scalar_t>
using dist_mtx_t = gko::experimental::distributed::Matrix<scalar_t, int, long>;
using Schwarz =
    gko::experimental::distributed::preconditioner::Schwarz<double, local_idx_t,
                                                            global_idx_t>;
using FGS = gko::solver::FwdGaussSeidel<double, local_idx_t>;
using Gmres = gko::solver::Gmres<double>;
using Vec = gko::matrix::Dense<double>;
using RVec = gko::matrix::Dense<gko::remove_complex<double>>;
using DistVec = dist_vec_t<double>;
using DistMtx = dist_mtx_t<double>;

/**
 * Generate a sine-wave RHS as a distributed vector.
 * Each rank fills its local portion based on global row indices.
 */
std::shared_ptr<const DistVec> generate_rhs(
    std::shared_ptr<const gko::Executor> exec, comm_t comm,
    const gko::size_type global_n, const gko::size_type local_n, const int rank)
{
    const auto offset = static_cast<gko::size_type>(rank) * local_n;
    auto local_b = Vec::create(exec->get_master(), gko::dim<2>{local_n, 1});
    auto vals = local_b->get_values();
    for (gko::size_type i = 0; i < local_n; ++i) {
        const auto gx = static_cast<double>(offset + i);
        vals[i] = 2.0 * std::sin(4.0 * 3.14159265358979 * gx / global_n);
    }
    auto local_dev = gko::clone(exec, local_b);
    return gko::share(DistVec::create(exec, comm, gko::dim<2>{global_n, 1},
                                      std::move(local_dev)));
}

/**
 * Compute a high-accuracy reference solution using distributed
 * GMRES + Schwarz(FGS).
 */
std::shared_ptr<const DistVec> compute_reference_solution(
    comm_t comm, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::experimental::distributed::Partition<
        local_idx_t, global_idx_t>>
        partition,
    const ProblemData<double, global_idx_t, local_idx_t>& prob,
    std::shared_ptr<const DistVec> rhs, const gko::size_type global_n,
    const gko::size_type local_n)
{
    // ---- Build ELL<double> distributed matrix ----
    auto system_mat = gko::share(
        DistMtx::create(exec, comm, gko::with_matrix_type<gko::matrix::Ell>()));
    system_mat->read_distributed(prob.mat_data, partition);

    const double ref_tol = 1e-14;

    auto local_factory = gko::share(
        FGS::build()
            .with_color_ptrs(prob.color_ptrs)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec));
    // auto local_factory = gko::share(
    //     gko::preconditioner::GaussSeidel<double, local_idx_t>::build()
    //     .on(exec));

    auto solver =
        Gmres::build()
            .with_krylov_dim(gko::size_type{60})
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2000u),
                gko::stop::ResidualNorm<double>::build().with_reduction_factor(
                    ref_tol))
            .with_preconditioner(
                Schwarz::build().with_local_solver(local_factory).on(exec))
            .on(exec)
            ->generate(system_mat);

    auto logger = gko::share(gko::log::Convergence<double>::create());
    solver->add_logger(logger);

    auto x = gko::share(DistVec::create(exec, comm, gko::dim<2>{global_n, 1},
                                        gko::dim<2>{local_n, 1}));
    x->fill(0.0);
    solver->apply(rhs, x);

    if (!logger->has_converged()) {
        auto resnorm = logger->get_residual_norm();
        auto hresnorm = RVec::create(exec->get_master());
        hresnorm->copy_from(resnorm);
        if (comm.rank() == 0) {
            std::cout << " Achieved reference solve residual = "
                      << hresnorm->at(0, 0) << std::endl;
        }
        comm.synchronize();
        throw std::runtime_error("Reference solve did not converge to " +
                                 std::to_string(ref_tol) + "!");
    }
    return x;
}

/**
 * Solve A*x = b using distributed GMRES + Schwarz(FGS) preconditioner.
 * Returns timing, iteration count, and solution error vs x_ref.
 */
GmresStats run_gmres(comm_t comm, std::shared_ptr<const gko::Executor> exec,
                     const std::vector<local_idx_t>& color_ptrs,
                     std::shared_ptr<const gko::LinOp> system_mat,
                     std::shared_ptr<const DistVec> b,
                     std::shared_ptr<const DistVec> x_ref,
                     const gko::size_type global_n,
                     const gko::size_type local_n, const Config& cfg)
{
    // --- Setup ---
    exec->synchronize();
    comm.synchronize();
    const auto t_setup0 = std::chrono::high_resolution_clock::now();

    auto fgs_factory = gko::share(
        FGS::build()
            .with_color_ptrs(color_ptrs)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec));
    auto solver =
        Gmres::build()
            .with_krylov_dim(static_cast<gko::size_type>(cfg.gmres_krylov_dim))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(
                    static_cast<unsigned>(cfg.gmres_max_iters)),
                gko::stop::ResidualNorm<double>::build().with_reduction_factor(
                    cfg.gmres_tol))
            .with_preconditioner(
                Schwarz::build().with_local_solver(fgs_factory).on(exec))
            .on(exec)
            ->generate(system_mat);
    exec->synchronize();
    comm.synchronize();
    const auto t_setup1 = std::chrono::high_resolution_clock::now();
    const double setup_ms =
        std::chrono::duration<double, std::milli>(t_setup1 - t_setup0).count();

    auto x = DistVec::create(exec, comm, gko::dim<2>{global_n, 1},
                             gko::dim<2>{local_n, 1});

    // --- Solve ---
    auto logger = gko::share(gko::log::Convergence<double>::create());
    solver->add_logger(logger);

    exec->synchronize();
    comm.synchronize();
    const auto t_solve0 = std::chrono::high_resolution_clock::now();
    for (int irep = 0; irep < cfg.solver_reps; ++irep) {
        x->fill(0.0);
        solver->apply(b, x);
    }
    exec->synchronize();
    comm.synchronize();
    const auto t_solve1 = std::chrono::high_resolution_clock::now();
    const double solve_ms =
        std::chrono::duration<double, std::milli>(t_solve1 - t_solve0).count();

    // --- Final residual norm ---
    auto rnorm = RVec::create(exec, gko::dim<2>{1, 1});
    auto residual = gko::clone(exec, b);
    const auto one = gko::initialize<Vec>({1.0}, exec);
    const auto neg_one = gko::initialize<Vec>({-1.0}, exec);
    system_mat->apply(neg_one, x, one, residual);  // residual = b - A*x
    residual->compute_norm2(rnorm);
    const double final_res_norm =
        gko::clone(exec->get_master(), rnorm)->at(0, 0);

    // --- Solution error vs reference ---
    const double err = relative_error(exec, x.get(), x_ref.get());

    return {setup_ms,
            solve_ms,
            static_cast<int>(logger->get_num_iterations()),
            err,
            logger->has_converged(),
            final_res_norm};
}


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    const auto num_procs = comm.size();
    const auto do_print = rank == 0;
    const Config cfg = (argc >= 2) ? load_config(argv[1]) : Config();

    if (do_print) {
        std::cout << "AMPLify Distributed GMRES + Schwarz(FGS) Benchmark\n";
        print_config(cfg);
        std::cout << "  GMRES tol: " << std::scientific << cfg.gmres_tol
                  << "  max_iters: " << cfg.gmres_max_iters
                  << "  krylov_dim: " << cfg.gmres_krylov_dim << "\n";
    }

    auto exec = make_executor(cfg.executor);

    // ---- Generate stencil data ----
    if (do_print) {
        std::cout << "\nBuilding 3D 27-pt stencil...";
        std::cout.flush();
    }
    OffdiagFn fn(42 + rank, cfg);
    const std::array<int, 3> local_grid_dims{cfg.nx, cfg.ny, cfg.nz};
    const auto data =
        generate_problem_data<double, global_idx_t>(comm, local_grid_dims, fn);

    const gko::size_type local_n =
        static_cast<gko::size_type>(cfg.nx) * cfg.ny * cfg.nz;
    const gko::size_type global_n = local_n * num_procs;

    auto mat_data = data.mat_data;
    mat_data.size = {global_n, global_n};
    if (do_print) {
        std::cout << " done.\n";
    }

    // ---- Create partition ----
    using partition_t =
        gko::experimental::distributed::Partition<local_idx_t, global_idx_t>;
    auto partition = gko::share(partition_t::build_from_global_size_uniform(
        exec->get_master(), num_procs, static_cast<global_idx_t>(global_n)));

    // Gather global nnz
    const int64_t local_nnz = static_cast<int64_t>(mat_data.nonzeros.size());
    int64_t global_nnz = 0;
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT64_T, MPI_SUM, comm.get());

    // ---- Generate RHS and reference solution ----
    if (do_print) {
        std::cout << "Generating RHS and reference solution...\n";
    }
    const auto b = generate_rhs(exec, comm, global_n, local_n, rank);

    // ---- Initial residual norm ----
    json results;
    {
        auto init_rnorm = RVec::create(exec, gko::dim<2>{1, 1});
        b->compute_norm2(init_rnorm);
        const double init_res_norm =
            gko::clone(exec->get_master(), init_rnorm)->at(0, 0);
        results["init_res_norm"] = init_res_norm;
        if (do_print) {
            std::cout << "\n  Initial residual norm ||b||_2 = "
                      << std::scientific << std::setprecision(4)
                      << init_res_norm << "\n";
        }
    }

    const auto x_ref = compute_reference_solution(comm, exec, partition, data,
                                                  b, global_n, local_n);
    if (do_print) {
        std::cout << "Generated reference solution.\n";
    }

    if (do_print) {
        std::cout << "\n=== GMRES + Schwarz(FGS) ===\n";
        std::cout << "  global_n = " << global_n
                  << "  global_nnz = " << global_nnz
                  << "  procs = " << num_procs << "\n";
        std::cout << std::left << std::setw(12) << "Format" << std::setw(13)
                  << "Setup (ms)" << std::setw(13) << "Solve (ms)"
                  << std::setw(13) << "Speedup" << std::setw(8) << "Iters"
                  << std::setw(11) << "Converged" << std::setw(13)
                  << "Final |r|" << std::setw(14) << "Rel. error"
                  << "\n"
                  << std::string(83, '-') << "\n";
    }

    results["config"] = {{"nx", cfg.nx},
                         {"ny", cfg.ny},
                         {"nz", cfg.nz},
                         {"local_n", local_n},
                         {"global_n", global_n},
                         {"local_nnz", local_nnz},
                         {"global_nnz", global_nnz},
                         {"num_procs", num_procs},
                         {"executor", cfg.executor},
                         {"amp_tolerance", cfg.amp_tolerance},
                         {"gmres_tol", cfg.gmres_tol},
                         {"gmres_max_iters", cfg.gmres_max_iters},
                         {"gmres_krylov_dim", cfg.gmres_krylov_dim}};
    json rows = json::array();

    auto print_row = [&](const std::string& label, const GmresStats& s,
                         const GmresStats& ref_s) {
        if (do_print) {
            std::cout << std::left << std::setw(12) << label << std::setw(13)
                      << std::fixed << std::setprecision(2) << s.setup_ms
                      << std::setw(13) << std::fixed << std::setprecision(2)
                      << s.solve_ms << std::setw(13) << std::fixed
                      << std::setprecision(2) << ref_s.solve_ms / s.solve_ms
                      << std::setw(8) << s.iters << std::setw(11)
                      << (s.converged ? "yes" : "no") << std::setw(13)
                      << std::scientific << std::setprecision(2)
                      << s.final_res_norm << std::setw(14) << std::scientific
                      << std::setprecision(2) << s.rel_error << "\n";
        }
        rows.push_back({{"format", label},
                        {"setup_ms", s.setup_ms},
                        {"solve_ms", s.solve_ms},
                        {"speedup", ref_s.solve_ms / s.solve_ms},
                        {"iters", s.iters},
                        {"converged", s.converged},
                        {"final_res_norm", s.final_res_norm},
                        {"rel_error_vs_exact", s.rel_error}});
    };

    GmresStats ref_s{};

    // ---- ELL<double> system ----
    {
        exec->synchronize();
        comm.synchronize();
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto ell_mat = gko::share(DistMtx::create(
            exec, comm, gko::with_matrix_type<gko::matrix::Ell>()));
        ell_mat->read_distributed(mat_data, partition);
        exec->synchronize();
        comm.synchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double setup_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto s = run_gmres(comm, exec, data.color_ptrs, ell_mat, b, x_ref,
                           global_n, local_n, cfg);
        s.setup_ms += setup_ms;
        print_row("ELL<double>", s, s);
        ref_s = s;
    }

    // ---- AMP<double> system ----
    std::string amp_details;
    {
        using Ell = gko::matrix::Ell<double, local_idx_t>;
        using Amp = gko::matrix::AMP<double, local_idx_t>;
        using Csr = gko::matrix::Csr<double, local_idx_t>;

        // Time AMP matrix generation
        exec->synchronize();
        comm.synchronize();
        const auto t0 = std::chrono::high_resolution_clock::now();

        auto ell_empty = gko::share(Ell::create(exec, gko::dim<2>{0, 0}));
        auto amp_template =
            Amp::build()
                .with_tolerance(cfg.amp_tolerance)
                .with_strategy(Amp::tolerance_type::componentwise)
                .on(exec)
                ->generate(ell_empty);
        auto csr_template = Csr::create(exec);

        auto amp_mat = gko::share(DistMtx::create(
            exec, comm, amp_template.get(), csr_template.get()));
        amp_mat->read_distributed(mat_data, partition);

        exec->synchronize();
        comm.synchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double setup_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto s = run_gmres(comm, exec, data.color_ptrs, amp_mat, b, x_ref,
                           global_n, local_n, cfg);
        s.setup_ms += setup_ms;
        print_row("AMP<double>", s, ref_s);

        const auto local_mat =
            dynamic_cast<const Amp*>(amp_mat->get_local_matrix().get());
        if (local_mat && do_print) {
            amp_details = compute_amp_details(local_mat, 0.0, rows);
        }
    }

    // ---- Output results (rank 0 only) ----
    if (do_print) {
        results["gmres"] = rows;

        const std::string out = "gmres_results.json";
        std::ofstream of(out);
        of << std::setw(2) << results << "\n";
        if (!amp_details.empty()) {
            std::cout << amp_details << std::endl;
        }
        std::cout << "Results written to " << out << "\n";
    }
    return 0;
}
