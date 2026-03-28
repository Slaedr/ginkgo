// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * AMPLify distributed SpMV benchmark
 *
 * Compares SpMV performance for ELL<double>, ELL<float>, ELL<half> (if
 * enabled at build time), and AMP<double> on a distributed 3D 27-point
 * stencil, using MPI for multi-GPU parallelism.
 *
 * The index space is uniformly partitioned across MPI ranks.  Each rank
 * generates its own local stencil rows (with global indices) and feeds
 * them to read_distributed.
 *
 * Usage: mpirun -np <P> benchmark_spmv [config.json]
 *
 * Config JSON keys (all optional, defaults shown):
 *   nx, ny, nz        : local grid dimensions per rank (64)
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


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    const auto num_procs = comm.size();
    const auto do_print = rank == 0;
    const Config cfg = (argc >= 2) ? load_config(argv[1]) : Config();

    if (do_print) {
        std::cout << "AMPLify Distributed SpMV Benchmark\n";
        print_config(cfg);
    }

    auto exec = make_executor(cfg.executor);

    // ---- Generate local stencil data (rows have global indices) ----
    if (do_print) {
        std::cout << "\nBuilding 3D 27-pt stencil...";
        std::cout.flush();
    }
    OffdiagFn fn(42 + rank, cfg);
    const std::array<int, 3> local_grid_dims{cfg.nx, cfg.ny, cfg.nz};
    const auto data = generate_problem_data(comm, local_grid_dims, fn);

    const gko::size_type local_n =
        static_cast<gko::size_type>(cfg.nx) * cfg.ny * cfg.nz;
    const gko::size_type global_n = local_n * num_procs;
    const int64_t local_nnz =
        static_cast<int64_t>(data.mat_data.nonzeros.size());
    if (do_print) {
        std::cout << " done.\n";
    }

    // Prepare global-sized matrix_data for read_distributed
    auto mat_data = data.mat_data;
    mat_data.size = {global_n, global_n};

    // ---- Create uniform partition ----
    using partition_t =
        gko::experimental::distributed::Partition<local_idx_t, global_idx_t>;
    auto partition = gko::share(partition_t::build_from_global_size_uniform(
        exec->get_master(), num_procs, static_cast<global_idx_t>(global_n)));

    // SpMV flops = 2 * nnz (one multiply + one add per nonzero)
    // Gather global nnz for GFLOP/s calculation
    int64_t global_nnz = 0;
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT64_T, MPI_SUM, comm.get());
    const double flops = 2.0 * static_cast<double>(global_nnz);

    if (do_print) {
        print_perf_header("SpMV", static_cast<int64_t>(global_n), global_nnz,
                          num_procs);
    }

    json results;
    results["config"] = {{"nx", cfg.nx},
                         {"ny", cfg.ny},
                         {"nz", cfg.nz},
                         {"local_n", local_n},
                         {"global_n", global_n},
                         {"local_nnz", local_nnz},
                         {"global_nnz", global_nnz},
                         {"num_procs", num_procs},
                         {"executor", cfg.executor},
                         {"amp_tolerance", cfg.amp_tolerance}};
    json rows = json::array();

    double baseline_ms = 1.0;
    std::shared_ptr<dist_vec_t<scalar_t>> ref_out;

    // Convenience lambda: compute error, print (rank 0), record JSON.
    auto record = [&](const std::string& label, const double ms,
                      std::shared_ptr<dist_vec_t<scalar_t>> x_dev) {
        const double gflops = flops / (ms * 1e6);
        const double err = relative_error(exec, x_dev.get(), ref_out.get());
        if (do_print) {
            print_perf_row(label, ms, gflops, baseline_ms, err);
        }
        rows.push_back({{"format", label},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", baseline_ms / ms},
                        {"rel_error_vs_ell_double", err}});
    };

    // ---- ELL<double> (reference) ----
    {
        auto mat = dist_mtx_t<double>::create(
            exec, comm, gko::with_matrix_type<gko::matrix::Ell>());
        mat->read_distributed(mat_data, partition);

        auto b = dist_vec_t<double>::create(
            exec, comm, gko::dim<2>{global_n, 1}, gko::dim<2>{local_n, 1});
        b->fill(1.0);
        auto x = dist_vec_t<double>::create(
            exec, comm, gko::dim<2>{global_n, 1}, gko::dim<2>{local_n, 1});
        x->fill(0.0);

        const double ms = time_ms(comm, exec, cfg.warmup_reps, cfg.bench_reps,
                                  [&] { mat->apply(b, x); });
        baseline_ms = ms;
        ref_out = gko::clone(exec, x);

        const double gflops = flops / (ms * 1e6);
        if (do_print) {
            print_perf_row("ELL<double>", ms, gflops, baseline_ms, 0.0);
        }
        rows.push_back({{"format", "ELL<double>"},
                        {"time_ms", ms},
                        {"gflops", gflops},
                        {"speedup", 1.0},
                        {"rel_error_vs_ell_double", 0.0}});
    }

    // ---- ELL<float> ----
    {
        // Convert matrix data to float
        gko::matrix_data<float, global_idx_t> fdata;
        fdata.size = mat_data.size;
        fdata.nonzeros.reserve(mat_data.nonzeros.size());
        for (const auto& nz : mat_data.nonzeros) {
            fdata.nonzeros.emplace_back(nz.row, nz.column,
                                        static_cast<float>(nz.value));
        }

        auto mat = dist_mtx_t<float>::create(
            exec, comm, gko::with_matrix_type<gko::matrix::Ell>());
        mat->read_distributed(fdata, partition);

        auto b = dist_vec_t<float>::create(exec, comm, gko::dim<2>{global_n, 1},
                                           gko::dim<2>{local_n, 1});
        b->fill(1.0f);
        auto x = dist_vec_t<float>::create(exec, comm, gko::dim<2>{global_n, 1},
                                           gko::dim<2>{local_n, 1});
        x->fill(0.0f);

        const double ms = time_ms(comm, exec, cfg.warmup_reps, cfg.bench_reps,
                                  [&] { mat->apply(b, x); });

        auto x_d = to_dist_double(exec, comm, x.get());
        record("ELL<float>", ms, x_d);
    }

#ifdef GINKGO_HAVE_AMP_HALF
    // ---- ELL<half> ----
    {
        using Half = gko::amp::half;

        gko::matrix_data<Half, global_idx_t> hdata;
        hdata.size = mat_data.size;
        hdata.nonzeros.reserve(mat_data.nonzeros.size());
        for (const auto& nz : mat_data.nonzeros) {
            hdata.nonzeros.emplace_back(nz.row, nz.column,
                                        static_cast<Half>(nz.value));
        }

        auto mat = dist_mtx_t<Half>::create(
            exec, comm, gko::with_matrix_type<gko::matrix::Ell>());
        mat->read_distributed(hdata, partition);

        auto b = dist_vec_t<Half>::create(exec, comm, gko::dim<2>{global_n, 1},
                                          gko::dim<2>{local_n, 1});
        b->fill(Half{1.0f});
        auto x = dist_vec_t<Half>::create(exec, comm, gko::dim<2>{global_n, 1},
                                          gko::dim<2>{local_n, 1});
        x->fill(Half{0.0f});

        const double ms = time_ms(comm, exec, cfg.warmup_reps, cfg.bench_reps,
                                  [&] { mat->apply(b, x); });

        auto x_d = to_dist_double(exec, comm, x.get());
        record("ELL<half>", ms, x_d);
    }
#endif

    std::string amp_details;
    // ---- AMP<double> ----
    {
        using Ell = gko::matrix::Ell<double, local_idx_t>;
        using Amp = gko::matrix::AMP<double, local_idx_t>;
        using Csr = gko::matrix::Csr<double, local_idx_t>;

        // Create AMP template from an empty ELL
        auto ell_empty =
            gko::share(Ell::create(exec->get_master(), gko::dim<2>{0, 0}));
        auto amp_template =
            Amp::build()
                .with_tolerance(cfg.amp_tolerance)
                .with_strategy(Amp::tolerance_type::componentwise)
                .on(exec)
                ->generate(ell_empty);
        auto csr_template = Csr::create(exec);

        auto mat = gko::share(dist_mtx_t<double>::create(
            exec, comm, amp_template.get(), csr_template.get()));
        mat->read_distributed(mat_data, partition);

        auto b = dist_vec_t<double>::create(
            exec, comm, gko::dim<2>{global_n, 1}, gko::dim<2>{local_n, 1});
        b->fill(1.0);
        auto x = dist_vec_t<double>::create(
            exec, comm, gko::dim<2>{global_n, 1}, gko::dim<2>{local_n, 1});
        x->fill(0.0);

        const double ms = time_ms(comm, exec, cfg.warmup_reps, cfg.bench_reps,
                                  [&] { mat->apply(b, x); });

        record("AMP<double>", ms, gko::share(std::move(x)));

        // AMP details from the local block
        const auto local_mat =
            dynamic_cast<const Amp*>(mat->get_local_matrix().get());
        if (local_mat && do_print) {
            amp_details = compute_amp_details(local_mat, rows);
        }
    }

    // ---- Output results (rank 0 only) ----
    if (do_print) {
        results["spmv"] = rows;

        const std::string out = "spmv_results.json";
        std::ofstream of(out);
        of << std::setw(2) << results << "\n";
        if (!amp_details.empty()) {
            std::cout << amp_details << std::endl;
        }
        std::cout << "Results written to " << out << "\n";
    }
    return 0;
}
