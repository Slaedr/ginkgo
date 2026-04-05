// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_EXAMPLES_DISTRIBUTED_AMP_SOLVER_UTILS_H_
#define GINKGO_EXAMPLES_DISTRIBUTED_AMP_SOLVER_UTILS_H_

#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include "poisson_amp.hpp"

using json = nlohmann::json;

namespace ex_dist_amp {

Config load_config(const std::string& path);

std::shared_ptr<gko::Executor> make_executor(const std::string& name);

namespace gkodist = gko::experimental::distributed;

/**
 * Create a distributed matrix using the configured base format for the
 * local and non-local parts.
 */
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<gkodist::Matrix<ValueType, LocalIndexType, GlobalIndexType>>
create_dist_matrix(std::shared_ptr<const gko::Executor> exec, comm_t comm,
                   const Config& cfg);

/**
 * Create an AMP distributed matrix.  The diagonal (local) block is an AMP
 * matrix whose bins use the configured base format, and the off-diagonal
 * (non-local) block uses CSR.
 */
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<gkodist::Matrix<ValueType, LocalIndexType, GlobalIndexType>>
create_amp_dist_matrix(std::shared_ptr<const gko::Executor> exec, comm_t comm,
                       const Config& cfg);

/**
 * Compute the relative L2 error between a computed distributed vector x
 * and a reference distributed vector ref_vec, both on exec.
 *
 *   error = ||x - ref||_2 / ||ref||_2
 *
 * Uses Ginkgo's add_scaled and compute_norm2 for parallel execution.
 * Only the two scalar norm values are copied back to the host (master)
 * executor.
 */
double relative_error(std::shared_ptr<const gko::Executor> exec,
                      const gkodist::Vector<double>* x,
                      const gkodist::Vector<double>* ref_vec);

/**
 * Extracts bin details for AMP matrix, and adds it to results dict and
 * returns it as a string.
 */
std::string compute_amp_details(const Config& cfg,
                                const gko::matrix::AMP<double, int>* const mtx,
                                const double amp_setup_ms, json& rows);

inline void print_config(const Config& cfg)
{
    std::cout << "  Grid: " << cfg.nx << "x" << cfg.ny << "x" << cfg.nz
              << "\n    stretching ratio: " << cfg.mesh_stretch_ratio << "\n"
              << "  executor: " << cfg.executor << "\n"
              << "  Base format: " << cfg.amp_base_format << "\n"
              << "  AMP tolerance: " << cfg.amp_tolerance << "\n"
              << "  Warmup / bench reps: " << cfg.warmup_reps << " / "
              << cfg.bench_reps << "\n";
}

inline void print_perf_header(const std::string& title, int64_t n, int64_t nnz,
                              const int num_procs)
{
    std::cout << "\n=== " << title << " ===\n";
    std::cout << "  n = " << n << "  nnz = " << nnz
              << "   procs = " << num_procs << "\n";
    std::cout << std::left << std::setw(20) << "Format" << std::setw(17)
              << "Setup time (ms)" << std::setw(14) << "Op Time (ms)"
              << std::setw(10) << "GFLOP/s" << std::setw(12) << "Op speedup"
              << std::setw(14) << "Rel. error"
              << "\n"
              << std::string(86, '-') << "\n";
}

inline void print_perf_row(const std::string& label, const double setup_ms,
                           double ms, double gflops, double baseline_ms,
                           double rel_error)
{
    std::cout << std::left << std::setw(20) << label << std::setw(17)
              << std::fixed << std::setprecision(3) << setup_ms << std::setw(14)
              << std::fixed << std::setprecision(3) << ms << std::setw(10)
              << std::fixed << std::setprecision(2) << gflops << std::setw(12)
              << std::fixed << std::setprecision(2) << baseline_ms / ms
              << std::setw(14) << std::scientific << std::setprecision(2)
              << rel_error << "\n";
}

}  // namespace ex_dist_amp

#endif  // GINKGO_EXAMPLES_DISTRIBUTED_AMP_SOLVER_UTILS_H_
