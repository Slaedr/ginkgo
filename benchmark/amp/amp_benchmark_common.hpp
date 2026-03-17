// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_AMP_AMP_BENCHMARK_COMMON_HPP_
#define GKO_BENCHMARK_AMP_AMP_BENCHMARK_COMMON_HPP_

/**
 * Common utilities for AMPLify single-GPU benchmarks:
 *   - Configuration (JSON file or defaults)
 *   - Executor creation
 *   - Wall-clock timing with GPU synchronization
 *   - 3D 27-point stencil generation with 8-color ordering
 *   - Solution error computation using Ginkgo Dense ops
 *   - Output helpers
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <ginkgo/ginkgo.hpp>


using json = nlohmann::json;
using int32 = gko::int32;

// ============================================================
// Configuration
// ============================================================

struct Config {
    int nx = 64;
    int ny = 64;
    int nz = 64;
    std::string executor = "cuda";
    int warmup_reps = 5;
    int bench_reps = 20;
    float amp_tolerance = 0.01f;
    double gmres_tol = 1e-8;
    int gmres_max_iters = 1000;
    int gmres_krylov_dim = 50;
};

inline Config load_config(const std::string& path)
{
    Config cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: could not open '" << path
                  << "', using defaults.\n";
        throw std::runtime_error("Could not find options file.");
    }
    json j;
    f >> j;
    if (j.contains("nx")) cfg.nx = j["nx"];
    if (j.contains("ny")) cfg.ny = j["ny"];
    if (j.contains("nz")) cfg.nz = j["nz"];
    if (j.contains("executor")) cfg.executor = j["executor"];
    if (j.contains("warmup_reps")) cfg.warmup_reps = j["warmup_reps"];
    if (j.contains("bench_reps")) cfg.bench_reps = j["bench_reps"];
    if (j.contains("amp_tolerance"))
        cfg.amp_tolerance = j["amp_tolerance"].get<float>();
    if (j.contains("gmres_tol")) cfg.gmres_tol = j["gmres_tol"];
    if (j.contains("gmres_max_iters"))
        cfg.gmres_max_iters = j["gmres_max_iters"];
    if (j.contains("gmres_krylov_dim"))
        cfg.gmres_krylov_dim = j["gmres_krylov_dim"];
    return cfg;
}

// ============================================================
// Executor factory
// ============================================================

inline std::shared_ptr<gko::Executor> make_executor(const std::string& name)
{
    auto omp = gko::OmpExecutor::create();
    if (name == "cuda") return gko::CudaExecutor::create(0, omp);
    if (name == "hip") return gko::HipExecutor::create(0, omp);
    if (name == "omp") return omp;
    if (name == "reference") return gko::ReferenceExecutor::create();
    throw std::runtime_error("Unknown executor: " + name);
}

// ============================================================
// Timing
// ============================================================

/**
 * Time an operation using wall-clock with executor synchronization.
 * Returns average time per rep in milliseconds.
 */
template <typename Fn>
double time_ms(std::shared_ptr<const gko::Executor> exec, int warmup, int reps,
               Fn&& fn)
{
    for (int i = 0; i < warmup; ++i) {
        fn();
        exec->synchronize();
    }
    exec->synchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    exec->synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

// ============================================================
// 3D 27-point stencil generator with 8-coloring
// ============================================================

/**
 * Builds matrix_data for a 3D NxNxN 27-point stencil with rows ordered by
 * 8-color partitioning (color = (i%2) + 2*(j%2) + 4*(k%2)).  Within each
 * color, all nodes are independent under the 27-point stencil, making this
 * ordering suitable for multi-color Gauss-Seidel.
 *
 * Stencil values: diagonal = 26, all 26 off-diagonal neighbors = -1.
 * The matrix is strictly diagonally dominant.
 *
 * @param nx, ny, nz   Grid dimensions.
 * @param color_ptrs   Output: color_ptrs[c] is the first row of color c,
 *                     color_ptrs[8] == n.  Size 9.
 * @return  matrix_data<double, int32> in the color-ordered layout.
 */
inline gko::matrix_data<double, int32> generate_stencil_data(
    int nx, int ny, int nz, std::vector<int32>& color_ptrs)
{
    const int64_t n = static_cast<int64_t>(nx) * ny * nz;

    std::vector<int32> old_to_new(n);
    std::vector<int32> new_to_old(n);
    std::vector<int32> cnt(8, 0);

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                ++cnt[(i % 2) + 2 * (j % 2) + 4 * (k % 2)];

    color_ptrs.resize(9);
    color_ptrs[0] = 0;
    for (int c = 0; c < 8; ++c) color_ptrs[c + 1] = color_ptrs[c] + cnt[c];

    std::vector<int32> fill(8, 0);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int32 old_idx = i + j * nx + k * nx * ny;
                int color = (i % 2) + 2 * (j % 2) + 4 * (k % 2);
                int32 new_idx = color_ptrs[color] + fill[color]++;
                old_to_new[old_idx] = new_idx;
                new_to_old[new_idx] = old_idx;
            }
        }
    }

    gko::matrix_data<double, int32> data(gko::dim<2>{
        static_cast<gko::size_type>(n), static_cast<gko::size_type>(n)});
    data.nonzeros.reserve(27 * n);

    for (int32 new_row = 0; new_row < static_cast<int32>(n); ++new_row) {
        int32 old_idx = new_to_old[new_row];
        int ki = old_idx / (nx * ny);
        int ji = (old_idx % (nx * ny)) / nx;
        int ii = old_idx % nx;

        for (int dk = -1; dk <= 1; ++dk) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    int ni = ii + di, nj = ji + dj, nk = ki + dk;
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 ||
                        nk >= nz)
                        continue;
                    int32 old_col = ni + nj * nx + nk * nx * ny;
                    double val = (di == 0 && dj == 0 && dk == 0) ? 26.0 : -1.0;
                    data.nonzeros.emplace_back(new_row, old_to_new[old_col],
                                               val);
                }
            }
        }
    }
    data.sort_row_major();
    return data;
}

// ============================================================
// Error computation (using Ginkgo Dense parallel ops)
// ============================================================

/**
 * Compute the relative L2 error between a computed Dense<double> vector x
 * and a reference Dense<double> vector ref_vec, both on exec.
 *
 *   error = ||x - ref||_2 / ||ref||_2
 *
 * Uses Ginkgo's add_scaled and compute_norm2 for parallel execution.
 * Only the two scalar norm values are copied back to the host (master)
 * executor.
 */
inline double relative_error(std::shared_ptr<const gko::Executor> exec,
                             const gko::matrix::Dense<double>* x,
                             const gko::matrix::Dense<double>* ref_vec)
{
    using Vec = gko::matrix::Dense<double>;
    using RVec = gko::matrix::Dense<gko::remove_complex<double>>;

    auto diff = gko::clone(exec, x);
    auto neg_one = gko::initialize<Vec>({-1.0}, exec);
    diff->add_scaled(neg_one, ref_vec);  // diff = x - ref_vec

    auto diff_norm = RVec::create(exec, gko::dim<2>{1, 1});
    auto ref_norm = RVec::create(exec, gko::dim<2>{1, 1});
    diff->compute_norm2(diff_norm);
    ref_vec->compute_norm2(ref_norm);

    auto master = exec->get_master();
    auto dn = gko::clone(master, diff_norm);
    auto rn = gko::clone(master, ref_norm);
    double rn_val = rn->at(0, 0);
    return (rn_val > 0.0) ? (dn->at(0, 0) / rn_val) : dn->at(0, 0);
}

// ============================================================
// Output helpers
// ============================================================

inline void print_config(const Config& cfg)
{
    std::cout << "  Grid: " << cfg.nx << "x" << cfg.ny << "x" << cfg.nz
              << "  executor: " << cfg.executor << "\n"
              << "  AMP tolerance: " << cfg.amp_tolerance << "\n"
              << "  Warmup / bench reps: " << cfg.warmup_reps << " / "
              << cfg.bench_reps << "\n";
}

inline void print_perf_header(const std::string& title, int64_t n, int64_t nnz)
{
    std::cout << "\n=== " << title << " ===\n";
    std::cout << "  n = " << n << "  nnz = " << nnz << "\n";
    std::cout << std::left << std::setw(20) << "Format" << std::setw(14)
              << "Time (ms)" << std::setw(14) << "GFLOP/s" << std::setw(10)
              << "Speedup" << std::setw(14) << "Rel. error"
              << "\n"
              << std::string(72, '-') << "\n";
}

inline void print_perf_row(const std::string& label, double ms, double gflops,
                           double baseline_ms, double rel_error)
{
    std::cout << std::left << std::setw(20) << label << std::setw(14)
              << std::fixed << std::setprecision(3) << ms << std::setw(14)
              << std::fixed << std::setprecision(2) << gflops << std::setw(10)
              << std::fixed << std::setprecision(2) << (baseline_ms / ms) << "x"
              << std::setw(14) << std::scientific << std::setprecision(2)
              << rel_error << "\n";
}

#endif  // GKO_BENCHMARK_AMP_AMP_BENCHMARK_COMMON_HPP_
