// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_EXAMPLES_DISTRIBUTED_AMP_SOLVER_POISSON_AMP_HPP
#define GINKGO_EXAMPLES_DISTRIBUTED_AMP_SOLVER_POISSON_AMP_HPP

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <ginkgo/ginkgo.hpp>

namespace ex_dist_amp {

using comm_t = gko::experimental::mpi::communicator;

/// Problem configuration
struct Config {
    int nx = 64;
    int ny = 64;
    int nz = 64;
    std::string executor = "cuda";
    int warmup_reps = 2;
    int bench_reps = 10;
    int solver_reps = 3;
    float amp_tolerance = 0.01f;
    double gmres_tol = 1e-8;
    int gmres_max_iters = 1000;
    int gmres_krylov_dim = 50;

    /**
     * (Isotropic) grid stretching ratio for a domain [0.0, 1.0]^3.
     * A stretching ratio < 1 causes narrower cells close to x = 1.0,
     * otherwise stretched cells are close to x = 0.0.
     */
    double mesh_stretch_ratio = 0.95;

    std::string output_file_prefix = "";
    std::string amp_base_format = "ell";
};

/**
 * Problem LHS matrix, RHS vector and exact solution in multi-color reordered
 * indices and order. Also stores the color pointers.
 */
template <typename scalar_t, typename global_idx_t, typename local_idx_t>
struct ProblemData {
    gko::matrix_data<scalar_t, global_idx_t> mat_data;
    std::vector<scalar_t> local_rhs;
    std::vector<scalar_t> local_exact_solution;
    std::vector<local_idx_t> color_ptrs;
};

/// Generate the problem data for the Poisson problem.
template <typename scalar_t, typename global_idx_t, typename local_idx_t>
ProblemData<scalar_t, global_idx_t, local_idx_t> generate_problem_data(
    comm_t comm, const Config& cfg);

/// A manufactured problem for the Poisson equation.
template <typename scalar_t>
class PoissonPDE {
    const scalar_t alpha_{0.1};

public:
    /// Exact solution
    scalar_t get_solution(const std::array<double, 3>& r) const
    {
        return alpha_ * std::sin(r[0] + r[1] + r[2]) + 1.0 -
               (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    }

    /// Forcing function: f = -laplacian(u) = 3*alpha*sin(x+y+z) + 6
    scalar_t get_forcing_function(const std::array<double, 3>& r) const
    {
        return 3.0 * alpha_ * std::sin(r[0] + r[1] + r[2]) + 6.0;
    }
};

struct MulticolorOrdering {
    std::vector<int> new_to_old;
    std::vector<int> old_to_new;
    std::vector<int> color_ptrs;
};

MulticolorOrdering compute_multicolor_ordering(
    const std::array<int, 3>& local_grid_dims);

template <typename T>
inline std::array<T, 3> get_natural_3d_indices_from_flat(
    const std::array<T, 3>& dims, const T flat_idx)
{
    T iz = flat_idx / (dims[1] * dims[0]);
    T iy = (flat_idx - iz * dims[1] * dims[0]) / dims[0];
    T ix = flat_idx % dims[0];
    return std::array<T, 3>{ix, iy, iz};
}

}  // namespace ex_dist_amp

#endif
