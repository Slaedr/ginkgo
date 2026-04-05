// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include "poisson_amp.hpp"

namespace {


using comm_t = gko::experimental::mpi::communicator;


class PoissonMatrixGeneration : public ::testing::Test {
public:
    using value_type = double;
    using global_index_type = long;
    using local_index_type = int;

    PoissonMatrixGeneration()
        : comm(MPI_COMM_WORLD), ref{gko::ReferenceExecutor::create()}
    {}

    comm_t comm;
    std::shared_ptr<const gko::ReferenceExecutor> ref;

    static constexpr double stretch_ratio = 0.95;
    static constexpr int grid_size = 3;
    static constexpr int n = 27;

    auto generate()
    {
        ex_dist_amp::Config cfg;
        cfg.nx = grid_size;
        cfg.ny = grid_size;
        cfg.nz = grid_size;
        cfg.mesh_stretch_ratio = stretch_ratio;
        return ex_dist_amp::generate_problem_data<value_type, global_index_type,
                                                  local_index_type>(comm, cfg);
    }

    /// Compute stretched mesh coordinate for index i in a direction with
    /// n_pts points and stretch ratio q.
    static double mesh_coord(const int i, const int n_pts, const double q)
    {
        if (i == 0) {
            return 0.0;
        }
        const double h0 = (q - 1.0) / (std::pow(q, n_pts - 1) - 1.0);
        return h0 * (std::pow(q, i) - 1.0) / (q - 1.0);
    }

    /// Get physical coordinates for a grid point given its (i,j,k) indices.
    static std::array<double, 3> point_coords(const std::array<int, 3>& ijk)
    {
        return {mesh_coord(ijk[0], grid_size, stretch_ratio),
                mesh_coord(ijk[1], grid_size, stretch_ratio),
                mesh_coord(ijk[2], grid_size, stretch_ratio)};
    }
};


TEST_F(PoissonMatrixGeneration, MatrixDimensionsAreCorrect)
{
    const auto problem = generate();

    EXPECT_EQ(problem.mat_data.size[0], n);
    EXPECT_EQ(problem.mat_data.size[1], n);
}


TEST_F(PoissonMatrixGeneration, ColorPointersAreCorrect)
{
    const auto problem = generate();

    // 8 colors -> 9 pointers
    ASSERT_EQ(problem.color_ptrs.size(), 9u);
    EXPECT_EQ(problem.color_ptrs[0], 0);
    EXPECT_EQ(problem.color_ptrs[8], n);

    // For 3x3x3: color sizes depend on parity counts.
    // Even count in each dir: ceil(3/2)=2, odd: floor(3/2)=1
    // color (e,e,e): 2*2*2=8, (o,e,e): 1*2*2=4, (e,o,e): 2*1*2=4,
    // (o,o,e): 1*1*2=2, (e,e,o): 2*2*1=4, (o,e,o): 1*2*1=2,
    // (e,o,o): 2*1*1=2, (o,o,o): 1*1*1=1
    // Total = 8+4+4+2+4+2+2+1 = 27
    const std::vector<int> expected_sizes{8, 4, 4, 2, 4, 2, 2, 1};
    for (int c = 0; c < 8; ++c) {
        EXPECT_EQ(problem.color_ptrs[c + 1] - problem.color_ptrs[c],
                  expected_sizes[c])
            << "Color " << c;
    }
}


TEST_F(PoissonMatrixGeneration, BoundaryRowsAreIdentity)
{
    const auto problem = generate();

    // Count nonzeros per row
    std::map<global_index_type,
             std::vector<std::pair<global_index_type, double>>>
        rows;
    for (const auto& nz : problem.mat_data.nonzeros) {
        rows[nz.row].emplace_back(nz.column, nz.value);
    }

    // All rows except the interior point (1,1,1) should be identity rows.
    // The interior point has multicolor index = color_ptrs[7] (color 7, the
    // only (odd,odd,odd) point in a 3x3x3 grid).
    const int interior_new_idx = problem.color_ptrs[7];

    for (const auto& [row, entries] : rows) {
        if (row == interior_new_idx) {
            continue;
        }
        ASSERT_EQ(entries.size(), 1u)
            << "Boundary row " << row << " should have 1 nonzero";
        EXPECT_EQ(entries[0].first, row)
            << "Boundary row " << row << " nonzero should be on diagonal";
        EXPECT_DOUBLE_EQ(entries[0].second, 1.0)
            << "Boundary row " << row << " diagonal should be 1.0";
    }
}


TEST_F(PoissonMatrixGeneration, InteriorRowHas7Nonzeros)
{
    const auto problem = generate();

    const int interior_new_idx = problem.color_ptrs[7];
    int count = 0;
    for (const auto& nz : problem.mat_data.nonzeros) {
        if (nz.row == interior_new_idx) {
            ++count;
        }
    }
    EXPECT_EQ(count, 7);
}


TEST_F(PoissonMatrixGeneration, TotalNonzerosAreCorrect)
{
    const auto problem = generate();

    // 26 boundary rows with 1 nnz each + 1 interior row with 7 nnz
    EXPECT_EQ(problem.mat_data.nonzeros.size(), 33u);
}


TEST_F(PoissonMatrixGeneration, InteriorRowValuesAreCorrect)
{
    const auto problem = generate();

    const int interior_new_idx = problem.color_ptrs[7];

    // Collect nonzeros for the interior row
    std::map<global_index_type, double> interior_row;
    for (const auto& nz : problem.mat_data.nonzeros) {
        if (nz.row == interior_new_idx) {
            interior_row[nz.column] = nz.value;
        }
    }

    ASSERT_EQ(interior_row.size(), 7u);

    // Interior point (1,1,1) on the stretched mesh.
    // Each direction has 3 points with stretch ratio q = 0.95.
    //   h0 = (q-1)/(q^2-1) = 20/39
    //   x0 = 0, x1 = 20/39, x2 = 1.0
    //   dr_plus  = x2 - x1 = 19/39
    //   dr_minus = x1 - x0 = 20/39
    //
    // Per direction:
    //   diagonal contrib   = 2/(dr+ + dr-) * (1/dr+ + 1/dr-)
    //                       = 2 * (39/19 + 39/20) = 2 * 39 * 39 / (19*20)
    //   off-diag (+dir)    = -2/(dr+ + dr-) * (1/dr+) = -2 * 39/19
    //   off-diag (-dir)    = -2/(dr+ + dr-) * (1/dr-) = -2 * 39/20
    const double q = stretch_ratio;
    const double h0 = (q - 1.0) / (q * q - 1.0);
    const double dr_plus = 1.0 - h0;  // 19/39
    const double dr_minus = h0;       // 20/39

    const double diag_per_dir =
        2.0 / (dr_plus + dr_minus) * (1.0 / dr_plus + 1.0 / dr_minus);
    const double expected_diag = 3.0 * diag_per_dir;
    const double expected_offdiag_plus =
        -2.0 / (dr_plus + dr_minus) * (1.0 / dr_plus);
    const double expected_offdiag_minus =
        -2.0 / (dr_plus + dr_minus) * (1.0 / dr_minus);

    // Check diagonal
    ASSERT_TRUE(interior_row.count(interior_new_idx));
    EXPECT_NEAR(interior_row[interior_new_idx], expected_diag, 1e-12)
        << "Diagonal value mismatch";

    // Check off-diagonals: there should be 3 pairs (one per direction),
    // each pair having one +dir and one -dir neighbor.
    double offdiag_sum = 0.0;
    int plus_count = 0;
    int minus_count = 0;
    for (const auto& [col, val] : interior_row) {
        if (col == interior_new_idx) {
            continue;
        }
        offdiag_sum += val;
        if (std::abs(val - expected_offdiag_plus) < 1e-12) {
            ++plus_count;
        } else if (std::abs(val - expected_offdiag_minus) < 1e-12) {
            ++minus_count;
        } else {
            ADD_FAILURE() << "Unexpected off-diagonal value " << val
                          << " at column " << col;
        }
    }
    EXPECT_EQ(plus_count, 3) << "Expected 3 off-diag entries with +dir value";
    EXPECT_EQ(minus_count, 3) << "Expected 3 off-diag entries with -dir value";

    // Row sum should equal zero (standard FD Laplacian property)
    const double row_sum = expected_diag + offdiag_sum;
    EXPECT_NEAR(row_sum, 0.0, 1e-12) << "Interior row should sum to zero";
}


TEST_F(PoissonMatrixGeneration, BoundaryRhsIsExactSolution)
{
    const auto problem = generate();

    const ex_dist_amp::PoissonPDE<double> pde;
    const int interior_new_idx = problem.color_ptrs[7];

    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{grid_size, grid_size, grid_size});

    for (int new_idx = 0; new_idx < n; ++new_idx) {
        if (new_idx == interior_new_idx) {
            continue;
        }
        const int old_idx = ordering.new_to_old[new_idx];
        const std::array<int, 3> ijk =
            ex_dist_amp::get_natural_3d_indices_from_flat(
                std::array<int, 3>{grid_size, grid_size, grid_size}, old_idx);
        const auto r = point_coords(ijk);

        EXPECT_NEAR(problem.local_rhs[new_idx], pde.get_solution(r), 1e-12)
            << "RHS mismatch at boundary point (" << ijk[0] << "," << ijk[1]
            << "," << ijk[2] << "), new_idx=" << new_idx;
    }
}


TEST_F(PoissonMatrixGeneration, InteriorRhsIsForcingFunction)
{
    const auto problem = generate();

    const ex_dist_amp::PoissonPDE<double> pde;
    const int interior_new_idx = problem.color_ptrs[7];

    // The only interior point is (1,1,1).
    const auto r = point_coords({1, 1, 1});

    EXPECT_NEAR(problem.local_rhs[interior_new_idx],
                pde.get_forcing_function(r), 1e-12)
        << "Interior RHS should equal the forcing function at (1,1,1)";
}


TEST_F(PoissonMatrixGeneration, BoundaryExactSolutionMatchesPDE)
{
    const auto problem = generate();

    const ex_dist_amp::PoissonPDE<double> pde;
    const int interior_new_idx = problem.color_ptrs[7];

    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{grid_size, grid_size, grid_size});

    for (int new_idx = 0; new_idx < n; ++new_idx) {
        if (new_idx == interior_new_idx) {
            continue;
        }
        const int old_idx = ordering.new_to_old[new_idx];
        const std::array<int, 3> ijk =
            ex_dist_amp::get_natural_3d_indices_from_flat(
                std::array<int, 3>{grid_size, grid_size, grid_size}, old_idx);
        const auto r = point_coords(ijk);

        EXPECT_NEAR(problem.local_exact_solution[new_idx], pde.get_solution(r),
                    1e-12)
            << "Exact solution mismatch at boundary point (" << ijk[0] << ","
            << ijk[1] << "," << ijk[2] << "), new_idx=" << new_idx;
    }
}


TEST_F(PoissonMatrixGeneration, InteriorExactSolutionMatchesPDE)
{
    const auto problem = generate();

    const ex_dist_amp::PoissonPDE<double> pde;
    const int interior_new_idx = problem.color_ptrs[7];

    // The only interior point is (1,1,1).
    const auto r = point_coords({1, 1, 1});

    EXPECT_NEAR(problem.local_exact_solution[interior_new_idx],
                pde.get_solution(r), 1e-12)
        << "Interior exact solution should equal PDE solution at (1,1,1)";
}


TEST_F(PoissonMatrixGeneration, RhsSizeMatchesLocalProblemSize)
{
    const auto problem = generate();

    EXPECT_EQ(problem.local_rhs.size(), n);
    EXPECT_EQ(problem.local_exact_solution.size(), n);
}


TEST(PoissonUniformMesh, InteriorStencilIsUniform)
{
    // stretch_ratio = 1.0 gives a uniform mesh with spacing h = 1/(n-1).
    // For 3x3x3 on 1 rank: h = 0.5 in all directions.
    //   diagonal = 3 * 2/(h+h) * (1/h + 1/h) = 3 * (1/h) * (2/h) = 6/h^2 = 24
    //   each off-diagonal = -2/(h*(h+h)) = -1/h^2 = -4
    comm_t comm(MPI_COMM_WORLD);
    ex_dist_amp::Config cfg;
    cfg.nx = 3;
    cfg.ny = 3;
    cfg.nz = 3;
    cfg.mesh_stretch_ratio = 1.0;
    const auto problem =
        ex_dist_amp::generate_problem_data<double, long, int>(comm, cfg);

    // Only interior point is (1,1,1) -> color 7, index = color_ptrs[7].
    const int interior_new_idx = problem.color_ptrs[7];

    std::map<long, double> interior_row;
    for (const auto& nz : problem.mat_data.nonzeros) {
        if (nz.row == interior_new_idx) {
            interior_row[nz.column] = nz.value;
        }
    }

    ASSERT_EQ(interior_row.size(), 7u);
    EXPECT_NEAR(interior_row[interior_new_idx], 24.0, 1e-12);

    for (const auto& [col, val] : interior_row) {
        if (col == interior_new_idx) {
            continue;
        }
        EXPECT_NEAR(val, -4.0, 1e-12) << "Off-diagonal at col " << col;
    }
}


}  // namespace
