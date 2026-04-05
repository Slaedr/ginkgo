// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <map>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include "poisson_amp.hpp"

namespace {


using comm_t = gko::experimental::mpi::communicator;


class PoissonMatrixGeneration4Rank : public ::testing::Test {
public:
    using value_type = double;
    using global_index_type = long;
    using local_index_type = int;

    PoissonMatrixGeneration4Rank()
        : comm(MPI_COMM_WORLD),
          ref{gko::ReferenceExecutor::create()},
          my_ranks{get_my_ranks()}
    {
        EXPECT_EQ(comm.size(), 4);
    }

    comm_t comm;
    std::shared_ptr<const gko::ReferenceExecutor> ref;

    static constexpr double stretch_ratio = 0.95;
    static constexpr int local_size = 3;
    static constexpr int local_n = 27;
    // cubic_radical_search(4) = {2, 1, 2}
    static constexpr std::array<int, 3> proc_dims{2, 1, 2};
    static constexpr std::array<long, 3> global_dims{6, 3, 6};
    static constexpr long global_n = 108;

    std::array<int, 3> my_ranks;

    auto generate()
    {
        ex_dist_amp::Config cfg;
        cfg.nx = local_size;
        cfg.ny = local_size;
        cfg.nz = local_size;
        cfg.mesh_stretch_ratio = stretch_ratio;
        return ex_dist_amp::generate_problem_data<value_type, global_index_type,
                                                  local_index_type>(comm, cfg);
    }

    /// Compute stretched mesh coordinate for a global index in a direction
    /// with n_global_pts total global points and stretch ratio q.
    static double mesh_coord(const int global_i, const int n_global_pts,
                             const double q)
    {
        if (global_i == 0) {
            return 0.0;
        }
        const double h0 = (q - 1.0) / (std::pow(q, n_global_pts - 1) - 1.0);
        return h0 * (std::pow(q, global_i) - 1.0) / (q - 1.0);
    }

    /// Get physical coordinates for a point given its global (i,j,k) indices.
    static std::array<double, 3> point_coords(
        const std::array<long, 3>& global_ijk)
    {
        return {mesh_coord(global_ijk[0], global_dims[0], stretch_ratio),
                mesh_coord(global_ijk[1], global_dims[1], stretch_ratio),
                mesh_coord(global_ijk[2], global_dims[2], stretch_ratio)};
    }

    /// Convert local (i,j,k) to global (i,j,k) for this rank.
    std::array<long, 3> to_global(const std::array<int, 3>& local_ijk) const
    {
        return {static_cast<long>(my_ranks[0]) * local_size + local_ijk[0],
                static_cast<long>(my_ranks[1]) * local_size + local_ijk[1],
                static_cast<long>(my_ranks[2]) * local_size + local_ijk[2]};
    }

    static bool is_global_boundary(const std::array<long, 3>& gijk)
    {
        for (int d = 0; d < 3; d++) {
            if (gijk[d] == 0 || gijk[d] == global_dims[d] - 1) {
                return true;
            }
        }
        return false;
    }

    /// Expected diagonal value at an interior point with global index gijk.
    /// Sum over directions of  2 / (h+ + h-) * (1/h+ + 1/h-).
    static double expected_diagonal(const std::array<long, 3>& gijk)
    {
        double val = 0.0;
        for (int d = 0; d < 3; d++) {
            const double ri =
                mesh_coord(gijk[d], global_dims[d], stretch_ratio);
            const double hp =
                mesh_coord(gijk[d] + 1, global_dims[d], stretch_ratio) - ri;
            const double hm =
                ri - mesh_coord(gijk[d] - 1, global_dims[d], stretch_ratio);
            val += 2.0 / (hp + hm) * (1.0 / hp + 1.0 / hm);
        }
        return val;
    }

    /// Expected off-diagonal value for the neighbor at gijk +/- 1 in direction
    /// dir.  sign = +1 for the forward neighbor, -1 for the backward neighbor.
    static double expected_offdiagonal(const std::array<long, 3>& gijk,
                                       const int dir, const int sign)
    {
        const double ri =
            mesh_coord(gijk[dir], global_dims[dir], stretch_ratio);
        const double hp =
            mesh_coord(gijk[dir] + 1, global_dims[dir], stretch_ratio) - ri;
        const double hm =
            ri - mesh_coord(gijk[dir] - 1, global_dims[dir], stretch_ratio);
        const double h = (sign > 0) ? hp : hm;
        return -2.0 / (h * (hp + hm));
    }

    /// Compute the expected global column index for a grid point at
    /// global 3D index nbd_gijk, given the multicolor ordering.
    static global_index_type expected_global_col(
        const std::array<long, 3>& nbd_gijk,
        const ex_dist_amp::MulticolorOrdering& ordering)
    {
        const std::array<int, 3> ldims{local_size, local_size, local_size};
        // Determine owning rank and local 3D index.
        const std::array<int, 3> nbd_rank{
            static_cast<int>(nbd_gijk[0] / local_size),
            static_cast<int>(nbd_gijk[1] / local_size),
            static_cast<int>(nbd_gijk[2] / local_size)};
        const std::array<int, 3> nbd_local{
            static_cast<int>(nbd_gijk[0] % local_size),
            static_cast<int>(nbd_gijk[1] % local_size),
            static_cast<int>(nbd_gijk[2] % local_size)};
        // Flat natural index -> multicolor index.
        const int nbd_flat = nbd_local[2] * ldims[1] * ldims[0] +
                             nbd_local[1] * ldims[0] + nbd_local[0];
        const int nbd_new = ordering.old_to_new[nbd_flat];
        // Flat rank -> global offset.
        const int flat_rank = nbd_rank[2] * proc_dims[1] * proc_dims[0] +
                              nbd_rank[1] * proc_dims[0] + nbd_rank[0];
        return static_cast<global_index_type>(flat_rank) * local_n + nbd_new;
    }

    /// Count how many local points are interior (not on any global boundary).
    int count_local_interior() const
    {
        int count = 0;
        for (int k = 0; k < local_size; ++k)
            for (int j = 0; j < local_size; ++j)
                for (int i = 0; i < local_size; ++i)
                    if (!is_global_boundary(to_global({i, j, k}))) ++count;
        return count;
    }

private:
    std::array<int, 3> get_my_ranks() const
    {
        return ex_dist_amp::get_natural_3d_indices_from_flat(proc_dims,
                                                             comm.rank());
    }
};


TEST_F(PoissonMatrixGeneration4Rank, MatrixDimensionsAreCorrect)
{
    const auto problem = generate();

    EXPECT_EQ(problem.mat_data.size[0], global_n);
    EXPECT_EQ(problem.mat_data.size[1], global_n);
}


TEST_F(PoissonMatrixGeneration4Rank, ColorPointersMatchLocalGrid)
{
    const auto problem = generate();

    // Local grid is 3x3x3, so color structure is the same on every rank.
    ASSERT_EQ(problem.color_ptrs.size(), 9u);
    EXPECT_EQ(problem.color_ptrs[0], 0);
    EXPECT_EQ(problem.color_ptrs[8], local_n);

    const std::vector<int> expected_sizes{8, 4, 4, 2, 4, 2, 2, 1};
    for (int c = 0; c < 8; ++c) {
        EXPECT_EQ(problem.color_ptrs[c + 1] - problem.color_ptrs[c],
                  expected_sizes[c])
            << "Color " << c;
    }
}


TEST_F(PoissonMatrixGeneration4Rank, VectorSizesMatchLocalProblemSize)
{
    const auto problem = generate();

    EXPECT_EQ(problem.local_rhs.size(), local_n);
    EXPECT_EQ(problem.local_exact_solution.size(), local_n);
}


TEST_F(PoissonMatrixGeneration4Rank, BoundaryRowsAreIdentity)
{
    const auto problem = generate();
    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{local_size, local_size, local_size});

    std::map<global_index_type,
             std::vector<std::pair<global_index_type, double>>>
        rows;
    for (const auto& nz : problem.mat_data.nonzeros) {
        rows[nz.row].emplace_back(nz.column, nz.value);
    }

    for (const auto& [row, entries] : rows) {
        // Map this global row back to local new_idx, then to local 3D,
        // then to global 3D to check if it is a boundary.
        const int new_idx = row % local_n;
        const int old_idx = ordering.new_to_old[new_idx];
        const auto local_ijk = ex_dist_amp::get_natural_3d_indices_from_flat(
            std::array<int, 3>{local_size, local_size, local_size}, old_idx);
        const auto global_ijk = to_global(local_ijk);

        if (!is_global_boundary(global_ijk)) {
            continue;
        }
        ASSERT_EQ(entries.size(), 1u)
            << "Boundary row " << row << " (global point " << global_ijk[0]
            << "," << global_ijk[1] << "," << global_ijk[2]
            << ") should have 1 nonzero";
        EXPECT_EQ(entries[0].first, row);
        EXPECT_DOUBLE_EQ(entries[0].second, 1.0);
    }
}


TEST_F(PoissonMatrixGeneration4Rank, InteriorRowsHave7Nonzeros)
{
    const auto problem = generate();
    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{local_size, local_size, local_size});

    std::map<global_index_type, int> nnz_per_row;
    for (const auto& nz : problem.mat_data.nonzeros) {
        nnz_per_row[nz.row]++;
    }

    for (const auto& [row, count] : nnz_per_row) {
        const int new_idx = row % local_n;
        const int old_idx = ordering.new_to_old[new_idx];
        const auto local_ijk = ex_dist_amp::get_natural_3d_indices_from_flat(
            std::array<int, 3>{local_size, local_size, local_size}, old_idx);
        const auto global_ijk = to_global(local_ijk);

        if (is_global_boundary(global_ijk)) {
            continue;
        }
        EXPECT_EQ(count, 7)
            << "Interior row " << row << " (global point " << global_ijk[0]
            << "," << global_ijk[1] << "," << global_ijk[2] << ")";
    }
}


TEST_F(PoissonMatrixGeneration4Rank, TotalNonzerosAreCorrect)
{
    const auto problem = generate();

    const int n_interior = count_local_interior();
    const int n_boundary = local_n - n_interior;
    const auto expected_nnz =
        static_cast<std::size_t>(n_boundary) + 7 * n_interior;
    EXPECT_EQ(problem.mat_data.nonzeros.size(), expected_nnz)
        << "rank " << comm.rank() << ": " << n_interior << " interior, "
        << n_boundary << " boundary";
}


TEST_F(PoissonMatrixGeneration4Rank, InteriorRowsSumToZero)
{
    const auto problem = generate();
    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{local_size, local_size, local_size});

    std::map<global_index_type, double> row_sums;
    std::map<global_index_type, bool> row_is_interior;
    for (const auto& nz : problem.mat_data.nonzeros) {
        row_sums[nz.row] += nz.value;
    }

    for (const auto& [row, sum] : row_sums) {
        const int new_idx = row % local_n;
        const int old_idx = ordering.new_to_old[new_idx];
        const auto local_ijk = ex_dist_amp::get_natural_3d_indices_from_flat(
            std::array<int, 3>{local_size, local_size, local_size}, old_idx);
        const auto global_ijk = to_global(local_ijk);

        if (is_global_boundary(global_ijk)) {
            continue;
        }
        EXPECT_NEAR(sum, 0.0, 1e-12)
            << "Interior row " << row << " (global point " << global_ijk[0]
            << "," << global_ijk[1] << "," << global_ijk[2]
            << ") should sum to zero";
    }
}


TEST_F(PoissonMatrixGeneration4Rank, InteriorStencilValuesAreCorrect)
{
    const auto problem = generate();
    const std::array<int, 3> ldims{local_size, local_size, local_size};
    const auto ordering = ex_dist_amp::compute_multicolor_ordering(ldims);

    // Collect nonzeros per row.
    std::map<global_index_type, std::map<global_index_type, double>> rows;
    for (const auto& nz : problem.mat_data.nonzeros) {
        rows[nz.row][nz.column] = nz.value;
    }

    // Build map: global row index -> global 3D index for this rank's rows.
    const int flat_rank = my_ranks[2] * proc_dims[1] * proc_dims[0] +
                          my_ranks[1] * proc_dims[0] + my_ranks[0];
    const global_index_type rank_offset =
        static_cast<global_index_type>(flat_rank) * local_n;

    for (const auto& [row, cols] : rows) {
        const int new_idx = static_cast<int>(row - rank_offset);
        const int old_idx = ordering.new_to_old[new_idx];
        const auto local_ijk =
            ex_dist_amp::get_natural_3d_indices_from_flat(ldims, old_idx);
        const auto gijk = to_global(local_ijk);

        if (is_global_boundary(gijk)) {
            continue;
        }

        // Check diagonal
        ASSERT_TRUE(cols.count(row));
        EXPECT_NEAR(cols.at(row), expected_diagonal(gijk), 1e-12)
            << "Diagonal at global (" << gijk[0] << "," << gijk[1] << ","
            << gijk[2] << ")";

        // Check each of the 6 off-diagonal entries: column index and value.
        const std::array<int, 2> signs{-1, +1};
        int offdiag_checked = 0;
        for (int d = 0; d < 3; d++) {
            for (const int s : signs) {
                std::array<long, 3> nbd_gijk = gijk;
                nbd_gijk[d] += s;
                const auto exp_col = expected_global_col(nbd_gijk, ordering);
                const auto exp_val = expected_offdiagonal(gijk, d, s);

                ASSERT_TRUE(cols.count(exp_col))
                    << "Missing neighbor at global (" << nbd_gijk[0] << ","
                    << nbd_gijk[1] << "," << nbd_gijk[2] << "), expected col "
                    << exp_col << " from row " << row;
                EXPECT_NEAR(cols.at(exp_col), exp_val, 1e-12)
                    << "Value mismatch for neighbor at global (" << nbd_gijk[0]
                    << "," << nbd_gijk[1] << "," << nbd_gijk[2] << ")";
                ++offdiag_checked;
            }
        }
        EXPECT_EQ(offdiag_checked, 6);

        // Verify no extra entries beyond diagonal + 6 off-diags.
        EXPECT_EQ(cols.size(), 7u)
            << "Interior row at global (" << gijk[0] << "," << gijk[1] << ","
            << gijk[2] << ") should have exactly 7 nonzeros";
    }
}


TEST_F(PoissonMatrixGeneration4Rank, BoundaryRhsIsExactSolution)
{
    const auto problem = generate();
    const ex_dist_amp::PoissonPDE<double> pde;
    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{local_size, local_size, local_size});

    for (int new_idx = 0; new_idx < local_n; ++new_idx) {
        const int old_idx = ordering.new_to_old[new_idx];
        const auto local_ijk = ex_dist_amp::get_natural_3d_indices_from_flat(
            std::array<int, 3>{local_size, local_size, local_size}, old_idx);
        const auto global_ijk = to_global(local_ijk);

        if (!is_global_boundary(global_ijk)) {
            continue;
        }
        const auto r = point_coords(global_ijk);
        EXPECT_NEAR(problem.local_rhs[new_idx], pde.get_solution(r), 1e-12)
            << "Boundary RHS at global (" << global_ijk[0] << ","
            << global_ijk[1] << "," << global_ijk[2] << ")";
    }
}


TEST_F(PoissonMatrixGeneration4Rank, InteriorRhsIsForcingFunction)
{
    const auto problem = generate();
    const ex_dist_amp::PoissonPDE<double> pde;
    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{local_size, local_size, local_size});

    for (int new_idx = 0; new_idx < local_n; ++new_idx) {
        const int old_idx = ordering.new_to_old[new_idx];
        const auto local_ijk = ex_dist_amp::get_natural_3d_indices_from_flat(
            std::array<int, 3>{local_size, local_size, local_size}, old_idx);
        const auto global_ijk = to_global(local_ijk);

        if (is_global_boundary(global_ijk)) {
            continue;
        }
        const auto r = point_coords(global_ijk);
        EXPECT_NEAR(problem.local_rhs[new_idx], pde.get_forcing_function(r),
                    1e-12)
            << "Interior RHS at global (" << global_ijk[0] << ","
            << global_ijk[1] << "," << global_ijk[2] << ")";
    }
}


TEST_F(PoissonMatrixGeneration4Rank, ExactSolutionMatchesPDE)
{
    const auto problem = generate();
    const ex_dist_amp::PoissonPDE<double> pde;
    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{local_size, local_size, local_size});

    for (int new_idx = 0; new_idx < local_n; ++new_idx) {
        const int old_idx = ordering.new_to_old[new_idx];
        const auto local_ijk = ex_dist_amp::get_natural_3d_indices_from_flat(
            std::array<int, 3>{local_size, local_size, local_size}, old_idx);
        const auto global_ijk = to_global(local_ijk);
        const auto r = point_coords(global_ijk);

        EXPECT_NEAR(problem.local_exact_solution[new_idx], pde.get_solution(r),
                    1e-12)
            << "Exact solution at global (" << global_ijk[0] << ","
            << global_ijk[1] << "," << global_ijk[2]
            << "), boundary=" << is_global_boundary(global_ijk);
    }
}


TEST_F(PoissonMatrixGeneration4Rank, LocalInteriorPointCountIsCorrect)
{
    // Verify our test helper against the actual matrix.
    const auto problem = generate();
    const auto ordering = ex_dist_amp::compute_multicolor_ordering(
        std::array<int, 3>{local_size, local_size, local_size});

    int actual_interior = 0;
    std::map<global_index_type, int> nnz_per_row;
    for (const auto& nz : problem.mat_data.nonzeros) {
        nnz_per_row[nz.row]++;
    }
    for (const auto& [row, count] : nnz_per_row) {
        if (count > 1) {
            ++actual_interior;
        }
    }

    EXPECT_EQ(actual_interior, count_local_interior())
        << "rank " << comm.rank();
}


}  // namespace
