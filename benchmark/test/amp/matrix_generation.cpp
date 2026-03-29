// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "benchmark/amp/matrix_generation.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include "test/utils/executor.hpp"

namespace {


TEST(Helpers, 3dFromFlatIsCorrect)
{
    const int flat = 33;
    const std::array<int, 3> dims{4, 4, 4};
    const std::array<int, 3> expected{1, 0, 2};

    const std::array<int, 3> result =
        get_natural_3d_indices_from_flat(dims, flat);

    ASSERT_EQ(result, expected);
}

TEST(Helpers, 3dToFlatIsCorrect)
{
    const int flat = 33;
    const std::array<int, 3> dims{4, 4, 4};
    const std::array<int, 3> coords{1, 0, 2};

    const auto result = get_natural_flat_index_from_3d(dims, coords);

    ASSERT_EQ(result, flat);
}

// Off-diagonal generator that always returns -1 (constant-coefficient stencil).
struct ConstOffdiag {
    double operator()() { return -1.0; }
};


class MatrixGeneration : public ::testing::Test {
public:
    using value_type = double;
    MatrixGeneration()
        : comm(MPI_COMM_WORLD),
          ref{gko::ReferenceExecutor::create()},
          exec{gko::OmpExecutor::create()}
    {
        guard = exec->get_scoped_device_id_guard();
    }

    void TearDown() final
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    gko::experimental::mpi::communicator comm;

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> exec;
    gko::scoped_device_id_guard guard;
};


// Helper: compute the expected set of (row, col) pairs for a 27-pt stencil
// with multicolor ordering on a single rank (comm.size() == 1).
//
// For 1 rank, global == local and proc_dims = {1,1,1}, so
//   global_flat_idx == local_flat_idx (the multicolor-ordered index).
//
// The stencil connects each grid point (ix,iy,iz) to all (ix+di,iy+dj,iz+dk)
// with di,dj,dk in {-1,0,1}, provided the neighbor is inside the grid.
std::set<std::pair<long, long>> expected_nonzero_locations(
    const std::array<int, 3>& dims, const MulticolorOrdering& ordering)
{
    const int n = dims[0] * dims[1] * dims[2];
    std::set<std::pair<long, long>> locs;

    for (int new_row = 0; new_row < n; ++new_row) {
        const int old_row = ordering.new_to_old[new_row];
        const auto old_idx = get_natural_3d_indices_from_flat(dims, old_row);

        for (int dk = -1; dk <= 1; ++dk) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    const std::array<int, 3> nbd{
                        old_idx[0] + di, old_idx[1] + dj, old_idx[2] + dk};
                    if (!is_valid_point(nbd, dims)) {
                        continue;
                    }
                    const int nbd_flat =
                        get_natural_flat_index_from_3d(dims, nbd);
                    const int new_col = ordering.old_to_new[nbd_flat];
                    locs.emplace(static_cast<long>(new_row),
                                 static_cast<long>(new_col));
                }
            }
        }
    }
    return locs;
}

// Hard-coded nonzero locations for a 3x3x3 grid with 27-pt stencil
// and 8-color multicolor ordering, on a single rank.
// 8 corners have 8 nnz, 12 edges have 12 nnz, 6 faces have 18 nnz,
// 1 interior has 27 nnz.  Total = 8*8 + 12*12 + 6*18 + 27 = 343.
std::set<std::pair<long, long>> nonzero_locations_3x3()
{
    std::set<std::pair<long, long>> locs;
    auto row = [&](long r, std::initializer_list<long> cols) {
        for (const long c : cols) {
            locs.emplace(r, c);
        }
    };
    // Color 0: 8 corners (8 nnz each)
    row(0, {0, 8, 12, 16, 18, 22, 24, 26});   // (0,0,0)
    row(1, {1, 8, 13, 16, 19, 22, 25, 26});   // (2,0,0)
    row(2, {2, 9, 12, 16, 20, 23, 24, 26});   // (0,2,0)
    row(3, {3, 9, 13, 16, 21, 23, 25, 26});   // (2,2,0)
    row(4, {4, 10, 14, 17, 18, 22, 24, 26});  // (0,0,2)
    row(5, {5, 10, 15, 17, 19, 22, 25, 26});  // (2,0,2)
    row(6, {6, 11, 14, 17, 20, 23, 24, 26});  // (0,2,2)
    row(7, {7, 11, 15, 17, 21, 23, 25, 26});  // (2,2,2)
    // Color 1: 4 x-edges (12 nnz each)
    row(8, {0, 1, 8, 12, 13, 16, 18, 19, 22, 24, 25, 26});    // (1,0,0)
    row(9, {2, 3, 9, 12, 13, 16, 20, 21, 23, 24, 25, 26});    // (1,2,0)
    row(10, {4, 5, 10, 14, 15, 17, 18, 19, 22, 24, 25, 26});  // (1,0,2)
    row(11, {6, 7, 11, 14, 15, 17, 20, 21, 23, 24, 25, 26});  // (1,2,2)
    // Color 2: 4 y-edges (12 nnz each)
    row(12, {0, 2, 8, 9, 12, 16, 18, 20, 22, 23, 24, 26});    // (0,1,0)
    row(13, {1, 3, 8, 9, 13, 16, 19, 21, 22, 23, 25, 26});    // (2,1,0)
    row(14, {4, 6, 10, 11, 14, 17, 18, 20, 22, 23, 24, 26});  // (0,1,2)
    row(15, {5, 7, 10, 11, 15, 17, 19, 21, 22, 23, 25, 26});  // (2,1,2)
    // Color 3: 2 xy-face centers (18 nnz each)
    row(16, {0, 1, 2, 3, 8, 9, 12, 13, 16, 18, 19, 20, 21, 22, 23, 24, 25,
             26});  // (1,1,0)
    row(17, {4, 5, 6, 7, 10, 11, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25,
             26});  // (1,1,2)
    // Color 4: 4 z-edges (12 nnz each)
    row(18, {0, 4, 8, 10, 12, 14, 16, 17, 18, 22, 24, 26});  // (0,0,1)
    row(19, {1, 5, 8, 10, 13, 15, 16, 17, 19, 22, 25, 26});  // (2,0,1)
    row(20, {2, 6, 9, 11, 12, 14, 16, 17, 20, 23, 24, 26});  // (0,2,1)
    row(21, {3, 7, 9, 11, 13, 15, 16, 17, 21, 23, 25, 26});  // (2,2,1)
    // Color 5: 2 xz-face centers (18 nnz each)
    row(22, {0, 1, 4, 5, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 25,
             26});  // (1,0,1)
    row(23, {2, 3, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 20, 21, 23, 24, 25,
             26});  // (1,2,1)
    // Color 6: 2 yz-face centers (18 nnz each)
    row(24, {0, 2, 4, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 20, 22, 23, 24,
             26});  // (0,1,1)
    row(25, {1, 3, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 19, 21, 22, 23, 25,
             26});  // (2,1,1)
    // Color 7: 1 interior center (27 nnz)
    row(26, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});  // (1,1,1)
    return locs;
}


// ---- Tests ----


TEST_F(MatrixGeneration, SmallGridNonzeroLocationsMatch)
{
    // Use a 3x3x3 grid: interior point (1,1,1) has 27 neighbors,
    // corners have 8, edges/faces have intermediate counts.
    const std::array<int, 3> dims{3, 3, 3};
    const auto ordering = compute_multicolor_ordering(dims);
    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, long, int>(comm, dims, gen, ordering);

    const auto expected = nonzero_locations_3x3();

    // Collect actual (row, col) pairs
    std::set<std::pair<long, long>> actual;
    for (const auto& nz : data.nonzeros) {
        actual.emplace(nz.row, nz.column);
    }

    EXPECT_EQ(actual, expected);
}


TEST_F(MatrixGeneration, DiagonalValuesAre26)
{
    const std::array<int, 3> dims{3, 3, 3};
    const auto ordering = compute_multicolor_ordering(dims);
    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, long, int>(comm, dims, gen, ordering);

    for (const auto& nz : data.nonzeros) {
        if (nz.row == nz.column) {
            EXPECT_DOUBLE_EQ(nz.value, 26.0)
                << "Diagonal at row " << nz.row << " should be 26.0";
        }
    }
}


TEST_F(MatrixGeneration, OffDiagonalValuesAreFromGenerator)
{
    const std::array<int, 3> dims{3, 3, 3};
    const auto ordering = compute_multicolor_ordering(dims);
    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, long, int>(comm, dims, gen, ordering);

    for (const auto& nz : data.nonzeros) {
        if (nz.row != nz.column) {
            EXPECT_DOUBLE_EQ(nz.value, -1.0)
                << "Off-diagonal at (" << nz.row << ", " << nz.column
                << ") should be -1.0";
        }
    }
}


TEST_F(MatrixGeneration, InteriorPointHas27Nonzeros)
{
    // 3x3x3 grid: (1,1,1) is the only interior point.
    const std::array<int, 3> dims{3, 3, 3};
    const auto ordering = compute_multicolor_ordering(dims);
    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, long, int>(comm, dims, gen, ordering);

    // Find the multicolor index of the natural interior point (1,1,1).
    const int old_flat =
        get_natural_flat_index_from_3d(dims, std::array<int, 3>{1, 1, 1});
    const int new_idx = ordering.old_to_new[old_flat];

    int count = 0;
    for (const auto& nz : data.nonzeros) {
        if (nz.row == new_idx) {
            ++count;
        }
    }
    EXPECT_EQ(count, 27);
}


TEST_F(MatrixGeneration, CornerPointHas8Nonzeros)
{
    // Corner (0,0,0) has 8 valid neighbors (itself + 7 that lie inside).
    const std::array<int, 3> dims{3, 3, 3};
    const auto ordering = compute_multicolor_ordering(dims);
    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, long, int>(comm, dims, gen, ordering);

    const int old_flat =
        get_natural_flat_index_from_3d(dims, std::array<int, 3>{0, 0, 0});
    const int new_idx = ordering.old_to_new[old_flat];

    int count = 0;
    for (const auto& nz : data.nonzeros) {
        if (nz.row == new_idx) {
            ++count;
        }
    }
    EXPECT_EQ(count, 8);
}


TEST_F(MatrixGeneration, MatrixDimensionsAreCorrect)
{
    const std::array<int, 3> dims{4, 3, 2};
    const int n = dims[0] * dims[1] * dims[2];
    const auto ordering = compute_multicolor_ordering(dims);
    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, long, int>(comm, dims, gen, ordering);

    EXPECT_EQ(data.size[0], n);
    EXPECT_EQ(data.size[1], n);
}


TEST_F(MatrixGeneration, EachRowHasExactlyOneDiagonal)
{
    const std::array<int, 3> dims{3, 3, 3};
    const int n = 27;
    const auto ordering = compute_multicolor_ordering(dims);
    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, long, int>(comm, dims, gen, ordering);

    std::vector<int> diag_count(n, 0);
    for (const auto& nz : data.nonzeros) {
        if (nz.row == nz.column) {
            diag_count[nz.row]++;
        }
    }
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(diag_count[i], 1)
            << "Row " << i << " has " << diag_count[i] << " diagonal entries";
    }
}


TEST_F(MatrixGeneration, MulticolorOrderingColorPtrs)
{
    const std::array<int, 3> dims{4, 4, 4};
    const auto ordering = compute_multicolor_ordering(dims);

    // 8 colors, 9 pointers
    ASSERT_EQ(ordering.color_ptrs.size(), 9u);
    EXPECT_EQ(ordering.color_ptrs[0], 0);
    EXPECT_EQ(ordering.color_ptrs[8], 64);

    // Color pointers should be monotonically non-decreasing
    for (int c = 0; c < 8; ++c) {
        EXPECT_LE(ordering.color_ptrs[c], ordering.color_ptrs[c + 1]);
    }

    // For a 4x4x4 grid, each color has exactly 8 points (2*2*2)
    for (int c = 0; c < 8; ++c) {
        EXPECT_EQ(ordering.color_ptrs[c + 1] - ordering.color_ptrs[c], 8)
            << "Color " << c << " should have 8 points";
    }
}


TEST_F(MatrixGeneration, NonzeroLocationsWithNonCubicGrid)
{
    const std::array<int, 3> dims{4, 3, 2};
    const auto ordering = compute_multicolor_ordering(dims);
    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, long, int>(comm, dims, gen, ordering);

    const auto expected = expected_nonzero_locations(dims, ordering);

    std::set<std::pair<long, long>> actual;
    for (const auto& nz : data.nonzeros) {
        actual.emplace(nz.row, nz.column);
    }

    EXPECT_EQ(actual, expected);
}


TEST_F(MatrixGeneration, GenerateProblemDataConsistency)
{
    const std::array<int, 3> dims{3, 3, 3};
    ConstOffdiag gen;
    const auto problem =
        generate_problem_data<double, long, int>(comm, dims, gen);

    EXPECT_EQ(problem.color_ptrs.size(), 9u);
    EXPECT_EQ(problem.color_ptrs[8], 27);
    EXPECT_EQ(problem.mat_data.size[0], 27);
    EXPECT_EQ(problem.mat_data.size[1], 27);
}


}  // namespace
