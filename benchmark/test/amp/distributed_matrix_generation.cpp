// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/amp/matrix_generation.hpp"
#include "test/utils/executor.hpp"

namespace {


// Off-diagonal generator that always returns -1 (constant-coefficient stencil).
struct ConstOffdiag {
    double operator()() { return -1.0; }
};


class DistMatrixGeneration : public ::testing::Test {
public:
    using value_type = double;
    using Vec = gko::matrix::Dense<value_type>;
    using DistMtx =
        gko::experimental::distributed::Matrix<value_type, int32_t, int64_t>;
    using DistVec = gko::experimental::distributed::Vector<value_type>;
    using partition_t = gko::experimental::distributed::Partition<int32_t, int64_t>;

    DistMatrixGeneration()
        : comm(MPI_COMM_WORLD),
          ref{gko::ReferenceExecutor::create()},
          exec{gko::OmpExecutor::create()},
          partition{gko::share(partition_t::build_from_global_size_uniform(
              exec, comm.size(), global_n))},
          ordering{compute_multicolor_ordering(ldims)}
    {
        guard = exec->get_scoped_device_id_guard();
    }

    void TearDown() final
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    // Generate a global unit vector with 1 at the specified global index.
    std::shared_ptr<const DistVec> generate_unit_vec(const int64_t nonzero_idx)
    {
        const int myrank = comm.rank();
        auto local_b = Vec::create(
            exec->get_master(), gko::dim<2>{static_cast<size_t>(local_n), 1});
        local_b->fill(0.0);
        const int64_t* range_bounds = partition->get_range_bounds();
        if (nonzero_idx >= range_bounds[myrank] &&
            nonzero_idx < range_bounds[myrank + 1]) {
            const int local_idx = nonzero_idx % local_n;
            const int nonzero_rank = nonzero_idx / local_n;
            if (nonzero_rank != myrank) {
                std::cout << "Unexpected partition!\n";
                std::exit(-1);
            }
            auto vals = local_b->get_values();
            vals[local_idx] = 1.0;
        }
        auto local_dev = gko::clone(exec, local_b);
        return gko::share(DistVec::create(
            exec, comm, gko::dim<2>{static_cast<size_t>(global_n), 1},
            std::move(local_dev)));
    }

    void check_mat_times_unit_vec(const int64_t input_nonzero_idx,
                                  const DistVec* const y)
    {
        const int myrank = comm.rank();
        const int local_new_idx = input_nonzero_idx % local_n;
        const int nonzero_rank = input_nonzero_idx / local_n;
        const int local_old_idx = ordering.new_to_old[local_new_idx];
        const int64_t global_old_idx =
            local_old_idx + static_cast<int64_t>(local_n) * nonzero_rank;
        const std::array<int32_t, 3> old_local_col =
            get_natural_3d_indices_from_flat(ldims, local_old_idx);

        const auto local_y = y->get_const_local_values();

        const int64_t* range_bounds = partition->get_range_bounds();
        if (input_nonzero_idx >= range_bounds[myrank] &&
            input_nonzero_idx < range_bounds[myrank + 1]) {
            ASSERT_EQ(nonzero_rank, myrank);
            // check local values
            EXPECT_EQ(local_y[local_new_idx], 26.0);
            // TODO: complete local values
            std::vector<int> nbd_pts{};
            if (global_old_idx == 0) {
                EXPECT_EQ(local_old_idx, 0);
                nbd_pts = std::vector<int>{1, 3, 4, 9, 10, 12, 13};
                for (auto pt : nbd_pts) {
                    auto npt = ordering.old_to_new[pt];
                    EXPECT_EQ(local_y[npt], -1.0);
                }
            } else if (global_old_idx == 1) {
                EXPECT_EQ(local_old_idx, 1);
                nbd_pts =
                    std::vector<int>{0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14};
            }
            for (auto pt : nbd_pts) {
                auto npt = ordering.old_to_new[pt];
                EXPECT_EQ(local_y[npt], -1.0);
            }
        } else {
            // TODO: complete nonlocal values
            std::vector<int> nbd_pts{};
            if (global_old_idx == 2) {
                if (myrank == 1) {
                    nbd_pts = std::vector<int>{0, 3, 9, 12};
                }
            }
            for (auto pt : nbd_pts) {
                auto npt = ordering.old_to_new[pt];
                EXPECT_EQ(local_y[npt], -1.0);
            }
        }
    }

    // Use a 3x3x3 grid: interior point (1,1,1) has 27 neighbors,
    // corners have 8, edges/faces have intermediate counts.
    const std::array<int32_t, 3> ldims{3, 3, 3};
    const int local_n{27};
    const int64_t global_n{local_n * 4};

    gko::experimental::mpi::communicator comm;

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> exec;
    gko::scoped_device_id_guard guard;
    std::shared_ptr<const partition_t> partition;
    MulticolorOrdering ordering;
};


TEST_F(DistMatrixGeneration, RankDecompositionWorksAsExpected)
{
    std::array<int32_t, 3> proc_dims = cubic_radical_search(4);
    std::array<int32_t, 3> expected{2, 1, 2};

    EXPECT_EQ(proc_dims, expected);

    proc_dims = cubic_radical_search(8);
    expected = {2, 2, 2};

    EXPECT_EQ(proc_dims, expected);
}


TEST_F(DistMatrixGeneration, Generated3x3MatrixAppliesCorrectlyToUnitVectors)
{
    ASSERT_EQ(comm.size(), 4);
    ASSERT_EQ(partition->get_num_parts(), 4);
    ASSERT_EQ(partition->get_num_ranges(), 4);
    ASSERT_EQ(partition->get_num_empty_parts(), 0);
    const int* part_sz = partition->get_part_sizes();
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(part_sz[i], 27);
    }

    ConstOffdiag gen;
    const auto data =
        generate_stencil_data<double, int64_t, int>(comm, ldims, gen, ordering);
    auto system_mat = gko::share(
        DistMtx::create(exec, comm, gko::with_matrix_type<gko::matrix::Ell>()));
    system_mat->read_distributed(data, partition);

    // Check each nonzero by checking each column
    for (int64_t j = 0; j < global_n; j++) {
        auto unitvec = generate_unit_vec(j);
        auto y = DistVec::create(exec, comm, gko::dim<2>(global_n, 1),
                                 gko::dim<2>(local_n, 1));
        system_mat->apply(unitvec, y);
        check_mat_times_unit_vec(j, y.get());
    }
}


}  // namespace
