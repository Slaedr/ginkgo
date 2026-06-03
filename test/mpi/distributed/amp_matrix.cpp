// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#ifndef GKO_COMPILING_DPCPP


class DistributedAmpMatrix : public CommonMpiTestFixture {
protected:
    using value_type = double;
    using local_index_type = int32_t;
    using global_index_type = int64_t;
    using Ell = gko::matrix::Ell<value_type, local_index_type>;
    using Amp = gko::matrix::AMP<value_type, local_index_type>;
    using Csr = gko::matrix::Csr<value_type, local_index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using dist_mat =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using dist_vec = gko::experimental::distributed::Vector<value_type>;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;

    static constexpr global_index_type n = 4;

    gko::matrix_data<value_type, global_index_type> mat_data{{n, n},
                                                             {{0, 0, 4.0},
                                                              {0, 1, -1.0},
                                                              {0, 2, -1.0},
                                                              {1, 0, -1.0},
                                                              {1, 1, 4.0},
                                                              {1, 3, -1.0},
                                                              {2, 0, -1.0},
                                                              {2, 2, 4.0},
                                                              {2, 3, -1.0},
                                                              {3, 1, -1.0},
                                                              {3, 2, -1.0},
                                                              {3, 3, 4.0}}};

    gko::matrix_data<value_type, global_index_type> x_data{
        {n, 1}, {{0, 0, 1.0}, {1, 0, 2.0}, {2, 0, 3.0}, {3, 0, 4.0}}};

    gko::matrix_data<value_type, global_index_type> zeros_data{
        {n, 1}, {{0, 0, 0.0}, {1, 0, 0.0}, {2, 0, 0.0}, {3, 0, 0.0}}};

    gko::matrix_data<value_type, global_index_type> ones_data{
        {n, 1}, {{0, 0, 1.0}, {1, 0, 1.0}, {2, 0, 1.0}, {3, 0, 1.0}}};

    std::shared_ptr<part_type> partition = gko::share(
        part_type::build_from_global_size_uniform(exec, comm.size(), n));

    std::shared_ptr<dist_mat> create_amp_dist_matrix()
    {
        auto ell_empty = gko::share(Ell::create(ref, gko::dim<2>{0, 0}));
        auto amp_template =
            Amp::build().with_tolerance(0.01f).on(exec)->generate(ell_empty);
        auto csr_template = Csr::create(exec);
        auto A = dist_mat::create(exec, comm, amp_template.get(),
                                  csr_template.get());
        A->read_distributed(mat_data, partition);
        return A;
    }

    std::shared_ptr<dist_mat> create_csr_dist_matrix()
    {
        auto A = dist_mat::create(exec, comm);
        A->read_distributed(mat_data, partition);
        return A;
    }
};


TEST_F(DistributedAmpMatrix, CanCreateWithAmpLocalBlock)
{
    auto ell_empty = gko::share(Ell::create(ref, gko::dim<2>{0, 0}));
    auto amp_template =
        Amp::build().with_tolerance(0.01f).on(exec)->generate(ell_empty);
    auto csr_template = Csr::create(exec);

    auto A =
        dist_mat::create(exec, comm, amp_template.get(), csr_template.get());

    ASSERT_NE(A, nullptr);
}


TEST_F(DistributedAmpMatrix, CanReadDistributedWithAmpLocalBlock)
{
    auto A = create_amp_dist_matrix();

    EXPECT_EQ(A->get_size(), gko::dim<2>(n, n));
}


TEST_F(DistributedAmpMatrix, SpMVProducesCorrectResult)
{
    auto A = create_amp_dist_matrix();
    auto A_ref = create_csr_dist_matrix();

    auto x = dist_vec::create(exec, comm);
    x->read_distributed(x_data, partition);

    auto y_amp = dist_vec::create(exec, comm);
    y_amp->read_distributed(zeros_data, partition);
    A->apply(x, y_amp);

    auto y_ref = dist_vec::create(exec, comm);
    y_ref->read_distributed(zeros_data, partition);
    A_ref->apply(x, y_ref);

    GKO_ASSERT_MTX_NEAR(y_amp->get_local_vector(), y_ref->get_local_vector(),
                        0.0);
}


TEST_F(DistributedAmpMatrix, AdvancedSpMVProducesCorrectResult)
{
    auto A = create_amp_dist_matrix();
    auto A_ref = create_csr_dist_matrix();

    auto x = dist_vec::create(exec, comm);
    x->read_distributed(x_data, partition);

    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({0.5}, exec);

    auto y_amp = dist_vec::create(exec, comm);
    y_amp->read_distributed(ones_data, partition);
    A->apply(alpha, x, beta, y_amp);

    auto y_ref = dist_vec::create(exec, comm);
    y_ref->read_distributed(ones_data, partition);
    A_ref->apply(alpha, x, beta, y_ref);

    GKO_ASSERT_MTX_NEAR(y_amp->get_local_vector(), y_ref->get_local_vector(),
                        0.0);
}


TEST_F(DistributedAmpMatrix, SpMVWorksWithAmpNonLocalBlock)
{
    auto ell_empty = gko::share(Ell::create(ref, gko::dim<2>{0, 0}));
    auto amp_local =
        Amp::build().with_tolerance(0.01f).on(exec)->generate(ell_empty);
    auto amp_nonlocal =
        Amp::build().with_tolerance(0.01f).on(exec)->generate(ell_empty);
    auto A = dist_mat::create(exec, comm, amp_local.get(), amp_nonlocal.get());
    A->read_distributed(mat_data, partition);

    auto A_ref = create_csr_dist_matrix();

    auto x = dist_vec::create(exec, comm);
    x->read_distributed(x_data, partition);

    auto y_amp = dist_vec::create(exec, comm);
    y_amp->read_distributed(zeros_data, partition);
    A->apply(x, y_amp);

    auto y_ref = dist_vec::create(exec, comm);
    y_ref->read_distributed(zeros_data, partition);
    A_ref->apply(x, y_ref);

    GKO_ASSERT_MTX_NEAR(y_amp->get_local_vector(), y_ref->get_local_vector(),
                        0.0);
}


#endif  // GKO_COMPILING_DPCPP
