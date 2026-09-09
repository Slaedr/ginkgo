// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/multicolor_kernels.hpp"

#include <fstream>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/test/utils/assertions.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/reordering.hpp"
#include "matrices/config.hpp"


class Multicolor : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;
    using perm_type = gko::matrix::Permutation<i_type>;

    Multicolor() : exec(gko::ReferenceExecutor::create())
    {
        auto mdata5 =
            gko::test::generate_laplacian_2d_5point_matrix_data<v_type, i_type>(
                dims2);
        laplace2d5 = gko::share(CsrMtx::create(exec));
        laplace2d5->read(mdata5);
        auto mdata27 =
            gko::test::generate_laplacian_3d_27point_matrix_data<v_type,
                                                                 i_type>(dims3);
        laplace3d27 = gko::share(CsrMtx::create(exec));
        laplace3d27->read(mdata27);
    }

    gko::dim<2> dims2{4, 4};
    gko::dim<3> dims3{4, 4, 4};
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<CsrMtx> laplace2d5;
    std::shared_ptr<CsrMtx> laplace3d27;

    static bool is_permutation(const perm_type* input_perm)
    {
        const auto perm_size = input_perm->get_size()[0];
        auto perm_sorted = std::vector<i_type>(perm_size);
        std::copy_n(input_perm->get_const_permutation(), perm_size,
                    perm_sorted.begin());
        std::sort(perm_sorted.begin(), perm_sorted.end());
        auto identity = std::vector<i_type>(perm_size);
        std::iota(identity.begin(), identity.end(), 0);
        return identity == perm_sorted;
    }
};


TEST_F(Multicolor, CreatesCorrectColorPtrs2d5p)
{
    const auto nrows = static_cast<i_type>(dims2[0] * dims2[1]);
    std::vector<i_type> perm(nrows);
    std::vector<i_type> invperm(nrows);
    std::vector<i_type> color_ptrs;

    gko::kernels::reference::multicolor::compute_permutation_csr(
        exec, nrows, laplace2d5->get_const_row_ptrs(),
        laplace2d5->get_const_col_idxs(), color_ptrs, perm.data(),
        invperm.data());

    ASSERT_EQ(color_ptrs.size(), 3);
    EXPECT_EQ(color_ptrs[0], 0);
    EXPECT_EQ(color_ptrs[1], nrows / 2);
    EXPECT_EQ(color_ptrs[2], nrows);
}


TEST_F(Multicolor, CreatesCorrectPermutations2d5p)
{
    const auto nrows = static_cast<i_type>(dims2[0] * dims2[1]);
    std::vector<i_type> perm(nrows);
    std::vector<i_type> invperm(nrows);
    std::vector<i_type> color_ptrs;
    auto expected_ordering =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(dims2);

    gko::kernels::reference::multicolor::compute_permutation_csr(
        exec, nrows, laplace2d5->get_const_row_ptrs(),
        laplace2d5->get_const_col_idxs(), color_ptrs, perm.data(),
        invperm.data());

    EXPECT_EQ(expected_ordering.new_to_old, perm);
    EXPECT_EQ(expected_ordering.old_to_new, invperm);
}

TEST_F(Multicolor, CreatesCorrectColorPtrs3d27p)
{
    const auto nrows = static_cast<i_type>(dims3[0] * dims3[1] * dims3[2]);
    std::vector<i_type> perm(nrows);
    std::vector<i_type> invperm(nrows);
    std::vector<i_type> color_ptrs;

    gko::kernels::reference::multicolor::compute_permutation_csr(
        exec, nrows, laplace3d27->get_const_row_ptrs(),
        laplace3d27->get_const_col_idxs(), color_ptrs, perm.data(),
        invperm.data());

    ASSERT_EQ(color_ptrs.size(), 9);
    for (int color = 0; color < 9; color++) {
        EXPECT_EQ(color_ptrs[color], color * nrows / 8);
    }
}


TEST_F(Multicolor, CreatesCorrectPermutations3d27p)
{
    const auto nrows = static_cast<i_type>(dims3[0] * dims3[1] * dims3[2]);
    std::vector<i_type> perm(nrows);
    std::vector<i_type> invperm(nrows);
    std::vector<i_type> color_ptrs;
    auto expected_ordering =
        gko::test::compute_multicolor_ordering_regular_box<i_type>(dims3);

    gko::kernels::reference::multicolor::compute_permutation_csr(
        exec, nrows, laplace3d27->get_const_row_ptrs(),
        laplace3d27->get_const_col_idxs(), color_ptrs, perm.data(),
        invperm.data());

    EXPECT_EQ(expected_ordering.new_to_old, perm);
    EXPECT_EQ(expected_ordering.old_to_new, invperm);
}


/**
 * Tests that reordering a matrix with the permutation returned by Multicolor
 * has to produce a matrix whose rows are grouped into the independent sets
 * described by the color pointers.
 *
 * Uses a real SuiteSparse matrix rather than a structured stencil, since the
 * benchmarks are run on SuiteSparse matrices and an irregular sparsity pattern
 * is what makes this non-trivial.
 */
class MulticolorSuiteSparse : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;

    MulticolorSuiteSparse()
        : exec(gko::ReferenceExecutor::create()),
          // 1138 x 1138, symmetric (hence structurally symmetric once read)
          mtx(gko::share(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_1138_bus_mtx, std::ios::in),
              exec)))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<CsrMtx> mtx;
};


TEST_F(MulticolorSuiteSparse, PermutedMatrixHasIndependentColorBlocks)
{
    const auto nrows = static_cast<i_type>(mtx->get_size()[0]);
    auto mc_base = reorder_type::build().on(exec)->generate(mtx);
    const auto* mc = gko::as<reorder_type>(mc_base.get());
    const auto color_ptrs = mc->get_color_pointers();

    // permute() expects new-to-old indices, see matrix::Permutation
    auto permuted = mtx->permute(mc->get_permutation());

    ASSERT_GE(color_ptrs.size(), 2);
    EXPECT_EQ(color_ptrs.front(), 0);
    EXPECT_EQ(color_ptrs.back(), nrows);
    EXPECT_TRUE(gko::test::colors_are_independent(
        nrows, permuted->get_const_row_ptrs(), permuted->get_const_col_idxs(),
        color_ptrs));
}


TEST_F(MulticolorSuiteSparse, PermutationAndInversePermutationAreInverses)
{
    const auto nrows = static_cast<i_type>(mtx->get_size()[0]);
    auto mc_base = reorder_type::build().on(exec)->generate(mtx);
    const auto* mc = gko::as<reorder_type>(mc_base.get());

    const auto* perm = mc->get_permutation()->get_const_permutation();
    const auto* invperm =
        mc->get_inverse_permutation()->get_const_permutation();

    for (i_type new_i = 0; new_i < nrows; new_i++) {
        ASSERT_GE(perm[new_i], 0);
        ASSERT_LT(perm[new_i], nrows);
        EXPECT_EQ(invperm[perm[new_i]], new_i) << "at new index " << new_i;
    }
}


/**
 * Tests that Multicolor produces a valid (independent-set) coloring on a
 * structurally nonsymmetric matrix, by default coloring the pattern of
 * A + A^T rather than A itself.
 *
 * ani1_nonsymm has 164 of its 238 entries structurally asymmetric, which is
 * enough for the naive (row-only) greedy coloring to place several coupled
 * rows in the same color: colors_are_independent is false without
 * symmetrization, and true with it.
 */
class MulticolorNonsymmetric : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;

    MulticolorNonsymmetric()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::share(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_ani1_nonsymm_mtx,
                            std::ios::in),
              exec)))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<CsrMtx> mtx;
};


TEST_F(MulticolorNonsymmetric, ColorsAreIndependentByDefault)
{
    const auto nrows = static_cast<i_type>(mtx->get_size()[0]);
    auto mc_base = reorder_type::build().on(exec)->generate(mtx);
    const auto* mc = gko::as<reorder_type>(mc_base.get());
    const auto color_ptrs = mc->get_color_pointers();

    auto permuted = mtx->permute(mc->get_permutation());

    ASSERT_GE(color_ptrs.size(), 2);
    EXPECT_EQ(color_ptrs.front(), 0);
    EXPECT_EQ(color_ptrs.back(), nrows);
    EXPECT_TRUE(gko::test::colors_are_independent(
        nrows, permuted->get_const_row_ptrs(), permuted->get_const_col_idxs(),
        color_ptrs));
}


TEST_F(MulticolorNonsymmetric, SkipSymmetrizeColorsTheUnsymmetrizedPattern)
{
    const auto nrows = static_cast<i_type>(mtx->get_size()[0]);
    auto mc_base =
        reorder_type::build().with_skip_symmetrize(true).on(exec)->generate(
            mtx);
    const auto* mc = gko::as<reorder_type>(mc_base.get());
    const auto color_ptrs = mc->get_color_pointers();

    auto permuted = mtx->permute(mc->get_permutation());

    // Document the opt-out's contract: without symmetrization, the naive
    // greedy coloring of this matrix is not a valid independent-set
    // coloring.
    EXPECT_FALSE(gko::test::colors_are_independent(
        nrows, permuted->get_const_row_ptrs(), permuted->get_const_col_idxs(),
        color_ptrs));
}


TEST_F(MulticolorNonsymmetric, PermutationAndInversePermutationAreInverses)
{
    const auto nrows = static_cast<i_type>(mtx->get_size()[0]);
    auto mc_base = reorder_type::build().on(exec)->generate(mtx);
    const auto* mc = gko::as<reorder_type>(mc_base.get());

    const auto* perm = mc->get_permutation()->get_const_permutation();
    const auto* invperm =
        mc->get_inverse_permutation()->get_const_permutation();

    for (i_type new_i = 0; new_i < nrows; new_i++) {
        ASSERT_GE(perm[new_i], 0);
        ASSERT_LT(perm[new_i], nrows);
        EXPECT_EQ(invperm[perm[new_i]], new_i) << "at new index " << new_i;
    }
}
