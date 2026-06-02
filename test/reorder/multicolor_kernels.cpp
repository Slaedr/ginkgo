// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/multicolor_kernels.hpp"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/test/utils/matrix_generator.hpp"
#include "test/utils/common_fixture.hpp"


class Multicolor : public CommonTestFixture {
protected:
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;

    Multicolor()
    {
        auto mdata5 = gko::test::generate_laplacian_2d_5point_matrix_data<
            value_type, index_type>(dims2);
        laplace2d5_ref = gko::share(CsrMtx::create(ref));
        laplace2d5_ref->read(mdata5);
        laplace2d5 = gko::share(gko::clone(exec, laplace2d5_ref));
        auto mdata27 = gko::test::generate_laplacian_3d_27point_matrix_data<
            value_type, index_type>(dims3);
        laplace3d27_ref = gko::share(CsrMtx::create(ref));
        laplace3d27_ref->read(mdata27);
        laplace3d27 = gko::share(gko::clone(exec, laplace3d27_ref));
    }

    // Returns max_color_size / avg_color_size; 1.0 is perfectly balanced.
    static double max_imbalance(const index_type nrows,
                                const std::vector<index_type>& color_ptrs)
    {
        const auto num_colors = static_cast<index_type>(color_ptrs.size()) - 1;
        const double avg = static_cast<double>(nrows) / num_colors;
        index_type max_size = 0;
        for (index_type c = 0; c < num_colors; c++) {
            max_size = std::max(max_size, color_ptrs[c + 1] - color_ptrs[c]);
        }
        return max_size / avg;
    }

    // Returns true iff no two adjacent nodes share the same color.
    // color_ptrs[c] .. color_ptrs[c+1]-1 are the new indices of color c;
    // perm maps old indices to new indices.
    static bool is_independent_set(const index_type nrows,
                                   const index_type* row_ptrs,
                                   const index_type* col_idxs,
                                   const std::vector<index_type>& color_ptrs,
                                   const index_type* perm)
    {
        std::vector<int> node_color(nrows);
        for (index_type old_i = 0; old_i < nrows; old_i++) {
            const index_type new_i = perm[old_i];
            const auto it =
                std::upper_bound(color_ptrs.begin(), color_ptrs.end(), new_i);
            node_color[old_i] =
                static_cast<int>(std::distance(color_ptrs.begin(), it)) - 1;
        }
        for (index_type i = 0; i < nrows; i++) {
            for (index_type jz = row_ptrs[i]; jz < row_ptrs[i + 1]; jz++) {
                const index_type j = col_idxs[jz];
                if (i != j && node_color[i] == node_color[j]) {
                    return false;
                }
            }
        }
        return true;
    }

    gko::dim<2> dims2{4, 4};
    gko::dim<3> dims3{4, 4, 4};
    std::shared_ptr<CsrMtx> laplace2d5_ref;
    std::shared_ptr<CsrMtx> laplace2d5;
    std::shared_ptr<CsrMtx> laplace3d27_ref;
    std::shared_ptr<CsrMtx> laplace3d27;
};


TEST_F(Multicolor, HasUpToTwiceOptimalColorCount2d5p)
{
    const auto nrows = static_cast<index_type>(dims2[0] * dims2[1]);
    gko::array<index_type> perm{exec, static_cast<size_t>(nrows)};
    gko::array<index_type> invperm{exec, static_cast<size_t>(nrows)};
    std::vector<index_type> color_ptrs;

    gko::kernels::GKO_DEVICE_NAMESPACE::multicolor::compute_permutation_csr(
        exec, nrows, laplace2d5->get_const_row_ptrs(),
        laplace2d5->get_const_col_idxs(), color_ptrs, perm.get_data(),
        invperm.get_data());

    EXPECT_LE(color_ptrs.size(), 5) << "Num colors= " << color_ptrs.size() - 1;
}

TEST_F(Multicolor, ColorsAreIndependentSets2d5p)
{
    const auto nrows = static_cast<index_type>(dims2[0] * dims2[1]);
    gko::array<index_type> perm{exec, static_cast<size_t>(nrows)};
    gko::array<index_type> invperm{exec, static_cast<size_t>(nrows)};
    std::vector<index_type> color_ptrs;

    gko::kernels::GKO_DEVICE_NAMESPACE::multicolor::compute_permutation_csr(
        exec, nrows, laplace2d5->get_const_row_ptrs(),
        laplace2d5->get_const_col_idxs(), color_ptrs, perm.get_data(),
        invperm.get_data());

    const gko::array<index_type> perm_host{ref, perm};
    EXPECT_TRUE(is_independent_set(nrows, laplace2d5_ref->get_const_row_ptrs(),
                                   laplace2d5_ref->get_const_col_idxs(),
                                   color_ptrs, perm_host.get_const_data()));
}

TEST_F(Multicolor, HasUpToTwiceOptimalColorCount3d27p)
{
    const auto nrows = static_cast<index_type>(dims3[0] * dims3[1] * dims3[2]);
    gko::array<index_type> perm{exec, static_cast<size_t>(nrows)};
    gko::array<index_type> invperm{exec, static_cast<size_t>(nrows)};
    std::vector<index_type> color_ptrs;

    gko::kernels::GKO_DEVICE_NAMESPACE::multicolor::compute_permutation_csr(
        exec, nrows, laplace3d27->get_const_row_ptrs(),
        laplace3d27->get_const_col_idxs(), color_ptrs, perm.get_data(),
        invperm.get_data());

    EXPECT_LE(color_ptrs.size(), 17) << "Num colors= " << color_ptrs.size() - 1;
}


TEST_F(Multicolor, ColorsAreIndependentSets3d27p)
{
    const auto nrows = static_cast<index_type>(dims3[0] * dims3[1] * dims3[2]);
    gko::array<index_type> perm{exec, static_cast<size_t>(nrows)};
    gko::array<index_type> invperm{exec, static_cast<size_t>(nrows)};
    std::vector<index_type> color_ptrs;

    gko::kernels::GKO_DEVICE_NAMESPACE::multicolor::compute_permutation_csr(
        exec, nrows, laplace3d27->get_const_row_ptrs(),
        laplace3d27->get_const_col_idxs(), color_ptrs, perm.get_data(),
        invperm.get_data());

    const gko::array<index_type> perm_host{ref, perm};
    EXPECT_TRUE(is_independent_set(nrows, laplace3d27_ref->get_const_row_ptrs(),
                                   laplace3d27_ref->get_const_col_idxs(),
                                   color_ptrs, perm_host.get_const_data()));
}


TEST_F(Multicolor, ColorClassesAreBalanced2d5p)
{
    const auto nrows = static_cast<index_type>(dims2[0] * dims2[1]);
    gko::array<index_type> perm{exec, static_cast<size_t>(nrows)};
    gko::array<index_type> invperm{exec, static_cast<size_t>(nrows)};
    std::vector<index_type> color_ptrs;

    gko::kernels::GKO_DEVICE_NAMESPACE::multicolor::compute_permutation_csr(
        exec, nrows, laplace2d5->get_const_row_ptrs(),
        laplace2d5->get_const_col_idxs(), color_ptrs, perm.get_data(),
        invperm.get_data());

    EXPECT_LE(max_imbalance(nrows, color_ptrs), 2.0);
}


TEST_F(Multicolor, ColorClassesAreBalanced3d27p)
{
    const auto nrows = static_cast<index_type>(dims3[0] * dims3[1] * dims3[2]);
    gko::array<index_type> perm{exec, static_cast<size_t>(nrows)};
    gko::array<index_type> invperm{exec, static_cast<size_t>(nrows)};
    std::vector<index_type> color_ptrs;

    gko::kernels::GKO_DEVICE_NAMESPACE::multicolor::compute_permutation_csr(
        exec, nrows, laplace3d27->get_const_row_ptrs(),
        laplace3d27->get_const_col_idxs(), color_ptrs, perm.get_data(),
        invperm.get_data());

    EXPECT_LE(max_imbalance(nrows, color_ptrs), 2.0);
}
