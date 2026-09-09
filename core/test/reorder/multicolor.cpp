// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/reordering.hpp"


template <typename IndexType>
class Multicolor : public ::testing::Test {
protected:
    using v_type = float;
    using i_type = IndexType;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;

    Multicolor()
        : exec(gko::ReferenceExecutor::create()),
          mc_factory(reorder_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename reorder_type::Factory> mc_factory;
};

TYPED_TEST_SUITE(Multicolor, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(Multicolor, MulticolorFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->mc_factory->get_executor(), this->exec);
}

TYPED_TEST(Multicolor, GeneratesCorrectOrderingWithCsrInput)
{
    using v_type = typename TestFixture::v_type;
    using i_type = typename TestFixture::i_type;
    const gko::dim<2> grid{5, 5};
    auto expected =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(grid);
    const auto size = 25u;
    auto mdata =
        gko::test::generate_laplacian_2d_5point_matrix_data<v_type, i_type>(
            grid);
    auto mat = gko::share(gko::matrix::Csr<v_type, i_type>::create(this->exec));
    mat->read(mdata);

    auto mc = this->mc_factory->generate(mat);

    auto color_ptrs = mc->get_color_pointers();
    auto perm = mc->get_permutation()->get_const_permutation();
    auto iperm = mc->get_inverse_permutation()->get_const_permutation();
    auto permv = std::vector<i_type>(perm, perm + size);
    auto ipermv = std::vector<i_type>(iperm, iperm + size);
    EXPECT_EQ(color_ptrs, expected.color_ptrs);
    EXPECT_EQ(permv, expected.new_to_old);
    EXPECT_EQ(ipermv, expected.old_to_new);
}

TYPED_TEST(Multicolor, GeneratesCorrectOrderingWithSparsityCsrInput)
{
    using v_type = typename TestFixture::v_type;
    using i_type = typename TestFixture::i_type;
    const gko::dim<2> grid{5, 5};
    auto expected =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(grid);
    const auto size = 25u;
    auto mdata =
        gko::test::generate_laplacian_2d_5point_matrix_data<v_type, i_type>(
            grid);
    auto mat = gko::matrix::Csr<v_type, i_type>::create(this->exec);
    mat->read(mdata);
    auto smat = gko::share(
        gko::matrix::SparsityCsr<v_type, i_type>::create(this->exec));
    mat->convert_to(smat.get());

    auto mc = this->mc_factory->generate(smat);

    auto color_ptrs = mc->get_color_pointers();
    auto perm = mc->get_permutation()->get_const_permutation();
    auto iperm = mc->get_inverse_permutation()->get_const_permutation();
    const auto permv = std::vector<i_type>(perm, perm + size);
    const auto ipermv = std::vector<i_type>(iperm, iperm + size);
    EXPECT_EQ(color_ptrs, expected.color_ptrs);
    EXPECT_EQ(permv, expected.new_to_old);
    EXPECT_EQ(ipermv, expected.old_to_new);
}

TYPED_TEST(Multicolor, DefaultParametersSymmetrizeAndSort)
{
    ASSERT_FALSE(this->mc_factory->get_parameters().skip_symmetrize);
    ASSERT_FALSE(this->mc_factory->get_parameters().skip_sorting);
}

TYPED_TEST(Multicolor, SymmetrizingDoesNotChangeOrderingOfSymmetricMatrix)
{
    using v_type = typename TestFixture::v_type;
    using i_type = typename TestFixture::i_type;
    using reorder_type = typename TestFixture::reorder_type;
    const gko::dim<2> grid{5, 5};
    auto expected =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(grid);
    const auto size = 25u;
    auto mdata =
        gko::test::generate_laplacian_2d_5point_matrix_data<v_type, i_type>(
            grid);
    auto mat = gko::share(gko::matrix::Csr<v_type, i_type>::create(this->exec));
    mat->read(mdata);

    auto mc_default = this->mc_factory->generate(mat);
    auto mc_skip = reorder_type::build()
                       .with_skip_symmetrize(true)
                       .on(this->exec)
                       ->generate(mat);

    for (auto* mc : {mc_default.get(), mc_skip.get()}) {
        auto color_ptrs = mc->get_color_pointers();
        auto perm = mc->get_permutation()->get_const_permutation();
        auto iperm = mc->get_inverse_permutation()->get_const_permutation();
        const auto permv = std::vector<i_type>(perm, perm + size);
        const auto ipermv = std::vector<i_type>(iperm, iperm + size);
        EXPECT_EQ(color_ptrs, expected.color_ptrs);
        EXPECT_EQ(permv, expected.new_to_old);
        EXPECT_EQ(ipermv, expected.old_to_new);
    }
}

TYPED_TEST(Multicolor, HandlesNonsymmetricCsrInput)
{
    using v_type = typename TestFixture::v_type;
    using i_type = typename TestFixture::i_type;
    using reorder_type = typename TestFixture::reorder_type;
    // Row 0 has an explicit entry in column 2, but row 2 has no entry in
    // column 0. The naive (row-only) greedy coloring colors rows in
    // increasing order and only looks at each row's own explicit entries, so
    // when processing row 2 it never learns that row 0 already used color 0,
    // and colors 0 and 2 identically despite being structurally coupled.
    gko::matrix_data<v_type, i_type> mdata{gko::dim<2>{4, 4},
                                           {{0, 0, gko::one<v_type>()},
                                            {0, 2, gko::one<v_type>()},
                                            {1, 1, gko::one<v_type>()},
                                            {2, 2, gko::one<v_type>()},
                                            {3, 3, gko::one<v_type>()}}};
    auto mat = gko::share(gko::matrix::Csr<v_type, i_type>::create(this->exec));
    mat->read(mdata);

    auto mc_base = this->mc_factory->generate(mat);
    auto* mc = gko::as<reorder_type>(mc_base.get());
    const auto color_ptrs = mc->get_color_pointers();
    auto permuted = mat->permute(mc->get_permutation());

    EXPECT_TRUE(gko::test::colors_are_independent(
        static_cast<i_type>(4), permuted->get_const_row_ptrs(),
        permuted->get_const_col_idxs(), color_ptrs));

    // With symmetrization skipped, this matrix reproduces the invalid
    // coloring described above: rows 0 and 2 land in the same color.
    auto mc_skip_base = reorder_type::build()
                            .with_skip_symmetrize(true)
                            .on(this->exec)
                            ->generate(mat);
    auto* mc_skip = gko::as<reorder_type>(mc_skip_base.get());
    const auto color_ptrs_skip = mc_skip->get_color_pointers();
    auto permuted_skip = mat->permute(mc_skip->get_permutation());

    EXPECT_FALSE(gko::test::colors_are_independent(
        static_cast<i_type>(4), permuted_skip->get_const_row_ptrs(),
        permuted_skip->get_const_col_idxs(), color_ptrs_skip));
}
