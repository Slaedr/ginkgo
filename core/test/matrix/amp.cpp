// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/matrix/amp_helpers.hpp"
#include "core/test/utils.hpp"


TEST(AMPTypes, NarrowTypesWorksCorrectly)
{
#if GINKGO_HAVE_AMP_HALF
    using mytypesd = gko::amp::narrow_types<double>::type;
    constexpr auto chkd =
        std::is_same<mytypesd,
                     std::tuple<double, float, gko::amp::half>>::value;
    static_assert(chkd, "Wrong types_list<double>!");
    using mytypescd = gko::amp::narrow_types<std::complex<double>>::type;
    constexpr auto chkcd =
        std::is_same<mytypescd,
                     std::tuple<std::complex<double>, std::complex<float>,
                                std::complex<gko::amp::half>>>::value;
    static_assert(chkcd, "Wrong types_list<cdouble>!");
    using mytypescf = gko::amp::narrow_types<std::complex<float>>::type;
    constexpr auto chkcf = std::is_same<
        mytypescf,
        std::tuple<std::complex<float>, std::complex<gko::amp::half>>>::value;
    static_assert(chkcf, "Wrong types_list<cfloat>!");
    static_assert(gko::amp::narrow_types<double>::num_types == 3);
    static_assert(gko::amp::narrow_types<std::complex<float>>::num_types == 2);
    static_assert(gko::amp::narrow_types<gko::amp::half>::num_types == 1);
    static_assert(
        gko::amp::narrow_types<std::complex<gko::amp::half>>::num_types == 1);
#else
    using mytypesd = gko::amp::narrow_types<double>::type;
    constexpr auto chkd =
        std::is_same<mytypesd, std::tuple<double, float>>::value;
    static_assert(chkd, "Wrong types_list<double>!");
    using mytypescd = gko::amp::narrow_types<std::complex<double>>::type;
    constexpr auto chkcd =
        std::is_same<mytypescd, std::tuple<std::complex<double>,
                                           std::complex<float>>>::value;
    static_assert(chkcd, "Wrong types_list<cdouble>!");
    using mytypescf = gko::amp::narrow_types<std::complex<float>>::type;
    constexpr auto chkcf =
        std::is_same<mytypescf, std::tuple<std::complex<float>>>::value;
    static_assert(chkcf, "Wrong types_list<cfloat>!");
    static_assert(gko::amp::narrow_types<double>::num_types == 2);
    static_assert(gko::amp::narrow_types<std::complex<double>>::num_types == 2);
    static_assert(gko::amp::narrow_types<float>::num_types == 1);
    static_assert(gko::amp::narrow_types<std::complex<float>>::num_types == 1);
#endif
}

template <typename T, typename I>
using Ell = gko::matrix::Ell<T, I>;


TEST(AMPHelpers, AllocatesEllBinsCorrectlyDouble)
{
    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
#if GINKGO_HAVE_AMP_HALF
    auto mnpr = gko::amp::precision_array<int, double>{3, 4, 5};

    auto bins = gko::amp::allocate_bins<double, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 3,
                  "wrong number of bins!");
    auto p = dynamic_cast<Ell<double, int>*>(bins[0].get());
    EXPECT_TRUE(p);
    EXPECT_EQ(p->get_size(), ds);
    EXPECT_EQ(p->get_num_stored_elements_per_row(), 3);
    EXPECT_TRUE(p->get_col_idxs()[0] = 1);
    auto q = dynamic_cast<Ell<float, int>*>(bins[1].get());
    EXPECT_TRUE(q);
    EXPECT_EQ(q->get_size(), ds);
    EXPECT_EQ(q->get_num_stored_elements_per_row(), 4);
    EXPECT_TRUE(q->get_col_idxs());
    auto r = dynamic_cast<Ell<gko::amp::half, int>*>(bins[2].get());
    EXPECT_TRUE(r);
    EXPECT_EQ(r->get_size(), ds);
    EXPECT_EQ(r->get_num_stored_elements_per_row(), 5);
    EXPECT_TRUE(r->get_col_idxs());
#else
    auto mnpr = gko::amp::precision_array<int, double>{3, 4};

    auto bins = gko::amp::allocate_bins<double, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 2,
                  "wrong number of bins!");
    auto p = dynamic_cast<Ell<double, int>*>(bins[0].get());
    EXPECT_TRUE(p);
    EXPECT_EQ(p->get_size(), ds);
    EXPECT_EQ(p->get_num_stored_elements_per_row(), 3);
    EXPECT_TRUE(p->get_col_idxs()[0] = 1);
    auto q = dynamic_cast<Ell<float, int>*>(bins[1].get());
    EXPECT_TRUE(q);
    EXPECT_EQ(q->get_size(), ds);
    EXPECT_EQ(q->get_num_stored_elements_per_row(), 4);
    EXPECT_TRUE(q->get_col_idxs());
#endif
}

#if GINKGO_HAVE_AMP_HALF
TEST(AMPHelpers, AllocatesEllBinsCorrectlyComplexFloat)
{
    using value_type = std::complex<float>;
    using half = gko::amp::half;
    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
    auto mnpr = gko::amp::precision_array<int, value_type>{4, 5};

    auto bins = gko::amp::allocate_bins<value_type, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 2,
                  "wrong number of bins!");
    auto p = dynamic_cast<gko::matrix::Ell<value_type, int>*>(bins[0].get());
    EXPECT_TRUE(p);
    EXPECT_EQ(p->get_size(), ds);
    EXPECT_EQ(p->get_num_stored_elements_per_row(), 4);
    auto r =
        dynamic_cast<gko::matrix::Ell<std::complex<half>, int>*>(bins[1].get());
    EXPECT_TRUE(r);
    EXPECT_EQ(r->get_size(), ds);
    EXPECT_EQ(r->get_num_stored_elements_per_row(), 5);
}
#else
TEST(AMPHelpers, AllocatesEllBinsCorrectlyComplexFloat)
{
    using value_type = std::complex<float>;
    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
    auto mnpr = gko::amp::precision_array<int, value_type>{4};

    auto bins = gko::amp::allocate_bins<value_type, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 1,
                  "wrong number of bins!");
    auto p = dynamic_cast<gko::matrix::Ell<value_type, int>*>(bins[0].get());
    EXPECT_TRUE(p);
    EXPECT_EQ(p->get_size(), ds);
    EXPECT_EQ(p->get_num_stored_elements_per_row(), 4);
}
#endif

#if GINKGO_HAVE_AMP_HALF

TEST(AMPHelpers, AllocatesEllBinsTupleCorrectlyComplexFloat)
{
    using value_type = std::complex<float>;
    using half = gko::amp::half;

    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
    auto mnpr = gko::amp::precision_array<int, value_type>{4, 5};

    auto bins = gko::amp::allocate_bins_tuple<value_type, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 2,
                  "wrong number of bins!");
    using bin0type = decltype(std::get<0>(bins));
    static_assert(
        std::is_same<bin0type,
                     std::unique_ptr<Ell<std::complex<float>, int>>&>::value,
        "Wrong static type of bin!");
    EXPECT_EQ(std::get<0>(bins)->get_size(), ds);
    EXPECT_EQ(std::get<0>(bins)->get_num_stored_elements_per_row(), 4);
    static_assert(
        std::is_same<decltype(std::get<1>(bins)),
                     std::unique_ptr<Ell<std::complex<half>, int>>&>::value,
        "Wrong static type of bin!");
    EXPECT_EQ(std::get<1>(bins)->get_size(), ds);
    EXPECT_EQ(std::get<1>(bins)->get_num_stored_elements_per_row(), 5);
}

#else

TEST(AMPHelpers, AllocatesEllBinsTupleCorrectlyComplexFloat)
{
    using value_type = std::complex<float>;

    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> ds{10, 12};
    auto mnpr = gko::amp::precision_array<int, value_type>{4};

    auto bins = gko::amp::allocate_bins_tuple<value_type, int>(exec, ds, mnpr);

    static_assert(std::tuple_size<decltype(bins)>{} == 1,
                  "wrong number of bins!");
    using bin0type = decltype(std::get<0>(bins));
    static_assert(
        std::is_same<bin0type,
                     std::unique_ptr<Ell<std::complex<float>, int>>&>::value,
        "Wrong static type of bin!");
    EXPECT_EQ(std::get<0>(bins)->get_size(), ds);
    EXPECT_EQ(std::get<0>(bins)->get_num_stored_elements_per_row(), 4);
}

#endif


TEST(AMPHelpers, ComputeLastActiveBinDoesNotFoldWhenAllBinsAreAboveThreshold)
{
    std::array<gko::int64, 3> bin_nnz{10, 10, 10};
    const auto last = gko::amp::compute_last_active_bin<gko::int64, 3>(
        bin_nnz, gko::int64{30}, 0.01f);

    EXPECT_EQ(last, 2);
    EXPECT_EQ(bin_nnz, (std::array<gko::int64, 3>{10, 10, 10}));
}


TEST(AMPHelpers, ComputeLastActiveBinFoldsSingleSparseTrailingBin)
{
    // bin 2 has only 1 out of 21 total nonzeros (< 10% threshold), so it
    // folds into bin 1. Bin 1 then holds 10 + 1 = 11 nonzeros, well above
    // the threshold, so folding stops there.
    std::array<gko::int64, 3> bin_nnz{10, 10, 1};
    const auto last = gko::amp::compute_last_active_bin<gko::int64, 3>(
        bin_nnz, gko::int64{21}, 0.1f);

    EXPECT_EQ(last, 1);
    EXPECT_EQ(bin_nnz, (std::array<gko::int64, 3>{10, 11, 0}));
}


TEST(AMPHelpers, ComputeLastActiveBinCascadesAllTheWayToBinZero)
{
    // Every bin is sparse relative to the (deliberately huge) threshold, so
    // folding cascades: bin 2 into bin 1, then the merged bin 1 into bin 0.
    std::array<gko::int64, 3> bin_nnz{1, 1, 1};
    const auto last = gko::amp::compute_last_active_bin<gko::int64, 3>(
        bin_nnz, gko::int64{3}, 1.0f);

    EXPECT_EQ(last, 0);
    EXPECT_EQ(bin_nnz, (std::array<gko::int64, 3>{3, 0, 0}));
}


TEST(AMPHelpers, ComputeLastActiveBinDoesNothingWhenRatioIsZero)
{
    // Same bin_nnz as ComputeLastActiveBinFoldsSingleSparseTrailingBin,
    // where a nonzero ratio would fold bin 2 into bin 1; with ratio 0,
    // folding must stay disabled and bin_nnz must be left untouched.
    std::array<gko::int64, 3> bin_nnz{10, 10, 1};
    const auto last = gko::amp::compute_last_active_bin<gko::int64, 3>(
        bin_nnz, gko::int64{21}, 0.0f);

    EXPECT_EQ(last, 2);
    EXPECT_EQ(bin_nnz, (std::array<gko::int64, 3>{10, 10, 1}));
}


TEST(AMPHelpers, ComputeLastActiveBinHandlesSingleBin)
{
    std::array<gko::int64, 1> bin_nnz{5};
    const auto last = gko::amp::compute_last_active_bin<gko::int64, 1>(
        bin_nnz, gko::int64{5}, 1.0f);

    // There is nowhere higher to fold to: bin 0 is always kept.
    EXPECT_EQ(last, 0);
    EXPECT_EQ(bin_nnz, (std::array<gko::int64, 1>{5}));
}


template <typename ValueIndexType>
class Amp : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::AMP<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;

    Amp() : exec(gko::ReferenceExecutor::create())
    {
        // Static tests
#if GINKGO_ENABLE_BFLOAT16 || GINKGO_ENABLE_HALF
        static_assert(gko::matrix::AMP<double, int>::num_precisions == 3,
                      "Wrong number of supported precisions for AMP<double>!");
        static_assert(gko::matrix::AMP<float, int>::num_precisions == 2,
                      "Wrong number of supported precisions for AMP<float>!");
        static_assert(
            gko::matrix::AMP<std::complex<float>, int>::num_precisions == 2,
            "Wrong number of supported precisions for AMP<complex float>!");
#else
        static_assert(gko::matrix::AMP<double, int>::num_precisions == 2,
                      "Wrong number of supported precisions for AMP<double>!");
        static_assert(gko::matrix::AMP<float, int>::num_precisions == 1,
                      "Wrong number of supported precisions for AMP<float>!");
        static_assert(
            gko::matrix::AMP<std::complex<float>, int>::num_precisions == 1,
            "Wrong number of supported precisions for AMP<complex float>!");
#endif
    }

    std::unique_ptr<Mtx> create_ampell_from_dense_ones(
        gko::dim<2> size, typename Mtx::strategy_type strat =
                              Mtx::strategy_type::monolithic_classical)
    {
        auto input = gko::share(Dense::create(exec, size));
        input->fill(gko::one<value_type>());
        auto inell = gko::share(Ell::create(exec));
        input->convert_to(inell.get());
        auto factory = Mtx::build().with_strategy(strat).on(exec);
        return factory->generate(inell);
    }

    std::unique_ptr<Mtx> create_ampcsr_from_dense_ones(
        gko::dim<2> size, typename Mtx::strategy_type strat =
                              Mtx::strategy_type::monolithic_classical)
    {
        auto input = gko::share(Dense::create(exec, size));
        input->fill(gko::one<value_type>());
        auto inmat = gko::share(Csr::create(exec));
        input->convert_to(inmat.get());
        auto factory = Mtx::build().with_strategy(strat).on(exec);
        return factory->generate(inmat);
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        for (int i = 0; i < Mtx::num_precisions; ++i) {
            ASSERT_EQ(m->get_bin_matrix(i), nullptr);
        }
    }

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(Amp, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


TYPED_TEST(Amp, HasCorrectExecutor)
{
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;

    auto empty_input = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto factory = Mtx::build().on(this->exec);
    auto mtx = factory->generate(empty_input);

    ASSERT_EQ(mtx->get_executor()->get_description(),
              this->exec->get_description());
}


TYPED_TEST(Amp, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;

    auto empty_input = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto factory = Mtx::build().on(this->exec);
    auto mtx = factory->generate(empty_input);

    ASSERT_EQ(mtx->get_size(), gko::dim<2>(0, 0));
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithDefaultParameters)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().on(this->exec);

    ASSERT_NE(factory, nullptr);
    ASSERT_EQ(factory->get_executor(), this->exec);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithCustomTolerance)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().with_tolerance(1e-6f).on(this->exec);

    EXPECT_EQ(factory->get_parameters().tolerance, 1e-6f);
}


TYPED_TEST(Amp, FactoryDefaultsToDefaultBinFoldupNnzRatio)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().on(this->exec);

    EXPECT_EQ(factory->get_parameters().bin_foldup_nnz_ratio, 0.01f);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithCustomBinFoldupNnzRatio)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().with_bin_foldup_nnz_ratio(0.25f).on(this->exec);

    EXPECT_EQ(factory->get_parameters().bin_foldup_nnz_ratio, 0.25f);
}


TYPED_TEST(Amp, FactoryDefaultsToHighPrecisionDiagonal)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().on(this->exec);

    EXPECT_EQ(factory->get_parameters().high_precision_diagonal, true);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithHighPrecisionDiagonalDisabled)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory =
        Mtx::build().with_high_precision_diagonal(false).on(this->exec);

    EXPECT_EQ(factory->get_parameters().high_precision_diagonal, false);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithNormwiseCriterion)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build()
                       .with_criterion(Mtx::criterion_type::normwise)
                       .on(this->exec);

    EXPECT_EQ(factory->get_parameters().criterion,
              Mtx::criterion_type::normwise);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithComponentwiseCriterion)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build()
                       .with_criterion(Mtx::criterion_type::componentwise)
                       .on(this->exec);

    EXPECT_EQ(factory->get_parameters().criterion,
              Mtx::criterion_type::componentwise);
}


TYPED_TEST(Amp, FactoryDefaultsToMonolithicClassicalStrategy)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().on(this->exec);

    EXPECT_EQ(factory->get_parameters().strategy,
              Mtx::strategy_type::monolithic_classical);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithIndependentBucketsStrategy)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build()
                       .with_strategy(Mtx::strategy_type::independent_buckets)
                       .on(this->exec);

    EXPECT_EQ(factory->get_parameters().strategy,
              Mtx::strategy_type::independent_buckets);
}


TYPED_TEST(Amp, FactoryDefaultsToAutomaticSubwarpSize)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().on(this->exec);

    EXPECT_EQ(factory->get_parameters().subwarp_size, 0);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithValidSubwarpSize)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().with_subwarp_size(8).on(this->exec);

    EXPECT_EQ(factory->get_parameters().subwarp_size, 8);
}


TYPED_TEST(Amp, GenerateRoundsDownInvalidSubwarpSize)
{
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;

    auto ell = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto mtx3 = Mtx::build().with_subwarp_size(3).on(this->exec)->generate(ell);
    auto mtx40 =
        Mtx::build().with_subwarp_size(40).on(this->exec)->generate(ell);
    auto mtx100 =
        Mtx::build().with_subwarp_size(100).on(this->exec)->generate(ell);

    // ReferenceExecutor has no native warp size, so validation falls back to
    // a warp size of 32.
    EXPECT_EQ(mtx3->get_parameters().subwarp_size, 2);
    EXPECT_EQ(mtx40->get_parameters().subwarp_size, 32);
    EXPECT_EQ(mtx100->get_parameters().subwarp_size, 32);
}


TYPED_TEST(Amp, FactoryDefaultsToAutomaticalCsrStrategy)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory = Mtx::build().on(this->exec);

    EXPECT_EQ(factory->get_parameters().csr_strategy,
              gko::matrix::amp_csr_strategy_type::automatical);
}


TYPED_TEST(Amp, FactoryCanBeCreatedWithClassicalCsrStrategy)
{
    using Mtx = typename TestFixture::Mtx;

    auto factory =
        Mtx::build()
            .with_csr_strategy(gko::matrix::amp_csr_strategy_type::classical)
            .on(this->exec);

    EXPECT_EQ(factory->get_parameters().csr_strategy,
              gko::matrix::amp_csr_strategy_type::classical);
}


TYPED_TEST(Amp, IndependentBucketsHonorsRequestedCsrStrategy)
{
    using Mtx = typename TestFixture::Mtx;
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;

    auto input = gko::share(Dense::create(this->exec, gko::dim<2>{3, 3}));
    input->fill(gko::one<typename TestFixture::value_type>());
    auto inmat = gko::share(Csr::create(this->exec));
    input->convert_to(inmat.get());
    auto mtx =
        Mtx::build()
            .with_strategy(Mtx::strategy_type::independent_buckets)
            .with_csr_strategy(gko::matrix::amp_csr_strategy_type::classical)
            .on(this->exec)
            ->generate(inmat);

    // Bin 0, the highest-precision bucket, shares Mtx's own value_type and
    // so is exactly Csr<value_type, index_type>. Narrower bins use their own
    // (smaller) precision and are not checked here.
    auto bin0 = dynamic_cast<const Csr*>(mtx->get_bin_matrix(0));
    ASSERT_TRUE(bin0);
    EXPECT_EQ(bin0->get_strategy()->get_name(), "classical");
}


TYPED_TEST(Amp, FactoryGenerateCompletesWithoutError)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using Ell = typename TestFixture::Ell;
    auto dinput = Dense::create(this->exec, gko::dim<2>{3, 3});
    dinput->fill(gko::one<typename TestFixture::value_type>());
    auto input = gko::share(Ell::create(this->exec));
    dinput->convert_to(input.get());
    auto factory = Mtx::build().on(this->exec);

    ASSERT_NO_THROW(auto mtx = factory->generate(input));
}


TYPED_TEST(Amp, GeneratedMatrixHasCorrectSize)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;
    auto input = gko::share(Ell::create(this->exec, gko::dim<2>{4, 5}));
    auto factory = Mtx::build().on(this->exec);

    auto mtx = factory->generate(input);

    EXPECT_EQ(mtx->get_size(), gko::dim<2>(4, 5));
    gko::constexpr_for<0, Mtx::num_precisions, 1>([&](auto k) {
        using types_list = typename gko::amp::narrow_types<value_type>::type;
        using vtype = typename std::tuple_element<k, types_list>::type;
        auto mell = dynamic_cast<const gko::matrix::Ell<vtype, index_type>*>(
            mtx->get_bin_matrix(k));
        EXPECT_EQ(mell->get_size(), input->get_size());
        EXPECT_GE(mell->get_num_stored_elements_per_row(), 0);
    });
    EXPECT_EQ(mtx->get_bin_matrix(Mtx::num_precisions), nullptr);
    EXPECT_EQ(mtx->get_bin_matrix(-1), nullptr);
}


TYPED_TEST(Amp, GetNumNonemptyBinsIsZeroForAnEmptyMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;
    auto input = gko::share(Ell::create(this->exec, gko::dim<2>{4, 5}));
    auto factory = Mtx::build().on(this->exec);

    auto mtx = factory->generate(input);

    EXPECT_EQ(mtx->get_num_nonempty_bins(), 0);
}


TYPED_TEST(Amp, GetNumNonemptyBinsCountsOnlyBinZeroForUniformMatrix)
{
    // Every entry has the same magnitude, so all of them land in bin 0
    // (the highest precision) regardless of the (default) tolerance: none
    // of the other bins are ever populated, folded or not.
    auto mtx_ell = this->create_ampell_from_dense_ones(gko::dim<2>{20, 20});
    auto mtx_csr = this->create_ampcsr_from_dense_ones(gko::dim<2>{20, 20});

    EXPECT_EQ(mtx_ell->get_num_nonempty_bins(), 1);
    EXPECT_EQ(mtx_csr->get_num_nonempty_bins(), 1);
}


TYPED_TEST(Amp, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{2, 3});

    auto copy = mtx->clone();

    auto copy_mtx = dynamic_cast<Mtx*>(copy.get());
    ASSERT_NE(copy_mtx, nullptr);
    EXPECT_EQ(copy_mtx->get_size(), mtx->get_size());
}


TYPED_TEST(Amp, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{2, 3});
    auto original_size = mtx->get_size();
    auto moved = mtx->clone();

    moved->move_from(mtx);

    auto moved_mtx = dynamic_cast<Mtx*>(moved.get());
    ASSERT_NE(moved_mtx, nullptr);
    EXPECT_EQ(moved_mtx->get_size(), original_size);
}


TYPED_TEST(Amp, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{3, 4});

    auto clone = mtx->clone();

    auto cloned_mtx = dynamic_cast<Mtx*>(clone.get());
    ASSERT_NE(cloned_mtx, nullptr);
    EXPECT_EQ(cloned_mtx->get_size(), mtx->get_size());
}


TYPED_TEST(Amp, CanBeCleared)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{2, 3});

    mtx->clear();

    this->assert_empty(mtx.get());
}


TYPED_TEST(Amp, GetBinMatrixReturnsNullForInvalidIndex)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{2, 3});

    EXPECT_EQ(mtx->get_bin_matrix(Mtx::num_precisions), nullptr);
    EXPECT_EQ(mtx->get_bin_matrix(Mtx::num_precisions + 1), nullptr);
    EXPECT_EQ(mtx->get_bin_matrix(-1), nullptr);
}


TYPED_TEST(Amp, ApplyCompletesWithoutError)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{3, 4});
    auto x = Dense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<value_type>());
    auto y = Dense::create(this->exec, gko::dim<2>{3, 1});

    ASSERT_NO_THROW(mtx->apply(x, y));
}


TYPED_TEST(Amp, ApplyCompletesWithoutErrorIndependentBuckets)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = this->create_ampell_from_dense_ones(
        gko::dim<2>{3, 4}, Mtx::strategy_type::independent_buckets);
    auto x = Dense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<value_type>());
    auto y = Dense::create(this->exec, gko::dim<2>{3, 1});

    ASSERT_NO_THROW(mtx->apply(x, y));
}


TYPED_TEST(Amp, AdvancedApplyCompletesWithoutError)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{3, 4});
    auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    auto x = Dense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<value_type>());
    auto y = Dense::create(this->exec, gko::dim<2>{3, 1});
    y->fill(gko::one<value_type>());

    ASSERT_NO_THROW(mtx->apply(alpha, x, beta, y));
}


TYPED_TEST(Amp, AdvancedApplyCompletesWithoutErrorIndependentBuckets)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = this->create_ampell_from_dense_ones(
        gko::dim<2>{3, 4}, Mtx::strategy_type::independent_buckets);
    auto alpha = gko::initialize<Dense>({2.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    auto x = Dense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<value_type>());
    auto y = Dense::create(this->exec, gko::dim<2>{3, 1});
    y->fill(gko::one<value_type>());

    ASSERT_NO_THROW(mtx->apply(alpha, x, beta, y));
}


TYPED_TEST(Amp, ApplySupportsVectorsOfDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherDense = gko::matrix::Dense<other_type>;
    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{3, 4});
    auto x = OtherDense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<other_type>());
    auto y = OtherDense::create(this->exec, gko::dim<2>{3, 1});

    ASSERT_NO_THROW(mtx->apply(x, y));
}


TYPED_TEST(Amp, ApplySupportsVectorsOfDifferentPrecisionIndependentBuckets)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherDense = gko::matrix::Dense<other_type>;
    auto mtx = this->create_ampell_from_dense_ones(
        gko::dim<2>{3, 4}, Mtx::strategy_type::independent_buckets);
    auto x = OtherDense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<other_type>());
    auto y = OtherDense::create(this->exec, gko::dim<2>{3, 1});

    ASSERT_NO_THROW(mtx->apply(x, y));
}


TYPED_TEST(Amp, AdvancedApplySupportsVectorsOfDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherDense = gko::matrix::Dense<other_type>;
    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{3, 4});
    auto alpha = gko::initialize<OtherDense>({2.0}, this->exec);
    auto beta = gko::initialize<OtherDense>({-1.0}, this->exec);
    auto x = OtherDense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<other_type>());
    auto y = OtherDense::create(this->exec, gko::dim<2>{3, 1});
    y->fill(gko::one<other_type>());

    ASSERT_NO_THROW(mtx->apply(alpha, x, beta, y));
}


TYPED_TEST(Amp,
           AdvancedApplySupportsVectorsOfDifferentPrecisionIndependentBuckets)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherDense = gko::matrix::Dense<other_type>;
    auto mtx = this->create_ampell_from_dense_ones(
        gko::dim<2>{3, 4}, Mtx::strategy_type::independent_buckets);
    auto alpha = gko::initialize<OtherDense>({2.0}, this->exec);
    auto beta = gko::initialize<OtherDense>({-1.0}, this->exec);
    auto x = OtherDense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<other_type>());
    auto y = OtherDense::create(this->exec, gko::dim<2>{3, 1});
    y->fill(gko::one<other_type>());

    ASSERT_NO_THROW(mtx->apply(alpha, x, beta, y));
}


TYPED_TEST(Amp, ApplyWithMixedPrecisionVectorsProducesCorrectResult)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherDense = gko::matrix::Dense<other_type>;

    // Create a known 3x3 matrix: all ones
    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{3, 3});

    // Apply with same-precision vectors
    auto x_same = Dense::create(this->exec, gko::dim<2>{3, 1});
    x_same->fill(gko::one<value_type>());
    auto y_same = Dense::create(this->exec, gko::dim<2>{3, 1});
    mtx->apply(x_same, y_same);

    // Apply with different-precision vectors
    auto x_other = OtherDense::create(this->exec, gko::dim<2>{3, 1});
    x_other->fill(gko::one<other_type>());
    auto y_other = OtherDense::create(this->exec, gko::dim<2>{3, 1});
    mtx->apply(x_other, y_other);

    // Convert both results to same type and compare
    auto y_same_dense = Dense::create(this->exec);
    y_same->convert_to(y_same_dense.get());
    // y_other should have the same result (3.0 in each entry for all-ones 3x3)
    GKO_ASSERT_MTX_NEAR(y_same_dense, y_other, 1e-5);
}


TYPED_TEST(
    Amp, ApplyWithMixedPrecisionVectorsProducesCorrectResultIndependentBuckets)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherDense = gko::matrix::Dense<other_type>;

    // Create a known 3x3 matrix: all ones
    auto mtx = this->create_ampell_from_dense_ones(
        gko::dim<2>{3, 3}, Mtx::strategy_type::independent_buckets);

    // Apply with same-precision vectors
    auto x_same = Dense::create(this->exec, gko::dim<2>{3, 1});
    x_same->fill(gko::one<value_type>());
    auto y_same = Dense::create(this->exec, gko::dim<2>{3, 1});
    mtx->apply(x_same, y_same);

    // Apply with different-precision vectors
    auto x_other = OtherDense::create(this->exec, gko::dim<2>{3, 1});
    x_other->fill(gko::one<other_type>());
    auto y_other = OtherDense::create(this->exec, gko::dim<2>{3, 1});
    mtx->apply(x_other, y_other);

    // Convert both results to same type and compare
    auto y_same_dense = Dense::create(this->exec);
    y_same->convert_to(y_same_dense.get());
    // y_other should have the same result (3.0 in each entry for all-ones 3x3)
    GKO_ASSERT_MTX_NEAR(y_same_dense, y_other, 1e-5);
}


TYPED_TEST(Amp, BothStrategiesProduceSameResult)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;

    auto mtx_monolithic = this->create_ampell_from_dense_ones(
        gko::dim<2>{3, 4}, Mtx::strategy_type::monolithic_classical);
    auto mtx_independent = this->create_ampell_from_dense_ones(
        gko::dim<2>{3, 4}, Mtx::strategy_type::independent_buckets);
    auto x = Dense::create(this->exec, gko::dim<2>{4, 1});
    x->fill(gko::one<value_type>());
    auto y_monolithic = Dense::create(this->exec, gko::dim<2>{3, 1});
    auto y_independent = Dense::create(this->exec, gko::dim<2>{3, 1});

    mtx_monolithic->apply(x, y_monolithic);
    mtx_independent->apply(x, y_independent);

    GKO_ASSERT_MTX_NEAR(y_monolithic, y_independent, 1e-5);
}


TYPED_TEST(Amp, CanConvertToDense)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto size = gko::dim<2>{2, 3};
    auto mtx = this->create_ampell_from_dense_ones(size);
    auto dense = Dense::create(this->exec);
    auto ref = gko::share(Dense::create(this->exec, size));
    ref->fill(gko::one<value_type>());

    mtx->convert_to(dense.get());

    EXPECT_EQ(dense->get_size(), mtx->get_size());
    GKO_ASSERT_MTX_NEAR(dense, ref, 0.0);
}


TYPED_TEST(Amp, CanMoveToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{2, 3});
    const auto original_size = mtx->get_size();
    const auto zero_size = gko::dim<2>{0, 0};
    auto dense = Dense::create(this->exec);

    mtx->move_to(dense.get());

    EXPECT_EQ(dense->get_size(), original_size);
}


TYPED_TEST(Amp, CanExtractDiagonal)
{
    using Mtx = typename TestFixture::Mtx;

    auto mtx = this->create_ampell_from_dense_ones(gko::dim<2>{3, 4});

    auto diag = mtx->extract_diagonal();

    ASSERT_NE(diag, nullptr);
    EXPECT_EQ(diag->get_size()[0],
              std::min(mtx->get_size()[0], mtx->get_size()[1]));
}


TYPED_TEST(Amp, ReadFromMatrixDataProducesCorrectSizeAndBinTypesEll)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;
    constexpr int q = gko::amp::narrow_types<value_type>::num_types;

    gko::matrix_data<value_type, index_type> data{
        {3, 3},
        {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {2, 2, 3.0}}};

    // Create empty AMP via factory with empty ELL input
    auto ell_empty = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto mtx = Mtx::build().on(this->exec)->generate(ell_empty);
    mtx->read(data);

    EXPECT_EQ(mtx->get_size(), gko::dim<2>(3, 3));
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using T = typename std::tuple_element<
            k, typename gko::amp::narrow_types<value_type>::type>::type;
        auto mptr = dynamic_cast<const gko::matrix::Ell<T, index_type>*>(
            mtx->get_bin_matrix(k));
        EXPECT_TRUE(mptr) << " Ell bin " << k;
    });
}


TYPED_TEST(Amp, ReadFromMatrixDataProducesCorrectSizeAndBinTypesCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    constexpr int q = gko::amp::narrow_types<value_type>::num_types;

    gko::matrix_data<value_type, index_type> data{
        {3, 3},
        {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {2, 2, 3.0}}};

    // Create empty AMP via factory with empty CSR input
    auto base_empty = gko::share(Csr::create(this->exec, gko::dim<2>{0, 0}));
    auto mtx = Mtx::build().on(this->exec)->generate(base_empty);
    mtx->read(data);

    EXPECT_EQ(mtx->get_size(), gko::dim<2>(3, 3));
    EXPECT_EQ(mtx->get_max_nnz_per_row_for_bin(0), 2);
    EXPECT_EQ(mtx->get_max_nnz_per_row_for_bin(1), 0);
    EXPECT_EQ(mtx->get_max_nnz_per_row_for_bin(2), 0);
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using T = typename std::tuple_element<
            k, typename gko::amp::narrow_types<value_type>::type>::type;
        auto mptr = dynamic_cast<const gko::matrix::Csr<T, index_type>*>(
            mtx->get_bin_matrix(k));
        EXPECT_TRUE(mptr) << " Csr bin " << k;
    });
}


TYPED_TEST(Amp, BinFoldupCascadesSparseTrailingBinIntoBinZero)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = typename TestFixture::Dense;
    constexpr int nrows = 200;

    // Every row has only a diagonal entry of magnitude 1e6, except the
    // last row, which also has one off-diagonal entry of magnitude 10. At
    // tolerance 1e-6, the lowest-precision bin's lower bound is
    // row_norm * tolerance ~= 1, and the next bin up starts at
    // row_norm * tolerance / epsilon(lowest precision type), which is at
    // least ~128 (bfloat16's epsilon ~1/128) or ~1024 (fp16's epsilon
    // ~1/1024) -- so 10 reliably lands in the lowest-precision bin (and
    // well above its underflow threshold) regardless of which of the two
    // is configured as the narrowest supported type. That lone
    // off-diagonal entry is the only one that ever lands outside bin 0,
    // and it is far too sparse (1 out of 201 nonzeros) to survive folding
    // at the default 1% threshold, so with folding enabled it is merged
    // all the way back into bin 0.
    gko::matrix_data<value_type, index_type> data(gko::dim<2>{nrows, nrows});
    for (int i = 0; i < nrows; i++) {
        data.nonzeros.emplace_back(i, i, value_type{1000000.0});
    }
    data.nonzeros.emplace_back(nrows - 1, 0, value_type{10.0});
    data.sort_row_major();

    auto base_empty = gko::share(Csr::create(this->exec, gko::dim<2>{0, 0}));
    auto folded = Mtx::build()
                      .with_tolerance(1e-6f)
                      .with_bin_foldup_nnz_ratio(0.01f)
                      .on(this->exec)
                      ->generate(base_empty);
    folded->read(data);
    auto unfolded = Mtx::build()
                        .with_tolerance(1e-6f)
                        .with_bin_foldup_nnz_ratio(0.0f)
                        .on(this->exec)
                        ->generate(base_empty);
    unfolded->read(data);

    if constexpr (Mtx::num_precisions >= 2) {
        // Without folding, the tiny entry survives in the lowest bin.
        EXPECT_EQ(unfolded->get_num_nonempty_bins(), 2);
        EXPECT_GT(
            unfolded->get_max_nnz_per_row_for_bin(Mtx::num_precisions - 1), 0);

        // With folding, it is merged all the way back into bin 0, leaving
        // every other bin empty.
        EXPECT_EQ(folded->get_num_nonempty_bins(), 1);
        EXPECT_EQ(folded->get_max_nnz_per_row_for_bin(Mtx::num_precisions - 1),
                  0);
    } else {
        // Only bin 0 exists, so there is nothing to fold either way.
        EXPECT_EQ(unfolded->get_num_nonempty_bins(), 1);
        EXPECT_EQ(folded->get_num_nonempty_bins(), 1);
    }

    // Every stored value here is a small integer, exactly representable in
    // every candidate precision
    auto dense_folded = Dense::create(this->exec);
    auto dense_unfolded = Dense::create(this->exec);
    folded->convert_to(dense_folded.get());
    unfolded->convert_to(dense_unfolded.get());
    GKO_ASSERT_MTX_NEAR(dense_folded, dense_unfolded, 0.0);
}


TYPED_TEST(Amp, ReadFromEllProducesSameResultAsFactoryGenerate)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;
    using Dense = typename TestFixture::Dense;

    gko::matrix_data<value_type, index_type> data{
        {3, 3},
        {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {2, 2, 3.0}}};

    // Method 1: read directly (via empty factory-generated AMP)
    auto ell_empty = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto mtx_read = Mtx::build().on(this->exec)->generate(ell_empty);
    mtx_read->read(data);

    // Method 2: factory generate from ELL
    auto ell = gko::share(Ell::create(this->exec));
    ell->read(data);
    auto mtx_gen = Mtx::build().on(this->exec)->generate(ell);

    // Both should produce the same dense result
    auto dense_read = Dense::create(this->exec);
    auto dense_gen = Dense::create(this->exec);
    mtx_read->convert_to(dense_read.get());
    mtx_gen->convert_to(dense_gen.get());

    GKO_ASSERT_MTX_NEAR(dense_read, dense_gen, 0.0);
}


TYPED_TEST(Amp, ReadFromCsrProducesSameResultAsFactoryGenerate)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using BaseMtx = gko::matrix::Csr<value_type, index_type>;
    using Dense = typename TestFixture::Dense;

    gko::matrix_data<value_type, index_type> data{
        {3, 3},
        {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {2, 2, 3.0}}};

    // Method 1: read directly (via empty factory-generated AMP)
    auto base_empty =
        gko::share(BaseMtx::create(this->exec, gko::dim<2>{0, 0}));
    auto mtx_read = Mtx::build().on(this->exec)->generate(base_empty);
    mtx_read->read(data);

    // Method 2: factory generate from CSR
    auto base = gko::share(BaseMtx::create(this->exec));
    base->read(data);
    auto mtx_gen = Mtx::build().on(this->exec)->generate(base);

    // Both should produce the same dense result
    auto dense_read = Dense::create(this->exec);
    auto dense_gen = Dense::create(this->exec);
    mtx_read->convert_to(dense_read.get());
    mtx_gen->convert_to(dense_gen.get());

    GKO_ASSERT_MTX_NEAR(dense_read, dense_gen, 0.0);
}


TYPED_TEST(Amp, ClonePreservesParameters)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;

    const float custom_tol = 1e-3f;
    auto ell = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto mtx =
        Mtx::build()
            .with_tolerance(custom_tol)
            .with_criterion(Mtx::criterion_type::normwise)
            .with_strategy(Mtx::strategy_type::independent_buckets)
            .with_subwarp_size(8)
            .with_csr_strategy(gko::matrix::amp_csr_strategy_type::classical)
            .on(this->exec)
            ->generate(ell);

    auto clone = gko::clone(mtx);

    EXPECT_EQ(clone->get_parameters().tolerance, custom_tol);
    EXPECT_EQ(clone->get_parameters().criterion, Mtx::criterion_type::normwise);
    EXPECT_EQ(clone->get_parameters().strategy,
              Mtx::strategy_type::independent_buckets);
    EXPECT_EQ(clone->get_parameters().subwarp_size, 8);
    EXPECT_EQ(clone->get_parameters().csr_strategy,
              gko::matrix::amp_csr_strategy_type::classical);
}


TYPED_TEST(Amp, ClonePreservesMaxNNZPerRowForCsrBins)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;
    auto mtx = this->create_ampcsr_from_dense_ones(gko::dim<2>{3, 3});
    ASSERT_EQ(mtx->get_max_nnz_per_row_for_bin(0), 3);
    for (int k = 1; k < 3; k++) {
        ASSERT_EQ(mtx->get_max_nnz_per_row_for_bin(k), 0);
    }

    auto clone = gko::clone(mtx);

    EXPECT_EQ(clone->get_max_nnz_per_row_for_bin(0), 3);
    for (int k = 1; k < 3; k++) {
        EXPECT_EQ(clone->get_max_nnz_per_row_for_bin(k), 0);
    }
}


TYPED_TEST(Amp, ReadAfterCloneUsesPreservedParameters)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Ell = typename TestFixture::Ell;
    using Dense = typename TestFixture::Dense;

    const float custom_tol = 1e-6f;
    gko::matrix_data<value_type, index_type> data{
        {3, 3},
        {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {2, 2, 3.0}}};

    // Create configured empty AMP, clone it, then read data into clone
    auto ell_empty = gko::share(Ell::create(this->exec, gko::dim<2>{0, 0}));
    auto amp_template = Mtx::build()
                            .with_tolerance(custom_tol)
                            .on(this->exec)
                            ->generate(ell_empty);
    auto amp_clone = gko::clone(amp_template);
    amp_clone->read(data);

    ASSERT_EQ(amp_clone->get_size(), gko::dim<2>(3, 3));
    EXPECT_EQ(amp_clone->get_parameters().tolerance, custom_tol);
    // Verify it produces a valid matrix by converting to dense
    auto dense = Dense::create(this->exec);
    amp_clone->convert_to(dense.get());
    EXPECT_EQ(dense->get_size(), gko::dim<2>(3, 3));
}


TYPED_TEST(Amp, FactoryGenerateWorksWithCsrInput)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    auto dinput = Dense::create(this->exec, gko::dim<2>{3, 3});
    dinput->fill(gko::one<value_type>());
    auto csr_input = gko::share(Csr::create(this->exec));
    dinput->convert_to(csr_input.get());
    auto factory = Mtx::build().on(this->exec);

    std::unique_ptr<Mtx> mtx;
    ASSERT_NO_THROW(mtx = factory->generate(csr_input));

    EXPECT_EQ(mtx->get_size(), csr_input->get_size());
    gko::constexpr_for<0, Mtx::num_precisions, 1>([&](auto k) {
        using types_list = typename gko::amp::narrow_types<value_type>::type;
        using vtype = typename std::tuple_element<k, types_list>::type;
        auto mcsr = dynamic_cast<const gko::matrix::Csr<vtype, index_type>*>(
            mtx->get_bin_matrix(k));
        ASSERT_NE(mcsr, nullptr);
        EXPECT_EQ(mcsr->get_size(), csr_input->get_size());
    });
}
