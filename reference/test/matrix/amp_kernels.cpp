// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <memory>
#include <numeric>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/test/utils.hpp"
#include "reference/matrix/amp_algorithms.hpp"


namespace {

namespace gkra = gko::kernels::reference::amp;

#if GKO_AMP_HALF_IS_FP16 || GKO_AMP_HALF_IS_BFLOAT16

TEST(AMPAlgorithm, GetsCorrectBinLowerBoundsByPrecision)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    const auto lbs1 =
        gkra::get_bins_precision_lower_bounds<double>(rownorm, tol);
    EXPECT_FLOAT_EQ(lbs1[0],
                    rownorm * tol / std::numeric_limits<float>::epsilon());
    EXPECT_FLOAT_EQ(
        lbs1[1],
        rownorm * tol / std::numeric_limits<gko::amp::half>::epsilon());
    EXPECT_FLOAT_EQ(lbs1[2], rownorm * tol);

    const auto lbs2 =
        gkra::get_bins_precision_lower_bounds<float>(rownorm, tol);
    EXPECT_FLOAT_EQ(
        lbs2[0],
        rownorm * tol / std::numeric_limits<gko::amp::half>::epsilon());
    EXPECT_FLOAT_EQ(lbs2[1], rownorm * tol);
}


TEST(AMPAlgorithm, GetsCorrectBinMinRepresentable)
{
    const auto mins_d = gkra::get_bins_min_representable<double>();
    auto doublemin = std::numeric_limits<double>::min();
    auto halfmin =
        static_cast<double>(std::numeric_limits<gko::amp::half>::min());
    EXPECT_EQ(mins_d[0], doublemin);
    EXPECT_EQ(mins_d[1],
              static_cast<double>(std::numeric_limits<float>::min()));
    EXPECT_EQ(mins_d[2], static_cast<double>(halfmin));

    const auto mins_f = gkra::get_bins_min_representable<float>();
    EXPECT_EQ(mins_f[0], std::numeric_limits<float>::min());
    EXPECT_EQ(mins_f[1], static_cast<float>(halfmin));
}


TEST(AMPAlgorithm, GetsCorrectPrecisionBin)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    const auto lbs =
        gkra::get_bins_precision_lower_bounds<double>(rownorm, tol);

    // Value larger than lb[0] goes to bin 0 (double)
    auto bin0 = gkra::get_precision_bin<double>(lbs, lbs[0] * 2.0, 0);
    EXPECT_EQ(bin0, 0);

    // Value between lb[0] and lb[1] goes to bin 1 (float)
    const double val_bin1 = (lbs[0] + lbs[1]) / 2.0;
    const auto bin1 = gkra::get_precision_bin<double>(lbs, val_bin1, 0);
    EXPECT_EQ(bin1, 1);

    // Value between lb[1] and lb[2] goes to bin 2 (half)
    const double val_bin2 = (lbs[1] + lbs[2]) / 2.0;
    const auto bin2 = gkra::get_precision_bin<double>(lbs, val_bin2, 0);
    EXPECT_EQ(bin2, 2);

    // Value smaller than lb[2] gets dropped (returns -1)
    const auto bin_drop = gkra::get_precision_bin<double>(lbs, lbs[2] * 0.5, 0);
    EXPECT_EQ(bin_drop, -1);

    // Starting from bin 1 should skip bin 0
    const auto bin_skip = gkra::get_precision_bin<double>(lbs, lbs[0] * 2.0, 1);
    EXPECT_EQ(bin_skip, 1);

    // Test with float as base type
    const auto lbs_f =
        gkra::get_bins_precision_lower_bounds<float>(rownorm, tol);
    const auto bin_f0 =
        gkra::get_precision_bin<float>(lbs_f, lbs_f[0] * 2.0f, 0);
    EXPECT_EQ(bin_f0, 0);
    const auto bin_f_drop =
        gkra::get_precision_bin<float>(lbs_f, lbs_f[1] * 0.5f, 0);
    EXPECT_EQ(bin_f_drop, -1);
}


TEST(AMPAlgorithm, AdjustsBinForUnderflow)
{
    const auto mins = gkra::get_bins_min_representable<double>();

    // Value representable in bin 2 stays in bin 2
    double val_ok = static_cast<double>(mins[2]) * 2.0;
    auto adj_ok = gkra::adjust_bin_for_underflow<double>(mins, val_ok, 2);
    EXPECT_EQ(adj_ok, 2);

    // Value below min of bin 2 should move to a higher-precision bin
    double val_underflow_half = static_cast<double>(mins[2]) * 0.5;
    int adjusted =
        gkra::adjust_bin_for_underflow<double>(mins, val_underflow_half, 2);
    ASSERT_GE(adjusted, 0);
    EXPECT_LT(adjusted, 2);
#if GKO_AMP_HALF_IS_FP16
    EXPECT_EQ(adjusted, 1);
#else
    EXPECT_EQ(adjusted, 0);
#endif
    // Should be representable in the adjusted bin
    EXPECT_GE(val_underflow_half, static_cast<double>(mins[adjusted]));

    // Too small a number that was originally in half bin goes to double bin.
    const double val_underflow_fl = static_cast<double>(mins[1]) * 0.5;
    const int adjusted_fl =
        gkra::adjust_bin_for_underflow<double>(mins, val_underflow_fl, 2);
    EXPECT_EQ(adjusted_fl, 0);
    // Should be representable in the adjusted bin
    EXPECT_GE(val_underflow_fl, static_cast<double>(mins[adjusted_fl]));

    // Dropped values (bin -1) stay dropped
    auto adj_drop = gkra::adjust_bin_for_underflow<double>(mins, 1e-100, -1);
    EXPECT_EQ(adj_drop, -1);

    // Bin 0 stays at bin 0 even for tiny values
    auto adj_tiny = gkra::adjust_bin_for_underflow<double>(mins, 1e-320, 0);
    EXPECT_EQ(adj_tiny, 0);

    // Test with float as base type
    const auto mins_f = gkra::get_bins_min_representable<float>();
    float val_ok_f = mins_f[1] * 2.0f;
    auto adj_f = gkra::adjust_bin_for_underflow<float>(mins_f, val_ok_f, 1);
    EXPECT_EQ(adj_f, 1);
}


TEST(AMPAlgorithm, GetsAdjustedBin)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    const auto lbs =
        gkra::get_bins_precision_lower_bounds<double>(rownorm, tol);
    const auto mins = gkra::get_bins_min_representable<double>();

    // Large value goes to bin 0
    auto bin_large = gkra::get_adjusted_bin<double>(lbs, mins, lbs[0] * 2.0);
    EXPECT_EQ(bin_large, 0);

    // Value in middle range: precision bin then adjusted for underflow
    double val_mid = (lbs[0] + lbs[1]) / 2.0;
    int bin_mid = gkra::get_adjusted_bin<double>(lbs, mins, val_mid);
    // Should be assigned to some bin (precision determined, then underflow
    // adjusted)
    EXPECT_GE(bin_mid, 0);
    // Should be representable in the assigned bin
    EXPECT_GE(val_mid, static_cast<double>(mins[bin_mid]));

    // Values just smaller than FP16 min are put in float bin
    //  but those smaller than bfloat16 min are discarded.
    const double val_under = mins[2] / 1.1;
    const int bin_under = gkra::get_adjusted_bin<double>(lbs, mins, val_under);
#if GKO_AMP_HALF_IS_FP16
    EXPECT_EQ(bin_under, 1);
#else
    EXPECT_EQ(bin_under, -1);
#endif

    // Very small value gets dropped
    auto bin_drop = gkra::get_adjusted_bin<double>(lbs, mins, lbs[2] * 0.5);
    EXPECT_EQ(bin_drop, -1);

    // Test with float as base type
    const auto lbs_f =
        gkra::get_bins_precision_lower_bounds<float>(rownorm, tol);
    const auto mins_f = gkra::get_bins_min_representable<float>();
    auto bin_f0 = gkra::get_adjusted_bin<float>(lbs_f, mins_f, lbs_f[0] * 2.0f);
    EXPECT_EQ(bin_f0, 0);
    auto bin_f_drop =
        gkra::get_adjusted_bin<float>(lbs_f, mins_f, lbs_f[1] * 0.5f);
    EXPECT_EQ(bin_f_drop, -1);
}


#else  // Only double and float available

TEST(AMPAlgorithm, GetsCorrectBinLowerBoundsByPrecision)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    const auto lbs1 =
        gkra::get_bins_precision_lower_bounds<double>(rownorm, tol);
    EXPECT_FLOAT_EQ(lbs1[0],
                    rownorm * tol / std::numeric_limits<float>::epsilon());
    EXPECT_FLOAT_EQ(lbs1[1], rownorm * tol);

    const auto lbs2 =
        gkra::get_bins_precision_lower_bounds<float>(rownorm, tol);
    EXPECT_FLOAT_EQ(lbs2[0], rownorm * tol);
}


TEST(AMPAlgorithm, GetsCorrectBinMinRepresentable)
{
    const auto mins_d = gkra::get_bins_min_representable<double>();
    auto doublemin = std::numeric_limits<double>::min();
    EXPECT_EQ(mins_d[0], doublemin);
    EXPECT_EQ(mins_d[1],
              static_cast<double>(std::numeric_limits<float>::min()));

    const auto mins_f = gkra::get_bins_min_representable<float>();
    EXPECT_EQ(mins_f[0], std::numeric_limits<float>::min());
}


TEST(AMPAlgorithm, GetsCorrectPrecisionBin)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    const auto lbs =
        gkra::get_bins_precision_lower_bounds<double>(rownorm, tol);

    // Value larger than lb[0] goes to bin 0 (double)
    auto bin0 = gkra::get_precision_bin<double>(lbs, lbs[0] * 2.0, 0);
    EXPECT_EQ(bin0, 0);

    // Value between lb[0] and lb[1] goes to bin 1 (float)
    double val_bin1 = (lbs[0] + lbs[1]) / 2.0;
    auto bin1 = gkra::get_precision_bin<double>(lbs, val_bin1, 0);
    EXPECT_EQ(bin1, 1);

    // Value smaller than lb[1] gets dropped (returns -1)
    auto bin_drop = gkra::get_precision_bin<double>(lbs, lbs[1] * 0.5, 0);
    EXPECT_EQ(bin_drop, -1);

    // Starting from bin 1 should skip bin 0
    auto bin_skip = gkra::get_precision_bin<double>(lbs, lbs[0] * 2.0, 1);
    EXPECT_EQ(bin_skip, 1);

    // Test with float as base type (1 bin only)
    const auto lbs_f =
        gkra::get_bins_precision_lower_bounds<float>(rownorm, tol);
    auto bin_f0 = gkra::get_precision_bin<float>(lbs_f, lbs_f[0] * 2.0f, 0);
    EXPECT_EQ(bin_f0, 0);
    auto bin_f_drop = gkra::get_precision_bin<float>(lbs_f, lbs_f[0] * 0.5f, 0);
    EXPECT_EQ(bin_f_drop, -1);
}


TEST(AMPAlgorithm, AdjustsBinForUnderflow)
{
    const auto mins = gkra::get_bins_min_representable<double>();

    // Value representable in bin 1 stays in bin 1
    double val_ok = static_cast<double>(mins[1]) * 2.0;
    auto adj_ok = gkra::adjust_bin_for_underflow<double>(mins, val_ok, 1);
    EXPECT_EQ(adj_ok, 1);

    // Value below min of bin 1 but above min of bin 0 moves to bin 0
    double val_underflow_float = static_cast<double>(mins[1]) * 0.5;
    auto adj_underflow =
        gkra::adjust_bin_for_underflow<double>(mins, val_underflow_float, 1);
    EXPECT_EQ(adj_underflow, 0);

    // Dropped values (bin -1) stay dropped
    auto adj_drop = gkra::adjust_bin_for_underflow<double>(mins, 1e-100, -1);
    EXPECT_EQ(adj_drop, -1);

    // Bin 0 stays at bin 0 even for tiny values
    auto adj_tiny = gkra::adjust_bin_for_underflow<double>(mins, 1e-320, 0);
    EXPECT_EQ(adj_tiny, 0);
}


TEST(AMPAlgorithm, GetsAdjustedBin)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    const auto lbs =
        gkra::get_bins_precision_lower_bounds<double>(rownorm, tol);
    const auto mins = gkra::get_bins_min_representable<double>();

    // Large value goes to bin 0
    auto bin_large = gkra::get_adjusted_bin<double>(lbs, mins, lbs[0] * 2.0);
    EXPECT_EQ(bin_large, 0);

    // Value that would go to bin 1 and is representable stays in bin 1
    double val_bin1 = (lbs[0] + lbs[1]) / 2.0;
    if (val_bin1 >= static_cast<double>(mins[1])) {
        auto bin1 = gkra::get_adjusted_bin<double>(lbs, mins, val_bin1);
        EXPECT_EQ(bin1, 1);
    }

    // Very small value gets dropped
    auto bin_drop = gkra::get_adjusted_bin<double>(lbs, mins, lbs[1] * 0.5);
    EXPECT_EQ(bin_drop, -1);
}


#endif


TEST(AMPAlgorithm, AssignsValueToTuple)
{
    auto t = std::make_tuple(3.0, -2.0f, -3, 'd');

    gkra::assign_value_to_tuple<0>(t, 1.0f, 1);
    gkra::assign_value_to_tuple<0>(t, 5, 2);
    gkra::assign_value_to_tuple<0>(t, 5, 3);
    // should do nothing:
    gkra::assign_value_to_tuple<0>(t, -6.4, 10);
    gkra::assign_value_to_tuple<0>(t, 'y', -2);

    EXPECT_EQ(std::get<0>(t), 3.0);
    EXPECT_EQ(std::get<1>(t), 1.0f);
    EXPECT_EQ(std::get<2>(t), 5);
    EXPECT_EQ(std::get<3>(t), 5);
}

TEST(AMPAlgorithm, AssignsValueToArrayTuple)
{
    double arr0[3] = {0.0, 0.0, 0.0};
    float arr1[3] = {0.0f, 0.0f, 0.0f};
    int arr2[3] = {0, 0, 0};
    auto t = std::make_tuple(arr0, arr1, arr2);

    // Assign values to different positions
    gkra::assign_value_to_array_tuple<0>(t, 1.5, 0, 0);   // arr0[0] = 1.5
    gkra::assign_value_to_array_tuple<0>(t, 2.5, 0, 2);   // arr0[2] = 2.5
    gkra::assign_value_to_array_tuple<0>(t, 3.5f, 1, 1);  // arr1[1] = 3.5
    gkra::assign_value_to_array_tuple<0>(t, 42, 2, 0);    // arr2[0] = 42

    EXPECT_EQ(arr0[0], 1.5);
    EXPECT_EQ(arr0[1], 0.0);
    EXPECT_EQ(arr0[2], 2.5);
    EXPECT_EQ(arr1[0], 0.0f);
    EXPECT_EQ(arr1[1], 3.5f);
    EXPECT_EQ(arr1[2], 0.0f);
    EXPECT_EQ(arr2[0], 42);
    EXPECT_EQ(arr2[1], 0);
    EXPECT_EQ(arr2[2], 0);
}


TEST(AMPAlgorithm, AssignsValueToArrayTupleWithOutOfRangeIndex)
{
    double arr0[2] = {1.0, 2.0};
    float arr1[2] = {3.0f, 4.0f};
    auto t = std::make_tuple(arr0, arr1);

    // Out-of-range tuple index should be a no-op (no crash, no modification)
    gkra::assign_value_to_array_tuple<0>(t, 99.0, 5,
                                         0);  // idx 5 is out of range
    gkra::assign_value_to_array_tuple<0>(t, 99.0, -1,
                                         0);  // idx -1 is out of range

    // Values should remain unchanged
    EXPECT_EQ(arr0[0], 1.0);
    EXPECT_EQ(arr0[1], 2.0);
    EXPECT_EQ(arr1[0], 3.0f);
    EXPECT_EQ(arr1[1], 4.0f);
}


template <typename ValueType>
class AMPDouble : public ::testing::Test {
protected:
    using value_type = ValueType;
    using index_type = int;
    using Mtx = gko::matrix::AMP<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;
    using Dns = gko::matrix::Dense<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;

    AMPDouble() : exec(gko::ReferenceExecutor::create())
    {
        // clang-format off
        mtx1 = gko::initialize<Dns>({{1.1, 3.0e-9, 0.0, 4.5e-4},
                                     {0.0, 1.2e-11, 2.0, 0.0},
                                     {0.0, 0.0, 0.8, 0.0},
                                     {1.2e-11, 0.0, 1.6e-4, 0.0},
                                     {-2e-5, 0.0, -2.0, 0.0}}, exec);
        mtx2 = gko::initialize<Dns>(
            {{1.0, 3.0, 2.0},
             {0.0, 5.0, 0.0}}, exec);
        // clang-format on
        ell1 = gko::share(Ell::create(exec));
        mtx1->convert_to(ell1.get());
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Dns> mtx1;
    std::unique_ptr<Dns> mtx2;
    std::shared_ptr<Ell> ell1;
    const float tol = 1e-10;
};

using double_types = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(AMPDouble, double_types, TypenameNameGenerator);


TYPED_TEST(AMPDouble, GenerateComputesCorrectRowNorms)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<typename TestFixture::value_type>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
    gko::amp::precision_array<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), this->tol, max_nnz, rownorms);

    auto rnv = rownorms.get_const_data();
    EXPECT_EQ(rnv[0], static_cast<real_T>(1.1) + static_cast<real_T>(3e-9) +
                          static_cast<real_T>(4.5e-4));
    EXPECT_EQ(rnv[1], static_cast<real_T>(2.0) + static_cast<real_T>(1.2e-11));
    EXPECT_EQ(rnv[2], static_cast<real_T>(0.8));
    EXPECT_EQ(rnv[3],
              static_cast<real_T>(1.2e-11) + static_cast<real_T>(1.6e-4));
    EXPECT_EQ(rnv[4], static_cast<real_T>(2.0) + static_cast<real_T>(2e-5));
}

TYPED_TEST(AMPDouble, GenerateComputesCorrectBinNNZs)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<typename TestFixture::value_type>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
#if GINKGO_HAVE_AMP_HALF
    static_assert(
        std::tuple_size<gko::amp::precision_array<int, T>>::value == 3,
        "should be 3 available precisions");
#else
    static_assert(
        std::tuple_size<gko::amp::precision_array<int, T>>::value == 2,
        "should be 2 available precisions");
#endif
    gko::amp::precision_array<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), this->tol, max_nnz, rownorms);

#if GKO_AMP_HALF_IS_FP16
    EXPECT_EQ(max_nnz[0], 1);
    EXPECT_EQ(max_nnz[1], 2);
    EXPECT_EQ(max_nnz[2], 0);
#elif GKO_AMP_HALF_IS_BFLOAT16
    EXPECT_EQ(max_nnz[0], 1);
    EXPECT_EQ(max_nnz[1], 1);
    EXPECT_EQ(max_nnz[2], 1);
#else
    EXPECT_EQ(max_nnz[0], 1);
    EXPECT_EQ(max_nnz[1], 2);
#endif
}

TYPED_TEST(AMPDouble, GenerateEllScattersBinsCorrectly)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<typename TestFixture::value_type>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
#if GINKGO_HAVE_AMP_HALF
    static_assert(
        std::tuple_size<gko::amp::precision_array<int, T>>::value == 3,
        "should be 3 available precisions");
#else
    static_assert(
        std::tuple_size<gko::amp::precision_array<int, T>>::value == 2,
        "should be 2 available precisions");
#endif
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
#if GKO_AMP_HALF_IS_FP16
    auto max_nnzs = gko::amp::precision_array<int, T>{1, 2, 0};
#elif GKO_AMP_HALF_IS_BFLOAT16
    auto max_nnzs = gko::amp::precision_array<int, T>{1, 1, 1};
#else
    auto max_nnzs = gko::amp::precision_array<int, T>{1, 2};
#endif
    auto abins = gko::amp::allocate_bins<T, int>(
        this->exec, this->ell1->get_size(), max_nnzs);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    gko::amp::precision_array<gko::LinOp*, T> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    gko::kernels::reference::amp::generate_ell_scatter_bins(
        rexec, this->ell1.get(), this->tol, amat);

    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, num_bins, 1>([&](auto k) {
        using value_type = typename std::tuple_element<k, types_list>::type;
        auto amat0 = dynamic_cast<gko::matrix::Ell<value_type, int>*>(amat[k]);
        ASSERT_TRUE(amat0);
        const auto nnzrow = amat0->get_num_stored_elements_per_row();
        auto vals = amat0->get_const_values();
        auto colids = amat0->get_const_col_idxs();
        if (k == 0) {
            EXPECT_EQ(nnzrow, 1);
            EXPECT_EQ(colids[0], 0);
            EXPECT_EQ(colids[1], 2);
            EXPECT_EQ(colids[2], 2);
            EXPECT_EQ(colids[3], 2);
            EXPECT_EQ(colids[4], 2);
            EXPECT_EQ(vals[0], static_cast<value_type>(1.1));
            EXPECT_EQ(vals[1], static_cast<value_type>(2.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.8));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.6e-4));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2.0));
        }
#if GKO_AMP_HALF_IS_FP16
        else if (k == 1) {
            EXPECT_EQ(nnzrow, 2);
            EXPECT_EQ(colids[0], 1);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], 0);
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(colids[5], 3);
            EXPECT_EQ(colids[6], gko::invalid_index<int>());
            EXPECT_EQ(colids[7], gko::invalid_index<int>());
            EXPECT_EQ(colids[8], gko::invalid_index<int>());
            EXPECT_EQ(colids[9], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(3e-9));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.2e-11));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
            EXPECT_EQ(vals[5], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[6], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[7], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[8], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[9], static_cast<value_type>(0.0));
        } else if (k == 2) {
            EXPECT_EQ(nnzrow, 0);
            EXPECT_FALSE(vals);
            EXPECT_FALSE(colids);
        }
#elif GKO_AMP_HALF_IS_BFLOAT16
        else if (k == 1) {
            EXPECT_EQ(nnzrow, 1);
            EXPECT_EQ(colids[0], 3);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], 0);
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(vals[0], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.2e-11));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
        } else if (k == 2) {
            EXPECT_EQ(colids[0], 1);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], gko::invalid_index<int>());
            EXPECT_EQ(colids[4], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(3e-9));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[4], static_cast<value_type>(0.0));
        }
#else
        // Only double and float available (no half)
        else if (k == 1) {
            EXPECT_EQ(nnzrow, 2);
            EXPECT_EQ(colids[0], 1);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], 0);
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(colids[5], 3);
            EXPECT_EQ(colids[6], gko::invalid_index<int>());
            EXPECT_EQ(colids[7], gko::invalid_index<int>());
            EXPECT_EQ(colids[8], gko::invalid_index<int>());
            EXPECT_EQ(colids[9], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(3e-9));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.2e-11));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
            EXPECT_EQ(vals[5], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[6], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[7], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[8], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[9], static_cast<value_type>(0.0));
        }
#endif
    });
}

TYPED_TEST(AMPDouble, ApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<T>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create test vector (matrix is 5x4)
    auto x = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    // Compute y_amp = AMP * x
    auto y_amp =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 1});

    amp_mtx->apply(x, y_amp);

    // Compute y_ref = original * x
    auto y_ref =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 1});
    this->ell1->apply(x, y_ref);
    // Check relative componentwise error
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        auto ref_val = y_ref_vals[i];
        auto amp_val = y_amp_vals[i];
        // real_T abs_ref = 0;
        // for(int j = 0; j < this->mtx1->get_size()[1]; j++) {
        //     abs_ref += std::abs(this->mtx1->at(i,j)*x->at(j));
        // }
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-14});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}

TYPED_TEST(AMPDouble, AdvancedApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<T>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create alpha and beta scalars
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    // Create test vector (matrix is 5x4)
    auto x = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    // Initialize y_amp with some values
    auto y_amp = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);
    // Initialize y_ref with the same values
    auto y_ref = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);

    amp_mtx->apply(alpha, x, beta, y_amp);

    // Compute y_ref = alpha * original * x + beta * y_ref
    this->ell1->apply(alpha, x, beta, y_ref);
    // Check relative componentwise error
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        auto ref_val = y_ref_vals[i];
        auto amp_val = y_amp_vals[i];
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-14});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}

TYPED_TEST(AMPDouble, ApplyWithMultipleRHSHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<T>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create test matrix with 2 RHS (matrix is 5x4, so x is 4x2)
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{1.0, 2.0},
         I<T>{1.0, 0.5},
         I<T>{1.0, 2.0},
         I<T>{1.0, 0.5}}, this->exec);
    // clang-format on
    // Compute y_amp = AMP * x (result is 5x2)
    auto y_amp =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 2});
    amp_mtx->apply(x, y_amp);
    // Compute y_ref = original * x
    auto y_ref =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 2});
    this->ell1->apply(x, y_ref);
    // Check relative componentwise error
    const auto nrows = y_ref->get_size()[0];
    const auto ncols = y_ref->get_size()[1];
    const auto stride_amp = y_amp->get_stride();
    const auto stride_ref = y_ref->get_stride();
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < nrows; i++) {
        for (gko::size_type j = 0; j < ncols; j++) {
            auto ref_val = y_ref_vals[i * stride_ref + j];
            auto amp_val = y_amp_vals[i * stride_amp + j];
            const auto abs_ref = std::abs(ref_val);
            ASSERT_GT(abs_ref, real_T{1e-14});
            auto rel_error =
                std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
            EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
                << "Component (" << i << "," << j << "): amp=" << amp_val
                << ", ref=" << ref_val;
        }
    }
}


TYPED_TEST(AMPDouble, AdvancedApplyWithMultipleRHSHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<T>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create alpha and beta scalars
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    // Create test matrix with 2 RHS (matrix is 5x4, so x is 4x2)
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{1.0, 2.0},
         I<T>{1.0, 0.5},
         I<T>{1.0, 2.0},
         I<T>{1.0, 0.5}}, this->exec);
    // Initialize y_amp and y_ref with identical values (5x2)
    auto y_amp = gko::initialize<Vec>(
        {I<T>{1.0, 2.0},
         I<T>{2.0, 1.0},
         I<T>{3.0, 3.0},
         I<T>{4.0, 2.0},
         I<T>{5.0, 1.0}}, this->exec);
    auto y_ref = gko::initialize<Vec>(
        {I<T>{1.0, 2.0},
         I<T>{2.0, 1.0},
         I<T>{3.0, 3.0},
         I<T>{4.0, 2.0},
         I<T>{5.0, 1.0}}, this->exec);
    // clang-format on
    amp_mtx->apply(alpha, x, beta, y_amp);
    // Compute y_ref = alpha * original * x + beta * y_ref
    this->ell1->apply(alpha, x, beta, y_ref);
    // Check relative componentwise error
    const auto nrows = y_ref->get_size()[0];
    const auto ncols = y_ref->get_size()[1];
    const auto stride_amp = y_amp->get_stride();
    const auto stride_ref = y_ref->get_stride();
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < nrows; i++) {
        for (gko::size_type j = 0; j < ncols; j++) {
            auto ref_val = y_ref_vals[i * stride_ref + j];
            auto amp_val = y_amp_vals[i * stride_amp + j];
            const auto abs_ref = std::abs(ref_val);
            ASSERT_GT(abs_ref, real_T{1e-14});
            auto rel_error =
                std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
            EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
                << "Component (" << i << "," << j << "): amp=" << amp_val
                << ", ref=" << ref_val;
        }
    }
}


TYPED_TEST(AMPDouble, FillInDenseReconstructsOriginalMatrix)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dns = typename TestFixture::Dns;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create result dense matrix with same size
    auto result = Dns::create(this->exec, this->ell1->get_size());

    gko::kernels::reference::amp::fill_in_dense(rexec, amp_mtx.get(),
                                                result.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, result, this->tol);
}

TYPED_TEST(AMPDouble, ExtractDiagonalSumsOverBins)
{
    using T = typename TestFixture::value_type;
    using real_T = gko::remove_complex<T>;
    using Mtx = typename TestFixture::Mtx;
    using Diag = gko::matrix::Diagonal<T>;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // The matrix is 5x4, so diagonal size is min(5,4) = 4
    auto diag = Diag::create(this->exec, 4);

    gko::kernels::reference::amp::extract_diagonal(rexec, amp_mtx.get(),
                                                   diag.get());

    // Diagonal entries from mtx1:
    // diag[0] = 1.1
    // diag[1] = 0 (dropped from 1.2e-11)
    // diag[2] = 0.8
    // diag[3] = 0.0
    auto diag_vals = diag->get_const_values();
    EXPECT_NEAR(std::abs(diag_vals[0] - static_cast<T>(1.1)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[1]), real_T{0}, 0.0);
    EXPECT_NEAR(std::abs(diag_vals[2] - static_cast<T>(0.8)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[3]), real_T{0}, 0.0);
}


TEST(AMPEmptyBin0, SpmvIsCorrectWhenBin0IsEmpty)
{
    using ValueType = double;
    using IndexType = int;
    using Mtx = gko::matrix::AMP<ValueType, IndexType>;
    using Ell = gko::matrix::Ell<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;
    auto exec = gko::ReferenceExecutor::create();
    // 4x4 constant 1D stencil: [-1, 2, -1] (tridiagonal)
    // clang-format off
    auto dns = gko::initialize<gko::matrix::Dense<ValueType>>(
        {{ 2.0, -1.0,  0.0,  0.0},
         {-1.0,  2.0, -1.0,  0.0},
         { 0.0, -1.0,  2.0, -1.0},
         { 0.0,  0.0, -1.0,  2.0}}, exec);
    // clang-format on
    auto ell = Ell::create(exec);
    dns->convert_to(ell.get());

    auto amp = Mtx::build().with_tolerance(0.01f).on(exec)->generate(
        gko::share(ell->clone()));

    auto bin0 = dynamic_cast<const Ell*>(amp->get_bin_matrix(0));
    ASSERT_NE(bin0, nullptr);
    EXPECT_EQ(bin0->get_num_stored_elements_per_row(), 0);

    auto x = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0}, exec);
    auto y_amp = Vec::create(exec, gko::dim<2>{4, 1});
    auto y_ref = Vec::create(exec, gko::dim<2>{4, 1});

    amp->apply(x, y_amp);
    ell->apply(x, y_ref);

    GKO_ASSERT_MTX_NEAR(y_amp, y_ref, 0.01);
}


TEST(AMPEmptyBin0, AdvancedSpmvIsCorrectWhenBin0IsEmpty)
{
    using ValueType = double;
    using IndexType = int;
    using Mtx = gko::matrix::AMP<ValueType, IndexType>;
    using Ell = gko::matrix::Ell<ValueType, IndexType>;
    using Vec = gko::matrix::Dense<ValueType>;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto dns = gko::initialize<gko::matrix::Dense<ValueType>>(
        {{ 2.0, -1.0,  0.0,  0.0},
         {-1.0,  2.0, -1.0,  0.0},
         { 0.0, -1.0,  2.0, -1.0},
         { 0.0,  0.0, -1.0,  2.0}}, exec);
    // clang-format on
    auto ell = Ell::create(exec);
    dns->convert_to(ell.get());
    auto amp = Mtx::build().with_tolerance(0.01f).on(exec)->generate(
        gko::share(ell->clone()));

    auto alpha = gko::initialize<Vec>({2.0}, exec);
    auto beta = gko::initialize<Vec>({-1.0}, exec);
    auto x = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0}, exec);
    auto y_amp = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, exec);
    auto y_ref = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, exec);

    amp->apply(alpha, x, beta, y_amp);
    ell->apply(alpha, x, beta, y_ref);

    GKO_ASSERT_MTX_NEAR(y_amp, y_ref, 0.01);
}


template <typename ValueType>
class AMPFloat : public ::testing::Test {
protected:
    using value_type = ValueType;
    using real_T = gko::remove_complex<value_type>;
    static_assert(std::is_same<real_T, float>::value, "float only!");
    using index_type = int;
    using Mtx = gko::matrix::AMP<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;
    using Dns = gko::matrix::Dense<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;

    AMPFloat() : exec(gko::ReferenceExecutor::create())
    {
        // clang-format off
        mtx1 = gko::initialize<Dns>({{1.1, 3.0e-9, 0.0, 4.5e-4},
                                     {0.0, 1.2e-11, 2.0, 0.0},
                                     {0.0, 0.0, 0.8, 0.0},
                                     {1.2e-11, 0.0, 1.6e-4, 0.0},
                                     {-2e-5, 0.0, -2.0, 0.0}}, exec);
        mtx2 = gko::initialize<Dns>(
            {{1.0, 3.0, 2.0},
             {0.0, 5.0, 0.0}}, exec);
        // clang-format on
        ell1 = gko::share(Ell::create(exec));
        mtx1->convert_to(ell1.get());
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Dns> mtx1;
    std::unique_ptr<Dns> mtx2;
    std::shared_ptr<Ell> ell1;
    const float tol = 1e-6;
};

using float_types = ::testing::Types<float, std::complex<float>>;
TYPED_TEST_SUITE(AMPFloat, float_types, TypenameNameGenerator);


TYPED_TEST(AMPFloat, GenerateComputesCorrectRowNorms)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    gko::amp::precision_array<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), this->tol, max_nnz, rownorms);

    auto rnv = rownorms.get_const_data();
    EXPECT_EQ(rnv[0], static_cast<real_T>(1.1) + static_cast<real_T>(3e-9) +
                          static_cast<real_T>(4.5e-4));
    EXPECT_EQ(rnv[1], static_cast<real_T>(2.0) + static_cast<real_T>(1.2e-11));
    EXPECT_EQ(rnv[2], static_cast<real_T>(0.8));
    EXPECT_EQ(rnv[3],
              static_cast<real_T>(1.2e-11) + static_cast<real_T>(1.6e-4));
    EXPECT_EQ(rnv[4], static_cast<real_T>(2.0) + static_cast<real_T>(2e-5));
}

TYPED_TEST(AMPFloat, GenerateComputesCorrectBinNNZs)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
#if GINKGO_HAVE_AMP_HALF
    static_assert(
        std::tuple_size<gko::amp::precision_array<int, T>>::value == 2,
        "should be 2 available precisions");
#else
    static_assert(
        std::tuple_size<gko::amp::precision_array<int, T>>::value == 1,
        "should be 1 available precision");
#endif
    gko::amp::precision_array<int, T> max_nnz;
    gko::array<real_T> rownorms(this->exec, this->ell1->get_size()[0]);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        rexec, this->ell1.get(), this->tol, max_nnz, rownorms);

#if GINKGO_HAVE_AMP_HALF
    EXPECT_EQ(max_nnz[0], 2);
    EXPECT_EQ(max_nnz[1], 1);
#else
    EXPECT_EQ(max_nnz[0], 2);
#endif
}

TYPED_TEST(AMPFloat, GenerateEllScattersBinsCorrectly)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
#if GINKGO_HAVE_AMP_HALF
    static_assert(
        std::tuple_size<gko::amp::precision_array<int, T>>::value == 2,
        "should be 2 available precisions");
#else
    static_assert(
        std::tuple_size<gko::amp::precision_array<int, T>>::value == 1,
        "should be 1 available precision");
#endif
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    const auto max_nnzs =
#if GINKGO_HAVE_AMP_HALF
        gko::amp::precision_array<int, T>{2, 1};
#else
        gko::amp::precision_array<int, T>{2};
#endif
    auto abins = gko::amp::allocate_bins<T, int>(
        this->exec, this->ell1->get_size(), max_nnzs);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    gko::amp::precision_array<gko::LinOp*, T> amat;
#if GINKGO_HAVE_AMP_HALF
    static_assert(num_bins == 2, "Wrong num bins!");
    ASSERT_EQ(amat.size(), 2);
#else
    static_assert(num_bins == 1, "Wrong num bins!");
    ASSERT_EQ(amat.size(), 1);
#endif
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    gko::kernels::reference::amp::generate_ell_scatter_bins(
        rexec, this->ell1.get(), this->tol, amat);

    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, num_bins, 1>([&](auto k) {
        using value_type = typename std::tuple_element<k, types_list>::type;
        auto amat0 = dynamic_cast<gko::matrix::Ell<value_type, int>*>(amat[k]);
        ASSERT_TRUE(amat0);
        auto vals = amat0->get_const_values();
        auto colids = amat0->get_const_col_idxs();
#if GKO_AMP_HALF_IS_FP16
        if (k == 0) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 2);
            EXPECT_EQ(colids[0], 0);
            EXPECT_EQ(colids[1], 2);
            EXPECT_EQ(colids[2], 2);
            EXPECT_EQ(colids[3], 2);
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(colids[5], gko::invalid_index<int>());
            EXPECT_EQ(colids[6], gko::invalid_index<int>());
            EXPECT_EQ(colids[7], gko::invalid_index<int>());
            EXPECT_EQ(colids[8], gko::invalid_index<int>());
            EXPECT_EQ(colids[9], 2);
            EXPECT_EQ(vals[0], static_cast<value_type>(1.1));
            EXPECT_EQ(vals[1], static_cast<value_type>(2.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.8));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.6e-4));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
            for (int j = 5; j < 9; j++) {
                EXPECT_EQ(vals[j], static_cast<value_type>(0));
            }
            EXPECT_EQ(vals[9], static_cast<value_type>(-2.0));
        } else if (k == 1) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 1);
            EXPECT_EQ(colids[0], 3);
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], gko::invalid_index<int>());
            EXPECT_EQ(colids[4], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[4], static_cast<value_type>(0));
        }
#elif GKO_AMP_HALF_IS_BFLOAT16
        if (k == 0) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 2);
            EXPECT_EQ(colids[0], 0);
            EXPECT_EQ(colids[1], 2);
            EXPECT_EQ(colids[2], 2);
            EXPECT_EQ(colids[3], 2);
            EXPECT_EQ(colids[4], 2);
            EXPECT_EQ(colids[5], 3);
            EXPECT_EQ(colids[6], gko::invalid_index<int>());
            EXPECT_EQ(colids[7], gko::invalid_index<int>());
            EXPECT_EQ(colids[8], gko::invalid_index<int>());
            EXPECT_EQ(colids[9], gko::invalid_index<int>());
            EXPECT_EQ(vals[0], static_cast<value_type>(1.1));
            EXPECT_EQ(vals[1], static_cast<value_type>(2.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.8));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.6e-4));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2.0));
            EXPECT_EQ(vals[5], static_cast<value_type>(4.5e-4));
            for (int j = 6; j < 10; j++) {
                EXPECT_EQ(vals[j], static_cast<value_type>(0));
            }
        } else if (k == 1) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 1);
            EXPECT_EQ(colids[0], gko::invalid_index<int>());
            EXPECT_EQ(colids[1], gko::invalid_index<int>());
            EXPECT_EQ(colids[2], gko::invalid_index<int>());
            EXPECT_EQ(colids[3], gko::invalid_index<int>());
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(vals[0], static_cast<value_type>(0));
            EXPECT_EQ(vals[1], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[3], static_cast<value_type>(0.0));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
        }
#else
        // Only float available (no half)
        if (k == 0) {
            EXPECT_EQ(amat0->get_num_stored_elements_per_row(), 2);
            EXPECT_EQ(colids[0], 0);
            EXPECT_EQ(colids[1], 2);
            EXPECT_EQ(colids[2], 2);
            EXPECT_EQ(colids[3], 2);
            EXPECT_EQ(colids[4], 0);
            EXPECT_EQ(colids[5], 3);
            EXPECT_EQ(colids[6], gko::invalid_index<int>());
            EXPECT_EQ(colids[7], gko::invalid_index<int>());
            EXPECT_EQ(colids[8], gko::invalid_index<int>());
            EXPECT_EQ(colids[9], 2);
            EXPECT_EQ(vals[0], static_cast<value_type>(1.1));
            EXPECT_EQ(vals[1], static_cast<value_type>(2.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.8));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.6e-4));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2e-5));
            EXPECT_EQ(vals[5], static_cast<value_type>(4.5e-4));
            for (int j = 6; j < 9; j++) {
                EXPECT_EQ(vals[j], static_cast<value_type>(0));
            }
            EXPECT_EQ(vals[9], static_cast<value_type>(-2.0));
        }
#endif
    });
}

TYPED_TEST(AMPFloat, ApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create test vector (matrix is 5x4)
    auto x = gko::initialize<Vec>({1.0, 2.0, 1.0, 2.0}, this->exec);
    // Compute y_amp = AMP * x
    auto y_amp =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 1});

    amp_mtx->apply(x, y_amp);

    // Compute y_ref = original * x
    auto y_ref =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 1});
    this->ell1->apply(x, y_ref);
    // Check relative componentwise error
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        const auto ref_val = y_ref_vals[i];
        const auto amp_val = y_amp_vals[i];
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-6});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}

TYPED_TEST(AMPFloat, AdvancedApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create alpha and beta scalars
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    // Create test vector (matrix is 5x4)
    auto x = gko::initialize<Vec>({1.0, 2.0, 1.0, 2.0}, this->exec);
    // Initialize y_amp with some values
    auto y_amp = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);
    // Initialize y_ref with the same values
    auto y_ref = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);

    amp_mtx->apply(alpha, x, beta, y_amp);

    // Compute y_ref = alpha * original * x + beta * y_ref
    this->ell1->apply(alpha, x, beta, y_ref);
    // Check relative componentwise error
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        const auto ref_val = y_ref_vals[i];
        const auto amp_val = y_amp_vals[i];
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-6});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}

TYPED_TEST(AMPFloat, ApplyWithMultipleRHSHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create test matrix with 2 RHS (matrix is 5x4, so x is 4x2)
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{1.0, 2.0},
         I<T>{2.0, 0.5},
         I<T>{1.0, 2.0},
         I<T>{2.0, 0.5}}, this->exec);
    // clang-format on
    // Compute y_amp = AMP * x (result is 5x2)
    auto y_amp =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 2});
    amp_mtx->apply(x, y_amp);
    // Compute y_ref = original * x
    auto y_ref =
        Vec::create(this->exec, gko::dim<2>{this->ell1->get_size()[0], 2});
    this->ell1->apply(x, y_ref);
    // Check relative componentwise error
    const auto nrows = y_ref->get_size()[0];
    const auto ncols = y_ref->get_size()[1];
    const auto stride_amp = y_amp->get_stride();
    const auto stride_ref = y_ref->get_stride();
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < nrows; i++) {
        for (gko::size_type j = 0; j < ncols; j++) {
            const auto ref_val = y_ref_vals[i * stride_ref + j];
            const auto amp_val = y_amp_vals[i * stride_amp + j];
            const auto abs_ref = std::abs(ref_val);
            ASSERT_GT(abs_ref, real_T{1e-6});
            auto rel_error =
                std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
            EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
                << "Component (" << i << "," << j << "): amp=" << amp_val
                << ", ref=" << ref_val;
        }
    }
}


TYPED_TEST(AMPFloat, AdvancedApplyWithMultipleRHSHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create alpha and beta scalars
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    // Create test matrix with 2 RHS (matrix is 5x4, so x is 4x2)
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{1.0, 2.0},
         I<T>{2.0, 0.5},
         I<T>{1.0, 2.0},
         I<T>{2.0, 0.5}}, this->exec);
    // Initialize y_amp and y_ref with identical values (5x2)
    auto y_amp = gko::initialize<Vec>(
        {I<T>{1.0, 2.0},
         I<T>{2.0, 1.0},
         I<T>{3.0, 3.0},
         I<T>{4.0, 2.0},
         I<T>{5.0, 1.0}}, this->exec);
    auto y_ref = gko::initialize<Vec>(
        {I<T>{1.0, 2.0},
         I<T>{2.0, 1.0},
         I<T>{3.0, 3.0},
         I<T>{4.0, 2.0},
         I<T>{5.0, 1.0}}, this->exec);
    // clang-format on
    amp_mtx->apply(alpha, x, beta, y_amp);
    // Compute y_ref = alpha * original * x + beta * y_ref
    this->ell1->apply(alpha, x, beta, y_ref);
    // Check relative componentwise error
    const auto nrows = y_ref->get_size()[0];
    const auto ncols = y_ref->get_size()[1];
    const auto stride_amp = y_amp->get_stride();
    const auto stride_ref = y_ref->get_stride();
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < nrows; i++) {
        for (gko::size_type j = 0; j < ncols; j++) {
            const auto ref_val = y_ref_vals[i * stride_ref + j];
            const auto amp_val = y_amp_vals[i * stride_amp + j];
            const auto abs_ref = std::abs(ref_val);
            ASSERT_GT(abs_ref, real_T{1e-6});
            auto rel_error =
                std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
            EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
                << "Component (" << i << "," << j << "): amp=" << amp_val
                << ", ref=" << ref_val;
        }
    }
}


TYPED_TEST(AMPFloat, FillInDenseReconstructsOriginalMatrix)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dns = typename TestFixture::Dns;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // Create result dense matrix with same size
    auto result = Dns::create(this->exec, this->ell1->get_size());

    gko::kernels::reference::amp::fill_in_dense(rexec, amp_mtx.get(),
                                                result.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, result, this->tol);
}

TYPED_TEST(AMPFloat, ExtractDiagonalSumsOverBins)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Diag = gko::matrix::Diagonal<T>;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    // Create AMP matrix from the ELL matrix
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->ell1);
    // The matrix is 5x4, so diagonal size is min(5,4) = 4
    auto diag = Diag::create(this->exec, 4);

    gko::kernels::reference::amp::extract_diagonal(rexec, amp_mtx.get(),
                                                   diag.get());

    // Diagonal entries from mtx1:
    // diag[0] = 1.1
    // diag[1] = 0 (dropped from 1.2e-11)
    // diag[2] = 0.8
    // diag[3] = 0.0
    auto diag_vals = diag->get_const_values();
    EXPECT_NEAR(std::abs(diag_vals[0] - static_cast<T>(1.1)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[1]), real_T{0}, 0.0);
    EXPECT_NEAR(std::abs(diag_vals[2] - static_cast<T>(0.8)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[3]), real_T{0}, 0.0);
}


#if GKO_AMP_HALF_IS_FP16 || GKO_AMP_HALF_IS_BFLOAT16

template <typename ValueType>
class AMPDoubleCsr : public ::testing::Test {
protected:
    using value_type = ValueType;
    using real_T = gko::remove_complex<ValueType>;
    static_assert(std::is_same<real_T, double>::value, "double only!");
    using index_type = int;
    using Mtx = gko::matrix::AMP<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dns = gko::matrix::Dense<value_type>;
    using Vec = gko::matrix::Dense<value_type>;

    AMPDoubleCsr() : exec(gko::ReferenceExecutor::create())
    {
        // clang-format off
        mtx1 = gko::initialize<Dns>({{1.1, 3.0e-9, 0.0, 4.5e-4},
                                     {0.0, 1.2e-11, 2.0, 0.0},
                                     {0.0, 0.0, 0.8, 0.0},
                                     {1.2e-11, 0.0, 1.6e-4, 0.0},
                                     {-2e-5, 0.0, -2.0, 0.0}}, exec);
        // clang-format on
        csr1 = gko::share(Csr::create(exec));
        mtx1->convert_to(csr1.get());
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Dns> mtx1;
    std::shared_ptr<Csr> csr1;
    const float tol = 1e-10;
};

using double_csr_types = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(AMPDoubleCsr, double_csr_types, TypenameNameGenerator);


TYPED_TEST(AMPDoubleCsr, GenerateComputesCorrectRowNorms)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_T = typename TestFixture::real_T;
    gko::amp::precision_array<gko::array<index_type>, T> row_sizes;
    for (int i = 0; i < gko::amp::narrow_types<double>::num_types; i++) {
        row_sizes[i].set_executor(this->exec);
        row_sizes[i].resize_and_reset(this->csr1->get_size()[0] + 1);
    }
    auto row_sz_ptrs = gko::amp::get_pointer_array<index_type, T>(row_sizes);
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);

    gko::kernels::reference::amp::generate_cwise_csr_calculate_row_sizes(
        rexec, this->csr1.get(), this->tol, row_sz_ptrs);

    auto rnv = row_sz_ptrs;
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(rnv[0][i], 1);
    }
#if GKO_AMP_HALF_IS_FP16
    EXPECT_EQ(rnv[1][0], 2);
    EXPECT_EQ(rnv[1][1], 0);
    EXPECT_EQ(rnv[1][2], 0);
    EXPECT_EQ(rnv[1][3], 1);
    EXPECT_EQ(rnv[1][4], 1);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(rnv[2][i], 0);
    }
#elif GKO_AMP_HALF_IS_BFLOAT16
    EXPECT_EQ(rnv[1][0], 1);
    EXPECT_EQ(rnv[1][1], 0);
    EXPECT_EQ(rnv[1][2], 0);
    EXPECT_EQ(rnv[1][3], 1);
    EXPECT_EQ(rnv[1][4], 1);
    EXPECT_EQ(rnv[2][0], 1);
    for (int i = 1; i < 5; i++) {
        EXPECT_EQ(rnv[2][i], 0);
    }
#else
    EXPECT_EQ(rnv[1][0], 2);
    EXPECT_EQ(rnv[1][1], 0);
    EXPECT_EQ(rnv[1][2], 0);
    EXPECT_EQ(rnv[1][3], 1);
    EXPECT_EQ(rnv[1][4], 1);
#endif
}


TYPED_TEST(AMPDoubleCsr, GenerateCsrScattersBinsCorrectly)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_T = typename TestFixture::real_T;
    constexpr int q = gko::amp::narrow_types<T>::num_types;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    gko::amp::precision_array<gko::array<index_type>, T> row_sizes;
    for (int i = 0; i < q; i++) {
        row_sizes[i].set_executor(this->exec);
        row_sizes[i].resize_and_reset(this->csr1->get_size()[0] + 1);
    }
    auto row_ptrs = gko::amp::get_pointer_array<index_type, T>(row_sizes);
    for (int i = 0; i < 5; i++) {
        row_ptrs[0][i] = 1;
    }
#if GKO_AMP_HALF_IS_FP16
    row_ptrs[1][0] = 2;
    row_ptrs[1][1] = 0;
    row_ptrs[1][2] = 0;
    row_ptrs[1][3] = 1;
    row_ptrs[1][4] = 1;
    for (int i = 0; i < 5; i++) {
        row_ptrs[2][i] = 0;
    }
#elif GKO_AMP_HALF_IS_BFLOAT16
    row_ptrs[1][0] = 1;
    row_ptrs[1][1] = 0;
    row_ptrs[1][2] = 0;
    row_ptrs[1][3] = 1;
    row_ptrs[1][4] = 1;
    row_ptrs[2][0] = 1;
    for (int i = 1; i < 5; i++) {
        row_ptrs[2][i] = 0;
    }
#else
    row_ptrs[1][0] = 2;
    row_ptrs[1][1] = 0;
    row_ptrs[1][2] = 0;
    row_ptrs[1][3] = 1;
    row_ptrs[1][4] = 1;
#endif
    for (int k = 0; k < q; k++) {
        std::exclusive_scan(row_ptrs[k], row_ptrs[k] + 6, row_ptrs[k], 0);
    }
    auto abins = gko::amp::allocate_csr_bins<T, int>(
        this->exec, this->csr1->get_size(), row_sizes);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    gko::amp::precision_array<gko::LinOp*, T> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    gko::kernels::reference::amp::generate_cwise_csr_scatter_bins(
        rexec, this->csr1.get(), this->tol, amat);

    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, num_bins, 1>([&](auto k) {
        using value_type = typename std::tuple_element<k, types_list>::type;
        auto cbin = dynamic_cast<gko::matrix::Csr<value_type, int>*>(amat[k]);
        ASSERT_TRUE(cbin);
        auto row_ptrs = cbin->get_const_row_ptrs();
        auto col_idxs = cbin->get_const_col_idxs();
        auto vals = cbin->get_const_values();

        if (k == 0) {
            // Bin 0: one dominant entry per row
            EXPECT_EQ(row_ptrs[0], 0);
            EXPECT_EQ(row_ptrs[1], 1);
            EXPECT_EQ(row_ptrs[2], 2);
            EXPECT_EQ(row_ptrs[3], 3);
            EXPECT_EQ(row_ptrs[4], 4);
            EXPECT_EQ(row_ptrs[5], 5);
            EXPECT_EQ(col_idxs[0], 0);
            EXPECT_EQ(col_idxs[1], 2);
            EXPECT_EQ(col_idxs[2], 2);
            EXPECT_EQ(col_idxs[3], 2);
            EXPECT_EQ(col_idxs[4], 2);
            EXPECT_EQ(vals[0], static_cast<value_type>(1.1));
            EXPECT_EQ(vals[1], static_cast<value_type>(2.0));
            EXPECT_EQ(vals[2], static_cast<value_type>(0.8));
            EXPECT_EQ(vals[3], static_cast<value_type>(1.6e-4));
            EXPECT_EQ(vals[4], static_cast<value_type>(-2.0));
        }
#if GKO_AMP_HALF_IS_FP16
        else if (k == 1) {
            // Bin 1 (float): row 0 has 2 entries, rows 3 and 4 have 1 each
            EXPECT_EQ(row_ptrs[0], 0);
            EXPECT_EQ(row_ptrs[1], 2);
            EXPECT_EQ(row_ptrs[2], 2);
            EXPECT_EQ(row_ptrs[3], 2);
            EXPECT_EQ(row_ptrs[4], 3);
            EXPECT_EQ(row_ptrs[5], 4);
            EXPECT_EQ(col_idxs[0], 1);
            EXPECT_EQ(col_idxs[1], 3);
            EXPECT_EQ(col_idxs[2], 0);
            EXPECT_EQ(col_idxs[3], 0);
            EXPECT_EQ(vals[0], static_cast<value_type>(3e-9));
            EXPECT_EQ(vals[1], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[2], static_cast<value_type>(1.2e-11));
            EXPECT_EQ(vals[3], static_cast<value_type>(-2e-5));
        } else if (k == 2) {
            // Bin 2 (half): empty
            EXPECT_EQ(cbin->get_num_stored_elements(), gko::size_type{0});
        }
#elif GKO_AMP_HALF_IS_BFLOAT16
        else if (k == 1) {
            // Bin 1 (float): rows 0, 3, 4 each have 1 entry
            EXPECT_EQ(row_ptrs[0], 0);
            EXPECT_EQ(row_ptrs[1], 1);
            EXPECT_EQ(row_ptrs[2], 1);
            EXPECT_EQ(row_ptrs[3], 1);
            EXPECT_EQ(row_ptrs[4], 2);
            EXPECT_EQ(row_ptrs[5], 3);
            EXPECT_EQ(col_idxs[0], 3);
            EXPECT_EQ(col_idxs[1], 0);
            EXPECT_EQ(col_idxs[2], 0);
            EXPECT_EQ(vals[0], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[1], static_cast<value_type>(1.2e-11));
            EXPECT_EQ(vals[2], static_cast<value_type>(-2e-5));
        } else if (k == 2) {
            // Bin 2 (bfloat16): row 0 has 1 entry
            EXPECT_EQ(row_ptrs[0], 0);
            EXPECT_EQ(row_ptrs[1], 1);
            EXPECT_EQ(row_ptrs[2], 1);
            EXPECT_EQ(row_ptrs[3], 1);
            EXPECT_EQ(row_ptrs[4], 1);
            EXPECT_EQ(row_ptrs[5], 1);
            EXPECT_EQ(col_idxs[0], 1);
            EXPECT_EQ(vals[0], static_cast<value_type>(3e-9));
        }
#else
        // Only double and float (no half)
        else if (k == 1) {
            EXPECT_EQ(row_ptrs[0], 0);
            EXPECT_EQ(row_ptrs[1], 2);
            EXPECT_EQ(row_ptrs[2], 2);
            EXPECT_EQ(row_ptrs[3], 2);
            EXPECT_EQ(row_ptrs[4], 3);
            EXPECT_EQ(row_ptrs[5], 4);
            EXPECT_EQ(col_idxs[0], 1);
            EXPECT_EQ(col_idxs[1], 3);
            EXPECT_EQ(col_idxs[2], 0);
            EXPECT_EQ(col_idxs[3], 0);
            EXPECT_EQ(vals[0], static_cast<value_type>(3e-9));
            EXPECT_EQ(vals[1], static_cast<value_type>(4.5e-4));
            EXPECT_EQ(vals[2], static_cast<value_type>(1.2e-11));
            EXPECT_EQ(vals[3], static_cast<value_type>(-2e-5));
        }
#endif
    });
}


TYPED_TEST(AMPDoubleCsr, ApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->csr1);
    auto x = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    auto y_amp =
        Vec::create(this->exec, gko::dim<2>{this->csr1->get_size()[0], 1});

    amp_mtx->apply(x, y_amp);

    auto y_ref =
        Vec::create(this->exec, gko::dim<2>{this->csr1->get_size()[0], 1});
    this->csr1->apply(x, y_ref);
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        auto ref_val = y_ref_vals[i];
        auto amp_val = y_amp_vals[i];
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-14});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}


TYPED_TEST(AMPDoubleCsr, AdvancedApplyHasCorrectRelativeError)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(this->csr1);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 1.0, 1.0, 1.0}, this->exec);
    auto y_amp = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);
    auto y_ref = gko::initialize<Vec>({1.0, 2.0, 3.0, 4.0, 5.0}, this->exec);

    amp_mtx->apply(alpha, x, beta, y_amp);

    this->csr1->apply(alpha, x, beta, y_ref);
    auto y_amp_vals = y_amp->get_const_values();
    auto y_ref_vals = y_ref->get_const_values();
    for (gko::size_type i = 0; i < y_ref->get_size()[0]; i++) {
        auto ref_val = y_ref_vals[i];
        auto amp_val = y_amp_vals[i];
        const auto abs_ref = std::abs(ref_val);
        ASSERT_GT(abs_ref, real_T{1e-14});
        auto rel_error =
            std::abs(amp_val - ref_val) / static_cast<real_T>(abs_ref);
        EXPECT_LE(rel_error, static_cast<real_T>(this->tol))
            << "Component " << i << ": amp=" << amp_val << ", ref=" << ref_val;
    }
}


TYPED_TEST(AMPDoubleCsr, FillInDenseReconstructsOriginalMatrix)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Dns = typename TestFixture::Dns;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->csr1->clone()));
    auto result = Dns::create(this->exec, this->csr1->get_size());

    gko::kernels::reference::amp::fill_in_dense(rexec, amp_mtx.get(),
                                                result.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, result, this->tol);
}


TYPED_TEST(AMPDoubleCsr, ExtractDiagonalIsCorrect)
{
    using T = typename TestFixture::value_type;
    using real_T = typename TestFixture::real_T;
    using Mtx = typename TestFixture::Mtx;
    using Diag = gko::matrix::Diagonal<T>;
    auto rexec =
        std::dynamic_pointer_cast<const gko::ReferenceExecutor>(this->exec);
    auto amp_mtx = Mtx::build()
                       .with_tolerance(this->tol)
                       .on(this->exec)
                       ->generate(gko::share(this->csr1->clone()));
    // Matrix is 5x4, so diagonal size is min(5,4) = 4
    auto diag = Diag::create(this->exec, 4);

    gko::kernels::reference::amp::extract_diagonal(rexec, amp_mtx.get(),
                                                   diag.get());

    // diag[0] = 1.1 (dominant, in bin 0)
    // diag[1] = dropped (1.2e-11 is too small), no diagonal entry
    // diag[2] = 0.8 (dominant, in bin 0)
    // diag[3] = 0.0 (no entry at (3,3) in the matrix)
    auto diag_vals = diag->get_const_values();
    EXPECT_NEAR(std::abs(diag_vals[0] - static_cast<T>(1.1)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[1]), real_T{0}, 0.0);
    EXPECT_NEAR(std::abs(diag_vals[2] - static_cast<T>(0.8)), real_T{0},
                static_cast<real_T>(this->tol));
    EXPECT_NEAR(std::abs(diag_vals[3]), real_T{0}, 0.0);
}

#endif  // GKO_AMP_HALF_IS_FP16 || GKO_AMP_HALF_IS_BFLOAT16


}  // namespace
