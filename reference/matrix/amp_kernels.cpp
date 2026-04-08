// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <cassert>
#include <vector>

#include <ginkgo/core/base/amp_types.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/mixed_precision_types.hpp"
#include "core/base/utils.hpp"
#include "reference/matrix/amp_algorithms.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The AMP matrix format namespace.
 * @ref Amp
 * @ingroup amp
 */
namespace amp {


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv_ell(std::shared_ptr<const ReferenceExecutor> exec,
              const matrix::AMP<MatrixValueType, IndexType>* const a,
              const matrix::Dense<InputValueType>* const b,
              matrix::Dense<OutputValueType>* const c)
{
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto ell0 = dynamic_cast<const matrix::Ell<MatrixValueType, IndexType>*>(
        a->get_bin_matrix(0));
    if (!ell0) {
        GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
    }
    auto y = c->get_values();
    const auto y_stride = c->get_stride();
    auto x = b->get_const_values();
    const auto x_stride = b->get_stride();
    const auto nrhs = b->get_size()[1];

    const auto nrows0 = static_cast<int>(a->get_size()[0]);
    const auto stride0 = ell0->get_stride();
    auto avals = ell0->get_const_values();
    auto acols = ell0->get_const_col_idxs();
    const auto max_nnz0 = ell0->get_num_stored_elements_per_row();
    // We need mult type because complex numbers of different precisions don't
    // get automatically promoted.
    using mult_type0 = gko::highest_precision<MatrixValueType, InputValueType>;
    using highest_type0 = gko::highest_precision<mult_type0, OutputValueType>;
    for (int i = 0; i < nrows0; i++) {
        for (int irhs = 0; irhs < nrhs; irhs++) {
            highest_type0 sum = 0;
            for (int j = 0; j < max_nnz0; j++) {
                if (acols[i + j * stride0] >= 0) {
                    sum += static_cast<highest_type0>(
                        static_cast<mult_type0>(avals[i + j * stride0]) *
                        static_cast<mult_type0>(
                            x[acols[i + j * stride0] * x_stride + irhs]));
                }
            }
            y[i * y_stride + irhs] = static_cast<OutputValueType>(sum);
        }
    }
    gko::constexpr_for<1, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        auto ellk = dynamic_cast<const matrix::Ell<value_type, IndexType>*>(
            a->get_bin_matrix(k));
        if (!ellk) {
            GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
        }
        using mult_type = gko::highest_precision<value_type, InputValueType>;
        using highest_type = gko::highest_precision<mult_type, OutputValueType>;
        const auto nrows = static_cast<int>(a->get_size()[0]);
        assert(nrows == nrows0);
        const auto stride = ellk->get_stride();
        auto avals = ellk->get_const_values();
        auto acols = ellk->get_const_col_idxs();
        const auto max_nnz = ellk->get_num_stored_elements_per_row();
        if (max_nnz > 0) {
            for (int i = 0; i < nrows; i++) {
                for (int irhs = 0; irhs < nrhs; irhs++) {
                    highest_type sum = 0;
                    for (int j = 0; j < max_nnz; j++) {
                        if (acols[i + j * stride] >= 0) {
                            sum += static_cast<highest_type>(
                                static_cast<mult_type>(avals[i + j * stride]) *
                                static_cast<mult_type>(
                                    x[acols[i + j * stride] * x_stride +
                                      irhs]));
                        }
                    }
                    y[i * y_stride + irhs] += static_cast<OutputValueType>(sum);
                }
            }
        }
    });
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_SPMV_ELL_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv_ell(std::shared_ptr<const ReferenceExecutor> exec,
                       const matrix::Dense<MatrixValueType>* alpha,
                       const matrix::AMP<MatrixValueType, IndexType>* a,
                       const matrix::Dense<InputValueType>* b,
                       const matrix::Dense<OutputValueType>* beta,
                       matrix::Dense<OutputValueType>* c)
{
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto ell0 = dynamic_cast<const matrix::Ell<MatrixValueType, IndexType>*>(
        a->get_bin_matrix(0));
    if (!ell0) {
        GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
    }
    auto y = c->get_values();
    auto x = b->get_const_values();
    const auto y_stride = c->get_stride();
    const auto x_stride = b->get_stride();
    const auto nrhs = b->get_size()[1];
    const auto alph = alpha->at(0, 0);
    const auto bet = beta->at(0, 0);
    const auto nrows0 = static_cast<int>(a->get_size()[0]);
    const auto stride0 = ell0->get_stride();
    auto avals = ell0->get_const_values();
    auto acols = ell0->get_const_col_idxs();
    const auto max_nnz0 = ell0->get_num_stored_elements_per_row();
    // We need mult type because complex numbers of different precisions don't
    // get automatically promoted.
    using mult_type0 = gko::highest_precision<MatrixValueType, InputValueType>;
    using highest_type0 = gko::highest_precision<mult_type0, OutputValueType>;
    for (int i = 0; i < nrows0; i++) {
        for (int irhs = 0; irhs < nrhs; irhs++) {
            y[i * y_stride + irhs] = bet * y[i * y_stride + irhs];
            highest_type0 sum = 0;
            for (int j = 0; j < max_nnz0; j++) {
                const auto col = acols[i + j * stride0];
                if (col >= 0) {
                    sum += static_cast<highest_type0>(
                        static_cast<mult_type0>(avals[i + j * stride0]) *
                        static_cast<mult_type0>(x[col * x_stride + irhs]));
                }
            }
            y[i * y_stride + irhs] += static_cast<OutputValueType>(
                static_cast<highest_type0>(alph) * sum);
        }
    }
    gko::constexpr_for<1, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        auto ellk = dynamic_cast<const matrix::Ell<value_type, IndexType>*>(
            a->get_bin_matrix(k));
        if (!ellk) {
            GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
        }
        using mult_type = gko::highest_precision<value_type, InputValueType>;
        using highest_type = gko::highest_precision<mult_type, OutputValueType>;
        const auto nrows = static_cast<int>(a->get_size()[0]);
        assert(nrows == nrows0);
        const auto stride = ellk->get_stride();
        auto avals = ellk->get_const_values();
        auto acols = ellk->get_const_col_idxs();
        const auto max_nnz = ellk->get_num_stored_elements_per_row();
        if (max_nnz > 0) {
            for (int i = 0; i < nrows; i++) {
                for (int irhs = 0; irhs < nrhs; irhs++) {
                    highest_type sum = 0;
                    for (int j = 0; j < max_nnz; j++) {
                        const auto col = acols[i + j * stride];
                        if (col >= 0) {
                            sum += static_cast<highest_type>(
                                static_cast<mult_type>(avals[i + j * stride]) *
                                static_cast<mult_type>(
                                    x[col * x_stride + irhs]));
                        }
                    }
                    y[i * y_stride + irhs] += static_cast<OutputValueType>(
                        static_cast<highest_type>(alph) * sum);
                }
            }
        }
    });
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_ADVANCED_SPMV_ELL_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv_csr(std::shared_ptr<const ReferenceExecutor> exec,
              const matrix::AMP<MatrixValueType, IndexType>* const a,
              const matrix::Dense<InputValueType>* const b,
              matrix::Dense<OutputValueType>* const c)
{
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto csr0 = dynamic_cast<const matrix::Csr<MatrixValueType, IndexType>*>(
        a->get_bin_matrix(0));
    if (!csr0) {
        GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
    }
    auto y = c->get_values();
    const auto y_stride = c->get_stride();
    auto x = b->get_const_values();
    const auto x_stride = b->get_stride();
    const auto nrhs = b->get_size()[1];
    const auto nrows0 = static_cast<int>(a->get_size()[0]);
    // We need mult type because complex numbers of different precisions don't
    // get automatically promoted.
    using mult_type0 = gko::highest_precision<MatrixValueType, InputValueType>;
    using highest_type0 = gko::highest_precision<mult_type0, OutputValueType>;

    auto avals0 = csr0->get_const_values();
    auto acols0 = csr0->get_const_col_idxs();
    auto arow_ptrs0 = csr0->get_const_row_ptrs();
    for (int i = 0; i < nrows0; i++) {
        for (int irhs = 0; irhs < nrhs; irhs++) {
            highest_type0 sum = 0;
            for (auto j = arow_ptrs0[i]; j < arow_ptrs0[i + 1]; j++) {
                sum += static_cast<highest_type0>(
                    static_cast<mult_type0>(avals0[j]) *
                    static_cast<mult_type0>(x[acols0[j] * x_stride + irhs]));
            }
            y[i * y_stride + irhs] = static_cast<OutputValueType>(sum);
        }
    }
    gko::constexpr_for<1, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        auto csrk = dynamic_cast<const matrix::Csr<value_type, IndexType>*>(
            a->get_bin_matrix(k));
        if (!csrk) {
            GKO_NOT_SUPPORTED(a->get_bin_matrix(k));
        }
        using mult_type = gko::highest_precision<value_type, InputValueType>;
        using highest_type = gko::highest_precision<mult_type, OutputValueType>;
        const auto nrows = static_cast<int>(a->get_size()[0]);
        assert(nrows == nrows0);
        auto avals = csrk->get_const_values();
        auto acols = csrk->get_const_col_idxs();
        auto arow_ptrs = csrk->get_const_row_ptrs();
        if (csrk->get_num_stored_elements() > 0) {
            for (int i = 0; i < nrows; i++) {
                for (int irhs = 0; irhs < nrhs; irhs++) {
                    highest_type sum = 0;
                    for (auto j = arow_ptrs[i]; j < arow_ptrs[i + 1]; j++) {
                        sum += static_cast<highest_type>(
                            static_cast<mult_type>(avals[j]) *
                            static_cast<mult_type>(
                                x[acols[j] * x_stride + irhs]));
                    }
                    y[i * y_stride + irhs] += static_cast<OutputValueType>(sum);
                }
            }
        }
    });
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_SPMV_CSR_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv_csr(std::shared_ptr<const ReferenceExecutor> exec,
                       const matrix::Dense<MatrixValueType>* alpha,
                       const matrix::AMP<MatrixValueType, IndexType>* a,
                       const matrix::Dense<InputValueType>* b,
                       const matrix::Dense<OutputValueType>* beta,
                       matrix::Dense<OutputValueType>* c)
{
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto csr0 = dynamic_cast<const matrix::Csr<MatrixValueType, IndexType>*>(
        a->get_bin_matrix(0));
    if (!csr0) {
        GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
    }
    auto y = c->get_values();
    auto x = b->get_const_values();
    const auto y_stride = c->get_stride();
    const auto x_stride = b->get_stride();
    const auto nrhs = b->get_size()[1];
    const auto alph = alpha->at(0, 0);
    const auto bet = beta->at(0, 0);
    const auto nrows0 = static_cast<int>(a->get_size()[0]);
    // We need mult type because complex numbers of different precisions don't
    // get automatically promoted.
    using mult_type0 = gko::highest_precision<MatrixValueType, InputValueType>;
    using highest_type0 = gko::highest_precision<mult_type0, OutputValueType>;

    auto avals0 = csr0->get_const_values();
    auto acols0 = csr0->get_const_col_idxs();
    auto arow_ptrs0 = csr0->get_const_row_ptrs();
    for (int i = 0; i < nrows0; i++) {
        for (int irhs = 0; irhs < nrhs; irhs++) {
            y[i * y_stride + irhs] = bet * y[i * y_stride + irhs];
            highest_type0 sum = 0;
            for (auto j = arow_ptrs0[i]; j < arow_ptrs0[i + 1]; j++) {
                sum += static_cast<highest_type0>(
                    static_cast<mult_type0>(avals0[j]) *
                    static_cast<mult_type0>(x[acols0[j] * x_stride + irhs]));
            }
            y[i * y_stride + irhs] += static_cast<OutputValueType>(
                static_cast<highest_type0>(alph) * sum);
        }
    }
    gko::constexpr_for<1, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        auto csrk = dynamic_cast<const matrix::Csr<value_type, IndexType>*>(
            a->get_bin_matrix(k));
        if (!csrk) {
            GKO_NOT_SUPPORTED(a->get_bin_matrix(k));
        }
        using mult_type = gko::highest_precision<value_type, InputValueType>;
        using highest_type = gko::highest_precision<mult_type, OutputValueType>;
        const auto nrows = static_cast<int>(a->get_size()[0]);
        assert(nrows == nrows0);
        auto avals = csrk->get_const_values();
        auto acols = csrk->get_const_col_idxs();
        auto arow_ptrs = csrk->get_const_row_ptrs();
        if (csrk->get_num_stored_elements() > 0) {
            for (int i = 0; i < nrows; i++) {
                for (int irhs = 0; irhs < nrhs; irhs++) {
                    highest_type sum = 0;
                    for (auto j = arow_ptrs[i]; j < arow_ptrs[i + 1]; j++) {
                        sum += static_cast<highest_type>(
                            static_cast<mult_type>(avals[j]) *
                            static_cast<mult_type>(
                                x[acols[j] * x_stride + irhs]));
                    }
                    y[i * y_stride + irhs] += static_cast<OutputValueType>(
                        static_cast<highest_type>(alph) * sum);
                }
            }
        }
    });
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_ADVANCED_SPMV_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void generate_cwise_ell_max_nnz_per_row(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a, const float tolerance,
    gko::amp::precision_array<int, ValueType>& max_nnz_per_row)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = gko::matrix::AMP<ValueType, IndexType>::num_precisions;
    // Compute minimum representable values for each bin
    const std::array<real_type, q> min_repr =
        get_bins_min_representable<real_type>();

    const auto nrows = a->get_size()[0];
    const auto ostride = a->get_stride();
    const auto omax_nnz = a->get_num_stored_elements_per_row();
    const ValueType* const ovals = a->get_const_values();
    const IndexType* const ocolids = a->get_const_col_idxs();
    for (int k = 0; k < q; k++) {
        max_nnz_per_row[k] = 0;
    }
    for (int irow = 0; irow < nrows; irow++) {
        // Compute row's 1-norm
        auto rnorm = static_cast<real_type>(0);
        for (int j = 0; j < omax_nnz; j++) {
            if (ocolids[j * ostride + irow] == invalid_index<IndexType>()) {
                break;
            } else {
                rnorm += std::abs(ovals[j * ostride + irow]);
            }
        }

        // Compute lower limits of each precision bin
        const std::array<float, q> min_bin =
            get_bins_precision_lower_bounds<real_type>(rnorm, tolerance);

        // Get max nnz per row for each precision bin matrix
        std::array<int, q> row_nnz = {};
        for (int j = 0; j < omax_nnz; j++) {
            const auto jcol = ocolids[j * ostride + irow];
            const int ibin = get_adjusted_bin_for_entry<real_type>(
                min_bin, min_repr, std::abs(ovals[j * ostride + irow]),
                jcol == static_cast<IndexType>(irow));
            if (ibin >= 0) {
                row_nnz[ibin]++;
            }
        }
        for (int k = 0; k < q; k++) {
            max_nnz_per_row[k] = std::max(max_nnz_per_row[k], row_nnz[k]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_CWISE_ELL_STEP1_KERNEL);


template <typename ValueType, typename IndexType>
void generate_ell_scatter_bins(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* const a, const float tolerance,
    gko::amp::precision_array<LinOp*, ValueType>& amat)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = gko::matrix::AMP<ValueType, IndexType>::num_precisions;
    // Compute minimum representable values for each bin
    const std::array<real_type, q> min_repr =
        get_bins_min_representable<real_type>();

    const auto nrows = a->get_size()[0];
    const auto ostride = a->get_stride();
    const auto omax_nnz = a->get_num_stored_elements_per_row();
    const ValueType* const ovals = a->get_const_values();
    const IndexType* const ocolidxs = a->get_const_col_idxs();
    for (int irow = 0; irow < nrows; irow++) {
        // Compute row's 1-norm
        auto rnorm = static_cast<real_type>(0);
        for (int j = 0; j < omax_nnz; j++) {
            if (ocolidxs[j * ostride + irow] == invalid_index<IndexType>()) {
                break;
            } else {
                rnorm += std::abs(ovals[j * ostride + irow]);
            }
        }

        // Compute lower limits of each precision bin
        const std::array<float, q> min_bin =
            get_bins_precision_lower_bounds<real_type>(rnorm, tolerance);

        using EllTuple = gko::instantiation_tuple_t<
            gko::generator_partial<gko::matrix::Ell, IndexType>,
            typename gko::amp::narrow_types<ValueType>::type>;
        using ScalarPtrTuple = gko::instantiation_tuple_t<
            gko::generator<gko::ptr_type>,
            typename gko::amp::narrow_types<ValueType>::type>;
        ScalarPtrTuple xvalues;
        gko::amp::precision_array<IndexType*, ValueType> xcol_idxs;
        gko::amp::precision_array<size_type, ValueType> bin_strides;
        std::array<int, q> ixj = {};

        // initialize bins
        gko::constexpr_for<0, q, 1>([&](auto k) {
            using EllType = typename std::tuple_element<k, EllTuple>::type;
            auto ematk = dynamic_cast<EllType*>(amat[k]);
            xcol_idxs[k] = ematk->get_col_idxs();
            bin_strides[k] = ematk->get_stride();
            std::get<k>(xvalues) = ematk->get_values();
            const auto nnz_row = ematk->get_num_stored_elements_per_row();
            const auto stride = ematk->get_stride();
            for (int j = 0; j < nnz_row; j++) {
                xcol_idxs[k][j * stride + irow] =
                    gko::invalid_index<IndexType>();
                std::get<k>(xvalues)[j * stride + irow] = 0;
            }
        });

        for (int j = 0; j < omax_nnz; j++) {
            const ptrdiff_t oloc = j * ostride + irow;
            const auto jcol = ocolidxs[oloc];
            const int ibin = get_adjusted_bin_for_entry<real_type>(
                min_bin, min_repr, std::abs(ovals[oloc]),
                jcol == static_cast<IndexType>(irow));
            if (ibin >= 0) {
                const auto nzloc =
                    ixj[ibin] * static_cast<ptrdiff_t>(bin_strides[ibin]) +
                    irow;
                xcol_idxs[ibin][nzloc] = jcol;
                assign_value_to_array_tuple<0>(xvalues, ovals[oloc], ibin,
                                               nzloc);
                ixj[ibin]++;
            }
        }
    }  // End loop over rows
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_ELL_SCATTER_BINS_KERNEL);


template <typename ValueType, typename IndexType>
void generate_cwise_csr_calculate_row_sizes(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* a, const float tolerance,
    gko::amp::precision_array<IndexType*, ValueType>& bin_row_sizes)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = gko::matrix::AMP<ValueType, IndexType>::num_precisions;
    const std::array<real_type, q> min_repr =
        get_bins_min_representable<real_type>();

    const auto nrows = a->get_size()[0];
    const ValueType* const ovals = a->get_const_values();
    const IndexType* const ocolidxs = a->get_const_col_idxs();
    const IndexType* const orow_ptrs = a->get_const_row_ptrs();
    for (int irow = 0; irow < nrows; irow++) {
        for (int k = 0; k < q; k++) {
            bin_row_sizes[k][irow] = 0;
        }
        // Compute row's 1-norm
        auto rnorm = static_cast<real_type>(0);
        for (auto j = orow_ptrs[irow]; j < orow_ptrs[irow + 1]; j++) {
            rnorm += std::abs(ovals[j]);
        }

        // Compute lower limits of each precision bin
        const std::array<float, q> min_bin =
            get_bins_precision_lower_bounds<real_type>(rnorm, tolerance);

        // Count NNZ per bin across all rows
        for (auto j = orow_ptrs[irow]; j < orow_ptrs[irow + 1]; j++) {
            const int ibin = get_adjusted_bin_for_entry<real_type>(
                min_bin, min_repr, std::abs(ovals[j]),
                ocolidxs[j] == static_cast<IndexType>(irow));
            if (ibin >= 0) {
                bin_row_sizes[ibin][irow]++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_CWISE_CSR_STEP1_KERNEL);


template <typename ValueType, typename IndexType>
void generate_cwise_csr_scatter_bins(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const a, const float tolerance,
    gko::amp::precision_array<LinOp*, ValueType>& amat)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = gko::matrix::AMP<ValueType, IndexType>::num_precisions;
    const std::array<real_type, q> min_repr =
        get_bins_min_representable<real_type>();

    const auto nrows = a->get_size()[0];
    const ValueType* const ovals = a->get_const_values();
    const IndexType* const ocolidxs = a->get_const_col_idxs();
    const IndexType* const orow_ptrs = a->get_const_row_ptrs();

    using CsrTuple = gko::instantiation_tuple_t<
        gko::generator_partial<gko::matrix::Csr, IndexType>,
        typename gko::amp::narrow_types<ValueType>::type>;
    using ScalarPtrTuple = gko::instantiation_tuple_t<
        gko::generator<gko::ptr_type>,
        typename gko::amp::narrow_types<ValueType>::type>;

    ScalarPtrTuple xvalues;
    gko::amp::precision_array<IndexType*, ValueType> xcol_idxs;
    gko::amp::precision_array<IndexType*, ValueType> xrow_ptrs;

    // Get pointers into each bin's arrays and zero-initialize row_ptrs
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using CsrType = typename std::tuple_element<k, CsrTuple>::type;
        auto cmatk = dynamic_cast<CsrType*>(amat[k]);
        xrow_ptrs[k] = cmatk->get_row_ptrs();
        xcol_idxs[k] = cmatk->get_col_idxs();
        std::get<k>(xvalues) = cmatk->get_values();
        for (size_type irow = 0; irow <= nrows; irow++) {
            xrow_ptrs[k][irow] = 0;
        }
    });

    // Pass 1: count NNZ per row per bin, stored in row_ptrs[irow+1]
    for (int irow = 0; irow < nrows; irow++) {
        auto rnorm = static_cast<real_type>(0);
        for (auto j = orow_ptrs[irow]; j < orow_ptrs[irow + 1]; j++) {
            rnorm += std::abs(ovals[j]);
        }
        const std::array<float, q> min_bin =
            get_bins_precision_lower_bounds<real_type>(rnorm, tolerance);
        for (auto j = orow_ptrs[irow]; j < orow_ptrs[irow + 1]; j++) {
            const int ibin = get_adjusted_bin_for_entry<real_type>(
                min_bin, min_repr, std::abs(ovals[j]),
                ocolidxs[j] == static_cast<IndexType>(irow));
            if (ibin >= 0) {
                xrow_ptrs[ibin][irow + 1]++;
            }
        }
    }

    // Convert per-row counts to row pointers via prefix sum
    gko::constexpr_for<0, q, 1>([&](auto k) {
        for (size_type irow = 1; irow <= nrows; irow++) {
            xrow_ptrs[k][irow] += xrow_ptrs[k][irow - 1];
        }
    });

    // Pass 2: scatter values into bins using a per-row cursor
    // cursor[ibin] is initialized to a copy of the row pointers of that bin.
    std::array<std::vector<IndexType>, q> cursors;
    gko::constexpr_for<0, q, 1>(
        [&](auto k) { cursors[k].assign(xrow_ptrs[k], xrow_ptrs[k] + nrows); });

    for (int irow = 0; irow < nrows; irow++) {
        auto rnorm = static_cast<real_type>(0);
        for (auto j = orow_ptrs[irow]; j < orow_ptrs[irow + 1]; j++) {
            rnorm += std::abs(ovals[j]);
        }
        const std::array<float, q> min_bin =
            get_bins_precision_lower_bounds<real_type>(rnorm, tolerance);
        for (auto j = orow_ptrs[irow]; j < orow_ptrs[irow + 1]; j++) {
            const int ibin = get_adjusted_bin_for_entry<real_type>(
                min_bin, min_repr, std::abs(ovals[j]),
                ocolidxs[j] == static_cast<IndexType>(irow));
            if (ibin >= 0) {
                const auto pos = cursors[ibin][irow]++;
                xcol_idxs[ibin][pos] = ocolidxs[j];
                assign_value_to_array_tuple<0>(xvalues, ovals[j], ibin, pos);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_CWISE_CSR_SCATTER_BINS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::AMP<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    constexpr int q = matrix::AMP<ValueType, IndexType>::num_precisions;
    const auto nrows = source->get_size()[0];
    const auto ncols = source->get_size()[1];

    // Zero out the result matrix
    for (gko::size_type i = 0; i < nrows; i++) {
        for (gko::size_type j = 0; j < ncols; j++) {
            result->at(i, j) = zero<ValueType>();
        }
    }

    // Accumulate each precision bin into the result
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using bin_value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<ValueType>::type>::type;
        auto ell = dynamic_cast<const matrix::Ell<bin_value_type, IndexType>*>(
            source->get_bin_matrix(k));
        auto csr = dynamic_cast<const matrix::Csr<bin_value_type, IndexType>*>(
            source->get_bin_matrix(k));
        if (!ell && !csr) {
            GKO_NOT_SUPPORTED(source->get_bin_matrix(k));
        }
        if (ell) {
            const auto stride = ell->get_stride();
            const auto max_nnz = ell->get_num_stored_elements_per_row();
            auto vals = ell->get_const_values();
            auto cols = ell->get_const_col_idxs();
            for (gko::size_type i = 0; i < nrows; i++) {
                for (IndexType j = 0; j < max_nnz; j++) {
                    const auto col = cols[i + j * stride];
                    if (col >= 0) {
                        result->at(i, col) +=
                            static_cast<ValueType>(vals[i + j * stride]);
                    }
                }
            }
        } else {
            auto vals = csr->get_const_values();
            auto cols = csr->get_const_col_idxs();
            auto row_ptrs = csr->get_const_row_ptrs();
            for (gko::size_type i = 0; i < nrows; i++) {
                for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; j++) {
                    result->at(i, cols[j]) += static_cast<ValueType>(vals[j]);
                }
            }
        }
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::AMP<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    constexpr int q = matrix::AMP<ValueType, IndexType>::num_precisions;
    const auto diag_size = diag->get_size()[0];
    auto diag_values = diag->get_values();
    for (int i = 0; i < diag_size; i++) {
        diag_values[i] = zero<ValueType>();
    }

    gko::constexpr_for<0, q, 1>([&](auto k) {
        using bin_value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<ValueType>::type>::type;
        auto ell = dynamic_cast<const matrix::Ell<bin_value_type, IndexType>*>(
            orig->get_bin_matrix(k));
        auto csr = dynamic_cast<const matrix::Csr<bin_value_type, IndexType>*>(
            orig->get_bin_matrix(k));
        if (!ell && !csr) {
            GKO_NOT_SUPPORTED(orig->get_bin_matrix(k));
        }
        if (ell) {
            const auto stride = ell->get_stride();
            const auto max_nnz = ell->get_num_stored_elements_per_row();
            auto vals = ell->get_const_values();
            auto cols = ell->get_const_col_idxs();
            for (gko::size_type row = 0; row < diag_size; row++) {
                for (IndexType j = 0; j < max_nnz; j++) {
                    const auto col = cols[row + j * stride];
                    if (col == static_cast<IndexType>(row)) {
                        diag_values[row] =
                            static_cast<ValueType>(vals[row + j * stride]);
                    }
                }
            }
        } else {
            auto vals = csr->get_const_values();
            auto cols = csr->get_const_col_idxs();
            auto row_ptrs = csr->get_const_row_ptrs();
            for (gko::size_type row = 0; row < diag_size; row++) {
                for (auto j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
                    if (cols[j] == static_cast<IndexType>(row)) {
                        diag_values[row] = static_cast<ValueType>(vals[j]);
                    }
                }
            }
        }
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_EXTRACT_DIAGONAL_KERNEL);


}  // namespace amp
}  // namespace reference
}  // namespace kernels
}  // namespace gko
