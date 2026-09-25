// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_AMP_HELPERS_H
#define GKO_CORE_MATRIX_AMP_HELPERS_H

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/utils.hpp"


namespace gko {
namespace amp {


/**
 * Determines the lowest-precision bin that should still be generated on its
 * own (the "last active bin"), folding sparse trailing bins into higher
 * precision as needed.
 *
 * Starting from the lowest-precision bin (index `q - 1`), a bin is folded
 * into the next higher precision bin (its count added to `bin_nnz[k - 1]`,
 * and its own count zeroed) whenever its nonzero count is below `ratio *
 * total_nnz`. The check then repeats for the new last bin, using its
 * (possibly already merged) count, until a bin meets the threshold or bin 0
 * is reached. Bin 0 is never folded away.
 *
 * @tparam CountType  Integer type used to count nonzeros per bin.
 * @tparam q  Number of precision bins.
 *
 * @param bin_nnz  Nonzero count for each bin, updated in place to reflect any
 *                 folding: entries above the returned index are zeroed out
 *                 and merged into the entry at the returned index.
 * @param total_nnz  Total number of nonzeros in the original (un-binned)
 *                   matrix; the folding threshold is `ratio * total_nnz`.
 * @param ratio  Folding threshold, as a ratio of `total_nnz`. A value <= 0
 *              disables folding.
 * @return  The index of the lowest-precision bin that remains active, i.e.
 *          all bins above it are empty.
 */
template <typename CountType, int q>
inline int compute_last_active_bin(std::array<CountType, q>& bin_nnz,
                                   CountType total_nnz, float ratio)
{
    int last = q - 1;
    if (ratio <= 0.0f) {
        return last;
    }
    const auto threshold =
        static_cast<double>(ratio) * static_cast<double>(total_nnz);
    while (last > 0 && static_cast<double>(bin_nnz[last]) < threshold) {
        bin_nnz[last - 1] += bin_nnz[last];
        bin_nnz[last] = CountType{};
        last--;
    }
    return last;
}


/**
 * Allocate an Ell matrix for each precision bin supported, starting at the
 * precision of the parameter ValueType.
 *
 * @tparam ValueType  Scalar type of the highest precision bin.
 * @tparam IndexType  Index type for the concrete matrix.
 *
 * @param dims  (Common) dimensions of all bins.
 * @param bin_max_nnz_row  Max nonzeros per row for each bin.
 * @return  Fixed-size array of LinOps, one for each allocated bin.
 */
template <typename ValueType, typename IndexType>
inline precision_array<std::unique_ptr<LinOp>, ValueType> allocate_bins(
    std::shared_ptr<const Executor> exec, const dim<2>& dims,
    const precision_array<IndexType, ValueType> bin_max_nnz_row)
{
    using last_precision =
        std::tuple_element<num_amp_precisions - 1, supported_precisions>::type;
    constexpr int highest_idx = precision_index<last_precision>::index;
    constexpr int starting_idx =
        precision_index<gko::remove_complex<ValueType>>::index;
    precision_array<std::unique_ptr<LinOp>, ValueType> bins;
    gko::constexpr_for<starting_idx, highest_idx + 1, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::supported_types<ValueType>::type>::type;
        bins[k - starting_idx] =
            std::move(matrix::Ell<value_type, IndexType>::create(
                exec, dims, bin_max_nnz_row[k - starting_idx]));
    });
    return bins;
}

/**
 * Allocate an Ell matrix for each precision bin supported, starting at the
 * precision of the parameter ValueType.
 * Unlike @ref allocate_bins, this function returns a tuple of the concrete
 * types.
 *
 * @tparam ValueType  Scalar type of the highest precision bin.
 * @tparam IndexType  Index type for the concrete matrix.
 *
 * @param dims  (Common) dimensions of all bins.
 * @param bin_max_nnz_row  Max nonzeros per row for each bin.
 * @return  Tuple of unique_ptrs to Ell matrices, one for each allocated bin.
 */
template <typename ValueType, typename IndexType>
inline auto allocate_bins_tuple(
    std::shared_ptr<const Executor> exec, const dim<2>& dims,
    const precision_array<IndexType, ValueType> bin_max_nnz_row)
{
    using last_precision =
        std::tuple_element<num_amp_precisions - 1, supported_precisions>::type;
    constexpr int highest_idx = precision_index<last_precision>::index;
    constexpr int starting_idx = precision_index<ValueType>::index;
    using EllTuple = gko::transformed_instantiation_tuple_t<
        std::unique_ptr, gko::generator_partial<gko::matrix::Ell, IndexType>,
        typename gko::amp::narrow_types<ValueType>::type>;
    EllTuple bins;
    gko::constexpr_for<starting_idx, highest_idx + 1, 1>([&](auto idx) {
        using value_type = typename std::tuple_element<
            idx, typename gko::amp::supported_types<ValueType>::type>::type;
        std::get<idx - starting_idx>(bins) =
            std::move(matrix::Ell<value_type, IndexType>::create(
                exec, dims, bin_max_nnz_row[idx - starting_idx]));
    });
    return bins;
}


/**
 * Builds the matrix::Csr strategy corresponding to the requested
 * matrix::amp_csr_strategy_type, for the given ValueType/IndexType bucket.
 *
 * `load_balance` and `automatical` need an executor-typed constructor;
 * mirrors matrix::Csr::make_default_strategy (include/ginkgo/core/matrix/
 * csr.hpp) so that amp_csr_strategy_type::automatical reproduces the same
 * fallback to `classical` on executors without a native warp concept.
 *
 * @tparam ValueType  Scalar type of the bucket.
 * @tparam IndexType  Index type for the concrete matrix.
 */
template <typename ValueType, typename IndexType>
inline std::shared_ptr<
    typename matrix::Csr<ValueType, IndexType>::strategy_type>
make_csr_strategy(std::shared_ptr<const Executor> exec,
                  matrix::amp_csr_strategy_type strategy)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    using csr_strategy_type = matrix::amp_csr_strategy_type;
    switch (strategy) {
    case csr_strategy_type::classical:
        return std::make_shared<typename Csr::classical>();
    case csr_strategy_type::merge_path:
        return std::make_shared<typename Csr::merge_path>();
    case csr_strategy_type::sparselib:
        return std::make_shared<typename Csr::sparselib>();
    case csr_strategy_type::load_balance:
    case csr_strategy_type::automatical:
    default: {
        const bool automatical = strategy == csr_strategy_type::automatical;
        if (auto cuda_exec =
                std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
            return automatical
                       ? std::static_pointer_cast<typename Csr::strategy_type>(
                             std::make_shared<typename Csr::automatical>(
                                 cuda_exec))
                       : std::static_pointer_cast<typename Csr::strategy_type>(
                             std::make_shared<typename Csr::load_balance>(
                                 cuda_exec));
        } else if (auto hip_exec =
                       std::dynamic_pointer_cast<const HipExecutor>(exec)) {
            return automatical
                       ? std::static_pointer_cast<typename Csr::strategy_type>(
                             std::make_shared<typename Csr::automatical>(
                                 hip_exec))
                       : std::static_pointer_cast<typename Csr::strategy_type>(
                             std::make_shared<typename Csr::load_balance>(
                                 hip_exec));
        } else if (auto dpcpp_exec =
                       std::dynamic_pointer_cast<const DpcppExecutor>(exec)) {
            return automatical
                       ? std::static_pointer_cast<typename Csr::strategy_type>(
                             std::make_shared<typename Csr::automatical>(
                                 dpcpp_exec))
                       : std::static_pointer_cast<typename Csr::strategy_type>(
                             std::make_shared<typename Csr::load_balance>(
                                 dpcpp_exec));
        } else {
            return std::make_shared<typename Csr::classical>();
        }
    }
    }
}


/**
 * Allocate a CSR matrix for each precision bin supported, starting at the
 * precision of the parameter ValueType.
 *
 * @tparam ValueType  Scalar type of the highest precision bin.
 * @tparam IndexType  Index type for the concrete matrix.
 *
 * @param dims  (Common) dimensions of all bins.
 * @param bin_row_ptrs  Pre-computed row pointer arrays for each bin.
 *                      To avoid a deep copy, pass an rvalue reference using
 *                      `std::move(bin_row_ptrs_arg)`.
 * @param csr_strategy  SpMV strategy to apply to every allocated bucket.
 * @return  Fixed-size array of LinOps, one for each allocated bin.
 */
template <typename ValueType, typename IndexType>
inline precision_array<std::unique_ptr<LinOp>, ValueType> allocate_csr_bins(
    std::shared_ptr<const Executor> exec, const dim<2>& dims,
    precision_array<gko::array<IndexType>, ValueType> bin_row_ptrs,
    matrix::amp_csr_strategy_type csr_strategy =
        matrix::amp_csr_strategy_type::automatical)
{
    constexpr int q = gko::amp::narrow_types<ValueType>::num_types;
    precision_array<std::unique_ptr<LinOp>, ValueType> bins;
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<ValueType>::type>::type;
        // copy nnz to host and allocate arrays
        auto ref = exec->get_master();
        IndexType nnz{};
        ref->copy_from(exec, 1, bin_row_ptrs[k].get_data() + dims[0], &nnz);
        array<IndexType> col_idxs(exec, nnz);
        array<value_type> values(exec, nnz);
        // auto row_ptrs = bin_row_ptrs[k];
        bins[k] = std::move(matrix::Csr<value_type, IndexType>::create(
            exec, dims, std::move(values), std::move(col_idxs),
            std::move(bin_row_ptrs[k]),
            make_csr_strategy<value_type, IndexType>(exec, csr_strategy)));
    });
    return bins;
}

/**
 * Helper function to get array of pointers from array of arrays.
 */
template <typename T, typename ValueType>
inline precision_array<T*, ValueType> get_pointer_array(
    precision_array<gko::array<T>, ValueType>& a)
{
    constexpr int q = gko::amp::narrow_types<ValueType>::num_types;
    precision_array<T*, ValueType> b;
    gko::constexpr_for<0, q, 1>([&](auto k) { b[k] = a[k].get_data(); });
    return b;
}

/**
 * Helper function to get array of pointers to const from const array of arrays.
 */
template <typename T, typename ValueType>
inline precision_array<const T*, ValueType> get_const_pointer_array(
    const precision_array<gko::array<T>, ValueType>& a)
{
    constexpr int q = gko::amp::narrow_types<ValueType>::num_types;
    precision_array<T*, ValueType> b;
    gko::constexpr_for<0, q, 1>([&](auto k) { b[k] = a[k]->get_const_data(); });
    return b;
}


}  // namespace amp
}  // namespace gko


#endif  // GKO_CORE_MATRIX_AMP_HELPERS_H
