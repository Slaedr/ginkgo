// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_AMP_HELPERS_H
#define GKO_CORE_MATRIX_AMP_HELPERS_H

#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/utils.hpp"


namespace gko {
namespace amp {


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
    const precision_array<int, ValueType> bin_max_nnz_row)
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
    const precision_array<int, ValueType> bin_max_nnz_row)
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
 * @return  Fixed-size array of LinOps, one for each allocated bin.
 */
template <typename ValueType, typename IndexType>
inline precision_array<std::unique_ptr<LinOp>, ValueType> allocate_csr_bins(
    std::shared_ptr<const Executor> exec, const dim<2>& dims,
    precision_array<gko::array<IndexType>, ValueType> bin_row_ptrs)
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
            std::move(bin_row_ptrs[k])));
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
