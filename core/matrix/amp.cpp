// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/allocator.hpp"
#include "core/base/array_access.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/amp_kernels.hpp"


namespace gko {
namespace matrix {
namespace amp {


template <typename ValueType, typename IndexType>
using bin_mtx_type = gko::matrix::Ell<ValueType, IndexType>;


namespace {


GKO_REGISTER_OPERATION(spmv_ell, amp::spmv_ell);
GKO_REGISTER_OPERATION(spmv_csr, amp::spmv_csr);
GKO_REGISTER_OPERATION(advanced_spmv_ell, amp::advanced_spmv_ell);
GKO_REGISTER_OPERATION(advanced_spmv_csr, amp::advanced_spmv_csr);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(fill_in_dense, amp::fill_in_dense);
GKO_REGISTER_OPERATION(generate_cwise_ell_max_nnz_per_row,
                       amp::generate_cwise_ell_max_nnz_per_row);
GKO_REGISTER_OPERATION(generate_ell_scatter_bins,
                       amp::generate_ell_scatter_bins);
GKO_REGISTER_OPERATION(generate_cwise_csr_calculate_row_sizes,
                       amp::generate_cwise_csr_calculate_row_sizes);
GKO_REGISTER_OPERATION(generate_cwise_csr_scatter_bins,
                       amp::generate_cwise_csr_scatter_bins);
GKO_REGISTER_OPERATION(extract_diagonal, amp::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);


}  // anonymous namespace
}  // namespace amp


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>& AMP<ValueType, IndexType>::operator=(
    const AMP& other)
{
    if (&other != this) {
        EnableLinOp<AMP>::operator=(other);
        this->parameters_ = other.parameters_;
        for (int i = 0; i < num_precisions; i++) {
            if (other.mat_bins_[i]) {
                this->mat_bins_[i] =
                    other.mat_bins_[i]->clone(this->get_executor());
            } else {
                this->mat_bins_[i] = nullptr;
            }
        }
        row_sizes_ = other.row_sizes_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>& AMP<ValueType, IndexType>::operator=(AMP&& other)
{
    if (&other != this) {
        EnableLinOp<AMP>::operator=(std::move(other));
        mat_bins_ = std::move(other.mat_bins_);
        row_sizes_ = std::move(other.row_sizes_);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    const bool csr_bins =
        dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(
            this->get_bin_matrix(0)) != nullptr;
    if (csr_bins) {
        this->get_executor()->run(
            amp::make_spmv_csr(this, gko::as<matrix::Dense<ValueType>>(b),
                               gko::as<matrix::Dense<ValueType>>(x)));
    } else {
        this->get_executor()->run(
            amp::make_spmv_ell(this, gko::as<matrix::Dense<ValueType>>(b),
                               gko::as<matrix::Dense<ValueType>>(x)));
    }
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    // mixed_precision_dispatch_real_complex<ValueType>(
    //     [this, alpha, beta](auto dense_b, auto dense_x) {
    //         auto d_alpha = make_temporary_conversion<ValueType>(alpha);
    //         auto d_beta = make_temporary_conversion<
    //             typename std::decay_t<decltype(*dense_x)>::value_type>(beta);
    //         this->get_executor()->run(amp::make_advanced_spmv(
    //             d_alpha.get(), this, dense_b, d_beta.get(), dense_x));
    //     },
    //     b, x);
    const bool csr_bins =
        dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(
            this->get_bin_matrix(0)) != nullptr;
    if (csr_bins) {
        this->get_executor()->run(amp::make_advanced_spmv_csr(
            gko::as<matrix::Dense<ValueType>>(alpha), this,
            gko::as<matrix::Dense<ValueType>>(b),
            gko::as<matrix::Dense<ValueType>>(beta),
            gko::as<matrix::Dense<ValueType>>(x)));
    } else {
        this->get_executor()->run(amp::make_advanced_spmv_ell(
            gko::as<matrix::Dense<ValueType>>(alpha), this,
            gko::as<matrix::Dense<ValueType>>(b),
            gko::as<matrix::Dense<ValueType>>(beta),
            gko::as<matrix::Dense<ValueType>>(x)));
    }
}


template <typename ValueType, typename IndexType>
auto generate_amp_impl(
    const matrix::Csr<ValueType, IndexType>* const mtx,
    std::shared_ptr<const Executor> exec, const float tol,
    gko::amp::precision_array<gko::array<IndexType>, ValueType>& row_sizes)
{
    constexpr int q = AMP<ValueType, IndexType>::num_precisions;
    gko::amp::precision_array<IndexType*, ValueType> bin_row_sizes;
    const size_type nrows = mtx->get_size()[0];
    gko::constexpr_for<0, q, 1>([&](auto k) {
        row_sizes[k].resize_and_reset(nrows + 1);
        bin_row_sizes[k] = row_sizes[k].get_data();
    });

    exec->run(amp::make_generate_cwise_csr_calculate_row_sizes(mtx, tol,
                                                               bin_row_sizes));

    for (int k = 0; k < q; k++) {
        exec->run(
            amp::make_prefix_sum_nonnegative(bin_row_sizes[k], nrows + 1));
    }
    exec->synchronize();

    auto abins = gko::amp::allocate_csr_bins<ValueType, IndexType>(
        exec, mtx->get_size(), std::move(row_sizes));
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    static_assert(num_bins == q, "Wrong number of bins!");
    gko::amp::precision_array<gko::LinOp*, ValueType> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    exec->run(amp::make_generate_cwise_csr_scatter_bins(mtx, tol, amat));

    gko::amp::precision_array<std::unique_ptr<const LinOp>, ValueType> cabins;
    for (int i = 0; i < q; i++) {
        cabins[i] = std::move(abins[i]);
    }
    return cabins;
}


template <typename ValueType, typename IndexType>
auto generate_amp_impl(const matrix::Ell<ValueType, IndexType>* const mtx,
                       std::shared_ptr<const Executor> exec, const float tol)
{
    gko::amp::precision_array<int, ValueType> max_nnz;
    exec->run(amp::make_generate_cwise_ell_max_nnz_per_row(mtx, tol, max_nnz));

    auto abins = gko::amp::allocate_bins<ValueType, IndexType>(
        exec, mtx->get_size(), max_nnz);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    static_assert(num_bins == AMP<ValueType, IndexType>::num_precisions,
                  "Wrong number of bins!");
    gko::amp::precision_array<gko::LinOp*, ValueType> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    exec->run(amp::make_generate_ell_scatter_bins(mtx, tol, amat));

    gko::amp::precision_array<std::unique_ptr<const LinOp>, ValueType> cabins;
    for (int i = 0; i < matrix::AMP<ValueType, IndexType>::num_precisions;
         i++) {
        cabins[i] = std::move(abins[i]);
    }
    return cabins;
}


template <typename ValueType, typename IndexType>
std::array<std::unique_ptr<const LinOp>,
           AMP<ValueType, IndexType>::num_precisions>
AMP<ValueType, IndexType>::generate_amp(const LinOp* const mtx)
{
    const auto tol = parameters_.tolerance;
    // typedef std::array<std::unique_ptr<const LinOp>, num_precisions>
    // ret_type;
    auto a_ell = dynamic_cast<const matrix::Ell<ValueType, IndexType>*>(mtx);
    if (a_ell) {
        return generate_amp_impl<ValueType, IndexType>(
            a_ell, this->get_executor(), tol);
    } else {
        auto a_csr =
            dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(mtx);
        if (a_csr) {
            return generate_amp_impl<ValueType, IndexType>(
                a_csr, this->get_executor(), tol, row_sizes_);
        } else {
            GKO_NOT_SUPPORTED(mtx);
        }
    }
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::convert_to(Dense<ValueType>* const result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(amp::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::move_to(Dense<ValueType>* const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
AMP<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(amp::make_extract_diagonal(this, diag.get()));
    return diag;
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>::AMP(std::shared_ptr<const Executor> exec)
    : EnableLinOp<AMP<ValueType, IndexType>>(std::move(exec)),
      row_sizes_(create_row_sizes())
{}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>::AMP(const AMP& other) : AMP(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>::AMP(AMP&& other) : AMP(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::read(
    const matrix_data<ValueType, IndexType>& data)
{
    auto exec = this->get_executor();
    auto ell = Ell<ValueType, IndexType>::create(exec);
    ell->read(data);
    this->set_size(ell->get_size());
    mat_bins_ = generate_amp(ell.get());
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::read(device_mat_data&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::read(
    const device_matrix_data<ValueType, IndexType>& data)
{
    auto exec = this->get_executor();
    auto ell = Ell<ValueType, IndexType>::create(exec);
    ell->read(data);
    this->set_size(ell->get_size());
    mat_bins_ = generate_amp(ell.get());
}


template <typename ValueType, typename IndexType>
std::array<gko::array<IndexType>, AMP<ValueType, IndexType>::num_precisions>
AMP<ValueType, IndexType>::create_row_sizes() const
{
    constexpr int q = AMP<ValueType, IndexType>::num_precisions;
    std::array<gko::array<IndexType>, num_precisions> arr;
    for (int i = 0; i < q; i++) {
        arr[i] = gko::array<IndexType>(this->get_executor());
    }
    return arr;
}


#define GKO_DECLARE_AMP_MATRIX(ValueType, IndexType) \
    class AMP<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(GKO_DECLARE_AMP_MATRIX);


}  // namespace matrix
}  // namespace gko
