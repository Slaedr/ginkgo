// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <iostream>
#include <utility>

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
#include "core/base/precision_dispatch_extra.hpp"
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
GKO_REGISTER_OPERATION(reduce_bins_max, amp::reduce_bins_max);
GKO_REGISTER_OPERATION(extract_diagonal, amp::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);


}  // anonymous namespace
}  // namespace amp


/*
 * Rounds `requested` down to the nearest power of two no larger than the
 * warp size of `exec` (defaulting to 32 on executors without a native warp
 * concept), leaving 0 ("automatic") untouched. Warns once via std::cerr if
 * the value had to change.
 */
template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::normalize_subwarp_size()
{
    const int requested = parameters_.subwarp_size;
    auto exec = this->get_executor();
    if (requested == 0) {
        parameters_.subwarp_size = 0;
        return;
    }
    int warp_size = 32;
    if (auto cuda_exec = std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
        warp_size = cuda_exec->get_warp_size();
    } else if (auto hip_exec =
                   std::dynamic_pointer_cast<const HipExecutor>(exec)) {
        warp_size = hip_exec->get_warp_size();
    }
    int clamped = std::min(std::max(requested, 1), warp_size);
    int rounded = 1;
    while (rounded * 2 <= clamped) {
        rounded *= 2;
    }
    if (rounded != requested) {
        std::cerr << "AMP: subwarp_size " << requested
                  << " is not a power of two <= " << warp_size << "; using "
                  << rounded << " instead." << std::endl;
    }
    parameters_.subwarp_size = rounded;
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>& AMP<ValueType, IndexType>::operator=(
    const AMP& other)
{
    if (&other != this) {
        EnableLinOp<AMP>::operator=(other);
        this->parameters_ = other.parameters_;
        this->max_nnz_per_row_ = other.max_nnz_per_row_;
        this->num_nonempty_bins_ = other.num_nonempty_bins_;
        for (int i = 0; i < num_precisions; i++) {
            if (other.mat_bins_[i]) {
                this->mat_bins_[i] =
                    other.mat_bins_[i]->clone(this->get_executor());
            } else {
                this->mat_bins_[i] = nullptr;
            }
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>& AMP<ValueType, IndexType>::operator=(AMP&& other)
{
    if (&other != this) {
        EnableLinOp<AMP>::operator=(std::move(other));
        mat_bins_ = std::move(other.mat_bins_);
        max_nnz_per_row_ = std::move(other.max_nnz_per_row_);
        num_nonempty_bins_ = std::exchange(other.num_nonempty_bins_, 0);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    switch (parameters_.strategy) {
    case strategy_type::monolithic_classical: {
        const bool csr_bins =
            dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(
                this->get_bin_matrix(0)) != nullptr;
        mixed_precision_base_dispatch_real_complex<ValueType>(
            [this, csr_bins](auto dense_b, auto dense_x) {
                if (csr_bins) {
                    this->get_executor()->run(amp::make_spmv_csr(
                        this, dense_b, dense_x, parameters_.subwarp_size));
                } else {
                    this->get_executor()->run(
                        amp::make_spmv_ell(this, dense_b, dense_x));
                }
            },
            b, x);
        break;
    }
    case strategy_type::independent_buckets: {
        // Each bucket is applied independently via its own (Ell/Csr) apply,
        // accumulating into x. The first live bucket overwrites x, later
        // ones accumulate with alpha = beta = 1. Note this is not
        // bit-identical to monolithic_classical (different summation order
        // and accumulator), and requires a mixed-precision build
        // (GINKGO_MIXED_PRECISION=ON) to avoid narrowing b/x to each
        // bucket's own precision.
        bool first = true;
        gko::constexpr_for<0, num_precisions, 1>([&](auto k) {
            if (!mat_bins_[k]) {
                return;
            }
            if (first) {
                mat_bins_[k]->apply(b, x);
                first = false;
            } else {
                mat_bins_[k]->apply(one_, b, one_, x);
            }
        });
        break;
    }
    default:
        GKO_NOT_SUPPORTED(parameters_.strategy);
    }
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    switch (parameters_.strategy) {
    case strategy_type::monolithic_classical: {
        const bool csr_bins =
            dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(
                this->get_bin_matrix(0)) != nullptr;
        mixed_precision_base_dispatch_real_complex<ValueType>(
            [this, csr_bins, alpha, beta](auto dense_b, auto dense_x) {
                auto d_alpha = make_temporary_conversion<ValueType>(alpha);
                auto d_beta = make_temporary_conversion<
                    typename std::decay_t<decltype(*dense_x)>::value_type>(
                    beta);
                if (csr_bins) {
                    this->get_executor()->run(amp::make_advanced_spmv_csr(
                        d_alpha.get(), this, dense_b, d_beta.get(), dense_x,
                        parameters_.subwarp_size));
                } else {
                    this->get_executor()->run(amp::make_advanced_spmv_ell(
                        d_alpha.get(), this, dense_b, d_beta.get(), dense_x));
                }
            },
            b, x);
        break;
    }
    case strategy_type::independent_buckets: {
        // x = alpha * bin_k * b + beta * x for the first live bucket, then
        // x = alpha * bin_k * b + 1 * x for every subsequent one, so beta
        // is only applied once.
        bool first = true;
        gko::constexpr_for<0, num_precisions, 1>([&](auto k) {
            if (!mat_bins_[k]) {
                return;
            }
            mat_bins_[k]->apply(alpha, b, first ? beta : one_.get(), x);
            first = false;
        });
        break;
    }
    default:
        GKO_NOT_SUPPORTED(parameters_.strategy);
    }
}


template <typename ValueType, typename IndexType>
gko::amp::precision_array<std::unique_ptr<const LinOp>, ValueType>
AMP<ValueType, IndexType>::generate_amp_impl(
    const matrix::Csr<ValueType, IndexType>* const mtx)
{
    auto exec = this->get_executor();
    constexpr int q = AMP<ValueType, IndexType>::num_precisions;
    const auto tol = parameters_.tolerance;
    const auto ratio = parameters_.bin_foldup_nnz_ratio;
    const bool force_diagonal_0 = parameters_.high_precision_diagonal;

    std::array<gko::array<IndexType>, num_precisions> row_sizes;

    gko::amp::precision_array<IndexType*, ValueType> bin_row_sizes;
    const size_type nrows = mtx->get_size()[0];
    gko::constexpr_for<0, q, 1>([&](auto k) {
        row_sizes[k].set_executor(exec);
        row_sizes[k].resize_and_reset(nrows + 1);
        bin_row_sizes[k] = row_sizes[k].get_data();
    });

    // Computes bin_row_sizes and max_nnz_per_row_ for the given max_bin,
    // turning the per-row counts into prefix-summed row pointers. Entries
    // that would land in a bin above max_bin are clamped down to it.
    auto compute_row_sizes = [&](int max_bin) {
        exec->run(amp::make_generate_cwise_csr_calculate_row_sizes(
            mtx, tol, max_bin, force_diagonal_0, bin_row_sizes));
        exec->run(amp::make_reduce_bins_max(mtx, row_sizes, max_nnz_per_row_));
        for (int k = 0; k < q; k++) {
            exec->run(
                amp::make_prefix_sum_nonnegative(bin_row_sizes[k], nrows + 1));
        }
        exec->synchronize();
    };
    // Reads the total nonzero count of each bin from the prefix-summed row
    // pointers computed by compute_row_sizes.
    auto read_bin_nnz = [&] {
        std::array<int64, q> counts{};
        for (int k = 0; k < q; k++) {
            counts[k] = static_cast<int64>(get_element(row_sizes[k], nrows));
        }
        return counts;
    };

    compute_row_sizes(q - 1);
    auto bin_nnz = read_bin_nnz();
    const int fold_max_bin = gko::amp::compute_last_active_bin<int64, q>(
        bin_nnz, static_cast<int64>(mtx->get_num_stored_elements()), ratio);
    if (fold_max_bin < q - 1) {
        // Some trailing bins were folded up: recompute the row sizes (and
        // max_nnz_per_row_) with entries clamped to the new max_bin.
        compute_row_sizes(fold_max_bin);
        bin_nnz = read_bin_nnz();
    }
    num_nonempty_bins_ = 0;
    for (int k = 0; k < q; k++) {
        if (bin_nnz[k] > 0) {
            num_nonempty_bins_++;
        }
    }

    auto abins = gko::amp::allocate_csr_bins<ValueType, IndexType>(
        exec, mtx->get_size(), std::move(row_sizes), parameters_.csr_strategy);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    static_assert(num_bins == q, "Wrong number of bins!");
    gko::amp::precision_array<gko::LinOp*, ValueType> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    exec->run(amp::make_generate_cwise_csr_scatter_bins(
        mtx, tol, fold_max_bin, force_diagonal_0, amat));

    gko::amp::precision_array<std::unique_ptr<const LinOp>, ValueType> cabins;
    for (int i = 0; i < q; i++) {
        cabins[i] = std::move(abins[i]);
    }
    return cabins;
}


template <typename ValueType, typename IndexType>
gko::amp::precision_array<std::unique_ptr<const LinOp>, ValueType>
AMP<ValueType, IndexType>::generate_amp_impl(
    const matrix::Ell<ValueType, IndexType>* const mtx)
{
    auto exec = this->get_executor();
    constexpr int q = AMP<ValueType, IndexType>::num_precisions;
    const auto tol = parameters_.tolerance;
    const auto ratio = parameters_.bin_foldup_nnz_ratio;
    const bool force_diagonal_0 = parameters_.high_precision_diagonal;

    // Computes max_nnz_per_row_ and bin_nnz for the given max_bin. Entries
    // that would land in a bin above max_bin are clamped down to it.
    gko::amp::precision_array<int64, ValueType> bin_nnz_dev;
    auto compute_max_nnz = [&](int max_bin) {
        exec->run(amp::make_generate_cwise_ell_max_nnz_per_row(
            mtx, tol, max_bin, force_diagonal_0, max_nnz_per_row_,
            bin_nnz_dev));
        exec->synchronize();
    };

    compute_max_nnz(q - 1);
    std::array<int64, q> bin_nnz{};
    int64 total_nnz = 0;
    for (int k = 0; k < q; k++) {
        bin_nnz[k] = bin_nnz_dev[k];
        total_nnz += bin_nnz[k];
    }
    const int fold_max_bin =
        gko::amp::compute_last_active_bin<int64, q>(bin_nnz, total_nnz, ratio);
    if (fold_max_bin < q - 1) {
        // Some trailing bins were folded up: recompute max_nnz_per_row_ with
        // entries clamped to the new max_bin (a merged bin's maximum row
        // length is not just the sum of the old maxima, so it must be
        // recomputed from the actual data).
        compute_max_nnz(fold_max_bin);
        for (int k = 0; k < q; k++) {
            bin_nnz[k] = bin_nnz_dev[k];
        }
    }
    num_nonempty_bins_ = 0;
    for (int k = 0; k < q; k++) {
        if (bin_nnz[k] > 0) {
            num_nonempty_bins_++;
        }
    }

    auto abins = gko::amp::allocate_bins<ValueType, IndexType>(
        exec, mtx->get_size(), max_nnz_per_row_);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    static_assert(num_bins == AMP<ValueType, IndexType>::num_precisions,
                  "Wrong number of bins!");
    gko::amp::precision_array<gko::LinOp*, ValueType> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    exec->run(amp::make_generate_ell_scatter_bins(mtx, tol, fold_max_bin,
                                                  force_diagonal_0, amat));

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
    auto a_ell = dynamic_cast<const matrix::Ell<ValueType, IndexType>*>(mtx);
    if (a_ell) {
        return generate_amp_impl(a_ell);
    } else {
        auto a_csr =
            dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(mtx);
        if (a_csr) {
            return generate_amp_impl(a_csr);
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
    : EnableLinOp<AMP<ValueType, IndexType>>(std::move(exec))
{
    init_one();
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::init_one()
{
    one_ = gko::share(gko::initialize<Dense<ValueType>>({gko::one<ValueType>()},
                                                        this->get_executor()));
}


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
    using Ell = matrix::Ell<ValueType, IndexType>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    // Determine underlying format from bin type.
    // Note that the bins would have been set by the factory build,
    //   which is the only way to create an AMP matrix from the outside.
    std::unique_ptr<LinOp> base_mtx;
    auto base_ell = dynamic_cast<const Ell*>(mat_bins_[0].get());
    auto base_csr = dynamic_cast<const Csr*>(mat_bins_[0].get());
    if (base_ell) {
        base_mtx = std::move(Ell::create(exec));
    } else if (base_csr) {
        base_mtx = std::move(Csr::create(exec));
    }
    as<ReadableFromMatrixData<ValueType, IndexType>>(base_mtx.get())
        ->read(data);
    this->set_size(base_mtx->get_size());
    mat_bins_ = generate_amp(base_mtx.get());
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
    using Ell = matrix::Ell<ValueType, IndexType>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    // Determine underlying format from bin type.
    // Note that the bins would have been set by the constructor.
    std::unique_ptr<LinOp> base_mtx;
    auto base_ell = dynamic_cast<const Ell*>(mat_bins_[0].get());
    auto base_csr = dynamic_cast<const Csr*>(mat_bins_[0].get());
    if (base_ell) {
        base_mtx = std::move(Ell::create(exec));
    } else if (base_csr) {
        base_mtx = std::move(Csr::create(exec));
    }
    as<ReadableFromMatrixData<ValueType, IndexType>>(base_mtx.get())
        ->read(data);
    this->set_size(base_mtx->get_size());
    mat_bins_ = generate_amp(base_mtx.get());
}


#define GKO_DECLARE_AMP_MATRIX(ValueType, IndexType) \
    class AMP<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(GKO_DECLARE_AMP_MATRIX);


}  // namespace matrix
}  // namespace gko
