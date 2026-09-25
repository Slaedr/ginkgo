// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/unified/matrix/amp_algorithms.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/base/utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Gauss Seidel solver namespace.
 *
 * @ingroup gssdl
 */
namespace gssdl {

namespace gkerd = gko::kernels::GKO_DEVICE_NAMESPACE;
using gkerd::reduce;

constexpr int default_block_size = 1024;

constexpr uint32 classical_oversubscription = 8;
using classical_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4, 2, 1>;

template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void mc_fgs_ell(
    const IndexType max_nnz_rows, const size_type stride,
    const IndexType* const __restrict__ col_idxs,
    const MatrixValueType* const __restrict__ values, const IndexType begin_row,
    const IndexType end_row, const size_type b_stride,
    const InputValueType* const __restrict__ b, const size_type x_stride,
    OutputValueType* const __restrict__ x,
    stopping_status* const __restrict__ stopstatus, const bool first_iter)
{
    using highest_type = gko::highest_precision<InputValueType, MatrixValueType,
                                                OutputValueType>;
    // TODO: Optimize cached loads/stores
    const auto row = begin_row + blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= end_row) {
        return;
    }
    const auto irhs = blockIdx.y;
    if (first_iter && blockIdx.x == 0 && threadIdx.x == 0) {
        stopstatus[irhs].reset();
    }
    auto sum = static_cast<highest_type>(b[row * b_stride + irhs]);
    auto diag = zero<MatrixValueType>();

    constexpr auto invalid = invalid_index<IndexType>();

    for (IndexType k = 0; k < max_nnz_rows; ++k) {
        const auto col = col_idxs[k * stride + row];
        if (col == invalid) {
            continue;
        }
        const auto val = values[k * stride + row];
        if (col == row) {
            diag = val;
        } else {
            sum -= static_cast<highest_type>(val) *
                   static_cast<highest_type>(x[col * x_stride + irhs]);
        }
    }

    if (diag != zero<MatrixValueType>()) {
        x[row * x_stride + irhs] =
            static_cast<OutputValueType>(sum / static_cast<highest_type>(diag));
    }
}


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void multicolor_fgs_ell(std::shared_ptr<const DefaultExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::Ell<MatrixValueType, IndexType>* const a,
                        const matrix::Dense<InputValueType>* const b,
                        matrix::Dense<OutputValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    if (color_ptrs.size() < 2) {
        return;
    }

    using d_m_val_type = typename gkerd::device_type<MatrixValueType>;
    using d_i_val_type = typename gkerd::device_type<InputValueType>;
    using d_o_val_type = typename gkerd::device_type<OutputValueType>;
    const auto num_colors = static_cast<int>(color_ptrs.size() - 1);
    const auto num_rhs = b->get_size()[1];
    const auto nnz_per_row = a->get_num_stored_elements_per_row();
    const auto stride = a->get_stride();
    const auto col_idxs = a->get_const_col_idxs();
    const auto values = as_device_type(a->get_const_values());
    const auto x_vals = as_device_type(x->get_values());
    const auto b_vals = as_device_type(b->get_const_values());
    const auto x_stride = x->get_stride();
    const auto b_stride = b->get_stride();

    for (int color = 0; color < num_colors; ++color) {
        const auto row_begin = color_ptrs[color];
        const auto row_end = color_ptrs[color + 1];
        const auto nrows = row_end - row_begin;
        const dim3 nblocks{
            static_cast<uint32>(ceildiv(nrows, default_block_size)),
            static_cast<uint32>(num_rhs), 1u};
        mc_fgs_ell<d_i_val_type, d_m_val_type, d_o_val_type, IndexType>
            <<<nblocks, default_block_size, 0, exec->get_stream()>>>(
                nnz_per_row, stride, col_idxs, values, row_begin, row_end,
                b_stride, b_vals, x_stride, x_vals, stop_status->get_data(),
                first_iter);
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL);


// Tuple of device const pointers to relevant scalar types
template <typename highest_type>
using ScalarDCPtrTuple =
    gko::instantiation_tuple_t<gko::generator<gko::ptr_to_const_type>,
                               typename amp::narrow_types<highest_type>::type>;

constexpr int amp_block_size = 512;

template <typename IValueType, typename MValueType, typename OValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void mc_fgs_amp(
    const amp::precision_array<size_type, MValueType> bin_strides,
    const amp::precision_array<uint32, MValueType> bin_max_nnz_rows,
    const amp::precision_array<const IndexType*, MValueType> bin_col_idxs,
    const ScalarDCPtrTuple<MValueType> bin_values, const IndexType begin_row,
    const IndexType end_row, const size_type b_stride,
    const IValueType* const __restrict__ b, const size_type x_stride,
    OValueType* const __restrict__ x,
    stopping_status* const __restrict__ stopstatus, const bool first_iter)
{
    constexpr int q = amp::narrow_types<MValueType>::num_types;
    using highest_type =
        gko::highest_precision<IValueType, MValueType, OValueType>;
    // TODO: Optimize cached loads/stores
    const auto row = begin_row + blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= end_row) {
        return;
    }
    const auto irhs = blockIdx.y;
    if (first_iter && blockIdx.x == 0 && threadIdx.x == 0) {
        stopstatus[irhs].reset();
    }
    auto sum = static_cast<highest_type>(b[row * b_stride + irhs]);
    auto diag = zero<MValueType>();

    constexpr auto invalid = invalid_index<IndexType>();

    gko::constexpr_for<0, q, 1>([&](auto k) {
        // using value_type = typename std::tuple_element<
        //     k, typename narrow_types<MValueType>::type>::type;
        // using mult_type = gko::highest_precision<value_type, IValueType>;
        // using highest_type = gko::highest_precision<mult_type, OValueType>;
        const auto stride = bin_strides[k];
        auto avals = std::get<k>(bin_values);
        auto acols = bin_col_idxs[k];
        const auto max_nnz = bin_max_nnz_rows[k];
        for (uint32 j = 0; j < max_nnz; ++j) {
            const auto col = acols[j * stride + row];
            if (col == invalid) {
                continue;
            }
            const auto val = avals[j * stride + row];
            if (col == row) {
                diag = static_cast<MValueType>(val);
            } else {
                sum -= static_cast<highest_type>(val) *
                       static_cast<highest_type>(x[col * x_stride + irhs]);
            }
        }
    });

    if (diag != zero<MValueType>()) {
        x[row * x_stride + irhs] =
            static_cast<OValueType>(sum / static_cast<highest_type>(diag));
    }
}

template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void multicolor_fgs_amp(std::shared_ptr<const DefaultExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::AMP<MatrixValueType, IndexType>* const a,
                        const matrix::Dense<InputValueType>* const b,
                        matrix::Dense<OutputValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    if (color_ptrs.size() < 2) {
        return;
    }

    using d_m_val_type = typename gkerd::device_type<MatrixValueType>;
    using d_i_val_type = typename gkerd::device_type<InputValueType>;
    using d_o_val_type = typename gkerd::device_type<OutputValueType>;
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    const auto num_colors = static_cast<int>(color_ptrs.size() - 1);
    const auto num_rhs = b->get_size()[1];
    const auto x_vals = as_device_type(x->get_values());
    const auto b_vals = as_device_type(b->get_const_values());
    const auto x_stride = x->get_stride();
    const auto b_stride = b->get_stride();

    // Get precision buckets' arrays
    ScalarDCPtrTuple<d_m_val_type> avalues;
    amp::precision_array<const IndexType*, d_m_val_type> acol_idxs;
    amp::precision_array<size_type, d_m_val_type> bin_strides;
    amp::precision_array<uint32, d_m_val_type> bin_max_nnzs;
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        using EllType = matrix::Ell<value_type, IndexType>;
        auto ematk = dynamic_cast<const EllType*>(a->get_bin_matrix(k));
        if (!ematk) {
            GKO_NOT_SUPPORTED(ematk);
        }
        acol_idxs[k] = ematk->get_const_col_idxs();
        bin_strides[k] = ematk->get_stride();
        bin_max_nnzs[k] =
            static_cast<uint32>(ematk->get_num_stored_elements_per_row());
        std::get<k>(avalues) = as_device_type(ematk->get_const_values());
    });

    for (int color = 0; color < num_colors; ++color) {
        const auto row_begin = color_ptrs[color];
        const auto row_end = color_ptrs[color + 1];
        const auto nrows = row_end - row_begin;
        const dim3 nblocks{static_cast<uint32>(ceildiv(nrows, amp_block_size)),
                           static_cast<uint32>(num_rhs), 1u};
        mc_fgs_amp<d_i_val_type, d_m_val_type, d_o_val_type, IndexType>
            <<<nblocks, amp_block_size, 0, exec->get_stream()>>>(
                bin_strides, bin_max_nnzs, acol_idxs, avalues, row_begin,
                row_end, b_stride, b_vals, x_stride, x_vals,
                stop_status->get_data(), first_iter);
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_AMP_KERNEL);


constexpr int warps_per_block = default_block_size / config::warp_size;

template <int subwarp_size, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void mc_fgs_csr(
    const IndexType* const __restrict__ row_ptrs,
    const IndexType* const __restrict__ col_idxs,
    const MatrixValueType* const __restrict__ values, const IndexType begin_row,
    const IndexType end_row, const size_type b_stride,
    const InputValueType* const __restrict__ b, const size_type x_stride,
    OutputValueType* const __restrict__ x,
    stopping_status* const __restrict__ stopstatus, const bool first_iter)
{
    using highest_type = gko::highest_precision<InputValueType, MatrixValueType,
                                                OutputValueType>;
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto lane = subwarp.thread_rank();
    const auto subwarp_id =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    const auto num_subwarps = thread::get_subwarp_num_flat<subwarp_size>();
    const auto irhs = blockIdx.y;
    if (first_iter && subwarp_id == 0 && lane == 0) {
        stopstatus[irhs].reset();
    }

    for (auto row = begin_row + subwarp_id; row < end_row;
         row += num_subwarps) {
        const auto row_start = row_ptrs[row];
        const auto row_end_nnz = row_ptrs[row + 1];
        auto partial_sum = zero<highest_type>();
        auto diag = zero<MatrixValueType>();

        for (auto k = row_start + lane; k < row_end_nnz; k += subwarp_size) {
            const auto col = col_idxs[k];
            const auto val = values[k];
            if (col == row) {
                diag = val;
            } else {
                partial_sum -=
                    static_cast<highest_type>(val) *
                    static_cast<highest_type>(x[col * x_stride + irhs]);
            }
        }

        auto sum = reduce(subwarp, partial_sum,
                          [](highest_type a, highest_type b) { return a + b; });
        auto subwarp_diag =
            reduce(subwarp, diag,
                   [](MatrixValueType a, MatrixValueType b) { return a + b; });

        if (lane == 0) {
            sum += static_cast<highest_type>(b[row * b_stride + irhs]);
            if (subwarp_diag != zero<MatrixValueType>()) {
                x[row * x_stride + irhs] = static_cast<OutputValueType>(
                    sum / static_cast<highest_type>(subwarp_diag));
            }
        }
    }
}


template <int subwarp_size, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void mc_fgs_csr(syn::value_list<int, subwarp_size>,
                std::shared_ptr<const DefaultExecutor> exec,
                const std::vector<IndexType>& color_ptrs,
                const matrix::Csr<MatrixValueType, IndexType>* const a,
                const matrix::Dense<InputValueType>* const b,
                matrix::Dense<OutputValueType>* const x, const bool first_iter,
                array<stopping_status>* const stop_status)
{
    if (color_ptrs.size() < 2) {
        return;
    }

    using d_m_val_type = typename gkerd::device_type<MatrixValueType>;
    using d_i_val_type = typename gkerd::device_type<InputValueType>;
    using d_o_val_type = typename gkerd::device_type<OutputValueType>;
    const auto num_colors = static_cast<int>(color_ptrs.size() - 1);
    const auto num_rhs = b->get_size()[1];
    const auto row_ptrs_dev = a->get_const_row_ptrs();
    const auto col_idxs_dev = a->get_const_col_idxs();
    const auto values_dev = as_device_type(a->get_const_values());
    const auto x_vals = as_device_type(x->get_values());
    const auto b_vals = as_device_type(b->get_const_values());
    const auto x_stride = x->get_stride();
    const auto b_stride = b->get_stride();

    const auto nwarps = static_cast<uint32>(exec->get_num_warps_per_sm() *
                                            exec->get_num_multiprocessor()) *
                        classical_oversubscription;

    for (int color = 0; color < num_colors; ++color) {
        const auto row_begin = color_ptrs[color];
        const auto row_end = color_ptrs[color + 1];
        const auto nrows = row_end - row_begin;
        const auto num_blocks =
            std::min(static_cast<uint32>(
                         ceildiv(nrows, default_block_size / subwarp_size)),
                     nwarps / warps_per_block);
        // One warp per row: warps_per_block rows per block
        const dim3 nblocks{num_blocks, static_cast<uint32>(num_rhs), 1u};
        mc_fgs_csr<subwarp_size, d_i_val_type, d_m_val_type, d_o_val_type,
                   IndexType>
            <<<nblocks, default_block_size, 0, exec->get_stream()>>>(
                row_ptrs_dev, col_idxs_dev, values_dev, row_begin, row_end,
                b_stride, b_vals, x_stride, x_vals, stop_status->get_data(),
                first_iter);
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_mc_fgs_csr, mc_fgs_csr);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void multicolor_fgs_csr(std::shared_ptr<const DefaultExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::Csr<MatrixValueType, IndexType>* const a,
                        const matrix::Dense<InputValueType>* const b,
                        matrix::Dense<OutputValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    IndexType max_length_per_row = 0;
    using Tcsr = matrix::Csr<MatrixValueType, IndexType>;
    if (auto strategy =
            std::dynamic_pointer_cast<const typename Tcsr::classical>(
                a->get_strategy())) {
        max_length_per_row = strategy->get_max_length_per_row();
    } else if (auto strategy =
                   std::dynamic_pointer_cast<const typename Tcsr::automatical>(
                       a->get_strategy())) {
        max_length_per_row = strategy->get_max_length_per_row();
    } else {
        // as a fall-back: use average row length, at least 1
        max_length_per_row = a->get_num_stored_elements() /
                             std::max<size_type>(a->get_size()[0], 1);
    }
    max_length_per_row = std::max<size_type>(max_length_per_row, 1);
    select_mc_fgs_csr(
        classical_kernels(),
        [&max_length_per_row](int compiled_len) {
            return max_length_per_row >= compiled_len;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, color_ptrs, a, b, x,
        first_iter, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_CSR_KERNEL);


template <typename ValueType, typename IndexType>
static inline IndexType eff_max_row_len(
    const matrix::AMP<ValueType, IndexType>* const a)
{
    constexpr int q = matrix::AMP<ValueType, IndexType>::num_precisions;
    auto maxnr = zero<IndexType>();
    for (int k = 0; k < q; k++) {
        auto binmax = a->get_max_nnz_per_row_for_bin(k);
        maxnr += binmax;
    }
    maxnr /= q;
    return maxnr;
}


template <int subwarp_size, typename IValueType, typename MValueType,
          typename OValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void mc_fgs_amp_csr(
    const amp::precision_array<const IndexType*, MValueType> bin_row_ptrs,
    const amp::precision_array<const IndexType*, MValueType> bin_col_idxs,
    const ScalarDCPtrTuple<MValueType> bin_values, const IndexType begin_row,
    const IndexType end_row, const size_type b_stride,
    const IValueType* const __restrict__ b, const size_type x_stride,
    OValueType* const __restrict__ x,
    stopping_status* const __restrict__ stopstatus, const bool first_iter)
{
    constexpr int q = amp::narrow_types<MValueType>::num_types;
    using highest_type =
        gko::highest_precision<IValueType, MValueType, OValueType>;
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto lane = subwarp.thread_rank();
    const auto subwarp_id =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    const auto num_subwarps =
        thread::get_subwarp_num_flat<subwarp_size, IndexType>();

    const auto irhs = blockIdx.y;
    if (first_iter && subwarp_id == 0 && lane == 0) {
        stopstatus[irhs].reset();
    }

    for (auto row = begin_row + subwarp_id; row < end_row;
         row += num_subwarps) {
        auto partial_sum = zero<highest_type>();
        auto diag = zero<MValueType>();

        gko::constexpr_for<0, q, 1>([&](auto k) {
            const auto row_start = bin_row_ptrs[k][row];
            const auto row_end_nnz = bin_row_ptrs[k][row + 1];
            auto avals = std::get<k>(bin_values);
            auto acols = bin_col_idxs[k];
            for (auto j = row_start + lane; j < row_end_nnz;
                 j += subwarp_size) {
                const auto col = acols[j];
                const auto val = avals[j];
                if (col == row) {
                    diag = static_cast<MValueType>(val);
                } else {
                    partial_sum -=
                        static_cast<highest_type>(val) *
                        static_cast<highest_type>(x[col * x_stride + irhs]);
                }
            }
        });

        auto sum = reduce(subwarp, partial_sum,
                          [](highest_type a, highest_type b) { return a + b; });
        auto subwarp_diag = reduce(
            subwarp, diag, [](MValueType a, MValueType b) { return a + b; });

        if (lane == 0) {
            sum += static_cast<highest_type>(b[row * b_stride + irhs]);
            if (subwarp_diag != zero<MValueType>()) {
                x[row * x_stride + irhs] = static_cast<OValueType>(
                    sum / static_cast<highest_type>(subwarp_diag));
            }
        }
    }
}


template <int subwarp_size, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void mc_fgs_amp_csr(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const std::vector<IndexType>& color_ptrs,
                    const matrix::AMP<MatrixValueType, IndexType>* const a,
                    const matrix::Dense<InputValueType>* const b,
                    matrix::Dense<OutputValueType>* const x,
                    const bool first_iter,
                    array<stopping_status>* const stop_status)
{
    if (color_ptrs.size() < 2) {
        return;
    }

    using d_m_val_type = typename gkerd::device_type<MatrixValueType>;
    using d_i_val_type = typename gkerd::device_type<InputValueType>;
    using d_o_val_type = typename gkerd::device_type<OutputValueType>;
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    const auto num_colors = static_cast<int>(color_ptrs.size() - 1);
    const auto num_rhs = b->get_size()[1];
    const auto x_vals = as_device_type(x->get_values());
    const auto b_vals = as_device_type(b->get_const_values());
    const auto x_stride = x->get_stride();
    const auto b_stride = b->get_stride();

    // Get precision buckets' arrays
    ScalarDCPtrTuple<d_m_val_type> avalues;
    amp::precision_array<const IndexType*, d_m_val_type> acol_idxs;
    amp::precision_array<const IndexType*, d_m_val_type> arow_ptrs;
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        using CsrType = matrix::Csr<value_type, IndexType>;
        auto cmatk = dynamic_cast<const CsrType*>(a->get_bin_matrix(k));
        if (!cmatk) {
            GKO_NOT_SUPPORTED(cmatk);
        }
        acol_idxs[k] = cmatk->get_const_col_idxs();
        arow_ptrs[k] = cmatk->get_const_row_ptrs();
        std::get<k>(avalues) = as_device_type(cmatk->get_const_values());
    });

    const auto nwarps = static_cast<uint32>(exec->get_num_warps_per_sm() *
                                            exec->get_num_multiprocessor()) *
                        classical_oversubscription;

    for (int color = 0; color < num_colors; ++color) {
        const auto row_begin = color_ptrs[color];
        const auto row_end = color_ptrs[color + 1];
        const auto nrows = row_end - row_begin;
        const auto num_blocks =
            std::min(static_cast<uint32>(
                         ceildiv(nrows, default_block_size / subwarp_size)),
                     nwarps / warps_per_block);
        const dim3 nblocks{num_blocks, static_cast<uint32>(num_rhs), 1u};
        mc_fgs_amp_csr<subwarp_size, d_i_val_type, d_m_val_type, d_o_val_type,
                       IndexType>
            <<<nblocks, default_block_size, 0, exec->get_stream()>>>(
                arow_ptrs, acol_idxs, avalues, row_begin, row_end, b_stride,
                b_vals, x_stride, x_vals, stop_status->get_data(), first_iter);
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_mc_fgs_amp_csr, mc_fgs_amp_csr);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void multicolor_fgs_amp_csr(
    std::shared_ptr<const DefaultExecutor> exec,
    const std::vector<IndexType>& color_ptrs,
    const matrix::AMP<MatrixValueType, IndexType>* const a,
    const matrix::Dense<InputValueType>* const b,
    matrix::Dense<OutputValueType>* const x, const bool first_iter,
    array<stopping_status>* const stop_status)
{
    const int sel_row_len = std::max(eff_max_row_len(a), one<IndexType>());
    select_mc_fgs_amp_csr(
        classical_kernels(),
        [&sel_row_len](int compiled_row_len) {
            return sel_row_len >= compiled_row_len;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, color_ptrs, a, b, x,
        first_iter, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_AMP_CSR_KERNEL);


}  // namespace gssdl
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
