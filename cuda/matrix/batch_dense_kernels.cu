/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/batch_dense_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/prefix_sum.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The BatchDense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {


// constexpr auto default_block_size = 512;


template <typename ValueType>
void simple_apply(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::BatchDense<ValueType> *a,
                  const matrix::BatchDense<ValueType> *b,
                  matrix::BatchDense<ValueType> *c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (cublas::is_supported<ValueType>::value) {
//        auto handle = exec->get_cublas_handle();
//        {
//            cublas::pointer_mode_guard pm_guard(handle);
//            auto alpha = one<ValueType>();
//            auto beta = zero<ValueType>();
//            cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c->get_size()[1],
//                         c->get_size()[0], a->get_size()[1], &alpha,
//                         b->get_const_values(), b->get_stride(),
//                         a->get_const_values(), a->get_stride(), &beta,
//                         c->get_values(), c->get_stride());
//        }
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const matrix::BatchDense<ValueType> *alpha,
           const matrix::BatchDense<ValueType> *a,
           const matrix::BatchDense<ValueType> *b,
           const matrix::BatchDense<ValueType> *beta,
           matrix::BatchDense<ValueType> *c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (cublas::is_supported<ValueType>::value) {
//        cublas::gemm(exec->get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
//                     c->get_size()[1], c->get_size()[0], a->get_size()[1],
//                     alpha->get_const_values(), b->get_const_values(),
//                     b->get_stride(), a->get_const_values(), a->get_stride(),
//                     beta->get_const_values(), c->get_values(),
//                     c->get_stride());
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const CudaExecutor> exec,
           const matrix::BatchDense<ValueType> *alpha,
           matrix::BatchDense<ValueType> *x) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (cublas::is_supported<ValueType>::value && x->get_size()[1] == 1) {
//        cublas::scal(exec->get_cublas_handle(), x->get_size()[0],
//                     alpha->get_const_values(), x->get_values(),
//                     x->get_stride());
//    } else {
//        // TODO: tune this parameter
//        constexpr auto block_size = default_block_size;
//        const dim3 grid_dim =
//            ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
//        const dim3 block_dim{config::warp_size, 1,
//                             block_size / config::warp_size};
//        kernel::scale<block_size><<<grid_dim, block_dim>>>(
//            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
//            as_cuda_type(alpha->get_const_values()),
//            as_cuda_type(x->get_values()), x->get_stride());
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const CudaExecutor> exec,
                const matrix::BatchDense<ValueType> *alpha,
                const matrix::BatchDense<ValueType> *x,
                matrix::BatchDense<ValueType> *y) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (cublas::is_supported<ValueType>::value && x->get_size()[1] == 1) {
//        cublas::axpy(exec->get_cublas_handle(), x->get_size()[0],
//                     alpha->get_const_values(), x->get_const_values(),
//                     x->get_stride(), y->get_values(), y->get_stride());
//    } else {
//        // TODO: tune this parameter
//        constexpr auto block_size = default_block_size;
//        const dim3 grid_dim =
//            ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
//        const dim3 block_dim{config::warp_size, 1,
//                             block_size / config::warp_size};
//        kernel::add_scaled<block_size><<<grid_dim, block_dim>>>(
//            x->get_size()[0], x->get_size()[1], alpha->get_size()[1],
//            as_cuda_type(alpha->get_const_values()),
//            as_cuda_type(x->get_const_values()), x->get_stride(),
//            as_cuda_type(y->get_values()), y->get_stride());
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const CudaExecutor> exec,
                     const matrix::BatchDense<ValueType> *alpha,
                     const matrix::Diagonal<ValueType> *x,
                     matrix::BatchDense<ValueType> *y) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    const auto size = y->get_size()[0];
//    const auto grid_dim = ceildiv(size, default_block_size);
//
//    kernel::add_scaled_diag<<<grid_dim, default_block_size>>>(
//        size, as_cuda_type(alpha->get_const_values()),
//        as_cuda_type(x->get_const_values()), as_cuda_type(y->get_values()),
//        y->get_stride());
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const CudaExecutor> exec,
                 const matrix::BatchDense<ValueType> *x,
                 const matrix::BatchDense<ValueType> *y,
                 matrix::BatchDense<ValueType> *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (cublas::is_supported<ValueType>::value) {
//        // TODO: write a custom kernel which does this more efficiently
//        for (size_type col = 0; col < x->get_size()[1]; ++col) {
//            cublas::dot(exec->get_cublas_handle(), x->get_size()[0],
//                        x->get_const_values() + col, x->get_stride(),
//                        y->get_const_values() + col, y->get_stride(),
//                        result->get_values() + col);
//        }
//    } else {
//        // TODO: these are tuning parameters obtained experimentally, once
//        // we decide how to handle this uniformly, they should be modified
//        // appropriately
//        constexpr auto work_per_thread = 32;
//        constexpr auto block_size = 1024;
//
//        constexpr auto work_per_block = work_per_thread * block_size;
//        const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
//        const dim3 block_dim{config::warp_size, 1,
//                             block_size / config::warp_size};
//        Array<ValueType> work(exec, grid_dim.x);
//        // TODO: write a kernel which does this more efficiently
//        for (size_type col = 0; col < x->get_size()[1]; ++col) {
//            kernel::compute_partial_dot<block_size><<<grid_dim, block_dim>>>(
//                x->get_size()[0], as_cuda_type(x->get_const_values() + col),
//                x->get_stride(), as_cuda_type(y->get_const_values() + col),
//                y->get_stride(), as_cuda_type(work.get_data()));
//            kernel::finalize_dot_computation<block_size><<<1, block_dim>>>(
//                grid_dim.x, as_cuda_type(work.get_const_data()),
//                as_cuda_type(result->get_values() + col));
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::BatchDense<ValueType> *x,
                   matrix::BatchDense<remove_complex<ValueType>> *result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (cublas::is_supported<ValueType>::value) {
//        for (size_type col = 0; col < x->get_size()[1]; ++col) {
//            cublas::norm2(exec->get_cublas_handle(), x->get_size()[0],
//                          x->get_const_values() + col, x->get_stride(),
//                          result->get_values() + col);
//        }
//    } else {
//        using norm_type = remove_complex<ValueType>;
//        // TODO: these are tuning parameters obtained experimentally, once
//        // we decide how to handle this uniformly, they should be modified
//        // appropriately
//        constexpr auto work_per_thread = 32;
//        constexpr auto block_size = 1024;
//
//        constexpr auto work_per_block = work_per_thread * block_size;
//        const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
//        const dim3 block_dim{config::warp_size, 1,
//                             block_size / config::warp_size};
//        Array<norm_type> work(exec, grid_dim.x);
//        // TODO: write a kernel which does this more efficiently
//        for (size_type col = 0; col < x->get_size()[1]; ++col) {
//            kernel::compute_partial_norm2<block_size><<<grid_dim,
//            block_dim>>>(
//                x->get_size()[0], as_cuda_type(x->get_const_values() + col),
//                x->get_stride(), as_cuda_type(work.get_data()));
//            kernel::finalize_norm2_computation<block_size><<<1, block_dim>>>(
//                grid_dim.x, as_cuda_type(work.get_const_data()),
//                as_cuda_type(result->get_values() + col));
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_batch_csr(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::BatchDense<ValueType> *source,
                          matrix::BatchCsr<ValueType, IndexType> *other)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERT_TO_BATCH_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::BatchDense<ValueType> *source,
                    size_type *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    const auto num_rows = source->get_size()[0];
//    auto nnz_per_row = Array<size_type>(exec, num_rows);
//
//    calculate_nonzeros_per_row(exec, source, &nnz_per_row);
//
//    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const CudaExecutor> exec,
                               const matrix::BatchDense<ValueType> *source,
                               size_type *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    const auto num_rows = source->get_size()[0];
//    auto nnz_per_row = Array<size_type>(exec, num_rows);
//
//    calculate_nonzeros_per_row(exec, source, &nnz_per_row);
//
//    const auto n = ceildiv(num_rows, default_block_size);
//    const size_type grid_dim =
//        (n <= default_block_size) ? n : default_block_size;
//
//    auto block_results = Array<size_type>(exec, grid_dim);
//
//    kernel::reduce_max_nnz<<<grid_dim, default_block_size,
//                             default_block_size * sizeof(size_type)>>>(
//        num_rows, as_cuda_type(nnz_per_row.get_const_data()),
//        as_cuda_type(block_results.get_data()));
//
//    auto d_result = Array<size_type>(exec, 1);
//
//    kernel::reduce_max_nnz<<<1, default_block_size,
//                             default_block_size * sizeof(size_type)>>>(
//        grid_dim, as_cuda_type(block_results.get_const_data()),
//        as_cuda_type(d_result.get_data()));
//
//    *result = exec->copy_val_to_host(d_result.get_const_data());
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const CudaExecutor> exec,
                                const matrix::BatchDense<ValueType> *source,
                                Array<size_type> *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    const dim3 block_size(default_block_size, 1, 1);
//    auto rows_per_block = ceildiv(default_block_size, config::warp_size);
//    const size_t grid_x = ceildiv(source->get_size()[0], rows_per_block);
//    const dim3 grid_size(grid_x, 1, 1);
//    if (grid_x > 0) {
//        kernel::count_nnz_per_row<<<grid_size, block_size>>>(
//            source->get_size()[0], source->get_size()[1],
//            source->get_stride(), as_cuda_type(source->get_const_values()),
//            as_cuda_type(result->get_data()));
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::BatchDense<ValueType> *source,
                          size_type *result, size_type *stride_factor,
                          size_type *slice_size) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    const auto num_rows = source->get_size()[0];
//
//    if (num_rows == 0) {
//        *result = 0;
//        return;
//    }
//
//    const auto num_cols = source->get_size()[1];
//    const auto slice_num = ceildiv(num_rows, slice_size);
//
//    auto nnz_per_row = Array<size_type>(exec, num_rows);
//
//    calculate_nonzeros_per_row(exec, source, &nnz_per_row);
//
//    auto max_nnz_per_slice = Array<size_type>(exec, slice_num);
//
//    auto grid_dim = ceildiv(slice_num * config::warp_size,
//    default_block_size);
//
//    kernel::reduce_max_nnz_per_slice<<<grid_dim, default_block_size>>>(
//        num_rows, slice_size, stride_factor,
//        as_cuda_type(nnz_per_row.get_const_data()),
//        as_cuda_type(max_nnz_per_slice.get_data()));
//
//    grid_dim = ceildiv(slice_num, default_block_size);
//    auto block_results = Array<size_type>(exec, grid_dim);
//
//    kernel::reduce_total_cols<<<grid_dim, default_block_size,
//                                default_block_size * sizeof(size_type)>>>(
//        slice_num, as_cuda_type(max_nnz_per_slice.get_const_data()),
//        as_cuda_type(block_results.get_data()));
//
//    auto d_result = Array<size_type>(exec, 1);
//
//    kernel::reduce_total_cols<<<1, default_block_size,
//                                default_block_size * sizeof(size_type)>>>(
//        grid_dim, as_cuda_type(block_results.get_const_data()),
//        as_cuda_type(d_result.get_data()));
//
//    *result = exec->copy_val_to_host(d_result.get_const_data());
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               const matrix::BatchDense<ValueType> *orig,
               matrix::BatchDense<ValueType> *trans) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (cublas::is_supported<ValueType>::value) {
//        auto handle = exec->get_cublas_handle();
//        {
//            cublas::pointer_mode_guard pm_guard(handle);
//            auto alpha = one<ValueType>();
//            auto beta = zero<ValueType>();
//            cublas::geam(
//                handle, CUBLAS_OP_T, CUBLAS_OP_N, orig->get_size()[0],
//                orig->get_size()[1], &alpha, orig->get_const_values(),
//                orig->get_stride(), &beta, static_cast<ValueType *>(nullptr),
//                trans->get_size()[1], trans->get_values(),
//                trans->get_stride());
//        }
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::BatchDense<ValueType> *orig,
                    matrix::BatchDense<ValueType> *trans) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (cublas::is_supported<ValueType>::value) {
//        auto handle = exec->get_cublas_handle();
//        {
//            cublas::pointer_mode_guard pm_guard(handle);
//            auto alpha = one<ValueType>();
//            auto beta = zero<ValueType>();
//            cublas::geam(
//                handle, CUBLAS_OP_C, CUBLAS_OP_N, orig->get_size()[0],
//                orig->get_size()[1], &alpha, orig->get_const_values(),
//                orig->get_stride(), &beta, static_cast<ValueType *>(nullptr),
//                trans->get_size()[1], trans->get_values(),
//                trans->get_stride());
//        }
//    } else {
//        GKO_NOT_IMPLEMENTED;
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);


}  // namespace batch_dense
}  // namespace cuda
}  // namespace kernels
}  // namespace gko