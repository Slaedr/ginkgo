/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/solver/gmres_mixed_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The GMRES_MIXED solver namespace.
 *
 * @ingroup gmres_mixed
 */
namespace gmres_mixed {


constexpr int default_block_size = 512;
// default_dot_dim can not be 64 in hip because 64 * 64 exceeds their max block
// size limit.
constexpr int default_dot_dim = 32;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


constexpr int default_update_dim = 32;


#include "common/solver/gmres_mixed_kernels.hpp.inc"


// Specialization, so the Accessor can use the same function as regular pointers
template <typename Type1, typename Type2>
GKO_INLINE accessor::ReducedStorage3d<cuda_type<Type1>, cuda_type<Type2>>
as_cuda_accessor(accessor::ReducedStorage3d<Type1, Type2> &acc)
{
    return {as_cuda_type(acc.get_storage()), acc.get_size(), acc.get_stride0(),
            acc.get_stride1()};
}

template <typename Type1, typename Type2>
GKO_INLINE accessor::ScaledReducedStorage3d<cuda_type<Type1>, cuda_type<Type2>>
as_cuda_accessor(accessor::ScaledReducedStorage3d<Type1, Type2> &acc)
{
    return {as_cuda_type(acc.get_storage()), acc.get_size(), acc.get_stride0(),
            acc.get_stride1(), as_cuda_type(acc.get_scale())};
}


template <typename ValueType>
void zero_matrix(size_type m, size_type n, size_type stride, ValueType *array)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);
    zero_matrix_kernel<<<grid_size, block_size, 0, 0>>>(m, n, stride,
                                                        as_cuda_type(array));
}


template <typename ValueType>
void initialize_1(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<remove_complex<ValueType>> *b_norm,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<stopping_status> *stop_status, size_type krylov_dim)
{
    const auto num_threads = std::max(b->get_size()[0] * b->get_stride(),
                                      krylov_dim * b->get_size()[1]);
    const dim3 grid_dim(ceildiv(num_threads, default_block_size), 1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    b->compute_norm2(b_norm);
    initialize_1_kernel<block_size><<<grid_dim, block_dim>>>(
        b->get_size()[0], b->get_size()[1], krylov_dim,
        as_cuda_type(b->get_const_values()), b->get_stride(),
        as_cuda_type(residual->get_values()), residual->get_stride(),
        as_cuda_type(givens_sin->get_values()), givens_sin->get_stride(),
        as_cuda_type(givens_cos->get_values()), givens_cos->get_stride(),
        as_cuda_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_GMRES_MIXED_INITIALIZE_1_KERNEL);


template <typename ValueType, typename Accessor3d>
void initialize_2(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<remove_complex<ValueType>> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<remove_complex<ValueType>> *arnoldi_norm,
                  Accessor3d krylov_bases,
                  matrix::Dense<ValueType> *next_krylov_basis,
                  Array<size_type> *final_iter_nums, size_type krylov_dim)
{
    const auto num_rows = residual->get_size()[0];
    const auto num_rhs = residual->get_size()[1];
    const dim3 grid_dim_1(ceildiv((krylov_dim + 1) * krylov_bases.get_stride0(),
                                  default_block_size),
                          1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    initialize_2_1_kernel<block_size><<<grid_dim_1, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1], krylov_dim,
        as_cuda_accessor(krylov_bases),
        as_cuda_type(residual_norm_collection->get_values()),
        residual_norm_collection->get_stride());
    residual->compute_norm2(residual_norm);

    components::fill_array(exec, arnoldi_norm->get_values() + 2 * num_rhs,
                           num_rhs, zero<remove_complex<ValueType>>());
    const dim3 grid_size_nrm(ceildiv(num_rhs, default_dot_dim),
                             exec->get_num_multiprocessor() * 2);
    const dim3 block_size_nrm(default_dot_dim, default_dot_dim);
    multinorminf_kernel_without_stop<<<grid_size_nrm, block_size_nrm>>>(
        num_rows, num_rhs, as_cuda_type(residual->get_const_values()), num_rhs,
        as_cuda_type(arnoldi_norm->get_values() + 2 * num_rhs), 0);

    const auto stride_arnoldi = arnoldi_norm->get_stride();
    set_scale_kernel<default_block_size>
        <<<ceildiv(num_rhs * (krylov_dim + 1), default_block_size),
           default_block_size>>>(
            num_rhs, krylov_dim + 1,
            as_cuda_type(residual_norm->get_const_values()), num_rhs,
            as_cuda_type(arnoldi_norm->get_const_values() + 2 * stride_arnoldi),
            stride_arnoldi, as_cuda_accessor(krylov_bases));

    const dim3 grid_dim_2(
        ceildiv(num_rows * krylov_bases.get_stride1(), default_block_size), 1,
        1);
    initialize_2_2_kernel<block_size><<<grid_dim_2, block_dim>>>(
        residual->get_size()[0], residual->get_size()[1],
        as_cuda_type(residual->get_const_values()), residual->get_stride(),
        as_cuda_type(residual_norm->get_const_values()),
        as_cuda_type(residual_norm_collection->get_values()),
        as_cuda_accessor(krylov_bases),
        as_cuda_type(next_krylov_basis->get_values()),
        next_krylov_basis->get_stride(),
        as_cuda_type(final_iter_nums->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_GMRES_MIXED_TYPE(
    GKO_DECLARE_GMRES_MIXED_INITIALIZE_2_KERNEL);


template <typename ValueType, typename Accessor3dim>
void finish_arnoldi_CGS2(std::shared_ptr<const CudaExecutor> exec,
                         matrix::Dense<ValueType> *next_krylov_basis,
                         Accessor3dim krylov_bases,
                         matrix::Dense<ValueType> *hessenberg_iter,
                         matrix::Dense<ValueType> *buffer_iter,
                         matrix::Dense<remove_complex<ValueType>> *arnoldi_norm,
                         size_type iter, const stopping_status *stop_status,
                         stopping_status *reorth_status,
                         Array<size_type> *num_reorth, int *num_reorth_steps,
                         int *num_reorth_vectors)
{
    using non_complex = remove_complex<ValueType>;
    // optimization parameter
    constexpr int singledot_block_size = default_dot_dim;
    constexpr bool use_scale =
        kernels::detail::is_3d_scaled_accessor<Accessor3dim>::value;
    const auto stride_next_krylov = next_krylov_basis->get_stride();
    const auto stride_hessenberg = hessenberg_iter->get_stride();
    const auto stride_buffer = buffer_iter->get_stride();
    const auto stride_arnoldi = arnoldi_norm->get_stride();
    const auto dim_size = next_krylov_basis->get_size();
    const dim3 grid_size(ceildiv(dim_size[1], default_dot_dim),
                         exec->get_num_multiprocessor() * 2);
    const dim3 grid_size_num_iters(
        ceildiv(dim_size[1] * (iter + 1), default_dot_dim),
        exec->get_num_multiprocessor() * 2);
    const dim3 grid_size_num_iters_2(ceildiv(dim_size[1], default_dot_dim),
                                     exec->get_num_multiprocessor() * 2,
                                     iter + 1);
    const dim3 block_size(default_dot_dim, default_dot_dim);
    // Note: having iter first (instead of row_idx information) is likely
    //       beneficial for avoiding atomic_add conflicts, but that needs
    //       further investigation.
    const dim3 grid_size_iters_single(exec->get_num_multiprocessor() * 2,
                                      iter + 1);
    const dim3 block_size_iters_single(singledot_block_size);
    size_type numReorth;

    components::fill_array(exec, arnoldi_norm->get_values(), dim_size[1],
                           zero<non_complex>());
    multinorm2_kernel<<<grid_size, block_size>>>(
        dim_size[0], dim_size[1],
        as_cuda_type(next_krylov_basis->get_const_values()), stride_next_krylov,
        as_cuda_type(arnoldi_norm->get_values()), as_cuda_type(stop_status));
    // nrmP = norm(next_krylov_basis
    zero_matrix(iter + 1, dim_size[1], stride_hessenberg,
                hessenberg_iter->get_values());
    if (dim_size[1] > 1) {
        multidot_kernel_num_iters_2<default_dot_dim>
            <<<grid_size_num_iters_2, block_size>>>(
                iter + 1, dim_size[0], dim_size[1],
                as_cuda_type(next_krylov_basis->get_const_values()),
                stride_next_krylov, as_cuda_accessor(krylov_bases),
                as_cuda_type(hessenberg_iter->get_values()), stride_hessenberg,
                as_cuda_type(stop_status));
    } else {
        singledot_kernel_num_iters_2<singledot_block_size>
            <<<grid_size_iters_single, block_size_iters_single>>>(
                iter + 1, dim_size[0],
                as_cuda_type(next_krylov_basis->get_const_values()),
                stride_next_krylov, as_cuda_accessor(krylov_bases),
                as_cuda_type(hessenberg_iter->get_values()), stride_hessenberg,
                as_cuda_type(stop_status));
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    // end
    update_next_krylov_kernel_num_iters<default_block_size>
        <<<ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
           default_block_size>>>(
            iter + 1, dim_size[0], dim_size[1],
            as_cuda_type(next_krylov_basis->get_values()), stride_next_krylov,
            as_cuda_accessor(krylov_bases),
            as_cuda_type(hessenberg_iter->get_const_values()),
            stride_hessenberg, as_cuda_type(stop_status));

    // for i in 1:iter
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end
    components::fill_array(exec, arnoldi_norm->get_values() + dim_size[1],
                           dim_size[1], zero<non_complex>());
    components::fill_array(exec, arnoldi_norm->get_values() + 2 * dim_size[1],
                           dim_size[1], zero<non_complex>());
    multinorm2_inf_kernel<use_scale><<<grid_size, block_size>>>(
        dim_size[0], dim_size[1],
        as_cuda_type(next_krylov_basis->get_const_values()), stride_next_krylov,
        as_cuda_type(arnoldi_norm->get_values() + dim_size[1]),
        as_cuda_type(arnoldi_norm->get_values() + 2 * dim_size[1]),
        as_cuda_type(stop_status));
    // nrmN = norm(next_krylov_basis)
    components::fill_array(exec, num_reorth->get_data(), 1, zero<size_type>());
    check_arnoldi_norms_new<default_block_size>
        <<<ceildiv(dim_size[1], default_block_size), default_block_size>>>(
            as_cuda_type(arnoldi_norm->get_values()), stride_arnoldi,
            as_cuda_type(hessenberg_iter->get_values()), stride_hessenberg,
            iter + 1, as_cuda_accessor(krylov_bases), as_cuda_type(stop_status),
            as_cuda_type(reorth_status), as_cuda_type(num_reorth->get_data()));
    numReorth = 0;
    exec->get_master()->copy_from(exec.get(), 1, num_reorth->get_const_data(),
                                  &numReorth);
    // numReorth <= number of next_krylov vector to be reorthogonalization
    for (size_type l = 1; (numReorth > 0) && (l < 3); l++) {
        (*num_reorth_steps)++;
        (*num_reorth_vectors) += iter;
        zero_matrix(iter + 1, dim_size[1], stride_buffer,
                    buffer_iter->get_values());
        if (dim_size[1] > 1) {
            multidot_kernel_num_iters_2<default_dot_dim>
                <<<grid_size_num_iters_2, block_size>>>(
                    iter + 1, dim_size[0], dim_size[1],
                    as_cuda_type(next_krylov_basis->get_const_values()),
                    stride_next_krylov, as_cuda_accessor(krylov_bases),
                    as_cuda_type(buffer_iter->get_values()), stride_buffer,
                    as_cuda_type(stop_status));
        } else {
            singledot_kernel_num_iters_2<singledot_block_size>
                <<<grid_size_iters_single, block_size_iters_single>>>(
                    iter + 1, dim_size[0],
                    as_cuda_type(next_krylov_basis->get_const_values()),
                    stride_next_krylov, as_cuda_accessor(krylov_bases),
                    as_cuda_type(buffer_iter->get_values()), stride_buffer,
                    as_cuda_type(stop_status));
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        // end
        //*
        update_next_krylov_kernel_num_iters_and_add<default_block_size>
            <<<ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
               default_block_size>>>(
                iter + 1, dim_size[0], dim_size[1],
                as_cuda_type(next_krylov_basis->get_values()),
                stride_next_krylov, as_cuda_accessor(krylov_bases),
                as_cuda_type(hessenberg_iter->get_values()), stride_hessenberg,
                as_cuda_type(buffer_iter->get_const_values()), stride_buffer,
                as_cuda_type(stop_status), as_cuda_type(reorth_status));
        // for i in 1:iter
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end
        components::fill_array(exec, arnoldi_norm->get_values() + dim_size[1],
                               dim_size[1], zero<non_complex>());
        components::fill_array(exec,
                               arnoldi_norm->get_values() + 2 * dim_size[1],
                               dim_size[1], zero<non_complex>());
        multinorm2_inf_kernel<use_scale><<<grid_size, block_size>>>(
            dim_size[0], dim_size[1],
            as_cuda_type(next_krylov_basis->get_const_values()),
            stride_next_krylov,
            as_cuda_type(arnoldi_norm->get_values() + dim_size[1]),
            as_cuda_type(arnoldi_norm->get_values() + 2 * dim_size[1]),
            as_cuda_type(stop_status));
        // nrmN = norm(next_krylov_basis)
        components::fill_array(exec, num_reorth->get_data(), 1,
                               zero<size_type>());
        check_arnoldi_norms_new<default_block_size>
            <<<ceildiv(dim_size[1], default_block_size), default_block_size>>>(
                as_cuda_type(arnoldi_norm->get_values()), stride_arnoldi,
                as_cuda_type(hessenberg_iter->get_values()), stride_hessenberg,
                iter + 1, as_cuda_accessor(krylov_bases),
                as_cuda_type(stop_status), as_cuda_type(reorth_status),
                as_cuda_type(num_reorth->get_data()));
        numReorth = 0;
        exec->get_master()->copy_from(exec.get(), 1,
                                      num_reorth->get_const_data(), &numReorth);
        // numReorth <= number of next_krylov vector to be reorthogonalization
    }

    update_krylov_next_krylov_kernel<default_block_size>
        <<<ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
           default_block_size>>>(
            iter, dim_size[0], dim_size[1],
            as_cuda_type(next_krylov_basis->get_values()), stride_next_krylov,
            as_cuda_accessor(krylov_bases),
            as_cuda_type(hessenberg_iter->get_const_values()),
            stride_hessenberg, as_cuda_type(stop_status));
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // krylov_bases(:, iter + 1) = next_krylov_basis
    // End of arnoldi
}

template <typename ValueType>
void givens_rotation(std::shared_ptr<const CudaExecutor> exec,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter,
                     matrix::Dense<remove_complex<ValueType>> *residual_norm,
                     matrix::Dense<ValueType> *residual_norm_collection,
                     const matrix::Dense<remove_complex<ValueType>> *b_norm,
                     size_type iter, const Array<stopping_status> *stop_status)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_cols = hessenberg_iter->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{
        static_cast<unsigned int>(ceildiv(num_cols, block_size)), 1, 1};

    givens_rotation_kernel<block_size><<<grid_dim, block_dim>>>(
        hessenberg_iter->get_size()[0], hessenberg_iter->get_size()[1], iter,
        as_cuda_type(hessenberg_iter->get_values()),
        hessenberg_iter->get_stride(), as_cuda_type(givens_sin->get_values()),
        givens_sin->get_stride(), as_cuda_type(givens_cos->get_values()),
        givens_cos->get_stride(), as_cuda_type(residual_norm->get_values()),
        as_cuda_type(residual_norm_collection->get_values()),
        residual_norm_collection->get_stride(),
        as_cuda_type(b_norm->get_const_values()),
        as_cuda_type(stop_status->get_const_data()));
}


template <typename ValueType, typename Accessor3d>
void step_1(std::shared_ptr<const CudaExecutor> exec,
            matrix::Dense<ValueType> *next_krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<remove_complex<ValueType>> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            Accessor3d krylov_bases, matrix::Dense<ValueType> *hessenberg_iter,
            matrix::Dense<ValueType> *buffer_iter,
            const matrix::Dense<remove_complex<ValueType>> *b_norm,
            matrix::Dense<remove_complex<ValueType>> *arnoldi_norm,
            size_type iter, Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status,
            Array<stopping_status> *reorth_status, Array<size_type> *num_reorth,
            int *num_reorth_steps, int *num_reorth_vectors)
{
    increase_final_iteration_numbers_kernel<<<
        static_cast<unsigned int>(
            ceildiv(final_iter_nums->get_num_elems(), default_block_size)),
        default_block_size>>>(as_cuda_type(final_iter_nums->get_data()),
                              as_cuda_type(stop_status->get_const_data()),
                              final_iter_nums->get_num_elems());
    finish_arnoldi_CGS2(exec, next_krylov_basis, krylov_bases, hessenberg_iter,
                        buffer_iter, arnoldi_norm, iter,
                        stop_status->get_const_data(),
                        reorth_status->get_data(), num_reorth, num_reorth_steps,
                        num_reorth_vectors);
    givens_rotation(exec, givens_sin, givens_cos, hessenberg_iter,
                    residual_norm, residual_norm_collection, b_norm, iter,
                    stop_status);
}

GKO_INSTANTIATE_FOR_EACH_GMRES_MIXED_TYPE(
    GKO_DECLARE_GMRES_MIXED_STEP_1_KERNEL);


template <typename ValueType>
void solve_upper_triangular(
    const matrix::Dense<ValueType> *residual_norm_collection,
    const matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
    const Array<size_type> *final_iter_nums)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_rhs = residual_norm_collection->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{static_cast<unsigned int>(ceildiv(num_rhs, block_size)),
                        1, 1};

    solve_upper_triangular_kernel<block_size><<<grid_dim, block_dim>>>(
        hessenberg->get_size()[1], num_rhs,
        as_cuda_type(residual_norm_collection->get_const_values()),
        residual_norm_collection->get_stride(),
        as_cuda_type(hessenberg->get_const_values()), hessenberg->get_stride(),
        as_cuda_type(y->get_values()), y->get_stride(),
        as_cuda_type(final_iter_nums->get_const_data()));
}


template <typename ValueType, typename ConstAccessor3d>
void calculate_qy(ConstAccessor3d krylov_bases, size_type num_krylov_bases,
                  const matrix::Dense<ValueType> *y,
                  matrix::Dense<ValueType> *before_preconditioner,
                  const Array<size_type> *final_iter_nums)
{
    const auto num_rows = before_preconditioner->get_size()[0];
    const auto num_cols = before_preconditioner->get_size()[1];
    const auto stride_before_preconditioner =
        before_preconditioner->get_stride();

    constexpr auto block_size = default_block_size;
    const dim3 grid_dim{
        static_cast<unsigned int>(
            ceildiv(num_rows * stride_before_preconditioner, block_size)),
        1, 1};
    const dim3 block_dim{block_size, 1, 1};


    calculate_Qy_kernel<block_size><<<grid_dim, block_dim>>>(
        num_rows, num_cols, as_cuda_accessor(krylov_bases),
        as_cuda_type(y->get_const_values()), y->get_stride(),
        as_cuda_type(before_preconditioner->get_values()),
        stride_before_preconditioner,
        as_cuda_type(final_iter_nums->get_const_data()));
    // Calculate qy
    // before_preconditioner = krylov_bases * y
}


template <typename ValueType, typename ConstAccessor3d>
void step_2(std::shared_ptr<const CudaExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            ConstAccessor3d krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *before_preconditioner,
            const Array<size_type> *final_iter_nums)
{
    // since hessenberg has dims:  iters x iters * num_rhs
    // krylov_bases has dims:  (iters + 1) x sysmtx[0] x num_rhs
    const auto iters =
        hessenberg->get_size()[1] / before_preconditioner->get_size()[1];
    const auto num_krylov_bases = iters + 1;
    solve_upper_triangular(residual_norm_collection, hessenberg, y,
                           final_iter_nums);
    calculate_qy(krylov_bases, num_krylov_bases, y, before_preconditioner,
                 final_iter_nums);
}

GKO_INSTANTIATE_FOR_EACH_GMRES_MIXED_CONST_TYPE(
    GKO_DECLARE_GMRES_MIXED_STEP_2_KERNEL);


}  // namespace gmres_mixed
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
