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


namespace {


template <typename BatchMatrixType, typename ValueType>
__device__ __forceinline__ void initialize(
    const size_type batch_id, const BatchMatrixType &a,
    const batch_dense::BatchEntryConfig mv_conf, const ValueType *const b_entry,
    const ValueType *const x_entry, ValueType *const rho_old_entry,
    ValueType *const omega_entry, ValueType *const alpha_entry,
    ValueType *const x_copy, ValueType *const r_entry,
    ValueType *const r_hat_entry, ValueType *const p_entry,
    ValueType *const v_entry,
    typename gko::remove_complex<ValueType> *const rhs_norms_entry,
    typename gko::remove_complex<ValueType> *const res_norms_entry)
{
    for (int c = threadIdx.x; c < mv_conf.ncols; c += blockDim.x) {
        rho_old_entry[c] = one<ValueType>();
        omega_entry[c] = one<ValueType>();
        alpha_entry[c] = one<ValueType>();
    }
    // Compute norms of rhs
    compute_norm2<ValueType>({mv_conf.stride, 1, mv_conf.ncols}, b_entry,
                             rhs_norms_entry);
    // copy x from global to shared memory
    copy(mv_conf, x_entry, x_copy);
    // r = b
    copy(mv_conf, b_entry, r_entry);
    __syncthreads();
    // r = b - A*x
    advanced_spmv_kernel(static_cast<ValueType>(-1.0), a, batch_id, mv_conf,
                         x_copy, static_cast<ValueType>(1.0), r_entry);
    __syncthreads();
    compute_norm2<ValueType>(mv_conf, r_entry, res_norms_entry);
    copy(mv_conf, r_entry, r_hat_entry);
    for (int li = threadIdx.x; li < mv_conf.nrows * mv_conf.ncols;
         li += blockDim.x) {
        const int r = li / mv_conf.ncols;
        const int c = li % mv_conf.ncols;
        p_entry[r * mv_conf.stride + c] = zero<ValueType>();
        v_entry[r * mv_conf.stride + c] = zero<ValueType>();
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_p(
    const gko::batch_dense::BatchEntryConfig &conf,
    const ValueType *const rho_new, const ValueType *const rho_old,
    const ValueType *const alpha, const ValueType *const omega,
    const ValueType *const r, const ValueType *const v, ValueType *const p,
    const uint32 &converged)
{
    for (int li = threadIdx.x; li < conf.nrows * conf.ncols; li += blockDim.x) {
        const int row = li / conf.ncols;
        const int col = li % conf.ncols;
        const uint32 conv = converged & (1 << col);
        if (conv) {
            continue;
        }
        const ValueType beta =
            (rho_new[col] / rho_old[col]) * (alpha[col] / omega[col]);
        p[row * conf.stride + col] =
            r[row * conf.stride + col] +
            beta * (p[row * conf.stride + col] -
                    omega[col] * v[row * conf.stride + col]);
    }
}

template <typename ValueType>
__device__ __forceinline__ void compute_alpha(
    const gko::batch_dense::BatchEntryConfig &mv_conf,
    const ValueType *const rho_new, const ValueType *const r_hat,
    const ValueType *const v, ValueType *const alpha, const uint32 &converged)
{
    compute_dot_product<ValueType>(mv_conf, r_hat, v, alpha, converged);
    __syncthreads();

    for (int c = threadIdx.x; c < mv_conf.ncols; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);
        if (conv) {
            continue;
        }
        alpha[c] = rho_new[c] / alpha[c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_s(
    const gko::batch_dense::BatchEntryConfig &mv_conf, const ValueType *const r,
    const ValueType *const alpha, const ValueType *const v, ValueType *const s,
    const uint32 &converged)
{
    for (int li = threadIdx.x; li < mv_conf.nrows * mv_conf.ncols;
         li += blockDim.x) {
        const int row = li / mv_conf.ncols;
        const int col = li % mv_conf.ncols;
        const uint32 conv = converged & (1 << col);
        if (conv) {
            continue;
        }
        s[row * mv_conf.stride + col] =
            r[row * mv_conf.stride + col] -
            alpha[col] * v[row * mv_conf.stride + col];
    }
}

template <typename ValueType>
__device__ __forceinline__ void compute_omega(
    const gko::batch_dense::BatchEntryConfig &mv_conf, const ValueType *const t,
    const ValueType *const s, ValueType *const temp, ValueType *const omega,
    const uint32 &converged)
{
    compute_dot_product<ValueType>(mv_conf, t, s, omega, converged);
    compute_dot_product<ValueType>(mv_conf, t, t, temp, converged);
    __syncthreads();
    for (int col = threadIdx.x; col < mv_conf.ncols; col += blockDim.x) {
        const uint32 conv = converged & (1 << col);
        if (conv) {
            continue;
        }
        omega[col] /= temp[col];
    }
}

template <typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    const gko::batch_dense::BatchEntryConfig &mv_conf,
    const ValueType *const p_hat, const ValueType *const s_hat,
    const ValueType *const alpha, const ValueType *const omega,
    const ValueType *const s, const ValueType *const t, ValueType *const x,
    ValueType *const r, const uint32 &converged)
{
    for (int li = threadIdx.x; li < mv_conf.nrows * mv_conf.ncols;
         li += blockDim.x) {
        const int row = li / mv_conf.ncols;
        const int col = li % mv_conf.ncols;
        const uint32 conv = converged & (1 << col);
        if (conv) {
            continue;
        }

        x[row * mv_conf.stride + col] =
            x[row * mv_conf.stride + col] +
            alpha[col] * p_hat[row * mv_conf.stride + col] +
            omega[col] * s_hat[row * mv_conf.stride + col];

        r[row * mv_conf.stride + col] =
            s[row * mv_conf.stride + col] -
            omega[col] * t[row * mv_conf.stride + col];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x_middle(
    const gko::batch_dense::BatchEntryConfig &conf,
    const ValueType *const alpha, const ValueType *const p_hat,
    ValueType *const x, const uint32 &converged_recent)
{
    for (int li = threadIdx.x; li < conf.nrows * conf.ncols; li += blockDim.x) {
        const int row = li / conf.ncols;
        const int col = li % conf.ncols;

        const uint32 conv = converged_recent & (1 << col);

        if (conv) {
            x[row * conf.stride + col] =
                x[row * conf.stride + col] +
                alpha[col] * p_hat[row * conf.stride + col];
        }
    }
}


}  // namespace

// unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(const int max_iter,
                             const gko::remove_complex<ValueType> tol,
                             LogType logger, PrecType prec_shared,
                             const BatchMatrixType a,
                             const ValueType *const __restrict__ left_scale,
                             const ValueType *const __restrict__ right_scale,
                             const gko::batch_dense::BatchEntryConfig mv_conf,
                             ValueType *const __restrict__ b,
                             ValueType *const __restrict__ x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = mv_conf.ncols;
    const auto norm_conf =
        gko::batch_dense::BatchEntryConfig{mv_conf.stride, 1, mv_conf.ncols};

    constexpr int max_nrhs = batch_config<ValueType>::max_num_rhs;
    assert(max_rhs >= nrhs);

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
#if GKO_CUDA_BATCH_USE_DYNAMIC_SHARED_MEM
        extern __shared__ char local_mem_sh[];
        ValueType *const r_sh = reinterpret_cast<ValueType *>(local_mem_sh);
        ValueType *const r_hat_sh = r_sh + nrows * nrhs;
        ValueType *const p_sh = r_hat_sh + nrows * nrhs;
        ValueType *const p_hat_sh = p_sh + nrows * nrhs;
        ValueType *const v_sh = p_hat_sh + nrows * nrhs;
        ValueType *const s_sh = v_sh + nrows * nrhs;
        ValueType *const s_hat_sh = s_sh + nrows * nrhs;
        ValueType *const t_sh = s_hat_sh + nrows * nrhs;
        ValueType *const x_sh = t_sh + nrows * nrhs;
        ValueType *const prec_work_sh = x_sh + nrows * nrhs;
        ValueType *const rho_old_sh =
            prec_work_sh + PrecType::dynamic_work_size(nrows, a.num_nnz);
        ValueType *const rho_new_sh = rho_old_sh + nrhs;
        ValueType *const omega_sh = rho_new_sh + nrhs;
        ValueType *const alpha_sh = omega_sh + nrhs;
        ValueType *const temp_sh = alpha_sh + nrhs;
        // real_type *const norms_rhs_sh =
        //    reinterpret_cast<real_type *>(temp_sh + nrhs);
        // real_type *const norms_res_sh = norms_rhs_sh + nrhs;
#else
        constexpr int max_nrows = batch_config<ValueType>::max_num_rows;
        if (max_nrows * max_nrhs < nrows * nrhs) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                printf(" Static vector size not enough!\n");
            }
            return;
        }
        __shared__ ValueType r_sh[max_nrows * max_nrhs];
        __shared__ ValueType r_hat_sh[max_nrows * max_nrhs];
        __shared__ ValueType p_sh[max_nrows * max_nrhs];
        __shared__ ValueType p_hat_sh[max_nrows * max_nrhs];
        __shared__ ValueType v_sh[max_nrows * max_nrhs];
        __shared__ ValueType s_sh[max_nrows * max_nrhs];
        __shared__ ValueType s_hat_sh[max_nrows * max_nrhs];
        __shared__ ValueType t_sh[max_nrows * max_nrhs];
        __shared__ ValueType x_sh[max_nrows * max_nrhs];
        __shared__ ValueType prec_work_sh[PrecType::work_size + 2];
        __shared__ ValueType rho_old_sh[max_nrhs];
        __shared__ ValueType rho_new_sh[max_nrhs];
        __shared__ ValueType omega_sh[max_nrhs];
        __shared__ ValueType alpha_sh[max_nrhs];
        __shared__ ValueType temp_sh[max_nrhs];
        __shared__ real_type norms_rhs_sh[max_nrhs];
        __shared__ real_type norms_res_sh[max_nrhs];
#endif
        __shared__ real_type norms_rhs_sh[max_nrhs];
        __shared__ real_type norms_res_sh[max_nrhs];

        uint32 converged = 0;

        if (left_scale) {
            batch_scale(left_scale + ibatch * nrows,
                        right_scale + ibatch * nrows, a, ibatch);
            const auto b_gl_entry =
                gko::batch::batch_pointer(b, ibatch, mv_conf);
            batch_scale(left_scale + ibatch * nrows, b_gl_entry, mv_conf);
            __syncthreads();
        }

        const auto b_gl_entry = gko::batch::batch_pointer(
            static_cast<const ValueType *>(b), ibatch, mv_conf);
        const auto x_gl_entry = gko::batch::batch_pointer(x, ibatch, mv_conf);

        // generate preconditioner
        prec_shared.generate(gko::batch::to_const(a), ibatch, prec_work_sh);

        // initialization
        // rho_old = 1, omega = 1, alpha = 1
        // compute b norms
        // copy x from global to shared memory
        // r = b - A*x
        // compute residual norms
        // r_hat = r
        // p = 0
        // v = 0
        initialize(ibatch, gko::batch::to_const(a), mv_conf, b_gl_entry,
                   x_gl_entry, rho_old_sh, omega_sh, alpha_sh, x_sh, r_sh,
                   r_hat_sh, p_sh, v_sh, norms_rhs_sh, norms_res_sh);
        __syncthreads();

        // stopping criterion object
        StopType stop(nrhs, max_iter, tol, norms_rhs_sh, converged);

        int iter = -1;

        while (1) {
            ++iter;

            bool all_converged = stop.check_converged(iter, norms_res_sh, NULL,
                                                      mv_conf, converged);
            logger.log_iteration(ibatch, iter, norms_res_sh, converged);
            if (all_converged) {
                break;
            }

            // rho_new =  < r_hat , r > = (r_hat)' * (r)
            compute_dot_product<ValueType>(mv_conf, r_hat_sh, r_sh, rho_new_sh,
                                           converged);
            __syncthreads();

            // beta = (rho_new / rho_old)*(alpha / omega)
            // p = r + beta*(p - omega * v)
            update_p(mv_conf, rho_new_sh, rho_old_sh, alpha_sh, omega_sh, r_sh,
                     v_sh, p_sh, converged);
            __syncthreads();

            // p_hat = precond * p
            prec_shared.apply(mv_conf, p_sh, p_hat_sh);
            __syncthreads();

            // v = A * p_hat
            spmv_kernel(gko::batch::to_const(a), ibatch, mv_conf, p_hat_sh,
                        v_sh);
            __syncthreads();

            // alpha = rho_new / < r_hat , v>
            compute_alpha(mv_conf, rho_new_sh, r_hat_sh, v_sh, alpha_sh,
                          converged);
            __syncthreads();

            // s = r - alpha*v
            update_s(mv_conf, r_sh, alpha_sh, v_sh, s_sh, converged);
            __syncthreads();

            // an estimate of residual norms
            compute_norm2<ValueType>(mv_conf, s_sh, norms_res_sh, converged);
            __syncthreads();

            const uint32 converged_prev = converged;

            all_converged = stop.check_converged(iter, norms_res_sh, NULL,
                                                 mv_conf, converged);

            // update x for the sytems (rhs) which converge at this point...  x
            // = x + alpha*p_hat
            // note bits could change from 0 to 1, not the other way round, so
            // we can use xor to get info about recent convergence...
            const uint32 converged_recent = converged_prev ^ converged;

            update_x_middle(mv_conf, alpha_sh, p_hat_sh, x_sh,
                            converged_recent);
            __syncthreads();

            logger.log_iteration(ibatch, iter, norms_res_sh, converged);

            if (all_converged) {
                break;
            }

            // s_hat = precond * s
            prec_shared.apply(mv_conf, s_sh, s_hat_sh);
            __syncthreads();

            // t = A * s_hat
            spmv_kernel(gko::batch::to_const(a), ibatch, mv_conf, s_hat_sh,
                        t_sh);
            __syncthreads();

            // omega = <t,s> / <t,t>
            compute_omega(mv_conf, t_sh, s_sh, temp_sh, omega_sh, converged);
            __syncthreads();

            // x = x + alpha*p_hat + omega *s_hat
            // r = s - omega * t
            update_x_and_r(mv_conf, p_hat_sh, s_hat_sh, alpha_sh, omega_sh,
                           s_sh, t_sh, x_sh, r_sh, converged);
            __syncthreads();

            compute_norm2<ValueType>(mv_conf, r_sh, norms_res_sh, converged);
            __syncthreads();

            // rho_old = rho_new
            copy(norm_conf, rho_new_sh, rho_old_sh, converged);
            __syncthreads();
        }

        if (left_scale) {
            batch_scale(right_scale + ibatch * nrows, x_sh, mv_conf);
        }
        __syncthreads();

        // copy x back to global memory
        copy(mv_conf, x_sh, x_gl_entry);
        __syncthreads();
    }
}
