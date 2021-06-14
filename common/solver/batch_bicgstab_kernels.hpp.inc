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
    const gko::batch_dense::BatchEntry<const ValueType> &t_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &temp_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_shared_entry,
    const uint32 &converged)
{
    compute_dot_product<ValueType>(t_shared_entry, s_shared_entry,
                                   omega_shared_entry, converged);
    compute_dot_product<ValueType>(t_shared_entry, t_shared_entry,
                                   temp_shared_entry, converged);

    __syncthreads();

    for (int c = threadIdx.x; c < omega_shared_entry.num_rhs; c += blockDim.x) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        omega_shared_entry.values[c] /= temp_shared_entry.values[c];
    }
}

template <typename ValueType>
__device__ __forceinline__ void update_x_and_r(
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_hat_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &t_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_shared_entry,
    const uint32 &converged)
{
    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / x_shared_entry.num_rhs;
        const int c = li % x_shared_entry.num_rhs;

        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        const ValueType omega = omega_shared_entry.values[c];

        x_shared_entry.values[r * x_shared_entry.stride + c] =
            x_shared_entry.values[r * x_shared_entry.stride + c] +
            alpha_shared_entry.values[c] *
                p_hat_shared_entry.values[r * p_hat_shared_entry.stride + c] +
            omega *
                s_hat_shared_entry.values[r * s_hat_shared_entry.stride + c];

        r_shared_entry.values[r * r_shared_entry.stride + c] =
            s_shared_entry.values[r * s_shared_entry.stride + c] -
            omega * t_shared_entry.values[r * t_shared_entry.stride + c];
    }
}


template <typename ValueType>
__device__ __forceinline__ void update_x_middle(

    const gko::batch_dense::BatchEntry<const ValueType> &alpha_shared_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_shared_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_shared_entry,
    const uint32 &converged_recent)
{
    for (int li = threadIdx.x;
         li < x_shared_entry.num_rows * x_shared_entry.num_rhs;
         li += blockDim.x) {
        const int r = li / x_shared_entry.num_rhs;
        const int c = li % x_shared_entry.num_rhs;

        const uint32 conv = converged_recent & (1 << c);

        if (conv) {
            x_shared_entry.values[r * x_shared_entry.stride + c] =
                x_shared_entry.values[r * x_shared_entry.stride + c] +
                alpha_shared_entry.values[c] *
                    p_hat_shared_entry
                        .values[r * p_hat_shared_entry.stride + c];
        }
    }
}


}  // namespace

// unnamed namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> tol,
    LogType logger, PrecType prec_shared, const BatchMatrixType a,
    const gko::batch_dense::UniformBatch<const ValueType> left,
    const gko::batch_dense::UniformBatch<const ValueType> right,
    const gko::batch_dense::UniformBatch<ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;
    const auto norm_conf =
        gko::batch_dense::BatchEntryConfig{b.stride, 1, nrhs};
    const auto mv_conf =
        gko::batch_dense::BatchEntryConfig{b.stride, nrows, nrhs};
    // required for static allocation in stopping criterion
    assert(batch_config<ValueType>::max_num_rhs >= nrhs);

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
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
        real_type *const norms_rhs_sh =
            reinterpret_cast<real_type *>(temp_sh + nrhs);
        real_type *const norms_res_sh = norms_rhs_sh + nrhs;

        uint32 converged = 0;

        const gko::batch_dense::BatchEntry<const ValueType> left_global_entry =
            gko::batch::batch_entry(left, ibatch);

        const gko::batch_dense::BatchEntry<const ValueType> right_global_entry =
            gko::batch::batch_entry(right, ibatch);

        if (left_global_entry.values) {
            const typename BatchMatrixType::entry_type A_global_entry =
                gko::batch::batch_entry(a, ibatch);
            const gko::batch_dense::BatchEntry<ValueType> b_global_entry =
                gko::batch::batch_entry(b, ibatch);
            batch_scale(left.values + ibatch * nrows,
                        right.values + ibatch * nrows, a, ibatch);
            batch_scale(left_global_entry.values, b_global_entry.values,
                        mv_conf);
            __syncthreads();
        }

        const auto A_global_entry =
            gko::batch::batch_entry(gko::batch::to_const(a), ibatch);

        const gko::batch_dense::BatchEntry<const ValueType> b_global_entry =
            gko::batch::batch_entry(gko::batch::to_const(b), ibatch);

        const gko::batch_dense::BatchEntry<ValueType> x_global_entry =
            gko::batch::batch_entry(x, ibatch);


        const gko::batch_dense::BatchEntry<ValueType> x_shared_entry{
            x_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> r_shared_entry{
            r_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> r_hat_shared_entry{
            r_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> p_shared_entry{
        //    p_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> p_hat_shared_entry{
            p_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> v_shared_entry{
        //    v_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> s_shared_entry{
            s_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> s_hat_shared_entry{
            s_hat_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> t_shared_entry{
            t_sh, static_cast<size_type>(nrhs), nrows, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> rho_old_shared_entry{
        //    rho_old_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<ValueType> rho_new_shared_entry{
        //    rho_new_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> omega_shared_entry{
            omega_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> alpha_shared_entry{
            alpha_sh, static_cast<size_type>(nrhs), 1, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> temp_shared_entry{
            temp_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<real_type> rhs_norms_shared_entry{
        //    norms_rhs_sh, static_cast<size_type>(nrhs), 1, nrhs};

        // const gko::batch_dense::BatchEntry<real_type> res_norms_shared_entry{
        //    norms_res_sh, static_cast<size_type>(nrhs), 1, nrhs};

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
        initialize(ibatch, gko::batch::to_const(a), mv_conf,
                   b_global_entry.values, x_global_entry.values, rho_old_sh,
                   omega_sh, alpha_sh, x_sh, r_sh, r_hat_sh, p_sh, v_sh,
                   norms_rhs_sh, norms_res_sh);
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

            update_x_middle(gko::batch::to_const(alpha_shared_entry),
                            gko::batch::to_const(p_hat_shared_entry),
                            x_shared_entry, converged_recent);
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
            compute_omega(gko::batch::to_const(t_shared_entry),
                          gko::batch::to_const(s_shared_entry),
                          temp_shared_entry, omega_shared_entry, converged);
            __syncthreads();


            // x = x + alpha*p_hat + omega *s_hat
            // r = s - omega * t
            update_x_and_r(gko::batch::to_const(p_hat_shared_entry),
                           gko::batch::to_const(s_hat_shared_entry),
                           gko::batch::to_const(alpha_shared_entry),
                           gko::batch::to_const(omega_shared_entry),
                           gko::batch::to_const(s_shared_entry),
                           gko::batch::to_const(t_shared_entry), x_shared_entry,
                           r_shared_entry, converged);
            __syncthreads();

            compute_norm2<ValueType>(mv_conf, r_sh, norms_res_sh, converged);
            __syncthreads();

            // rho_old = rho_new
            copy(norm_conf, rho_new_sh, rho_old_sh, converged);
            __syncthreads();
        }

        if (left_global_entry.values) {
            batch_scale(right.values + ibatch * nrows, x_sh, mv_conf);
        }
        __syncthreads();

        // copy x back to global memory
        copy(mv_conf, x_sh, x_global_entry.values);
        __syncthreads();
    }
}
