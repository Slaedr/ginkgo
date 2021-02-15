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

#ifndef GKO_CUDA_SOLVER_COMMON_BLK_TRS_KERNELS_CUH_
#define GKO_CUDA_SOLVER_COMMON_BLK_TRS_KERNELS_CUH_


#include <functional>
#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cusparse_block_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace solver {


struct SolveStruct {
    virtual void dummy() {}
};


namespace cuda {


struct SolveStruct : gko::solver::SolveStruct {
    bsrsm2Info_t solve_info;
    cusparseSolvePolicy_t policy;
    cusparseMatDescr_t factor_descr;
    size_t factor_work_size;
    void *factor_work_vec;
    SolveStruct()
    {
        factor_work_vec = nullptr;
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&factor_descr));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatIndexBase(factor_descr, CUSPARSE_INDEX_BASE_ZERO));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatType(factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatDiagType(factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateBsrsm2Info(&solve_info));
        policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    }

    SolveStruct(const SolveStruct &) = delete;

    SolveStruct(SolveStruct &&) = delete;

    SolveStruct &operator=(const SolveStruct &) = delete;

    SolveStruct &operator=(SolveStruct &&) = delete;

    ~SolveStruct()
    {
        cusparseDestroyMatDescr(factor_descr);
        if (solve_info) {
            cusparseDestroyBsrsm2Info(solve_info);
        }
        if (factor_work_vec != nullptr) {
            cudaFree(factor_work_vec);
            factor_work_vec = nullptr;
        }
    }
};


}  // namespace cuda
}  // namespace solver


namespace kernels {
namespace cuda {
namespace {


void should_perform_transpose_kernel(std::shared_ptr<const CudaExecutor> exec,
                                     bool &do_transpose)
{
    do_transpose = false;
}


void init_struct_kernel(std::shared_ptr<const CudaExecutor> exec,
                        std::shared_ptr<solver::SolveStruct> &solve_struct)
{
    solve_struct = std::make_shared<solver::cuda::SolveStruct>();
}


template <typename ValueType, typename IndexType>
void generate_kernel(std::shared_ptr<const CudaExecutor> exec,
                     const matrix::Fbcsr<ValueType, IndexType> *const matrix,
                     solver::SolveStruct *const solve_struct, const int num_rhs,
                     const bool is_upper, const bool is_unit)
{
    const int bs = matrix->get_block_size();
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        if (auto cuda_solve_struct =
                dynamic_cast<solver::cuda::SolveStruct *>(solve_struct)) {
            auto handle = exec->get_cusparse_handle();
            // auto handle =
            // static_cast<cusparseHandle_t>(exec->get_cusparse_handle());
            if (is_upper) {
                GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatFillMode(
                    cuda_solve_struct->factor_descr, CUSPARSE_FILL_MODE_UPPER));
            } else {
                GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatFillMode(
                    cuda_solve_struct->factor_descr, CUSPARSE_FILL_MODE_LOWER));
            }

            if (is_unit) {
                GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatDiagType(
                    cuda_solve_struct->factor_descr, CUSPARSE_DIAG_TYPE_UNIT));
            }

            {
                cusparse::pointer_mode_guard pm_guard(handle);
                cuda_solve_struct->factor_work_size =
                    cusparse::bsrsm2_buffer_size(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_TRANSPOSE,
                        matrix->get_num_block_rows(),
                        static_cast<IndexType>(num_rhs),
                        static_cast<IndexType>(matrix->get_num_stored_blocks()),
                        cuda_solve_struct->factor_descr,
                        const_cast<ValueType *>(matrix->get_const_values()),
                        matrix->get_const_row_ptrs(),
                        matrix->get_const_col_idxs(), bs,
                        cuda_solve_struct->solve_info);

                // allocate workspace
                if (cuda_solve_struct->factor_work_vec != nullptr) {
                    exec->free(cuda_solve_struct->factor_work_vec);
                }
                cuda_solve_struct->factor_work_vec =
                    exec->alloc<void *>(cuda_solve_struct->factor_work_size);

                cusparse::bsrsm2_analysis(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE, matrix->get_num_block_rows(),
                    static_cast<IndexType>(num_rhs),
                    static_cast<IndexType>(matrix->get_num_stored_blocks()),
                    cuda_solve_struct->factor_descr, matrix->get_const_values(),
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    bs, cuda_solve_struct->solve_info,
                    cuda_solve_struct->policy,
                    cuda_solve_struct->factor_work_vec);
            }

        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void solve_kernel(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Fbcsr<ValueType, IndexType> *const matrix,
                  const solver::SolveStruct *const solve_struct,
                  const matrix::Dense<ValueType> *const b,
                  matrix::Dense<ValueType> *const x)
{
    using vec = matrix::Dense<ValueType>;
    const int bs = matrix->get_block_size();
    const auto nrhs = static_cast<IndexType>(b->get_size()[1]);

    if (cusparse::is_supported<ValueType, IndexType>::value) {
        if (auto cuda_solve_struct =
                dynamic_cast<const solver::cuda::SolveStruct *>(solve_struct)) {
            ValueType one = 1.0;
            auto handle = exec->get_cusparse_handle();

            {
                cusparse::pointer_mode_guard pm_guard(handle);
                cusparse::bsrsm2_solve(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE, matrix->get_num_block_rows(),
                    nrhs,
                    static_cast<IndexType>(matrix->get_num_stored_blocks()),
                    &one, cuda_solve_struct->factor_descr,
                    matrix->get_const_values(), matrix->get_const_row_ptrs(),
                    matrix->get_const_col_idxs(), bs,
                    cuda_solve_struct->solve_info, b->get_const_values(),
                    b->get_stride(), x->get_values(), x->get_stride(),
                    cuda_solve_struct->policy,
                    cuda_solve_struct->factor_work_vec);
            }

        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


}  // namespace
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_
