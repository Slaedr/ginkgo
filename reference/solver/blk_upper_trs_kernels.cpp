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

#include "core/solver/blk_upper_trs_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "reference/components/fixed_block_operations.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The block upper tri-solve namespace.
 *
 * @ingroup blk_upper_trs
 */
namespace blk_upper_trs {


void should_perform_transpose(std::shared_ptr<const ReferenceExecutor>,
                              bool &do_transpose)
{
    do_transpose = false;
}


void init_struct(std::shared_ptr<const ReferenceExecutor> exec,
                 std::shared_ptr<solver::SolveStruct> &solve_struct)
{
    // This init kernel is here to allow initialization of the solve struct for
    // a more sophisticated implementation as for other executors.
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const ReferenceExecutor>,
              const matrix::Fbcsr<ValueType, IndexType> *,
              solver::SolveStruct *, const gko::size_type)
{
    // This generate kernel is here to allow for a more sophisticated
    // implementation as for other executors. This kernel would perform the
    // "analysis" phase for the triangular matrix.
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLK_UPPER_TRS_GENERATE_KERNEL);


template <int mat_blk_sz, typename ValueType, typename IndexType>
static void solve_kernel(const IndexType nbrows, const int nrhs,
                         const IndexType *const row_ptrs,
                         const IndexType *const col_idxs,
                         const ValueType *const values,
                         const matrix::Dense<ValueType> *const b,
                         matrix::Dense<ValueType> *const x)
{
    constexpr int bs2 = mat_blk_sz * mat_blk_sz;
    using Blk_t = blockutils::FixedBlock<ValueType, mat_blk_sz, mat_blk_sz>;
    const auto mvals = reinterpret_cast<const Blk_t *>(values);
    for (IndexType brow = nbrows - 1; brow >= 0; --brow) {
        for (IndexType j = 0; j < nrhs; ++j) {
            ValueType sum[mat_blk_sz];
            for (int ib = 0; ib < mat_blk_sz; ib++) {
                sum[ib] = b->at(brow * mat_blk_sz + ib, j);
            }

            for (auto k = row_ptrs[brow]; k < row_ptrs[brow + 1]; ++k) {
                const auto bcol = col_idxs[k];
                if (bcol > brow) {
                    for (int jb = 0; jb < mat_blk_sz; jb++) {
                        for (int ib = 0; ib < mat_blk_sz; ib++) {
                            sum[ib] -= mvals[k](ib, jb) *
                                       x->at(bcol * mat_blk_sz + jb, j);
                        }
                    }
                }
            }

            Blk_t invD = mvals[row_ptrs[brow]];
            const bool invflag = invert_block_complete(invD);
            if (!invflag) {
                printf(" Could not invert upper diag block at blk row %ld!",
                       static_cast<long int>(brow));
            }

            for (int ib = 0; ib < mat_blk_sz; ib++) {
                x->at(brow * mat_blk_sz + ib, j) = 0;
            }
            for (int jb = 0; jb < mat_blk_sz; jb++) {
                for (int ib = 0; ib < mat_blk_sz; ib++) {
                    x->at(brow * mat_blk_sz + ib, j) += invD(ib, jb) * sum[jb];
                }
            }
        }
    }
}

/**
 * The parameters trans_x and trans_b are used only in the CUDA executor for
 * versions <=9.1 due to a limitation in the cssrsm_solve algorithm and hence
 * here essentially unused.
 */
template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Fbcsr<ValueType, IndexType> *const matrix,
           const solver::SolveStruct *const solve_struct,
           matrix::Dense<ValueType> *, matrix::Dense<ValueType> *,
           const matrix::Dense<ValueType> *const b,
           matrix::Dense<ValueType> *const x)
{
    auto row_ptrs = matrix->get_const_row_ptrs();
    auto col_idxs = matrix->get_const_col_idxs();
    auto vals = matrix->get_const_values();
    const int bs = matrix->get_block_size();
    const IndexType nbrows = matrix->get_num_block_rows();
    const auto nrhs = static_cast<int>(b->get_size()[1]);

    if (bs == 2) {
        solve_kernel<2>(nbrows, nrhs, row_ptrs, col_idxs, vals, b, x);
    } else if (bs == 3) {
        solve_kernel<3>(nbrows, nrhs, row_ptrs, col_idxs, vals, b, x);
    } else if (bs == 4) {
        solve_kernel<4>(nbrows, nrhs, row_ptrs, col_idxs, vals, b, x);
    } else if (bs == 7) {
        solve_kernel<7>(nbrows, nrhs, row_ptrs, col_idxs, vals, b, x);
    } else
        GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLK_UPPER_TRS_SOLVE_KERNEL);


}  // namespace blk_upper_trs
}  // namespace reference
}  // namespace kernels
}  // namespace gko
