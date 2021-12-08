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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {


/**
 * Exact ILU(0) factorization for batch solvers.
 */
template <typename ValueType, typename TriSolver>
class BatchIlu0 final {
public:
    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    static int dynamic_work_size(const int nrows, const int nnz) { return nnz; }

    /**
     * Generates the ILU(0) preconditioner in the supplied work vector.
     *
     * @param mat  Matrix for which to build a Jacobi preconditioner.
     * @param work  A 'work-vector', used here to store the both the lower and
     *              upper triangular factors. It must be allocated with at
     *              least the amount of memory given by dynamic_work_size.
     *
     * @note Assumes each row is sorted by column index.
     */
    void generate(const gko::batch_csr::BatchEntry<const ValueType>& mat,
                  void* const work)
    {
        work_ = reinterpret_cast<ValueType*>(work);
        for (int i = 0; i < mat.num_nnz; i++) {
            work_[i] = mat.values[i];
        }
        for (int i = 0; i < mat.num_rows; i++) {
            for (int j = mat.row_ptrs[i]; j < mat.row_ptrs[i + 1]; j++) {
                ValueType diag_val = zero<ValueType>();
                const auto col = mat.col_idxs[j];
                auto sum = zero<ValueType>();
                for (int k = mat.row_ptrs[i]; k < j; k++) {
                    const auto col_k = mat.col_idxs[k];
                    // Quit if reached diagonal (while computing U entries)
                    if (col_k == i) {
                        diag_val = work_[k];
                        break;
                    }
                    if (col_k > i) {
                        break;
                    }
                    bool found = false;
                    for (int l = mat.row_ptrs[col_k];
                         l < mat.row_ptrs[col_k + 1]; l++) {
                        if (mat.col_idxs[l] == col) {
                            sum += work_[k] * work_[l];
                            break;
                        }
                    }
                }
                if (col < i) {
                    // L
                    assert(diag_val != zero<ValueType>());
                    work_[j] = (mat.values[j] - sum) / diag_val;
                } else {
                    // U
                    work_[j] = mat.values[j] - sum;
                }
            }
        }

        static_assert(TriSolver::is_batch_tri_solve,
                      "Need a batched triangular solver!");
        TriSolver::generate(mat, work_);
    }

    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
    {
        TriSolver::apply(r, z);
    }

private:
    ValueType* work_ = nullptr;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_JACOBI_HPP_
