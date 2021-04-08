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


/**
 *  Identity preconditioner for batch solvers. ( To be able to have
 * unpreconditioned solves )
 */
template <typename ValueType>
class BatchIdentity final {
public:
    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int work_size = 1;

    /**
     * The size of the work vector required in case of dynamic allocation.
     *
     * For the Identity preconditioner, this is unnecessary, but this function
     * is part of a 'batch preconditioner interface' because other
     * preconditioners may need it.
     */
    static constexpr int dynamic_work_size(int, int) { return 0; }

    /**
     * Sets the input and generates the identity preconditioner.(Nothing needs
     * to be actually generated.)
     *
     * @param mat  Matrix for which to build an Ideniity preconditioner.
     * @param work  A 'work-vector', which is unneecessary here as no
     * preconditioner values are to be stored.
     */
    GKO_ATTRIBUTES GKO_INLINE
    BatchIdentity(const gko::batch_csr::BatchEntry<const ValueType> &mat,
                  ValueType *const __restrict__ work)
        : matrix_{mat}, work_{work}
    {}

    __device__ __forceinline__ void apply(
        const gko::batch_dense::BatchEntry<const ValueType> &r,
        const gko::batch_dense::BatchEntry<ValueType> &z) const
    {
        for (int li = threadIdx.x; li < r.num_rows * r.num_rhs;
             li += blockDim.x) {
            const int row = li / r.num_rhs;

            const int col = li % r.num_rhs;

            z.values[row * z.stride + col] = r.values[row * r.stride + col];
        }
    }

private:
    ValueType *const __restrict__ work_;
    const gko::batch_csr::BatchEntry<const ValueType> &matrix_;
};