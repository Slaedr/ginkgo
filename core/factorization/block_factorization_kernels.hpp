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

#ifndef GKO_CORE_BLOCK_FACTORIZATION_FACTORIZATION_KERNELS_HPP_
#define GKO_CORE_BLOCK_FACTORIZATION_FACTORIZATION_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_BLOCKS_KERNEL(ValueType,   \
                                                             IndexType)   \
    void add_diagonal_blocks(std::shared_ptr<const DefaultExecutor> exec, \
                             matrix::Fbcsr<ValueType, IndexType> *mtx,    \
                             bool is_sorted)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_BLU_KERNEL(ValueType, \
                                                                 IndexType) \
    void initialize_row_ptrs_BLU(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::Fbcsr<ValueType, IndexType> *system_matrix,           \
        IndexType *l_row_ptrs, IndexType *u_row_ptrs)

/** @fn initialize_BLU
 *
 * Initialize block L and block U factors with a given matrix
 * @param[in] exec  The executor to run the kernel on
 * @param[in] system_matrix  The original matrix
 * @param[in,out] l_factor  A unit block lower triangular matrix whose strictly
 *   lower block triangular part equals that of system_matrix on output
 * @param[in,out] u_factor  Upper block triangular matrix (non-transposed,
 *   in BCSR format, not BCSC) which equals the upper block triangular
 *   part of system_matrix on output
 */
#define GKO_DECLARE_FACTORIZATION_INITIALIZE_BLU_KERNEL(ValueType, IndexType) \
    void initialize_BLU(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Fbcsr<ValueType, IndexType> *system_matrix,             \
        matrix::Fbcsr<ValueType, IndexType> *l_factor,                        \
        matrix::Fbcsr<ValueType, IndexType> *u_factor)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                     \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_BLOCKS_KERNEL(ValueType,      \
                                                         IndexType);     \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_BLU_KERNEL(ValueType,  \
                                                             IndexType); \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_BLU_KERNEL(ValueType, IndexType)


namespace omp {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace omp


namespace cuda {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace cuda


namespace reference {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace reference


namespace hip {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace hip


namespace dpcpp {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_FACTORIZATION_KERNELS_HPP_