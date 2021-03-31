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

#ifndef GKO_CUDA_BASE_CUSPARSE_BLOCK_BINDINGS_HPP_
#define GKO_CUDA_BASE_CUSPARSE_BLOCK_BINDINGS_HPP_


#include "cuda/base/cusparse_bindings.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CUSPARSE namespace.
 *
 * @ingroup cusparse
 */
namespace cusparse {


/// Default storage layout within each small dense block
constexpr cusparseDirection_t blockDir = CUSPARSE_DIRECTION_ROW;


#define GKO_BIND_CUSPARSE32_BSRMV(ValueType, CusparseName)                    \
    inline void bsrmv(cusparseHandle_t handle, cusparseOperation_t transA,    \
                      int32 mb, int32 nb, int32 nnzb, const ValueType *alpha, \
                      const cusparseMatDescr_t descrA, const ValueType *valA, \
                      const int32 *rowPtrA, const int32 *colIndA,             \
                      int blocksize, const ValueType *x,                      \
                      const ValueType *beta, ValueType *y)                    \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                           \
            handle, blockDir, transA, mb, nb, nnzb, as_culibs_type(alpha),    \
            descrA, as_culibs_type(valA), rowPtrA, colIndA, blocksize,        \
            as_culibs_type(x), as_culibs_type(beta), as_culibs_type(y)));     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRMV(ValueType, CusparseName)                     \
    inline void bsrmv(cusparseHandle_t handle, cusparseOperation_t transA,     \
                      int64 mb, int64 nb, int64 nnzb, const ValueType *alpha,  \
                      const cusparseMatDescr_t descrA,                         \
                      const ValueType *csrValA, const int64 *csrRowPtrA,       \
                      const int64 *csrColIndA, int block_size,                 \
                      const ValueType *x, const ValueType *beta, ValueType *y) \
        GKO_NOT_IMPLEMENTED;                                                   \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRMV(float, cusparseSbsrmv);
GKO_BIND_CUSPARSE32_BSRMV(double, cusparseDbsrmv);
GKO_BIND_CUSPARSE32_BSRMV(std::complex<float>, cusparseCbsrmv);
GKO_BIND_CUSPARSE32_BSRMV(std::complex<double>, cusparseZbsrmv);
GKO_BIND_CUSPARSE64_BSRMV(float, cusparseSbsrmv);
GKO_BIND_CUSPARSE64_BSRMV(double, cusparseDbsrmv);
GKO_BIND_CUSPARSE64_BSRMV(std::complex<float>, cusparseCbsrmv);
GKO_BIND_CUSPARSE64_BSRMV(std::complex<double>, cusparseZbsrmv);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRMV(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRMV(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_BSRMV
#undef GKO_BIND_CUSPARSE64_BSRMV


#define GKO_BIND_CUSPARSE32_BSRMM(ValueType, CusparseName)                     \
    inline void bsrmm(cusparseHandle_t handle, cusparseOperation_t transA,     \
                      cusparseOperation_t transB, int32 mb, int32 n, int32 kb, \
                      int32 nnzb, const ValueType *alpha,                      \
                      const cusparseMatDescr_t descrA, const ValueType *valA,  \
                      const int32 *rowPtrA, const int32 *colIndA,              \
                      int block_size, const ValueType *B, int32 ldb,           \
                      const ValueType *beta, ValueType *C, int32 ldc)          \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, blockDir, transA, transB, mb, n, kb, nnzb,    \
                         as_culibs_type(alpha), descrA, as_culibs_type(valA),  \
                         rowPtrA, colIndA, block_size, as_culibs_type(B), ldb, \
                         as_culibs_type(beta), as_culibs_type(C), ldc));       \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRMM(ValueType, CusparseName)                    \
    inline void bsrmm(                                                        \
        cusparseHandle_t handle, cusparseOperation_t transA,                  \
        cusparseOperation_t transB, int64 mb, int64 n, int64 kb, int64 nnzb,  \
        const ValueType *alpha, const cusparseMatDescr_t descrA,              \
        const ValueType *valA, const int64 *rowPtrA, const int64 *colIndA,    \
        int block_size, const ValueType *B, int64 ldb, const ValueType *beta, \
        ValueType *C, int64 ldc) GKO_NOT_IMPLEMENTED;                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRMM(float, cusparseSbsrmm);
GKO_BIND_CUSPARSE32_BSRMM(double, cusparseDbsrmm);
GKO_BIND_CUSPARSE32_BSRMM(std::complex<float>, cusparseCbsrmm);
GKO_BIND_CUSPARSE32_BSRMM(std::complex<double>, cusparseZbsrmm);
GKO_BIND_CUSPARSE64_BSRMM(float, cusparseSbsrmm);
GKO_BIND_CUSPARSE64_BSRMM(double, cusparseDbsrmm);
GKO_BIND_CUSPARSE64_BSRMM(std::complex<float>, cusparseCbsrmm);
GKO_BIND_CUSPARSE64_BSRMM(std::complex<double>, cusparseZbsrmm);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRMM(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRMM(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_BSRMM
#undef GKO_BIND_CUSPARSE64_BSRMM

template <typename ValueType, typename IndexType>
inline int transpose_buffersize(cusparseHandle_t handle, IndexType mb,
                                IndexType nb, IndexType nnzb,
                                const ValueType *OrigValA,
                                const IndexType *OrigRowPtrA,
                                const IndexType *OrigColIndA, int rowblocksize,
                                int colblocksize) GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
inline void transpose(cusparseHandle_t handle, IndexType mb, IndexType nb,
                      IndexType nnzb, const ValueType *OrigValA,
                      const IndexType *OrigRowPtrA,
                      const IndexType *OrigColIndA, int rowblocksize,
                      int colblocksize, ValueType *TransValA,
                      IndexType *TransRowIndA, IndexType *TransColPtrA,
                      cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                      void *pBuffer) GKO_NOT_IMPLEMENTED;

// cuSparse does not transpose the blocks themselves,
//  only the sparsity pattern
#define GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(ValueType, CusparseName)           \
    template <>                                                                \
    inline int transpose_buffersize<ValueType, int32>(                         \
        cusparseHandle_t handle, int32 mb, int32 nb, int32 nnzb,               \
        const ValueType *OrigValA, const int32 *OrigRowPtrA,                   \
        const int32 *OrigColIndA, int rowblocksize, int colblocksize)          \
    {                                                                          \
        int pBufferSize = -1;                                                  \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName##_bufferSize(               \
            handle, mb, nb, nnzb, as_culibs_type(OrigValA), OrigRowPtrA,       \
            OrigColIndA, rowblocksize, colblocksize, &pBufferSize));           \
        return pBufferSize;                                                    \
    }                                                                          \
    template <>                                                                \
    inline void transpose<ValueType, int32>(                                   \
        cusparseHandle_t handle, int32 mb, int32 nb, int32 nnzb,               \
        const ValueType *OrigValA, const int32 *OrigRowPtrA,                   \
        const int32 *OrigColIndA, int rowblocksize, int colblocksize,          \
        ValueType *TransValA, int32 *TransRowIdxA, int32 *TransColPtrA,        \
        cusparseAction_t copyValues, cusparseIndexBase_t idxBase,              \
        void *pBuffer)                                                         \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, mb, nb, nnzb, as_culibs_type(OrigValA),       \
                         OrigRowPtrA, OrigColIndA, rowblocksize, colblocksize, \
                         as_culibs_type(TransValA), TransRowIdxA,              \
                         TransColPtrA, copyValues, idxBase, pBuffer));         \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(float, cusparseSgebsr2gebsc);
GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(double, cusparseDgebsr2gebsc);
GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(std::complex<float>, cusparseCgebsr2gebsc);
GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(std::complex<double>, cusparseZgebsr2gebsc);

#undef GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32


inline bsrsm2Info_t create_bsr_trsm_info()
{
    bsrsm2Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateBsrsm2Info(&info));
    return info;
}


inline void destroy(bsrsm2Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyBsrsm2Info(info));
}


inline bsrilu02Info_t create_bilu0_info()
{
    bsrilu02Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateBsrilu02Info(&info));
    return info;
}


inline void destroy(bsrilu02Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyBsrilu02Info(info));
}


#define GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(ValueType, CusparseName)          \
    inline int bsrsm2_buffer_size(                                             \
        cusparseHandle_t handle, cusparseOperation_t transA,                   \
        cusparseOperation_t transX, int32 mb, int32 n, int32 nnzb,             \
        const cusparseMatDescr_t descr, ValueType *val, const int32 *rowPtr,   \
        const int32 *colInd, int block_sz, bsrsm2Info_t factor_info)           \
    {                                                                          \
        int factor_work_size = -1;                                             \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, blockDir, transA, transX, mb, n, nnzb, descr, \
                         as_culibs_type(val), rowPtr, colInd, block_sz,        \
                         factor_info, &factor_work_size));                     \
        return factor_work_size;                                               \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(ValueType, CusparseName)        \
    inline int64 bsrsm2_buffer_size(                                         \
        cusparseHandle_t handle, cusparseOperation_t transA,                 \
        cusparseOperation_t transX, int64 mb, int64 n, int64 nnzb,           \
        const cusparseMatDescr_t descr, ValueType *val, const int64 *rowPtr, \
        const int64 *colInd, int block_size, bsrsm2Info_t factor_info)       \
        GKO_NOT_IMPLEMENTED;                                                 \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(float, cusparseSbsrsm2_bufferSize);
GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(double, cusparseDbsrsm2_bufferSize);
GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(std::complex<float>,
                                     cusparseCbsrsm2_bufferSize);
GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(std::complex<double>,
                                     cusparseZbsrsm2_bufferSize);
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(float, cusparseSbsrsm2_bufferSize);
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(double, cusparseDbsrsm2_bufferSize);
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(std::complex<float>,
                                     cusparseCbsrsm2_bufferSize);
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(std::complex<double>,
                                     cusparseZbsrsm2_bufferSize);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE
#undef GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE


#define GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(ValueType, CusparseName)           \
    inline void bsrsm2_analysis(                                               \
        cusparseHandle_t handle, cusparseOperation_t trans1,                   \
        cusparseOperation_t trans2, int32 mb, int32 n, int32 nnzb,             \
        const cusparseMatDescr_t descr, const ValueType *val,                  \
        const int32 *rowPtr, const int32 *colInd, int block_size,              \
        bsrsm2Info_t factor_info, cusparseSolvePolicy_t policy,                \
        void *factor_work_vec)                                                 \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, blockDir, trans1, trans2, mb, n, nnzb, descr, \
                         as_culibs_type(val), rowPtr, colInd, block_size,      \
                         factor_info, policy, factor_work_vec));               \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(ValueType, CusparseName)           \
    inline void bsrsm2_analysis(                                               \
        cusparseHandle_t handle, cusparseOperation_t trans1,                   \
        cusparseOperation_t trans2, size_type mb, size_type n, size_type nnzb, \
        const cusparseMatDescr_t descr, const ValueType *val,                  \
        const int64 *rowPtr, const int64 *colInd, int block_size,              \
        bsrsm2Info_t factor_info, cusparseSolvePolicy_t policy,                \
        void *factor_work_vec) GKO_NOT_IMPLEMENTED;                            \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(float, cusparseSbsrsm2_analysis);
GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(double, cusparseDbsrsm2_analysis);
GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(std::complex<float>,
                                    cusparseCbsrsm2_analysis);
GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(std::complex<double>,
                                    cusparseZbsrsm2_analysis);
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(float, cusparseSbsrsm2_analysis);
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(double, cusparseDbsrsm2_analysis);
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(std::complex<float>,
                                    cusparseCbsrsm2_analysis);
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(std::complex<double>,
                                    cusparseZbsrsm2_analysis);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS
#undef GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS


#define GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(ValueType, CusparseName)            \
    inline void bsrsm2_solve(                                                \
        cusparseHandle_t handle, cusparseOperation_t transA,                 \
        cusparseOperation_t transX, int32 mb, int32 n, int32 nnzb,           \
        const ValueType *alpha, const cusparseMatDescr_t descrA,             \
        const ValueType *valA, const int32 *rowPtrA, const int32 *colIndA,   \
        int blockSizeA, bsrsm2Info_t factor_info, const ValueType *B,        \
        int32 ldb, ValueType *X, int32 ldx, cusparseSolvePolicy_t policy,    \
        void *factor_work_vec)                                               \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                          \
            handle, blockDir, transA, transX, mb, n, nnzb,                   \
            as_culibs_type(alpha), descrA, as_culibs_type(valA), rowPtrA,    \
            colIndA, blockSizeA, factor_info, as_culibs_type(B), ldb,        \
            as_culibs_type(X), ldx, policy, factor_work_vec));               \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(ValueType, CusparseName)              \
    inline void bsrsm2_solve(                                                  \
        cusparseHandle_t handle, cusparseOperation_t trans1,                   \
        cusparseOperation_t trans2, size_type mb, size_type n, size_type nnzb, \
        const ValueType *alpha, const cusparseMatDescr_t descr,                \
        const ValueType *val, const int64 *rowPtr, const int64 *colInd,        \
        int block_size, bsrsm2Info_t factor_info, const ValueType *B,          \
        int64 ldb, ValueType *X, int64 ldx, cusparseSolvePolicy_t policy,      \
        void *factor_work_vec) GKO_NOT_IMPLEMENTED;                            \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(float, cusparseSbsrsm2_solve);
GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(double, cusparseDbsrsm2_solve);
GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(std::complex<float>, cusparseCbsrsm2_solve);
GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(std::complex<double>, cusparseZbsrsm2_solve);
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(float, cusparseSbsrsm2_solve);
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(double, cusparseDbsrsm2_solve);
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(std::complex<float>, cusparseCbsrsm2_solve);
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(std::complex<double>, cusparseZbsrsm2_solve);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BSRSM2_SOLVE
#undef GKO_BIND_CUSPARSE64_BSRSM2_SOLVE


template <typename ValueType, typename IndexType>
int bilu0_buffer_size(cusparseHandle_t handle, IndexType mb, IndexType nnzb,
                      const cusparseMatDescr_t descr, const ValueType *vals,
                      const IndexType *row_ptrs, const IndexType *col_idxs,
                      int block_sz, bsrilu02Info_t info) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(ValueType, CusparseName)           \
    template <>                                                                \
    inline int bilu0_buffer_size<ValueType, int32>(                            \
        cusparseHandle_t handle, int32 mb, int32 nnzb,                         \
        const cusparseMatDescr_t descr, const ValueType *vals,                 \
        const int32 *row_ptrs, const int32 *col_idxs, int block_sz,            \
        bsrilu02Info_t info)                                                   \
    {                                                                          \
        int tmp_buffer_sz{};                                                   \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, blockDir, mb, nnzb, descr,                    \
                         as_culibs_type(const_cast<ValueType *>(vals)),        \
                         row_ptrs, col_idxs, block_sz, info, &tmp_buffer_sz)); \
        return tmp_buffer_sz;                                                  \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(float, cusparseSbsrilu02_bufferSize);
GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(double, cusparseDbsrilu02_bufferSize);
GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(std::complex<float>,
                                    cusparseCbsrilu02_bufferSize);
GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(std::complex<double>,
                                    cusparseZbsrilu02_bufferSize);

#undef GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE


template <typename ValueType, typename IndexType>
inline void bilu0_analysis(cusparseHandle_t handle, IndexType mb,
                           IndexType nnzb, const cusparseMatDescr_t descr,
                           ValueType *vals, const IndexType *row_ptrs,
                           const IndexType *col_idxs, int block_sz,
                           bsrilu02Info_t info, cusparseSolvePolicy_t policy,
                           void *buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_BILU0_ANALYSIS(ValueType, CusparseName)              \
    template <>                                                                \
    inline void bilu0_analysis<ValueType, int32>(                              \
        cusparseHandle_t handle, int32 mb, int32 nnzb,                         \
        const cusparseMatDescr_t descr, ValueType *vals,                       \
        const int32 *row_ptrs, const int32 *col_idxs, int block_sz,            \
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *buffer)       \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                            \
            handle, blockDir, mb, nnzb, descr, as_culibs_type(vals), row_ptrs, \
            col_idxs, block_sz, info, policy, buffer));                        \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_BILU0_ANALYSIS(float, cusparseSbsrilu02_analysis);
GKO_BIND_CUSPARSE_BILU0_ANALYSIS(double, cusparseDbsrilu02_analysis);
GKO_BIND_CUSPARSE_BILU0_ANALYSIS(std::complex<float>,
                                 cusparseCbsrilu02_analysis);
GKO_BIND_CUSPARSE_BILU0_ANALYSIS(std::complex<double>,
                                 cusparseZbsrilu02_analysis);

#undef GKO_BIND_CUSPARSE_BILU0_ANALYSIS


template <typename ValueType, typename IndexType>
void bilu0(cusparseHandle_t handle, IndexType mb, IndexType nnzb,
           const cusparseMatDescr_t descr, ValueType *vals,
           const IndexType *row_ptrs, const IndexType *col_idxs, int block_sz,
           bsrilu02Info_t info, cusparseSolvePolicy_t policy,
           void *buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_BILU0(ValueType, CusparseName)                       \
    template <>                                                                \
    inline void bilu0<ValueType, int32>(                                       \
        cusparseHandle_t handle, int32 mb, int32 nnzb,                         \
        const cusparseMatDescr_t descr, ValueType *vals,                       \
        const int32 *row_ptrs, const int32 *col_idxs, int block_sz,            \
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *buffer)       \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                            \
            handle, blockDir, mb, nnzb, descr, as_culibs_type(vals), row_ptrs, \
            col_idxs, block_sz, info, policy, buffer));                        \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_BILU0(float, cusparseSbsrilu02);
GKO_BIND_CUSPARSE_BILU0(double, cusparseDbsrilu02);
GKO_BIND_CUSPARSE_BILU0(std::complex<float>, cusparseCbsrilu02);
GKO_BIND_CUSPARSE_BILU0(std::complex<double>, cusparseZbsrilu02);

#undef GKO_BIND_CUSPARSE_BILU0


}  // namespace cusparse
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_
