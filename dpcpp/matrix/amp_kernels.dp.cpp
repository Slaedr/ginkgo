// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/mixed_precision_types.hpp"
#include "core/matrix/amp_helpers.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The AMP matrix format namespace.
 *
 * @ingroup amp
 */
namespace amp {


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv_ell(std::shared_ptr<const DpcppExecutor> exec,
              const matrix::AMP<MatrixValueType, IndexType>* a,
              const matrix::Dense<InputValueType>* b,
              matrix::Dense<OutputValueType>* c)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_SPMV_ELL_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv_ell(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Dense<MatrixValueType>* alpha,
                       const matrix::AMP<MatrixValueType, IndexType>* a,
                       const matrix::Dense<InputValueType>* b,
                       const matrix::Dense<OutputValueType>* beta,
                       matrix::Dense<OutputValueType>* c)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_ADVANCED_SPMV_ELL_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv_csr(std::shared_ptr<const DpcppExecutor> exec,
              const matrix::AMP<MatrixValueType, IndexType>* a,
              const matrix::Dense<InputValueType>* b,
              matrix::Dense<OutputValueType>* c)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_SPMV_CSR_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv_csr(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Dense<MatrixValueType>* alpha,
                       const matrix::AMP<MatrixValueType, IndexType>* a,
                       const matrix::Dense<InputValueType>* b,
                       const matrix::Dense<OutputValueType>* beta,
                       matrix::Dense<OutputValueType>* c)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_ADVANCED_SPMV_CSR_KERNEL);


}  // namespace amp
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
