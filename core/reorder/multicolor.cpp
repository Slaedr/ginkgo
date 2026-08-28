// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/reorder/multicolor_kernels.hpp"


namespace gko {
namespace reorder {
namespace multicolor {
namespace {


GKO_REGISTER_OPERATION(compute_permutation_csr,
                       multicolor::compute_permutation_csr);


}  // anonymous namespace
}  // namespace multicolor


template <template <typename, typename> class MatrixTempl, typename ValueType,
          typename IndexType>
void multicolor_reorder(const MatrixTempl<ValueType, IndexType>* const mtx,
                        std::vector<IndexType>& color_ptrs,
                        IndexType* const permutation,
                        IndexType* const inv_permutation)
{
    const auto exec = mtx->get_executor();
    const IndexType num_rows = mtx->get_size()[0];
    exec->run(multicolor::make_compute_permutation_csr(
        num_rows, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
        color_ptrs, permutation, inv_permutation));
}


template <typename ValueType, typename IndexType>
Multicolor<ValueType, IndexType>::Multicolor(
    std::shared_ptr<const Executor> exec)
    : EnablePolymorphicObject<Multicolor, ReorderingBase<IndexType>>(
          std::move(exec))
{}


template <typename ValueType, typename IndexType>
Multicolor<ValueType, IndexType>::Multicolor(const Factory* factory,
                                             const ReorderingBaseArgs& args)
    : EnablePolymorphicObject<Multicolor, ReorderingBase<IndexType>>(
          factory->get_executor()),
      parameters_{factory->get_parameters()}
{
    using CsrType = matrix::Csr<ValueType, IndexType>;
    using FloatCsr = matrix::Csr<float, IndexType>;
    auto exec = this->get_executor();
    auto sysmat = args.system_matrix;

    // The adjacency matrix has to be square.
    GKO_ASSERT_IS_SQUARE_MATRIX(sysmat);
    auto const size = sysmat->get_size()[0];

    IndexType nnz{};
    const IndexType* row_ptrs{};
    const IndexType* col_idxs{};
    if (auto csrmat = std::dynamic_pointer_cast<const CsrType>(sysmat)) {
        nnz = static_cast<IndexType>(csrmat->get_num_stored_elements());
        row_ptrs = csrmat->get_const_row_ptrs();
        col_idxs = csrmat->get_const_col_idxs();
    } else if (auto smat =
                   std::dynamic_pointer_cast<const SparsityMatrix>(sysmat)) {
        nnz = static_cast<IndexType>(smat->get_num_nonzeros());
        row_ptrs = smat->get_const_row_ptrs();
        col_idxs = smat->get_const_col_idxs();
    } else {
        GKO_NOT_SUPPORTED(sysmat);
    }

    permutation_ = PermutationMatrix::create(exec, size);

    // To make it explicit.
    inv_permutation_ = nullptr;
    if (parameters_.construct_inverse_permutation) {
        inv_permutation_ = PermutationMatrix::create(exec, size);
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    if (parameters_.skip_symmetrize) {
        // Color the pattern as-is, without allocating a copy.
        auto pattern = SparsityMatrix::create_const(
            exec, sysmat->get_size(),
            make_const_array_view<IndexType>(exec, nnz, col_idxs),
            make_const_array_view<IndexType>(exec, size + 1, row_ptrs));
        multicolor_reorder(
            pattern.get(), color_ptrs_, permutation_->get_permutation(),
            inv_permutation_ ? inv_permutation_->get_permutation() : nullptr);
    } else {
        // Color the pattern of A + A^T, so that no independent set has an
        // entry of A in either direction between two of its rows. Only the
        // pattern matters, so this uses a value type decoupled from
        // ValueType (as gko::experimental::reorder::Amd does), keeping
        // half/bfloat16 instantiations out of spgeam.
        auto pattern = matrix::SparsityCsr<float, IndexType>::create_const(
            exec, sysmat->get_size(),
            make_const_array_view<IndexType>(exec, nnz, col_idxs),
            make_const_array_view<IndexType>(exec, size + 1, row_ptrs));
        auto sym = FloatCsr::create(exec);
        pattern->convert_to(sym);
        if (!parameters_.skip_sorting) {
            sym->sort_by_column_index();
        }
        auto scalar = initialize<matrix::Dense<float>>({one<float>()}, exec);
        auto id = matrix::Identity<float>::create(exec, size);
        // compute A^T + A
        sym->transpose()->apply(scalar, id, scalar, sym);

        multicolor_reorder(
            sym.get(), color_ptrs_, permutation_->get_permutation(),
            inv_permutation_ ? inv_permutation_->get_permutation() : nullptr);
    }
}


#define GKO_DECLARE_MULTICOLOR(ValueType, IndexType) \
    class Multicolor<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MULTICOLOR);


}  // namespace reorder
}  // namespace gko
