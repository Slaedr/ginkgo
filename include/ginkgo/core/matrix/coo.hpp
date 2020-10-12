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

#ifndef GKO_CORE_MATRIX_COO_HPP_
#define GKO_CORE_MATRIX_COO_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
/**
 * @brief The matrix namespace.
 *
 * @ingroup matrix
 */
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;


template <typename ValueType>
class Dense;


template <typename ValueType, typename IndexType>
class CooBuilder;


/**
 * COO stores a matrix in the coordinate matrix format.
 *
 * The nonzero elements are stored in an array row-wise (but not neccessarily
 * sorted by column index within a row). Two extra arrays contain the row and
 * column indexes of each nonzero element of the matrix.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup coo
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Coo : public EnableLinOp<Coo<ValueType, IndexType>>,
            public EnableCreateMethod<Coo<ValueType, IndexType>>,
            public EnableDistributedCreateMethod<Coo<ValueType, IndexType>>,
            public ConvertibleTo<Coo<next_precision<ValueType>, IndexType>>,
            public ConvertibleTo<Csr<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public DiagonalExtractable<ValueType>,
            public ReadableFromMatrixData<ValueType, IndexType>,
            public WritableToMatrixData<ValueType, IndexType>,
            public EnableAbsoluteComputation<
                remove_complex<Coo<ValueType, IndexType>>> {
    friend class EnableCreateMethod<Coo>;
    friend class EnableDistributedCreateMethod<Coo>;
    friend class EnablePolymorphicObject<Coo, LinOp>;
    friend class Csr<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class CooBuilder<ValueType, IndexType>;
    friend class Coo<to_complex<ValueType>, IndexType>;

public:
    using EnableLinOp<Coo>::convert_to;
    using EnableLinOp<Coo>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Coo>;

    friend class Coo<next_precision<ValueType>, IndexType>;

    void convert_to(
        Coo<next_precision<ValueType>, IndexType> *result) const override;

    void move_to(Coo<next_precision<ValueType>, IndexType> *result) override;

    void convert_to(Csr<ValueType, IndexType> *other) const override;

    void move_to(Csr<ValueType, IndexType> *other) override;

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void read(const mat_data &data) override;

    void read(const mat_data &data, const Array<size_type> &dist) override;

    void write(mat_data &data) const override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc Csr::get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type *get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc Csr::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns the row indexes of the matrix.
     *
     * @return the row indexes of the matrix.
     */
    index_type *get_row_idxs() noexcept { return row_idxs_.get_data(); }

    /**
     * @copydoc Csr::get_row_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_row_idxs() const noexcept
    {
        return row_idxs_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
    }

    /**
     * Applies Coo matrix axpy to a vector (or a sequence of vectors).
     *
     * Performs the operation x = Coo * b + x
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     *
     * @return this
     */
    LinOp *apply2(const LinOp *b, LinOp *x)
    {
        auto exec = this->get_executor();
        if (auto chk = dynamic_cast<const gko::MpiExecutor *>(exec.get())) {
            this->validate_distributed_application_parameters(b, x);
            this->distributed_apply2_impl(make_temporary_clone(exec, b).get(),
                                          make_temporary_clone(exec, x).get());
        } else {
            this->validate_application_parameters(b, x);
            this->apply2_impl(make_temporary_clone(exec, b).get(),
                              make_temporary_clone(exec, x).get());
        }
        return this;
    }

    /**
     * @copydoc apply2(cost LinOp *, LinOp *)
     */
    const LinOp *apply2(const LinOp *b, LinOp *x) const
    {
        auto exec = this->get_executor();
        if (auto chk = dynamic_cast<const gko::MpiExecutor *>(exec.get())) {
            this->validate_distributed_application_parameters(b, x);
            this->distributed_apply2_impl(make_temporary_clone(exec, b).get(),
                                          make_temporary_clone(exec, x).get());
        } else {
            this->validate_application_parameters(b, x);
            this->apply2_impl(make_temporary_clone(exec, b).get(),
                              make_temporary_clone(exec, x).get());
        }
        return this;
    }

    /**
     * Performs the operation x = alpha * Coo * b + x.
     *
     * @param alpha  scaling of the result of Coo * b
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     *
     * @return this
     */
    LinOp *apply2(const LinOp *alpha, const LinOp *b, LinOp *x)
    {
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        auto exec = this->get_executor();
        if (auto chk = dynamic_cast<const gko::MpiExecutor *>(exec.get())) {
            this->validate_distributed_application_parameters(b, x);
            this->distributed_apply2_impl(
                make_temporary_clone(exec, alpha).get(),
                make_temporary_clone(exec, b).get(),
                make_temporary_clone(exec, x).get());
        } else {
            this->validate_application_parameters(b, x);
            this->apply2_impl(make_temporary_clone(exec, alpha).get(),
                              make_temporary_clone(exec, b).get(),
                              make_temporary_clone(exec, x).get());
        }
        return this;
    }

    /**
     * @copydoc apply2(const LinOp *, const LinOp *, LinOp *)
     */
    const LinOp *apply2(const LinOp *alpha, const LinOp *b, LinOp *x) const
    {
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        auto exec = this->get_executor();
        if (auto chk = dynamic_cast<const gko::MpiExecutor *>(exec.get())) {
            this->validate_application_parameters(b, x);
            this->distributed_apply2_impl(
                make_temporary_clone(exec, alpha).get(),
                make_temporary_clone(exec, b).get(),
                make_temporary_clone(exec, x).get());
        } else {
            this->validate_distributed_application_parameters(b, x);
            this->apply2_impl(make_temporary_clone(exec, alpha).get(),
                              make_temporary_clone(exec, b).get(),
                              make_temporary_clone(exec, x).get());
        }
        return this;
    }

protected:
    /**
     * Creates an uninitialized COO matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     */
    Coo(std::shared_ptr<const Executor> exec, const dim<2> &size = dim<2>{},
        size_type num_nonzeros = {})
        : EnableLinOp<Coo>(exec, size, size),
          index_set_(size[0] + 1),
          values_(exec, num_nonzeros),
          col_idxs_(exec, num_nonzeros),
          row_idxs_(exec, num_nonzeros)
    {}


    /**
     * Creates an uninitialized COO matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     */
    Coo(std::shared_ptr<const Executor> exec, const dim<2> &size,
        const IndexSet<size_type> &index_set, size_type num_nonzeros)
        : EnableLinOp<Coo>(exec, size, size),
          index_set_(index_set),
          values_(exec, num_nonzeros),
          col_idxs_(exec, num_nonzeros),
          row_idxs_(exec, num_nonzeros)
    {
        this->set_size(dim<2>(index_set_.get_num_elems(), size[1]));
    }

    /**
     * Creates a COO matrix from already allocated (and initialized) row
     * index, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowIdxArray  type of `row_idxs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param row_idxs  array of row pointers
     *
     * @note If one of `row_idxs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowIdxsArray>
    Coo(std::shared_ptr<const Executor> exec, const dim<2> &size,
        ValuesArray &&values, ColIdxsArray &&col_idxs, RowIdxsArray &&row_idxs)
        : EnableLinOp<Coo>(exec, size, size),
          index_set_(size[0] + 1),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_idxs_{exec, std::forward<RowIdxsArray>(row_idxs)}
    {
        GKO_ASSERT_EQ(values_.get_num_elems(), col_idxs_.get_num_elems());
        GKO_ASSERT_EQ(values_.get_num_elems(), row_idxs_.get_num_elems());
    }

    /**
     * Creates a COO matrix from already allocated (and initialized) row
     * index, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowIdxArray  type of `row_idxs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param row_idxs  array of row pointers
     *
     * @note If one of `row_idxs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowIdxsArray>
    Coo(std::shared_ptr<const Executor> exec, const dim<2> &size,
        IndexSet<size_type> &index_set, ValuesArray &&values,
        ColIdxsArray &&col_idxs, RowIdxsArray &&row_idxs)
        : EnableLinOp<Coo>(exec, size, size),
          index_set_(index_set),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_idxs_{exec, std::forward<RowIdxsArray>(row_idxs)}
    {
        this->set_size(gko::dim<2>(index_set_.get_num_elems(), size[1]));
        GKO_ASSERT_EQ(values_.get_num_elems(), col_idxs_.get_num_elems());
        GKO_ASSERT_EQ(values_.get_num_elems(), row_idxs_.get_num_elems());
    }

    template <typename ExecType, typename ValuesArray, typename ColIdxsArray,
              typename RowIdxsArray>
    static std::unique_ptr<Coo> distribute_impl(
        ExecType &exec, const dim<2> &global_size, IndexSet<size_type> &row_set,
        ValuesArray &&values, ColIdxsArray &&col_idxs, RowIdxsArray &&row_idxs)
    {
        using itype = index_type;
        auto mpi_exec = as<gko::MpiExecutor>(exec.get());
        auto sub_exec = exec->get_sub_executor();
        auto num_ranks = mpi_exec->get_num_ranks();
        auto my_rank = mpi_exec->get_my_rank();
        auto root_rank = mpi_exec->get_root_rank();
        itype num_rows = row_set.get_num_elems();
        // Can also be the last element of the row_idx array as we sort the
        // row_idxs by row
        itype total_num_rows = global_size[0];
        itype total_num_nnz = row_idxs.get_num_elems();
        mpi_exec->broadcast(&total_num_nnz, 1, root_rank);
        auto row_ptrs = Array<itype>{exec->get_master()};
        auto row_idxs_cpy = Array<itype>{exec->get_master()};
        row_idxs_cpy = row_idxs;
        if (my_rank == root_rank) {
            row_ptrs = Array<itype>(exec->get_master(),
                                    size_type(total_num_rows + 1), itype(0));
            std::for_each(row_idxs_cpy.get_const_data(),
                          row_idxs_cpy.get_const_data() + total_num_nnz,
                          [&](size_type v) {
                              if (v + 1 < total_num_rows + 1) {
                                  ++row_ptrs.get_data()[v + 1];
                              }
                          });
            std::partial_sum(row_ptrs.get_data(),
                             row_ptrs.get_data() + total_num_rows + 1,
                             row_ptrs.get_data());
        }

        // TODO: Can possibly be moved to the exec instead of master.
        auto nnz_per_row = Array<itype>{sub_exec->get_master()};
        auto row_ptr_clone = Array<itype>{sub_exec->get_master()};
        if (my_rank == root_rank) {
            row_ptr_clone =
                Array<itype>(sub_exec->get_master(), row_ptrs.get_data() + 1,
                             row_ptrs.get_data() + 1 + total_num_rows);
            nnz_per_row =
                Array<itype>(sub_exec->get_master(), row_ptrs.get_data() + 1,
                             row_ptrs.get_data() + 1 + total_num_rows);
            std::adjacent_difference(nnz_per_row.get_data(),
                                     nnz_per_row.get_data() + total_num_rows,
                                     nnz_per_row.get_data());
        }
        auto num_nnz_per_row = nnz_per_row.distribute(exec, row_set);
        num_nnz_per_row.set_executor(exec->get_master());
        auto row_start = row_ptr_clone.distribute(exec, row_set);
        row_start.set_executor(exec->get_master());
        auto max_index_size = row_set.get_largest_element_in_set();
        auto index_set =
            gko::IndexSet<itype>{(max_index_size + 1) * global_size[1]};
        for (auto i = 0; i < num_rows; ++i) {
            index_set.add_subset(row_start.get_const_data()[i] -
                                     num_nnz_per_row.get_const_data()[i],
                                 row_start.get_const_data()[i]);
        }
        auto updated_values = values.distribute(exec, index_set);
        auto updated_col_idxs = col_idxs.distribute(exec, index_set);
        auto updated_row_idxs = row_idxs.distribute(exec, index_set);
        return Coo::create(exec, global_size, row_set, updated_values,
                           updated_col_idxs, updated_row_idxs);
    }

    template <typename ExecType>
    static std::unique_ptr<Coo> distribute_impl(ExecType &exec,
                                                const dim<2> &size)
    {
        return Coo::create(exec, size);
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    void distributed_apply_impl(const LinOp *b, LinOp *x) const override
    {
        this->apply_impl(b, x);
    }

    void distributed_apply_impl(const LinOp *alpha, const LinOp *b,
                                const LinOp *beta, LinOp *x) const override
    {
        this->apply_impl(alpha, b, beta, x);
    }

    void apply2_impl(const LinOp *b, LinOp *x) const;

    void apply2_impl(const LinOp *alpha, const LinOp *b, LinOp *x) const;

    void distributed_apply2_impl(const LinOp *b, LinOp *x) const
    {
        this->apply2_impl(b, x);
    }

    void distributed_apply2_impl(const LinOp *alpha, const LinOp *b,
                                 LinOp *x) const
    {
        this->apply2_impl(alpha, b, x);
    }

private:
    IndexSet<size_type> index_set_;
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    Array<index_type> row_idxs_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_COO_HPP_
