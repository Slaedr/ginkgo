// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_AMP_HPP_
#define GKO_PUBLIC_CORE_MATRIX_AMP_HPP_


#include <limits>

#include <ginkgo/core/base/amp_types.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;

template <typename ValueType, typename IndexType>
class Ell;


/**
 * SpMV strategy applied to each CSR precision bucket of an adaptive mixed
 * precision (AMP) matrix.
 *
 * Defined outside AMP, so that it is the same type across every
 * AMP<ValueType, IndexType> instantiation: helpers that build a bucket's
 * strategy operate on matrix::Csr<bucket_value_type, IndexType>, a different
 * specialization than AMP's own ValueType, and need to accept this type
 * regardless of which AMP instantiation it came from.
 *
 * `sparselib` (cu/hipSPARSE) and, on some executors, `merge_path` may not
 * support every precision bucket (e.g. half or bfloat16); requesting them
 * is a user opt-in and may raise an error at apply time for those bins.
 */
enum class amp_csr_strategy_type {
    classical,
    merge_path,
    load_balance,
    sparselib,
    automatical
};


/**
 * AMP is an adaptive mixed precision matrix class.
 *
 * It takes any sparse matrix and sorts the nonzeros into 'bins' or 'buckets'
 * of different precisions, where each bin is a sparse matrix with a
 * specific value type.
 *
 * @tparam ValueType  Highest precision of matrix elements
 * @tparam IndexType  Integer type of matrix indexes
 *
 * @ingroup amp
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class AMP : public EnableLinOp<AMP<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public DiagonalExtractable<ValueType>,
            public ReadableFromMatrixData<ValueType, IndexType> {
    friend class EnablePolymorphicObject<AMP, LinOp>;
    friend class Dense<ValueType>;
    friend class AMP<to_complex<ValueType>, IndexType>;

    GKO_ASSERT_SUPPORTED_INDEX_TYPE;
    static_assert(
        std::is_same<remove_complex<ValueType>, double>::value ||
            std::is_same<remove_complex<ValueType>, float>::value,
        "AMP is currently only supported for real types double and float!");

public:
    using EnableLinOp<AMP<ValueType, IndexType>>::convert_to;
    using EnableLinOp<AMP<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using real_type = remove_complex<ValueType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;

    // Maximum number of supported precisions.
    static constexpr int num_precisions =
        gko::amp::num_amp_precisions -
        gko::amp::precision_index<real_type>::index;

    void convert_to(Dense<ValueType>* other) const override;

    void move_to(Dense<ValueType>* other) override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    /**
     * Reads in a matrix from nonzero data on the host.
     *
     * The underlying bin type is determined by the type of bin set during
     * the factory build of this AMP matrix.
     */
    void read(const matrix_data<ValueType, IndexType>& data) override;

    void read(device_mat_data&& data) override;

    void read(const device_mat_data& data) override;

    /**
     * Returns the number of precision bins that are non-empty (contain at
     * least one nonzero) after generation.
     *
     * Sparse trailing bins may be "folded up" into the next higher precision
     * bin during generation (see @ref parameters_type::min_bin_nnz_ratio), and
     * a bin may also be empty on its own even without folding. This function
     * reports how many of the `num_precisions` bins actually hold nonzeros.
     *
     * @return  Number of non-empty precision bins.
     */
    int get_num_nonempty_bins() const { return num_nonempty_bins_; }

    /**
     * Returns a pointer to the i-th bin matrix.
     *
     * The zeroth entry always refers to FP64, the 1st entry to FP32,
     * if supported, entry 2 to BF16, and if supported, entry 3 to FP16
     * (and so on).
     * If particular precision is not supported or if the corredponding bin
     * is not necessary for this matrix, its slot is left un-allocated and
     * should return `nullptr`.
     *
     * @param i  bin index (0 to num_precisions-1)
     * @return pointer to the bin matrix, or nullptr if index out of range
     */
    const LinOp* get_bin_matrix(int i) const
    {
        return i >= 0 && i < num_precisions ? mat_bins_[i].get() : nullptr;
    }

    /**
     * Maximum number of nonzero entries per row for a precision bin.
     *
     * @param i  bin index.
     * @return  Max. number of nonzeros per row in bin i.
     */
    IndexType get_max_nnz_per_row_for_bin(const int i) const
    {
        return i >= 0 && i < num_precisions ? max_nnz_per_row_[i] : 0;
    }

    /// Meaning of the tolerance - componentwise or normwise backward error.
    enum class criterion_type { normwise, componentwise };

    /// Algorithm used to perform the AMP SpMV.
    enum class strategy_type {
        /**
         * A single kernel reads all precision buckets and
         * accumulates each row in ValueType.
         */
        monolithic_classical,
        /**
         * One independent SpMV per precision bucket,
         * accumulated into the output vector. Each bucket
         * is free to use its own (Ell/Csr) kernel.
         */
        independent_buckets
    };

    /**
     * SpMV strategy applied to each CSR precision bucket.
     *
     * See @ref amp_csr_strategy_type. Only has an effect for CSR bins
     * (see the `amp_base_format` used at generation) under
     * strategy_type::independent_buckets.
     * The monolithic kernel never consults the buckets' strategies.
     */
    using csr_strategy_type = amp_csr_strategy_type;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The tolerance "epsilon" for adaptive mixed precision generation.
         */
        float GKO_FACTORY_PARAMETER_SCALAR(
            tolerance, std::numeric_limits<real_type>::epsilon() * 100);

        /**
         * Meaning of the tolerance - componentwise or normwise tolerance.
         */
        criterion_type GKO_FACTORY_PARAMETER_SCALAR(
            criterion, criterion_type::componentwise);

        /**
         * Strategy to use for SpMV.
         */
        strategy_type GKO_FACTORY_PARAMETER_SCALAR(
            strategy, strategy_type::monolithic_classical);

        /**
         * Subwarp size used by the CSR kernels.
         *
         * Must be a power of two no larger than
         * the executor's warp size; other values are rounded
         * down to the nearest valid size, with a warning printed once. 0
         * means "automatic": derive it from the maximum number of nonzeros
         * per row over all precision bins (the default behaviour).
         * Ignored for ELL bins and on non-CUDA/HIP executors.
         *
         * This value is normalized at generation time, so get_parameters()
         * reports the subwarp size actually in use, which may differ from
         * the one requested.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(subwarp_size, 0);

        /**
         * Strategy to use for each CSR precision bucket's own SpMV.
         * See @ref csr_strategy_type.
         */
        csr_strategy_type GKO_FACTORY_PARAMETER_SCALAR(
            csr_strategy, csr_strategy_type::automatical);

        /**
         * Threshold, as a ratio of the original matrix's number of nonzeros,
         * below which a trailing (lowest-precision) bin is folded into the
         * next higher precision bin instead of being generated on its own.
         *
         * During generation, the lowest-precision bin (bin index
         * `num_precisions - 1`) is checked first: if its nonzero count is
         * below `ratio * nnz(original matrix)`, its entries are merged into
         * the next higher precision bin and it is left empty. The next
         * higher bin (now holding the merged count) is checked the same
         * way, and so on, until a bin meets the threshold or bin 0 is
         * reached. Bin 0 itself is never folded away.
         *
         * A value of 0 disables folding.
         */
        float GKO_FACTORY_PARAMETER_SCALAR(bin_foldup_nnz_ratio, 0.01f);
    };
    GKO_ENABLE_LIN_OP_FACTORY(AMP, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Copy-assigns an AMP matrix. Preserves the executor while copying each
     * precision bin, using its `copy_from` function, as well as the size.
     */
    AMP& operator=(const AMP&);

    /**
     * Move-assigns an AMP matrix. Preserves the executor, moves the data over
     * Leaves the moved-from object in an empty state (0x0 with empty array).
     */
    AMP& operator=(AMP&&);

    /**
     * Copy-constructs an AMP matrix. Inherits executor and dimensions, but
     * copies data without padding.
     */
    AMP(const AMP&);

    /**
     * Move-constructs an AMP matrix. Inherits executor, dimensions and data
     * with padding. The moved-from object is empty (0x0 with empty Array).
     */
    AMP(AMP&&);

protected:
    /// Creates an empty matrix.
    explicit AMP(std::shared_ptr<const Executor>);

    /**
     * Constructs an AMP matrix from a given (high-precision) matrix.
     * Inherits the executor and size; runs an analysis step.
     */
    explicit AMP(const Factory* factory, std::shared_ptr<const LinOp> lin_op)
        : EnableLinOp<AMP>(factory->get_executor(), lin_op->get_size()),
          parameters_{factory->get_parameters()},
          mat_bins_(generate_amp(lin_op.get()))
    {
        normalize_subwarp_size();
        init_one();
    }

    /**
     * Rounds parameters_.subwarp_size down to the nearest power of two no
     * larger than the executor's warp size (0 stays "automatic"), warning
     * once if the requested value had to change. Defined in amp.cpp.
     */
    void normalize_subwarp_size();

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    /**
     * Generate binned adaptive precision matrix from given (fixed precision)
     * matrix.
     */
    std::array<std::unique_ptr<const LinOp>, num_precisions> generate_amp(
        const LinOp* matrix);

protected:
    /// Max. number of nonzeros per row for each precision bin.
    std::array<IndexType, num_precisions> max_nnz_per_row_{};

    /// Number of precision bins that hold at least one nonzero, after any
    /// bin folding performed during generation. @sa get_num_nonempty_bins
    int num_nonempty_bins_{0};

    /// Array of bins of the different precisions.
    std::array<std::unique_ptr<const LinOp>, num_precisions> mat_bins_;

    /**
     * Sets #one_ to a scalar 1.0 on the current executor. Used as alpha/beta
     * when accumulating buckets for the `independent_buckets` strategy.
     */
    void init_one();

    /// Scalar one, used as alpha/beta when accumulating buckets in the
    /// `independent_buckets` SpMV strategy.
    std::shared_ptr<const LinOp> one_;

private:
    /// CSR-bin generation
    gko::amp::precision_array<std::unique_ptr<const LinOp>, ValueType>
    generate_amp_impl(const matrix::Csr<ValueType, IndexType>* mtx);

    /// ELL-bin generation
    gko::amp::precision_array<std::unique_ptr<const LinOp>, ValueType>
    generate_amp_impl(const matrix::Ell<ValueType, IndexType>* mtx);
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_AMP_HPP_
