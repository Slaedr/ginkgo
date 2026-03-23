// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_SOLVER_GAUSS_SEIDEL_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_SOLVER_GAUSS_SEIDEL_HPP_

#include <vector>

#include <ginkgo/config.hpp>

#if GINKGO_BUILD_MPI

#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/distributed/vector_cache.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

namespace gko {
namespace experimental {
namespace distributed {
namespace solver {


template <typename ValueType = default_precision,
          typename LocalIndexType = int32, typename GlobalIndexType = int64>
class FwdGaussSeidel
    : public EnableLinOp<
          FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>>,
      public gko::solver::EnableSolverBase<
          FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>>,
      public gko::solver::EnableIterativeBase<
          FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>>,
      public DistributedBase {
    friend class EnableLinOp<FwdGaussSeidel>;
    friend class EnablePolymorphicObject<FwdGaussSeidel, LinOp>;

public:
    using value_type = ValueType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;

    bool apply_uses_initial_guess() const override
    {
        return (this->parameters_.init_guess_mode ==
                gko::solver::initial_guess_mode::provided);
    }

    class Factory;

    struct parameters_type
        : gko::solver::enable_iterative_solver_factory_parameters<
              parameters_type, Factory> {
        std::vector<LocalIndexType> GKO_FACTORY_PARAMETER_SCALAR(
            color_ptrs, std::vector<LocalIndexType>());

        gko::solver::initial_guess_mode GKO_FACTORY_PARAMETER_SCALAR(
            init_guess_mode, gko::solver::initial_guess_mode::zero);
    };
    GKO_ENABLE_LIN_OP_FACTORY(FwdGaussSeidel, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, LocalIndexType,
                                         GlobalIndexType>());

protected:
    explicit FwdGaussSeidel(std::shared_ptr<const Executor> exec,
                            mpi::communicator comm)
        : EnableLinOp<FwdGaussSeidel>(std::move(exec)),
          DistributedBase(std::move(comm))
    {}

    explicit FwdGaussSeidel(const Factory* factory,
                            std::shared_ptr<const LinOp> system_matrix);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    template <typename VectorType>
    void apply_dense_impl(const VectorType* dense_b, VectorType* dense_x) const;

    std::vector<LocalIndexType> color_ptrs_;
    std::shared_ptr<const LinOp> local_mtx_;
    std::shared_ptr<const LinOp> non_local_mtx_;
    std::shared_ptr<const RowGatherer<LocalIndexType>> row_gatherer_;
    std::vector<std::shared_ptr<const LinOp>> color_solvers_;
    mutable gko::detail::DenseCache<ValueType> b_corrected_cache_;
    // Double-buffered recv caches for overlapping halo exchange with compute
    mutable detail::VectorCache<ValueType> recv_cache_[2];
    mutable detail::VectorCache<ValueType> host_recv_cache_[2];
};


}  // namespace solver
}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif  // GINKGO_BUILD_MPI

#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_SOLVER_GAUSS_SEIDEL_HPP_
