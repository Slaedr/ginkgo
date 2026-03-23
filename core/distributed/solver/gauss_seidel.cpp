// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/solver/gauss_seidel.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gauss_seidel.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/config/config_helper.hpp"
#include "core/distributed/helpers.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace solver {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>::FwdGaussSeidel(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix)
    : EnableLinOp<FwdGaussSeidel>(factory->get_executor(),
                                  gko::transpose(system_matrix->get_size())),
      gko::solver::EnableSolverBase<FwdGaussSeidel>{system_matrix},
      gko::solver::EnableIterativeBase<FwdGaussSeidel>{
          stop::combine(factory->get_parameters().criteria)},
      DistributedBase(
          as<experimental::distributed::DistributedBase>(system_matrix.get())
              ->get_communicator()),
      parameters_{factory->get_parameters()},
      color_ptrs_{parameters_.color_ptrs}
{
    if (color_ptrs_.size() != 0 && color_ptrs_.size() < 2) {
        GKO_INVALID_STATE("Color row pointers array has invalid size!");
    }

    using dist_mat_type =
        distributed::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    auto dist_mat = as<dist_mat_type>(system_matrix.get());
    local_mtx_ = dist_mat->get_local_matrix();
    non_local_mtx_ = dist_mat->get_non_local_matrix();
    row_gatherer_ = dist_mat->get_row_gatherer();

    // Create per-color local FGS solvers
    auto exec = factory->get_executor();
    const auto num_colors = static_cast<int>(color_ptrs_.size()) - 1;
    using LocalFgs = gko::solver::FwdGaussSeidel<ValueType, LocalIndexType>;
    for (int c = 0; c < num_colors; ++c) {
        std::vector<LocalIndexType> single_color_ptrs = {color_ptrs_[c],
                                                         color_ptrs_[c + 1]};
        auto local_fgs =
            LocalFgs::build()
                .with_criteria(stop::Iteration::build().with_max_iters(1u))
                .with_color_ptrs(single_color_ptrs)
                .with_init_guess_mode(gko::solver::initial_guess_mode::provided)
                .on(exec)
                ->generate(local_mtx_);
        color_solvers_.push_back(gko::share(std::move(local_fgs)));
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
template <typename VectorType>
void FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>::
    apply_dense_impl(const VectorType* dense_b, VectorType* dense_x) const
{
    using Dense = matrix::Dense<ValueType>;

    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    const auto ncols = dense_b->get_size()[1];
    const auto local_nrows = local_mtx_->get_size()[0];

    // Initialize x based on init_guess_mode
    if (parameters_.init_guess_mode == gko::solver::initial_guess_mode::zero) {
        dense_x->fill(zero<typename VectorType::value_type>());
    } else if (parameters_.init_guess_mode ==
               gko::solver::initial_guess_mode::rhs) {
        dense_x->copy_from(dense_b);
    }

    // Setup recv buffers (following init_recv_buffers pattern from matrix.cpp)
    auto coll_comm = row_gatherer_->get_collective_communicator();
    auto base_comm = coll_comm->get_base_communicator();
    auto global_recv_dim =
        dim<2>{static_cast<size_type>(row_gatherer_->get_size()[0]), ncols};
    auto local_recv_dim =
        dim<2>{static_cast<size_type>(coll_comm->get_recv_size()), ncols};
    recv_cache_.init(exec, base_comm, global_recv_dim, local_recv_dim);
    host_recv_cache_.init(exec->get_master(), base_comm, global_recv_dim,
                          local_recv_dim);

    // Setup b_corrected cache (local Dense)
    b_corrected_cache_.init(exec, dim<2>{local_nrows, ncols});

    // Stopping criterion setup, TODO: Move to outer factory ?
    constexpr uint8 stopping_id{1};
    array<stopping_status> stop_status(exec, ncols);
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x);

    // Create scalar constants for advanced apply, TODO: Move to workspace
    auto one_op = gko::initialize<Dense>({one<ValueType>()}, exec);
    auto neg_one_op = gko::initialize<Dense>({-one<ValueType>()}, exec);

    const auto num_colors = static_cast<int>(color_solvers_.size());

    auto* local_b = gko::detail::get_local(dense_b);
    auto* local_x = gko::detail::get_local(dense_x);

    int iter = -1;

    while (true) {
        ++iter;

        for (int c = 0; c < num_colors; ++c) {
            // Halo exchange: gather remote x values
            auto recv_ptr = mpi::requires_host_buffer(exec, comm)
                                ? host_recv_cache_.get()
                                : recv_cache_.get();
            auto req = row_gatherer_->apply_async(dense_x, recv_ptr);
            req.wait();

            if (recv_ptr != recv_cache_.get()) {
                recv_cache_->copy_from(host_recv_cache_.get());
            }

            auto* recv_local = recv_cache_->get_local_vector();

            // RHS correction: b_corrected = b_local - A_nonlocal * recv
            b_corrected_cache_->copy_from(local_b);
            non_local_mtx_->apply(neg_one_op, recv_local, one_op,
                                  b_corrected_cache_.get());

            // Local FGS sweep for this color
            color_solvers_[c]->apply(b_corrected_cache_.get(), local_x);
        }

        // Check stopping criterion
        bool one_changed = false;
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .solution(dense_x)
                .check(stopping_id, true, &stop_status, &one_changed);
        if (all_stopped) {
            break;
        }
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::distributed::precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
typename FwdGaussSeidel<ValueType, LocalIndexType,
                        GlobalIndexType>::parameters_type
FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = FwdGaussSeidel::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("criteria")) {
        params.with_criteria(
            config::parse_or_get_criteria(obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("color_ptrs")) {
        auto arr = obj.get_array();
        std::vector<LocalIndexType> ptrs;
        ptrs.reserve(arr.size());
        for (const auto& elem : arr) {
            ptrs.push_back(config::get_value<LocalIndexType>(elem));
        }
        params.with_color_ptrs(std::move(ptrs));
    }
    if (auto& obj = config_check.get("init_guess_mode")) {
        params.with_init_guess_mode(
            config::get_value<gko::solver::initial_guess_mode>(obj));
    }
    return params;
}


#define GKO_DECLARE_DIST_FWD_GS(ValueType, LocalIndexType, GlobalIndexType) \
    class FwdGaussSeidel<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_DIST_FWD_GS);


}  // namespace solver
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
