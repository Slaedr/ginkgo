// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/gauss_seidel.hpp"

#include <string>

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

#include "core/base/precision_dispatch_extra.hpp"
#include "core/distributed/helpers.hpp"
#include "core/matrix/amp_helpers.hpp"
#include "core/solver/gauss_seidel_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace gssdl {
namespace {


GKO_REGISTER_OPERATION(multicolor_fgs_ell, gssdl::multicolor_fgs_ell);
GKO_REGISTER_OPERATION(multicolor_fgs_amp, gssdl::multicolor_fgs_amp);
GKO_REGISTER_OPERATION(multicolor_fgs_csr, gssdl::multicolor_fgs_csr);
GKO_REGISTER_OPERATION(multicolor_fgs_amp_csr, gssdl::multicolor_fgs_amp_csr);


}  // anonymous namespace
}  // namespace gssdl


template <typename ValueType, typename IndexType>
FwdGaussSeidel<ValueType, IndexType>&
FwdGaussSeidel<ValueType, IndexType>::operator=(const FwdGaussSeidel& other)
{
    if (&other != this) {
        EnableLinOp<FwdGaussSeidel>::operator=(other);
        EnableSolverBase<FwdGaussSeidel>::operator=(other);
        EnableIterativeBase<FwdGaussSeidel>::operator=(other);
        this->parameters_ = other.parameters_;
        this->color_row_ptrs_ = other.color_row_ptrs_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
FwdGaussSeidel<ValueType, IndexType>&
FwdGaussSeidel<ValueType, IndexType>::operator=(FwdGaussSeidel&& other)
{
    if (&other != this) {
        EnableLinOp<FwdGaussSeidel>::operator=(std::move(other));
        EnableSolverBase<FwdGaussSeidel>::operator=(std::move(other));
        EnableIterativeBase<FwdGaussSeidel>::operator=(std::move(other));
        this->parameters_ = std::exchange(other.parameters_, parameters_type{});
        this->color_row_ptrs_ =
            std::exchange(other.color_row_ptrs_, std::vector<IndexType>{});
    }
    return *this;
}


template <typename ValueType, typename IndexType>
FwdGaussSeidel<ValueType, IndexType>::FwdGaussSeidel(
    const FwdGaussSeidel& other)
    : FwdGaussSeidel(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
FwdGaussSeidel<ValueType, IndexType>::FwdGaussSeidel(FwdGaussSeidel&& other)
    : FwdGaussSeidel(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void FwdGaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                      LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    mixed_precision_base_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
template <typename VectorType_b, typename VectorType_x>
void FwdGaussSeidel<ValueType, IndexType>::apply_dense_impl(
    const VectorType_b* dense_b, VectorType_x* dense_x) const
{
    using ws = workspace_traits<FwdGaussSeidel>;
    constexpr uint8 stopping_id{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    auto& stop_status = this->template create_workspace_array<stopping_status>(
        ws::stop, dense_b->get_size()[1]);

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x);

    if (parameters_.init_guess_mode == initial_guess_mode::zero) {
        dense_x->fill(zero<typename VectorType_x::value_type>());
    } else if (parameters_.init_guess_mode == initial_guess_mode::rhs) {
        dense_x->copy_from(dense_b);
    }

    int iter = -1;

    auto ellmat =
        std::dynamic_pointer_cast<const matrix::Ell<ValueType, IndexType>>(
            this->get_system_matrix());
    auto ampmat =
        std::dynamic_pointer_cast<const matrix::AMP<ValueType, IndexType>>(
            this->get_system_matrix());
    auto csrmat =
        std::dynamic_pointer_cast<const matrix::Csr<ValueType, IndexType>>(
            this->get_system_matrix());
    if (!ellmat && !ampmat && !csrmat) {
        GKO_NOT_SUPPORTED(this->get_system_matrix());
    }

    while (true) {
        ++iter;

        if (ellmat) {
            exec->run(gssdl::make_multicolor_fgs_ell(
                color_row_ptrs_, ellmat.get(), gko::detail::get_local(dense_b),
                gko::detail::get_local(dense_x), iter == 0, &stop_status));
        } else if (ampmat) {
            const bool csr_bins =
                dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(
                    ampmat->get_bin_matrix(0)) != nullptr;
            if (csr_bins) {
                exec->run(gssdl::make_multicolor_fgs_amp_csr(
                    color_row_ptrs_, ampmat.get(),
                    gko::detail::get_local(dense_b),
                    gko::detail::get_local(dense_x), iter == 0, &stop_status));
            } else {
                exec->run(gssdl::make_multicolor_fgs_amp(
                    color_row_ptrs_, ampmat.get(),
                    gko::detail::get_local(dense_b),
                    gko::detail::get_local(dense_x), iter == 0, &stop_status));
            }
        } else {
            exec->run(gssdl::make_multicolor_fgs_csr(
                color_row_ptrs_, csrmat.get(), gko::detail::get_local(dense_b),
                gko::detail::get_local(dense_x), iter == 0, &stop_status));
        }

        bool one_changed = false;
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .solution(dense_x)
                .check(stopping_id, true, &stop_status, &one_changed);
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, iter, nullptr, nullptr, nullptr,
            &stop_status, all_stopped);
        if (all_stopped) {
            break;
        }
    }
}


template <typename ValueType, typename IndexType>
void FwdGaussSeidel<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                      const LinOp* b,
                                                      const LinOp* beta,
                                                      LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    mixed_precision_base_dispatch_real_complex<ValueType>(
        [this, alpha, beta](auto dense_b, auto dense_x) {
            // auto d_alpha = make_temporary_conversion<ValueType>(alpha);
            // auto d_beta = make_temporary_conversion<
            //     typename std::decay_t<decltype(*dense_x)>::value_type>(beta);
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(beta);
            dense_x->add_scaled(alpha, x_clone);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
int workspace_traits<FwdGaussSeidel<ValueType, IndexType>>::num_arrays(
    const Solver&)
{
    return 1;
}


template <typename ValueType, typename IndexType>
int workspace_traits<FwdGaussSeidel<ValueType, IndexType>>::num_vectors(
    const Solver&)
{
    return 0;
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<FwdGaussSeidel<ValueType, IndexType>>::op_names(const Solver&)
{
    return std::vector<std::string>{"fwd_gauss_seidel"};
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<FwdGaussSeidel<ValueType, IndexType>>::array_names(
    const Solver&)
{
    return {"stop"};
}


template <typename ValueType, typename IndexType>
std::vector<int>
workspace_traits<FwdGaussSeidel<ValueType, IndexType>>::scalars(const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int>
workspace_traits<FwdGaussSeidel<ValueType, IndexType>>::vectors(const Solver&)
{
    return {};
}


#define GKO_DECLARE_FWD_GS(_type, _index) class FwdGaussSeidel<_type, _index>
#define GKO_DECLARE_FWD_GS_TRAITS(_type, _index) \
    struct workspace_traits<FwdGaussSeidel<_type, _index>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(GKO_DECLARE_FWD_GS);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(GKO_DECLARE_FWD_GS_TRAITS);


}  // namespace solver
}  // namespace gko
