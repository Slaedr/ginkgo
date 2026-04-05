// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/distributed/solver/gauss_seidel.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/solver/gauss_seidel.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"


namespace {


using value_type = double;
using local_index_type = gko::int32;
using global_index_type = gko::int64;
using dist_mtx_type =
    gko::experimental::distributed::Matrix<value_type, local_index_type,
                                           global_index_type>;
using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
using part_type = gko::experimental::distributed::Partition<local_index_type,
                                                            global_index_type>;
using Dense = gko::matrix::Dense<value_type>;
using Ell = gko::matrix::Ell<value_type, local_index_type>;
using Csr = gko::matrix::Csr<value_type, local_index_type>;
using Gmres = gko::solver::Gmres<value_type>;
using Fgs = gko::solver::FwdGaussSeidel<value_type, local_index_type>;
using DistFgs = gko::experimental::distributed::solver::FwdGaussSeidel<
    value_type, local_index_type, global_index_type>;
using Schwarz = gko::experimental::distributed::preconditioner::Schwarz<
    value_type, local_index_type, global_index_type>;


/**
 * Generates a 3D 27-point stencil subdomain with 8-color ordering.
 * Simplified version of the benchmark stencil for testing.
 */
std::pair<gko::matrix_data<value_type, global_index_type>,
          std::vector<local_index_type>>
generate_test_stencil(local_index_type gx, local_index_type gy,
                      local_index_type gz, local_index_type oz,
                      local_index_type lz, global_index_type my_row_offset,
                      const std::vector<global_index_type>& rank_row_offsets,
                      const std::vector<local_index_type>& rank_z_offsets,
                      const std::vector<local_index_type>& rank_z_sizes,
                      int nranks)
{
    const auto lx = gx, ly = gy;
    const auto local_n = static_cast<global_index_type>(lx) * ly * lz;
    const auto global_n = static_cast<global_index_type>(gx) * gy * gz;

    // Build color permutation
    std::vector<local_index_type> old_to_new(local_n);
    std::array<local_index_type, 8> cnt{};
    for (local_index_type k = 0; k < lz; ++k)
        for (local_index_type j = 0; j < ly; ++j)
            for (local_index_type i = 0; i < lx; ++i)
                ++cnt[(i % 2) + 2 * (j % 2) + 4 * (k % 2)];

    std::vector<local_index_type> color_ptrs(9);
    color_ptrs[0] = 0;
    for (int c = 0; c < 8; ++c) color_ptrs[c + 1] = color_ptrs[c] + cnt[c];

    std::array<local_index_type, 8> fill{};
    for (local_index_type k = 0; k < lz; ++k)
        for (local_index_type j = 0; j < ly; ++j)
            for (local_index_type i = 0; i < lx; ++i) {
                auto local_idx = i + j * lx + k * lx * ly;
                int color = (i % 2) + 2 * (j % 2) + 4 * (k % 2);
                old_to_new[local_idx] = color_ptrs[color] + fill[color]++;
            }

    std::vector<local_index_type> new_to_old(local_n);
    for (local_index_type i = 0; i < local_n; ++i)
        new_to_old[old_to_new[i]] = i;

    // Helper: compute color-ordered global index for any grid point
    auto color_global_idx = [&](local_index_type gi, local_index_type gj,
                                local_index_type gk) -> global_index_type {
        for (int r = 0; r < nranks; ++r) {
            auto r_oz = rank_z_offsets[r];
            auto r_nz = rank_z_sizes[r];
            if (gk >= r_oz && gk < r_oz + r_nz) {
                auto li = gi, lj = gj, lk = gk - r_oz;
                int color = (li % 2) + 2 * (lj % 2) + 4 * (lk % 2);
                std::array<local_index_type, 8> r_cnt{};
                for (local_index_type kk = 0; kk < r_nz; ++kk)
                    for (local_index_type jj = 0; jj < ly; ++jj)
                        for (local_index_type ii = 0; ii < lx; ++ii)
                            ++r_cnt[(ii % 2) + 2 * (jj % 2) + 4 * (kk % 2)];
                std::array<local_index_type, 9> r_offsets{};
                for (int c = 0; c < 8; ++c)
                    r_offsets[c + 1] = r_offsets[c] + r_cnt[c];
                local_index_type pos = 0;
                auto local_nat = li + lj * lx + lk * lx * ly;
                for (local_index_type kk = 0; kk < r_nz; ++kk)
                    for (local_index_type jj = 0; jj < ly; ++jj)
                        for (local_index_type ii = 0; ii < lx; ++ii) {
                            if ((ii % 2) + 2 * (jj % 2) + 4 * (kk % 2) != color)
                                continue;
                            auto nat = ii + jj * lx + kk * lx * ly;
                            if (nat < local_nat) ++pos;
                        }
                return rank_row_offsets[r] + r_offsets[color] + pos;
            }
        }
        return global_index_type{-1};
    };

    gko::matrix_data<value_type, global_index_type> data(
        gko::dim<2>{static_cast<gko::size_type>(global_n),
                    static_cast<gko::size_type>(global_n)});
    data.nonzeros.reserve(27 * static_cast<std::size_t>(local_n));

    for (local_index_type new_local = 0; new_local < local_n; ++new_local) {
        auto old_local = new_to_old[new_local];
        auto li = old_local % lx;
        auto lj = (old_local / lx) % ly;
        auto lk = old_local / (lx * ly);
        auto gi = li, gj = lj, gk = oz + lk;

        auto global_row = my_row_offset + new_local;
        double offdiag_sum = 0.0;

        for (local_index_type dk = -1; dk <= 1; ++dk)
            for (local_index_type dj = -1; dj <= 1; ++dj)
                for (local_index_type di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0 && dk == 0) continue;
                    auto ni = gi + di, nj = gj + dj, nk = gk + dk;
                    if (ni < 0 || ni >= gx || nj < 0 || nj >= gy || nk < 0 ||
                        nk >= gz)
                        continue;
                    global_index_type global_col;
                    auto nli = ni, nlj = nj, nlk = nk - oz;
                    if (nlk >= 0 && nlk < lz) {
                        auto neighbor_nat = nli + nlj * lx + nlk * lx * ly;
                        global_col = my_row_offset + old_to_new[neighbor_nat];
                    } else {
                        global_col = color_global_idx(ni, nj, nk);
                    }
                    double val = -0.01;
                    offdiag_sum += std::abs(val);
                    data.nonzeros.emplace_back(global_row, global_col, val);
                }

        data.nonzeros.emplace_back(global_row, global_row, 1.1 * offdiag_sum);
    }

    data.sort_row_major();
    return {std::move(data), std::move(color_ptrs)};
}


/**
 * Helper struct to set up a distributed 3D stencil problem.
 */
struct StencilProblem {
    std::shared_ptr<dist_mtx_type> A;
    std::shared_ptr<dist_vec_type> b;
    std::shared_ptr<dist_vec_type> ones;
    std::shared_ptr<part_type> partition;
    std::vector<local_index_type> color_ptrs;
    gko::experimental::mpi::communicator comm;
    std::shared_ptr<const gko::Executor> exec;

    StencilProblem(std::shared_ptr<const gko::Executor> exec_,
                   gko::experimental::mpi::communicator comm_,
                   local_index_type grid_dim, const gko::LinOp* local_template)
        : exec(exec_), comm(comm_)
    {
        const auto rank = comm.rank();
        const auto nranks = comm.size();
        const auto global_n =
            static_cast<global_index_type>(grid_dim) * grid_dim * grid_dim;
        const auto local_nz_base = grid_dim / nranks;
        const auto remainder = grid_dim % nranks;
        const auto my_nz = local_nz_base + (rank < remainder ? 1 : 0);
        local_index_type my_oz =
            local_nz_base * rank + std::min(rank, static_cast<int>(remainder));
        const auto local_n =
            static_cast<global_index_type>(grid_dim) * grid_dim * my_nz;

        global_index_type global_row_offset = 0;
        for (int r = 0; r < rank; ++r) {
            auto r_nz = local_nz_base + (r < remainder ? 1 : 0);
            global_row_offset +=
                static_cast<global_index_type>(grid_dim) * grid_dim * r_nz;
        }

        std::vector<local_index_type> rank_z_offsets(nranks);
        std::vector<local_index_type> rank_z_sizes(nranks);
        std::vector<global_index_type> rank_row_offsets(nranks);
        global_index_type offset = 0;
        for (int r = 0; r < nranks; ++r) {
            auto r_nz = local_nz_base + (r < remainder ? 1 : 0);
            rank_z_offsets[r] =
                local_nz_base * r + std::min(r, static_cast<int>(remainder));
            rank_z_sizes[r] = r_nz;
            rank_row_offsets[r] = offset;
            offset +=
                static_cast<global_index_type>(grid_dim) * grid_dim * r_nz;
        }

        auto [data, cptrs] = generate_test_stencil(
            grid_dim, grid_dim, grid_dim, my_oz, my_nz, global_row_offset,
            rank_row_offsets, rank_z_offsets, rank_z_sizes, nranks);
        color_ptrs = std::move(cptrs);

        std::vector<global_index_type> ranges_vec(nranks + 1);
        ranges_vec[0] = 0;
        for (int r = 0; r < nranks; ++r) {
            auto r_nz = local_nz_base + (r < remainder ? 1 : 0);
            ranges_vec[r + 1] =
                ranges_vec[r] +
                static_cast<global_index_type>(grid_dim) * grid_dim * r_nz;
        }
        gko::array<global_index_type> ranges{exec, ranges_vec.begin(),
                                             ranges_vec.end()};
        partition = gko::share(part_type::build_from_contiguous(exec, ranges));

        auto csr_template = Csr::create(exec);
        A = gko::share(dist_mtx_type::create(exec, comm, local_template,
                                             csr_template.get()));
        A->read_distributed(data, partition);

        // RHS: b = A * ones
        ones = dist_vec_type::create(exec, comm);
        gko::matrix_data<value_type, global_index_type> ones_data;
        ones_data.size = {static_cast<gko::size_type>(global_n), 1};
        for (global_index_type i = global_row_offset;
             i < global_row_offset + local_n; ++i) {
            ones_data.nonzeros.emplace_back(i, global_index_type{0}, 1.0);
        }
        ones->read_distributed(ones_data, partition);
        b = dist_vec_type::create(exec, comm);
        gko::matrix_data<value_type, global_index_type> zero_data;
        zero_data.size = ones_data.size;
        for (global_index_type i = global_row_offset;
             i < global_row_offset + local_n; ++i) {
            zero_data.nonzeros.emplace_back(i, global_index_type{0}, 0.0);
        }
        b->read_distributed(zero_data, partition);
        A->apply(ones, b);
    }
};


class DistFwdGaussSeidel : public ::testing::Test {
protected:
    DistFwdGaussSeidel()
        : ref(gko::ReferenceExecutor::create()),
          comm(gko::experimental::mpi::communicator(MPI_COMM_WORLD))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::experimental::mpi::communicator comm;
};


TEST_F(DistFwdGaussSeidel, SingleRankMatchesLocalFGS)
{
    if (comm.size() != 1) {
        GTEST_SKIP() << "This test requires exactly 1 MPI rank";
    }

    // Create a small 4x4 diagonally-dominant matrix in ELL format
    auto ell_template = Ell::create(ref);
    StencilProblem problem(ref, comm, 4, ell_template.get());

    // Run local FGS
    auto local_fgs =
        Fgs::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_color_ptrs(problem.color_ptrs)
            .on(ref)
            ->generate(problem.A->get_local_matrix());
    auto x_local =
        Dense::create(ref, problem.b->get_local_vector()->get_size());
    x_local->fill(0.0);
    local_fgs->apply(problem.b->get_local_vector(), x_local);

    // Run distributed FGS
    auto dist_fgs =
        DistFgs::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_color_ptrs(problem.color_ptrs)
            .on(ref)
            ->generate(problem.A);
    auto x_dist = gko::clone(ref, problem.b);
    x_dist->fill(0.0);
    dist_fgs->apply(problem.b, x_dist);

    // Compare: should be identical for single rank
    auto x_dist_local = x_dist->get_local_vector();
    for (gko::size_type i = 0; i < x_local->get_size()[0]; ++i) {
        EXPECT_NEAR(x_local->at(i, 0), x_dist_local->at(i, 0), 1e-14)
            << "Mismatch at row " << i;
    }
}


TEST_F(DistFwdGaussSeidel, ConvergesWithMultipleRanks)
{
    auto ell_template = Ell::create(ref);
    StencilProblem problem(ref, comm, 6, ell_template.get());

    auto solver =
        Gmres::build()
            .with_krylov_dim(gko::size_type{50})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-8))
            .with_preconditioner(
                DistFgs::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .with_color_ptrs(problem.color_ptrs))
            .on(ref)
            ->generate(problem.A);

    auto x = gko::clone(ref, problem.b);
    x->fill(0.0);

    auto logger = gko::share(gko::log::Convergence<value_type>::create());
    solver->add_logger(logger);
    solver->apply(problem.b, x);
    solver->remove_logger(logger);

    ASSERT_TRUE(logger->has_converged());
}


TEST_F(DistFwdGaussSeidel, FewerItersThanSchwarz)
{
    auto ell_template = Ell::create(ref);
    StencilProblem problem(ref, comm, 6, ell_template.get());

    // Distributed FGS preconditioner
    auto solver_dist =
        Gmres::build()
            .with_krylov_dim(gko::size_type{50})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-8))
            .with_preconditioner(
                DistFgs::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .with_color_ptrs(problem.color_ptrs))
            .on(ref)
            ->generate(problem.A);

    auto x1 = gko::clone(ref, problem.b);
    x1->fill(0.0);
    auto logger1 = gko::share(gko::log::Convergence<value_type>::create());
    solver_dist->add_logger(logger1);
    solver_dist->apply(problem.b, x1);
    solver_dist->remove_logger(logger1);

    // Schwarz + local FGS preconditioner
    auto solver_schwarz =
        Gmres::build()
            .with_krylov_dim(gko::size_type{50})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-8))
            .with_preconditioner(Schwarz::build().with_local_solver(
                Fgs::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .with_color_ptrs(problem.color_ptrs)))
            .on(ref)
            ->generate(problem.A);

    auto x2 = gko::clone(ref, problem.b);
    x2->fill(0.0);
    auto logger2 = gko::share(gko::log::Convergence<value_type>::create());
    solver_schwarz->add_logger(logger2);
    solver_schwarz->apply(problem.b, x2);
    solver_schwarz->remove_logger(logger2);

    ASSERT_TRUE(logger1->has_converged());
    ASSERT_TRUE(logger2->has_converged());
    // Distributed FGS should need fewer or equal GMRES iterations
    EXPECT_LE(logger1->get_num_iterations(), logger2->get_num_iterations());
}


TEST_F(DistFwdGaussSeidel, WorksWithCSR)
{
    auto csr_template = Csr::create(ref);
    StencilProblem problem(ref, comm, 6, csr_template.get());

    auto solver =
        Gmres::build()
            .with_krylov_dim(gko::size_type{50})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(1e-8))
            .with_preconditioner(
                DistFgs::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .with_color_ptrs(problem.color_ptrs))
            .on(ref)
            ->generate(problem.A);

    auto x = gko::clone(ref, problem.b);
    x->fill(0.0);
    auto logger = gko::share(gko::log::Convergence<value_type>::create());
    solver->add_logger(logger);
    solver->apply(problem.b, x);
    solver->remove_logger(logger);

    ASSERT_TRUE(logger->has_converged());
}


}  // namespace
