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


#include <mpi.h>

#include <gtest/gtest.h>

#include "gtest-mpi-listener.hpp"
#include "gtest-mpi-main.hpp"


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class DistributedFcg : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Fcg<value_type>;

    DistributedFcg() : mpi_exec(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        exec = gko::ReferenceExecutor::create();
        mpi_exec = gko::MpiExecutor::create(gko::ReferenceExecutor::create());
        sub_exec = mpi_exec->get_sub_executor();
        auto comm = mpi_exec->get_communicator();
        rank = mpi_exec->get_my_rank(comm);
        ASSERT_GT(mpi_exec->get_num_ranks(comm), 1);
        mtx = gko::initialize<Mtx>(
            {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, sub_exec);
        fcg_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(30u).on(
                        mpi_exec),
                    gko::stop::ResidualNormReduction<value_type>::build()
                        .with_reduction_factor(gko::remove_complex<T>{1e-6})
                        .on(mpi_exec))
                .on(mpi_exec);
        solver = fcg_factory->generate(mtx);
    }

    void TearDown()
    {
        if (mpi_exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi_exec->synchronize());
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi_exec;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const gko::Executor> sub_exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> fcg_factory;
    std::unique_ptr<gko::LinOp> solver;
    int rank;

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        for (gko::size_type i = 0; i < m1->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < m2->get_size()[1]; ++j) {
                EXPECT_EQ(m1->at(i, j), m2->at(i, j));
            }
        }
    }
};

TYPED_TEST_CASE(DistributedFcg, gko::test::ValueTypes);


TYPED_TEST(DistributedFcg, DistributedFcgFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->fcg_factory->get_executor(), this->mpi_exec);
}


TYPED_TEST(DistributedFcg, DistributedFcgFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto fcg_solver = static_cast<Solver *>(this->solver.get());
    ASSERT_NE(fcg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(fcg_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(DistributedFcg, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->fcg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(DistributedFcg, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->fcg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(DistributedFcg, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver *>(clone.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(DistributedFcg, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver *>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(DistributedFcg, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(DistributedFcg, CanSetPreconditionerGenerator)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec),
                gko::stop::ResidualNormReduction<value_type>::build()
                    .with_reduction_factor(
                        gko::remove_complex<value_type>(1e-6))
                    .on(this->exec))
            .with_preconditioner(
                Solver::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(3u).on(
                            this->exec))
                    .on(this->exec))
            .on(this->exec);
    auto solver = fcg_factory->generate(this->mtx);
    auto precond = dynamic_cast<const gko::solver::Fcg<value_type> *>(
        static_cast<gko::solver::Fcg<value_type> *>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(DistributedFcg, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .with_generated_preconditioner(fcg_precond)
            .on(this->exec);
    auto solver = fcg_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), fcg_precond.get());
}


TYPED_TEST(DistributedFcg, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto fcg_factory = Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((fcg_factory->get_parameters().criteria).back(), init_crit);

    auto solver = fcg_factory->generate(this->mtx);
    std::shared_ptr<gko::stop::CriterionFactory> new_crit =
        gko::stop::Iteration::build().with_max_iters(5u).on(this->exec);

    solver->set_stop_criterion_factory(new_crit);
    auto new_crit_fac = solver->get_stop_criterion_factory();
    auto niter =
        static_cast<const gko::stop::Iteration::Factory *>(new_crit_fac.get())
            ->get_parameters()
            .max_iters;

    ASSERT_EQ(niter, 5);
}


TYPED_TEST(DistributedFcg, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{2, 2});
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .with_generated_preconditioner(fcg_precond)
            .on(this->exec);

    ASSERT_THROW(fcg_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(DistributedFcg, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->fcg_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(DistributedFcg, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    auto fcg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec);
    auto solver = fcg_factory->generate(this->mtx);
    solver->set_preconditioner(fcg_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), fcg_precond.get());
}


TYPED_TEST(DistributedFcg, CanSolveIndependentLocalSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->sub_exec)
            ->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->sub_exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->sub_exec);

    auto fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->sub_exec);
    auto solver = fcg_factory->generate(this->mtx);
    solver->set_preconditioner(fcg_precond);
    auto precond = solver->get_preconditioner();
    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(DistributedFcg, CanSolveDistributedSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    using Solver = typename TestFixture::Solver;
    gko::IndexSet<size_type> row_dist{4};
    if (this->rank == 0) {
        row_dist.add_index(0);
        row_dist.add_index(2);
    } else {
        row_dist.add_index(1);
    }
    std::shared_ptr<Mtx> dist_mtx = gko::initialize_and_distribute<Mtx>(
        row_dist, {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}},
        this->mpi_exec);
    auto b = gko::initialize_and_distribute<Mtx>(1, row_dist, {-1.0, 3.0, 1.0},
                                                 this->mpi_exec);
    auto x = gko::initialize_and_distribute<Mtx>(1, row_dist, {0.0, 0.0, 0.0},
                                                 this->mpi_exec);

    auto fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->mpi_exec);
    auto solver = fcg_factory->generate(dist_mtx);
    solver->apply(b.get(), x.get());

    if (this->rank == 0) {
        GKO_ASSERT_MTX_NEAR(x, l({1.0, 2.0}), r<value_type>::value);
    } else {
        GKO_ASSERT_MTX_NEAR(x, l({3.0}), r<value_type>::value);
    }
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
