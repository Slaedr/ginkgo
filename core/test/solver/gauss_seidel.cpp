// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/solver/gauss_seidel.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class FwdGaussSeidel : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Ell<value_type, index_type>;
    using Solver = gko::solver::FwdGaussSeidel<value_type, index_type>;

    FwdGaussSeidel()
        : exec(gko::ReferenceExecutor::create()),
          // 4x4 symmetric positive-definite matrix (two-color ordering):
          //   color 0: rows 0, 1  color 1: rows 2, 3
          mtx(gko::initialize<Mtx>(
              // clang-format off
              {{2.0, 0.0, 1.0, 0.0},
               {0.0, 3.0, 0.0, 1.0},
               {1.0, 0.0, 4.0, 0.0},
               {0.0, 1.0, 0.0, 5.0}},
              // clang-format on
              exec)),
          gs_factory(Solver::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(1u))
                         .with_color_ptrs(std::vector<index_type>{0, 2, 4})
                         .on(exec)),
          solver(gs_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> gs_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(FwdGaussSeidel, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


TYPED_TEST(FwdGaussSeidel, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->gs_factory->get_executor(), this->exec);
}


TYPED_TEST(FwdGaussSeidel, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(4, 4));
    auto gs = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(gs->get_system_matrix(), nullptr);
    ASSERT_EQ(gs->get_system_matrix(), this->mtx);
}


TYPED_TEST(FwdGaussSeidel, ApplyUsesInitialGuessReturnsFalseByDefault)
{
    // Default init_guess_mode is zero, so the output x is not used as a guess.
    ASSERT_FALSE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(FwdGaussSeidel, ApplyUsesInitialGuessReturnsTrueWhenModeIsProvided)
{
    using Solver = typename TestFixture::Solver;
    using index_type = typename TestFixture::index_type;

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .with_init_guess_mode(gko::solver::initial_guess_mode::provided)
            .on(this->exec)
            ->generate(this->mtx);

    ASSERT_TRUE(solver->apply_uses_initial_guess());
}


TYPED_TEST(FwdGaussSeidel, ApplyUsesInitialGuessReturnsFalseWhenModeIsRhs)
{
    using Solver = typename TestFixture::Solver;
    using index_type = typename TestFixture::index_type;

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .with_init_guess_mode(gko::solver::initial_guess_mode::rhs)
            .on(this->exec)
            ->generate(this->mtx);

    ASSERT_FALSE(solver->apply_uses_initial_guess());
}


TYPED_TEST(FwdGaussSeidel, DefaultInitGuessModeIsZero)
{
    using Solver = typename TestFixture::Solver;

    auto gs = static_cast<const Solver*>(this->solver.get());

    ASSERT_EQ(gs->get_parameters().init_guess_mode,
              gko::solver::initial_guess_mode::zero);
}


// Zero mode: any initial x is overwritten with zeros, so the result is
// independent of the starting vector.
TYPED_TEST(FwdGaussSeidel, ZeroModeIgnoresInitialX)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;

    auto b = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x_from_zero = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto x_from_nonzero =
        gko::initialize<Vec>({10.0, -5.0, 3.0, 7.0}, this->exec);

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .with_init_guess_mode(gko::solver::initial_guess_mode::zero)
            .on(this->exec)
            ->generate(this->mtx);

    solver->apply(b, x_from_zero);
    solver->apply(b, x_from_nonzero);

    GKO_ASSERT_MTX_NEAR(x_from_nonzero, x_from_zero, 0.0);
}


// Provided mode: the exact solution is a fixed point of Gauss-Seidel, so
// starting from it should leave x unchanged after any number of iterations.
TYPED_TEST(FwdGaussSeidel, ProvidedModeKeepsExactSolutionAFixedPoint)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;

    // x* = {4/7, -1, 6/7, 0} is the exact solution for b = {2, -3, 4, -1}.
    auto b = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<Vec>({value_type{4.0 / 7.0}, value_type{-1.0},
                                   value_type{6.0 / 7.0}, value_type{0.0}},
                                  this->exec);
    auto exact = x->clone();

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .with_init_guess_mode(gko::solver::initial_guess_mode::provided)
            .on(this->exec)
            ->generate(this->mtx);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, exact, 1e-6);
}


// Rhs mode: x is initialised to b before iterating. Applying with rhs mode
// must give the same result as explicitly setting x = b and using provided
// mode.
TYPED_TEST(FwdGaussSeidel, RhsModeMatchesProvidedModeWithXEqualToB)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;

    auto b = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);

    // Run with rhs mode (x will be set to b internally).
    auto x_rhs = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto solver_rhs =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .with_init_guess_mode(gko::solver::initial_guess_mode::rhs)
            .on(this->exec)
            ->generate(this->mtx);
    solver_rhs->apply(b, x_rhs);

    // Run with provided mode, starting x explicitly from b.
    auto x_provided = b->clone();
    auto solver_provided =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .with_init_guess_mode(gko::solver::initial_guess_mode::provided)
            .on(this->exec)
            ->generate(this->mtx);
    solver_provided->apply(b, x_provided);

    GKO_ASSERT_MTX_NEAR(x_rhs, x_provided, 0.0);
}


TYPED_TEST(FwdGaussSeidel, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto empty = this->gs_factory->generate(Mtx::create(this->exec));

    empty->copy_from(this->solver);

    ASSERT_EQ(empty->get_size(), gko::dim<2>(4, 4));
    auto copy_mtx = static_cast<Solver*>(empty.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
}


TYPED_TEST(FwdGaussSeidel, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto empty = this->gs_factory->generate(Mtx::create(this->exec));

    empty->move_from(this->solver);

    ASSERT_EQ(empty->get_size(), gko::dim<2>(4, 4));
    auto moved_mtx = static_cast<Solver*>(empty.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(moved_mtx), this->mtx, 0.0);
}


TYPED_TEST(FwdGaussSeidel, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;

    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(4, 4));
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_mtx), this->mtx, 0.0);
}


TYPED_TEST(FwdGaussSeidel, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(static_cast<Solver*>(this->solver.get())->get_system_matrix(),
              nullptr);
}


TYPED_TEST(FwdGaussSeidel, ColorPtrsAreStoredFromParameters)
{
    using Solver = typename TestFixture::Solver;
    using index_type = typename TestFixture::index_type;

    auto gs = static_cast<const Solver*>(this->solver.get());
    const auto& stored = gs->get_parameters().color_ptrs;

    ASSERT_EQ(stored.size(), 3u);
    EXPECT_EQ(stored[0], index_type{0});
    EXPECT_EQ(stored[1], index_type{2});
    EXPECT_EQ(stored[2], index_type{4});
}


TYPED_TEST(FwdGaussSeidel, IterationConvergesTowardKnownExactSolution)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;

    // RHS with mixed signs; exact solution x* = {4/7, -1, 6/7, 0}.
    // Error in x[0] contracts by factor 1/8 per sweep: after 5 iterations
    // max component error ≈ 1e-4, well within the 1e-3 tolerance below.
    auto b = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto exact = gko::initialize<Vec>({value_type{4.0 / 7.0}, value_type{-1.0},
                                       value_type{6.0 / 7.0}, value_type{0.0}},
                                      this->exec);

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .on(this->exec)
            ->generate(this->mtx);
    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, exact, 1e-3);
}


TYPED_TEST(FwdGaussSeidel, ApplySupportsVectorsOfDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;
    auto b = gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<OtherVec>({0.0, 0.0, 0.0, 0.0}, this->exec);

    ASSERT_NO_THROW(this->solver->apply(b, x));
}


TYPED_TEST(FwdGaussSeidel, AdvancedApplySupportsVectorsOfDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;
    auto alpha = gko::initialize<OtherVec>({2.0}, this->exec);
    auto beta = gko::initialize<OtherVec>({-1.0}, this->exec);
    auto b = gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<OtherVec>({1.0, 1.0, 1.0, 1.0}, this->exec);

    ASSERT_NO_THROW(this->solver->apply(alpha, b, beta, x));
}


TYPED_TEST(FwdGaussSeidel, ApplyWithMixedPrecisionVectorsProducesCorrectResult)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .on(this->exec)
            ->generate(this->mtx);

    // Apply with same-precision vectors
    auto b_same = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x_same = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    solver->apply(b_same, x_same);

    // Apply with different-precision vectors
    auto b_other =
        gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x_other = gko::initialize<OtherVec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    solver->apply(b_other, x_other);

    GKO_ASSERT_MTX_NEAR(x_same, x_other, 1e-5);
}


template <typename ValueIndexType>
class FwdGaussSeidelAMP : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using EllMtx = gko::matrix::Ell<value_type, index_type>;
    using AMPMtx = gko::matrix::AMP<value_type, index_type>;
    using Solver = gko::solver::FwdGaussSeidel<value_type, index_type>;

    FwdGaussSeidelAMP()
        : exec(gko::ReferenceExecutor::create()),
          mtx(AMPMtx::build().with_tolerance(1e-6f).on(exec)->generate(
              gko::share(gko::initialize<EllMtx>(
                  // clang-format off
                      {{2.0, 0.0, 1.0, 0.0},
                       {0.0, 3.0, 0.0, 1.0},
                       {1.0, 0.0, 4.0, 0.0},
                       {0.0, 1.0, 0.0, 5.0}},
                  // clang-format on
                  exec)))),
          gs_factory(Solver::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(1u))
                         .with_color_ptrs(std::vector<index_type>{0, 2, 4})
                         .on(exec)),
          solver(gs_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<AMPMtx> mtx;
    std::unique_ptr<typename Solver::Factory> gs_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(FwdGaussSeidelAMP, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


TYPED_TEST(FwdGaussSeidelAMP, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    using AMPMtx = typename TestFixture::AMPMtx;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(4, 4));
    auto gs = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(gs->get_system_matrix(), nullptr);
    ASSERT_EQ(gs->get_system_matrix(), this->mtx);
}


TYPED_TEST(FwdGaussSeidelAMP, IterationConvergesTowardKnownExactSolution)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;

    // Same system as the ELL test: A x = b, x* = {4/7, -1, 6/7, 0}.
    auto b = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto exact = gko::initialize<Vec>({value_type{4.0 / 7.0}, value_type{-1.0},
                                       value_type{6.0 / 7.0}, value_type{0.0}},
                                      this->exec);

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .on(this->exec)
            ->generate(this->mtx);
    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, exact, 1e-3);
}


TYPED_TEST(FwdGaussSeidelAMP, ApplySupportsVectorsOfDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;
    auto b = gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<OtherVec>({0.0, 0.0, 0.0, 0.0}, this->exec);

    ASSERT_NO_THROW(this->solver->apply(b, x));
}


TYPED_TEST(FwdGaussSeidelAMP, AdvancedApplySupportsVectorsOfDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;
    auto alpha = gko::initialize<OtherVec>({2.0}, this->exec);
    auto beta = gko::initialize<OtherVec>({-1.0}, this->exec);
    auto b = gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<OtherVec>({1.0, 1.0, 1.0, 1.0}, this->exec);

    ASSERT_NO_THROW(this->solver->apply(alpha, b, beta, x));
}


TYPED_TEST(FwdGaussSeidelAMP,
           ApplyWithMixedPrecisionVectorsProducesCorrectResult)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .on(this->exec)
            ->generate(this->mtx);

    auto b_same = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x_same = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    solver->apply(b_same, x_same);

    auto b_other =
        gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x_other = gko::initialize<OtherVec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    solver->apply(b_other, x_other);

    GKO_ASSERT_MTX_NEAR(x_same, x_other, 1e-5);
}


// ============================================================
// FwdGaussSeidel with AMP-CSR system matrix
// ============================================================

template <typename ValueIndexType>
class FwdGaussSeidelAMPCSR : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using AMPMtx = gko::matrix::AMP<value_type, index_type>;
    using Solver = gko::solver::FwdGaussSeidel<value_type, index_type>;

    FwdGaussSeidelAMPCSR()
        : exec(gko::ReferenceExecutor::create()),
          mtx(AMPMtx::build().with_tolerance(1e-6f).on(exec)->generate(
              gko::share(gko::initialize<CsrMtx>(
                  // clang-format off
                      {{2.0, 0.0, 1.0, 0.0},
                       {0.0, 3.0, 0.0, 1.0},
                       {1.0, 0.0, 4.0, 0.0},
                       {0.0, 1.0, 0.0, 5.0}},
                  // clang-format on
                  exec)))),
          gs_factory(Solver::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(1u))
                         .with_color_ptrs(std::vector<index_type>{0, 2, 4})
                         .on(exec)),
          solver(gs_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<AMPMtx> mtx;
    std::unique_ptr<typename Solver::Factory> gs_factory;
    std::unique_ptr<gko::LinOp> solver;
};

TYPED_TEST_SUITE(FwdGaussSeidelAMPCSR, gko::test::ValueIndexTypesBase,
                 PairTypenameNameGenerator);


TYPED_TEST(FwdGaussSeidelAMPCSR, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    using AMPMtx = typename TestFixture::AMPMtx;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(4, 4));
    auto gs = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(gs->get_system_matrix(), nullptr);
    ASSERT_EQ(gs->get_system_matrix(), this->mtx);
}


TYPED_TEST(FwdGaussSeidelAMPCSR, IterationConvergesTowardKnownExactSolution)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;

    // Same system as the ELL test: A x = b, x* = {4/7, -1, 6/7, 0}.
    auto b = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    auto exact = gko::initialize<Vec>({value_type{4.0 / 7.0}, value_type{-1.0},
                                       value_type{6.0 / 7.0}, value_type{0.0}},
                                      this->exec);

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .on(this->exec)
            ->generate(this->mtx);
    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, exact, 1e-3);
}


TYPED_TEST(FwdGaussSeidelAMPCSR, ApplySupportsVectorsOfDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;
    auto b = gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<OtherVec>({0.0, 0.0, 0.0, 0.0}, this->exec);

    ASSERT_NO_THROW(this->solver->apply(b, x));
}


TYPED_TEST(FwdGaussSeidelAMPCSR,
           AdvancedApplySupportsVectorsOfDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;
    auto alpha = gko::initialize<OtherVec>({2.0}, this->exec);
    auto beta = gko::initialize<OtherVec>({-1.0}, this->exec);
    auto b = gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x = gko::initialize<OtherVec>({1.0, 1.0, 1.0, 1.0}, this->exec);

    ASSERT_NO_THROW(this->solver->apply(alpha, b, beta, x));
}


TYPED_TEST(FwdGaussSeidelAMPCSR,
           ApplyWithMixedPrecisionVectorsProducesCorrectResult)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Solver = typename TestFixture::Solver;
    using Vec = gko::matrix::Dense<value_type>;
    using other_type =
        typename gko::detail::next_precision_base_impl<value_type>::type;
    using OtherVec = gko::matrix::Dense<other_type>;

    auto solver =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(5u))
            .with_color_ptrs(std::vector<index_type>{0, 2, 4})
            .on(this->exec)
            ->generate(this->mtx);

    auto b_same = gko::initialize<Vec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x_same = gko::initialize<Vec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    solver->apply(b_same, x_same);

    auto b_other =
        gko::initialize<OtherVec>({2.0, -3.0, 4.0, -1.0}, this->exec);
    auto x_other = gko::initialize<OtherVec>({0.0, 0.0, 0.0, 0.0}, this->exec);
    solver->apply(b_other, x_other);

    GKO_ASSERT_MTX_NEAR(x_same, x_other, 1e-5);
}


}  // namespace
