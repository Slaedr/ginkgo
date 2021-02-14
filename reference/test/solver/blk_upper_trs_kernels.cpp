/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/solver/blk_upper_trs.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/blk_upper_trs_kernels.hpp"
#include "core/test/solver/blk_trs_utils.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class BlkUpperTrs : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Solver = gko::solver::BlkUpperTrs<value_type, index_type>;

    std::shared_ptr<const gko::ReferenceExecutor> ref;

    BlkUpperTrs() : ref(gko::ReferenceExecutor::create()) {}

    void test_tri_system(const index_type nbrows, const int bs, const int nrhs)
    {
        const gko::testing::BlkTriSystem<value_type, index_type> tsys =
            gko::testing::get_block_tri_system<value_type, index_type>(
                ref, nbrows, bs, nrhs, false, gko::testing::UPPER_TRI);
        auto upper_trs_factory = Solver::build().on(ref);
        auto solver = upper_trs_factory->generate(tsys.tri_mtx);
        auto x = Dense::create(ref, gko::dim<2>(nbrows * bs, nrhs));

        solver->apply(tsys.b.get(), x.get());

        const double tol = 1e-9;
        GKO_ASSERT_MTX_NEAR(x, tsys.x, tol);
    }
};

using SomeTypes =
    ::testing::Types<std::tuple<double, gko::int32>,
                     std::tuple<std::complex<double>, gko::int32>>;

TYPED_TEST_SUITE(BlkUpperTrs, SomeTypes);


TYPED_TEST(BlkUpperTrs, RefUpperTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;

    gko::kernels::reference::blk_upper_trs::should_perform_transpose(
        this->ref, trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TYPED_TEST(BlkUpperTrs, SolvesTriangularSystem)
{
    this->test_tri_system(10, 7, 1);
}


TYPED_TEST(BlkUpperTrs, SolvesMultipleTriangularSystems)
{
    this->test_tri_system(20, 4, 3);
}


TYPED_TEST(BlkUpperTrs, SolvesNonUnitTriangularSystem)
{
    this->test_tri_system(10, 7, 1);
}


// TYPED_TEST(BlkUpperTrs, SolvesTriangularSystemUsingAdvancedApply)
// {
//     using Mtx = typename TestFixture::Mtx;
//     using value_type = typename TestFixture::value_type;
//     auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
//     auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
//     std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0},
//     this->exec); auto x = gko::initialize<Mtx>({1.0, -1.0, 1.0}, this->exec);
//     auto solver = this->lower_trs_factory->generate(this->mtx);

//     solver->apply(alpha.get(), b.get(), beta.get(), x.get());

//     GKO_ASSERT_MTX_NEAR(x, l({1.0, -1.0, 3.0}), r<value_type>::value);
// }


// TYPED_TEST(BlkUpperTrs, SolvesMultipleTriangularSystemsUsingAdvancedApply)
// {
//     using Mtx = typename TestFixture::Mtx;
//     using value_type = typename TestFixture::value_type;
//     using T = value_type;
//     auto alpha = gko::initialize<Mtx>({-1.0}, this->exec);
//     auto beta = gko::initialize<Mtx>({2.0}, this->exec);
//     std::shared_ptr<Mtx> b = gko::initialize<Mtx>(
//         {I<T>{3.0, 4.0}, I<T>{1.0, 0.0}, I<T>{1.0, -1.0}}, this->exec);
//     auto x = gko::initialize<Mtx>(
//         {I<T>{1.0, 2.0}, I<T>{-1.0, -1.0}, I<T>{0.0, -2.0}}, this->exec);
//     auto solver = this->lower_trs_factory_mrhs->generate(this->mtx);

//     solver->apply(alpha.get(), b.get(), beta.get(), x.get());

//     GKO_ASSERT_MTX_NEAR(x, l({{-1.0, 0.0}, {6.0, 10.0}, {-14.0, -23.0}}),
//                         r<value_type>::value);
// }


// TYPED_TEST(BlkUpperTrs, SolvesTransposedTriangularSystem)
// {
//     using Mtx = typename TestFixture::Mtx;
//     using value_type = typename TestFixture::value_type;
//     std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0},
//     this->exec); auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
//     auto solver = this->lower_trs_factory->generate(this->mtx);

//     solver->transpose()->apply(b.get(), x.get());

//     GKO_ASSERT_MTX_NEAR(x, l({0.0, 0.0, 1.0}), r<value_type>::value);
// }


// TYPED_TEST(BlkUpperTrs, SolvesConjTransposedTriangularSystem)
// {
//     using Mtx = typename TestFixture::Mtx;
//     using value_type = typename TestFixture::value_type;
//     std::shared_ptr<Mtx> b = gko::initialize<Mtx>({1.0, 2.0, 1.0},
//     this->exec); auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
//     auto solver = this->lower_trs_factory->generate(this->mtx);

//     solver->conj_transpose()->apply(b.get(), x.get());

//     GKO_ASSERT_MTX_NEAR(x, l({0.0, 0.0, 1.0}), r<value_type>::value);
// }


}  // namespace
