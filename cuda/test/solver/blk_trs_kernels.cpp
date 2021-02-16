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

#include <ginkgo/core/solver/blk_lower_trs.hpp>
#include <ginkgo/core/solver/blk_upper_trs.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/solver/blk_lower_trs_kernels.hpp"
#include "core/solver/blk_upper_trs_kernels.hpp"
#include "core/test/solver/blk_trs_utils.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class BlkLowerTrs : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using LSolver = gko::solver::BlkLowerTrs<value_type, index_type>;
    using USolver = gko::solver::BlkUpperTrs<value_type, index_type>;

    std::shared_ptr<const gko::CudaExecutor> cuda;

    BlkLowerTrs() {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        auto ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    /* Generates a diagonal-dominant triangular system for testing.
     * With a random non-diagonal-dominant system
     * with 10 block rows and 7x7 blocks, (using cuSparse bsrsm)
     * using complex double value type leads to accumulating errors.
     * For a lower-triangular solve, at the first couple of block-rows,
     * the rel. soln. error is about 1e-16.
     * But the relative error in the last block-row is about 1.0.
     * On the other hand, a random but diagonal dominant matrix of
     * the same sizes is well-behaved.
     */
    void test_tri_system(const index_type nbrows, const int bs, const int nrhs,
                         const bool diag_identity,
                         const gko::testing::TriSystemType type,
                         const bool strict_triangular)
    {
        const gko::testing::BlkTriSystem<value_type, index_type> tsys =
            gko::testing::get_block_tri_system<value_type, index_type>(
                cuda, nbrows, bs, nrhs, diag_identity, type, true,
                strict_triangular);
        auto lower_trs_factory =
            LSolver::build().with_diag_identity(diag_identity).on(cuda);
        auto upper_trs_factory = USolver::build().on(cuda);
        std::shared_ptr<gko::LinOp> solver;
        if (type == gko::testing::LOWER_TRI)
            solver = lower_trs_factory->generate(tsys.tri_mtx);
        else
            solver = upper_trs_factory->generate(tsys.tri_mtx);
        auto x = Dense::create(cuda, gko::dim<2>(nbrows * bs, nrhs));

        solver->apply(tsys.b.get(), x.get());

        const double tol =
            50 *
            std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
        GKO_ASSERT_MTX_NEAR(x, tsys.x, tol);
    }
};

using SomeTypes =
    ::testing::Types<std::tuple<double, gko::int32>,
                     std::tuple<std::complex<double>, gko::int32>>;

TYPED_TEST_SUITE(BlkLowerTrs, SomeTypes);


TYPED_TEST(BlkLowerTrs, RefLowerTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;

    gko::kernels::cuda::blk_lower_trs::should_perform_transpose(this->cuda,
                                                                trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TYPED_TEST(BlkLowerTrs, SolvesUnitLowerBlockTriangularSystem)
{
    const bool diag_identity = true;
    const bool strict_triangular = false;

    this->test_tri_system(10, 7, 1, diag_identity, gko::testing::LOWER_TRI,
                          strict_triangular);
}


TYPED_TEST(BlkLowerTrs, SolvesNonUnitLowerBlockTriangularSystem)
{
    // test with lower triangular matrix, not block-triangular
    const bool diag_identity = false;
    const bool strict_triangular = true;

    this->test_tri_system(10, 7, 1, diag_identity, gko::testing::LOWER_TRI,
                          strict_triangular);
}


TYPED_TEST(BlkLowerTrs, SolvesUpperBlockTriangularSystem)
{
    // test only works with upper triangular matrix, not block-triangular
    const bool diag_identity = false;
    const bool strict_triangular = true;

    this->test_tri_system(10, 7, 1, diag_identity, gko::testing::UPPER_TRI,
                          strict_triangular);
}


}  // namespace
