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

#include <ginkgo/core/base/range_accessors.hpp>


#include <gtest/gtest.h>


#include <tuple>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/types.hpp>


namespace {


class RowMajorAccessor : public ::testing::Test {
protected:
    using span = gko::span;

    using row_major_int_range = gko::range<gko::accessor::row_major<int, 2>>;

    // clang-format off
    int data[9]{
        1, 2, -1,
        3, 4, -2,
        5, 6, -3
    };
    // clang-format on
    row_major_int_range r{data, 3u, 2u, 3u};
};


TEST_F(RowMajorAccessor, CanAccessData)
{
    EXPECT_EQ(r(0, 0), 1);
    EXPECT_EQ(r(0, 1), 2);
    EXPECT_EQ(r(1, 0), 3);
    EXPECT_EQ(r(1, 1), 4);
    EXPECT_EQ(r(2, 0), 5);
    EXPECT_EQ(r(2, 1), 6);
}


TEST_F(RowMajorAccessor, CanWriteData)
{
    r(0, 0) = 4;

    EXPECT_EQ(r(0, 0), 4);
}


TEST_F(RowMajorAccessor, CanCreateSubrange)
{
    auto subr = r(span{1, 3}, span{0, 2});

    EXPECT_EQ(subr(0, 0), 3);
    EXPECT_EQ(subr(0, 1), 4);
    EXPECT_EQ(subr(1, 0), 5);
    EXPECT_EQ(subr(1, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateRowVector)
{
    auto subr = r(2, span{0, 2});

    EXPECT_EQ(subr(0, 0), 5);
    EXPECT_EQ(subr(0, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateColumnVector)
{
    auto subr = r(span{0, 3}, 0);

    EXPECT_EQ(subr(0, 0), 1);
    EXPECT_EQ(subr(1, 0), 3);
    EXPECT_EQ(subr(2, 0), 5);
}


TEST_F(RowMajorAccessor, CanAssignValues)
{
    r(1, 1) = r(0, 0);

    EXPECT_EQ(data[4], 1);
}


TEST_F(RowMajorAccessor, CanAssignSubranges)
{
    r(0, span{0, 2}) = r(1, span{0, 2});

    EXPECT_EQ(data[0], 3);
    EXPECT_EQ(data[1], 4);
    EXPECT_EQ(data[2], -1);
    EXPECT_EQ(data[3], 3);
    EXPECT_EQ(data[4], 4);
    EXPECT_EQ(data[5], -2);
    EXPECT_EQ(data[6], 5);
    EXPECT_EQ(data[7], 6);
    EXPECT_EQ(data[8], -3);
}


class ReducedStorage3d : public ::testing::Test {
protected:
    using span = gko::span;
    using ar_type = double;
    using st_type = double;

    using accessor = gko::accessor::reduced_row_major<3, ar_type, st_type>;
    using const_accessor =
        gko::accessor::reduced_row_major<3, ar_type, const st_type>;

    using reduced_storage = gko::range<accessor>;
    using const_reduced_storage = gko::range<const_accessor>;

    // clang-format off
    st_type data[4 * 3 * 2]{
        // 0, y, z
        1.0, 2.01,
        -1.02, 3.03,
        4.04, -2.05,
        // 1, y, z
        5.06, 6.07,
        2.08, 3.09,
        -1.1, -9.11,
        // 2, y, z
        -2.12, 2.13,
        0.14, 15.15,
        -9.16, 8.17,
        // 3, y, z
        7.18, -6.19,
        5.2, -4.21,
        3.22, -2.23
    };
    // clang-format on
    reduced_storage r{data, gko::dim<3>{4u, 3u, 2u}};
    const_reduced_storage cr{data, gko::dim<3>{4u, 3u, 2u}};

    template <typename Accessor>
    static void check_accessor_correctness(const Accessor &a,
                                           std::tuple<int, int, int> ignore = {
                                               99, 99, 99})
    {
        using t = std::tuple<int, int, int>;
        // clang-format off
        if (ignore != t{0, 0, 0}) { EXPECT_EQ(a(0, 0, 0), 1.0);     }
        if (ignore != t{0, 0, 1}) { EXPECT_EQ(a(0, 0, 1), 2.01);    }
        if (ignore != t{0, 1, 0}) { EXPECT_EQ(a(0, 1, 0), -1.02);   }
        if (ignore != t{0, 1, 1}) { EXPECT_EQ(a(0, 1, 1), 3.03);    }
        if (ignore != t{0, 2, 0}) { EXPECT_EQ(a(0, 2, 0), 4.04);    }
        if (ignore != t{0, 2, 1}) { EXPECT_EQ(a(0, 2, 1), -2.05);   }
        if (ignore != t{1, 0, 0}) { EXPECT_EQ(a(1, 0, 0), 5.06);    }
        if (ignore != t{1, 0, 1}) { EXPECT_EQ(a(1, 0, 1), 6.07);    }
        if (ignore != t{1, 1, 0}) { EXPECT_EQ(a(1, 1, 0), 2.08);    }
        if (ignore != t{1, 1, 1}) { EXPECT_EQ(a(1, 1, 1), 3.09);    }
        if (ignore != t{1, 2, 0}) { EXPECT_EQ(a(1, 2, 0), -1.1);    }
        if (ignore != t{1, 2, 1}) { EXPECT_EQ(a(1, 2, 1), -9.11);   }
        if (ignore != t{2, 0, 0}) { EXPECT_EQ(a(2, 0, 0), -2.12);   }
        if (ignore != t{2, 0, 1}) { EXPECT_EQ(a(2, 0, 1), 2.13);    }
        if (ignore != t{2, 1, 0}) { EXPECT_EQ(a(2, 1, 0), 0.14);    }
        if (ignore != t{2, 1, 1}) { EXPECT_EQ(a(2, 1, 1), 15.15);   }
        if (ignore != t{2, 2, 0}) { EXPECT_EQ(a(2, 2, 0), -9.16);   }
        if (ignore != t{2, 2, 1}) { EXPECT_EQ(a(2, 2, 1), 8.17);    }
        if (ignore != t{3, 0, 0}) { EXPECT_EQ(a(3, 0, 0), 7.18);    }
        if (ignore != t{3, 0, 1}) { EXPECT_EQ(a(3, 0, 1), -6.19);   }
        if (ignore != t{3, 1, 0}) { EXPECT_EQ(a(3, 1, 0), 5.2);     }
        if (ignore != t{3, 1, 1}) { EXPECT_EQ(a(3, 1, 1), -4.21);   }
        if (ignore != t{3, 2, 0}) { EXPECT_EQ(a(3, 2, 0), 3.22);    }
        if (ignore != t{3, 2, 1}) { EXPECT_EQ(a(3, 2, 1), -2.23);   }
        // clang-format on
    }
};


TEST_F(ReducedStorage3d, CanReadData)
{
    check_accessor_correctness(r);
    check_accessor_correctness(cr);
}


TEST_F(ReducedStorage3d, ToConstWorking)
{
    auto cr2 = r->to_const();

    static_assert(std::is_same<decltype(cr2), const_reduced_storage>::value,
                  "Types must be equal!");
    check_accessor_correctness(cr2);
}


TEST_F(ReducedStorage3d, CanWriteData)
{
    r(0, 1, 0) = 100.0;

    check_accessor_correctness(r, {0, 1, 0});
    EXPECT_EQ(r(0, 1, 0), 100.0);
}


TEST_F(ReducedStorage3d, CanCreateSubrange)
{
    auto subr = r(span{1, 3}, span{0, 2}, 0);

    EXPECT_EQ(subr(0, 0, 0), 5.06);
    EXPECT_EQ(subr(0, 1, 0), 2.08);
    EXPECT_EQ(subr(1, 0, 0), -2.12);
    EXPECT_EQ(subr(1, 1, 0), 0.14);
}


TEST_F(ReducedStorage3d, CanCreateSubrange2)
{
    auto subr = r(span{1, 3}, span{0, 2}, span{0, 1});

    EXPECT_EQ(subr(0, 0, 0), 5.06);
    EXPECT_EQ(subr(0, 1, 0), 2.08);
    EXPECT_EQ(subr(1, 0, 0), -2.12);
    EXPECT_EQ(subr(1, 1, 0), 0.14);
}


class ScaledReducedStorage3d : public ::testing::Test {
protected:
    using span = gko::span;
    using ar_type = double;
    using st_type = gko::int32;

    using accessor =
        gko::accessor::scaled_reduced_row_major<3, ar_type, st_type>;
    using const_accessor =
        gko::accessor::scaled_reduced_row_major<3, ar_type, const st_type>;

    using reduced_storage = gko::range<accessor>;
    using const_reduced_storage = gko::range<const_accessor>;

    // clang-format off
    st_type data[8]{
        1, 2,
        -3, 4,
        55, 6,
        -777, 8
    };
    ar_type scale[4]{
        1., 1.,
        1., 1.
    };
    // clang-format on
    reduced_storage r{data, scale, gko::dim<3>{2u, 2u, 2u}};
    const_reduced_storage cr{data, scale, gko::dim<3>{2u, 2u, 2u}};
};


TEST_F(ScaledReducedStorage3d, CanUseConst)
{
    EXPECT_EQ(cr(0, 0, 0), 1.);
    EXPECT_EQ(cr(0, 0, 1), 2.);
    EXPECT_EQ(cr(0, 1, 0), -3.);
    EXPECT_EQ(cr(0, 1, 1), 4.);
    EXPECT_EQ(cr(1, 0, 0), 55.);
    EXPECT_EQ(cr(1, 0, 1), 6.);
    EXPECT_EQ(cr(1, 1, 0), -777.);
    EXPECT_EQ(cr(1, 1, 1), 8.);

    auto subr = cr(span{0, 2}, 0, 0);

    EXPECT_EQ(subr(0, 0, 0), 1.0);
    EXPECT_EQ(subr(1, 0, 0), 55.);

    // cr(0, 0, 0) = 2.0;
    r->write_scale(0, 0, 2.);
    EXPECT_EQ(r(0, 0, 0), 2.);
}


}  // namespace
