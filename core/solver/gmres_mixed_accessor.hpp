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

#ifndef GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_
#define GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_


#include <cinttypes>
#include <limits>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#include <iostream>
#define GKO_DEBUG_OUTPUT false


namespace gko {
namespace kernels {  // TODO maybe put into another separate namespace


namespace detail {


template <typename Accessor>
struct is_3d_scaled_accessor : public std::false_type {
};

template <typename... Args>
struct is_3d_scaled_accessor<accessor::ScaledReducedStorage3d<Args...>>
    : public std::true_type {
};

template <typename StorageType, bool = std::is_integral<StorageType>::value>
struct helper_require_scale {
};

template <typename StorageType>
struct helper_require_scale<StorageType, false> : public std::false_type {
};

template <typename StorageType>
struct helper_require_scale<StorageType, true> : public std::true_type {
};


}  // namespace detail


template <typename ValueType, typename StorageType,
          bool = detail::helper_require_scale<StorageType>::value>
class Accessor3dHelper {
};


template <typename ValueType, typename StorageType>
class Accessor3dHelper<ValueType, StorageType, true> {
public:
    using accessor = accessor::ScaledReducedStorage3d<ValueType, StorageType>;

    Accessor3dHelper() = default;

    Accessor3dHelper(std::shared_ptr<const Executor> exec, dim<3> krylov_dim)
        : krylov_dim_{krylov_dim},
          bases_{exec, krylov_dim_[0] * krylov_dim_[1] * krylov_dim_[2]},
          scale_{exec, krylov_dim_[0] * krylov_dim_[2]}
    {
        // For testing, initialize scale to ones
        // Array<ValueType> h_scale{exec->get_master(), krylov_dim[0]};
        Array<ValueType> h_scale{exec->get_master(),
                                 krylov_dim[0] * krylov_dim[2]};
        for (size_type i = 0; i < h_scale.get_num_elems(); ++i) {
            h_scale.get_data()[i] = one<ValueType>();
        }
        scale_ = h_scale;
    }

    accessor get_accessor()
    {
        return {bases_.get_data(), krylov_dim_, scale_.get_data()};
    }

    gko::Array<StorageType> &get_bases() { return bases_; }

private:
    dim<3> krylov_dim_;
    Array<StorageType> bases_;
    Array<ValueType> scale_;
};


template <typename ValueType, typename StorageType>
class Accessor3dHelper<ValueType, StorageType, false> {
public:
    using accessor = accessor::ReducedStorage3d<ValueType, StorageType>;

    Accessor3dHelper() = default;

    Accessor3dHelper(std::shared_ptr<const Executor> exec, dim<3> krylov_dim)
        : krylov_dim_{krylov_dim},
          bases_{std::move(exec),
                 krylov_dim_[0] * krylov_dim_[1] * krylov_dim_[2]}
    {}

    accessor get_accessor() { return {bases_.get_data(), krylov_dim_}; }

    gko::Array<StorageType> &get_bases() { return bases_; }

private:
    dim<3> krylov_dim_;
    Array<StorageType> bases_;
};

//----------------------------------------------

template <typename Accessor3d,
          bool = detail::is_3d_scaled_accessor<Accessor3d>::value>
struct helper_functions_accessor {
};

// Accessors having a scale
template <typename Accessor3d>
struct helper_functions_accessor<Accessor3d, true> {
    using arithmetic_type = typename Accessor3d::arithmetic_type;
    static_assert(detail::is_3d_scaled_accessor<Accessor3d>::value,
                  "Accessor must have a scale here!");
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scale(Accessor3d krylov_bases,
                                                  IndexType vector_idx,
                                                  IndexType col_idx,
                                                  arithmetic_type value)
    {
        using storage_type = typename Accessor3d::storage_type;
        constexpr arithmetic_type correction =
            std::is_integral<storage_type>::value
                // Use 2 instead of 1 here to allow for a bit more room
                ? 2 / static_cast<arithmetic_type>(
                          std::numeric_limits<storage_type>::max())
                : 1;
        krylov_bases.set_scale(vector_idx, col_idx, value * correction);
    }
};

// Accessors not having a scale
template <typename Accessor3d>
struct helper_functions_accessor<Accessor3d, false> {
    using arithmetic_type = typename Accessor3d::arithmetic_type;
    static_assert(!detail::is_3d_scaled_accessor<Accessor3d>::value,
                  "Accessor must not have a scale here!");
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scale(Accessor3d krylov_bases,
                                                  IndexType vector_idx,
                                                  IndexType col_idx,
                                                  arithmetic_type value)
    {}
};

// calling it with:
// helper_functions_accessor<Accessor3d>::write_scale(krylov_bases, col_idx,
// value);

//----------------------------------------------

}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_MIXED_ACCESSOR_HPP_
