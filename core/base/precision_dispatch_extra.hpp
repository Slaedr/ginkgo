// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_PRECISION_DISPATCH_EXTRA_HPP
#define GKO_CORE_BASE_PRECISION_DISPATCH_EXTRA_HPP

#include <type_traits>

#include <ginkgo/core/base/precision_dispatch.hpp>


namespace gko {


/**
 * Dispatches the given function with the input/output LinOps dynamically cast
 * to Dense<double> or Dense<float> (all four combinations), but only if
 * GINKGO_MIXED_PRECISION was set to ON during configuration. Otherwise,
 * it falls back to temporary conversions.
 *
 * Unlike mixed_precision_dispatch, this does NOT walk the full
 * next_precision chain (which may include half/bfloat16). It only considers
 * double and float, which is sufficient for AMP matrix apply where the
 * vectors should be either double or float.
 *
 * @tparam ValueType  used only to decide whether the dispatch is over real
 *                    or complex types. If complex, dispatches over
 *                    complex<double> and complex<float>.
 */
template <typename ValueType, typename Function>
void mixed_precision_base_dispatch(Function fn, const LinOp* in, LinOp* out)
{
#ifdef GINKGO_MIXED_PRECISION
    using fst_type = matrix::Dense<ValueType>;
    using snd_type = matrix::Dense<
        typename detail::next_precision_base_impl<ValueType>::type>;
    auto dispatch_out = [&](auto dense_in) {
        if (auto dense_out = dynamic_cast<fst_type*>(out)) {
            fn(dense_in, dense_out);
        } else if (auto dense_out = dynamic_cast<snd_type*>(out)) {
            fn(dense_in, dense_out);
        } else {
            GKO_NOT_SUPPORTED(out);
        }
    };
    if (auto dense_in = dynamic_cast<const fst_type*>(in)) {
        dispatch_out(dense_in);
    } else if (auto dense_in = dynamic_cast<const snd_type*>(in)) {
        dispatch_out(dense_in);
    } else {
        GKO_NOT_SUPPORTED(in);
    }
#else
    precision_dispatch<ValueType>(fn, in, out);
#endif
}


/**
 * Like mixed_precision_base_dispatch, but handles the case where ValueType
 * is real and the vectors are complex: converts via create_real_view().
 *
 * Supports Dense<double>, Dense<float>, Dense<complex<double>>, and
 * Dense<complex<float>> as vector types.
 *
 * @note When ValueType is complex, real vectors are not supported and will
 *       throw NotSupported. This is acceptable because multiplying a complex
 *       matrix by a real vector is not a typical use case.
 */
template <typename ValueType, typename Function,
          std::enable_if_t<is_complex<ValueType>()>* = nullptr>
void mixed_precision_base_dispatch_real_complex(Function fn, const LinOp* in,
                                                LinOp* out)
{
#ifdef GINKGO_MIXED_PRECISION
    mixed_precision_base_dispatch<ValueType>(fn, in, out);
#else
    precision_dispatch<ValueType>(fn, in, out);
#endif
}


template <typename ValueType, typename Function,
          std::enable_if_t<!is_complex<ValueType>()>* = nullptr>
void mixed_precision_base_dispatch_real_complex(Function fn, const LinOp* in,
                                                LinOp* out)
{
#ifdef GINKGO_MIXED_PRECISION
    if (!dynamic_cast<const ConvertibleTo<matrix::Dense<>>*>(in)) {
        mixed_precision_base_dispatch<to_complex<ValueType>>(
            [&fn](auto dense_in, auto dense_out) {
                fn(dense_in->create_real_view().get(),
                   dense_out->create_real_view().get());
            },
            in, out);
    } else {
        mixed_precision_base_dispatch<ValueType>(fn, in, out);
    }
#else
    precision_dispatch_real_complex<ValueType>(fn, in, out);
#endif
}


}  // namespace gko

#endif  // GKO_CORE_BASE_PRECISION_DISPATCH_EXTRA_HPP
