// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARKS_AMP_TYPES_HPP_
#define GKO_BENCHMARKS_AMP_TYPES_HPP_

#include <ginkgo/ginkgo.hpp>

using int32 = gko::int32;
using global_idx_t = long;
using local_idx_t = int;
using scalar_t = double;

using comm_t = gko::experimental::mpi::communicator;

template <typename value_type>
using dist_vec_t = gko::experimental::distributed::Vector<value_type>;

template <typename value_type>
using dist_mtx_t =
    gko::experimental::distributed::Matrix<value_type, local_idx_t,
                                           global_idx_t>;

#endif
