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


#ifndef GKO_CORE_TEST_BLK_TRS_UTILS_HPP
#define GKO_CORE_TEST_BLK_TRS_UTILS_HPP


#include <random>


#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/factorization/block_factorization_kernels.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"


namespace gko {
namespace testing {


enum TriSystemType { LOWER_TRI, UPPER_TRI };

template <typename ValueType, typename IndexType>
struct BlkTriSystem {
    using Fbcsr = gko::matrix::Fbcsr<ValueType, IndexType>;
    using Dense = gko::matrix::Dense<ValueType>;
    std::shared_ptr<const Fbcsr> mtx;
    std::shared_ptr<const Fbcsr> tri_mtx;
    std::shared_ptr<const Dense> b;
    std::shared_ptr<const Dense> x;
};


template <typename ValueType, typename IndexType>
inline void copy_diagonal_blocks(
    const matrix::Fbcsr<ValueType, IndexType> *const mtx,
    matrix::Fbcsr<ValueType, IndexType> *const trimtx)
{
    if (!mtx->get_executor()->memory_accessible(ReferenceExecutor::create()))
        GKO_NOT_SUPPORTED(mtx->get_executor());
    if (!trimtx->get_executor()->memory_accessible(ReferenceExecutor::create()))
        GKO_NOT_SUPPORTED(trimtx->get_executor());

    const IndexType nbrows = mtx->get_num_block_rows();
    const int bs = mtx->get_block_size();
    assert(bs == trimtx->get_block_size());
    assert(nbrows == trimtx->get_num_block_rows());

    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto values = mtx->get_const_values();
    const auto tri_row_ptrs = trimtx->get_const_row_ptrs();
    const auto tri_col_idxs = trimtx->get_const_col_idxs();
    const auto tri_values = trimtx->get_values();
    for (IndexType ibrow = 0; ibrow < nbrows; ibrow++) {
        const ValueType *diag{};
        for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
             ibz++) {
            if (col_idxs[ibz] == ibrow) {
                diag = &values[ibz * bs * bs];
            }
        }
        assert(diag);
        bool written = false;
        for (IndexType ibz = tri_row_ptrs[ibrow]; ibz < tri_row_ptrs[ibrow + 1];
             ibz++) {
            if (tri_col_idxs[ibz] == ibrow) {
                for (int i = 0; i < bs * bs; i++) {
                    tri_values[ibz * bs * bs + i] = diag[i];
                }
                written = true;
            }
        }
        assert(written);
    }
}

// Assumes column-major blocks
template <typename ValueType, typename IndexType>
inline void make_strict_triangular(
    const TriSystemType type, matrix::Fbcsr<ValueType, IndexType> *const trimtx)
{
    const IndexType nbrows = trimtx->get_num_block_rows();
    const int bs = trimtx->get_block_size();

    const auto row_ptrs = trimtx->get_const_row_ptrs();
    const auto col_idxs = trimtx->get_const_col_idxs();
    const auto values = trimtx->get_values();
    for (IndexType ibrow = 0; ibrow < nbrows; ibrow++) {
        ValueType *diag{};
        for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
             ibz++) {
            if (col_idxs[ibz] == ibrow) {
                diag = &values[ibz * bs * bs];
            }
        }
        assert(diag);
        for (int ib = 0; ib < bs; ib++) {
            for (int jb = 0; jb < bs; jb++) {
                if ((type == LOWER_TRI && ib < jb) ||
                    (type == UPPER_TRI && ib > jb)) {
                    diag[ib + jb * bs] = static_cast<ValueType>(0.0);
                }
            }
        }
    }
}

/**
 * Randomly generates a matrix, extracts its block-triangular part,
 * randomly generates a solution vector and 'manufactures' a RHS vector for the
 * the triangular system.
 */
template <typename ValueType, typename IndexType>
inline BlkTriSystem<ValueType, IndexType> get_block_tri_system(
    const std::shared_ptr<const Executor> exec, const IndexType nbrows,
    const int blk_sz, const int nrhs, const bool diag_identity,
    const TriSystemType type, const bool diag_dominant = false,
    const bool strict_triangular = false)
{
    using real_type = gko::remove_complex<ValueType>;
    using Fbcsr = gko::matrix::Fbcsr<ValueType, IndexType>;
    using Dense = gko::matrix::Dense<ValueType>;
    auto refexec = gko::ReferenceExecutor::create();
    const IndexType nrows = nbrows * blk_sz;
    std::shared_ptr<const Fbcsr> mtx =
        gko::test::generate_random_fbcsr<ValueType, IndexType>(
            refexec, std::ranlux48(42), nbrows, nbrows, blk_sz, diag_dominant,
            false);
    Array<IndexType> l_row_ptrs(refexec, nbrows + 1);
    Array<IndexType> u_row_ptrs(refexec, nbrows + 1);
    gko::kernels::reference::factorization::initialize_row_ptrs_BLU(
        refexec, mtx.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data());
    const IndexType l_nbnz = l_row_ptrs.get_const_data()[nbrows];
    const IndexType u_nbnz = u_row_ptrs.get_const_data()[nbrows];
    Array<IndexType> l_col_idxs(refexec, l_nbnz);
    Array<IndexType> u_col_idxs(refexec, u_nbnz);
    Array<ValueType> l_values(refexec, l_nbnz * blk_sz * blk_sz);
    Array<ValueType> u_values(refexec, u_nbnz * blk_sz * blk_sz);
    std::shared_ptr<Fbcsr> l_factor =
        Fbcsr::create(refexec, dim<2>(nrows, nrows), blk_sz, l_values,
                      l_col_idxs, l_row_ptrs);
    std::shared_ptr<Fbcsr> u_factor =
        Fbcsr::create(refexec, dim<2>(nrows, nrows), blk_sz, u_values,
                      u_col_idxs, u_row_ptrs);
    gko::kernels::reference::factorization::initialize_BLU(
        refexec, mtx.get(), l_factor.get(), u_factor.get());
    if (!diag_identity && (type == LOWER_TRI)) {
        copy_diagonal_blocks(mtx.get(), l_factor.get());
    }

    const std::shared_ptr<Fbcsr> tri_factor =
        (type == LOWER_TRI) ? l_factor : u_factor;
    if (strict_triangular) {
        make_strict_triangular(type, tri_factor.get());
    }

    std::shared_ptr<Dense> x = Dense::create(refexec, dim<2>(nrows, nrhs));
    ValueType *const xarr = x->get_values();
    for (IndexType i = 0; i < nrows * nrhs; i++) {
        xarr[i] = static_cast<ValueType>(std::sin(static_cast<ValueType>(
            i / 2.0 + gko::test::get_some_number<ValueType>())));
    }
    std::shared_ptr<Dense> b = Dense::create(refexec, gko::dim<2>(nrows, nrhs));

    std::shared_ptr<Fbcsr> d_mtx = Fbcsr::create(exec);
    std::shared_ptr<Fbcsr> d_tri_factor = Fbcsr::create(exec);
    std::shared_ptr<Dense> d_x = Dense::create(exec);
    std::shared_ptr<Dense> d_b = Dense::create(exec);
    d_mtx->copy_from(mtx.get());
    d_x->copy_from(x.get());
    d_tri_factor->copy_from(tri_factor.get());
    tri_factor->apply(x.get(), b.get());
    d_b->copy_from(b.get());

    return BlkTriSystem<ValueType, IndexType>{d_mtx, d_tri_factor, d_b, d_x};
}

}  // namespace testing
}  // namespace gko

#endif
