// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_REORDERING_HPP_
#define GKO_CORE_TEST_UTILS_REORDERING_HPP_


#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

#include <ginkgo/core/base/dim.hpp>


namespace gko {
namespace test {


template <typename T>
inline T get_natural_flat_index_from_3d(const std::array<T, 3>& dims,
                                        const std::array<T, 3>& idx)
{
    return idx[2] * dims[1] * dims[0] + idx[1] * dims[0] + idx[0];
}

/**
 * Checks that every color of an already-reordered matrix is an independent set.
 *
 * The matrix is expected to be in the multicolor ordering, so that the rows of
 * color `c` are exactly `[color_ptrs[c], color_ptrs[c+1])`. The coloring is
 * usable by a multicolor Gauss-Seidel sweep only if no row of a color has an
 * off-diagonal entry in a column belonging to the same color; otherwise the
 * rows of that color cannot be updated in parallel.
 *
 * Unlike @ref compute_multicolor_ordering_regular_box and its 2d counterpart,
 * this works for an arbitrary sparsity pattern.
 *
 * @param num_rows  Number of rows of the reordered matrix.
 * @param row_ptrs  Row pointers of the reordered matrix (CSR).
 * @param col_idxs  Column indices of the reordered matrix (CSR).
 * @param color_ptrs  Row at which each color starts, @see
 *                    gko::reorder::Multicolor::get_color_pointers.
 */
template <typename IndexType>
inline bool colors_are_independent(const IndexType num_rows,
                                   const IndexType* const row_ptrs,
                                   const IndexType* const col_idxs,
                                   const std::vector<IndexType>& color_ptrs)
{
    if (color_ptrs.size() < 2) {
        return false;
    }
    for (IndexType row = 0; row < num_rows; row++) {
        const auto it =
            std::upper_bound(color_ptrs.begin(), color_ptrs.end(), row);
        if (it == color_ptrs.begin()) {
            // row lies before the first color
            return false;
        }
        const auto color =
            static_cast<int>(std::distance(color_ptrs.begin(), it)) - 1;
        const auto color_begin = color_ptrs[color];
        const auto color_end = color_ptrs[color + 1];
        for (auto jz = row_ptrs[row]; jz < row_ptrs[row + 1]; jz++) {
            const auto col = col_idxs[jz];
            if (col != row && col >= color_begin && col < color_end) {
                return false;
            }
        }
    }
    return true;
}

template <typename itype>
struct MulticolorOrdering {
    std::vector<itype> new_to_old;
    std::vector<itype> old_to_new;
    std::vector<itype> color_ptrs;
};

/**
 * Compute 8-color independent-set ordering for a 3d box (27-pt) stencil.
 */
template <typename itype>
MulticolorOrdering<itype> compute_multicolor_ordering_regular_box(
    const gko::dim<3>& local_grid_dims)
{
    const std::array<int, 3> ldims{static_cast<int>(local_grid_dims[0]),
                                   static_cast<int>(local_grid_dims[1]),
                                   static_cast<int>(local_grid_dims[2])};
    const int ln = ldims[0] * ldims[1] * ldims[2];
    std::vector<itype> old_to_new(ln);
    std::vector<itype> new_to_old(ln);
    std::vector<itype> cnt(8, 0);

    for (itype k = 0; k < ldims[2]; ++k)
        for (itype j = 0; j < ldims[1]; ++j)
            for (itype i = 0; i < ldims[0]; ++i)
                ++cnt[(i % 2) + 2 * (j % 2) + 4 * (k % 2)];

    std::vector<int> color_ptrs(9);
    color_ptrs[0] = 0;
    for (int c = 0; c < 8; ++c) {
        color_ptrs[c + 1] = color_ptrs[c] + cnt[c];
    }

    std::vector<itype> fill(8, 0);
    for (itype k = 0; k < ldims[2]; ++k) {
        for (itype j = 0; j < ldims[1]; ++j) {
            for (itype i = 0; i < ldims[0]; ++i) {
                const std::array<itype, 3> idx{i, j, k};
                const int old_idx = get_natural_flat_index_from_3d(ldims, idx);
                const int color = (i % 2) + 2 * (j % 2) + 4 * (k % 2);
                const int new_idx = color_ptrs[color] + fill[color]++;
                old_to_new[old_idx] = new_idx;
                new_to_old[new_idx] = old_idx;
            }
        }
    }
    return MulticolorOrdering<itype>{new_to_old, old_to_new, color_ptrs};
}

/**
 * Compute 2-color independent-set ordering for a 2d star (5-pt) stencil.
 */
template <typename itype>
MulticolorOrdering<itype> compute_multicolor_ordering_regular_star(
    const gko::dim<2>& local_grid_dims)
{
    const int ln = local_grid_dims[0] * local_grid_dims[1];
    std::vector<itype> old_to_new(ln);
    std::vector<itype> new_to_old(ln);
    std::vector<itype> cnt(2, 0);

    for (itype j = 0; j < local_grid_dims[1]; ++j) {
        for (itype i = 0; i < local_grid_dims[0]; ++i) {
            ++cnt[(i + j) % 2];
        }
    }

    std::vector<itype> color_ptrs(3);
    color_ptrs[0] = 0;
    for (int c = 0; c < 2; ++c) {
        color_ptrs[c + 1] = color_ptrs[c] + cnt[c];
    }

    std::vector<itype> fill(2, 0);
    for (itype j = 0; j < local_grid_dims[1]; ++j) {
        for (itype i = 0; i < local_grid_dims[0]; ++i) {
            const int old_idx = local_grid_dims[0] * j + i;
            const int color = (i + j) % 2;
            const int new_idx = color_ptrs[color] + fill[color]++;
            old_to_new[old_idx] = new_idx;
            new_to_old[new_idx] = old_idx;
        }
    }
    return MulticolorOrdering<itype>{new_to_old, old_to_new, color_ptrs};
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_REORDERING_HPP_
