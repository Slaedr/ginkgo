// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/multicolor_kernels.hpp"

#include <algorithm>
#include <random>
#include <vector>

#include <omp.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/allocator.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace multicolor {


template <typename IndexType>
std::vector<IndexType> generate_random(const size_type N, const IndexType lo,
                                       const IndexType hi)
{
    std::vector<IndexType> v(N);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        // Each thread initializes its own RNG
        // std::random_device rd;
        // std::seed_seq seq{rd(), rd(), rd(), rd(),
        // static_cast<uint32_t>(tid)};
        std::seed_seq seq{1u, 101u, 42u, 23u, static_cast<uint32_t>(tid)};
        std::mt19937 rng{seq};
        std::uniform_int_distribution<IndexType> dist{lo, hi};

#pragma omp for schedule(static)
        for (size_type i = 0; i < N; ++i) {
            v[i] = dist(rng);
        }
    }

    return v;
}

// TODO: Implement early exit.
template <typename T>
bool check_value_exists(const std::vector<T>& vec, const T& target)
{
    bool found = false;
#pragma omp parallel for shared(found)
    for (size_t i = 0; i < vec.size(); i++) {
        if (vec[i] == target) {
#pragma omp atomic write
            found = true;
        }
    }
    return found;
}

/* Compute an independent set according to Luby.
 * and return whether there are remaining uncolored vertices.
 */
template <typename IndexType>
void independent_set(const IndexType num_vertices,
                     const IndexType* const row_ptrs,
                     const IndexType* const col_idxs,
                     const std::vector<IndexType>& randvec,
                     const int current_color,
                     const std::vector<int>& prev_color,
                     std::vector<int>& new_color)
{
    int indset_empty = 0;
    std::vector<int> cur_neigh(num_vertices, 0);
    std::vector<int> prev_neigh(num_vertices);
    while (!indset_empty) {
        // In each iteration, we remove the independent set vertices and their
        // (uncolored) neighbors from the graph, and try to find more
        // independent vertices to augment the independent set.
        prev_neigh = cur_neigh;
        indset_empty = 1;
#pragma omp parallel for shared(indset_empty)
        for (IndexType irow = 0; irow < num_vertices; irow++) {
            // go over un-colored nodes
            int this_thread_indset_not_empy = 0;
            const bool ipoin_in_graph =
                prev_color[irow] == -1 && prev_neigh[irow] == 0;
            if (ipoin_in_graph) {
                // Use lexicographic (randvec, index) comparison to break ties.
                bool irow_dominated = false;
                for (int jz = row_ptrs[irow]; jz < row_ptrs[irow + 1]; jz++) {
                    const int j = col_idxs[jz];
                    const bool j_in_graph =
                        prev_color[j] == -1 && prev_neigh[j] == 0;
                    if (j_in_graph && j != irow) {
                        if (randvec[j] > randvec[irow] ||
                            (randvec[j] == randvec[irow] && j < irow)) {
                            irow_dominated = true;
                            break;
                        }
                    }
                }
                if (!irow_dominated) {
                    // irow is to be added to this color
                    new_color[irow] = current_color;
#pragma omp atomic write
                    cur_neigh[irow] = 1;
                    // record the neighbours of the current color
                    for (int jz = row_ptrs[irow]; jz < row_ptrs[irow + 1];
                         jz++) {
                        const int j = col_idxs[jz];
                        if (prev_color[j] == -1 && j != irow) {
#pragma omp atomic write
                            cur_neigh[j] = 1;
                        }
                    }
                } else {
                    // this node may remain in the graph at the end of this
                    // iteration
#pragma omp atomic write
                    indset_empty = 0;
                }
            }
        }  // end parallel for
    }      // end independent set
}

struct Coloring {
    int num_colors;
    std::vector<int> vertex_colors;
};

template <typename IndexType>
Coloring compute_coloring(const IndexType num_vertices,
                          const IndexType* const row_ptrs,
                          const IndexType* const col_idxs,
                          const std::vector<IndexType>& randvec)
{
    std::vector<int> color(num_vertices);
    std::vector<int> new_color(num_vertices);
    std::vector<int> cur_neigh(num_vertices);
#pragma omp parallel for schedule(static)
    for (IndexType i = 0; i < num_vertices; i++) {
        color[i] = -1;
        new_color[i] = -1;
        cur_neigh[i] = 0;
    }
    bool uncolored_exist = true;
    int current_color = 0;
    while (uncolored_exist) {
        // In each iteration, compute one color (independent set)
        independent_set(num_vertices, row_ptrs, col_idxs, randvec,
                        current_color, color, new_color);
        uncolored_exist = check_value_exists(new_color, -1);
        if (current_color >= 100) {
            break;
        }
        if (uncolored_exist) {
            color = new_color;
        }
        current_color++;
    }
    return Coloring{current_color, new_color};
}

template <typename IndexType>
void compute_color_ptrs(std::shared_ptr<const OmpExecutor> exec,
                        const Coloring& coloring, const IndexType num_vertices,
                        std::vector<IndexType>& color_vec,
                        IndexType* const old_to_new,
                        IndexType* const new_to_old)
{
    color_vec.assign(coloring.num_colors + 1, 0);
#pragma omp parallel for
    for (IndexType old_i = 0; old_i < num_vertices; old_i++) {
        const int color = coloring.vertex_colors[old_i];
        IndexType ind = 0;
#pragma omp atomic capture
        ind = color_vec[color]++;
        old_to_new[old_i] = ind;
    }

    gko::kernels::omp::components::prefix_sum_nonnegative(
        exec, color_vec.data(), coloring.num_colors);
    color_vec[coloring.num_colors] = num_vertices;

#pragma omp parallel for
    for (IndexType old_i = 0; old_i < num_vertices; old_i++) {
        const int color = coloring.vertex_colors[old_i];
        old_to_new[old_i] += color_vec[color];
        new_to_old[old_to_new[old_i]] = old_i;
    }
}


template <typename IndexType>
void compute_permutation_csr(std::shared_ptr<const OmpExecutor> exec,
                             const IndexType num_vertices,
                             const IndexType* const row_ptrs,
                             const IndexType* const col_idxs,
                             std::vector<IndexType>& color_ptrs,
                             IndexType* const permutation,
                             IndexType* const inv_permutation)
{
    constexpr int rand_mult = 4;
    const auto randvec = generate_random<IndexType>(
        num_vertices, 1, rand_mult * num_vertices - 1);
    const auto coloring =
        compute_coloring<IndexType>(num_vertices, row_ptrs, col_idxs, randvec);
    // The permutation maps a new index to the old index,
    // so that it can be used directly with LinOp::permute. That is what
    // compute_color_ptrs calls new_to_old.
    compute_color_ptrs<IndexType>(exec, coloring, num_vertices, color_ptrs,
                                  inv_permutation, permutation);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_MULTICOLOR_COMPUTE_PERMUTATION_CSR_KERNEL);


}  // namespace multicolor
}  // namespace omp
}  // namespace kernels
}  // namespace gko
