// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_AMP_MATRIX_GENERATION_HPP_
#define GKO_BENCHMARK_AMP_MATRIX_GENERATION_HPP_

#include <array>
#include <vector>

#include "benchmark/amp/types.hpp"

inline int min3(int a, int b, int c) { return std::min(a, std::min(b, c)); }

inline int max3(int a, int b, int c) { return std::max(a, std::max(b, c)); }

inline std::array<int, 3> cubic_radical_search(const int n)
{
    int x = n, y = 1, z = 1;
    double best = 0.0;

    for (int f1 = (int)(std::pow(n, 1.0 / 3.0) + 0.5); f1 > 0; --f1) {
        if (n % f1 == 0) {
            int n1 = n / f1;
            for (int f2 = (int)(pow(n1, 0.5) + 0.5); f2 > 0; --f2) {
                if (n1 % f2 == 0) {
                    int f3 = n1 / f2;
                    double current =
                        (double)min3(f1, f2, f3) / max3(f1, f2, f3);
                    if (current > best) {
                        best = current;
                        x = f1;
                        y = f2;
                        z = f3;
                    }
                }
            }
        }
    }
    return std::array<int, 3>{x, y, z};
}

template <typename T, typename U, typename V>
std::array<T, 3> mult(const std::array<U, 3>& a, const std::array<V, 3>& b)
{
    std::array<T, 3> out;
    for (int i = 0; i < 3; i++) {
        out[i] = static_cast<T>(a[i]) * static_cast<T>(b[i]);
    }
    return out;
}

template <typename T, typename U, typename V>
std::array<T, 3> add(const std::array<U, 3>& a, const std::array<V, 3>& b)
{
    std::array<T, 3> out;
    for (int i = 0; i < 3; i++) {
        out[i] = static_cast<T>(a[i]) + static_cast<T>(b[i]);
    }
    return out;
}

/**
 * Get 3D indices corresponding to a flattened 1D index in natural ordering.
 *
 * Note that in all arrays, index 0 is x, 1 is y and 2 is z axis.
 */
template <typename T>
inline std::array<T, 3> get_natural_3d_indices_from_flat(
    const std::array<T, 3>& dims, const T flat_idx)
{
    T iz = flat_idx / (dims[1] * dims[0]);
    T iy = (flat_idx - iz * dims[1] * dims[0]) / dims[0];
    T ix = flat_idx % dims[0];
    return std::array<T, 3>{ix, iy, iz};
}

template <typename T>
inline T get_natural_flat_index_from_3d(const std::array<T, 3>& dims,
                                        const std::array<T, 3>& idx)
{
    return idx[2] * dims[1] * dims[0] + idx[1] * dims[0] + idx[0];
}

struct MulticolorOrdering {
    std::vector<int> new_to_old;
    std::vector<int> old_to_new;
    std::vector<int> color_ptrs;
};

inline MulticolorOrdering compute_multicolor_ordering(
    const std::array<int, 3>& local_grid_dims)
{
    const int ln = local_grid_dims[0] * local_grid_dims[1] * local_grid_dims[2];
    std::vector<int> old_to_new(ln);
    std::vector<int> new_to_old(ln);
    std::vector<int> cnt(8, 0);

    for (int k = 0; k < local_grid_dims[2]; ++k)
        for (int j = 0; j < local_grid_dims[1]; ++j)
            for (int i = 0; i < local_grid_dims[0]; ++i)
                ++cnt[(i % 2) + 2 * (j % 2) + 4 * (k % 2)];

    std::vector<int> color_ptrs(9);
    color_ptrs[0] = 0;
    for (int c = 0; c < 8; ++c) {
        color_ptrs[c + 1] = color_ptrs[c] + cnt[c];
    }

    std::vector<int> fill(8, 0);
    for (int k = 0; k < local_grid_dims[2]; ++k) {
        for (int j = 0; j < local_grid_dims[1]; ++j) {
            for (int i = 0; i < local_grid_dims[0]; ++i) {
                const std::array<int, 3> idx{i, j, k};
                const int old_idx =
                    get_natural_flat_index_from_3d(local_grid_dims, idx);
                const int color = (i % 2) + 2 * (j % 2) + 4 * (k % 2);
                const int new_idx = color_ptrs[color] + fill[color]++;
                old_to_new[old_idx] = new_idx;
                new_to_old[new_idx] = old_idx;
            }
        }
    }
    return MulticolorOrdering{new_to_old, old_to_new, color_ptrs};
}

/**
 * Compute global index of a point from its local index and other details.
 *
 * @param owner_rank  Rank of the subdomain owning the given point.
 * @param comm_size_dir  Subdomain rank of the owning subdomain in each
 *                       logical direction.
 * @param local_grid_dims  The common size of the local grid of each subdomain.
 * @param local_flat_idx  Flattened local index of the point in question.
 */
global_idx_t get_global_from_local(const std::array<int, 3>& owner_rank,
                                   const std::array<int, 3>& comm_size_dir,
                                   const std::array<int, 3>& local_grid_dims,
                                   const int local_flat_idx)
{
    const auto global_grid_dims =
        mult<global_idx_t>(local_grid_dims, comm_size_dir);
    const auto global_idx_base =
        mult<global_idx_t>(local_grid_dims, owner_rank);
    return get_natural_flat_index_from_3d(global_grid_dims, global_idx_base) +
           local_flat_idx;
}

// ============================================================
// 3D 27-point stencil generator with 8-coloring
// ============================================================

/**
 * Builds matrix_data for a 3D 27-point stencil with rows locally ordered by
 * 8-color partitioning (color = (i%2) + 2*(j%2) + 4*(k%2)).  Within each
 * color in a particular subdomain, all nodes are independent under
 * the 27-point stencil.
 *
 * Diagonal entry = 26.  Off-diagonal values are drawn independently from
 * @p gen.  To recover the original constant-coefficient stencil,
 * pass a distribution that always returns -1.
 *
 * @param comm  Ginkgo MPI communicator.
 * @param local_grid_dims  Local grid dimensions (entry[0] is x-direction).
 *                         NOTE: assumed to be the same on all ranks.
 * @param gen          Generator function for off-diagonal values,
 *                     called with no arguments.
 * @param color_ptrs   Output: color_ptrs[c] is the first row of color c,
 *                     color_ptrs[8] == n.  Size 9.
 * @return  matrix_data<double, int32> in the color-ordered layout.
 */
template <typename OffdiagFn>
inline gko::matrix_data<scalar_t, global_idx_t> generate_stencil_data(
    comm_t comm, const std::array<local_idx_t, 3>& local_grid_dims,
    OffdiagFn& gen, const MulticolorOrdering& ordering)
{
    const std::array<int, 3> proc_dims = cubic_radical_search(comm.size());
    const std::array<int, 3> my_ranks =
        get_natural_3d_indices_from_flat(proc_dims, comm.rank());
    const int ln = local_grid_dims[0] * local_grid_dims[1] * local_grid_dims[2];
    const auto my_offsets = mult<global_idx_t>(my_ranks, local_grid_dims);

    gko::matrix_data<double, global_idx_t> data(gko::dim<2>{
        static_cast<gko::size_type>(ln), static_cast<gko::size_type>(ln)});
    data.nonzeros.reserve(27 * ln);


    scalar_t max_val{0.0}, min_val{100.0};
    for (int new_row = 0; new_row < ln; ++new_row) {
        const global_idx_t global_new_row = get_global_from_local(
            my_ranks, proc_dims, local_grid_dims, new_row);
        const int old_row = ordering.new_to_old[new_row];
        const std::array<int, 3> old_idx =
            get_natural_3d_indices_from_flat(local_grid_dims, old_row);

        for (int dk = -1; dk <= 1; ++dk) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    const std::array<int, 3> dst{di, dj, dk};
                    std::array<int, 3> nbd_old_idx = add<int>(old_idx, dst);
                    // Get new local index of the stencil point,
                    //   in the subdomain that point belongs to
                    std::array<int, 3> nbd_rank = my_ranks;
                    for (int i = 0; i < 3; i++) {
                        if (nbd_old_idx[i] < 0) {
                            nbd_rank[i]--;
                            nbd_old_idx[i] = local_grid_dims[i] - 1;
                        } else if (nbd_old_idx[i] >= local_grid_dims[i]) {
                            nbd_rank[i]++;
                            nbd_old_idx[i] -= local_grid_dims[i];
                        }
                    }
                    const local_idx_t local_new_flat_idx =
                        ordering.old_to_new[get_natural_flat_index_from_3d(
                            local_grid_dims, nbd_old_idx)];
                    const global_idx_t global_new_col = get_global_from_local(
                        nbd_rank, proc_dims, local_grid_dims,
                        local_new_flat_idx);
                    const bool diag = (di == 0 && dj == 0 && dk == 0);
                    const auto val = static_cast<scalar_t>(diag ? 26.0 : gen());
                    if (!diag) {
                        max_val = std::max(max_val, std::abs(val));
                        min_val = std::min(min_val, std::abs(val));
                    }
                    data.nonzeros.emplace_back(global_new_row, global_new_col,
                                               val);
                }
            }
        }
    }
    data.sort_row_major();
    std::cout << "\n  Generated matrix off-diagonals: max abs val = " << max_val
              << ", min abs val = " << min_val << std::endl;
    return data;
}

struct ProblemData {
    gko::matrix_data<scalar_t, global_idx_t> mat_data;
    std::vector<int> color_ptrs;
};

template <typename OffdiagFn>
inline ProblemData generate_problem_data(
    comm_t comm, const std::array<local_idx_t, 3>& local_grid_dims,
    OffdiagFn& gen)
{
    const auto ordering = compute_multicolor_ordering(local_grid_dims);
    auto matdata = generate_stencil_data(comm, local_grid_dims, gen, ordering);
    return ProblemData{matdata, ordering.color_ptrs};
}


#endif
