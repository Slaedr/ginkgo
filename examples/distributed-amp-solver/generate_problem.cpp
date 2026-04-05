// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "poisson_amp.hpp"

namespace ex_dist_amp {

std::array<int, 3> cubic_radical_search(const int n);

class Mesh {
public:
    Mesh(comm_t comm, const std::array<int, 3>& local_dims,
         const double mesh_stretch_ratio)
        : comm_my_rank_{comm.rank()},
          comm_size_{comm.size()},
          local_dims_{local_dims},
          proc_dims_{cubic_radical_search(comm.size())},
          global_dims_{static_cast<long>(proc_dims_[0]) * local_dims[0],
                       static_cast<long>(proc_dims_[1]) * local_dims[1],
                       static_cast<long>(proc_dims_[2]) * local_dims[2]},
          q_{mesh_stretch_ratio},
          q_nearly_one_{ratio_nearly_one()},
          h0_{get_first_point()}
    {}

    const std::array<int, 3>& get_local_dims() const { return local_dims_; }

    int comm_size() const { return comm_size_; }

    /**
     * Get the physical location along the given direction of a point.
     *
     * @param i  Global index of the point in the required direction.
     * @param dir  The required direction (0, 1 or 2 for x, y or z).
     */
    double get_1D_point_coordinate(const long i, const int dir) const
    {
        assert(i >= 0 && i < global_dims_[dir]);
        if (i == 0) {
            return domain_min_;
        } else {
            if (q_nearly_one_) {
                return h0_[dir] * i;
            } else {
                return h0_[dir] * (std::pow(q_, i) - 1) / (q_ - 1);
            }
        }
    }

    std::array<double, 3> get_point_coordinates(
        const std::array<long, 3>& idx) const
    {
        std::array<double, 3> coords;
        for (int i = 0; i < 3; i++) {
            coords[i] = get_1D_point_coordinate(idx[i], i);
        }
        return coords;
    }

    std::array<double, 3> get_first_point() const
    {
        std::array<double, 3> h0;
        for (int i = 0; i < 3; i++) {
            if (q_nearly_one_) {
                h0[i] = 1.0 / (global_dims_[i] - 1);
            } else {
                h0[i] = (q_ - 1) / (std::pow(q_, global_dims_[i] - 1) - 1);
            }
        }
        return h0;
    }

private:
    int comm_my_rank_;
    int comm_size_;
    std::array<int, 3> local_dims_;
    std::array<int, 3> proc_dims_;
    std::array<long, 3> global_dims_;
    const double domain_min_ = 0.0;
    const double domain_max_ = 1.0;
    double q_;
    bool q_nearly_one_;
    std::array<double, 3> h0_;

    bool ratio_nearly_one() const
    {
        if (std::abs(q_ - 1) < 1e-10) {
            return true;
        } else {
            return false;
        }
    }
};

inline int min3(int a, int b, int c) { return std::min(a, std::min(b, c)); }

inline int max3(int a, int b, int c) { return std::max(a, std::max(b, c)); }

std::array<int, 3> cubic_radical_search(const int n)
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

/**
 * Get 3D indices corresponding to a flattened 1D index in natural ordering.
 *
 * Note that in all arrays, index 0 is x, 1 is y and 2 is z axis.
 */
template <typename T>
inline T get_natural_flat_index_from_3d(const std::array<T, 3>& dims,
                                        const std::array<T, 3>& idx)
{
    return idx[2] * dims[1] * dims[0] + idx[1] * dims[0] + idx[0];
}

MulticolorOrdering compute_multicolor_ordering(
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

/// Sign of a number or zero
template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

template <typename global_idx_t>
std::enable_if_t<std::is_integral_v<global_idx_t>, int>
get_local_flat_from_global(const std::array<int, 3>& local_grid_dims,
                           const global_idx_t global_flat)
{
    const int local_size =
        local_grid_dims[0] * local_grid_dims[1] * local_grid_dims[2];
    return global_flat % local_size;
}

template <typename global_idx_t>
std::array<int, 3> get_proc_coords_from_global_point_flat(
    const std::array<int, 3> proc_dims,
    const std::array<int, 3>& local_grid_dims,
    const global_idx_t global_flat_id)
{
    const int local_size =
        local_grid_dims[0] * local_grid_dims[1] * local_grid_dims[2];
    const int rank_flat = global_flat_id / local_size;
    return get_natural_3d_indices_from_flat(proc_dims, rank_flat);
}

/**
 * Compute global index of a point from its local index.
 *
 * It is assumed that ranks are factored into 3 dimensions using the usual
 * natural ordering.
 *
 * @param owner_rank  Rank of the subdomain owning the given point.
 * @param comm_size_dir  Subdomain rank of the owning subdomain in each
 *                       logical direction.
 * @param local_grid_dims  The common size of the local grid of each subdomain.
 * @param local_flat_idx  Flattened local index of the point in question.
 */
template <typename global_idx_t>
std::enable_if_t<std::is_integral_v<global_idx_t>, global_idx_t>
get_global_from_local(const std::array<int, 3>& owner_rank,
                      const std::array<int, 3>& comm_size_dir,
                      const std::array<int, 3>& local_grid_dims,
                      const int local_flat_idx)
{
    const int flat_rank =
        get_natural_flat_index_from_3d(comm_size_dir, owner_rank);
    const int local_size =
        local_grid_dims[0] * local_grid_dims[1] * local_grid_dims[2];
    const global_idx_t global_base = static_cast<global_idx_t>(local_size) *
                                     static_cast<global_idx_t>(flat_rank);
    return global_base + local_flat_idx;
}

bool is_valid_point(const std::array<long, 3>& point,
                    const std::array<long, 3>& grid_dims)
{
    for (int i = 0; i < 3; i++) {
        if (point[i] < 0 || point[i] >= grid_dims[i]) {
            return false;
        }
    }
    return true;
}

bool is_boundary_point(const std::array<long, 3>& point,
                       const std::array<long, 3>& grid_dims)
{
    for (int i = 0; i < 3; i++) {
        if (point[i] == 0 || point[i] == grid_dims[i] - 1) {
            return true;
        }
    }
    return false;
}

/**
 * 7-point finite difference discretization of the Poisson equation
 * $ -div(grad(u)) = f $.
 */
template <typename scalar_t>
class PoissonFDDiscretization {
public:
    PoissonFDDiscretization(const Mesh* mesh) : mesh_{mesh} {}

    /**
     * Get diagonal value of the matrix corresponding to one row,
     * with the central point given.
     *
     * @param ipt  The global indices of the point corresponding to
     *             the desired row.
     */
    double diagonal_value(const std::array<long, 3>& gpt) const
    {
        double val{};
        for (int dir = 0; dir < 3; dir++) {
            const auto ri = mesh_->get_1D_point_coordinate(gpt[dir], dir);
            const auto rip1 = mesh_->get_1D_point_coordinate(gpt[dir] + 1, dir);
            const auto rim1 = mesh_->get_1D_point_coordinate(gpt[dir] - 1, dir);
            const auto dri = rip1 - ri;
            const auto drim1 = ri - rim1;
            val += 2.0 / (dri + drim1) * (1.0 / dri + 1.0 / drim1);
        }
        return val;
    }

    /**
     * Get off-diagonal value of the matrix corresponding to one row,
     * with the central point and neighboring offset given.
     *
     * @param gpt  The global indices of the point corresponding to
     *             the desired row.
     * @param d  Local offset (w.r.t point gpt) that defines the "neighboring"
     *           point that is the desired column.
     */
    double offdiagonal_value(const std::array<long, 3>& gpt,
                             const std::array<int, 3>& d) const
    {
        // For now, this is a 7-point stencil.
        std::array<int, 3> is_displaced = {d[0] != 0, d[1] != 0, d[2] != 0};
        const int disp_sum =
            is_displaced[0] + is_displaced[1] + is_displaced[2];
        if (disp_sum > 1) {
            return 0;
        }
        for (int dir = 0; dir < 3; dir++) {
            if (d[dir] == 0) {
                continue;
            }
            const auto ri = mesh_->get_1D_point_coordinate(gpt[dir], dir);
            const auto rip1 = mesh_->get_1D_point_coordinate(gpt[dir] + 1, dir);
            const auto rim1 = mesh_->get_1D_point_coordinate(gpt[dir] - 1, dir);
            const auto dri = rip1 - ri;
            const auto drim1 = ri - rim1;
            if (d[dir] < 0) {
                return -2.0 / (dri + drim1) * (1.0 / drim1);
            } else {
                return -2.0 / (dri + drim1) * (1.0 / dri);
            }
        }
        // If all neighbor increments are zero, return diagonal value.
        return diagonal_value(gpt);
    }

private:
    const Mesh* mesh_;
};

// ============================================================
// 3D 27-point stencil generator with 8-coloring
// ============================================================

/**
 * Builds matrix_data for a 3D 27-point stencil with rows locally ordered by
 * 8-color partitioning (color = (i%2) + 2*(j%2) + 4*(k%2)).  Within each
 * color in a particular subdomain, all nodes are independent under
 * the 27-point stencil.
 *
 * @param comm  Ginkgo MPI communicator.
 * @param local_grid_dims  Local grid dimensions (entry[0] is x-direction).
 *                         NOTE: assumed to be the same on all ranks.
 * @return  matrix nonzeros in the color-ordered layout.
 */
template <typename scalar_t, typename global_idx_t, typename local_idx_t>
inline ProblemData<scalar_t, global_idx_t, local_idx_t> generate_stencil_data(
    comm_t comm, const Mesh& mesh)
{
    const auto ordering = compute_multicolor_ordering(mesh.get_local_dims());
    const std::array<int, 3> proc_dims = cubic_radical_search(comm.size());
    if (comm.rank() == 0) {
        std::cout << "  Matrix generation: Process grid is factored into "
                  << proc_dims[0] << "x" << proc_dims[1] << "x" << proc_dims[2]
                  << std::endl;
    }
    const std::array<int, 3> my_ranks =
        get_natural_3d_indices_from_flat(proc_dims, comm.rank());
    const int ln = mesh.get_local_dims()[0] * mesh.get_local_dims()[1] *
                   mesh.get_local_dims()[2];
    const auto my_offsets = mult<global_idx_t>(my_ranks, mesh.get_local_dims());
    const auto global_grid_dims =
        mult<global_idx_t>(proc_dims, mesh.get_local_dims());
    const auto global_n =
        global_grid_dims[0] * global_grid_dims[1] * global_grid_dims[2];

    gko::matrix_data<scalar_t, global_idx_t> data(
        gko::dim<2>{static_cast<gko::size_type>(global_n),
                    static_cast<gko::size_type>(global_n)});
    data.nonzeros.reserve(7 * ln);
    std::vector<scalar_t> rhs(ln);
    std::vector<scalar_t> exact_soln(ln);

    const PoissonPDE<scalar_t> pde;
    const PoissonFDDiscretization<scalar_t> pdiscr(&mesh);

    scalar_t max_val{0.0}, min_val{100.0};
    for (int new_row = 0; new_row < ln; ++new_row) {
        const auto global_new_row = get_global_from_local<global_idx_t>(
            my_ranks, proc_dims, mesh.get_local_dims(), new_row);
        const int old_row = ordering.new_to_old[new_row];
        const std::array<int, 3> old_idx =
            get_natural_3d_indices_from_flat(mesh.get_local_dims(), old_row);
        const auto global_old_idx = add<global_idx_t>(my_offsets, old_idx);
        const auto ri = mesh.get_point_coordinates(global_old_idx);

        // Check for boundary
        if (is_boundary_point(global_old_idx, global_grid_dims)) {
            rhs[new_row] = pde.get_solution(ri);
            exact_soln[new_row] = pde.get_solution(ri);
            data.nonzeros.emplace_back(global_new_row, global_new_row, 1.0);
            continue;
        }

        // Compute RHS and exact solution
        rhs[new_row] = pde.get_forcing_function(ri);
        exact_soln[new_row] = pde.get_solution(ri);

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
                            nbd_old_idx[i] =
                                mesh.get_local_dims()[i] + nbd_old_idx[i];
                        } else if (nbd_old_idx[i] >= mesh.get_local_dims()[i]) {
                            nbd_rank[i]++;
                            nbd_old_idx[i] =
                                nbd_old_idx[i] % mesh.get_local_dims()[i];
                        }
                    }
                    const std::array<global_idx_t, 3> nbd_offsets =
                        mult<global_idx_t>(nbd_rank, mesh.get_local_dims());
                    const std::array<global_idx_t, 3> global_nbd_old_idx =
                        add<global_idx_t>(nbd_offsets, nbd_old_idx);
                    if (!is_valid_point(global_nbd_old_idx, global_grid_dims)) {
                        continue;
                    }
                    const local_idx_t local_new_flat_idx =
                        ordering.old_to_new[get_natural_flat_index_from_3d(
                            mesh.get_local_dims(), nbd_old_idx)];
                    const auto global_new_col =
                        get_global_from_local<global_idx_t>(
                            nbd_rank, proc_dims, mesh.get_local_dims(),
                            local_new_flat_idx);
                    const bool diag = (di == 0 && dj == 0 && dk == 0);
                    const auto val = static_cast<scalar_t>(
                        diag ? pdiscr.diagonal_value(global_old_idx)
                             : pdiscr.offdiagonal_value(global_old_idx, dst));
                    if (val != 0) {
                        max_val = std::max(max_val, std::abs(val));
                        min_val = std::min(min_val, std::abs(val));
                        data.nonzeros.emplace_back(global_new_row,
                                                   global_new_col, val);
                    }
                }
            }
        }
    }
    data.sort_row_major();
    if (comm.rank() == 0) {
        std::cout << "\n  Generated matrix values: max abs val = " << max_val
                  << ", min abs val = " << min_val << std::endl;
    }
    return ProblemData<scalar_t, global_idx_t, local_idx_t>{
        data, rhs, exact_soln, ordering.color_ptrs};
}

template <typename scalar_t, typename global_idx_t, typename local_idx_t>
ProblemData<scalar_t, global_idx_t, local_idx_t> generate_problem_data(
    comm_t comm, const Config& cfg)
{
    const Mesh mesh(comm, std::array<int, 3>{cfg.nx, cfg.ny, cfg.nz},
                    cfg.mesh_stretch_ratio);
    return generate_stencil_data<scalar_t, global_idx_t, local_idx_t>(comm,
                                                                      mesh);
}

template ProblemData<double, long, int> generate_problem_data(
    comm_t comm, const Config& cfg);

}  // namespace ex_dist_amp
