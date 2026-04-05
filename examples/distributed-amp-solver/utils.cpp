// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "utils.hpp"

namespace ex_dist_amp {

Config load_config(const std::string& path)
{
    Config cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: could not open '" << path
                  << "', using defaults.\n";
        throw std::runtime_error("Could not find options file.");
    }
    json j;
    f >> j;
    if (j.contains("nx")) cfg.nx = j["nx"];
    if (j.contains("ny")) cfg.ny = j["ny"];
    if (j.contains("nz")) cfg.nz = j["nz"];
    if (j.contains("mesh_stretch_ratio")) {
        cfg.mesh_stretch_ratio = j["mesh_stretch_ratio"];
    }
    if (j.contains("executor")) cfg.executor = j["executor"];
    if (j.contains("warmup_reps")) cfg.warmup_reps = j["warmup_reps"];
    if (j.contains("bench_reps")) cfg.bench_reps = j["bench_reps"];
    if (j.contains("solver_reps")) {
        cfg.solver_reps = j["solver_reps"];
    }
    if (j.contains("amp_tolerance"))
        cfg.amp_tolerance = j["amp_tolerance"].get<float>();
    if (j.contains("gmres_tol")) cfg.gmres_tol = j["gmres_tol"];
    if (j.contains("gmres_max_iters"))
        cfg.gmres_max_iters = j["gmres_max_iters"];
    if (j.contains("gmres_krylov_dim"))
        cfg.gmres_krylov_dim = j["gmres_krylov_dim"];
    if (j.contains("output_file_prefix")) {
        cfg.output_file_prefix = j["output_file_prefix"];
    }
    if (j.contains("amp_base_format")) {
        const std::string fmt = j["amp_base_format"];
        if (fmt == "ell" || fmt == "csr") {
            cfg.amp_base_format = fmt;
        } else {
            throw std::runtime_error("Invalid amp_base_format '" + fmt +
                                     "': supported values are 'ell' and 'csr'");
        }
    }
    return cfg;
}

std::shared_ptr<gko::Executor> make_executor(const std::string& name)
{
    auto ref = gko::ReferenceExecutor::create();
    if (name == "cuda") return gko::CudaExecutor::create(0, ref);
    if (name == "hip") return gko::HipExecutor::create(0, ref);
    if (name == "omp") return gko::OmpExecutor::create();
    if (name == "reference") return ref;
    throw std::runtime_error("Unknown executor: " + name);
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<gko::experimental::distributed::Matrix<
    ValueType, LocalIndexType, GlobalIndexType>>
create_dist_matrix(std::shared_ptr<const gko::Executor> exec, comm_t comm,
                   const Config& cfg)
{
    using DistMtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;
    if (cfg.amp_base_format == "csr") {
        return DistMtx::create(exec, comm,
                               gko::with_matrix_type<gko::matrix::Csr>());
    } else if (cfg.amp_base_format == "ell") {
        return DistMtx::create(exec, comm,
                               gko::with_matrix_type<gko::matrix::Ell>());
    } else {
        return nullptr;
    }
}

/**
 * Create an AMP distributed matrix.  The diagonal (local) block is an AMP
 * matrix whose bins use the configured base format, and the off-diagonal
 * (non-local) block uses CSR.
 */
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<gko::experimental::distributed::Matrix<
    ValueType, LocalIndexType, GlobalIndexType>>
create_amp_dist_matrix(std::shared_ptr<const gko::Executor> exec, comm_t comm,
                       const Config& cfg)
{
    using DistMtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;
    using Amp = gko::matrix::AMP<ValueType, LocalIndexType>;
    using Csr = gko::matrix::Csr<ValueType, LocalIndexType>;

    std::shared_ptr<const gko::LinOp> base_empty;
    if (cfg.amp_base_format == "csr") {
        base_empty = gko::share(Csr::create(exec, gko::dim<2>{0, 0}));
    } else {
        using Ell = gko::matrix::Ell<ValueType, LocalIndexType>;
        base_empty = gko::share(Ell::create(exec, gko::dim<2>{0, 0}));
    }
    auto amp_template = Amp::build()
                            .with_tolerance(cfg.amp_tolerance)
                            .with_strategy(Amp::tolerance_type::componentwise)
                            .on(exec)
                            ->generate(base_empty);
    auto csr_template = Csr::create(exec);
    return DistMtx::create(exec, comm, amp_template.get(), csr_template.get());
}

template std::unique_ptr<
    gko::experimental::distributed::Matrix<double, int, long>>
create_dist_matrix(std::shared_ptr<const gko::Executor> exec, comm_t comm,
                   const Config& cfg);

template std::unique_ptr<
    gko::experimental::distributed::Matrix<double, int, long>>
create_amp_dist_matrix(std::shared_ptr<const gko::Executor> exec, comm_t comm,
                       const Config& cfg);

double relative_error(std::shared_ptr<const gko::Executor> exec,
                      const gkodist::Vector<double>* const x,
                      const gkodist::Vector<double>* const ref_vec)
{
    using Vec = gko::matrix::Dense<double>;
    using RVec = gko::matrix::Dense<gko::remove_complex<double>>;

    auto diff = gko::clone(exec, x);
    auto neg_one = gko::initialize<Vec>({-1.0}, exec);
    diff->add_scaled(neg_one, ref_vec);  // diff = x - ref_vec

    auto diff_norm = RVec::create(exec, gko::dim<2>{1, 1});
    auto ref_norm = RVec::create(exec, gko::dim<2>{1, 1});
    diff->compute_norm2(diff_norm);
    ref_vec->compute_norm2(ref_norm);

    auto master = exec->get_master();
    auto dn = gko::clone(master, diff_norm);
    auto rn = gko::clone(master, ref_norm);
    double rn_val = rn->at(0, 0);
    return (rn_val > 0.0) ? (dn->at(0, 0) / rn_val) : dn->at(0, 0);
}

std::string compute_amp_details(const Config& cfg,
                                const gko::matrix::AMP<double, int>* const mtx,
                                const double amp_setup_ms, json& rows)
{
    std::stringstream sstream;
    using Ell = gko::matrix::Ell<double, int>;
    using Csr = gko::matrix::Csr<double, int>;
    constexpr int q = gko::matrix::AMP<double, int>::num_precisions;
    sstream << "\nAMP details";
    sstream << "\n  AMP matrix precision buckets:\n";
    json amps = json::array();
    for (int k = 0; k < q; k++) {
        const auto* bin = mtx->get_bin_matrix(k);
        GKO_ASSERT(bin);
        if (cfg.amp_base_format == "ell") {
            // Apparently, this works:
            const auto* ellmat = static_cast<const Ell*>(bin);
            const auto max_nnz_per_row =
                ellmat->get_num_stored_elements_per_row();
            sstream << "     Bin " << k
                    << ": max_nnz_per_row = " << max_nnz_per_row << "\n";
            amps.push_back({{"bin", k}, {"max_nnz_per_row", max_nnz_per_row}});
        } else if (cfg.amp_base_format == "csr") {
            const auto* csrmat = static_cast<const Csr*>(bin);
            const auto nnz = csrmat->get_num_stored_elements();
            sstream << "     Bin " << k << ": nnz = " << nnz << "\n";
            amps.push_back({{"bin", k}, {"nnz", nnz}});
        }
    }
    amps.push_back({"amp_setup_ms", amp_setup_ms});
    sstream << "  AMP setup time (ms) = " << std::setprecision(3)
            << amp_setup_ms << "\n";
    rows.back()["amp_details"] = amps;
    return sstream.str();
}

}  // namespace ex_dist_amp
