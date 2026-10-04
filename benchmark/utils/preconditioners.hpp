// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_
#define GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_


#include <map>
#include <string>

#include <gflags/gflags.h>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/overhead_linop.hpp"
#include "benchmark/utils/types.hpp"


// MSVC has different way to expand macro than linux, so we can not put the #if
// inside the DEFINE_string macro
#define PRECONDITIONERS_COMMON                                        \
    "A comma-separated list of preconditioners to use. "              \
    "Supported values are: none, jacobi, mg, paric, parict, parilu, " \
    "parilut, ic, ilu, paric-isai, parict-isai, parilu-isai, "        \
    "parilut-isai, ic-isai, ilu-isai, fgs, sor, overhead"
#if GINKGO_BUILD_MPI
DEFINE_string(preconditioners, "none",
              PRECONDITIONERS_COMMON
              ", schwarz-jacobi, schwarz-ilu, schwarz-ic, schwarz-lu");
#else
DEFINE_string(preconditioners, "none", PRECONDITIONERS_COMMON);
#endif

#undef PRECONDITIONERS_COMMON

DEFINE_uint32(parilu_iterations, 5,
              "The number of iterations for ParIC(T)/ParILU(T)");

DEFINE_bool(parilut_approx_select, true,
            "Use approximate selection for ParICT/ParILUT");

DEFINE_double(parilut_limit, 2.0, "The fill-in limit for ParICT/ParILUT");

DEFINE_int32(
    isai_power, 1,
    "Which power of the sparsity structure to use for ISAI preconditioners");

DEFINE_string(jacobi_storage, "0,0",
              "Defines the kind of storage optimization to perform on "
              "preconditioners that support it. Supported values are: "
              "autodetect and <X>,<Y> where <X> and <Y> are the input "
              "parameters used to construct a precision_reduction object.");

DEFINE_double(jacobi_accuracy, 1e-1,
              "This value is used as the accuracy flag of the adaptive Jacobi "
              "preconditioner.");

DEFINE_uint32(jacobi_max_block_size, 32,
              "Maximal block size of the block-Jacobi preconditioner");

DEFINE_double(sor_relaxation_factor, 1.0,
              "The relaxation factor for the SOR preconditioner");

DEFINE_bool(sor_symmetric, false,
            "Apply the SOR preconditioner symmetrically, i.e. use SSOR");

DEFINE_bool(pgm_deterministic, false,
            "Use deterministic computation of the aggregated group within PGM");

DEFINE_uint32(
    mg_max_num_levels, false,
    "The maximum number of levels to use for the Multigrid preconditioner");

DEFINE_double(mg_tolerance, false, "The tolerance for the coarse solver");

DEFINE_uint32(mg_max_iters, false,
              "The max number of iterations for the coarse solver");

DEFINE_uint32(fgs_sweeps, 1,
              "Number of forward Gauss-Seidel sweeps per preconditioner "
              "application (requires --reorder=multicolor)");


// parses the Jacobi storage optimization command line argument
gko::precision_reduction parse_storage_optimization(const std::string& flag)
{
    if (flag == "autodetect") {
        return gko::precision_reduction::autodetect();
    }
    const auto parts = split(flag, ',');
    if (parts.size() != 2) {
        throw std::runtime_error(
            "storage_optimization has to be a list of two integers");
    }
    return gko::precision_reduction(std::stoi(parts[0]), std::stoi(parts[1]));
}


/**
 * Arguments passed to each factory returned by get_precond_factory.
 *
 * Most preconditioners only use exec; some, like FGS, additionally require
 * color_ptrs from a prior multicolor reordering.
 */
struct PrecondArgs {
    std::shared_ptr<const gko::Executor> exec;
    std::vector<itype> color_ptrs;
};


template <typename ValueType>
std::function<std::unique_ptr<gko::LinOpFactory>(const PrecondArgs&)>
get_precond_factory(const std::string& prec)
{
    using rc_type = gko::remove_complex<ValueType>;
    using lower_trs = gko::solver::LowerTrs<ValueType, itype>;
    using upper_trs = gko::solver::UpperTrs<ValueType, itype>;
    using lower_isai = gko::preconditioner::LowerIsai<ValueType, itype>;
    using upper_isai = gko::preconditioner::UpperIsai<ValueType, itype>;

    if (prec == "none") {
        return [](const PrecondArgs& args) {
            return gko::matrix::IdentityFactory<ValueType>::create(args.exec);
        };
    } else if (prec == "jacobi") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            return gko::preconditioner::Jacobi<ValueType, itype>::build()
                .with_max_block_size(FLAGS_jacobi_max_block_size)
                .with_storage_optimization(
                    parse_storage_optimization(FLAGS_jacobi_storage))
                .with_accuracy(static_cast<rc_type>(FLAGS_jacobi_accuracy))
                .with_skip_sorting(true)
                .on(exec);
        };
    } else if (prec == "paric") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact =
                gko::share(gko::factorization::ParIc<ValueType, itype>::build()
                               .with_iterations(FLAGS_parilu_iterations)
                               .with_skip_sorting(true)
                               .on(exec));
            return gko::preconditioner::Ic<lower_trs, itype>::build()
                .with_factorization(fact)
                .on(exec);
        };
    } else if (prec == "parict") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::factorization::ParIct<ValueType, itype>::build()
                    .with_iterations(FLAGS_parilu_iterations)
                    .with_approximate_select(FLAGS_parilut_approx_select)
                    .with_fill_in_limit(FLAGS_parilut_limit)
                    .with_skip_sorting(true)
                    .on(exec));
            return gko::preconditioner::Ilu<lower_trs, upper_trs, false,
                                            itype>::build()
                .with_factorization(fact)
                .on(exec);
        };
    } else if (prec == "parilu") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact =
                gko::share(gko::factorization::ParIlu<ValueType, itype>::build()
                               .with_iterations(FLAGS_parilu_iterations)
                               .with_skip_sorting(true)
                               .on(exec));
            return gko::preconditioner::Ilu<lower_trs, upper_trs, false,
                                            itype>::build()
                .with_factorization(fact)
                .on(exec);
        };
    } else if (prec == "parilut") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::factorization::ParIlut<ValueType, itype>::build()
                    .with_iterations(FLAGS_parilu_iterations)
                    .with_approximate_select(FLAGS_parilut_approx_select)
                    .with_fill_in_limit(FLAGS_parilut_limit)
                    .with_skip_sorting(true)
                    .on(exec));
            return gko::preconditioner::Ilu<lower_trs, upper_trs, false,
                                            itype>::build()
                .with_factorization(fact)
                .on(exec);
        };
    } else if (prec == "ic") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::factorization::Ic<ValueType, itype>::build().on(exec));
            return gko::preconditioner::Ic<lower_trs, itype>::build()
                .with_factorization(fact)
                .on(exec);
        };
    } else if (prec == "ilu") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::factorization::Ilu<ValueType, itype>::build().on(exec));
            return gko::preconditioner::Ilu<lower_trs, upper_trs, false,
                                            itype>::build()
                .with_factorization(fact)
                .on(exec);
        };
    } else if (prec == "paric-isai") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact =
                gko::share(gko::factorization::ParIc<ValueType, itype>::build()
                               .with_iterations(FLAGS_parilu_iterations)
                               .with_skip_sorting(true)
                               .on(exec));
            auto lisai = gko::share(lower_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            return gko::preconditioner::Ic<lower_isai, itype>::build()
                .with_factorization(fact)
                .with_l_solver(lisai)
                .on(exec);
        };
    } else if (prec == "parict-isai") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::factorization::ParIct<ValueType, itype>::build()
                    .with_iterations(FLAGS_parilu_iterations)
                    .with_approximate_select(FLAGS_parilut_approx_select)
                    .with_fill_in_limit(FLAGS_parilut_limit)
                    .with_skip_sorting(true)
                    .on(exec));
            auto lisai = gko::share(lower_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            return gko::preconditioner::Ic<lower_isai, itype>::build()
                .with_factorization(fact)
                .with_l_solver(lisai)
                .on(exec);
        };
    } else if (prec == "parilu-isai") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact =
                gko::share(gko::factorization::ParIlu<ValueType, itype>::build()
                               .with_iterations(FLAGS_parilu_iterations)
                               .with_skip_sorting(true)
                               .on(exec));
            auto lisai = gko::share(lower_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            auto uisai = gko::share(upper_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            return gko::preconditioner::Ilu<lower_isai, upper_isai, false,
                                            itype>::build()
                .with_factorization(fact)
                .with_l_solver(lisai)
                .with_u_solver(uisai)
                .on(exec);
        };
    } else if (prec == "parilut-isai") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::factorization::ParIlut<ValueType, itype>::build()
                    .with_iterations(FLAGS_parilu_iterations)
                    .with_approximate_select(FLAGS_parilut_approx_select)
                    .with_fill_in_limit(FLAGS_parilut_limit)
                    .with_skip_sorting(true)
                    .on(exec));
            auto lisai = gko::share(lower_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            auto uisai = gko::share(upper_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            return gko::preconditioner::Ilu<lower_isai, upper_isai, false,
                                            itype>::build()
                .with_factorization(fact)
                .with_l_solver(lisai)
                .with_u_solver(uisai)
                .on(exec);
        };
    } else if (prec == "ic-isai") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::factorization::Ic<ValueType, itype>::build().on(exec));
            auto lisai = gko::share(lower_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            return gko::preconditioner::Ic<lower_isai, itype>::build()
                .with_factorization(fact)
                .with_l_solver(lisai)
                .on(exec);
        };
    } else if (prec == "ilu-isai") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::factorization::Ilu<ValueType, itype>::build().on(exec));
            auto lisai = gko::share(lower_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            auto uisai = gko::share(upper_isai::build()
                                        .with_sparsity_power(FLAGS_isai_power)
                                        .on(exec));
            return gko::preconditioner::Ilu<lower_isai, upper_isai, false,
                                            itype>::build()
                .with_factorization(fact)
                .with_l_solver(lisai)
                .with_u_solver(uisai)
                .on(exec);
        };
    } else if (prec == "general-isai") {
        return [](const PrecondArgs& args) {
            return gko::preconditioner::GeneralIsai<ValueType, itype>::build()
                .with_sparsity_power(FLAGS_isai_power)
                .on(args.exec);
        };
    } else if (prec == "spd-isai") {
        return [](const PrecondArgs& args) {
            return gko::preconditioner::SpdIsai<ValueType, itype>::build()
                .with_sparsity_power(FLAGS_isai_power)
                .on(args.exec);
        };
    } else if (prec == "fgs") {
        return [](const PrecondArgs& args) {
            if (args.color_ptrs.empty()) {
                throw std::runtime_error{
                    "fgs preconditioner requires --reorder=multicolor"};
            }
            return gko::solver::FwdGaussSeidel<ValueType, itype>::build()
                .with_criteria(gko::stop::Iteration::build()
                                   .with_max_iters(FLAGS_fgs_sweeps)
                                   .on(args.exec))
                .with_color_ptrs(args.color_ptrs)
                .on(args.exec);
        };
    } else if (prec == "sor") {
        return [](const PrecondArgs& args) {
            return gko::preconditioner::Sor<ValueType, itype>::build()
                .with_relaxation_factor(
                    static_cast<rc_type>(FLAGS_sor_relaxation_factor))
                .with_symmetric(FLAGS_sor_symmetric)
                .on(args.exec);
        };
    } else if (prec == "overhead") {
        return [](const PrecondArgs& args) {
            return gko::Overhead<ValueType>::build()
                .with_criteria(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_reduction_factor(rc_type{}))
                .on(args.exec);
        };
    } else if (prec == "mg") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            using ir = gko::solver::Ir<ValueType>;
            auto iter_stop = gko::share(gko::stop::Iteration::build()
                                            .with_max_iters(FLAGS_mg_max_iters)
                                            .on(exec));
            auto tol_stop =
                gko::share(gko::stop::ResidualNorm<ValueType>::build()
                               .with_baseline(gko::stop::mode::absolute)
                               .with_reduction_factor(FLAGS_mg_tolerance)
                               .on(exec));
            return gko::solver::Multigrid::build()
                .with_mg_level(gko::multigrid::Pgm<ValueType, itype>::build()
                                   .with_deterministic(FLAGS_pgm_deterministic))
                .with_criteria(iter_stop, tol_stop)
                .with_max_levels(FLAGS_mg_max_num_levels)
                .on(exec);
        };
#if GINKGO_BUILD_MPI
    } else if (prec == "schwarz-jacobi") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            return gko::experimental::distributed::preconditioner::Schwarz<
                       ValueType>::build()
                .with_local_solver(
                    gko::preconditioner::Jacobi<ValueType>::build()
                        .with_max_block_size(FLAGS_jacobi_max_block_size)
                        .with_storage_optimization(
                            parse_storage_optimization(FLAGS_jacobi_storage))
                        .with_accuracy(
                            static_cast<rc_type>(FLAGS_jacobi_accuracy))
                        .with_skip_sorting(true)
                        .on(exec))
                .on(exec);
        };
    } else if (prec == "schwarz-general-isai") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            return gko::experimental::distributed::preconditioner::Schwarz<
                       ValueType, itype>::build()
                .with_local_solver(
                    gko::preconditioner::GeneralIsai<ValueType, itype>::build()
                        .with_sparsity_power(FLAGS_isai_power)
                        .on(exec))
                .on(exec);
        };
    } else if (prec == "schwarz-spd-isai") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            return gko::experimental::distributed::preconditioner::Schwarz<
                       ValueType, itype>::build()
                .with_local_solver(
                    gko::preconditioner::SpdIsai<ValueType, itype>::build()
                        .with_sparsity_power(FLAGS_isai_power)
                        .on(exec))
                .on(exec);
        };
    } else if (prec == "schwarz-ilu") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact =
                gko::share(gko::factorization::Ilu<ValueType, itype>::build()
                               .with_skip_sorting(true)
                               .on(exec));
            return gko::experimental::distributed::preconditioner::Schwarz<
                       ValueType, itype>::build()
                .with_local_solver(
                    gko::preconditioner::Ilu<lower_trs, upper_trs, false,
                                             itype>::build()
                        .with_factorization(fact)
                        .on(exec))
                .on(exec);
        };
    } else if (prec == "schwarz-ic") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact =
                gko::share(gko::factorization::Ic<ValueType, itype>::build()
                               .with_skip_sorting(true)
                               .on(exec));
            return gko::experimental::distributed::preconditioner::Schwarz<
                       ValueType, itype>::build()
                .with_local_solver(
                    gko::preconditioner::Ic<lower_trs, itype>::build()
                        .with_factorization(fact)
                        .on(exec))
                .on(exec);
        };
    } else if (prec == "schwarz-lu") {
        return [](const PrecondArgs& args) {
            const auto& exec = args.exec;
            auto fact = gko::share(
                gko::experimental::factorization::Lu<ValueType, itype>::build()
                    .on(exec));
            return gko::experimental::distributed::preconditioner::Schwarz<
                       ValueType, itype>::build()
                .with_local_solver(
                    gko::experimental::solver::Direct<ValueType, itype>::build()
                        .with_factorization(fact)
                        .on(exec))
                .on(exec);
        };
#endif
    }
    throw std::out_of_range("Unknown preconditioner: " + prec);
}

#endif  // GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_
