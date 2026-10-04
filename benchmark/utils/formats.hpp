// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_FORMATS_HPP_
#define GKO_BENCHMARK_UTILS_FORMATS_HPP_


#include <algorithm>
#include <map>
#include <string>

#include <gflags/gflags.h>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/json.hpp"
#include "benchmark/utils/sparselib_linops.hpp"
#include "benchmark/utils/types.hpp"


namespace formats {


std::string available_format =
    "coo, csr, csrc, csri, csrm, csrs, ell, ell_mixed, sellp, hybrid, "
    "hybrid0, hybrid25, hybrid33, "
    "hybrid40, "
    "hybrid60, hybrid80, hybridlimit0, hybridlimit25, hybridlimit33, "
    "hybridminstorage, amp, ampib"
#ifdef HAS_CUDA
    ", cusparse_csr, cusparse_csrex, cusparse_coo"
    ", cusparse_csrmp, cusparse_csrmm, cusparse_ell, cusparse_hybrid"
    ", cusparse_gcsr, cusparse_gcsr2, cusparse_gcoo"
#endif  // HAS_CUDA
#ifdef HAS_HIP
    ", hipsparse_csr, hipsparse_csrmm, hipsparse_coo, hipsparse_ell, "
    "hipsparse_hybrid"
#endif  // HAS_HIP
#ifdef HAS_DPCPP
    ", onemkl_csr, onemkl_optimized_csr"
#endif  // HAS_DPCPP
    ".\n";

std::string format_description =
    "coo: Coordinate storage. The GPU kernels use the load-balancing "
    "approach\n"
    "     suggested in Flegar et al.: Overcoming Load Imbalance for\n"
    "     Irregular Sparse Matrices.\n"
    "csr: Compressed Sparse Row storage. Ginkgo implementation with\n"
    "     automatic strategy.\n"
    "csrc: Ginkgo's CSR implementation with classical strategy.\n"
    "csri: Ginkgo's CSR implementation with load_balance (imbalance) "
    "strategy.\n"
    "csrm: Ginkgo's CSR implementation with merge_path strategy.\n"
    "csrs: Ginkgo's CSR implementation with sparselib strategy.\n"
    "ell: Ellpack format according to Bell and Garland: Efficient Sparse\n"
    "     Matrix-Vector Multiplication on CUDA.\n"
    "ell_mixed: Mixed Precision Ellpack format according to Bell and Garland:\n"
    "           Efficient Sparse Matrix-Vector Multiplication on CUDA.\n"
    "sellp: Sliced Ellpack uses a default block size of 32.\n"
    "hybrid: Hybrid uses ELL and COO to represent the matrix.\n"
    "hybrid0, hybrid25, hybrid33, hybrid40, hybrid60, hybrid80:\n"
    "    Use 0%, 25%, ... quantiles of the row length distribution\n"
    "    to choose number of entries stored in the ELL part.\n"
    "hybridlimit0, hybridlimit25, hybrid33: Similar to hybrid0\n"
    "    but with an additional absolute limit on the number of entries\n"
    "    per row stored in ELL.\n"
    "hybridminstorage: Use the minimal storage to store the matrix.\n"
    "amp: Adaptive Mixed Precision format. Sorts nonzeros into bins of\n"
    "     different precisions (FP64/FP32/BF16/FP16). Base format is\n"
    "     controlled by --amp_base_type (ell or csr), tolerance type by\n"
    "     --amp_tolerance_type (componentwise or normwise), and tolerance\n"
    "     value by --amp_tolerance. A sparse trailing (lowest-precision)\n"
    "     bin is folded into the next higher precision bin below the\n"
    "     --amp_bin_foldup_nnz_ratio threshold. Diagonal entries are always\n"
    "     placed in the highest precision (FP64) bin unless\n"
    "     --amp_high_precision_diagonal=false. Uses the monolithic_classical\n"
    "     SpMV strategy: a single kernel reads all precision buckets and\n"
    "     accumulates each row.\n"
    "     Note: AMP[CSR] uses a classical-style SpMV kernel internally, so\n"
    "     compare against csrc (classical) rather than csr (automatical)\n"
    "     for a like-for-like fixed-precision baseline.\n"
    "ampib: AMP with the independent_buckets SpMV strategy -- one SpMV per\n"
    "       precision bucket, accumulated into the output vector. Same\n"
    "       storage (and --amp_* flags) as amp; differs only in the apply.\n"
    "       The SpMV strategy of each CSR precision bucket is controlled by\n"
    "       --amp_csr_strategy."
#ifdef HAS_CUDA
    "\n"
    "cusparse_coo: cuSPARSE COO SpMV, using cusparseXhybmv with \n"
    "              CUSPARSE_HYB_PARTITION_USER for CUDA < 10.2, or\n"
    "              the Generic API otherwise\n"
    "cusparse_csr: cuSPARSE CSR SpMV, using cusparseXcsrmv for CUDA < 10.2,\n"
    "              or the Generic API with default algorithm otherwise\n"
    "cusparse_csrex: cuSPARSE CSR SpMV using cusparseXcsrmvEx\n"
    "cusparse_ell: cuSPARSE ELL SpMV using cusparseXhybmv with\n"
    "              CUSPARSE_HYB_PARTITION_MAX, available for CUDA < 11.0\n"
    "cusparse_csrmp: cuSPARSE CSR SpMV using cusparseXcsrmv_mp,\n"
    "                available for CUDA < 11.0\n"
    "cusparse_csrmm: cuSPARSE CSR SpMV using cusparseXcsrmv_mm,\n"
    "                available for CUDA < 11.0\n"
    "cusparse_hybrid: cuSPARSE Hybrid SpMV using cusparseXhybmv\n"
    "                 with an automatic partition, available for CUDA < 11.0\n"
    "cusparse_gcsr: cuSPARSE CSR SpMV using Generic API with default\n"
    "               algorithm, available for CUDA >= 10.2\n"
    "cusparse_gcsr2: cuSPARSE CSR SpMV using Generic API with\n"
    "                CUSPARSE_CSRMV_ALG2, available for CUDA >= 10.2\n"
    "cusparse_gcoo: cuSPARSE Generic API with default COO SpMV,\n"
    "               available for CUDA >= 10.2\n"
#endif  // HAS_CUDA
#ifdef HAS_HIP
    "\n"
    "hipsparse_csr: hipSPARSE CSR SpMV using hipsparseXcsrmv\n"
    "hipsparse_csrmm: hipSPARSE CSR SpMV using hipsparseXcsrmv_mm\n"
    "hipsparse_hybrid: hipSPARSE CSR SpMV using hipsparseXhybmv\n"
    "                  with an automatic partition\n"
    "hipsparse_coo: hipSPARSE CSR SpMV using hipsparseXhybmv\n"
    "               with HIPSPARSE_HYB_PARTITION_USER\n"
    "hipsparse_ell: hipSPARSE CSR SpMV using hipsparseXhybmv\n"
    "               with HIPSPARSE_HYB_PARTITION_MAX\n"
#endif  // HAS_HIP
#ifdef HAS_DPCPP
    "onemkl_csr: oneMKL Csr SpMV\n"
    "onemkl_optimized_csr: oneMKL optimized Csr SpMV using optimize_gemv after "
    "reading the matrix"
#endif  // HAS_DPCPP
    ;

std::string format_command =
    "A comma-separated list of formats to run. Supported values are: " +
    available_format + format_description;


}  // namespace formats


// the formats command-line argument
DEFINE_string(formats, "coo", formats::format_command.c_str());

DEFINE_int64(ell_imbalance_limit, 100,
             "Maximal storage overhead above which ELL benchmarks will be "
             "skipped. Negative values mean no limit.");

DEFINE_double(amp_tolerance, 1e-14,
              "Backward error (componentwise/normwise) "
              "tolerance for AMP matrix type.");

DEFINE_string(amp_base_type, "ell",
              "Base matrix format for AMP: \"ell\" or \"csr\".");

DEFINE_string(amp_tolerance_type, "componentwise",
              "Tolerance type for AMP: \"componentwise\" or \"normwise\".");

DEFINE_string(amp_csr_strategy, "automatical",
              "SpMV strategy applied to each CSR precision bucket of an AMP "
              "matrix: \"automatical\", \"classical\", \"load_balance\" or "
              "\"merge_path\". Only has an effect for the \"ampib\" format "
              "with --amp_base_type=csr; the monolithic kernel used by "
              "\"amp\" never consults the buckets' strategies.");

DEFINE_double(amp_bin_foldup_nnz_ratio, 0.01,
              "Threshold, as a ratio of the original matrix's number of "
              "nonzeros, below which a trailing (lowest-precision) bin of "
              "an AMP matrix is folded into the next higher precision bin "
              "instead of being generated on its own. 0 disables folding.");

DEFINE_bool(amp_high_precision_diagonal, true,
            "Whether diagonal entries of an AMP matrix are always placed in "
            "the highest precision (FP64) bin regardless of magnitude. If "
            "false, diagonal entries are binned by magnitude like any other "
            "entry.");


namespace formats {


// some shortcuts
// using hybrid = gko::matrix::Hybrid<etype, itype>;
template <typename vtype>
using csr = gko::matrix::Csr<vtype, itype>;
// using coo = gko::matrix::Coo<etype, itype>;
// using ell = gko::matrix::Ell<etype, itype>;
// using ell_mixed = gko::matrix::Ell<gko::next_precision_base<etype>, itype>;

template <typename vtype>
using amp_type = gko::matrix::AMP<vtype, itype>;


/**
 * Parses the SpMV strategy applied to each CSR precision bucket of an AMP
 * matrix from a string. Supports "automatical", "classical", "load_balance"
 * and "merge_path". "sparselib" is a valid gko::matrix::amp_csr_strategy_type
 * enumerator but is deliberately not offered here, since it does not support
 * every precision bucket (e.g. half or bfloat16) and would fail at apply
 * time; unlike the CSR bucket's own value type, which is chosen internally
 * by AMP itself, this cannot be gated in advance.
 *
 * @throws gko::Error if the string does not match a supported value.
 */
template <typename ValueType>
typename amp_type<ValueType>::csr_strategy_type parse_amp_csr_strategy(
    const std::string& s)
{
    if (s == "automatical") {
        return amp_type<ValueType>::csr_strategy_type::automatical;
    } else if (s == "classical") {
        return amp_type<ValueType>::csr_strategy_type::classical;
    } else if (s == "load_balance") {
        return amp_type<ValueType>::csr_strategy_type::load_balance;
    } else if (s == "merge_path") {
        return amp_type<ValueType>::csr_strategy_type::merge_path;
    } else {
        throw gko::Error(__FILE__, __LINE__,
                         "Invalid amp_csr_strategy '" + s +
                             "': supported values are 'automatical', "
                             "'classical', 'load_balance' and 'merge_path'");
    }
}


template <typename ValueType>
std::string to_string(typename amp_type<ValueType>::csr_strategy_type s)
{
    switch (s) {
    case amp_type<ValueType>::csr_strategy_type::classical:
        return "classical";
    case amp_type<ValueType>::csr_strategy_type::load_balance:
        return "load_balance";
    case amp_type<ValueType>::csr_strategy_type::merge_path:
        return "merge_path";
    case amp_type<ValueType>::csr_strategy_type::sparselib:
        return "sparselib";
    case amp_type<ValueType>::csr_strategy_type::automatical:
    default:
        return "automatical";
    }
}


/**
 * Creates a CSR strategy of the given type for the given executor if possible,
 * falls back to csr::classical for executors without support for this strategy.
 *
 * @tparam Strategy  one of csr::automatical or csr::load_balance
 */
template <typename Strategy, typename ValueType>
std::shared_ptr<typename csr<ValueType>::strategy_type> create_gpu_strategy(
    std::shared_ptr<const gko::Executor> exec)
{
    if (auto cuda = dynamic_cast<const gko::CudaExecutor*>(exec.get())) {
        return std::make_shared<Strategy>(cuda->shared_from_this());
    } else if (auto hip = dynamic_cast<const gko::HipExecutor*>(exec.get())) {
        return std::make_shared<Strategy>(hip->shared_from_this());
    } else if (auto dpcpp =
                   dynamic_cast<const gko::DpcppExecutor*>(exec.get())) {
        return std::make_shared<Strategy>(dpcpp->shared_from_this());
    } else {
        return std::make_shared<typename csr<ValueType>::classical>();
    }
}


/**
 * Checks whether the given matrix data exceeds the ELL imbalance limit set by
 * the --ell_imbalance_limit flag
 *
 * @throws gko::Error if the imbalance limit is exceeded
 */
template <typename ValueType>
void check_ell_admissibility(const gko::matrix_data<ValueType, itype>& data)
{
    if (data.size[0] == 0 || FLAGS_ell_imbalance_limit < 0) {
        return;
    }
    std::vector<gko::size_type> row_lengths(data.size[0]);
    for (auto nz : data.nonzeros) {
        row_lengths[nz.row]++;
    }
    auto max_len = *std::max_element(row_lengths.begin(), row_lengths.end());
    auto avg_len = data.nonzeros.size() / std::max<double>(data.size[0], 1);
    if (max_len / avg_len > FLAGS_ell_imbalance_limit) {
        throw gko::Error(__FILE__, __LINE__,
                         "Matrix exceeds ELL imbalance limit");
    }
}


template <typename MatrixType, typename... Args>
auto create_matrix_type(Args&&... args)
{
    return [=](std::shared_ptr<const gko::Executor> exec)
               -> std::unique_ptr<MatrixType> {
        return MatrixType::create(std::move(exec), args...);
    };
}


template <typename MatrixType, typename Strategy>
auto create_matrix_type_with_gpu_strategy()
{
    using vtype = typename MatrixType::value_type;
    return [&](std::shared_ptr<const gko::Executor> exec)
               -> std::unique_ptr<MatrixType> {
        return MatrixType::create(exec,
                                  create_gpu_strategy<Strategy, vtype>(exec));
    };
}


template <typename ValueType>
std::function<std::unique_ptr<gko::LinOp>(std::shared_ptr<const gko::Executor>)>
get_matrix_factory(const std::string& mattype)
{
    using hybrid = gko::matrix::Hybrid<ValueType, itype>;
    using csr = gko::matrix::Csr<ValueType, itype>;
    using coo = gko::matrix::Coo<ValueType, itype>;
    using ell = gko::matrix::Ell<ValueType, itype>;
    using amp_type = gko::matrix::AMP<ValueType, itype>;

    if (mattype == "csr") {
        return create_matrix_type_with_gpu_strategy<
            csr, typename csr::automatical>();
    } else if (mattype == "csri") {
        return create_matrix_type_with_gpu_strategy<
            csr, typename csr::load_balance>();
    } else if (mattype == "csrm") {
        return create_matrix_type<csr>(
            std::make_shared<typename csr::merge_path>());
    } else if (mattype == "csrc") {
        return create_matrix_type<csr>(
            std::make_shared<typename csr::classical>());
    } else if (mattype == "csrs") {
        return create_matrix_type<csr>(
            std::make_shared<typename csr::sparselib>());
    } else if (mattype == "coo") {
        return create_matrix_type<coo>();
    } else if (mattype == "ell") {
        return create_matrix_type<ell>();
    } else if (mattype == "ell_mixed") {
        if constexpr (std::is_same_v<gko::remove_complex<ValueType>, double> ||
                      std::is_same_v<gko::remove_complex<ValueType>, float>) {
            using ell_mixed =
                gko::matrix::Ell<gko::next_precision_base<ValueType>, itype>;
            return create_matrix_type<ell_mixed>();
        } else {
            std::runtime_error(
                "ell_mixed is only supported for float and double!");
        }
#ifdef HAS_CUDA
    } else if (mattype == "cusparse_csr") {
        return create_sparselib_linop<cusparse_csr>;
    } else if (mattype == "cusparse_csrmp") {
        return create_sparselib_linop<cusparse_csrmp>;
    } else if (mattype == "cusparse_csrmm") {
        return create_sparselib_linop<cusparse_csrmm>;
    } else if (mattype == "cusparse_hybrid") {
        return create_sparselib_linop<cusparse_hybrid>;
    } else if (mattype == "cusparse_coo") {
        return create_sparselib_linop<cusparse_coo>;
    } else if (mattype == "cusparse_ell") {
        return create_sparselib_linop<cusparse_ell>;
    } else if (mattype == "cusparse_csrex") {
        return create_sparselib_linop<cusparse_csrex>;
    } else if (mattype == "cusparse_gcsr") {
        return create_sparselib_linop<cusparse_gcsr>;
    } else if (mattype == "cusparse_gcsr2") {
        return create_sparselib_linop<cusparse_gcsr2>;
    } else if (mattype == "cusparse_gcoo") {
        return create_sparselib_linop<cusparse_gcoo>;
#endif  // HAS_CUDA
#ifdef HAS_HIP
    } else if (mattype == "hipsparse_csr") {
        return create_sparselib_linop<hipsparse_csr>;
    } else if (mattype == "hipsparse_csrmm") {
        return create_sparselib_linop<hipsparse_csrmm>;
    } else if (mattype == "hipsparse_hybrid") {
        return create_sparselib_linop<hipsparse_hybrid>;
    } else if (mattype == "hipsparse_coo") {
        return create_sparselib_linop<hipsparse_coo>;
    } else if (mattype == "hipsparse_ell") {
        return create_sparselib_linop<hipsparse_ell>;
#endif  // HAS_HIP
#ifdef HAS_DPCPP
    } else if (mattype == "onemkl_csr") {
        return create_sparselib_linop<onemkl_csr>;
    } else if (mattype == "onemkl_optimized_csr") {
        return create_sparselib_linop<onemkl_optimized_csr>;
#endif  // HAS_DPCPP
    } else if (mattype == "hybrid") {
        return create_matrix_type<hybrid>();
    } else if (mattype == "hybrid0") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_limit>(0));
    } else if (mattype == "hybrid25") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_limit>(0.25));
    } else if (mattype == "hybrid33") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_limit>(1.0 / 3.0));
    } else if (mattype == "hybrid40") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_limit>(0.4));
    } else if (mattype == "hybrid60") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_limit>(0.6));
    } else if (mattype == "hybrid80") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_limit>(0.8));
    } else if (mattype == "hybridlimit0") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_bounded_limit>(0));
    } else if (mattype == "hybridlimit25") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_bounded_limit>(0.25));
    } else if (mattype == "hybridlimit33") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::imbalance_bounded_limit>(1.0 /
                                                                       3.0));
    } else if (mattype == "hybridminstorage") {
        return create_matrix_type<hybrid>(
            std::make_shared<typename hybrid::minimal_storage_limit>());
    } else if (mattype == "sellp") {
        return create_matrix_type<gko::matrix::Sellp<ValueType, itype>>();
    }
    throw std::invalid_argument("Unknown matrix format: " + mattype);
}


/**
 * Returns true if the given format string names one of the AMP formats
 * ("amp": monolithic_classical, "ampib": independent_buckets).
 */
inline bool is_amp_format(const std::string& format)
{
    return format == "amp" || format == "ampib";
}

template <typename ValueType>
inline std::unique_ptr<gko::LinOpFactory> create_amp_matrix_factory(
    std::shared_ptr<const gko::Executor> exec, const std::string& format)
{
    using amp_type = gko::matrix::AMP<ValueType, itype>;
    const auto criterion = (FLAGS_amp_tolerance_type == "normwise")
                               ? amp_type::criterion_type::normwise
                               : amp_type::criterion_type::componentwise;
    const auto strategy = (format == "ampib")
                              ? amp_type::strategy_type::independent_buckets
                              : amp_type::strategy_type::monolithic_classical;
    return amp_type::build()
        .with_criterion(criterion)
        .with_strategy(strategy)
        .with_csr_strategy(
            parse_amp_csr_strategy<ValueType>(FLAGS_amp_csr_strategy))
        .with_tolerance(static_cast<float>(FLAGS_amp_tolerance))
        .with_bin_foldup_nnz_ratio(
            static_cast<float>(FLAGS_amp_bin_foldup_nnz_ratio))
        .with_high_precision_diagonal(FLAGS_amp_high_precision_diagonal)
        .on(exec);
}

template <typename ValueType>
inline std::shared_ptr<gko::LinOp> create_amp_base_matrix(
    std::shared_ptr<const gko::Executor> exec,
    const gko::LinOpFactory* const amp_factory,
    const gko::matrix_data<ValueType, itype>& data)
{
    using csr = gko::matrix::Csr<ValueType, itype>;
    using ell = gko::matrix::Ell<ValueType, itype>;
    using amp_type = gko::matrix::AMP<ValueType, itype>;
    std::shared_ptr<gko::LinOp> base_mat;
    if (FLAGS_amp_base_type == "csr" || FLAGS_amp_base_type == "csrc") {
        auto csr_mat = csr::create(exec);
        csr_mat->read(data);
        base_mat = std::move(csr_mat);
    } else {
        check_ell_admissibility(data);
        auto ell_mat = ell::create(exec);
        ell_mat->read(data);
        base_mat = std::move(ell_mat);
    }
    return base_mat;
}


template <typename ValueType>
std::unique_ptr<gko::LinOp> matrix_factory_generic(
    const std::string& format, std::shared_ptr<const gko::Executor> exec,
    const gko::matrix_data<ValueType, itype>& data)
{
    using hybrid = gko::matrix::Hybrid<ValueType, itype>;
    using csr = gko::matrix::Csr<ValueType, itype>;
    using coo = gko::matrix::Coo<ValueType, itype>;
    using ell = gko::matrix::Ell<ValueType, itype>;
    using ell_mixed =
        gko::matrix::Ell<gko::next_precision_base<ValueType>, itype>;
    using amp_type = gko::matrix::AMP<ValueType, itype>;
    if (is_amp_format(format)) {
        auto factory = create_amp_matrix_factory<ValueType>(exec, format);
        auto base_mat =
            create_amp_base_matrix<ValueType>(exec, factory.get(), data);
        return factory->generate(std::move(base_mat));
    }
    auto mat = get_matrix_factory<ValueType>(format)(exec);
    if (format == "ell" || format == "ell_mixed") {
        check_ell_admissibility(data);
    }
    if (format == "ell_mixed") {
        if constexpr (std::is_same_v<gko::remove_complex<ValueType>, double> ||
                      std::is_same_v<gko::remove_complex<ValueType>, float>) {
            gko::matrix_data<gko::next_precision_base<ValueType>, itype>
                conv_data;
            conv_data.size = data.size;
            conv_data.nonzeros.resize(data.nonzeros.size());
            auto it = conv_data.nonzeros.begin();
            for (auto& el : data.nonzeros) {
                it->row = el.row;
                it->column = el.column;
                it->value = el.value;
                ++it;
            }
            gko::as<gko::ReadableFromMatrixData<
                gko::next_precision_base<ValueType>, itype>>(mat.get())
                ->read(conv_data);
        } else {
            throw std::runtime_error(
                "Ell_mixed is only supported for float and double!");
        }
    } else {
        gko::as<gko::ReadableFromMatrixData<ValueType, itype>>(mat.get())->read(
            data);
    }
    return mat;
}

std::unique_ptr<gko::LinOp> matrix_factory(
    const std::string& format, std::shared_ptr<const gko::Executor> exec,
    const gko::matrix_data<etype, itype>& data)
{
    return matrix_factory_generic<etype>(format, exec, data);
}


/**
 * If the given LinOp is an AMP matrix, writes per-bin nonzero metadata into
 * the provided JSON object under an "amp_bins" key, and the full set of
 * explicitly requested/effective AMP factory parameters under an
 * "amp_config" key.
 *
 * "amp_bins": for AMP[CSR], records total nonzeros per bin; for AMP[ELL],
 * records max nonzeros per row per bin.
 *
 * "amp_config" is read back from the generated matrix's own
 * get_parameters() rather than from the --amp_* flags directly, so that it
 * always reflects exactly what this matrix was built with (this matters in
 * particular for amp_subwarp_size, which is normalized at generation time
 * and so may differ from the value requested on the command line).
 */
template <typename ValueType>
void write_amp_info(const gko::LinOp* mtx, json& format_case)
{
    const auto* amp_mat = dynamic_cast<const amp_type<ValueType>*>(mtx);
    if (!amp_mat) {
        return;
    }
    using csr = gko::matrix::Csr<ValueType, itype>;
    using ell = gko::matrix::Ell<ValueType, itype>;
    const bool is_csr =
        dynamic_cast<const csr*>(amp_mat->get_bin_matrix(0)) != nullptr;
    auto bins_json = json::object();
    for (int k = 0; k < amp_type<ValueType>::num_precisions; ++k) {
        const auto* bin = amp_mat->get_bin_matrix(k);
        if (!bin) {
            continue;
        }
        const auto key = "bin_" + std::to_string(k);
        if (is_csr) {
            const auto* bin_csr = static_cast<const csr*>(bin);
            bins_json[key] = bin_csr->get_num_stored_elements();
        } else {
            const auto* bin_ell = static_cast<const ell*>(bin);
            bins_json[key] = bin_ell->get_num_stored_elements_per_row();
        }
    }
    bins_json["base_type"] = is_csr ? "csr" : "ell";
    bins_json["csr_strategy"] =
        to_string<ValueType>(amp_mat->get_parameters().csr_strategy);
    format_case["amp_bins"] = std::move(bins_json);

    const auto& p = amp_mat->get_parameters();
    format_case["amp_config"] = {
        {"amp_tolerance", p.tolerance},
        {"amp_tolerance_type",
         p.criterion == amp_type<ValueType>::criterion_type::normwise
             ? "normwise"
             : "componentwise"},
        {"amp_base_type", is_csr ? "csr" : "ell"},
        {"amp_strategy",
         p.strategy == amp_type<ValueType>::strategy_type::independent_buckets
             ? "independent_buckets"
             : "monolithic_classical"},
        {"amp_csr_strategy", to_string<ValueType>(p.csr_strategy)},
        {"amp_subwarp_size", p.subwarp_size},
        {"amp_bin_foldup_nnz_ratio", p.bin_foldup_nnz_ratio},
        {"amp_high_precision_diagonal", p.high_precision_diagonal}};
}


}  // namespace formats

#endif  // GKO_BENCHMARK_UTILS_FORMATS_HPP_
