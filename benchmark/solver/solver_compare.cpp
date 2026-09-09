// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// A benchmark driver for comparing two solver configurations ("config_a" and
// "config_b") on the same set of matrices, with most settings shared between
// them ("common") and a few deliberately different. Typical uses: comparing
// two preconditioners (e.g. Jacobi vs. multicolor FwdGaussSeidel) under an
// otherwise identical solver, or comparing two reorderings of the same
// solver+preconditioner to sanity-check a reordering implementation.
//
// Unlike benchmark/solver/solver.cpp, this driver is not built around
// run_test_cases (benchmark/utils/runner.hpp): that framework has no place to
// compare the *outputs* of two operations against each other, since
// Benchmark<State>::postprocess() never sees the per-operation State. Instead
// this is a hand-rolled main() that reuses the same lower-level building
// blocks (generate_solver, precond_factory, reorder, formats::matrix_factory,
// SolverGenerator) that solver_common.hpp assembles into run_test_cases.
//
// Configuration is a single JSON file. Its "common", "config_a" and
// "config_b" objects are keyed by the *same names* as the gflags defined in
// benchmark/solver/solver_common.hpp and the headers it includes (e.g.
// "solvers", "preconditioners", "reorder", "rel_res_goal", "gmres_restart",
// "fgs_sweeps", ...); applying a block means calling
// gflags::SetCommandLineOption for each key, inside a gflags::FlagSaver scope
// so config_a's settings never leak into config_b. This lets every existing
// benchmark helper run completely unmodified. See
// benchmark/solver/config_compare_example.json for a full example.
//
// A handful of flags may only appear in "common" (see common_only_keys
// below): they are latched into function-local statics on first use
// (get_executor, get_engine), or are consumed once before either
// configuration runs (the RHS and initial guess are generated a single time,
// in the original matrix ordering, specifically so that a "sinus" or
// "random" right-hand side is not silently regenerated -- and thus changed
// -- for the second configuration).
//
// Correctness note on reordering: when a configuration reorders the matrix
// with permutation P, it solves (P A P^T) y = P b and this driver maps the
// result back with x = P^T y (permute_mode::inverse_rows) before comparing
// solutions or computing the "authoritative" residual. That residual is
// always computed against the untouched, unreordered, unquantized system
// matrix (A_orig, format "csr") rather than against a config's own
// (possibly permuted and/or reduced-precision, e.g. "amp") operator, so that
// the two configs' residuals -- and the relative solution difference -- stay
// comparable even when the two configs disagree on both reordering and
// format. A secondary "residual_norm_in_format" field reports the residual
// against the config's own operator, which should match residual_norm to
// near machine precision whenever formats agree exactly, and getting the
// un-permute direction wrong would show up as an O(1) relative solution
// difference in a reorder-only comparison.

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/solver/solver_common.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"


namespace {


// Flags that must only be set in "common", never in "config_a"/"config_b":
// setting them per-configuration would either be silently ignored (the first
// two rows are latched into function-local statics on first use) or would
// invalidate the comparison outright (the rest are consumed once, before
// either configuration runs, or control global output/profiling behavior
// this driver does not implement).
const std::set<std::string> common_only_keys{
    "executor",       "device_id",
    "allocator",      "gpu_timer",
    "seed",           "nrhs",
    "rhs_generation", "initial_guess_generation",
    "detailed",       "nested_names",
    "overhead",       "input",
    "input_matrix",   "backup",
    "double_buffer",  "overwrite",
    "profile",        "profiler_hook",
};


// Converts a JSON config value into the string form that
// gflags::SetCommandLineOption expects.
std::string json_to_flag_value(const json& value)
{
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "true" : "false";
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<std::int64_t>());
    }
    if (value.is_number_float()) {
        std::ostringstream os;
        os << std::setprecision(17) << value.get<double>();
        return os.str();
    }
    throw std::runtime_error(
        "configuration values must be strings, numbers or booleans");
}


void apply_flag(const std::string& name, const json& value)
{
    gflags::CommandLineFlagInfo info;
    if (!gflags::GetCommandLineFlagInfo(name.c_str(), &info)) {
        throw std::runtime_error(
            "unknown configuration key '" + name +
            "' (keys must match a benchmark flag name, e.g. \"solvers\", "
            "\"reorder\", \"rel_res_goal\")");
    }
    const auto str = json_to_flag_value(value);
    if (gflags::SetCommandLineOption(name.c_str(), str.c_str()).empty()) {
        throw std::runtime_error("invalid value '" + str + "' for flag --" +
                                 name + " (expected type: " + info.type + ")");
    }
}


void apply_flags(const json& block)
{
    for (auto it = block.begin(); it != block.end(); ++it) {
        apply_flag(it.key(), it.value());
    }
}


void check_no_common_only_keys(const json& block, const std::string& label)
{
    for (auto it = block.begin(); it != block.end(); ++it) {
        if (common_only_keys.count(it.key())) {
            throw std::runtime_error(
                "'" + it.key() +
                "' may only be set in the \"common\" "
                "block, not in \"" +
                label + "\" (it is fixed before either configuration runs)");
        }
    }
}


// Turns one entry of the top-level "matrices" array into the JSON object
// that SolverGenerator::generate_matrix_data expects: a bare filename string
// is sugar for {"filename": <string>}; an object (e.g.
// {"stencil": "7pt", "size": 1000}) is passed through unchanged.
json to_matrix_config(const json& entry)
{
    if (entry.is_string()) {
        return json{{"filename", entry.get<std::string>()}};
    }
    if (entry.is_object()) {
        return entry;
    }
    throw std::runtime_error(
        "each entry of \"matrices\" must be a filename string or an "
        "object, e.g. {\"stencil\": \"7pt\", \"size\": 1000}");
}


std::string basename_only(const std::string& path)
{
    const auto pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}


// Result of running one configuration ("a" or "b") on one matrix.
struct ConfigResult {
    bool ok = false;
    std::string error;
    std::string solver;
    std::string preconditioner;
    std::string format;
    std::string reorder = "none";
    gko::size_type num_colors = 0;
    gko::size_type iterations = 0;
    bool converged = false;
    bool is_direct = false;
    // Residual against the exact, unreordered, unquantized system matrix --
    // the authoritative number, comparable across configs regardless of
    // reorder/format.
    double residual_norm = 0.0;
    // Residual against this config's own (possibly permuted and/or
    // quantized) operator; should match residual_norm closely whenever the
    // format is exact.
    double residual_norm_in_format = 0.0;
    // The absolute residual the stopping criterion was aiming for, i.e.
    // rel_res_goal times ||b|| (or times the initial residual norm under
    // --rel_residual), and whether residual_norm actually reached it.
    double residual_goal = 0.0;
    bool residual_goal_met = false;
    double generate_time = 0.0;
    double apply_time = 0.0;
    unsigned repetitions = 0;
    // Solution, mapped back to the original (unreordered) row ordering.
    std::unique_ptr<vec<etype>> solution;
    // Extra per-config info (e.g. "reordered", "amp_bins") collected along
    // the way.
    json detail = json::object();

    double solve_time() const { return generate_time + apply_time; }

    json to_json() const
    {
        json j;
        j["completed"] = ok;
        if (!ok) {
            j["error"] = error;
            return j;
        }
        j["solver"] = solver;
        j["preconditioner"] = preconditioner;
        j["format"] = format;
        j["reorder"] = reorder;
        j["num_colors"] = num_colors;
        j["iterations"] = iterations;
        j["converged"] = converged;
        j["direct"] = is_direct;
        j["residual_norm"] = residual_norm;
        j["residual_norm_in_format"] = residual_norm_in_format;
        j["residual_goal"] = residual_goal;
        j["residual_goal_met"] = residual_goal_met;
        j["generate_time"] = generate_time;
        j["apply_time"] = apply_time;
        j["solve_time"] = solve_time();
        j["repetitions"] = repetitions;
        for (auto it = detail.begin(); it != detail.end(); ++it) {
            j[it.key()] = it.value();
        }
        return j;
    }
};


// Runs the configuration currently active in the gflags (the caller is
// expected to have applied "common" and one of "config_a"/"config_b" inside
// a gflags::FlagSaver scope) on the given matrix, and returns timing,
// iteration and residual information plus the solution mapped back to the
// original ordering.
ConfigResult run_config(std::shared_ptr<gko::Executor> exec,
                        const gko::matrix_data<etype, itype>& data_orig,
                        const gko::LinOp* A_orig, const vec<etype>* b_orig,
                        const vec<etype>* x0_orig)
{
    ConfigResult r;
    r.solver = FLAGS_solvers;
    r.preconditioner = FLAGS_preconditioners;
    r.format = FLAGS_formats;
    r.reorder = FLAGS_reorder;
    if (split(FLAGS_solvers, ',').size() != 1) {
        throw std::runtime_error(
            "a configuration must select exactly one solver");
    }
    if (split(FLAGS_preconditioners, ',').size() != 1) {
        throw std::runtime_error(
            "a configuration must select exactly one preconditioner");
    }
    if (split(FLAGS_formats, ',').size() != 1) {
        throw std::runtime_error(
            "a configuration must select exactly one format");
    }

    // Reorder a private copy of the matrix data, if requested. reorder()
    // (benchmark/utils/general_matrix.hpp) permutes symmetrically
    // (A' = P A P^T) and writes test_case["reordered"] into r.detail.
    gko::matrix_data<etype, itype> data_local;
    const gko::matrix_data<etype, itype>* data_ptr = &data_orig;
    ReorderResult<itype> ro;
    if (FLAGS_reorder != "none") {
        data_local = data_orig;
        ro = reorder(data_local, r.detail);
        data_ptr = &data_local;
    }
    r.num_colors = ro.color_ptrs.empty() ? 0 : ro.color_ptrs.size() - 1;

    auto A =
        gko::share(formats::matrix_factory(FLAGS_formats, exec, *data_ptr));
    if (FLAGS_formats == "amp") {
        formats::write_amp_bin_info(A.get(), r.detail);
    }

    auto b = gko::clone(b_orig);
    auto x0 = gko::clone(x0_orig);
    if (ro.permutation) {
        b = b->permute(ro.permutation, gko::matrix::permute_mode::rows);
        x0 = x0->permute(ro.permutation, gko::matrix::permute_mode::rows);
    }

    const PrecondArgs prec_args{exec, ro.color_ptrs};

    IterationControl ic{get_timer(exec, FLAGS_gpu_timer)};
    auto generate_timer = get_timer(exec, FLAGS_gpu_timer);
    auto apply_timer = ic.get_timer();

    // Warmup: fixed number of iterations (FLAGS_warmup_max_iters),
    // independent of --repetitions, exactly as in
    // SolverBenchmark::run (benchmark/solver/solver_common.hpp).
    for (auto _ : ic.warmup_run()) {
        auto x_clone = gko::clone(x0);
        auto precond = precond_factory.at(FLAGS_preconditioners)(prec_args);
        auto warmup_solver = generate_solver(exec, give(precond), FLAGS_solvers,
                                             FLAGS_warmup_max_iters)
                                 ->generate(A);
        warmup_solver->apply(b, x_clone);
        exec->synchronize();
    }

    // Timed run, mirroring SolverBenchmark::run
    // (benchmark/solver/solver_common.hpp:591-636), including the
    // --benchmark_from_scratch distinction between reusing one generated
    // solver and regenerating it (cold) every repetition.
    auto conv_logger = gko::share(gko::log::Convergence<etype>::create());
    std::shared_ptr<gko::LinOp> solver;
    auto x = gko::clone(x0);
    if (!FLAGS_benchmark_from_scratch) {
        auto precond = precond_factory.at(FLAGS_preconditioners)(prec_args);
        solver = gko::share(
            generate_solver(exec, give(precond), FLAGS_solvers, FLAGS_max_iters)
                ->generate(A));
        solver->apply(b, x);
    }
    for (auto status : ic.run(false)) {
        x = gko::clone(x0);
        {
            exec->synchronize();
            generate_timer->tic();
            auto precond = precond_factory.at(FLAGS_preconditioners)(prec_args);
            auto generated_solver =
                gko::share(generate_solver(exec, give(precond), FLAGS_solvers,
                                           FLAGS_max_iters)
                               ->generate(A));
            generate_timer->toc();
            if (FLAGS_benchmark_from_scratch) {
                solver = generated_solver;
            }
        }
        exec->synchronize();
        if (ic.get_num_repetitions() == 0) {
            solver->add_logger(conv_logger);
        }
        apply_timer->tic();
        solver->apply(b, x);
        apply_timer->toc();
        if (ic.get_num_repetitions() == 0) {
            solver->remove_logger(conv_logger);
        }
    }

    r.iterations = conv_logger->get_num_iterations();
    r.converged = conv_logger->has_converged();
    // A solver with no stopping criteria (the *_trs/*_direct solvers) never
    // triggers a criterion check, so no iterations are ever logged -- the
    // same "0 iterations means direct" heuristic SolverBenchmark::run uses.
    r.is_direct = (r.iterations == 0);
    r.residual_norm_in_format =
        compute_residual_norm(A.get(), b.get(), x.get());

    if (ro.permutation) {
        r.solution =
            x->permute(ro.permutation, gko::matrix::permute_mode::inverse_rows);
    } else {
        r.solution = std::move(x);
    }
    r.residual_norm = compute_residual_norm(A_orig, b_orig, r.solution.get());

    // The stopping criterion built by create_criterion() tests the solver's
    // *recurrent* residual, which for GMRES can keep descending long after the
    // true residual has stagnated. A solver can therefore report convergence
    // while the residual recomputed above is still orders of magnitude above
    // the requested goal -- which is easy to mistake for a bug in whatever the
    // two configurations differ by. Record the goal the criterion was actually
    // aiming for so that case can be flagged. Both candidate baselines are
    // permutation-invariant, so evaluating them on the original system matches
    // what the (possibly permuted) criterion saw.
    const auto baseline_norm =
        FLAGS_rel_residual ? compute_residual_norm(A_orig, b_orig, x0_orig)
                           : compute_norm2(b_orig);
    r.residual_goal = FLAGS_rel_res_goal * baseline_norm;
    r.residual_goal_met = r.residual_norm <= r.residual_goal;

    r.generate_time = generate_timer->compute_time(FLAGS_timer_method);
    r.apply_time = apply_timer->compute_time(FLAGS_timer_method);
    r.repetitions = apply_timer->get_num_repetitions();
    r.ok = true;
    return r;
}


std::string fmt_num(double v)
{
    std::ostringstream os;
    os << std::scientific << std::setprecision(2) << v;
    return os.str();
}


std::string fmt_iters(const json& c)
{
    if (!c.value("completed", false)) {
        return "---";
    }
    if (c.value("direct", false)) {
        return "D";
    }
    std::string s = std::to_string(c.value("iterations", gko::int64{0}));
    if (!c.value("converged", false)) {
        s += "*";
    }
    return s;
}


// Formats a config's true residual, appending "!" when the solver reported
// convergence even though this recomputed residual never reached the goal the
// stopping criterion was asking for -- see the comment in run_config().
std::string fmt_residual(const json& c)
{
    if (!c.value("completed", false)) {
        return "---";
    }
    auto s = fmt_num(c.value("residual_norm", 0.0));
    if (c.value("converged", false) && !c.value("residual_goal_met", true)) {
        s += "!";
    }
    return s;
}


void print_header(const std::string& label_a, const json& cfg_a,
                  const std::string& label_b, const json& cfg_b,
                  std::shared_ptr<const gko::Executor> exec)
{
    std::cout << "solver_compare -- " << exec->get_description() << "\n"
              << "  A (" << label_a << "): " << cfg_a.dump() << "\n"
              << "  B (" << label_b << "): " << cfg_b.dump() << "\n\n";
}


constexpr int kMatrixWidth = 24;
constexpr int kNumWidth = 10;
constexpr int kValWidth = 11;


void print_table_header()
{
    std::cout << std::left << std::setw(kMatrixWidth) << "matrix" << std::right
              << std::setw(kNumWidth) << "rows" << std::setw(kNumWidth) << "nnz"
              << std::setw(kValWidth) << "||b||" << std::setw(kValWidth)
              << "resA" << std::setw(kValWidth) << "resB" << std::setw(6)
              << "itA" << std::setw(6) << "itB" << std::setw(kValWidth)
              << "timeA(s)" << std::setw(kValWidth) << "timeB(s)"
              << std::setw(kValWidth) << "rel.diff"
              << "\n";
    std::cout << std::string(kMatrixWidth + 2 * kNumWidth + 6 * kValWidth, '-')
              << "\n";
}


void print_table_row(const json& row)
{
    std::cout << std::left << std::setw(kMatrixWidth)
              << basename_only(row.value("matrix", std::string{"?"}));
    if (row.contains("error")) {
        std::cout << "FAILED: " << row["error"].get<std::string>() << "\n";
        return;
    }
    const auto& a = row["a"];
    const auto& b = row["b"];
    const bool a_ok = a.value("completed", false);
    const bool b_ok = b.value("completed", false);
    std::cout << std::right << std::setw(kNumWidth) << row.value("rows", 0)
              << std::setw(kNumWidth) << row.value("nnz", 0)
              << std::setw(kValWidth) << fmt_num(row.value("rhs_norm", 0.0))
              << std::setw(kValWidth) << fmt_residual(a) << std::setw(kValWidth)
              << fmt_residual(b) << std::setw(6) << fmt_iters(a) << std::setw(6)
              << fmt_iters(b) << std::setw(kValWidth)
              << (a_ok ? fmt_num(a.value("solve_time", 0.0))
                       : std::string{"---"})
              << std::setw(kValWidth)
              << (b_ok ? fmt_num(b.value("solve_time", 0.0))
                       : std::string{"---"})
              << std::setw(kValWidth)
              << (row.contains("rel_solution_diff") &&
                          !row["rel_solution_diff"].is_null()
                      ? fmt_num(row["rel_solution_diff"].get<double>())
                      : std::string{"---"})
              << "\n";
    if (!a_ok) {
        std::cout << "    [A] " << a.value("error", std::string{}) << "\n";
    }
    if (!b_ok) {
        std::cout << "    [B] " << b.value("error", std::string{}) << "\n";
    }
}


}  // namespace


int main(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>\n\n"
                  << "Compares two solver configurations (\"config_a\" vs "
                     "\"config_b\") on a common list of matrices. See\n"
                  << "benchmark/solver/config_compare_example.json for the "
                     "expected format.\n";
        return 1;
    }

    json cfg;
    {
        std::ifstream in(argv[1]);
        if (!in) {
            std::cerr << "Could not open config file '" << argv[1] << "'\n";
            return 1;
        }
        in >> cfg;
    }
    if (!cfg.contains("matrices") || !cfg["matrices"].is_array()) {
        std::cerr << "config must contain a \"matrices\" array\n";
        return 1;
    }

    const auto common = cfg.value("common", json::object());
    const auto cfg_a = cfg.value("config_a", json::object());
    const auto cfg_b = cfg.value("config_b", json::object());
    const auto label_a = cfg.value("label_a", std::string{"config_a"});
    const auto label_b = cfg.value("label_b", std::string{"config_b"});
    const auto output_path =
        cfg.value("output_file", std::string{"solver_compare_results.json"});

    try {
        check_no_common_only_keys(cfg_a, "config_a");
        check_no_common_only_keys(cfg_b, "config_b");
    } catch (const std::exception& e) {
        std::cerr << "Error in configuration: " << e.what() << std::endl;
        return 1;
    }
    try {
        apply_flags(common);
    } catch (const std::exception& e) {
        std::cerr << "Error applying \"common\": " << e.what() << std::endl;
        return 1;
    }
    if (FLAGS_nrhs != 1) {
        std::cerr << "solver_compare requires nrhs == 1\n";
        return 1;
    }

    auto exec = get_executor(FLAGS_gpu_timer);
    std::cerr << gko::version_info::get() << "\nRunning on "
              << exec->get_description() << std::endl;

    print_header(label_a, cfg_a, label_b, cfg_b, exec);
    print_table_header();

    SolverGenerator gen;
    json out;
    out["label_a"] = label_a;
    out["label_b"] = label_b;
    out["common"] = common;
    out["config_a"] = cfg_a;
    out["config_b"] = cfg_b;
    out["results"] = json::array();

    for (const auto& entry : cfg["matrices"]) {
        json row;
        try {
            auto mcase = to_matrix_config(entry);
            if (!SolverGenerator::validate_config(mcase)) {
                throw std::runtime_error("invalid matrix entry");
            }
            row["matrix"] = SolverGenerator::describe_config(mcase);

            auto [data, size] = SolverGenerator::generate_matrix_data(mcase);
            if (data.size[0] == 0 || data.size[0] != data.size[1]) {
                throw std::runtime_error(
                    "solver_compare requires a nonempty, square matrix");
            }
            row["rows"] = data.size[0];
            row["nnz"] = data.nonzeros.size();

            // Built once, in the original ordering, under "common" flags
            // only -- crucially *before* either configuration reorders
            // anything, so both configs solve for the exact same right-hand
            // side and initial guess.
            auto A_orig =
                gko::share(formats::matrix_factory("csr", exec, data));
            auto b_orig = gen.generate_rhs(exec, A_orig.get(), mcase);
            auto x0_orig =
                gen.generate_initial_guess(exec, A_orig.get(), b_orig.get());
            row["rhs_norm"] = compute_norm2(b_orig.get());

            ConfigResult ra;
            try {
                gflags::FlagSaver saver;
                apply_flags(cfg_a);
                ra = run_config(exec, data, A_orig.get(), b_orig.get(),
                                x0_orig.get());
            } catch (const std::exception& e) {
                ra.ok = false;
                ra.error = e.what();
            }
            ConfigResult rb;
            try {
                gflags::FlagSaver saver;
                apply_flags(cfg_b);
                rb = run_config(exec, data, A_orig.get(), b_orig.get(),
                                x0_orig.get());
            } catch (const std::exception& e) {
                rb.ok = false;
                rb.error = e.what();
            }
            row["a"] = ra.to_json();
            row["b"] = rb.to_json();

            if (ra.ok && rb.ok) {
                const auto ref_norm = compute_norm2(ra.solution.get());
                // compute_max_relative_norm2 is destructive in its first
                // argument, hence the clone.
                auto diff = gko::clone(rb.solution);
                if (ref_norm > 0) {
                    row["rel_solution_diff"] = compute_max_relative_norm2(
                        diff.get(), ra.solution.get());
                } else {
                    auto neg_one = gko::initialize<vec<etype>>({-1.0}, exec);
                    diff->add_scaled(neg_one, ra.solution);
                    row["rel_solution_diff"] = nullptr;
                    row["abs_solution_diff"] = compute_norm2(diff.get());
                }
            }
        } catch (const std::exception& e) {
            row["error"] = e.what();
        }

        print_table_row(row);
        out["results"].push_back(row);

        // Write after every matrix, so a crash or interrupt still leaves the
        // completed results on disk.
        std::ofstream of(output_path);
        of << std::setw(2) << out << std::endl;
    }

    std::cout << std::string(kMatrixWidth + 2 * kNumWidth + 6 * kValWidth, '-')
              << "\n"
              << "* = solver stopped without reporting convergence, "
                 "D = direct solver (0 iterations)\n"
              << "! = reported converged, but this recomputed residual is "
                 "still above rel_res_goal;\n"
              << "    the stopping criterion tests the solver's recurrent "
                 "residual, not this one, so\n"
              << "    a goal below the attainable floor is reached only on "
                 "paper\n";
    std::cerr << "\nResults written to " << output_path << std::endl;
    return 0;
}
