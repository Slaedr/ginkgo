#!/usr/bin/env bash
################################################################################
# Environment variable detection
#

print_default() {
    local var=$1
    echo "$var  environment variable not set - assuming \"${!var}\"" 1>&2
}

if [ ! "${BENCHMARK}" ]; then
    BENCHMARK="spmv"
    print_default BENCHMARK
fi

if [ ! "${DRY_RUN}" ]; then
    DRY_RUN="false"
    print_default DRY_RUN
fi

if [ ! "${EXECUTOR}" ]; then
    EXECUTOR="cuda"
    print_default EXECUTOR
fi

if [ ! "${REPETITIONS}" ]; then
    REPETITIONS=10
    print_default REPETITIONS
fi

if [ ! "${SOLVER_REPETITIONS}" ]; then
    SOLVER_REPETITIONS=1
    print_default SOLVER_REPETITIONS
fi

if [ ! "${SEGMENTS}" ]; then
    echo "SEGMENTS  environment variable not set - running entire suite" 1>&2
    SEGMENTS=1
    SEGMENT_ID=1
elif [ ! "${SEGMENT_ID}" ]; then
    echo "SEGMENT_ID  environment variable not set - exiting" 1>&2
    exit 1
fi

if [ ! "${PRECONDS}" ]; then
    PRECONDS="none"
    print_default PRECONDS
fi

if [ ! "${FORMATS}" ]; then
    FORMATS="csr,coo,ell,hybrid,sellp"
    print_default FORMATS
fi

if [ ! "${ELL_IMBALANCE_LIMIT}" ]; then
    ELL_IMBALANCE_LIMIT=100
    print_default ELL_IMBALANCE_LIMIT
fi

if [ ! "${AMP_BASE_TYPE}" ]; then
    AMP_BASE_TYPE="ell"
    print_default AMP_BASE_TYPE
fi

if [ "${AMP_BASE_TYPE}" != "ell" ] && [ "${AMP_BASE_TYPE}" != "csrc" ]; then
    echo "AMP_BASE_TYPE is set to the unsupported \"${AMP_BASE_TYPE}\"." 1>&2
    echo "Currently supported values: \"ell\" and \"csrc\"" 1>&2
    exit 1
fi

if [ ! "${AMP_TOLERANCE_TYPE}" ]; then
    AMP_TOLERANCE_TYPE="componentwise"
    print_default AMP_TOLERANCE_TYPE
fi

if [ "${AMP_TOLERANCE_TYPE}" != "componentwise" ] && [ "${AMP_TOLERANCE_TYPE}" != "normwise" ]; then
    echo "AMP_TOLERANCE_TYPE is set to the unsupported \"${AMP_TOLERANCE_TYPE}\"." 1>&2
    echo "Currently supported values: \"componentwise\" and \"normwise\"" 1>&2
    exit 1
fi

if [ ! "${AMP_CSR_STRATEGY}" ]; then
    AMP_CSR_STRATEGY="automatical"
    print_default AMP_CSR_STRATEGY
fi

case "${AMP_CSR_STRATEGY}" in
    automatical|classical|load_balance|merge_path) ;;
    *)
        echo "AMP_CSR_STRATEGY is set to the unsupported \"${AMP_CSR_STRATEGY}\"." 1>&2
        echo "Currently supported values: \"automatical\", \"classical\", \"load_balance\" and \"merge_path\"" 1>&2
        exit 1
        ;;
esac

if [ ! "${AMP_TOLERANCE}" ]; then
    AMP_TOLERANCE="1e-14"
    print_default AMP_TOLERANCE
fi

# Unlike the other AMP_* variables above, this one has no default here: if
# unset, the flag is simply omitted, and the benchmark falls back to the
# default declared on the AMP class itself.
if [ "${AMP_BIN_FOLDUP_NNZ_RATIO}" ]; then
    AMP_BIN_FOLDUP_NNZ_RATIO_FLAG="--amp_bin_foldup_nnz_ratio=${AMP_BIN_FOLDUP_NNZ_RATIO}"
else
    echo "AMP_BIN_FOLDUP_NNZ_RATIO environment variable not set - using the AMP class's own default" 1>&2
    AMP_BIN_FOLDUP_NNZ_RATIO_FLAG=""
fi

# Same convention as AMP_BIN_FOLDUP_NNZ_RATIO above: no default here, so an
# unset variable falls back to the AMP class's own default.
if [ "${AMP_HIGH_PRECISION_DIAGONAL}" ]; then
    AMP_HIGH_PRECISION_DIAGONAL_FLAG="--amp_high_precision_diagonal=${AMP_HIGH_PRECISION_DIAGONAL}"
else
    echo "AMP_HIGH_PRECISION_DIAGONAL environment variable not set - using the AMP class's own default" 1>&2
    AMP_HIGH_PRECISION_DIAGONAL_FLAG=""
fi

if [ ! "${SOLVERS}" ]; then
    SOLVERS="bicgstab,cg,cgs,fcg,gmres,cb_gmres_reduce1,idr"
    print_default SOLVERS
fi

if [ ! "${SOLVERS_PRECISION}" ]; then
    SOLVERS_PRECISION=1e-6
    print_default SOLVERS_PRECISION
fi

if [ ! "${SOLVERS_MAX_ITERATIONS}" ]; then
    SOLVERS_MAX_ITERATIONS=10000
    print_default SOLVERS_MAX_ITERATIONS
fi

if [ ! "${SOLVERS_WARMUP_MAX_ITERATIONS}" ]; then
    SOLVERS_WARMUP_MAX_ITERATIONS=100
    print_default SOLVERS_WARMUP_MAX_ITERATIONS
fi

if [ ! "${SOLVERS_GMRES_RESTART}" ]; then
    SOLVERS_GMRES_RESTART=100
    print_default SOLVERS_GMRES_RESTART
fi

if [ ! "${SOLVERS_REORDER}" ]; then
    SOLVERS_REORDER="none"
    print_default SOLVERS_REORDER
fi

# "fgs" in SOLVERS runs FwdGaussSeidel as a standalone iterative solver
# (as opposed to the "fgs" *preconditioner* in PRECONDS, which applies a
# fixed, small number of sweeps per apply). As a standalone solver it
# requires a multicolor-ordered matrix; the benchmark itself exits with an
# explanatory error if this is missing, but warn here too, since that error
# would otherwise only surface partway through a potentially long sweep over
# many matrices.
case ",${SOLVERS}," in
    *,fgs,*)
        if [ "${SOLVERS_REORDER}" != "multicolor" ]; then
            echo "WARNING: SOLVERS includes \"fgs\" but SOLVERS_REORDER=\"${SOLVERS_REORDER}\" - the fgs solver requires SOLVERS_REORDER=multicolor and will fail otherwise." 1>&2
        fi
        ;;
esac

if [ ! "${SOLVERS_FGS_SWEEPS}" ]; then
    SOLVERS_FGS_SWEEPS=1
    print_default SOLVERS_FGS_SWEEPS
fi

if [ ! "${SYSTEM_NAME}" ]; then
    SYSTEM_NAME="unknown"
    print_default SYSTEM_NAME
fi

if [ ! "${DEVICE_ID}" ]; then
    DEVICE_ID="0"
    print_default DEVICE_ID
fi

if [ ! "${SOLVERS_JACOBI_MAX_BS}" ]; then
    SOLVERS_JACOBI_MAX_BS="32"
    print_default SOLVERS_JACOBI_MAX_BS
fi

if [ ! "${BENCHMARK_PRECISION}" ]; then
    BENCHMARK_PRECISION="double"
    print_default BENCHMARK_PRECISION
fi

if [ "${BENCHMARK_PRECISION}" == "double" ]; then
    BENCH_SUFFIX=""
elif [ "${BENCHMARK_PRECISION}" == "single" ]; then
    BENCH_SUFFIX="_single"
elif [ "${BENCHMARK_PRECISION}" == "dcomplex" ]; then
    BENCH_SUFFIX="_dcomplex"
elif [ "${BENCHMARK_PRECISION}" == "scomplex" ]; then
    BENCH_SUFFIX="_scomplex"
else
    echo "BENCHMARK_PRECISION is set to the not supported \"${BENCHMARK_PRECISION}\"." 1>&2
    echo "Currently supported values: \"double\", \"single\", \"dcomplex\" and \"scomplex\"" 1>&2
    exit 1
fi

if [ ! "${SOLVERS_RHS}" ]; then
    SOLVERS_RHS="1"
    echo "SOLVERS_RHS environment variable not set - assuming \"${SOLVERS_RHS}\"" 1>&2
fi

if [ "${SOLVERS_RHS}" == "random" ]; then
    SOLVERS_RHS_FLAG="--rhs_generation=random"
elif [ "${SOLVERS_RHS}" == "1" ]; then
    SOLVERS_RHS_FLAG="--rhs_generation=1"
elif [ "${SOLVERS_RHS}" == "sinus" ]; then
    SOLVERS_RHS_FLAG="--rhs_generation=sinus"
else
    echo "SOLVERS_RHS does not support the value \"${SOLVERS_RHS}\"." 1>&2
    echo "The following values are supported: \"1\", \"random\" and \"sinus\"" 1>&2
    exit 1
fi

if [ ! "${SOLVERS_INITIAL_GUESS}" ]; then
    SOLVERS_INITIAL_GUESS="rhs"
    print_default SOLVERS_INITIAL_GUESS
fi

if [ "${SOLVERS_INITIAL_GUESS}" == "random" ]; then
    SOLVERS_INITIAL_GUESS_FLAG="--initial_guess_generation=random"
elif [ "${SOLVERS_INITIAL_GUESS}" == "0" ]; then
    SOLVERS_INITIAL_GUESS_FLAG="--initial_guess_generation=0"
elif [ "${SOLVERS_INITIAL_GUESS}" == "rhs" ]; then
    SOLVERS_INITIAL_GUESS_FLAG="--initial_guess_generation=rhs"
else
    echo "SOLVERS_RHS does not support the value \"${SOLVERS_RHS}\"." 1>&2
    echo "The following values are supported: \"0\", \"random\" and \"rhs\"" 1>&2
    exit 1
fi

if [ ! "${GPU_TIMER}" ]; then
    GPU_TIMER="false"
    print_default GPU_TIMER
fi

if [ ! "${TIMER_OUTPUT}" ]; then
    TIMER_OUTPUT="average"
    print_default TIMER_OUTPUT
fi

# Control whether to run detailed benchmarks or not.
# Default setting is detailed=false. To activate, set DETAILED=1.
if  [ ! "${DETAILED}" ] || [ "${DETAILED}" -eq 0 ]; then
    DETAILED_STR="--detailed=false"
else
    DETAILED_STR="--detailed=true"
fi

# This allows using a matrix list file for benchmarking.
# The file should contains a suitesparse matrix on each line.
# The allowed formats to target suitesparse matrix is:
#   id or group/name or name.
# Example:
# 1903
# Freescale/circuit5M
# thermal2
if [ ! "${MATRIX_LIST_FILE}" ]; then
    use_matrix_list_file=0
elif [ -f "${MATRIX_LIST_FILE}" ]; then
    use_matrix_list_file=1
else
    echo -e "A matrix list file was set to ${MATRIX_LIST_FILE} but it cannot be found."
    exit 1
fi

# Location handling. This makes the script usable from any working directory.
#
#   GINKGO_BUILD_DIR  Root of the Ginkgo build tree (the directory that contains
#                     "benchmark/"). If unset, executables are looked up relative
#                     to the current directory (the original behavior: run the
#                     script from <build>/benchmark).
#   RESULTS_DIR       If set, results are written directly to
#                     $RESULTS_DIR/<group>/<name>.json (SYSTEM_NAME, EXECUTOR
#                     and the collection name are not part of the path). If
#                     unset, the original tree
#                     results/<SYSTEM_NAME>/<EXECUTOR>/<collection>/<group>/<name>.json
#                     is created in the current directory.
#                     When RESULTS_DIR is set, existing files are never
#                     overwritten or deleted: a problem whose result file (or
#                     leftover .imd/.bkp/.bkp2/.tmp file) already exists is
#                     skipped.
if [ "${GINKGO_BUILD_DIR}" ]; then
    BENCH_BIN_DIR="${GINKGO_BUILD_DIR}/benchmark"
    if [ ! -d "${BENCH_BIN_DIR}" ]; then
        echo "GINKGO_BUILD_DIR is set to \"${GINKGO_BUILD_DIR}\", but \"${BENCH_BIN_DIR}\" is not a directory." 1>&2
        exit 1
    fi
else
    BENCH_BIN_DIR="."
    echo "GINKGO_BUILD_DIR environment variable not set - using executables relative to the current directory" 1>&2
fi

if [ "${RESULTS_DIR}" ]; then
    RESULTS_BASE="${RESULTS_DIR%/}"
    SUITESPARSE_RESULT_DIR="${RESULTS_BASE}"
    GENERATED_RESULT_DIR="${RESULTS_BASE}"
    NO_CLOBBER=1
    if [ "${DRY_RUN}" != "true" ]; then
        if ! mkdir -p "${RESULTS_BASE}" || [ ! -w "${RESULTS_BASE}" ]; then
            echo "RESULTS_DIR \"${RESULTS_DIR}\" cannot be created or is not writable." 1>&2
            exit 1
        fi
    fi
else
    RESULTS_BASE="results"
    SUITESPARSE_RESULT_DIR="${RESULTS_BASE}/${SYSTEM_NAME}/${EXECUTOR}/SuiteSparse"
    GENERATED_RESULT_DIR="${RESULTS_BASE}/${SYSTEM_NAME}/${EXECUTOR}/Generated"
    NO_CLOBBER=0
    echo "RESULTS_DIR environment variable not set - assuming \"${RESULTS_BASE}/${SYSTEM_NAME}/${EXECUTOR}/<collection>\" in the current directory" 1>&2
fi

# Aborts early if an executable required by the selected benchmark is missing,
# instead of failing partway through a long sweep.
require_executables() {
    [ "${DRY_RUN}" == "true" ] && return
    local exe missing=0
    for exe in "$@"; do
        if [ ! -x "${BENCH_BIN_DIR}/${exe}${BENCH_SUFFIX}" ]; then
            echo "Missing executable: ${BENCH_BIN_DIR}/${exe}${BENCH_SUFFIX}" 1>&2
            missing=1
        fi
    done
    [ "${missing}" -eq 0 ] || exit 1
}

case "${BENCHMARK}" in
    preconditioner)
        require_executables matrix_generator/matrix_generator \
            matrix_statistics/matrix_statistics preconditioner/preconditioner ;;
    conversions)
        require_executables matrix_statistics/matrix_statistics spmv/spmv \
            conversions/conversions ;;
    solver)
        require_executables matrix_statistics/matrix_statistics spmv/spmv \
            solver/solver ;;
    *)
        require_executables matrix_statistics/matrix_statistics spmv/spmv ;;
esac

# By default, each SuiteSparse matrix is removed from ssget's cache once it has
# been benchmarked ("ssget -c"). Set KEEP_MATRICES=true to keep it instead. This
# is required when running several jobs at the same time (e.g. different AMP
# tolerances) or when using a pre-populated, shared collection: otherwise one job
# deletes a matrix that another job is still reading, or that others still need.
if [ ! "${KEEP_MATRICES}" ]; then
    KEEP_MATRICES="false"
    print_default KEEP_MATRICES
fi

# Removes the downloaded copy of SuiteSparse problem $1 unless KEEP_MATRICES=true.
clean_suite_sparse_problem() {
    [ "${DRY_RUN}" == "true" ] && return
    [ "${KEEP_MATRICES}" == "true" ] && return
    ${SSGET} -i "$1" -c >/dev/null
}

# Returns success if $1 or any of its intermediate files already exists. Only
# used when RESULTS_DIR is set, to avoid clobbering earlier results.
result_exists() {
    [ "${NO_CLOBBER}" -eq 1 ] || return 1
    local suffix
    for suffix in "" .imd .bkp .bkp2 .tmp; do
        [ -e "$1${suffix}" ] && return 0
    done
    return 1
}


################################################################################
# Utilities

# Checks the creation dates of a set of files $@, replaces file $1
# with the newest file from the set, and deletes the other
# files.
keep_latest() {
    RESULT="$1"
    for file in $@; do
        if [ "${file}" -nt "${RESULT}" ]; then
            RESULT="${file}"
        fi
    done
    if [ "${RESULT}" != "$1" ]; then
        cp "${RESULT}" "$1"
    fi
    for file in ${@:2}; do
        rm -f "${file}"
    done
}


# Computes matrix statistics of the problems described in file $1, and updates
# the file with results. Backups are created after each processed problem to
# prevent data loss in case of a crash. Once the extraction is completed, the
# backups and the results are combined, and the newest file is taken as the
# final result.
compute_matrix_statistics() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.imd" # make sure we're not losing the original input
    "${BENCH_BIN_DIR}"/matrix_statistics/matrix_statistics${BENCH_SUFFIX} \
        --backup="$1.bkp" --double_buffer="$1.bkp2" \
        <"$1.imd" 2>&1 >"$1"
    keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the conversion benchmarks for all matrix formats by using file $1 as the
# input, and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_conversion_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.imd" # make sure we're not losing the original input
    "${BENCH_BIN_DIR}"/conversions/conversions${BENCH_SUFFIX} --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --formats="${FORMATS}" \
                --device_id="${DEVICE_ID}" --gpu_timer=${GPU_TIMER} --timer_method=${TIMER_OUTPUT} \
                --repetitions="${REPETITIONS}" \
                --ell_imbalance_limit="${ELL_IMBALANCE_LIMIT}" \
                --amp_base_type="${AMP_BASE_TYPE}" \
                --amp_tolerance_type="${AMP_TOLERANCE_TYPE}" \
                --amp_tolerance="${AMP_TOLERANCE}" \
                --amp_csr_strategy="${AMP_CSR_STRATEGY}" \
                ${AMP_BIN_FOLDUP_NNZ_RATIO_FLAG} \
                ${AMP_HIGH_PRECISION_DIAGONAL_FLAG} \
                <"$1.imd" 2>&1 >"$1"
    keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the SpMV benchmarks for all SpMV formats by using file $1 as the input,
# and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_spmv_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.imd" # make sure we're not losing the original input
    "${BENCH_BIN_DIR}"/spmv/spmv${BENCH_SUFFIX} --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --formats="${FORMATS}" \
                --device_id="${DEVICE_ID}" --gpu_timer=${GPU_TIMER} --timer_method=${TIMER_OUTPUT} \
                --repetitions="${REPETITIONS}" \
                --ell_imbalance_limit="${ELL_IMBALANCE_LIMIT}" \
                --amp_base_type="${AMP_BASE_TYPE}" \
                --amp_tolerance_type="${AMP_TOLERANCE_TYPE}" \
                --amp_tolerance="${AMP_TOLERANCE}" \
                --amp_csr_strategy="${AMP_CSR_STRATEGY}" \
                ${AMP_BIN_FOLDUP_NNZ_RATIO_FLAG} \
                ${AMP_HIGH_PRECISION_DIAGONAL_FLAG} \
                <"$1.imd" 2>&1 >"$1"
    keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the solver benchmarks for all supported solvers by using file $1 as the
# input, and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_solver_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.imd" # make sure we're not losing the original input
    "${BENCH_BIN_DIR}"/solver/solver${BENCH_SUFFIX} --backup="$1.bkp" --double_buffer="$1.bkp2" \
                    --executor="${EXECUTOR}" --solvers="${SOLVERS}" \
                    --preconditioners="${PRECONDS}" \
                    --max_iters=${SOLVERS_MAX_ITERATIONS} --rel_res_goal=${SOLVERS_PRECISION} \
                    ${SOLVERS_RHS_FLAG} ${DETAILED_STR} ${SOLVERS_INITIAL_GUESS_FLAG} \
                    --gpu_timer=${GPU_TIMER} --timer_method=${TIMER_OUTPUT} \
                    --jacobi_max_block_size=${SOLVERS_JACOBI_MAX_BS} --device_id="${DEVICE_ID}" \
                    --gmres_restart="${SOLVERS_GMRES_RESTART}" \
                    --repetitions="${SOLVER_REPETITIONS}" \
                    --amp_base_type="${AMP_BASE_TYPE}" \
                    --amp_tolerance_type="${AMP_TOLERANCE_TYPE}" \
                    --amp_tolerance="${AMP_TOLERANCE}" \
                    --amp_csr_strategy="${AMP_CSR_STRATEGY}" \
                    ${AMP_BIN_FOLDUP_NNZ_RATIO_FLAG} \
                    ${AMP_HIGH_PRECISION_DIAGONAL_FLAG} \
                    --reorder="${SOLVERS_REORDER}" \
                    --fgs_sweeps="${SOLVERS_FGS_SWEEPS}" \
                    <"$1.imd" 2>&1 >"$1"
    keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
}

# A list of block sizes that should be run for the block-Jacobi preconditioner
BLOCK_SIZES="$(seq 1 32)"
# A lis of precision reductions to run the block-Jacobi preconditioner for
PRECISIONS="0,0 0,1 0,2 1,0 1,1 2,0 autodetect"
# Runs the preconditioner benchmarks for all supported preconditioners by using
# file $1 as the input, and updating it with the results. Backups are created
# after each benchmark run, to prevent data loss in case of a crash. Once the
# benchmarking is completed, the backups and the results are combined, and the
# newest file is taken as the final result.
run_preconditioner_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    local bsize
    for bsize in ${BLOCK_SIZES}; do
        for prec in ${PRECISIONS}; do
            echo -e "\t\t running jacobi ($prec) for block size ${bsize}" 1>&2
            cp "$1" "$1.imd" # make sure we're not losing the original input
            "${BENCH_BIN_DIR}"/preconditioner/preconditioner${BENCH_SUFFIX} \
                --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --preconditioners="jacobi" \
                --jacobi_max_block_size="${bsize}" \
                --jacobi_storage="${prec}" \
                --device_id="${DEVICE_ID}" --gpu_timer=${GPU_TIMER} --timer_method=${TIMER_OUTPUT} \
                --repetitions="${REPETITIONS}" \
                <"$1.imd" 2>&1 >"$1"
            keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
        done
    done
}


################################################################################
# SuiteSparse collection

SSGET=ssget
NUM_PROBLEMS="$(${SSGET} -n)"

# Creates an input file for $1-th problem in the SuiteSparse collection
generate_suite_sparse_input() {
    INPUT=$(${SSGET} -i "$1" -e)
    cat << EOT
[{
    "filename": "${INPUT}",
    "problem": $(${SSGET} -i "$1" -j)
}]
EOT
}

parse_matrix_list() {
    local source_list_file=$1
    local benchmark_list=""
    local id=0
    for mtx in $(cat ${source_list_file}); do
        if [[ ! "$mtx" =~ ^[0-9]+$ ]]; then
            if [[ "$mtx" =~ ^[a-zA-Z0-9_-]+$ ]]; then
                id=$(${SSGET} -s "[ @name == $mtx ]")
            elif [[ "$mtx" =~ ^([a-zA-Z0-9_-]+)\/([a-zA-Z0-9_-]+)$ ]]; then
                local group="${BASH_REMATCH[1]}"
                local name="${BASH_REMATCH[2]}"
                id=$(${SSGET} -s "[ @name == $name ] && [ @group == $group ]")
            else
                >&2 echo -e "Could not recognize entry $mtx."
            fi
        else
            id=$mtx
        fi
        benchmark_list="$benchmark_list $id"
    done
    echo "$benchmark_list"
}

if [ $use_matrix_list_file -eq 1 ]; then
    MATRIX_LIST=($(parse_matrix_list $MATRIX_LIST_FILE))
    NUM_PROBLEMS=${#MATRIX_LIST[@]}
fi

LOOP_START=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID} - 1) / ${SEGMENTS}))
LOOP_END=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID}) / ${SEGMENTS}))
for (( p=${LOOP_START}; p < ${LOOP_END}; ++p )); do
    if [ $use_matrix_list_file -eq 1 ]; then
        i=${MATRIX_LIST[$((p-1))]}
    else
        i=$p
    fi
    if [ "${BENCHMARK}" == "preconditioner" ]; then
        break
    fi
    if [ "$(${SSGET} -i "$i" -preal)" = "0" ]; then
        clean_suite_sparse_problem "$i"
        continue
    fi
    RESULT_DIR="${SUITESPARSE_RESULT_DIR}"
    GROUP=$(${SSGET} -i "$i" -pgroup)
    NAME=$(${SSGET} -i "$i" -pname)
    RESULT_FILE="${RESULT_DIR}/${GROUP}/${NAME}.json"
    PREFIX="($i/${NUM_PROBLEMS}):\t"
    if result_exists "${RESULT_FILE}"; then
        echo -e "${PREFIX}Skipping ${GROUP}/${NAME}: ${RESULT_FILE} (or its intermediate files) already exists" 1>&2
        continue
    fi
    mkdir -p "$(dirname "${RESULT_FILE}")"
    generate_suite_sparse_input "$i" >"${RESULT_FILE}"

    echo -e "${PREFIX}Extracting statistics for ${GROUP}/${NAME}" 1>&2
    compute_matrix_statistics "${RESULT_FILE}"

    echo -e "${PREFIX}Running SpMV for ${GROUP}/${NAME}" 1>&2

    run_spmv_benchmarks "${RESULT_FILE}"

    if [ "${BENCHMARK}" == "conversions" ]; then
        echo -e "${PREFIX}Running Conversion for ${GROUP}/${NAME}" 1>&2
        run_conversion_benchmarks "${RESULT_FILE}"
    fi

    if [ "${BENCHMARK}" != "solver" -o \
         "$(${SSGET} -i "$i" -prows)" != "$(${SSGET} -i "$i" -pcols)" ]; then
        clean_suite_sparse_problem "$i"
        continue
    fi

    echo -e "${PREFIX}Running solvers for ${GROUP}/${NAME}" 1>&2
    run_solver_benchmarks "${RESULT_FILE}"

    echo -e "${PREFIX}Cleaning up problem ${GROUP}/${NAME}" 1>&2
    clean_suite_sparse_problem "$i"
done


if [ "${BENCHMARK}" != "preconditioner" ]; then
    exit
fi


################################################################################
# Generated collection

count_tokens() { echo "$#"; }

BLOCK_SIZES="$(seq 1 32)"
NUM_BLOCKS="$(seq 10000 2000 50000)"
NUM_PROBLEMS=$((
    $(count_tokens ${BLOCK_SIZES}) * $(count_tokens ${NUM_BLOCKS}) ))
ID=0


# Creates an input file for a block diagonal matrix with block size $1 and
# number of blocks $2. The location of the matrix is given by $3.
generate_block_diagonal_input() {
    cat << EOT
[{
    "filename": "$3",
    "problem": {
        "collection": "generated",
        "group": "block-diagonal",
        "name": "$2-$1",
        "type": "block-diagonal",
        "real": true,
        "binary": false,
        "2d3d": false,
        "posdef": false,
        "psym": 1,
        "nsym": 0,
        "kind": "artificially generated problem",
        "num_blocks": $2,
        "block_size": $1
    }
}]
EOT
}


# Generates the problem data using the input file $1
generate_problem() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.tmp"
    "${BENCH_BIN_DIR}"/matrix_generator/matrix_generator${BENCH_SUFFIX} <"$1.tmp" 2>&1 >"$1"
    keep_latest "$1" "$1.tmp"
}


LOOP_START=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID} - 1) / ${SEGMENTS}))
LOOP_END=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID}) / ${SEGMENTS}))
for bsize in ${BLOCK_SIZES}; do
    for nblocks in ${NUM_BLOCKS}; do
        ID=$((${ID} + 1))
        if [ "${ID}" -ge "${LOOP_END}" ]; then
            break
        fi
        if [ "${ID}" -lt "${LOOP_START}" ]; then
            continue
        fi
        RESULT_DIR="${GENERATED_RESULT_DIR}"
        GROUP="block-diagonal"
        NAME="${nblocks}-${bsize}"
        RESULT_FILE="${RESULT_DIR}/${GROUP}/${NAME}.json"
        PREFIX="(${ID}/${NUM_PROBLEMS}):\t"
        if result_exists "${RESULT_FILE}"; then
            echo -e "${PREFIX}Skipping ${GROUP}/${NAME}: ${RESULT_FILE} (or its intermediate files) already exists" 1>&2
            continue
        fi
        mkdir -p "$(dirname "${RESULT_FILE}")"
        mkdir -p "/tmp/${GROUP}"
        generate_block_diagonal_input \
            "${bsize}" "${nblocks}" "/tmp/${GROUP}/${NAME}.mtx" \
                >"${RESULT_FILE}"

        echo -e "${PREFIX}Generating problem ${GROUP}/${NAME}" 1>&2
        generate_problem "${RESULT_FILE}"

        echo -e "${PREFIX}Extracting statistics for ${GROUP}/${NAME}" 1>&2
        compute_matrix_statistics "${RESULT_FILE}"

        echo -e "${PREFIX}Running preconditioners for ${GROUP}/${NAME}" 1>&2
        BLOCK_SIZES="${bsize}"
        run_preconditioner_benchmarks "${RESULT_FILE}"

        echo -e "${PREFIX}Cleaning up problem ${GROUP}/${NAME}" 1>&2
        [ "${DRY_RUN}" != "true" ] && rm -r "/tmp/${GROUP}/${NAME}.mtx"
    done
    if [ "${ID}" -ge "${LOOP_END}" ]; then
        break
    fi
done
