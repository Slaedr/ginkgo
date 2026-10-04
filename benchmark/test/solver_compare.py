#!/usr/bin/env python3
# Regression test for solver_compare with the GMRES-IR solver. The outer
# iteration always uses the double CSR matrix; the inner GMRES+FGS solver uses
# either the same double CSR matrix (baseline, config_a) or, in config_b, an
# AMP matrix (ir_inner_format) or a single-precision matrix
# (ir_inner_precision). Every variant must converge to the requested residual
# goal, and in the same number of outer iterations as the baseline up to a
# small tolerance.
#
# Unlike the other benchmark tests, solver_compare reports its results in an
# output file and prints a human-readable table to stdout, so this test checks
# the results file instead of comparing against reference output.
import json
import subprocess
import sys
import tempfile
import pathlib
import test_framework

common = {
    "executor": "reference",
    "solvers": "gmres_ir",
    "gmres_restart": 100,
    "max_iters": 500,
    "rel_res_goal": 1e-8,
    "rhs_generation": "1",
    "initial_guess_generation": "0",
    "repetitions": "1",
    "warmup": 0,
    "precision": "double",
    "preconditioners": "fgs",
    "reorder": "multicolor",
    "fgs_sweeps": 1,
}
baseline = {"formats": "csrc", "ir_inner_format": "csrc",
            "ir_inner_precision": "double"}
variants = {
    "amp inner": {"formats": "csrc", "ir_inner_format": "amp",
                  "amp_base_type": "csrc", "amp_tolerance": 1e-6,
                  "ir_inner_precision": "double"},
    "single inner": {"formats": "csrc", "ir_inner_format": "csrc",
                     "ir_inner_precision": "single"},
}
matrices = [
    str(test_framework.matrixpath),
    {"stencil": "7pt", "size": 100},
]
# Allowed difference in outer iterations between a variant and the baseline.
# A reduced-precision inner solver may need a few more outer iterations.
max_extra_iterations = 5

failures = []


def check(condition: bool, message: str):
    if not condition:
        failures.append(message)


def run(tmpdir: pathlib.Path, label: str, config_b: dict):
    config = {
        "label_a": "baseline",
        "label_b": label,
        "output_file": str(tmpdir / "results.json"),
        "common": common,
        "config_a": baseline,
        "config_b": config_b,
        "matrices": matrices,
    }
    config_file = tmpdir / "config.json"
    config_file.write_text(json.dumps(config))
    command = [sys.argv[1], str(config_file)]
    print("TEST: {}".format(" ".join("'{}'".format(arg) for arg in command)))
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print("FAIL: solver_compare exited with code", result.returncode)
        print(result.stdout.decode())
        print(result.stderr.decode())
        exit(1)
    return json.loads((tmpdir / "results.json").read_text())["results"]


with tempfile.TemporaryDirectory() as tmpdir:
    for label, config_b in variants.items():
        results = run(pathlib.Path(tmpdir), label, config_b)
        check(len(results) == len(matrices),
              f"[{label}] unexpected number of results")
        for entry in results:
            name = f"[{label}] {entry['matrix']}"
            runs = {key: entry[key] for key in ["a", "b"]}
            for key, run_result in runs.items():
                check(run_result.get("completed", False),
                      f"{name} [{key}]: did not complete: "
                      f"{run_result.get('error')}")
                if not run_result.get("completed", False):
                    continue
                check(run_result["converged"],
                      f"{name} [{key}]: did not converge")
                check(run_result["residual_goal_met"],
                      f"{name} [{key}]: residual "
                      f"{run_result['residual_norm']} above goal "
                      f"{run_result['residual_goal']}")
                check(run_result["iterations"] < common["max_iters"],
                      f"{name} [{key}]: hit the iteration limit")
            if all(r.get("completed", False) for r in runs.values()):
                extra = runs["b"]["iterations"] - runs["a"]["iterations"]
                check(extra <= max_extra_iterations,
                      f"{name}: needed {runs['b']['iterations']} outer "
                      f"iterations, baseline needed "
                      f"{runs['a']['iterations']}")

if failures:
    print("FAIL")
    print("\n".join(failures))
    exit(1)
print("PASS")
