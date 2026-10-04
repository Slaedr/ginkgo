#!/usr/bin/env python3
# Regression test for solver_compare with the GMRES-IR solver: the same
# system is solved with a plain CSR matrix and with an AMP matrix, and both
# must converge to the requested residual goal in the same number of outer
# iterations. The inner solver precision is selected by the format alone, so
# every AMP bin is expected to hold the full-precision entries here.
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

config = {
    "label_a": "gmres-ir+fgs",
    "label_b": "gmres-ir+fgs(amp)",
    "common": {
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
    },
    "config_a": {
        "formats": "csrc",
        "preconditioners": "fgs",
        "reorder": "multicolor",
        "fgs_sweeps": 1,
        "ir_inner_precision": "double",
    },
    "config_b": {
        "formats": "amp",
        "amp_base_type": "csrc",
        "preconditioners": "fgs",
        "reorder": "multicolor",
        "fgs_sweeps": 1,
        "ir_inner_precision": "double",
    },
    "matrices": [
        str(test_framework.matrixpath),
        {"stencil": "7pt", "size": 100},
    ],
}

failures = []


def check(condition: bool, message: str):
    if not condition:
        failures.append(message)


with tempfile.TemporaryDirectory() as tmpdir:
    config["output_file"] = str(pathlib.Path(tmpdir) / "results.json")
    config_file = pathlib.Path(tmpdir) / "config.json"
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
    results = json.loads(pathlib.Path(config["output_file"]).read_text())["results"]

check(len(results) == len(config["matrices"]), "unexpected number of results")
for entry in results:
    name = entry["matrix"]
    runs = {}
    for key in ["a", "b"]:
        run = entry[key]
        runs[key] = run
        check(run.get("completed", False),
              f"{name} [{key}]: did not complete: {run.get('error')}")
        if not run.get("completed", False):
            continue
        check(run["converged"], f"{name} [{key}]: did not converge")
        check(run["residual_goal_met"],
              f"{name} [{key}]: residual {run['residual_norm']} above goal "
              f"{run['residual_goal']}")
        check(run["iterations"] < config["common"]["max_iters"],
              f"{name} [{key}]: hit the iteration limit")
    if all(run.get("completed", False) for run in runs.values()):
        check(abs(runs["a"]["iterations"] - runs["b"]["iterations"]) <= 1,
              f"{name}: AMP needed {runs['b']['iterations']} iterations, "
              f"CSR needed {runs['a']['iterations']}")

if failures:
    print("FAIL")
    print("\n".join(failures))
    exit(1)
print("PASS")
