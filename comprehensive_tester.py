#!/usr/bin/env python3
"""Independent, offline final-project tester; never include this file in submission.

Run: python3 comprehensive_tester.py [--jobs 2] [--sanitizers]
Requires the course's Python, NumPy, scikit-learn, setuptools, make, and gcc.
No packages are installed and no network requests are made. Sources are copied
to a fresh temporary directory; reports and reproducible inputs are retained.
Exit codes: 0=no confirmed required failures, 1=required failures, 2=tester/setup
error. SKIP is never counted as a successful numerical comparison.

References implement Final_Project.pdf sections 1.1--1.5 independently. A zero
H entry stays zero; no epsilon is added to the denominator. Normalization with
zero floating-point degrees, overflow, unspecified empty-cluster policies, and
ambiguous ties are diagnosed rather than invented assignment requirements.
The local HW1 helpers factorize/fit are optional observations: absence is SKIP.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ATOL = 1e-4
EPSILON = 1e-4
MAX_ITER = 300
SEED = 1234
ERROR = "An Error Has Occurred"
K_ERROR = "Incorrect number of clusters!"
FLAGS = ["-ansi", "-Wall", "-Wextra", "-Werror", "-pedantic-errors"]
GOALS = ("sym", "ddg", "norm", "symnmf")
NUMBER = r"-?\d+\.\d{4}"


def json_safe(value):
    """Keep reports valid JSON, even when a failed program produces NaN/inf."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def compare_array(actual, expected):
    """Compare shape, finiteness and absolute error; never use relative tolerance."""
    try:
        actual = np.asarray(actual, dtype=float)
    except (TypeError, ValueError) as error:
        return False, None, "Non-numeric matrix: " + str(error)
    if actual.shape != expected.shape:
        return False, None, "Wrong shape: {} instead of {}".format(actual.shape, expected.shape)
    if not np.isfinite(actual).all():
        return False, None, "Output contains NaN or infinity"
    difference = float(np.max(np.abs(actual - expected)))
    return difference <= ATOL, difference, "Absolute tolerance is 1e-4"


def parse_matrix(text, shape):
    """Require only numeric CSV, exact dimensions, and exactly four decimals."""
    body = text[:-1] if text.endswith("\n") else text
    lines = body.split("\n")
    pattern = re.compile(NUMBER + r"(?:," + NUMBER + r")*")
    if len(lines) != shape[0] or any(not pattern.fullmatch(line) for line in lines):
        raise ValueError("Expected {} CSV rows, four decimals, and no extra text".format(shape[0]))
    rows = [line.split(",") for line in lines]
    if any(len(row) != shape[1] for row in rows):
        raise ValueError("Expected {} columns per row".format(shape[1]))
    return np.asarray(rows, dtype=float)


def reference_h_gram(w, h):
    """A second independent update arrangement corroborates a suspected H error."""
    current = h.copy()
    for _ in range(MAX_ITER):
        top = w.dot(current)
        bottom = current.dot(current.T.dot(current))
        active = current != 0
        if np.any(active & (bottom <= 0)):
            return None
        following = np.zeros_like(current)
        following[active] = current[active] * (0.5 + 0.5 * top[active] / bottom[active])
        delta = np.sum(np.square(following - current))
        current = following
        if delta < EPSILON:
            break
    return current


def scalar_graph(points, goal):
    """Use scalar math and fsum to corroborate a failed vectorized graph oracle."""
    n = len(points)
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = math.fsum((float(x) - float(y)) ** 2
                                 for x, y in zip(points[i], points[j]))
            a[i, j] = a[j, i] = math.exp(-distance / 2)
    if goal == "sym":
        return a
    degrees = [math.fsum(row) for row in a]
    if goal == "ddg":
        return np.diag(degrees)
    if min(degrees) <= 0:
        return None
    return np.array([[a[i, j] / math.sqrt(degrees[i]) / math.sqrt(degrees[j])
                      for j in range(n)] for i in range(n)])


def scalar_kmeans(points, k):
    """A scalar Lloyd implementation is used only to corroborate a suspected bug."""
    data = points.tolist()
    centers = [point[:] for point in data[:k]]
    for _ in range(MAX_ITER):
        groups = [[] for _ in range(k)]
        for point in data:
            distances = [math.fsum((x-y)**2 for x, y in zip(point, center)) for center in centers]
            groups[distances.index(min(distances))].append(point)
        updated = [([math.fsum(row[j] for row in group) / len(group) for j in range(len(data[0]))]
                    if group else centers[index][:]) for index, group in enumerate(groups)]
        converged = all(math.sqrt(math.fsum((x-y)**2 for x, y in zip(old, new))) < EPSILON
                        for old, new in zip(centers, updated))
        centers = updated
        if converged:
            break
    labels = []
    for point in data:
        distances = [math.fsum((x-y)**2 for x, y in zip(point, center)) for center in centers]
        labels.append(distances.index(min(distances)))
    return np.asarray(centers), np.asarray(labels)


class ComprehensiveTester:
    """Execute isolated actual code; keep independent expected results in parent."""

    def __init__(self, project, folder, args):
        self.project, self.folder, self.args = project, folder, args
        self.results, self.datasets = [], []
        self.case_scopes = {}
        self.started = time.monotonic()
        self.counter = 0
        self.environment = os.environ.copy()
        self.environment.pop("PYTHONPATH", None)
        self.environment.update(LC_ALL="C", PYTHONHASHSEED="0")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            self.environment[name] = "1"
        self.environment["CC"] = shutil.which("gcc") or "gcc"
        if sys.platform.startswith("linux"):
            self.environment["LDSHARED"] = self.environment["CC"] + " -shared"
        self.binary = str(folder / "symnmf")
        self.driver = folder / "_comprehensive_driver.py"
        self.c_ready = self.python_ready = False
        self.requests = folder / "requests"
        self.requests.mkdir()
        (folder / "failures").mkdir()
        self.executor = ThreadPoolExecutor(max_workers=args.jobs)

    def record(self, status, category, name, description, scope="required", **details):
        if scope == "required" and self.case_scopes.get(details.get("dataset")) == "numerical_limit":
            scope = "diagnostic"
            if status == "FAIL":
                details["diagnostic_outcome"] = "Numerical-limit observation; no assignment failure claim"
                status = "SKIP"
        item = {"id": len(self.results) + 1, "status": status, "category": category,
                "name": name, "description": description, "scope": scope}
        item.update(json_safe(details))
        if status == "FAIL":
            item.setdefault("reason", description)
            actual = details.get("actual")
            item.setdefault("command", actual.get("command") if isinstance(actual, dict) else details.get("function_call"))
            item.setdefault("dataset", None)
            item.setdefault("max_abs_difference", None)
            target = self.folder / "failures" / ("failure_{:04d}.json".format(item["id"]))
            target.write_text(json.dumps(item, indent=2, allow_nan=False), encoding="utf-8")
            item["failure_report"] = str(target)
        self.results.append(item)
        print("[{}] {} / {}: {}".format(status, category, name, description), flush=True)
        return item

    def execute(self, command, timeout=None, environment=None):
        started = time.monotonic()
        try:
            result = subprocess.run(command, cwd=str(self.folder),
                                    env=environment or self.environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=timeout or self.args.timeout)
            return {"command": command, "exit_code": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr,
                    "elapsed_seconds": time.monotonic() - started}
        except subprocess.TimeoutExpired as error:
            return {"command": command, "exit_code": None, "timeout": True,
                    "stdout": (error.stdout or b"").decode(errors="replace")
                    if isinstance(error.stdout, bytes) else error.stdout or "",
                    "stderr": "Command exceeded {} seconds".format(timeout or self.args.timeout),
                    "elapsed_seconds": time.monotonic() - started}
        except OSError as error:
            return {"command": command, "exit_code": None, "stdout": "", "stderr": str(error),
                    "elapsed_seconds": time.monotonic() - started}

    def build(self):
        """Test the actual Makefile binary separately from a strict GCC build."""
        make = self.execute(["make"], max(120, self.args.timeout))
        okay = make["exit_code"] == 0 and not re.search(r"\bwarning:", make["stdout"] + make["stderr"], re.I)
        self.c_ready = okay and Path(self.binary).is_file()
        self.record("PASS" if self.c_ready else "FAIL", "build", "make",
                    "Makefile builds the standalone executable without warnings",
                    expected="exit 0, symnmf executable, no compiler warnings", actual=make,
                    implementation="Makefile")
        dry = self.execute(["make", "-B", "-n"])
        compile_lines = [line for line in dry["stdout"].splitlines()
                         if " -c " in " " + line + " "]
        flags_ok = dry["exit_code"] == 0 and bool(compile_lines)
        flags_ok = flags_ok and all(all(flag in shlex.split(line) for flag in FLAGS)
                                    for line in compile_lines)
        self.record("PASS" if flags_ok else "FAIL", "build", "Makefile flags",
                    "Each compilation command includes all five required flags",
                    expected=FLAGS, actual=dry, implementation="Makefile")
        c_sources = sorted(p.name for p in self.folder.glob("*.c") if p.name != "symnmfmodule.c")
        command = [self.environment["CC"]] + FLAGS + c_sources + ["-o", "symnmf_strict", "-lm"]
        strict = self.execute(command, max(120, self.args.timeout))
        okay = strict["exit_code"] == 0 and not re.search(r"\bwarning:", strict["stdout"] + strict["stderr"], re.I)
        self.record("PASS" if okay else "FAIL", "build", "strict C89 compilation",
                    "Fresh independent GCC build with the assignment flags",
                    expected="exit 0 without warnings", actual=strict, implementation="C source files")
        extension = self.execute([sys.executable, "setup.py", "build_ext", "--inplace"],
                                 max(120, self.args.timeout))
        self.python_ready = extension["exit_code"] == 0 and not re.search(
            r"\bwarning:", extension["stdout"] + extension["stderr"], re.I)
        self.record("PASS" if self.python_ready else "FAIL", "build", "Python extension",
                    "setup.py builds the extension without warnings",
                    expected="exit 0 without warnings", actual=extension, implementation="setup.py")

    def write_dataset(self, case):
        """Validate tester-generated data before any program can see it."""
        points, k = np.asarray(case["points"], dtype=float), int(case["k"])
        if points.ndim != 2 or points.shape[1] < 1 or not 1 < k < len(points):
            raise ValueError("Invalid tester dimensions or k: " + case["name"])
        if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Non-finite or duplicate tester points: " + case["name"])
        path = self.folder / (case["name"] + ".txt")
        np.savetxt(path, points, delimiter=",", fmt="%.17g")
        restored = np.loadtxt(path, delimiter=",", ndmin=2)
        if not np.array_equal(restored, points):
            raise ValueError("Tester serialization lost precision: " + case["name"])
        case["points"], case["filename"] = points, path.name
        self.case_scopes[path.name] = case.get("scope", "required")
        self.datasets.append({"name": case["name"], "file": str(path), "n": len(points),
                              "dimension": points.shape[1], "k": k,
                              "description": case["description"], "tags": case.get("tags", []),
                              "scope": case.get("scope", "required")})

    def corroborate(self, case, goal, expected, h0=None, w=None):
        try:
            other = reference_h_gram(w, h0) if goal == "symnmf" else scalar_graph(case["points"], goal)
            return other is not None and compare_array(other, expected)[0]
        except (ArithmeticError, ValueError, FloatingPointError):
            return False

    def matrix_result(self, case, category, name, observation, expected, goal, h0=None, w=None):
        """A numerical FAIL requires an independently corroborated expectation."""
        detail = dict(command=observation["command"], dataset=case["filename"],
                      implementation="symnmf.c / " + goal if category.startswith("C") else "symnmf.py / " + goal)
        if observation["exit_code"] != 0 or observation["stderr"]:
            return self.record("FAIL", category, name, "Expected successful output without stderr",
                               expected=expected, actual=observation, reason="process failure", **detail)
        try:
            actual = parse_matrix(observation["stdout"], expected.shape)
        except ValueError as error:
            return self.record("FAIL", category, name, str(error), expected=expected,
                               actual=observation, reason="output structure or formatting", **detail)
        okay, difference, reason = compare_array(actual, expected)
        if not okay and not self.corroborate(case, goal, expected, h0, w):
            return self.record("SKIP", category, name,
                               "Numerical discrepancy needs investigation: independent oracles disagree",
                               expected=expected, actual=observation, max_abs_difference=difference,
                               scope="diagnostic", reason="suspected numerical sensitivity", **detail)
        return self.record("PASS" if okay else "FAIL", category, name,
                           "Four-decimal matrix agrees with independent reference" if okay else reason,
                           max_abs_difference=difference,
                           **({} if okay else {"expected": expected, "actual": observation,
                                              "reason": "corroborated numerical mismatch"}), **detail)

    def worker_request(self, name, request):
        self.counter += 1
        path = self.requests / ("{:04d}_{}.json".format(self.counter, name))
        path.write_text(json.dumps(json_safe(request), allow_nan=False), encoding="utf-8")
        command = [sys.executable, str(self.driver), "--worker", str(path)]
        return command

    def decode_worker(self, observation):
        if observation["exit_code"] != 0 or observation["stderr"]:
            return None
        try:
            value = json.loads(observation["stdout"])
            return value if isinstance(value, dict) else None
        except (ValueError, TypeError):
            return None

    def extension_results(self, case, observation, graph, h_result, h0, only_symnmf=False):
        response = self.decode_worker(observation)
        goals = [] if only_symnmf else [goal for goal in ("sym", "ddg", "norm") if graph.get(goal) is not None]
        if h_result is not None and h_result.get("h") is not None:
            goals.append("symnmf")
        if response is None or "error" in response:
            self.record("FAIL", "extension", case["name"], "Direct extension process failed",
                        command=observation["command"], dataset=case["filename"],
                        expected="JSON observations for valid matrix calls", actual=observation,
                        implementation="symnmfmodule.c", reason="extension import, crash or call error")
            return
        if Path(response["module_file"]).resolve().parent != self.folder.resolve():
            raise RuntimeError("Worker imported a stale extension outside the temporary build")
        for goal in goals:
            name = case["name"] + ": " + goal
            expected = h_result["h"] if goal == "symnmf" else graph[goal]
            details = dict(command=observation["command"], function_call="symnmfmodule." + goal,
                           dataset=case["filename"], implementation="symnmfmodule.c / " + goal)
            if goal in response.get("errors", {}):
                self.record("FAIL", "extension", name, "Valid direct extension call raised an exception",
                            expected=expected, actual=response["errors"][goal], reason="exception", **details)
                continue
            actual = response.get("matrices", {}).get(goal)
            okay, difference, reason = compare_array(actual, expected)
            if not okay and not self.corroborate(case, goal, expected, h0, graph.get("norm")):
                self.record("SKIP", "extension", name, "Independent oracles disagree on this numerical edge",
                            scope="diagnostic", expected=expected, actual=actual,
                            max_abs_difference=difference, **details)
            else:
                self.record("PASS" if okay else "FAIL", "extension", name,
                            "Full matrix matches independent reference" if okay else reason,
                            max_abs_difference=difference,
                            **({} if okay else {"expected": expected, "actual": actual,
                                               "reason": "corroborated numerical mismatch"}), **details)
            metadata = response.get("metadata", {}).get(goal, {})
            good_type = (metadata.get("rectangular") and metadata.get("shape") == list(expected.shape)
                         and all(item in ("float", "float64", "float32")
                                 for item in metadata.get("element_types", ["missing"])))
            self.record("PASS" if good_type else "FAIL", "extension types", name,
                        "Returns a rectangular matrix of floating-point numbers",
                        expected={"shape": list(expected.shape), "elements": "floating-point"},
                        actual=metadata, **details)
            for key, category, description in (("input_unchanged", "input mutation", "Caller-owned matrices remain unchanged"),
                                                ("repeat_equal", "repeatability", "Repeated calls return identical results")):
                okay = response.get(key, {}).get(goal, False)
                self.record("PASS" if okay else "FAIL", category, name, description,
                            expected=True, actual=okay, **details)

    def analysis_result(self, case, observation, h):
        expected = reference_analysis(case["points"], case["k"], h)
        details = dict(command=observation["command"], dataset=case["filename"], implementation="analysis.py")
        if observation.get("timeout") or (observation["exit_code"] is not None and observation["exit_code"] < 0):
            self.record("FAIL", "analysis", case["name"], "Valid data caused a crash or exceeded the timeout",
                        expected="bounded execution without a crash", actual=observation, **details)
            return
        km = expected.get("kmeans", {})
        ambiguous = (expected.get("nmf_ambiguous") or km.get("empty_clusters")
                     or km.get("tie") or km.get("near_tie"))
        if ambiguous:
            self.record("SKIP", "analysis", case["name"],
                        "Tied assignments or unspecified empty-cluster policy prevent a unique score oracle",
                        expected=expected, actual=observation, **details)
            return
        if expected.get("scores") is None:
            accepted = (observation["exit_code"] is not None and observation["exit_code"] > 0
                        and observation["stdout"] == ERROR + "\n" and not observation["stderr"])
            if observation["exit_code"] is not None and observation["exit_code"] < 0:
                self.record("FAIL", "analysis undefined", case["name"],
                            "Valid data caused a process crash even though scores are undefined",
                            expected={"stdout": ERROR + "\n", "exit": "positive"}, actual=observation,
                            reason=expected.get("reason"), **details)
            else:
                self.record("SKIP", "analysis", case["name"], expected.get("reason", "Undefined score"),
                            accepted_error_response=accepted, actual=observation,
                            reason="No mathematically defined numerical score to compare", **details)
            return
        pattern = (r"nmf: (" + NUMBER + r")\nkmeans: (" + NUMBER + r")\nari: (" + NUMBER + r")\n?")
        match = re.fullmatch(pattern, observation["stdout"])
        if observation["exit_code"] != 0 or observation["stderr"] or match is None:
            self.record("FAIL", "analysis", case["name"], "Need three correctly formatted score lines",
                        expected=expected["scores"], actual=observation, reason="process/format failure", **details)
            return
        actual = np.asarray([float(x) for x in match.groups()])
        okay, difference, reason = compare_array(actual, np.asarray(expected["scores"]))
        if not okay:
            _, scalar_labels = scalar_kmeans(case["points"], case["k"])
            graph = reference_graph(case["points"])
            second_h = reference_h_gram(graph["norm"], reference_initial_h(graph["norm"], case["k"]))
            if second_h is None or not compare_array(second_h, h)[0]:
                self.record("SKIP", "analysis", case["name"], "Independent H oracles disagree",
                            expected=expected, actual=observation, scope="diagnostic", **details)
                return
            nmf_labels = second_h.argmax(axis=1)
            corroborated = np.asarray([silhouette_score(case["points"], nmf_labels),
                                       silhouette_score(case["points"], scalar_labels),
                                       adjusted_rand_score(nmf_labels, scalar_labels)])
            if not compare_array(corroborated, np.asarray(expected["scores"]))[0]:
                self.record("SKIP", "analysis", case["name"], "Scalar and vector score oracles disagree",
                            expected=expected, actual=observation, scope="diagnostic", **details)
                return
        self.record("PASS" if okay else "FAIL", "analysis", case["name"],
                    "Independent NMF/K-means silhouettes and ARI match" if okay else reason,
                    max_abs_difference=difference,
                    **({} if okay else {"expected": expected, "actual": observation,
                                       "reason": "independent clustering score mismatch"}), **details)

    def initialization_result(self, case, observation, h0, expected_h):
        response = self.decode_worker(observation)
        details = dict(command=observation["command"], dataset=case["filename"], implementation="symnmf.py:factorize")
        if response is None or response.get("error") or response.get("optional_unavailable"):
            # factorize is a local helper, not a required external API name.
            missing = response and response.get("optional_unavailable")
            self.record("SKIP" if missing else "FAIL", "initialization", case["name"],
                        "Local factorize helper unavailable" if missing else "Valid factorization call failed",
                        actual=observation, expected="two successful factorizations", **details)
            return
        outputs = response.get("uniform_calls", [])
        seed_ok = response.get("seed_calls") == [SEED, SEED]
        events_ok = response.get("events") == ["seed", "uniform", "seed", "uniform"]
        uniform_ok = len(outputs) == 2 and all(compare_array(call["result"], h0)[0] for call in outputs)
        self.record("PASS" if seed_ok and events_ok and uniform_ok else "FAIL", "initialization",
                    case["name"], "Seed 1234 immediately precedes uniform H with the required bounds and shape",
                    expected={"seed_calls": [SEED, SEED], "events": ["seed", "uniform"] * 2, "H": h0},
                    actual={key: response.get(key) for key in ("seed_calls", "events", "uniform_calls")}, **details)
        okay, difference, reason = compare_array(response["first"], expected_h)
        if not okay and not self.corroborate(case, "symnmf", expected_h, h0, reference_graph(case["points"])["norm"]):
            self.record("SKIP", "factorization", case["name"], "Independent final-H oracles disagree",
                        expected=expected_h, actual=response["first"], scope="diagnostic", **details)
            okay = None
        if okay is not None:
            self.record("PASS" if okay else "FAIL", "factorization", case["name"],
                        "Final H matches independent updates" if okay else reason,
                        expected=expected_h if not okay else "independent final H",
                        actual=response["first"] if not okay else "matching matrix", max_abs_difference=difference, **details)
        for key, description in (("repeat_equal", "Unrelated random draws do not change repeated H"),
                                  ("points_unchanged", "Factorization does not mutate input points")):
            self.record("PASS" if response.get(key) else "FAIL", "factorization", case["name"] + ": " + key,
                        description, expected=True, actual=response.get(key), **details)

    def kmeans_result(self, case, observation):
        expected = reference_kmeans(case["points"], case["k"])
        details = dict(command=observation["command"], dataset=case["filename"], implementation="kmeans.py")
        response = self.decode_worker(observation)
        if observation.get("timeout") or (observation["exit_code"] is not None and observation["exit_code"] < 0):
            self.record("FAIL", "K-means", case["name"], "Valid data caused a crash or exceeded the timeout",
                        expected="bounded execution without a crash", actual=observation, **details)
            return
        if expected.get("empty_clusters") or expected.get("tie") or expected.get("near_tie") or expected.get("numerical_issue"):
            self.record("SKIP", "K-means", case["name"],
                        "Empty-cluster/tie policy or floating-point limit is outside specified behavior",
                        actual=response or observation, **details)
            return
        if response is None or response.get("error") or response.get("optional_unavailable"):
            missing = response and response.get("optional_unavailable")
            self.record("SKIP" if missing else "FAIL", "K-means", case["name"],
                        "Optional HW1 helper unavailable" if missing else "Valid K-means call failed",
                        expected=expected, actual=response or observation, **details)
            return
        okay, difference, reason = compare_array(response["centroids"], expected["centers"])
        okay = okay and response["labels"] == expected["labels"].tolist()
        if not okay:
            centers, labels = scalar_kmeans(case["points"], case["k"])
            if not (compare_array(centers, expected["centers"])[0] and np.array_equal(labels, expected["labels"])):
                self.record("SKIP", "K-means", case["name"], "Scalar and vector oracles disagree",
                            expected=expected, actual=response, scope="diagnostic", **details)
                return
        self.record("PASS" if okay else "FAIL", "K-means", case["name"],
                    "Independent HW1 centroids and assignments match" if okay else reason,
                    max_abs_difference=difference, expected=expected if not okay else "reference centroids/labels",
                    actual=response if not okay else "matching centroids/labels", **details)
        for key in ("input_unchanged", "repeat_equal"):
            self.record("PASS" if response.get(key) else "FAIL", "K-means", case["name"] + ": " + key,
                        "HW1 computation preserves inputs and is deterministic", expected=True,
                        actual=response.get(key), **details)

    def run_dataset(self, case):
        self.write_dataset(case)
        graph = reference_graph(case["points"])
        h0 = h_result = None
        if graph.get("norm") is not None:
            h0 = reference_initial_h(graph["norm"], case["k"])
            h_result = reference_symnmf(graph["norm"], initial_h=h0)
        metadata = self.datasets[-1]
        if h_result:
            metadata.update(iterations=h_result["iterations"], stopped=h_result["stopped"],
                            final_change=h_result["changes"][-1] if h_result["changes"] else None)
        # Original cases retain their original coverage; new cases cover every goal.
        goals = [case["original_goal"]] if case.get("original_goal") else list(GOALS)
        jobs = []
        for goal in goals:
            expected = h_result.get("h") if goal == "symnmf" and h_result else graph.get(goal)
            if expected is None:
                for category in (["Python CLI"] if goal == "symnmf" else ["C CLI", "Python CLI"]):
                    self.record("SKIP", category, case["name"] + ": " + goal,
                                graph.get("reason") or (h_result or {}).get("reason") or "Undefined normalization/update",
                                dataset=case["filename"], scope="diagnostic")
                continue
            if goal != "symnmf":
                command = [self.binary, goal, case["filename"]]
                jobs.append(("matrix", "C CLI", goal, expected,
                             self.executor.submit(self.execute, command) if self.c_ready else None))
            command = [sys.executable, "symnmf.py", str(case["k"]), goal, case["filename"]]
            jobs.append(("matrix", "Python CLI", goal, expected,
                         self.executor.submit(self.execute, command) if self.python_ready else None))
        if h_result is not None and h_result.get("h") is not None:
            command = [sys.executable, "analysis.py", str(case["k"]), case["filename"]]
            jobs.append(("analysis", "analysis", None, None,
                         self.executor.submit(self.execute, command) if self.python_ready else None))
        else:
            self.record("SKIP", "analysis", case["name"], "No mathematically defined normalized reference",
                        dataset=case["filename"], scope="diagnostic")
        if not case.get("original_goal"):
            request = {"kind": "extension", "points": case["points"], "k": case["k"],
                       "goals": [goal for goal in ("sym", "ddg", "norm") if graph.get(goal) is not None]}
            if h_result is not None and h_result.get("h") is not None:
                request.update(h=h0, w=graph["norm"])
                request["goals"].append("symnmf")
            command = self.worker_request(case["name"] + "_extension", request)
            jobs.append(("extension", "extension", None, None,
                         self.executor.submit(self.execute, command) if self.python_ready else None))
            if h_result is not None and h_result.get("h") is not None:
                for kind in ("factorize", "kmeans"):
                    command = self.worker_request(case["name"] + "_" + kind,
                                                  {"kind": kind, "points": case["points"], "k": case["k"]})
                    jobs.append((kind, kind, None, None,
                                 self.executor.submit(self.execute, command) if self.python_ready else None))
        # Drain this dataset's concurrent work before confirming a timeout alone.
        completed = [(kind, category, goal, expected, future.result() if future else None)
                     for kind, category, goal, expected, future in jobs]
        for kind, category, goal, expected, observation in completed:
            if observation is None:
                self.record("SKIP", category, case["name"], "Required build unavailable", dataset=case["filename"])
                continue
            if observation.get("timeout"):
                first_attempt = observation
                observation = self.execute(first_attempt["command"])
                observation["initial_timeout"] = first_attempt
                if not observation.get("timeout"):
                    self.record("SKIP", "timing confirmation", case["name"] + ": " + kind + ": " + str(goal),
                                "Concurrent timeout did not recur in an isolated rerun; numerical checks still apply",
                                scope="diagnostic", command=first_attempt["command"], dataset=case["filename"],
                                actual={"initial": first_attempt, "isolated": observation})
            if kind == "matrix":
                self.matrix_result(case, category, case["name"] + ": " + goal, observation,
                                   expected, goal, h0, graph.get("norm"))
            elif kind == "analysis":
                self.analysis_result(case, observation, h_result["h"])
            elif kind == "extension":
                self.extension_results(case, observation, graph, h_result, h0)
            elif kind == "factorize":
                self.initialization_result(case, observation, h0, h_result["h"])
            elif kind == "kmeans":
                self.kmeans_result(case, observation)

    def arguments_and_formats(self):
        """Invalid arguments use valid files; analysis inputs are always legal."""
        points = np.array([[0., 0.], [.1, 0.], [0., .1], [3., 3.], [3.1, 3.], [3., 3.1]])
        case = dict(name="argument_fixture", points=points, k=2, description="Distinct valid CLI fixture")
        self.write_dataset(case)
        graph = reference_graph(points)
        h0 = reference_initial_h(graph["norm"], 2)
        h = reference_symnmf(graph["norm"], initial_h=h0)["h"]
        pending = []
        for goal in GOALS:
            for k in ("0", "1", "-2", "6", "7", "hello", "2.5", "nan", "inf"):
                command = [sys.executable, "symnmf.py", k, goal, case["filename"]]
                pending.append(("Invalid k={} goal={}".format(k, goal), command, K_ERROR))
        for name, tail in (("no arguments", []), ("only k", ["2"]), ("missing file", ["2", "sym"]),
                           ("extra argument", ["2", "sym", case["filename"], "extra"]),
                           ("unknown goal", ["2", "oops", case["filename"]])):
            pending.append(("Python " + name, [sys.executable, "symnmf.py"] + tail, ERROR))
        for name, tail in (("no arguments", []), ("missing file", ["sym"]),
                           ("extra argument", ["sym", case["filename"], "extra"]),
                           ("unknown goal", ["oops", case["filename"]]),
                           ("unsupported symnmf", ["symnmf", case["filename"]])):
            pending.append(("C " + name, [self.binary] + tail, ERROR))
        def available(command):
            return self.c_ready if command[0] == self.binary else self.python_ready

        futures = [self.executor.submit(self.execute, command) if available(command) else None
                   for _, command, _ in pending]
        for (name, command, message), future in zip(pending, futures):
            if future is None:
                self.record("SKIP", "arguments", name, "Corresponding build unavailable", command=command)
                continue
            observation = future.result()
            okay = (observation["exit_code"] is not None and observation["exit_code"] > 0
                    and observation["stdout"] == message + "\n" and not observation["stderr"])
            self.record("PASS" if okay else "FAIL", "arguments", name,
                        "Required error text and non-crash failure status", command=command,
                        dataset=case["filename"], expected={"message": message, "exit": "positive"},
                        actual=observation, implementation="symnmf.py / main" if "Python" in name or "goal=" in name else "symnmf.c / main")
        for goal in GOALS:
            expected = h if goal == "symnmf" else graph[goal]
            if not self.python_ready:
                self.record("SKIP", "arguments", "valid k: " + goal, "Python extension build unavailable")
                continue
            for k in ("2", "2.0", "2.000", "+2", "+2.000", "0002"):
                command = [sys.executable, "symnmf.py", k, goal, case["filename"]]
                self.matrix_result(case, "arguments", "Accepted k={} goal={}".format(k, goal),
                                   self.execute(command), expected, goal, h0, graph["norm"])
        normal = "\n".join(",".join("%.17g" % value for value in row) for row in points)
        for label, content in (("LF", normal + "\n"), ("CRLF", normal.replace("\n", "\r\n") + "\r\n"),
                               ("no final newline", normal)):
            file = self.folder / ("format_" + label.replace(" ", "_") + ".txt")
            file.write_bytes(content.encode("ascii"))
            variant = dict(case, filename=file.name)
            for goal in GOALS:
                expected = h if goal == "symnmf" else graph[goal]
                commands = [[sys.executable, "symnmf.py", "2", goal, file.name]]
                if goal != "symnmf":
                    commands.append([self.binary, goal, file.name])
                for command in commands:
                    if not available(command):
                        self.record("SKIP", "formats", label + ": " + goal, "Corresponding build unavailable", command=command)
                        continue
                    self.matrix_result(variant, "formats", label + ": " + goal + ": " + Path(command[0]).name,
                                       self.execute(command), expected, goal, h0, graph["norm"])
        if (self.folder / "symnmf_strict").is_file():
            command = [str(self.folder / "symnmf_strict"), "norm", case["filename"]]
            self.matrix_result(case, "build", "Strict-build executable smoke test", self.execute(command),
                               graph["norm"], "norm")

    def custom_h_tests(self):
        """Exercise legal sparse/zero H and absolute stopping/max-iteration limits."""
        points = np.array([[0., 0.], [.3, 0.], [1., .2], [2., 1.]])
        case = dict(name="custom_H_fixture", points=points, k=2, description="Valid zero/sparse H matrices")
        self.write_dataset(case)
        graph = reference_graph(points)
        initial = reference_initial_h(graph["norm"], 2)
        variants = [("all zero H", np.zeros_like(initial), graph["norm"]),
                    ("zero column H", np.column_stack((initial[:, 0], np.zeros(len(points)))), graph["norm"]),
                    ("zero row H", initial.copy(), graph["norm"]),
                    ("tiny positive H", initial * 1e-12, graph["norm"]),
                    ("max_iter_300", np.full_like(initial, 1e100), graph["norm"])]
        variants[2][1][0] = 0
        for name, h0, w in variants:
            expected = reference_symnmf(w, initial_h=h0)
            if expected["h"] is None:
                self.record("SKIP", "custom H", name, expected.get("reason", "Undefined update"))
                continue
            command = self.worker_request(name.replace(" ", "_"), {"kind": "symnmf", "h": h0, "w": w, "repeat": 5})
            observation = self.execute(command)
            variant = dict(case, name="custom H " + name)
            self.extension_results(variant, observation, {"norm": w}, expected, h0, only_symnmf=True)
            if name == "max_iter_300":
                assert expected["iterations"] == 300 and expected["stopped"] == "max_iter"
                self.record("PASS", "convergence coverage", "300-iteration fixture",
                            "Independent updates reach the maximum, rather than stopping early", scope="tester",
                            actual={"iterations": expected["iterations"], "stopped": expected["stopped"]})

    def memory_tests(self):
        valgrind = shutil.which("valgrind")
        cases = [case for case in self.datasets if case["scope"] == "required"]
        selected = [case for case in cases if case["name"] in
                    ("argument_fixture", "custom_H_fixture", "long_line_1024_dimensions", "minimum_positive_1d", "larger_signed")]
        if not selected:
            selected = cases[-3:]
        if valgrind is None:
            self.record("SKIP", "memory", "Valgrind", "Valgrind is unavailable; no Valgrind claim")
        elif self.c_ready:
            for case in selected:
                for goal in ("sym", "ddg", "norm"):
                    command = [valgrind, "--leak-check=full", "--show-leak-kinds=all", "--errors-for-leak-kinds=all",
                               "--track-origins=yes", "--error-exitcode=97", self.binary, goal, case["file"]]
                    observation = self.execute(command, max(120, self.args.timeout))
                    okay = observation["exit_code"] == 0 and re.search(r"ERROR SUMMARY:\s+0 errors", observation["stderr"])
                    self.record("PASS" if okay else "FAIL", "memory", "Valgrind " + case["name"] + ": " + goal,
                                "No leaks, invalid access, double free or uninitialized use",
                                command=command, dataset=case["file"], expected="exit 0 and 0 errors", actual=observation,
                                implementation="standalone C allocation/cleanup")
        self.allocation_tests()
        if valgrind and (self.folder / "allocation_harness").is_file():
            command = [valgrind, "--leak-check=full", "--show-leak-kinds=all", "--errors-for-leak-kinds=all",
                       "--track-origins=yes", "--error-exitcode=97", str(self.folder / "allocation_harness"),
                       "allocation_fixture.txt"]
            observed = self.execute(command, max(120, self.args.timeout))
            okay = observed["exit_code"] == 0 and re.search(r"ERROR SUMMARY:\s+0 errors", observed["stderr"])
            self.record("PASS" if okay else "FAIL", "memory", "Valgrind all C APIs and failure cleanup",
                        "Graph, SymNMF update and allocation failure paths have no Valgrind errors",
                        command=command, dataset="allocation_fixture.txt", expected="exit 0 and zero errors",
                        actual=observed, implementation="input.c, matrix.c, symnmf.c")
        if not self.args.sanitizers:
            self.record("SKIP", "memory", "AddressSanitizer", "Optional separate build disabled; enable --sanitizers")
            return
        self.sanitizer_tests(selected)

    def allocation_tests(self):
        case = dict(name="allocation_fixture", points=np.arange(40 * 64, dtype=float).reshape(40, 64) / 1000,
                    k=2, description="Long rows and point-capacity growth for allocation failures")
        self.write_dataset(case)
        source = self.folder / "_allocation_harness.c"
        source.write_text(ALLOCATION_HARNESS, encoding="utf-8")
        command = [self.environment["CC"]] + FLAGS + ["-DSYMNMF_EXTENSION", "-I.", source.name,
                   "symnmf.c", "matrix.c", "input.c", "-o", "allocation_harness", "-lm",
                   "-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free"]
        built = self.execute(command, 120)
        if built["exit_code"] != 0:
            self.record("SKIP", "memory", "allocation fault injection", "Linker wrapping is unavailable",
                        scope="diagnostic", actual=built)
            return
        command = [str(self.folder / "allocation_harness"), case["filename"]]
        observed = self.execute(command)
        try:
            value = json.loads(observed["stdout"])
            okay = observed["exit_code"] == 0 and not value["failed"] and value["checks"] > 0
        except (ValueError, KeyError, TypeError):
            value, okay = observed, False
        self.record("PASS" if okay else "FAIL", "memory", "allocation fault injection",
                    "Each injected malloc/calloc/realloc failure returns cleanly without tracked leaks",
                    command=command, dataset=case["filename"], expected="no outstanding allocations after every failure",
                    actual=value, implementation="input.c, matrix.c, symnmf.c")

    def sanitizer_tests(self, selected):
        flags = ["-g", "-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        probe = self.folder / "_sanitizer_probe.c"
        probe.write_text("int main(void) { return 0; }\n", encoding="ascii")
        built = self.execute([self.environment["CC"]] + FLAGS + flags + [probe.name, "-o", "sanitizer_probe"])
        environment = dict(self.environment, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
        if built["exit_code"] == 0:
            probes = [self.execute([str(self.folder / "sanitizer_probe")], environment=environment) for _ in range(3)]
        else:
            probes = []
        if not probes or any(result["exit_code"] != 0 for result in probes):
            self.record("SKIP", "memory", "AddressSanitizer availability",
                        "Compiler/runtime sanitizer probe unavailable or unstable", actual={"build": built, "probes": probes})
            return
        command = [self.environment["CC"]] + FLAGS + flags + ["symnmf.c", "matrix.c", "input.c", "-o", "symnmf_asan", "-lm"]
        built = self.execute(command, 120)
        if built["exit_code"] != 0:
            self.record("SKIP", "memory", "AddressSanitizer build", "Optional instrumented build unsupported", actual=built)
            return
        for case in selected:
            for goal in ("sym", "ddg", "norm"):
                command = [str(self.folder / "symnmf_asan"), goal, case["file"]]
                observation = self.execute(command, environment=environment)
                if observation["exit_code"] != 0:
                    # A bare DEADLYSIGNAL without a diagnostic is not a confirmed project bug.
                    actionable = re.search(r"ERROR: (AddressSanitizer|LeakSanitizer)|runtime error:", observation["stderr"])
                    self.record("FAIL" if actionable else "SKIP", "memory", "sanitizer " + case["name"] + ": " + goal,
                                "Instrumented execution reported a memory defect" if actionable else "Sanitizer runtime failed without a usable defect report",
                                command=command, dataset=case["file"], actual=observation,
                                expected="exit 0 with no sanitizer diagnostics", implementation="standalone C")
                else:
                    self.record("PASS", "memory", "sanitizer " + case["name"] + ": " + goal,
                                "Address/undefined sanitizers completed without diagnostics", command=command, dataset=case["file"])
        command = [self.environment["CC"]] + FLAGS + flags + ["-DSYMNMF_EXTENSION", "-I.",
                   "_allocation_harness.c", "symnmf.c", "matrix.c", "input.c", "-o", "allocation_harness_asan", "-lm",
                   "-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free"]
        built = self.execute(command, 120)
        if built["exit_code"] != 0:
            self.record("SKIP", "memory", "sanitizer all C APIs", "Optional wrapped sanitizer build unsupported", actual=built)
            return
        command = [str(self.folder / "allocation_harness_asan"), "allocation_fixture.txt"]
        observed = self.execute(command, environment=environment)
        actionable = re.search(r"ERROR: (AddressSanitizer|LeakSanitizer)|runtime error:", observed["stderr"])
        status = "PASS" if observed["exit_code"] == 0 else "FAIL" if actionable else "SKIP"
        self.record(status, "memory", "sanitizer all C APIs and failure cleanup",
                    "Instrumented graph/SymNMF updates and failure paths; uninformative runtime failures are skipped",
                    command=command, dataset="allocation_fixture.txt", expected="exit 0 and no sanitizer errors",
                    actual=observed, implementation="input.c, matrix.c, symnmf.c")

    def finish(self, incomplete=False):
        self.executor.shutdown(wait=True)
        counts = Counter(item["status"] for item in self.results)
        required = Counter(item["status"] for item in self.results if item["scope"] == "required")
        counts.update({"PASS": 0, "FAIL": 0, "SKIP": 0})
        required.update({"PASS": 0, "FAIL": 0, "SKIP": 0})
        groups = defaultdict(Counter)
        for item in self.results:
            groups[item["category"]][item["status"]] += 1
        for group in groups.values():
            group.update({"PASS": 0, "FAIL": 0, "SKIP": 0})
        report = {"total_tests": len(self.results), "counts": dict(counts), "required_counts": dict(required),
                  "categories": {key: dict(value) for key, value in groups.items()}, "incomplete": incomplete,
                  "elapsed_seconds": time.monotonic() - self.started, "project": str(self.project),
                  "artifacts": str(self.folder), "python": sys.version, "numpy": np.__version__,
                  "sklearn": importlib.import_module("sklearn").__version__,
                  "absolute_tolerance": ATOL, "epsilon": EPSILON, "max_iter": MAX_ITER,
                  "stress_seeds": self.args.seeds, "datasets": self.datasets, "results": self.results,
                  "assumptions": ["Existing, nonempty valid .txt files with distinct finite equal-dimension points",
                                  "HW1 K-means starts from the first k points; unspecified empty-cluster/tie policies are skipped",
                                  "Zero H entries remain zero; no denominator epsilon",
                                  "Zero-degree normalization and unsupported extreme floating-point operations are diagnosed",
                                  "Undefined silhouettes accept the required error response and remain SKIP",
                                  "Local factorize/fit helper names are observed when available, not required external APIs",
                                  "No invalid files, automatic installs or network access"]}
        destination = self.args.report or self.folder / "comprehensive_report.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
        print("\nTOTAL: {} tests | {} PASS | {} FAIL | {} SKIP".format(len(self.results), counts["PASS"], counts["FAIL"], counts["SKIP"]))
        for category in sorted(groups):
            group = groups[category]
            print("  {}: {} PASS, {} FAIL, {} SKIP".format(category, group["PASS"], group["FAIL"], group["SKIP"]))
        print("OVERALL: " + ("INCOMPLETE" if incomplete else "CONFIRMED REQUIRED FAILURES" if required["FAIL"] else "ALL EXECUTED REQUIRED CHECKS PASSED"))
        print("SKIP is not PASS. Report: " + str(destination))
        print("Retained inputs, requests, build artifacts and failure diagnostics: " + str(self.folder))
        return 2 if incomplete else 1 if required["FAIL"] else 0


def oracle_self_checks(tester):
    """Check hand-derived answers and tester rejection paths before project tests."""
    a, b = math.exp(-0.5), math.exp(-2)
    expected = np.array([[0, a, b], [a, 0, a], [b, a, 0]])
    points = np.array([[0.], [1.], [2.]])
    graph = reference_graph(points)
    assert np.allclose(graph["sym"], expected, rtol=0, atol=1e-15)
    assert np.allclose(graph["ddg"], np.diag([a+b, 2*a, a+b]), rtol=0, atol=1e-15)
    assert np.allclose(graph["norm"], scalar_graph(points, "norm"), rtol=0, atol=1e-15)
    for malformed in ("0.000,1.0000\n", "0.0000,1.0000\n\n", "debug\n0.0000,1.0000\n",
                      "nan,1.0000\n", "0.0000\n", "0.0000,1.0000,2.0000\n"):
        try:
            parse_matrix(malformed, (1, 2))
        except ValueError:
            continue
        raise AssertionError("Tester accepted malformed matrix output: " + repr(malformed))
    assert compare_array([[0.0001]], np.zeros((1, 1)))[0]
    assert not compare_array([[0.0002]], np.zeros((1, 1)))[0]
    h = np.array([[.1, .2], [.3, .4], [.5, .6]])
    result = reference_symnmf(graph["norm"], initial_h=h)
    assert compare_array(reference_h_gram(graph["norm"], h), result["h"])[0]
    zero = reference_symnmf(graph["norm"], initial_h=np.zeros_like(h))
    assert zero["iterations"] == 1 and np.array_equal(zero["h"], np.zeros_like(h))
    tester.record("PASS", "oracle self-check", "independent references and strict comparison",
                  "Hand-derived graph, two H arrangements, zero H, tolerance and malformed-output rejection", scope="tester")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--timeout", type=float, default=60, help="Per-process timeout in seconds")
    parser.add_argument("--jobs", type=int, default=2, help="Bounded subprocess concurrency (default 2)")
    parser.add_argument("--seeds", default="7,42,2026", help="Comma-separated deterministic stress seeds")
    parser.add_argument("--stress-cases", type=int, default=12, help="Total stress cases over all seeds")
    parser.add_argument("--sanitizers", action="store_true", help="Also try an isolated Address/UndefinedSanitizer build")
    parser.add_argument("--report", type=Path, help="JSON destination (default: retained temporary directory)")
    args = parser.parse_args()
    try:
        args.seeds = tuple(int(value) for value in args.seeds.split(","))
        if not args.seeds or any(not 0 <= value < 2**32 for value in args.seeds):
            raise ValueError("seeds must be in [0, 2**32)")
    except ValueError as error:
        parser.error(str(error))
    if args.timeout <= 0 or not 1 <= args.jobs <= 8 or args.stress_cases < 0:
        parser.error("timeout must be positive, jobs 1..8, stress-cases nonnegative")
    project = args.project.resolve()
    required = ("symnmf.py", "analysis.py", "kmeans.py", "setup.py", "Makefile", "symnmf.c",
                "matrix.c", "input.c", "symnmfmodule.c", "symnmf.h", "input.h")
    try:
        global np, silhouette_score, adjusted_rand_score
        np = importlib.import_module("numpy")
        metrics = importlib.import_module("sklearn.metrics")
        silhouette_score, adjusted_rand_score = metrics.silhouette_score, metrics.adjusted_rand_score
        importlib.import_module("setuptools")
        missing = [name for name in required if not (project / name).is_file()]
        if missing or not shutil.which("gcc") or not shutil.which("make"):
            raise RuntimeError("Missing prerequisites: " + ", ".join(missing or ["gcc and make required"]))
    except (ImportError, RuntimeError) as error:
        print("[SETUP ERROR] {}. No dependencies were installed.".format(error))
        return 2
    protected = [path for path in project.iterdir() if path.is_file() and
                 (path.suffix in (".py", ".c", ".h", ".pdf") or path.name == "Makefile")]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    if args.report and args.report.resolve() in {path.resolve() for path in protected}:
        parser.error("The report destination must not overwrite a source file or the existing tester")
    folder = Path(tempfile.mkdtemp(prefix="comprehensive_symnmf_"))
    for path in protected:
        if path.suffix in (".c", ".h", ".py") or path.name == "Makefile":
            if path.name not in ("tester_symnmf.py", "comprehensive_tester.py"):
                shutil.copy2(str(path), str(folder / path.name))
    shutil.copy2(str(Path(__file__).resolve()), str(folder / "_comprehensive_driver.py"))
    tester = ComprehensiveTester(project, folder, args)
    print("Project: {}\nTemporary copy: {}\nStress seeds: {}".format(project, folder, args.seeds), flush=True)
    incomplete = False
    try:
        oracle_self_checks(tester)
        tester.build()
        for case in make_datasets(stress_seeds=args.seeds, stress_cases=args.stress_cases):
            tester.run_dataset(case)
        if tester.python_ready or tester.c_ready:
            tester.arguments_and_formats()
        if tester.python_ready:
            tester.custom_h_tests()
        if tester.c_ready:
            tester.memory_tests()
    except KeyboardInterrupt:
        incomplete = True
        print("\nInterrupted; partial report and artifacts retained.", flush=True)
    except Exception as error:
        incomplete = True
        tester.record("SKIP", "tester error", type(error).__name__, str(error), scope="tester", traceback=traceback.format_exc())
    changed = [str(path) for path, digest in hashes.items()
               if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest]
    tester.record("FAIL" if changed else "PASS", "source protection", "original files",
                  "Implementation and existing tester hashes are unchanged", scope="tester", actual=changed)
    return tester.finish(incomplete or bool(changed))


# Independent numerical oracles. No expected result calls the project.
# np and the two sklearn metrics are imported by the standalone tester.
def reference_graph(points):
    """Calculate A, D and D^(-1/2) A D^(-1/2), per PDF sections 1.1-1.3."""
    points = np.asarray(points, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        difference = points[:, None, :] - points[None, :, :]
        similarity = np.exp(-np.sum(difference * difference, axis=2) / 2.0)
    np.fill_diagonal(similarity, 0.0)
    degree = np.sum(similarity, axis=1)
    reason = None
    normalized = None
    if not np.isfinite(degree).all():
        reason = "Non-finite degree after double-precision arithmetic"
    elif np.any(degree == 0.0):
        reason = "A degree is zero after floating-point exponential underflow; D^(-1/2) is undefined"
    else:
        # Dividing separately avoids underflow in a product of two tiny degrees.
        root = np.sqrt(degree)
        normalized = (similarity / root[:, None]) / root[None, :]
    return {"sym": similarity, "ddg": np.diag(degree), "norm": normalized,
            "degrees": degree, "reason": reason}


def reference_initial_h(w, k):
    """The legacy seeded uniform generator matches seed(1234); no global RNG mutation."""
    w = np.asarray(w, dtype=float)
    random = np.random.RandomState(1234)
    return random.uniform(0.0, 2.0 * np.sqrt(np.mean(w) / k), (len(w), k))


def reference_symnmf(w, initial_h=None, k=None, max_iter=300, epsilon=1e-4):
    """Original multiplicative rule and strictly squared-Frobenius stopping test."""
    w = np.asarray(w, dtype=float)
    if initial_h is None:
        if k is None:
            raise ValueError("The reference requires k or an initial H")
        h = reference_initial_h(w, k)
    else:
        h = np.array(initial_h, dtype=float, copy=True)
    changes = []
    zero_denominator = False
    for iteration in range(1, max_iter + 1):
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            numerator = w @ h
            # Literal PDF denominator, evaluated independently of C's Gram matrix.
            denominator = (h @ h.T) @ h
            positive = h != 0.0
            zero_denominator = zero_denominator or bool(np.any(denominator == 0.0))
            if np.any((denominator <= 0.0) & positive):
                return {"h": None, "iterations": iteration - 1, "changes": changes,
                        "stopped": "undefined", "zero_denominator": zero_denominator,
                        "reason": "A positive H entry has a zero denominator in double precision"}
            updated = np.zeros_like(h)
            # Multiplicative zeros remain zero, without evaluating undefined 0/0.
            updated[positive] = h[positive] * (0.5 + 0.5 * numerator[positive] / denominator[positive])
            change = float(np.sum((updated - h) ** 2))
        if not np.isfinite(updated).all() or not np.isfinite(change):
            return {"h": None, "iterations": iteration - 1, "changes": changes,
                    "stopped": "undefined", "zero_denominator": zero_denominator,
                    "reason": "The original SymNMF update exceeds double-precision arithmetic"}
        changes.append(change)
        h = updated
        if change < epsilon:
            return {"h": h, "iterations": iteration, "changes": changes,
                    "stopped": "converged", "zero_denominator": zero_denominator,
                    "reason": None}
    return {"h": h, "iterations": max_iter, "changes": changes,
            "stopped": "max_iter", "zero_denominator": zero_denominator,
            "reason": None}


def reference_kmeans(points, k, max_iter=300, epsilon=1e-4):
    """HW1 Lloyd iteration: first k points, first-min assignment, Euclidean stopping."""
    points = np.asarray(points, dtype=float)
    centers = points[:k].copy()
    empty_clusters = False
    tie = False
    near_tie = False
    for iteration in range(1, max_iter + 1):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            distance = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        if not np.isfinite(distance).all():
            return {"labels": None, "centers": centers, "iterations": iteration - 1,
                    "empty_clusters": empty_clusters, "tie": tie, "near_tie": near_tie,
                    "numerical_issue": True, "reason": "K-means squared distances overflow double precision"}
        labels = np.argmin(distance, axis=1)
        ordered = np.sort(distance, axis=1)
        gap = ordered[:, 1] - ordered[:, 0]
        tie = tie or bool(np.any(gap == 0.0))
        near_tie = near_tie or bool(np.any(gap <= 1e-12 * np.maximum(ordered[:, 1], 1e-100)))
        counts = np.bincount(labels, minlength=k)
        empty_clusters = empty_clusters or bool(np.any(counts == 0))
        # Keeping an empty center is just one common policy; callers skip score
        # comparisons when an empty cluster occurs because the PDF gives no policy.
        updated = centers.copy()
        for cluster in range(k):
            if counts[cluster]:
                updated[cluster] = np.mean(points[labels == cluster], axis=0)
        movement = np.linalg.norm(updated - centers, axis=1)
        centers = updated
        if np.all(movement < epsilon):
            break
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distance = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    if not np.isfinite(distance).all() or not np.isfinite(centers).all():
        return {"labels": None, "centers": centers, "iterations": iteration,
                "empty_clusters": empty_clusters, "tie": tie, "near_tie": near_tie,
                "numerical_issue": True, "reason": "Non-finite final K-means calculation"}
    # The last centroid move can create ties absent from earlier assignments.
    ordered = np.sort(distance, axis=1)
    gap = ordered[:, 1] - ordered[:, 0]
    tie = tie or bool(np.any(gap == 0.0))
    near_tie = near_tie or bool(np.any(gap <= 1e-12 * np.maximum(ordered[:, 1], 1e-100)))
    return {"labels": np.argmin(distance, axis=1), "centers": centers,
            "iterations": iteration, "empty_clusters": empty_clusters, "tie": tie,
            "near_tie": near_tie, "numerical_issue": False, "reason": None}


def reference_analysis(points, k, h):
    """Independently derive labels, both sklearn silhouettes, and sklearn ARI."""
    km = reference_kmeans(points, k)
    if h is None:
        return {"scores": None, "reason": "SymNMF reference is undefined", "kmeans": km,
                "nmf_labels": None, "kmeans_labels": km["labels"], "nmf_ambiguous": False}
    h = np.asarray(h, dtype=float)
    nmf_labels = np.argmax(h, axis=1)
    ordered = np.sort(h, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    ambiguous = bool(np.any(margin <= 1e-10))
    result = {"scores": None, "reason": km["reason"], "kmeans": km,
              "nmf_labels": nmf_labels, "kmeans_labels": km["labels"],
              "nmf_ambiguous": ambiguous, "nmf_min_margin": float(np.min(margin))}
    if km["numerical_issue"]:
        return result
    for label, values in (("SymNMF", nmf_labels), ("K-means", km["labels"])):
        if not 2 <= len(np.unique(values)) < len(points):
            result["reason"] = label + " has undefined Silhouette (need 2 through N-1 nonempty clusters)"
            return result
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            scores = np.array([silhouette_score(points, nmf_labels),
                               silhouette_score(points, km["labels"]),
                               adjusted_rand_score(nmf_labels, km["labels"])])
    except (ValueError, OverflowError, FloatingPointError) as error:
        result["reason"] = "Sklearn score is numerically undefined: " + str(error)
        return result
    if not np.isfinite(scores).all():
        result["reason"] = "Sklearn score is non-finite in double precision"
        return result
    result["scores"] = scores
    result["reason"] = None
    return result


# Historical tester configurations are retained verbatim; curated cases extend
# them rather than silently replacing their size/k/dimension combinations.
ORIGINAL_CASES = [
    (2, "sym", 5, 2), (2, "ddg", 5, 3), (2, "norm", 7, 4),
    (2, "symnmf", 6, 2), (2, "sym", 5, 1), (2, "ddg", 5, 1),
    (2, "norm", 7, 1), (2, "symnmf", 6, 1), (2, "symnmf", 5, 3),
    (3, "ddg", 6, 4), (2, "norm", 4, 2), (2, "symnmf", 5, 3),
    (4, "sym", 10, 3), (5, "ddg", 12, 5), (3, "norm", 8, 4),
    (3, "symnmf", 7, 3), (6, "sym", 15, 5), (4, "ddg", 10, 2),
    (3, "norm", 6, 3), (2, "symnmf", 4, 2), (5, "sym", 11, 4),
    (3, "ddg", 7, 3), (4, "norm", 9, 4), (6, "symnmf", 14, 5),
    (2, "sym", 5, 2), (3, "ddg", 6, 3), (4, "norm", 8, 4),
    (5, "symnmf", 10, 5), (7, "sym", 20, 6), (5, "ddg", 15, 5),
    (4, "norm", 10, 4), (3, "symnmf", 8, 3), (8, "sym", 25, 7),
    (6, "ddg", 18, 6), (5, "norm", 12, 5), (4, "symnmf", 9, 4),
    (9, "sym", 30, 8), (7, "ddg", 21, 7), (6, "norm", 14, 6),
    (5, "symnmf", 11, 5), (10, "sym", 35, 9), (8, "ddg", 24, 8),
    (7, "norm", 16, 7), (6, "symnmf", 13, 6),
]


def make_datasets(stress_seeds=(7, 42, 2026), stress_cases=12):
    """Reproducible finite, distinct, rectangular inputs; no network or project calls."""
    datasets = []

    def add(name, points, k, description, tags=(), scope="required", **metadata):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 1 or not 1 < k < len(points):
            raise ValueError("Invalid tester-generated shape/k: " + name)
        if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Tester generated non-finite or duplicate points: " + name)
        record = {"name": name, "points": points, "k": k,
                  "description": description, "scope": scope, "tags": list(tags)}
        record.update(metadata)
        datasets.append(record)

    original_random = np.random.RandomState(2026)
    for index, (k, goal, n, dimension) in enumerate(ORIGINAL_CASES, 1):
        add("original_{:02d}".format(index), original_random.uniform(0, 1, (n, dimension)),
            k, "Historical configuration {}: N={}, d={}, k={}, goal={}".format(index, n, dimension, k, goal),
            ("historical",), original_goal=goal)

    random = np.random.RandomState(1667)
    add("minimum_positive_1d", [[0.1], [0.4], [1.3]], 2,
        "Minimum valid N=3, dimension one, k=N-1", ("minimum_n", "1d", "k_n_minus_1"))
    add("minimum_signed_2d", [[-1, 0], [0, 0], [1, 0]], 2,
        "Minimum signed line with exact zero coordinates", ("minimum_n", "signed", "zeros", "symmetric"))
    add("signed_1d", np.arange(-1.5, 1.51, 0.5)[:, None], 2,
        "Signed one-dimensional arithmetic progression", ("1d", "signed", "line"))
    add("all_negative", -random.uniform(0.02, 1.9, (17, 3)), 4,
        "Strictly negative coordinates in three dimensions", ("negative", "3d"))
    add("positive_three_dimensions", random.uniform(0.01, 2.0, (25, 3)), 5,
        "Positive three-dimensional points and moderate k/N", ("positive", "3d"))
    zero_axes = np.vstack((np.zeros((1, 4)), np.eye(4), -np.eye(4)))
    add("zeros_and_axis_points", zero_axes, 3,
        "Axis-aligned vectors with zero entries and origin", ("zeros", "signed", "symmetric"))
    add("tiny_distinct_1d", np.arange(9, dtype=float)[:, None] * 1e-12, 3,
        "Distinct tiny values with similarity rounding to one", ("tiny", "near_identical", "1d"))
    add("small_signed_3d", random.normal(0, 1e-8, (15, 3)), 4,
        "Mixed signed coordinates close to zero", ("tiny", "signed", "3d"))
    add("near_identical_offset", 0.125 + random.normal(0, 1e-10, (12, 2)), 3,
        "Nearly identical distinct points with a common nonzero offset", ("near_identical", "decimals"))
    add("many_decimal_digits", [[0.123456789012345, -0.987654321098765],
                               [0.123456789112345, -0.987654320998765],
                               [0.333333333333333, 0.142857142857143],
                               [-0.271828182845904, 0.314159265358979],
                               [1.414213562373095, -1.732050807568877],
                               [0.693147180559945, 0.577215664901533]], 2,
        "Many significant decimal digits and two coordinates differing by 1e-10", ("decimals", "near_identical"))
    # exp(-d^2/2) lies just above/below rounding boundaries 0.12345 and 0.54325.
    distances = np.sqrt(-2 * np.log(np.array([0.1234499, 0.1234501, 0.5432499, 0.5432501])))
    add("four_decimal_boundaries", np.r_[0.0, distances][:, None], 2,
        "Similarity values straddling four-place rounding boundaries", ("rounding", "1d", "decimals"))
    add("large_common_offset", 1e6 + random.normal(0, 0.7, (21, 3)), 4,
        "Large finite coordinates with ordinary pairwise distances", ("large", "3d"))
    add("high_dimension_64", random.normal(0, 0.08, (22, 64)), 5,
        "High-dimensional vectors with well-scaled nonzero similarities", ("high_dimension", "dimension_gt_n"))
    add("high_dimension_128", random.normal(0, 0.05, (14, 128)), 4,
        "Dimension 128 exceeds number of points", ("high_dimension", "dimension_gt_n"))
    add("long_line_1024_dimensions", random.normal(0, 0.015, (9, 1024)), 3,
        "Dynamic line reading with 1024 dimensions and long input rows", ("high_dimension", "dimension_gt_n", "long_lines"))
    add("line_uneven_spacing", np.array([-2.5, -1.7, -1.69, -0.3, 0.1, 0.31, 0.9, 2.0, 2.5])[:, None], 3,
        "Uneven spacing along a signed line", ("line", "1d", "signed"))
    angle = np.arange(20) * 2.0 * np.pi / 20
    add("circle", np.column_stack((np.cos(angle), np.sin(angle))), 4,
        "Symmetric circle with equal angular spacing", ("circle", "symmetric", "ties"))
    add("uniform_square", random.uniform(-2, 2, (36, 2)), 6,
        "Uniform mixed-sign points with overlapping clusters", ("uniform", "overlap", "signed"))
    for k, dimension in ((2, 2), (3, 3), (4, 5)):
        centers = random.normal(0, 1.7, (k, dimension))
        points = np.vstack([random.normal(center, 0.25, (12, dimension)) for center in centers])
        add("separated_k{}".format(k), points, k,
            "Well-separated Gaussian groups in d={}".format(dimension), ("separated", "gaussian"))
    centers = np.array([[-0.2, 0.0], [0.2, 0.0], [0.0, 0.25]])
    add("overlapping_gaussians", np.vstack([random.normal(c, 0.65, (14, 2)) for c in centers]), 3,
        "Strongly overlapping Gaussian groups", ("overlap", "gaussian"))
    add("highly_unbalanced", np.vstack((random.normal([0.0, 0.0], 0.12, (45, 2)),
                                       random.normal([2.6, 0.1], 0.1, (4, 2)),
                                       random.normal([-1.8, 1.2], 0.08, (2, 2)))), 3,
        "Highly unbalanced 45/4/2-point groups", ("unbalanced", "separated"))
    add("ordinary_outliers", np.vstack((random.normal(0, 0.2, (24, 2)),
                                      [[3.0, 3.5], [-4.0, 2.0], [0.0, -4.2]])), 3,
        "A dense cluster plus isolated outliers with nonzero degrees", ("outliers", "unbalanced"))
    add("k_near_n", random.normal(0, 0.8, (13, 4)), 12,
        "Almost one requested cluster per point", ("k_n_minus_1", "high_k_ratio"))
    add("large_k_ratio", random.normal(0, 0.9, (45, 3)), 25,
        "More than half as many requested clusters as points", ("high_k_ratio", "3d"))
    add("tiny_similarity", np.arange(7, dtype=float)[:, None] * 12.0, 3,
        "Extremely small positive similarities, mathematically defined normalization", ("tiny_similarity", "near_zero_degree", "1d"))
    add("near_zero_degrees", np.arange(5, dtype=float)[:, None] * 37.0, 2,
        "Degrees around 1e-298, positive and above the subnormal range", ("tiny_similarity", "near_zero_degree", "1d"))
    add("subnormal_degrees", np.arange(5, dtype=float)[:, None] * 38.5, 2,
        "Subnormal nonzero degree tests: normalization remains mathematically defined", ("tiny_similarity", "near_zero_degree", "subnormal", "1d"))
    add("separated_blocks", np.vstack((random.normal([0.0, 0.0], 0.08, (8, 2)),
                                      random.normal([1000.0, 0.0], 0.08, (8, 2)))), 2,
        "Between-group weights underflow to zero while all degrees stay positive", ("separated", "zero_weights"))
    add("larger_signed", random.normal(0, 1.0, (70, 6)), 5,
        "Historical larger signed coverage, N=70", ("larger", "signed"))
    add("larger_180", random.normal(0, 0.75, (180, 3)), 9,
        "Practical larger case for repeated allocation growth and runtime", ("larger", "allocation_growth"))
    threshold_random = np.random.RandomState(95)
    add("convergence_threshold", threshold_random.normal(0, 0.1, (12, 3)), 7,
        "Converges at squared change 9.9918626918e-5, just below the strict threshold",
        ("convergence", "near_threshold"))
    slow_random = np.random.RandomState(177)
    add("slow_convergence", slow_random.normal(0, 8.0, (90, 2)), 6,
        "Requires about 186 original-formula updates with standard seeded initialization",
        ("convergence", "many_iterations"))
    empty_random = np.random.RandomState(821)
    add("empty_kmeans_cluster", np.vstack((empty_random.normal(0, 0.05, (5, 2)),
                                          empty_random.normal(0, 2.0, (25, 2)))), 5,
        "An empty Lloyd cluster occurs; PDF leaves empty-cluster policy unspecified",
        ("empty_kmeans", "unbalanced"))
    add("zero_degree_underflow", np.array([[0.0], [50.0], [100.0], [150.0]]), 2,
        "All similarities underflow; normalization is undefined, not a normal accuracy requirement",
        ("underflow", "zero_degree", "1d"), scope="numerical_limit")
    add("huge_finite_coordinates", [[1e200, 1e200], [-1e200, 1e200],
                                    [1e200, -1e200], [-1e200, -1e200]], 2,
        "Finite coordinates whose squared distances overflow double precision",
        ("huge", "overflow", "zero_degree"), scope="numerical_limit")

    if stress_cases and not stress_seeds:
        raise ValueError("At least one seed is required for randomized stress tests")
    streams = [np.random.RandomState(seed) for seed in stress_seeds]
    for index in range(stress_cases):
        stream_index = index % len(streams)
        rng = streams[stream_index]
        n = (11, 18, 29, 40, 56, 82, 120, 150)[index % 8]
        dimension = (1, 2, 3, 7, 16, 64)[index % 6]
        k = (2, min(3, n - 1), n - 1, min(7, n - 1))[index % 4]
        style = index % 4
        scale = (0.015, 0.25, 0.7, 1.5)[index % 4]
        if style == 0:
            points = rng.uniform(-scale, scale, (n, dimension))
            description = "Uniform stress with tiny/ordinary scales"
        elif style == 1:
            centers = rng.normal(0, 1.7 / np.sqrt(dimension), (k, dimension))
            labels = np.arange(n) % k
            points = centers[labels] + rng.normal(0, scale / np.sqrt(dimension), (n, dimension))
            description = "Separated random groups with deterministic label order"
        elif style == 2:
            points = rng.normal(0, scale / np.sqrt(dimension), (n, dimension))
            description = "Signed normal stress with k=N-1"
        else:
            points = rng.normal(0, scale / np.sqrt(dimension), (n, dimension))
            points[-1] += 2.5 / np.sqrt(dimension)
            description = "Overlapping random groups with an outlier"
        add("stress_s{}_case{:02d}".format(stress_seeds[stream_index], index + 1), points, k,
            description + "; seed={}, N={}, d={}, k={}".format(stress_seeds[stream_index], n, dimension, k),
            ("stress", "seed_{}".format(stress_seeds[stream_index])), stress_seed=stress_seeds[stream_index])
    return datasets


def _worker_json_value(value):
    """Encode observations without changing the separately recorded return types."""
    if hasattr(value, "tolist"):
        return _worker_json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _worker_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_worker_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"unexpected_type": type(value).__name__, "representation": repr(value)}


def _worker_matrix_metadata(matrix):
    """Keep structural evidence that JSON's automatic list conversion would lose."""
    metadata = {"outer_type": type(matrix).__name__}
    try:
        rows = list(matrix)
        lengths = [len(row) for row in rows]
        metadata["row_types"] = sorted({type(row).__name__ for row in rows})
        metadata["element_types"] = sorted(
            {type(item).__name__ for row in rows for item in row}
        )
        metadata["shape"] = [len(rows), lengths[0] if lengths else 0]
        metadata["row_lengths"] = lengths
        metadata["rectangular"] = len(set(lengths)) <= 1
        metadata["all_python_float"] = all(
            type(item) is float for row in rows for item in row
        )
    except (TypeError, ValueError) as error:
        metadata["structure_error"] = str(error)
    return metadata


def _worker_same(first, second):
    """Compare repeated observations exactly, including accidental non-list outputs."""
    import json

    return json.dumps(_worker_json_value(first), sort_keys=True) == json.dumps(
        _worker_json_value(second), sort_keys=True
    )


def _worker_extension(request):
    """Observe the current extension with caller-supplied points, H, and W."""
    import copy
    import traceback
    import symnmfmodule

    count = max(2, int(request.get("repeat", 3)))
    result = {
        "module_file": symnmfmodule.__file__,
        "matrices": {},
        "metadata": {},
        "input_unchanged": {},
        "repeat_equal": {},
        "errors": {},
        "repeat_count": count,
    }
    names = ("symnmf",) if request["kind"] == "symnmf" else ("sym", "ddg", "norm")
    if request["kind"] == "extension" and "h" in request and "w" in request:
        names += ("symnmf",)
    names = tuple(request.get("goals", names))
    for name in names:
        if name == "symnmf":
            arguments = [copy.deepcopy(request["h"]), copy.deepcopy(request["w"])]
        else:
            arguments = [copy.deepcopy(request["points"])]
        before = copy.deepcopy(arguments)
        first = None
        unchanged = True
        equal = True
        try:
            for index in range(count):
                actual = getattr(symnmfmodule, name)(*arguments)
                unchanged = unchanged and _worker_same(arguments, before)
                if index == 0:
                    first = copy.deepcopy(actual)
                    result["metadata"][name] = _worker_matrix_metadata(actual)
                else:
                    equal = equal and _worker_same(actual, first)
        except Exception as error:
            result["errors"][name] = {
                "error": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            result["metadata"].pop(name, None)
            continue
        result["matrices"][name] = _worker_json_value(first)
        result["input_unchanged"][name] = unchanged
        result["repeat_equal"][name] = equal
    return result


def _worker_factorize(request):
    """Record actual initialization calls without using them to form expectations."""
    import copy
    import numpy as np
    import symnmf
    import symnmfmodule

    if not callable(getattr(symnmf, "factorize", None)):
        return {"optional_unavailable": "symnmf.factorize"}
    points = copy.deepcopy(request["points"])
    before = copy.deepcopy(points)
    original_seed = np.random.seed
    original_uniform = np.random.uniform
    seed_calls, uniform_calls, events = [], [], []

    def observed_seed(*args, **kwargs):
        seed_calls.append(_worker_json_value(args[0] if args else kwargs))
        events.append("seed")
        return original_seed(*args, **kwargs)

    def observed_uniform(*args, **kwargs):
        actual = original_uniform(*args, **kwargs)
        uniform_calls.append({
            "args": _worker_json_value(args),
            "kwargs": _worker_json_value(kwargs),
            "result": _worker_json_value(actual),
        })
        events.append("uniform")
        return actual

    try:
        np.random.seed = observed_seed
        np.random.uniform = observed_uniform
        original_seed(7529)
        original_uniform(size=19)
        first = symnmf.factorize(points, request["k"])
        unchanged_first = _worker_same(points, before)
        original_seed(19003)
        original_uniform(size=37)
        second = symnmf.factorize(points, request["k"])
    finally:
        np.random.seed = original_seed
        np.random.uniform = original_uniform
    return {
        "first": _worker_json_value(first),
        "second": _worker_json_value(second),
        "repeat_equal": _worker_same(first, second),
        "points_unchanged": unchanged_first and _worker_same(points, before),
        "seed_calls": seed_calls,
        "uniform_calls": uniform_calls,
        "events": events,
        "metadata": _worker_matrix_metadata(first),
        "module_file": symnmf.__file__,
        "extension_file": symnmfmodule.__file__,
    }


def _worker_kmeans(request):
    """Observe the project's original HW1 centroid and assignment routines."""
    import copy
    import kmeans

    if not all(callable(getattr(kmeans, name, None)) for name in ("fit", "closest_centroid")):
        return {"optional_unavailable": "kmeans.fit / closest_centroid"}
    points = copy.deepcopy(request["points"])
    before = copy.deepcopy(points)
    first = kmeans.fit(points, request["k"])
    first_labels = [kmeans.closest_centroid(point, first) for point in points]
    unchanged_first = _worker_same(points, before)
    second = kmeans.fit(points, request["k"])
    second_labels = [kmeans.closest_centroid(point, second) for point in points]
    return {
        "centroids": _worker_json_value(first),
        "labels": _worker_json_value(first_labels),
        "repeat_equal": _worker_same(first, second) and first_labels == second_labels,
        "input_unchanged": unchanged_first and _worker_same(points, before),
        "metadata": _worker_matrix_metadata(first),
        "module_file": kmeans.__file__,
    }


def run_worker(request_path):
    """Run project code only in an isolated subprocess and emit one JSON observation."""
    import json
    import os
    import sys
    import traceback

    sys.path.insert(0, os.getcwd())
    try:
        with open(request_path, "r", encoding="utf-8") as handle:
            request = json.load(handle)
        if request["kind"] in ("extension", "extension_graph", "symnmf"):
            result = _worker_extension(request)
        elif request["kind"] == "factorize":
            result = _worker_factorize(request)
        elif request["kind"] == "kmeans":
            result = _worker_kmeans(request)
        else:
            raise ValueError("Unknown worker kind: " + str(request["kind"]))
        print(json.dumps(result, sort_keys=True, allow_nan=True))
    except Exception as error:
        print(json.dumps({
            "error": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }, sort_keys=True))
    return 0


ALLOCATION_HARNESS = r'''
#include <stdio.h>
#include <stdlib.h>
#include "symnmf.h"
#include "input.h"

#define LIMIT 4096
void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void *__real_realloc(void *pointer, size_t size);
void __real_free(void *pointer);
static void *owned[LIMIT];
static int enabled, calls, fail_at, invalid_frees, overflow;
static const char *names[] = {"read_points", "sym", "ddg", "norm", "symnmf"};
static int baselines[5], checks, failures;
typedef struct {
    int api, allocation, null_result, leaks, bad_frees, overflow;
} Failure;
static Failure failed[LIMIT];

static int find_slot(void *pointer)
{
    int i;
    for (i = 0; i < LIMIT; i++) {
        if (owned[i] == pointer) return i;
    }
    return -1;
}

static void remember(void *pointer)
{
    int index;
    if (pointer == NULL) return;
    index = find_slot(NULL);
    if (index < 0) overflow = 1;
    else owned[index] = pointer;
}

static int inject(void)
{
    if (!enabled) return 0;
    calls++;
    return fail_at > 0 && calls == fail_at;
}

void *__wrap_malloc(size_t size)
{
    void *result;
    if (inject()) return NULL;
    result = __real_malloc(size);
    if (enabled) remember(result);
    return result;
}

void *__wrap_calloc(size_t count, size_t size)
{
    void *result;
    if (inject()) return NULL;
    result = __real_calloc(count, size);
    if (enabled) remember(result);
    return result;
}

void *__wrap_realloc(void *pointer, size_t size)
{
    void *result;
    int index;
    if (inject()) return NULL;
    index = pointer == NULL ? -1 : find_slot(pointer);
    result = __real_realloc(pointer, size);
    if (enabled && (result != NULL || size == 0)) {
        if (index >= 0) owned[index] = NULL;
        remember(result);
    }
    return result;
}

void __wrap_free(void *pointer)
{
    int index;
    if (pointer == NULL) return;
    if (enabled) {
        index = find_slot(pointer);
        if (index < 0) {
            invalid_frees++;
            return;
        }
        owned[index] = NULL;
    }
    __real_free(pointer);
}

static double **invoke(int api, const char *path, int *rows)
{
    static double point_data[4][2] = {{0,0}, {1,0}, {0,2}, {3,1}};
    static double h_data[4][2] = {{.1,.2}, {.2,.3}, {.3,.1}, {.4,.2}};
    static double w_data[4][4] = {
        {0,.3,.3,.3}, {.3,0,.3,.3}, {.3,.3,0,.3}, {.3,.3,.3,0}
    };
    double *points[4], *h[4], *w[4];
    int i, dim;
    *rows = 4;
    for (i = 0; i < 4; i++) {
        points[i] = point_data[i];
        h[i] = h_data[i];
        w[i] = w_data[i];
    }
    if (api == 0) return read_points(path, rows, &dim);
    if (api == 1) return sym(points, 4, 2);
    if (api == 2) return ddg(points, 4, 2);
    if (api == 3) return norm(points, 4, 2);
    return symnmf(h, w, 4, 2);
}

static int one_check(int api, int allocation, const char *path)
{
    double **result;
    int rows, i, leaked, null_result, count;
    calls = invalid_frees = overflow = 0;
    fail_at = allocation;
    enabled = 1;
    result = invoke(api, path, &rows);
    null_result = result == NULL;
    if (result != NULL) free_matrix(result, rows);
    enabled = 0;
    count = calls;
    leaked = 0;
    for (i = 0; i < LIMIT; i++) {
        if (owned[i] != NULL) {
            leaked++;
            __real_free(owned[i]);
            owned[i] = NULL;
        }
    }
    checks++;
    if ((allocation == 0 ? null_result : !null_result) ||
        leaked || invalid_frees || overflow) {
        if (failures < LIMIT) {
            failed[failures].api = api;
            failed[failures].allocation = allocation;
            failed[failures].null_result = null_result;
            failed[failures].leaks = leaked;
            failed[failures].bad_frees = invalid_frees;
            failed[failures].overflow = overflow;
            failures++;
        }
    }
    return count;
}

int main(int argc, char **argv)
{
    int api, position, i;
    Failure *failure;
    if (argc != 2) {
        puts("{\"checks\":0,\"baseline_calls\":{},\"failed\":[\"fixture path required\"]}");
        return 1;
    }
    for (api = 0; api < 5; api++) {
        baselines[api] = one_check(api, 0, argv[1]);
        for (position = 1; position <= baselines[api]; position++) {
            one_check(api, position, argv[1]);
        }
    }
    printf("{\"checks\":%d,\"baseline_calls\":{", checks);
    for (api = 0; api < 5; api++) {
        printf("%s\"%s\":%d", api ? "," : "", names[api], baselines[api]);
    }
    printf("},\"failed\":[");
    for (i = 0; i < failures; i++) {
        failure = &failed[i];
        printf("%s{\"api\":\"%s\",\"fail_at\":%d,\"returned_null\":%s,",
               i ? "," : "", names[failure->api], failure->allocation,
               failure->null_result ? "true" : "false");
        printf("\"leaks\":%d,\"invalid_frees\":%d,\"tracker_overflow\":%s}",
               failure->leaks, failure->bad_frees,
               failure->overflow ? "true" : "false");
    }
    puts("]}");
    return failures ? 1 : 0;
}
'''

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.exit(run_worker(sys.argv[2]))
    sys.exit(main())
