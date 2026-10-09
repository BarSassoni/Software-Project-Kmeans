#!/usr/bin/env python3
# Standalone tester for this SymNMF project. Run: python3 tester_symnmf.py
#
# Adapted test design and 44 case configurations from avivjan/TAU_Testers:
# https://github.com/avivjan/TAU_Testers/blob/main/SoftwareProject/compareTester.sh
# Upstream blob inspected: d7d57424e748bbbeca46a6bfe8953303a301bf14
#
# Rewritten Python tester, not the original shell script. Independent NumPy
# references replace Prev_final_100. Uses epsilon=1e-4, max_iter=300, seed=1234,
# and checks both Silhouette scores plus ARI. Expected answers never use the
# project's own functions. No make, sudo, network access, or automatic installs.
# Builds a source-only temporary copy; original files are not changed/deleted.
# Requires Linux/macOS, gcc, Python development headers, setuptools, numpy,
# and scikit-learn. Valgrind checks standalone C if available; otherwise SKIP.
# Undefined Silhouette cases are SKIP, never a false PASS.
# Exit: 0=all executed checks passed, 1=test failure, 2=setup/tester error.
# For local use only; do not include this file in the assignment submission.
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

EPSILON = 1e-4
MAX_ITER = 300
ATOL = 5.0001e-5  # Half one printed decimal unit, plus floating-point slack.
ERROR = "An Error Has Occurred"
NUMBER = r"-?\d+\.\d{4}"
REQUIRED = ("symnmf.py", "analysis.py", "kmeans.py", "setup.py", "symnmf.c",
            "symnmf.h", "symnmfmodule.c", "matrix.c", "input.c", "input.h")

# (k, goal, number_of_points, dimensions), in the original tester's order.
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


def reference_matrices(points):
    delta = points[:, None, :] - points[None, :, :]
    similarity = np.exp(-np.sum(delta * delta, axis=2) / 2.0)
    np.fill_diagonal(similarity, 0.0)
    degrees = np.sum(similarity, axis=1)
    inverse = np.zeros(len(points))
    np.divide(1.0, np.sqrt(degrees), out=inverse, where=degrees > 0)
    normalized = (similarity * inverse[:, None]) * inverse[None, :]
    return {"sym": similarity, "ddg": np.diag(degrees), "norm": normalized}


def reference_h(normalized, k):
    random = np.random.RandomState(1234)
    h = random.uniform(0, 2 * np.sqrt(normalized.mean() / k),
                       size=(len(normalized), k))
    for _ in range(MAX_ITER):
        numerator = normalized @ h
        denominator = (h @ h.T) @ h
        ratio = np.zeros_like(h)
        np.divide(numerator, denominator, out=ratio, where=denominator > 0)
        updated = h * (0.5 + 0.5 * ratio)
        change = np.sum((updated - h) ** 2)
        h = updated
        if change < EPSILON:
            break
    return h


def reference_kmeans(points, k):
    centers = points[:k].copy()
    for _ in range(MAX_ITER):
        distances = np.sum((points[:, None] - centers[None, :]) ** 2, axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(k):
            if np.any(labels == cluster):
                updated[cluster] = points[labels == cluster].mean(axis=0)
        converged = np.all(np.linalg.norm(updated - centers, axis=1) < EPSILON)
        centers = updated
        if converged:
            break
    distances = np.sum((points[:, None] - centers[None, :]) ** 2, axis=2)
    return distances.argmin(axis=1)


def reference_scores(points, k, h):
    nmf_labels = h.argmax(axis=1)
    km_labels = reference_kmeans(points, k)
    for labels in (nmf_labels, km_labels):
        if not 2 <= len(np.unique(labels)) < len(points):
            return None
    return np.array([silhouette_score(points, nmf_labels),
                     silhouette_score(points, km_labels),
                     adjusted_rand_score(nmf_labels, km_labels)])


class Tester:
    def __init__(self, folder, timeout):
        self.folder = folder
        self.timeout = timeout
        self.results = []
        self.environment = os.environ.copy()
        self.environment.pop("PYTHONPATH", None)
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            self.environment[key] = "1"
        self.environment["LC_ALL"] = "C"
        self.c_program = str(folder / "symnmf")
        self.memory_inputs = []

    def record(self, status, name, detail=""):
        self.results.append({"status": status, "name": name, "detail": detail})
        print("[{}] {}{}".format(status, name, ": " + detail if detail else ""),
              flush=True)

    def execute(self, command, timeout=None):
        try:
            result = subprocess.run(command, cwd=str(self.folder),
                                    env=self.environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=timeout or self.timeout)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout: {} seconds".format(timeout or self.timeout)
        except OSError as error:
            return -1, "", str(error)

    def failure(self, name, reason, command, stdout="", stderr="", expected=None):
        log = self.folder / ("failure_{:03d}.txt".format(len(self.results) + 1))
        content = "Reason: {}\nCommand: {}\n\nSTDOUT:\n{}\nSTDERR:\n{}\n".format(
            reason, repr(command), stdout, stderr)
        if expected is not None:
            content += "\nExpected:\n" + str(expected) + "\n"
        log.write_text(content, encoding="utf-8")
        self.record("FAIL", name, reason + " (" + log.name + ")")

    def build(self):
        gcc = shutil.which("gcc")
        command = [gcc, "-ansi", "-Wall", "-Wextra", "-Werror",
                   "-pedantic-errors", "symnmf.c", "matrix.c", "input.c",
                   "-o", "symnmf", "-lm"]
        self.environment["CC"] = gcc
        if sys.platform.startswith("linux"):
            self.environment["LDSHARED"] = gcc + " -shared"
        commands = [("Build standalone C (no Makefile)", command),
                    ("Build Python extension", [sys.executable, "setup.py",
                                                "build_ext", "--inplace", "--force"])]
        for name, command in commands:
            code, stdout, stderr = self.execute(command, max(120, self.timeout))
            if code != 0 or re.search(r"\bwarning:", stdout + stderr, re.I):
                self.failure(name, "Build failed or emitted compiler warnings",
                             command, stdout, stderr)
                return False
            self.record("PASS", name)
        return True

    def matrix_test(self, name, command, expected):
        code, stdout, stderr = self.execute(command)
        if code != 0 or stderr:
            self.failure(name, "Unsuccessful process or stderr output", command,
                         stdout, stderr)
            return
        lines = stdout.splitlines()
        pattern = re.compile(NUMBER + r"(?:," + NUMBER + r")*")
        if len(lines) != len(expected) or any(not pattern.fullmatch(x) for x in lines):
            self.failure(name, "Wrong shape/format; need CSV with 4 decimal places",
                         command, stdout, stderr)
            return
        rows = [line.split(",") for line in lines]
        if any(len(row) != expected.shape[1] for row in rows):
            self.failure(name, "Incorrect number of columns", command, stdout)
            return
        actual = np.array(rows, dtype=float)
        if not np.isfinite(actual).all() or not np.allclose(actual, expected,
                                                          rtol=0, atol=ATOL):
            difference = float(np.max(np.abs(actual - expected)))
            formatted = "\n".join(",".join("%.4f" % x for x in row) for row in expected)
            self.failure(name, "Numeric mismatch; max difference={:.8g}".format(difference),
                         command, stdout, stderr, formatted)
            return
        self.record("PASS", name)

    def analysis_test(self, name, filename, points, k, h):
        expected = reference_scores(points, k, h)
        if expected is None:
            self.record("SKIP", name, "Reference clustering has undefined Silhouette")
            return
        command = [sys.executable, "analysis.py", str(k), filename]
        code, stdout, stderr = self.execute(command)
        pattern = r"nmf: (" + NUMBER + r")\nkmeans: (" + NUMBER + r")\nari: (" + NUMBER + r")\n?"
        match = re.fullmatch(pattern, stdout)
        if code != 0 or stderr or match is None:
            self.failure(name, "Need successful output with nmf, kmeans AND ari",
                         command, stdout, stderr)
            return
        actual = np.array([float(x) for x in match.groups()])
        if not np.allclose(actual, expected, rtol=0, atol=ATOL):
            formatted = "nmf: %.4f\nkmeans: %.4f\nari: %.4f" % tuple(expected)
            self.failure(name, "Silhouette/ARI mismatch", command, stdout, stderr,
                         formatted)
            return
        self.record("PASS", name)

    def dataset(self, name, points, k, goals):
        filename = name + ".txt"
        np.savetxt(str(self.folder / filename), points, delimiter=",", fmt="%.17g")
        matrices = reference_matrices(points)
        matrices["symnmf"] = reference_h(matrices["norm"], k)
        for goal in goals:
            self.matrix_test(name + ": Python " + goal,
                             [sys.executable, "symnmf.py", str(k), goal, filename],
                             matrices[goal])
            if goal != "symnmf":
                self.matrix_test(name + ": C " + goal,
                                 [self.c_program, goal, filename], matrices[goal])
                self.memory_inputs.append((goal, filename))
        self.analysis_test(name + ": analysis", filename, points, k, matrices["symnmf"])

    def expected_error(self, name, command, message=ERROR):
        code, stdout, stderr = self.execute(command)
        if code <= 0 or stdout.strip() != message or stderr:
            self.failure(name, "Need error message and nonzero (non-crash) exit",
                         command, stdout, stderr, message)
        else:
            self.record("PASS", name)

    def errors_and_formats(self):
        points = np.array([[0., 0.], [.1, 0.], [0., .1],
                           [3., 3.], [3.1, 3.], [3., 3.1]])
        filename = "format_input.txt"
        path = self.folder / filename
        normal = "\n".join(",".join(str(x) for x in row) for row in points)
        variants = [("no-final-newline", normal),
                    ("CRLF", normal.replace("\n", "\r\n") + "\r\n"),
                    ("blank-lines", "\n" + normal.replace("\n", "\n\n") + "\n\n"),
                    ("spaces", normal.replace(",", " , ") + "\n")]
        expected = reference_matrices(points)["norm"]
        for name, content in variants:
            path.write_bytes(content.encode("ascii"))
            for label, command in (("C", [self.c_program, "norm", filename]),
                                   ("Python", [sys.executable, "symnmf.py", "2", "norm", filename])):
                self.matrix_test("CSV " + name + ": " + label, command, expected)
        for goal in ("sym", "ddg", "norm", "symnmf"):
            self.expected_error("Missing file: Python " + goal,
                                [sys.executable, "symnmf.py", "2", goal, "absent.txt"])
            if goal != "symnmf":
                self.expected_error("Missing file: C " + goal,
                                    [self.c_program, goal, "absent.txt"])
        self.expected_error("Missing file: analysis",
                            [sys.executable, "analysis.py", "2", "absent.txt"])
        for k in ("0", "1", "6", "7", "-2", "2.5", "hello"):
            self.expected_error("Invalid k=" + k,
                                [sys.executable, "symnmf.py", k, "sym", filename],
                                "Incorrect number of clusters!")
        for name, command in (("C arguments", [self.c_program]),
                              ("Python arguments", [sys.executable, "symnmf.py"]),
                              ("Analysis arguments", [sys.executable, "analysis.py"]),
                              ("C unknown goal", [self.c_program, "oops", filename]),
                              ("Python unknown goal", [sys.executable, "symnmf.py", "2", "oops", filename])):
            self.expected_error(name, command)
        bad_files = ["", "1,2\n3\n", "1,\n2,3\n", "nan,1\n2,3\n", "inf,1\n2,3\n",
                     "hello,1\n2,3\n", "1,,2\n3,4,5\n", "1,2junk\n3,4\n"]
        for index, content in enumerate(bad_files, 1):
            name = "bad_input_{:02d}.txt".format(index)
            (self.folder / name).write_text(content)
            self.expected_error("Malformed CSV {}: C".format(index), [self.c_program, "sym", name])
            self.expected_error("Malformed CSV {}: Python".format(index),
                                [sys.executable, "symnmf.py", "2", "sym", name])

    def memory_tests(self):
        valgrind = shutil.which("valgrind")
        if valgrind is None:
            self.record("SKIP", "Valgrind (standalone C)", "Not installed; no memory-check claim")
            return
        targets = self.memory_inputs + [("sym", "absent.txt"), ("norm", "bad_input_02.txt")]
        for goal, filename in targets:
            command = [valgrind, "--leak-check=full", "--show-leak-kinds=all",
                       "--errors-for-leak-kinds=all", "--error-exitcode=97",
                       self.c_program, goal, filename]
            code, stdout, stderr = self.execute(command, max(120, self.timeout))
            expected_exit = 1 if filename in ("absent.txt", "bad_input_02.txt") else 0
            if code != expected_exit or not re.search(r"ERROR SUMMARY:\s+0 errors", stderr):
                self.failure("Valgrind " + goal + " " + filename,
                             "Memory check failed or could not run", command, stdout, stderr)
            else:
                self.record("PASS", "Valgrind " + goal + " " + filename)

    def summary(self):
        counts = {status: sum(r["status"] == status for r in self.results)
                  for status in ("PASS", "FAIL", "SKIP")}
        print("\nRESULT: {PASS} PASS | {FAIL} FAIL | {SKIP} SKIP".format(**counts))
        if counts["FAIL"]:
            print("FAILED: inspect the failure logs below.")
        else:
            print("ALL EXECUTED CHECKS PASSED. SKIP means not checked, not passed.")
        report = {"counts": counts, "python": sys.version, "numpy": np.__version__,
                  "results": self.results}
        (self.folder / "tester_results.json").write_text(json.dumps(report, indent=2))
        return counts["FAIL"] == 0


def check_requirements(project):
    global np, silhouette_score, adjusted_rand_score
    missing = [name for name in REQUIRED if not (project / name).is_file()]
    if missing:
        raise RuntimeError("Missing project files: " + ", ".join(missing))
    if os.name == "nt":
        raise RuntimeError("Use the Linux/course terminal in VS Code (not Windows PowerShell).")
    if shutil.which("gcc") is None:
        raise RuntimeError("gcc is not available in this terminal.")
    try:
        np = importlib.import_module("numpy")
        metrics = importlib.import_module("sklearn.metrics")
        importlib.import_module("setuptools")
    except ImportError as error:
        raise RuntimeError("Missing dependency: {}. Use the course Python environment.".format(error))
    silhouette_score = metrics.silhouette_score
    adjusted_rand_score = metrics.adjusted_rand_score


def run_suite(tester, seed):
    random = np.random.RandomState(seed)
    for index, (k, goal, n, dimension) in enumerate(ORIGINAL_CASES, 1):
        # Deterministic full-precision points avoid duplicate 1D points.
        points = random.uniform(0, 1, size=(n, dimension))
        tester.dataset("original_{:02d}".format(index), points, k, [goal])
    all_goals = ["sym", "ddg", "norm", "symnmf"]
    for k, dimension in ((2, 2), (3, 3), (4, 5)):
        centers = random.normal(size=(k, dimension)) * 1.7
        points = np.vstack([random.normal(loc=center, scale=.25, size=(12, dimension))
                            for center in centers])
        tester.dataset("separated_k{}".format(k), points, k, all_goals)
    points = random.normal(size=(70, 6))
    tester.dataset("larger_signed", points, 5, all_goals)
    tester.errors_and_formats()
    tester.memory_tests()


def main():
    parser = argparse.ArgumentParser(description="Standalone tester for this SymNMF project.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder containing your source files (default: folder of this tester)")
    parser.add_argument("--seed", type=int, default=2026, help="Reproducible test-data seed")
    parser.add_argument("--timeout", type=int, default=60, help="Seconds allowed per tested command")
    parser.add_argument("--keep", action="store_true", help="Keep temporary inputs/logs even when checks pass")
    args = parser.parse_args()
    if args.timeout <= 0 or not 0 <= args.seed <= 2**32 - 1:
        parser.error("timeout must be positive; seed must be between 0 and 2**32-1")
    project = args.project.resolve()
    try:
        check_requirements(project)
    except RuntimeError as error:
        print("[SETUP ERROR] " + str(error), flush=True)
        return 2
    folder = Path(tempfile.mkdtemp(prefix="symnmf_tester_"))
    # Never copy stale binaries, .so, build folders or Makefile.
    for source in project.iterdir():
        if source.is_file() and source.suffix in (".c", ".h", ".py"):
            if source.resolve() != Path(__file__).resolve():
                shutil.copy2(str(source), str(folder / source.name))
    tester = Tester(folder, args.timeout)
    print("Source project: " + str(project), flush=True)
    print("Building a temporary copy, without Makefile. Test-data seed: " + str(args.seed), flush=True)
    result = 2
    try:
        if tester.build():
            run_suite(tester, args.seed)
            result = 0 if tester.summary() else 1
    except KeyboardInterrupt:
        print("\nInterrupted. Temporary files retained.")
    except Exception as error:
        print("[TESTER ERROR] {}: {}".format(type(error).__name__, error))
    if result == 0 and not args.keep:
        shutil.rmtree(str(folder))
    else:
        print("Inputs, build files and logs retained at: " + str(folder))
    return result


if __name__ == "__main__":
    sys.exit(main())
