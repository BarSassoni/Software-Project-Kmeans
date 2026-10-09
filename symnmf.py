"""Python interface for the SymNMF C implementation."""

import math
import sys
import numpy as np

np.random.seed(1234)


class ClusterError(ValueError):
    """An invalid number of clusters was supplied."""


def isNumValid(s):
    """Accept the integer notation used in HW1, including +2 and 2.000."""
    if s.startswith("+"):
        s = s[1:]
    whole, dot, fraction = s.partition(".")
    if not whole or any(ch < "0" or ch > "9" for ch in whole):
        return None
    if dot and any(ch != "0" for ch in fraction):
        return None
    try:
        return int(whole)
    except ValueError:
        return None


def parse_k(text, n=None):
    """Check the cluster count with the same bounds as HW1."""
    k = isNumValid(text)
    if k is None or k <= 1 or (n is not None and k >= n):
        raise ClusterError("Incorrect number of clusters!")
    return k


def read_points(lines):
    """Read comma-separated, finite vectors, ignoring empty lines."""
    points = []
    dimension = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        point = [float(value) for value in line.split(",")]
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Non-finite vector element")
        if dimension is None:
            dimension = len(point)
        elif len(point) != dimension:
            raise ValueError("Inconsistent vector dimension")
        points.append(point)
    if not points:
        raise ValueError("Empty input")
    return points


def read_data(file_name):
    """Load the input vectors from a text file."""
    with open(file_name, "r") as input_file:
        return read_points(input_file)


def factorize(points, k):
    """Initialize H in Python and optimize it using the C extension."""
    import symnmfmodule

    w = symnmfmodule.norm(points)
    initial_h = np.random.uniform(
        0, 2 * np.sqrt(np.mean(w) / k), (len(points), k)
    )
    return symnmfmodule.symnmf(initial_h.tolist(), w)


def calculate_matrix(points, k, goal):
    """Send each goal to the corresponding C extension method."""
    import symnmfmodule

    if goal == "symnmf":
        return factorize(points, k)
    if goal == "sym":
        return symnmfmodule.sym(points)
    if goal == "ddg":
        return symnmfmodule.ddg(points)
    if goal == "norm":
        return symnmfmodule.norm(points)
    raise ValueError("Unknown goal")


def print_matrix(matrix):
    """Print one row per line using the required precision."""
    for row in matrix:
        print(",".join("%.4f" % value for value in row))


def main():
    """Validate command-line input and print the requested matrix."""
    try:
        if len(sys.argv) != 4:
            raise ValueError("Incorrect argument count")
        k = parse_k(sys.argv[1])
        goal = sys.argv[2]
        if goal not in ("symnmf", "sym", "ddg", "norm"):
            raise ValueError("Unknown goal")
        points = read_data(sys.argv[3])
        k = parse_k(sys.argv[1], len(points))
        print_matrix(calculate_matrix(points, k, goal))
        return 0
    except ClusterError:
        print("Incorrect number of clusters!")
    except Exception:
        print("An Error Has Occurred")
    return 1


if __name__ == "__main__":
    sys.exit(main())
