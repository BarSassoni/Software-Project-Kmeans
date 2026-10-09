import math
import sys
import numpy as np

np.random.seed(1234)


def isNumValid(text):
    # check if num is valid
    if text.startswith("+"):
        text = text[1:]
    whole, _, fraction = text.partition(".")
    if not whole:
        return None
    for digit in whole:
        if digit < "0" or digit > "9":
            return None
    for digit in fraction:
        if digit != "0":
            return None
    try:
        return int(whole)
    except ValueError:
        return None


def parse_k(text, n=None):
    # check if k is valid (as in HW1)
    k = isNumValid(text)
    if k is None or k <= 1 or (n is not None and k >= n):
        return None
    return k


def read_points(lines):
    # read vectors, ignore the empty lines
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
    # load vectors from file
    with open(file_name, "r") as inputFile:
        return read_points(inputFile)


def factorize(points, k):
    # create H and optimize using C module
    import symnmfmodule

    w = symnmfmodule.norm(points)
    initialH = np.random.uniform(
        0, 2 * np.sqrt(np.mean(w) / k), (len(points), k)
    )
    return symnmfmodule.symnmf(initialH.tolist(), w)


def calculate_matrix(points, k, goal):
    # send goal to matching C method
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
    # Print one row per line using the required precision.
    for row in matrix:
        print(",".join("%.4f" % value for value in row))


def main():
    # Validate command-line input and print the requested matrix.
    try:
        if len(sys.argv) != 4:
            print("An Error Has Occurred")
            return 1
        k = parse_k(sys.argv[1])
        if k is None:
            print("Incorrect number of clusters!")
            return 1
        goal = sys.argv[2]
        if goal not in ("symnmf", "sym", "ddg", "norm"):
            print("An Error Has Occurred")
            return 1
        points = read_data(sys.argv[3])
        k = parse_k(sys.argv[1], len(points))
        if k is None:
            print("Incorrect number of clusters!")
            return 1
        print_matrix(calculate_matrix(points, k, goal))
        return 0
    except Exception:
        print("An Error Has Occurred")
    return 1


if __name__ == "__main__":
    sys.exit(main())
