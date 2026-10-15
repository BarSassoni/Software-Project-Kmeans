import sys
import numpy as np


def isNumValid(text):
    # check if num is valid
    # input: text; output: integer or None
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
    # input: text, n; output: k or None
    k = isNumValid(text)
    if k is None or k <= 1 or (n is not None and k >= n):
        return None
    return k


def read_points(lines):
    # read comma-separated vectors
    # input: lines; output: list of points
    points = []
    for line in lines:
        point = []
        for value in line.split(","):
            point.append(float(value))
        points.append(point)
    return points


def read_data(file_name):
    # load vectors from file
    # input: file_name; output: list of points
    with open(file_name, "r") as inputFile:
        return read_points(inputFile)


def factorize(points, k):
    # create H and optimize using C module
    # input: points, k; output: optimized H matrix
    import symnmfmodule

    w = symnmfmodule.norm(points)
    np.random.seed(1234)
    initialH = np.random.uniform(
        0, 2 * np.sqrt(np.mean(w) / k), (len(points), k)
    )
    return symnmfmodule.symnmf(initialH.tolist(), w)


def print_matrix(matrix):
    # print the matrix
    # input: matrix; output: prints matrix
    for row in matrix:
        print(",".join("%.4f" % value for value in row))


def main():
    # get input and print the matrix
    # input: command arguments; output: exit status
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
        import symnmfmodule
        if goal == "symnmf":
            result = factorize(points, k)
        elif goal == "sym":
            result = symnmfmodule.sym(points)
        elif goal == "ddg":
            result = symnmfmodule.ddg(points)
        elif goal == "norm":
            result = symnmfmodule.norm(points)
        print_matrix(result)
        return 0
    except Exception:
        print("An Error Has Occurred")
    return 1


if __name__ == "__main__":
    sys.exit(main())
