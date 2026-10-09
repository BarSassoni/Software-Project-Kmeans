"""HW1 K-means, refactored for reuse with the project's convergence limits."""

import sys

epsilon = 0.0001
iters = 300


def squared_distance(point, centroid):
    """Compute squared Euclidean distance as in HW1."""
    distance = 0.0
    for j in range(len(point)):
        distance += (point[j] - centroid[j]) ** 2
    return distance


def closest_centroid(point, centroids):
    """Return the first closest centroid, preserving the HW1 tie rule."""
    minidx = 0
    min_dist = squared_distance(point, centroids[0])
    for i in range(1, len(centroids)):
        distance = squared_distance(point, centroids[i])
        if distance < min_dist:
            min_dist = distance
            minidx = i
    return minidx


def predict(points, centroids):
    """Assign each vector to its nearest final centroid."""
    return [closest_centroid(point, centroids) for point in points]


def update_centroids(points, centroids):
    """Average each cluster; keep the old centroid for an empty cluster."""
    clusters = [[] for _ in centroids]
    for point in points:
        clusters[closest_centroid(point, centroids)].append(point)
    new_centroids = []
    for i in range(len(centroids)):
        cluster = clusters[i]
        if len(cluster) == 0:
            new_centroids.append(centroids[i][:])
            continue
        new_centroid = []
        for j in range(len(centroids[i])):
            total = 0.0
            for point in cluster:
                total += point[j]
            new_centroid.append(total / len(cluster))
        new_centroids.append(new_centroid)
    return new_centroids


def fit(points, k, max_iter=iters, eps=epsilon):
    """Run HW1 K-means, starting from the first k input vectors."""
    centroids = [points[i][:] for i in range(k)]
    for _ in range(max_iter):
        new_centroids = update_centroids(points, centroids)
        converged = True
        for i in range(k):
            if squared_distance(new_centroids[i], centroids[i]) >= eps ** 2:
                converged = False
        centroids = new_centroids
        if converged:
            break
    return centroids


def main():
    """Keep the HW1 stdin interface, with project defaults for convergence."""
    from symnmf import ClusterError, isNumValid, parse_k
    from symnmf import print_matrix, read_points

    try:
        if len(sys.argv) not in (2, 3):
            raise ValueError("Incorrect argument count")
        k = parse_k(sys.argv[1])
        max_iter = iters
        if len(sys.argv) == 3:
            max_iter = isNumValid(sys.argv[2])
            if max_iter is None or max_iter <= 1 or max_iter >= 800:
                print("Incorrect maximum iteration!")
                return 1
        points = read_points(sys.stdin)
        k = parse_k(sys.argv[1], len(points))
        print_matrix(fit(points, k, max_iter))
        return 0
    except ClusterError:
        print("Incorrect number of clusters!")
    except Exception:
        print("An Error Has Occurred")
    return 1


if __name__ == "__main__":
    sys.exit(main())
