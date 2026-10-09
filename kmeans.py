# HW1 K-means, refactored for reuse with the project's convergence limits.

import sys

epsilon = 0.0001
iters = 300


def squared_distance(point, centroid):
    # Compute squared Euclidean distance as in HW1.
    distance = 0.0
    for j in range(len(point)):
        distance += (point[j] - centroid[j]) ** 2
    return distance


def closest_centroid(point, centroids):
    # Return the first closest centroid, preserving the HW1 tie rule.
    closestIndex = 0
    closestDistance = squared_distance(point, centroids[0])
    for i in range(1, len(centroids)):
        distance = squared_distance(point, centroids[i])
        if distance < closestDistance:
            closestDistance = distance
            closestIndex = i
    return closestIndex


def predict(points, centroids):
    # Assign each vector to its nearest final centroid.
    return [closest_centroid(point, centroids) for point in points]


def update_centroids(points, centroids):
    # Average each cluster; keep the old centroid for an empty cluster.
    clusters = [[] for _ in centroids]
    for point in points:
        clusterIndex = closest_centroid(point, centroids)
        clusters[clusterIndex].append(point)
    newCentroids = []
    for i, cluster in enumerate(clusters):
        if not cluster:
            newCentroids.append(centroids[i][:])
            continue
        newCentroid = []
        for j in range(len(centroids[i])):
            total = 0.0
            for point in cluster:
                total += point[j]
            newCentroid.append(total / len(cluster))
        newCentroids.append(newCentroid)
    return newCentroids


def fit(points, k, max_iter=iters, eps=epsilon):
    # Run HW1 K-means, starting from the first k input vectors.
    centroids = [points[i][:] for i in range(k)]
    for _ in range(max_iter):
        newCentroids = update_centroids(points, centroids)
        converged = True
        for i in range(k):
            if squared_distance(newCentroids[i], centroids[i]) >= eps ** 2:
                converged = False
        centroids = newCentroids
        if converged:
            break
    return centroids


def main():
    # Keep the HW1 stdin interface, with project defaults for convergence.
    from symnmf import isNumValid, parse_k
    from symnmf import print_matrix, read_points

    try:
        if len(sys.argv) not in (2, 3):
            print("An Error Has Occurred")
            return 1
        k = parse_k(sys.argv[1])
        if k is None:
            print("Incorrect number of clusters!")
            return 1
        maxIter = iters
        if len(sys.argv) == 3:
            maxIter = isNumValid(sys.argv[2])
            if maxIter is None or maxIter <= 1 or maxIter >= 800:
                print("Incorrect maximum iteration!")
                return 1
        points = read_points(sys.stdin)
        k = parse_k(sys.argv[1], len(points))
        if k is None:
            print("Incorrect number of clusters!")
            return 1
        print_matrix(fit(points, k, maxIter))
        return 0
    except Exception:
        print("An Error Has Occurred")
    return 1


if __name__ == "__main__":
    sys.exit(main())
