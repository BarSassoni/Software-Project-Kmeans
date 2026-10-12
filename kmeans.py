# HW1 Kmeans code with minor changes (adjusted)

import sys

epsilon = 0.0001
iters = 300


def squared_distance(point, centroid):
    # calculate squared euclidean distance
    distance = 0.0
    for j in range(len(point)):
        distance += (point[j] - centroid[j]) ** 2
    return distance


def closest_centroid(point, centroids):
    # return the first closest centroid
    closestIndex = 0
    closestDistance = squared_distance(point, centroids[0])
    for i in range(1, len(centroids)):
        distance = squared_distance(point, centroids[i])
        if distance < closestDistance:
            closestDistance = distance
            closestIndex = i
    return closestIndex


def predict(points, centroids):
    # match vector to final centroid
    return [closest_centroid(point, centroids) for point in points]


def update_centroids(points, centroids):
    # avg cluster, and keep old centroid for empty cluster
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
    # running HW1 Kmeans, starting from the first k input vectors
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
    # keep HW1 stdin interface, with project defaults for convergence
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
