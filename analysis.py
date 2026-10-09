# Compare SymNMF and the HW1 K-means implementation.

import sys
import numpy as np
import kmeans
from symnmf import factorize, parse_k, read_data

np.random.seed(1234)


def compare_clusterings(points, k):
    # Return both silhouette scores and their adjusted Rand index.
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    h = factorize(points, k)
    nmfLabels = np.argmax(h, axis=1)
    centroids = kmeans.fit(points, k)
    kmeansLabels = kmeans.predict(points, centroids)
    nmfScore = silhouette_score(points, nmfLabels)
    kmeansScore = silhouette_score(points, kmeansLabels)
    ari = adjusted_rand_score(nmfLabels, kmeansLabels)
    return nmfScore, kmeansScore, ari


def main():
    # Read k and a filename, then print the three required scores.
    try:
        if len(sys.argv) != 3:
            print("An Error Has Occurred")
            return 1
        k = parse_k(sys.argv[1])
        if k is None:
            print("Incorrect number of clusters!")
            return 1
        points = read_data(sys.argv[2])
        k = parse_k(sys.argv[1], len(points))
        if k is None:
            print("Incorrect number of clusters!")
            return 1
        nmfScore, kmeansScore, ari = compare_clusterings(points, k)
        print("nmf: %.4f" % nmfScore)
        print("kmeans: %.4f" % kmeansScore)
        print("ari: %.4f" % ari)
        return 0
    except Exception:
        print("An Error Has Occurred")
    return 1


if __name__ == "__main__":
    sys.exit(main())
