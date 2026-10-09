"""Compare SymNMF and the HW1 K-means implementation."""

import sys
import numpy as np
import kmeans
from symnmf import ClusterError, factorize, parse_k, read_data

np.random.seed(1234)


def compare_clusterings(points, k):
    """Return both silhouette scores and their adjusted Rand index."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    h = factorize(points, k)
    nmf_labels = np.argmax(h, axis=1)
    centroids = kmeans.fit(points, k)
    kmeans_labels = kmeans.predict(points, centroids)
    nmf_score = silhouette_score(points, nmf_labels)
    kmeans_score = silhouette_score(points, kmeans_labels)
    ari = adjusted_rand_score(nmf_labels, kmeans_labels)
    return nmf_score, kmeans_score, ari


def main():
    """Read k and a filename, then print the three required scores."""
    try:
        if len(sys.argv) != 3:
            raise ValueError("Incorrect argument count")
        k = parse_k(sys.argv[1])
        points = read_data(sys.argv[2])
        k = parse_k(sys.argv[1], len(points))
        nmf_score, kmeans_score, ari = compare_clusterings(points, k)
        print("nmf: %.4f" % nmf_score)
        print("kmeans: %.4f" % kmeans_score)
        print("ari: %.4f" % ari)
        return 0
    except ClusterError:
        print("Incorrect number of clusters!")
    except Exception:
        print("An Error Has Occurred")
    return 1


if __name__ == "__main__":
    sys.exit(main())
