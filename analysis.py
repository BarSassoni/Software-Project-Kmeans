import sys
import numpy as np
import kmeans
from symnmf import factorize, parse_k, read_data


def main():
    # get k and filename, print required scores
    # input: command arguments; output: exit status
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
        from sklearn.metrics import adjusted_rand_score, silhouette_score

        h = factorize(points, k)
        nmfLabels = np.argmax(h, axis=1)
        centroids = kmeans.fit(points, k)
        kmeansLabels = [kmeans.closest_centroid(point, centroids) for point in points]
        nmfScore = silhouette_score(points, nmfLabels)
        kmeansScore = silhouette_score(points, kmeansLabels)
        ari = adjusted_rand_score(nmfLabels, kmeansLabels)
        print("nmf: %.4f" % nmfScore)
        print("kmeans: %.4f" % kmeansScore)
        print("ari: %.4f" % ari)
        return 0
    except Exception:
        print("An Error Has Occurred")
    return 1


if __name__ == "__main__":
    sys.exit(main())
