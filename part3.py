"""Part 3 - Segmentation and dimensionality reduction (25 marks).

All 800 rows and the five continuous predictors X1..X5:

* k-means from scratch with the pinned starting centroids,
* the elbow rule,
* a three-segment profile built from the clusters,
* agglomerative clustering under single, complete and average linkage,
* PCA by the seven pinned steps.

Run from the repository root:  python part3.py
Writes:  outputs/part3_results.txt
         outputs/part3_segments.csv
         outputs/part3_hierarchical.csv
"""

from __future__ import annotations

import numpy as np

from common import (
    Results,
    banner,
    load_records,
    mean,
    sd,
    summarise_results,
    write_csv,
)

K_VALUES = (2, 3, 4, 5, 6)
SEGMENT_K = 3
ELBOW_CANDIDATES = (3, 4, 5)

# The supplied dissimilarity table for Task 3.4 (E1..E6, symmetric, zero diagonal).
COHORTS = ["E1", "E2", "E3", "E4", "E5", "E6"]
DISSIMILARITY = np.array(
    [
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ],
    dtype=float,
)


# ======================================================================================
# K-means
# ======================================================================================
def kmeans(Z: np.ndarray, k: int):
    """K-means with the pinned seeding, tie-break and stopping rule.

    Starting centroids are the standardised rows at 0, floor(n/k), 2*floor(n/k), ...
    A row joins the nearest centroid by squared Euclidean distance (ties to the
    lower-numbered centroid).  The loop stops at the first assignment pass that
    changes nothing.  Returns the final labels, the centroids and the WCSS.
    """
    n = Z.shape[0]
    step = n // k
    centroids = Z[[i * step for i in range(k)]].copy()

    assignments = np.full(n, -1, dtype=int)
    passes = 0

    while True:
        distances = np.sum((Z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(distances, axis=1)  # argmin keeps the first (lowest) index on ties
        passes += 1

        if np.array_equal(new_assignments, assignments):
            break

        assignments = new_assignments
        for cluster in range(k):
            members = assignments == cluster
            if np.any(members):
                centroids[cluster] = Z[members].mean(axis=0)
            # a centroid with no members stays where it is

    wcss = float(np.sum((Z - centroids[assignments]) ** 2))
    return assignments, centroids, wcss, passes


# ======================================================================================
# Agglomerative clustering
# ======================================================================================
def linkage_distance(a: np.ndarray, b: np.ndarray, rule: str) -> float:
    """Linkage distance between two clusters given as member index arrays."""
    block = DISSIMILARITY[np.ix_(a, b)]
    if rule == "single":
        return float(np.min(block))
    if rule == "complete":
        return float(np.max(block))
    if rule == "average":
        return float(np.mean(block))
    raise ValueError(rule)


def agglomerate(rule: str):
    """Run five merges under one linkage rule.

    Returns the merge log as ``(merge_no, cluster_a, cluster_b, height)`` with
    cluster labels written as ascending member names joined by ``+``.
    """
    clusters: list[list[int]] = [[i] for i in range(len(COHORTS))]
    log = []

    for merge_no in range(1, len(COHORTS)):
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                height = linkage_distance(
                    np.asarray(clusters[i]), np.asarray(clusters[j]), rule
                )
                if best is None or height < best[0]:
                    best = (height, i, j)

        assert best is not None
        height, i, j = best

        first, second = clusters[i], clusters[j]
        # cluster_a is the one whose smallest member number is the smaller one.
        if min(first) <= min(second):
            left, right = first, second
        else:
            left, right = second, first

        label_a = "+".join(COHORTS[m] for m in left)
        label_b = "+".join(COHORTS[m] for m in right)
        log.append((merge_no, label_a, label_b, height))

        merged = sorted(first + second)
        clusters = [c for index, c in enumerate(clusters) if index not in (i, j)]
        clusters.append(merged)

    return log


def clusters_at_cut(log) -> tuple[int, list[list[str]]]:
    """Cut the dendrogram in the largest gap between consecutive merge heights.

    The number of clusters is ``6 - (number of merges below the cut)``.
    """
    heights = [row[3] for row in log]
    gaps = [heights[i + 1] - heights[i] for i in range(len(heights) - 1)]
    largest = int(np.argmax(gaps))
    merges_below = largest + 1  # gap i sits between merge i and merge i+2 (1-based)
    n_clusters = len(COHORTS) - merges_below

    # Rebuild the memberships that exist after `merges_below` merges.
    clusters: list[list[int]] = [[i] for i in range(len(COHORTS))]
    for merge_no, label_a, label_b, _height in log[:merges_below]:
        members_a = [COHORTS.index(name) for name in label_a.split("+")]
        members_b = [COHORTS.index(name) for name in label_b.split("+")]
        merged = sorted(set(members_a) | set(members_b))
        clusters = [
            c
            for c in clusters
            if not set(c).issubset(merged)
        ]
        clusters.append(merged)

    named = sorted(
        ("+".join(COHORTS[m] for m in sorted(c)) for c in clusters),
        key=lambda name: COHORTS.index(name.split("+")[0]),
    )
    return n_clusters, named


# ======================================================================================
# PCA
# ======================================================================================
def pca(Z: np.ndarray):
    """The seven pinned PCA steps, with the pinned eigenvector sign convention.

    Step 1: the mean vector (zero after standardising).
    Step 2: centre the data.
    Step 3: the covariance matrix S = 1/(n-1) * Z^T Z.
    Step 4: eigen-decomposition with ``numpy.linalg.eigh`` (the one allowed
            library routine here).
    Step 5: sort by eigenvalue, largest first.
    Step 6: keep the top two eigenvectors.
    Step 7: project Z = X @ W.
    """
    n = Z.shape[0]

    mean_vector = Z.mean(axis=0)
    centred = Z - mean_vector
    covariance = (centred.T @ centred) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Sign convention: flip each eigenvector so its largest-magnitude entry is positive.
    for column in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            eigenvectors[:, column] = -vector

    components = eigenvectors[:, :2]
    scores = centred @ components

    return eigenvalues, eigenvectors, components, scores, mean_vector


# ======================================================================================
# Main
# ======================================================================================
def main() -> None:
    ids, X, y, value = load_records()
    n = X.shape[0]

    X5 = X[:, 0:5]
    mu = X5.mean(axis=0)
    sigma = X5.std(axis=0, ddof=1)
    Z = (X5 - mu) / sigma

    banner("PART 3 - Segmentation and dimensionality reduction")

    results = Results()

    # ---------------------------------------------------------------- 3.1 K-means
    fits = {}
    for k in K_VALUES:
        assignments, centroids, wcss, passes = kmeans(Z, k)
        fits[k] = {"labels": assignments, "centroids": centroids, "wcss": wcss, "passes": passes}
        results.add(f"wcss_k{k}", wcss)
        print(f"  k={k}: WCSS {wcss:.4f} after {passes} assignment passes")

    # ---------------------------------------------------------------- 3.2 Elbow
    def drop(k: int) -> float:
        return fits[k]["wcss"] - fits[k + 1]["wcss"]

    elbow_k = None
    for k in ELBOW_CANDIDATES:
        if drop(k) < 0.5 * drop(k - 1):
            elbow_k = k
            break
    if elbow_k is None:
        elbow_k = ELBOW_CANDIDATES[-1]
    results.add("elbow_k", elbow_k)
    print(
        "  successive WCSS drops: "
        + ", ".join(f"{k}->{k + 1}: {drop(k):.4f}" for k in range(2, 6))
    )
    print(f"  elbow rule selects K = {elbow_k}; the Part continues with K = {SEGMENT_K}")

    # ---------------------------------------------------------------- 3.3 Segments
    segment_fit = fits[SEGMENT_K]
    labels = segment_fit["labels"]

    # Order clusters by ascending mean of X3 in original units.
    means_x3 = [mean(X5[labels == c, 2]) for c in range(SEGMENT_K)]
    order = np.argsort(means_x3, kind="stable")
    name_of_cluster = {int(cluster): "ABC"[position] for position, cluster in enumerate(order)}

    segment_of_row = np.array([name_of_cluster[int(c)] for c in labels])
    sizes = {name: int(np.sum(segment_of_row == name)) for name in "ABC"}

    for name in "ABC":
        results.add(f"size_segment_{name}", sizes[name])

    segment_rows = []
    for name in "ABC":
        mask = segment_of_row == name
        segment_rows.append(
            [
                name,
                int(np.sum(mask)),
                mean(X5[mask, 0]),
                mean(X5[mask, 1]),
                mean(X5[mask, 2]),
                mean(X5[mask, 3]),
                mean(X5[mask, 4]),
                float(np.mean(y[mask])),
                mean(value[mask]),
            ]
        )
    write_csv(
        "part3_segments.csv",
        [
            "segment",
            "size",
            "mean_X1",
            "mean_X2",
            "mean_X3",
            "mean_X4",
            "mean_X5",
            "outcome_rate",
            "mean_value",
        ],
        segment_rows,
    )

    print(
        "  segments: "
        + ", ".join(
            f"{row[0]} n={row[1]} X3={row[4]:.3f} churn={row[7]:.3f}" for row in segment_rows
        )
    )

    # ---------------------------------------------------------------- 3.4 Hierarchical
    hierarchical_rows = []
    cut_clusters = {}
    for rule in ("single", "complete", "average"):
        log = agglomerate(rule)
        for merge_no, label_a, label_b, height in log:
            hierarchical_rows.append([rule, merge_no, label_a, label_b, height])
        n_clusters, members = clusters_at_cut(log)
        cut_clusters[rule] = (n_clusters, members)
        results.add(f"{rule}_clusters_at_cut", n_clusters)
        print(f"  {rule:<8} cut -> {n_clusters} clusters: {', '.join(members)}")

    write_csv(
        "part3_hierarchical.csv",
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hierarchical_rows,
    )

    # ---------------------------------------------------------------- 3.5 PCA
    eigenvalues, eigenvectors, components, scores, mean_vector = pca(Z)
    total = float(np.sum(eigenvalues))
    explained = eigenvalues / total
    cumulative = np.cumsum(explained)
    components_for_80 = int(np.searchsorted(cumulative, 0.8 - 1e-12) + 1)

    results.add("eigenvalue_sum", total)
    for index in range(5):
        results.add(f"eigenvalue_{index + 1}", float(eigenvalues[index]))
    for index in range(5):
        results.add(f"explained_ratio_{index + 1}", float(explained[index]))
    for index in range(5):
        results.add(f"cumulative_ratio_{index + 1}", float(cumulative[index]))
    results.add("components_for_80pct", components_for_80)

    for index in range(5):
        results.add(f"pc1_loading_X{index + 1}", float(components[index, 0]))
    for index in range(5):
        results.add(f"pc2_loading_X{index + 1}", float(components[index, 1]))

    for row in range(3):
        results.add(f"row{row + 1}_pc1", float(scores[row, 0]))
        results.add(f"row{row + 1}_pc2", float(scores[row, 1]))

    for name in "ABC":
        mask = segment_of_row == name
        results.add(f"mean_pc1_segment_{name}", float(np.mean(scores[mask, 0])))

    path = results.write("part3_results.txt")
    summarise_results(path)

    print(
        "  PCA explained: "
        + ", ".join(f"PC{i + 1}={explained[i]:.4f}" for i in range(5))
    )
    print(f"  components needed for 80% of variance: {components_for_80}")
    print(
        "  mean PC1 by segment: "
        + ", ".join(
            f"{name}={np.mean(scores[segment_of_row == name, 0]):.4f}" for name in "ABC"
        )
    )


if __name__ == "__main__":
    main()
