"""Part 3 - Segmentation and dimensionality reduction.
Run from the repository root:  python part3.py
Needs data/records.csv (from generate_data.py) and common.py.
Writes outputs/part3_results.txt, outputs/part3_segments.csv, outputs/part3_hierarchical.csv.
K-means, the three linkage rules and the PCA steps are written here by hand;
numpy.linalg.eigh is the one library routine used (allowed for PCA step 4)."""
import os

import numpy as np

from common import Results, f4, load_records, sample_sd_cols

res = Results()
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------------ data: X1..X5 only
X_all, outcome, value = load_records()
X_raw = X_all[:, :5]                                  # five continuous predictors, original units
n = X_raw.shape[0]
mu = X_raw.mean(axis=0)
sd = sample_sd_cols(X_raw)
Z = (X_raw - mu) / sd                                 # standardised over all 800 rows


# ------------------------------------------------------------------ Task 3.1
def kmeans(data, k):
    """Pinned rules: start centroids = rows 0, floor(n/k), 2*floor(n/k), ...; assign by squared
    Euclidean distance (ties -> lower-numbered centroid); centroid := mean of members (an empty
    cluster keeps its centroid); stop at the first assignment pass where no row changes."""
    step = data.shape[0] // k
    centroids = data[[i * step for i in range(k)]].copy()
    labels = np.full(data.shape[0], -1)
    passes = 0
    while True:
        d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)            # argmin returns the first (lowest) index on ties
        passes += 1
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = data[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
    wcss = float(((data - centroids[labels]) ** 2).sum())
    return labels, centroids, wcss, passes


wcss = {}
runs = {}
for k in range(2, 7):
    labels, centroids, w, passes = kmeans(Z, k)
    wcss[k] = w
    runs[k] = (labels, passes)
    res.put(f"wcss_k{k}", w)

# ------------------------------------------------------------------ Task 3.2
elbow_k = None
for k in (3, 4, 5):
    if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k]):
        elbow_k = k
        break
res.put("elbow_k", elbow_k if elbow_k is not None else "none")
# whatever elbow_k is, K = 3 is used for the rest of this Part

# ------------------------------------------------------------------ Task 3.3
labels3 = runs[3][0]
x3_mean = [float(X_raw[labels3 == j, 2].mean()) for j in range(3)]
order = sorted(range(3), key=lambda j: x3_mean[j])    # ascending mean of X3 -> segments A, B, C
segment_of_cluster = {cl: name for cl, name in zip(order, "ABC")}
seg = np.array([segment_of_cluster[c] for c in labels3])

sizes = {}
with open(os.path.join("outputs", "part3_segments.csv"), "w") as f:
    f.write("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value\n")
    for name in "ABC":
        mask = seg == name
        sizes[name] = int(mask.sum())
        means = [f4(X_raw[mask, c].mean()) for c in range(5)]
        f.write(",".join([name, str(sizes[name])] + means +
                         [f4(outcome[mask].mean()), f4(value[mask].mean())]) + "\n")
for name in "ABC":
    res.put(f"size_segment_{name}", sizes[name])

# ------------------------------------------------------------------ Task 3.4
D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
])


def link_distance(a, b, rule):
    """Linkage distance between two clusters (sets of 0-based member indices)."""
    dists = [D[i, j] for i in a for j in b]
    if rule == "single":
        return min(dists)
    if rule == "complete":
        return max(dists)
    return sum(dists) / len(dists)                    # average: mean of all member-to-member distances


def name_of(cluster):
    return "+".join(f"E{i + 1}" for i in sorted(cluster))


def agglomerate(rule):
    clusters = [frozenset([i]) for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                h = link_distance(clusters[x], clusters[y], rule)
                if best is None or h < best[0]:       # strict < : no ties occur on this table
                    best = (h, x, y)
        h, x, y = best
        a, b = clusters[x], clusters[y]
        if min(b) < min(a):                           # cluster_a holds the smaller smallest-member number
            a, b = b, a
        merges.append((a, b, h))
        clusters = [c for idx, c in enumerate(clusters) if idx not in (x, y)] + [a | b]
    return merges


cut_info = {}
with open(os.path.join("outputs", "part3_hierarchical.csv"), "w") as f:
    f.write("linkage,merge_no,cluster_a,cluster_b,height\n")
    for rule in ("single", "complete", "average"):
        merges = agglomerate(rule)
        for no, (a, b, h) in enumerate(merges, start=1):
            f.write(f"{rule},{no},{name_of(a)},{name_of(b)},{f4(h)}\n")
        heights = [m[2] for m in merges]
        gaps = [heights[i + 1] - heights[i] for i in range(4)]
        g = int(np.argmax(gaps))                      # largest gap (first one if tied)
        merges_below_cut = g + 1                      # merges 1..g+1 lie below the cut
        n_clusters = 6 - merges_below_cut
        # members of the clusters at the cut (console only, for the README)
        groups = [frozenset([i]) for i in range(6)]
        for a, b, _h in merges[:merges_below_cut]:
            groups = [c for c in groups if c not in (a, b)] + [a | b]
        cut_info[rule] = (n_clusters, sorted(name_of(c) for c in groups))
for rule in ("single", "complete", "average"):
    res.put(f"{rule}_clusters_at_cut", cut_info[rule][0])

# ------------------------------------------------------------------ Task 3.5  PCA, seven steps
# (1) mean vector: all zeros after standardising (state this in the README)
mean_vec = Z.mean(axis=0)
# (2) centre the data
Xc = Z - mean_vec
# (3) covariance matrix S = Xc^T Xc / (n - 1)
S = (Xc.T @ Xc) / (n - 1)
# (4) eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eigh(S)
# (5) sort by eigenvalue, largest first
idx = np.argsort(-eigvals, kind="stable")
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
# sign convention: the entry of largest absolute value in each eigenvector is positive
for j in range(eigvecs.shape[1]):
    if eigvecs[np.argmax(np.abs(eigvecs[:, j])), j] < 0:
        eigvecs[:, j] = -eigvecs[:, j]
# (6) keep the top two eigenvectors
W = eigvecs[:, :2]
# (7) project
P = Xc @ W

total = float(eigvals.sum())
res.put("eigenvalue_sum", total)
for j in range(5):
    res.put(f"eigenvalue_{j + 1}", eigvals[j])
ratios = eigvals / total
for j in range(5):
    res.put(f"explained_ratio_{j + 1}", ratios[j])
cum = np.cumsum(ratios)
for j in range(5):
    res.put(f"cumulative_ratio_{j + 1}", cum[j])
res.put("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1)
for j in range(5):
    res.put(f"pc1_loading_X{j + 1}", eigvecs[j, 0])
for j in range(5):
    res.put(f"pc2_loading_X{j + 1}", eigvecs[j, 1])
for r in range(3):                                   # first three data rows (indices 0, 1, 2)
    res.put(f"row{r + 1}_pc1", P[r, 0])
    res.put(f"row{r + 1}_pc2", P[r, 1])
for name in "ABC":
    res.put(f"mean_pc1_segment_{name}", float(P[seg == name, 0].mean()))

# ------------------------------------------------------------------ write
res.write(os.path.join("outputs", "part3_results.txt"))
print(open(os.path.join("outputs", "part3_results.txt")).read())

# console-only extras to help with the README (not part of any graded file)
print("--- console only ---")
print("assignment passes (K=4, 5, 6):", runs[4][1], runs[5][1], runs[6][1])
for rule in ("single", "complete", "average"):
    print(f"{rule} cut ->", cut_info[rule][0], "clusters:", cut_info[rule][1])
print("successive WCSS drops 2-3, 3-4, 4-5, 5-6:",
      [round(wcss[k] - wcss[k + 1], 4) for k in range(2, 6)])
