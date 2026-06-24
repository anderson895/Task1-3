"""
Task 3 - Part 1: DBSCAN Clustering on the Wine Dataset
Dataset: wine-clustering.csv (13 physicochemical features)

Pipeline:
  1. Preprocess (standardise features)
  2. k-distance graph to choose eps (min_samples = 2*dim is a common rule)
  3. Fit DBSCAN, visualise via PCA and t-SNE, identify noise points
  4. Experiment with several (eps, min_samples) combinations
All figures saved next to this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(os.path.join(HERE, "wine-clustering.csv"))
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum().sum(), "total")
    print("\nDescribe:\n", df.describe().T)

    X = StandardScaler().fit_transform(df.values)
    dim = X.shape[1]
    # With only 178 samples, 2*dim (=26) is too strict; min_samples=10 is a
    # better balance (> dim+1, still demands a reasonably dense neighbourhood).
    min_samples = 10
    print(f"\nFeatures={dim}, using min_samples={min_samples} for the k-graph")

    # --------------------------------------------- k-distance graph for eps
    nn = NearestNeighbors(n_neighbors=min_samples)
    dist, _ = nn.fit(X).kneighbors(X)
    kdist = np.sort(dist[:, -1])  # distance to the k-th neighbour, sorted

    # Geometric knee (Kneedle): point on the sorted curve furthest from the
    # straight line joining its first and last point.
    x_idx = np.arange(len(kdist))
    p1, p2 = np.array([0, kdist[0]]), np.array([len(kdist) - 1, kdist[-1]])
    line = p2 - p1
    line = line / np.linalg.norm(line)
    vecs = np.column_stack([x_idx, kdist]) - p1
    proj = vecs - np.outer(vecs @ line, line)
    knee = int(np.argmax(np.linalg.norm(proj, axis=1)))
    eps_guess = round(float(kdist[knee]), 2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(kdist, color="steelblue")
    ax.set_title(f"k-distance graph (k={min_samples}) - look for the 'knee'")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"Distance to {min_samples}-th nearest neighbour")
    ax.axhline(eps_guess, color="red", ls="--", label=f"eps ~ {eps_guess} (knee)")
    ax.axvline(knee, color="green", ls=":", alpha=.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_kdistance.png"), dpi=120)
    plt.close(fig)
    print(f"Chosen eps from knee heuristic: {eps_guess}")

    # ------------------------------- experiment with eps / min_samples grid
    # Run this first so we can pick the most informative configuration.
    print("\nParameter experiments:")
    rows = []
    for eps in [2.0, 2.2, 2.5, 2.8, 3.11, 3.5]:
        for ms in [5, 10]:
            lb = DBSCAN(eps=eps, min_samples=ms).fit(X).labels_
            nc = len(set(lb)) - (1 if -1 in lb else 0)
            nn_ = int((lb == -1).sum())
            sil = np.nan
            if nc >= 2:
                m = lb != -1
                sil = silhouette_score(X[m], lb[m])
            rows.append({"eps": eps, "min_samples": ms, "clusters": nc,
                         "noise": nn_,
                         "silhouette": round(sil, 3) if sil == sil else np.nan})
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    grid.to_csv(os.path.join(HERE, "dbscan_param_experiments.csv"), index=False)

    # --------------------------------------------------- final DBSCAN model
    # The knee eps over-merges (clusters of similar density form one blob);
    # pick the configuration that yields >=2 clusters with the best silhouette.
    valid = grid.dropna(subset=["silhouette"])
    best = valid.loc[valid["silhouette"].idxmax()]
    eps_final, ms_final = float(best["eps"]), int(best["min_samples"])
    print(f"\nKnee eps={eps_guess} over-merges; selecting best-silhouette "
          f"config: eps={eps_final}, min_samples={ms_final}")

    db = DBSCAN(eps=eps_final, min_samples=ms_final).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"[DBSCAN eps={eps_final}, min_samples={ms_final}] "
          f"clusters={n_clusters}, noise points={n_noise}")
    if n_clusters >= 2:
        mask = labels != -1
        print("Silhouette (excl. noise):",
              round(silhouette_score(X[mask], labels[mask]), 4))

    # ------------------------------- PCA & t-SNE 2D visualisation
    pcs = PCA(n_components=2).fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=30, init="pca").fit_transform(X)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for a, emb, name in [(ax[0], pcs, "PCA"), (ax[1], tsne, "t-SNE")]:
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels,
                        palette="tab10", style=(labels == -1),
                        ax=a, s=55, legend="full")
        a.set_title(f"DBSCAN clusters in {name} space "
                    f"(noise marked, n={n_noise})")
        a.set_xlabel(f"{name}-1"); a.set_ylabel(f"{name}-2")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_dbscan_pca_tsne.png"), dpi=120)
    plt.close(fig)

    # store labelled data + cluster chemical profiles
    out = df.copy()
    out["Cluster"] = labels
    profile = out.groupby("Cluster")[["Alcohol", "Malic_Acid", "Color_Intensity",
                                      "Flavanoids", "Proline"]].mean().round(2)
    print("\nCluster chemical profile (selected features):\n", profile)
    out.to_csv(os.path.join(HERE, "wine_dbscan_clustered.csv"), index=False)
    print("\nSaved figures + CSVs")


if __name__ == "__main__":
    main()
