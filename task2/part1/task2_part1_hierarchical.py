"""
Task 2 - Part 1: Customer Segmentation with Hierarchical (Agglomerative) Clustering
Dataset: Wholesale customers data.csv

Pipeline:
  1. Load & preprocess (drop Channel/Region IDs, standardise spend features)
  2. Dendrograms with different linkage methods (single, complete, average, ward)
     to decide the number of clusters
  3. Fit AgglomerativeClustering and visualise the resulting segments
  4. Profile clusters -> business insights
All figures saved next to this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(os.path.join(HERE, "Wholesale customers data.csv"))
    df.columns = [c.strip() for c in df.columns]
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())

    # Channel & Region are categorical IDs, not spending behaviour -> drop them
    spend_cols = ["Fresh", "Milk", "Grocery", "Frozen",
                  "Detergents_Paper", "Delicatessen"]
    # the file header uses lowercase 'delicatessen'
    rename = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename)
    spend_cols = [c for c in ["Fresh", "Milk", "Grocery", "Frozen",
                              "Detergents_Paper", "Delicatessen"] if c in df.columns]
    print("\nSpend columns used:", spend_cols)
    print("\nDescribe:\n", df[spend_cols].describe().T)

    X = df[spend_cols].values
    # Spend is right-skewed across orders of magnitude -> log1p then standardise
    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)

    # ------------------------------------------- dendrograms per linkage
    methods = ["single", "complete", "average", "ward"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, m in zip(axes.ravel(), methods):
        Z = linkage(X_scaled, method=m)
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=20,
                   show_leaf_counts=True, no_labels=True)
        ax.set_title(f"Dendrogram - {m} linkage")
        ax.set_xlabel("Samples (truncated)")
        ax.set_ylabel("Distance")
    fig.suptitle("Hierarchical clustering dendrograms by linkage method", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_dendrograms.png"), dpi=120)
    plt.close(fig)

    # ------------------------ compare linkage methods at k=3 by silhouette
    print("\nSilhouette by linkage (k=3):")
    for m in methods:
        labels = AgglomerativeClustering(n_clusters=3, linkage=m).fit_predict(X_scaled)
        print(f"  {m:9s}: {silhouette_score(X_scaled, labels):.4f}  "
              f"sizes={np.bincount(labels)}")

    # --------------------------------------------- final model (ward, k=3)
    k = 3
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    df["Cluster"] = model.fit_predict(X_scaled)
    print(f"\nFinal model: ward, k={k}, "
          f"silhouette={silhouette_score(X_scaled, df['Cluster']):.4f}")

    # ------------------------------------------------------ visualisation
    # PCA projection for a 2D scatter
    pcs = PCA(n_components=2).fit_transform(X_scaled)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Cluster"],
                    palette="tab10", s=60, ax=ax[0])
    ax[0].set_title("Clusters in PCA space")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")

    sns.scatterplot(data=df, x="Grocery", y="Detergents_Paper",
                    hue="Cluster", palette="tab10", s=60, ax=ax[1])
    ax[1].set_title("Grocery vs Detergents_Paper spend")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_clusters_scatter.png"), dpi=120)
    plt.close(fig)

    # ----------------------------------------------------- cluster profiles
    profile = df.groupby("Cluster")[spend_cols].mean().round(0)
    profile["n"] = df.groupby("Cluster").size()
    print("\nCluster profiles (mean annual spend):\n", profile)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.groupby("Cluster")[spend_cols].mean(), annot=True, fmt=".0f",
                cmap="YlGnBu", ax=ax)
    ax.set_title("Mean annual spend per cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig3_cluster_heatmap.png"), dpi=120)
    plt.close(fig)

    df.to_csv(os.path.join(HERE, "wholesale_clustered.csv"), index=False)
    print("\nSaved figures + wholesale_clustered.csv")


if __name__ == "__main__":
    main()
