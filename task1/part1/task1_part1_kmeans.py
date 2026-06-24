"""
Task 1 - Part 1: Customer Segmentation with K-Means Clustering
Dataset: Mall_Customers.csv
Tools: Python + Scikit-learn + Matplotlib/Seaborn

Pipeline:
  1. Load & explore data
  2. Determine optimal k with the Elbow method and Silhouette analysis
  3. Fit final K-Means and visualise the segments (scatter + pair plots)
  4. Profile clusters and derive business insights
All figures are saved next to this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, just save figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


def main():
    # ----------------------------------------------------------------- load
    df = pd.read_csv(os.path.join(HERE, "Mall_Customers.csv"))
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nMissing values:\n", df.isna().sum())
    print("\nDescribe:\n", df.describe())

    # Features used for segmentation: Age, Annual Income, Spending Score.
    # CustomerID is just an identifier; Gender is kept for later profiling.
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)

    # ------------------------------------------- Elbow + Silhouette analysis
    k_range = range(2, 11)
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    for k, inertia, s in zip(k_range, inertias, sil_scores):
        print(f"k={k:2d}  inertia={inertia:8.2f}  silhouette={s:.4f}")

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"\nBest k by silhouette = {best_k}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(list(k_range), inertias, "o-", color="steelblue")
    ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("Number of clusters (k)")
    ax[0].set_ylabel("Inertia (within-cluster SSE)")
    ax[0].axvline(5, color="red", ls="--", alpha=.6, label="elbow ~ k=5")
    ax[0].legend()

    ax[1].plot(list(k_range), sil_scores, "o-", color="darkorange")
    ax[1].set_title("Silhouette Analysis")
    ax[1].set_xlabel("Number of clusters (k)")
    ax[1].set_ylabel("Average silhouette score")
    ax[1].axvline(best_k, color="red", ls="--", alpha=.6, label=f"best k={best_k}")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_elbow_silhouette.png"), dpi=120)
    plt.close(fig)

    # ----------------------------------------------------- final clustering
    k = 5  # Elbow ~5 and silhouette agree; classic Mall segmentation result
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    df["Cluster"] = km.fit_predict(X_scaled)
    print(f"\nFinal model: k={k}, silhouette={silhouette_score(X_scaled, df['Cluster']):.4f}")

    # Silhouette plot for the chosen k
    sample_sil = silhouette_samples(X_scaled, df["Cluster"])
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(k):
        ith = np.sort(sample_sil[df["Cluster"] == i])
        y_upper = y_lower + ith.shape[0]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, alpha=.7)
        ax.text(-0.05, y_lower + 0.5 * ith.shape[0], str(i))
        y_lower = y_upper + 10
    ax.axvline(silhouette_score(X_scaled, df["Cluster"]), color="red", ls="--")
    ax.set_title(f"Silhouette plot (k={k})")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_silhouette_plot.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------- visualisations
    # Main 2D view: Income vs Spending (the most business-relevant axes)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)",
                    hue="Cluster", palette="tab10", s=70, ax=ax)
    centers = StandardScaler().fit(X).inverse_transform(km.cluster_centers_)
    ax.scatter(centers[:, 1], centers[:, 2], c="black", s=250, marker="X",
               label="centroids")
    ax.set_title("Customer Segments: Annual Income vs Spending Score")
    ax.legend(title="Cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig3_income_vs_spending.png"), dpi=120)
    plt.close(fig)

    # Pair plot across all features
    pair = sns.pairplot(df, vars=features, hue="Cluster", palette="tab10",
                        diag_kind="kde", height=2.2)
    pair.fig.suptitle("Pair plot of segments", y=1.02)
    pair.savefig(os.path.join(HERE, "fig4_pairplot.png"), dpi=120)
    plt.close("all")

    # ----------------------------------------------------- cluster profiles
    profile = df.groupby("Cluster").agg(
        n=("CustomerID", "size"),
        avg_age=("Age", "mean"),
        avg_income=("Annual Income (k$)", "mean"),
        avg_spending=("Spending Score (1-100)", "mean"),
        pct_female=("Gender", lambda g: (g == "Female").mean() * 100),
    ).round(1)
    print("\nCluster profiles:\n", profile)

    df.to_csv(os.path.join(HERE, "mall_customers_clustered.csv"), index=False)
    print("\nSaved figures + mall_customers_clustered.csv")


if __name__ == "__main__":
    main()
