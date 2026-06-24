"""
Task 1 - Part 2: Principal Component Analysis (PCA) on the Wine Quality Dataset
Dataset: winequality-red.csv

Pipeline:
  1. Preprocessing  -> check missing values + basic stats, standardise features
                       (target `quality` is excluded from PCA)
  2. Apply PCA      -> explained variance ratio + cumulative variance
  3. Identify minimum number of components to retain >= 90% variance
  4. Visualisation  -> 2D scatter of PC1 vs PC2, coloured by quality category
                       (Low: 3-5, Medium: 6, High: 7-8)
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid")


def quality_category(q):
    if q <= 5:
        return "Low (3-5)"
    elif q == 6:
        return "Medium (6)"
    return "High (7-8)"


def main():
    # ----------------------------------------------------------------- load
    df = pd.read_csv(os.path.join(HERE, "winequality-red.csv"))
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nBasic statistics:\n", df.describe().T)

    target = "quality"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values

    # ------------------------------------------------------- standardise
    X_scaled = StandardScaler().fit_transform(X)

    # --------------------------------------------------------------- PCA
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    print("\nExplained variance ratio per component:")
    for i, (e, c) in enumerate(zip(evr, cum), start=1):
        print(f"  PC{i:2d}: {e*100:6.2f}%   cumulative: {c*100:6.2f}%")

    n_90 = int(np.argmax(cum >= 0.90) + 1)
    print(f"\nMinimum components to retain >= 90% variance: {n_90}")

    # Scree + cumulative variance plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    comps = np.arange(1, len(evr) + 1)
    ax[0].bar(comps, evr * 100, color="steelblue")
    ax[0].set_title("Explained Variance Ratio (Scree plot)")
    ax[0].set_xlabel("Principal component")
    ax[0].set_ylabel("Variance explained (%)")

    ax[1].plot(comps, cum * 100, "o-", color="darkorange")
    ax[1].axhline(90, color="red", ls="--", label="90% threshold")
    ax[1].axvline(n_90, color="green", ls="--", label=f"{n_90} components")
    ax[1].set_title("Cumulative Explained Variance")
    ax[1].set_xlabel("Number of components")
    ax[1].set_ylabel("Cumulative variance (%)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_explained_variance.png"), dpi=120)
    plt.close(fig)

    # ----------------------------------------------- 2D scatter by category
    df["quality_category"] = df[target].apply(quality_category)
    order = ["Low (3-5)", "Medium (6)", "High (7-8)"]
    pc_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1],
                          "quality_category": df["quality_category"]})

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue="quality_category",
                    hue_order=order, palette={"Low (3-5)": "tomato",
                    "Medium (6)": "gold", "High (7-8)": "seagreen"},
                    alpha=.7, s=45, ax=ax)
    ax.set_title("Wine samples projected onto PC1 vs PC2")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.legend(title="Quality category")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_pca_scatter.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------- PCA loadings (PC1 & PC2)
    loadings = pd.DataFrame(pca.components_[:2].T, index=feature_cols,
                            columns=["PC1", "PC2"])
    print("\nLoadings on PC1 / PC2 (feature contributions):\n",
          loadings.round(3))

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature loadings on PC1 and PC2")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig3_loadings.png"), dpi=120)
    plt.close(fig)

    # ------------------------------- classification: original vs reduced
    # Compare a classifier trained on the 11 original (standardised) features
    # versus the reduced PCA components, predicting the 3 quality categories.
    y = df["quality_category"].values
    clf = lambda: LogisticRegression(max_iter=2000, multi_class="auto")

    comparisons = {}
    # (a) original standardised features
    comparisons["Original 11 features"] = cross_val_score(
        clf(), X_scaled, y, cv=5, scoring="accuracy").mean()
    # (b) PCA reduced to the components retaining >=90% variance
    comparisons[f"PCA {n_90} comps (>=90% var)"] = cross_val_score(
        make_pipeline(PCA(n_components=n_90), clf()), X_scaled, y,
        cv=5, scoring="accuracy").mean()
    # (c) PCA reduced to just the first 2 components (the 2D plot)
    comparisons["PCA 2 comps"] = cross_val_score(
        make_pipeline(PCA(n_components=2), clf()), X_scaled, y,
        cv=5, scoring="accuracy").mean()

    print("\n5-fold CV accuracy (LogisticRegression, 3 quality classes):")
    for k, v in comparisons.items():
        print(f"  {k:28s}: {v:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(comparisons.keys())
    ax.barh(names, [comparisons[n] for n in names], color="slateblue")
    for i, n in enumerate(names):
        ax.text(comparisons[n] + 0.003, i, f"{comparisons[n]:.3f}", va="center")
    ax.set_xlabel("5-fold CV accuracy")
    ax.set_title("Classifier performance: original vs PCA-reduced features")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig4_classification_compare.png"), dpi=120)
    plt.close(fig)

    print("\nSaved figures.")


if __name__ == "__main__":
    main()
