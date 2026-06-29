"""
Part 2 - Task 2: Preprocessing, Scaling & Feature Selection on Telco Customer Churn
Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv   |   Target: Churn (Yes/No)

Steps:
  1. Preprocessing & feature scaling (MinMax / Standard / Robust) + boxplots
  2. VarianceThreshold(0.01)
  3. SelectKBest(chi2) top 10  vs  RFE(LogisticRegression) top 10
  4. LASSO (LassoCV) feature selection
  5. Final pipeline: impute + scale + encode -> SelectFromModel(LassoCV)
     -> LogisticRegression, evaluated with Accuracy / ROC-AUC / RMSE / MAE
All figures saved next to this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler,
                                   OneHotEncoder)
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, chi2,
                                       RFE, SelectFromModel)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error)

HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid")
RNG = 42


def main():
    # ------------------------------------------------- 1. load & preprocess
    df = pd.read_csv(os.path.join(HERE, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    print("Shape:", df.shape)
    df = df.drop(columns=["customerID"])

    # TotalCharges has blank strings -> coerce to numeric then impute median
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    print(f"TotalCharges missing after coercion: {n_missing}")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target
    y = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn"])

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    # SeniorCitizen is already 0/1; treat the rest of the objects as categorical
    cat_cols = [c for c in df.columns if c not in numeric_cols + ["SeniorCitizen"]]
    print(f"Numeric: {numeric_cols}\nCategorical ({len(cat_cols)}): {cat_cols}")

    # --------------------------- scaler comparison via boxplots
    scalers = {"MinMaxScaler": MinMaxScaler(),
               "StandardScaler": StandardScaler(),
               "RobustScaler": RobustScaler()}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, sc) in zip(axes, scalers.items()):
        scaled = pd.DataFrame(sc.fit_transform(df[numeric_cols]),
                              columns=numeric_cols)
        sns.boxplot(data=scaled, ax=ax)
        ax.set_title(name); ax.set_ylabel("scaled value")
    fig.suptitle("Numeric features under different scalers", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_scaler_boxplots.png"), dpi=120)
    plt.close(fig)

    # --------------------------- one-hot encode the full feature matrix
    ohe = OneHotEncoder(drop="if_binary", sparse_output=False,
                        handle_unknown="ignore")
    X_cat = pd.DataFrame(ohe.fit_transform(df[cat_cols]),
                         columns=ohe.get_feature_names_out(cat_cols),
                         index=df.index)
    X_num = df[numeric_cols + ["SeniorCitizen"]].reset_index(drop=True)
    X_full = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
    print(f"\nEncoded feature matrix shape: {X_full.shape}")

    # MinMax-scaled version (needed for chi2 which requires non-negative input)
    X_minmax = pd.DataFrame(MinMaxScaler().fit_transform(X_full),
                            columns=X_full.columns)

    # ------------------------------------------------- 2. VarianceThreshold
    # Applied to the encoded matrix (binary dummies have variance p*(1-p));
    # this flags near-constant columns coming from very rare categories.
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_full)
    kept = X_full.columns[vt.get_support()]
    removed = X_full.columns[~vt.get_support()]
    print(f"\n[VarianceThreshold 0.01] removed {len(removed)} features:")
    print(list(removed))

    # ------------------------------------------------- 3. SelectKBest + RFE
    skb = SelectKBest(chi2, k=10).fit(X_minmax, y)
    skb_feats = list(X_full.columns[skb.get_support()])
    print(f"\n[SelectKBest chi2] top 10:\n{skb_feats}")

    rfe = RFE(LogisticRegression(max_iter=2000, solver="liblinear"),
              n_features_to_select=10)
    rfe.fit(X_minmax, y)
    rfe_feats = list(X_full.columns[rfe.get_support()])
    print(f"\n[RFE LogisticRegression] top 10:\n{rfe_feats}")

    overlap = sorted(set(skb_feats) & set(rfe_feats))
    print(f"\nOverlap ({len(overlap)}): {overlap}")

    # ------------------------------------------------- 4. LASSO selection
    X_std = StandardScaler().fit_transform(X_full)
    lasso = LassoCV(cv=5, random_state=RNG, max_iter=5000).fit(X_std, y)
    coef = pd.Series(lasso.coef_, index=X_full.columns)
    nonzero = coef[coef != 0].sort_values(key=abs, ascending=False)
    print(f"\n[LassoCV] alpha={lasso.alpha_:.5f}, "
          f"{(coef != 0).sum()} non-zero of {len(coef)} features")
    print(nonzero.round(4))

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(nonzero))))
    nonzero.sort_values().plot.barh(ax=ax, color="teal")
    ax.set_title("LASSO non-zero feature coefficients")
    ax.set_xlabel("coefficient")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_lasso_coefficients.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------- 5. final pipeline
    X_raw = df.copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=RNG, stratify=y)

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols + ["SeniorCitizen"]),
        ("cat", cat_pipe, cat_cols)])

    pipe = Pipeline([
        ("pre", pre),
        ("select", SelectFromModel(LassoCV(cv=5, random_state=RNG, max_iter=5000))),
        ("clf", LogisticRegression(max_iter=2000))])
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "ROC-AUC": roc_auc_score(y_te, y_proba),
        "RMSE": np.sqrt(mean_squared_error(y_te, y_proba)),
        "MAE": mean_absolute_error(y_te, y_proba),
    }
    n_selected = pipe.named_steps["select"].get_support().sum()
    print(f"\n[Final pipeline] LASSO selected {n_selected} features")
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k:9s}: {v:.4f}")

    pd.Series(metrics).round(4).to_csv(os.path.join(HERE, "pipeline_metrics.csv"))
    print("\nSaved figures + pipeline_metrics.csv")


if __name__ == "__main__":
    main()
