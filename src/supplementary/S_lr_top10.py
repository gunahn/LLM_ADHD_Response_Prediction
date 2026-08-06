#!/usr/bin/env python3
"""
S_lr_top10.py
Logistic-regression top-10 feature importance (illustrative; manuscript
supplementary). Ranks features by |standardized coefficient|, averaged over
LOOCV folds for stability, and plots the top 10 with direction of effect.

NOTE: selection is performed on the full data, so the refit AUC is an
optimistic, in-sample figure (selection leakage). Reported as illustrative
interpretability, not as a leakage-safe generalization estimate.

Usage:
  python src/supplementary/S_lr_top10.py --data "data/Prediction Data for GA (02.13.26).xls"
"""
import argparse
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

SEED = 42
ID = ["study_id", "clinic_id", "mit_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/figures/lr_top10_feature_importance.png")
    args = ap.parse_args()
    df = pd.read_excel(args.data, sheet_name="Data")
    y = df["responder"].astype(int).values
    X = df.drop(columns=[c for c in ID + ["responder"] if c in df.columns]).copy()
    X["gender"] = (X["gender"] == 2).astype(int)
    Xd = pd.get_dummies(X, columns=["race"]).astype(float)
    Xs = StandardScaler().fit_transform(Xd)
    coefs = []
    for tr, _ in LeaveOneOut().split(Xs):
        m = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED).fit(Xs[tr], y[tr])
        coefs.append(m.coef_[0])
    mean_c = np.mean(coefs, axis=0); sd_c = np.std(coefs, axis=0)
    rank = pd.DataFrame({"feature": Xd.columns, "coef": mean_c, "sd": sd_c})
    rank["absc"] = rank.coef.abs()
    top = rank.sort_values("absc", ascending=False).head(10)
    print(top[["feature", "coef", "sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
