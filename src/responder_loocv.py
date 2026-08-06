#!/usr/bin/env python3
"""
Responder vs non-responder binary classification — LOOCV baseline.

Dataset: Prediction Data for GA (02.13.26).xls  (n=36, balanced 18/18)
Models:  LR, RandomForest, XGBoost, TabNet, TabPFN, TabFM

Evaluation: Leave-One-Out CV. Because each LOOCV test fold has a single
sample, per-fold AUC is undefined, so we POOL the 36 out-of-fold
predictions and compute a single set of metrics over them.

Preprocessing (fit inside each training fold to avoid leakage):
  - drop ID columns (study_id, clinic_id, mit_id)
  - one-hot encode `race` (5 nominal levels)
  - pass through binary `gender`, `stim_type`
  - standardize all remaining continuous features
"""
import argparse, json, time, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, f1_score, confusion_matrix)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

ID_COLS = ["study_id", "clinic_id", "mit_id"]
TARGET = "responder"
CAT_COLS = ["race"]            # nominal -> one-hot
BIN_COLS = ["gender", "stim_type"]  # binary -> passthrough (gender remapped 1/2 -> 0/1)


def load_data(path):
    df = pd.read_excel(path, sheet_name="Data")
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    y = df[TARGET].astype(int).values
    X = df.drop(columns=[TARGET]).copy()
    if "gender" in X.columns:
        X["gender"] = (X["gender"] == 2).astype(int)  # 1=M,2=F -> 0/1
    num_cols = [c for c in X.columns if c not in CAT_COLS + BIN_COLS]
    return X, y, num_cols


def make_preprocessor(num_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", [c for c in BIN_COLS]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="drop",
    )


# ----------------------------- model factories -----------------------------
def get_models():
    """Return dict name -> dict(factory, per_fold, available, err).
    per_fold=True  -> new estimator every fold (trained from scratch)
    per_fold=False -> single estimator reused across folds (pretrained FMs)
    """
    models = {}

    # 1. Logistic Regression
    models["LR"] = dict(
        factory=lambda: LogisticRegression(max_iter=5000, C=1.0, random_state=SEED),
        per_fold=True)

    # 2. Random Forest
    models["RF"] = dict(
        factory=lambda: RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
        per_fold=True)

    # 3. XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = dict(
            factory=lambda: XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=SEED, n_jobs=-1, verbosity=0),
            per_fold=True)
    except Exception as e:
        models["XGBoost"] = dict(factory=None, per_fold=True, err=repr(e))

    # 4. TabNet
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch
        def tabnet_factory():
            return TabNetClassifier(seed=SEED, verbose=0,
                                    device_name="cpu",
                                    optimizer_params=dict(lr=2e-2))
        models["TabNet"] = dict(factory=tabnet_factory, per_fold=True, is_tabnet=True)
    except Exception as e:
        models["TabNet"] = dict(factory=None, per_fold=True, err=repr(e))

    # 5. TabPFN
    try:
        from tabpfn import TabPFNClassifier
        models["TabPFN"] = dict(
            factory=lambda: TabPFNClassifier(device="cpu"),
            per_fold=False)
    except Exception as e:
        models["TabPFN"] = dict(factory=None, per_fold=False, err=repr(e))

    # 6. TabFM (google-research/tabfm) — sklearn-compatible, zero-shot
    try:
        import tabfm as _tabfm
        try:
            _tabfm_model = _tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")
        except Exception:
            _tabfm_model = _tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
        models["TabFM"] = dict(
            factory=lambda: _tabfm.TabFMClassifier(model=_tabfm_model),
            per_fold=False)
    except Exception as e:
        models["TabFM"] = dict(factory=None, per_fold=False, err=repr(e))

    return models


# ----------------------------- LOOCV runner --------------------------------
def run_loocv(name, cfg, X, y, num_cols):
    if cfg.get("factory") is None:
        return dict(model=name, status="unavailable", error=cfg.get("err"))

    loo = LeaveOneOut()
    n = len(y)
    oof_pred = np.zeros(n, dtype=int)
    oof_proba = np.zeros(n, dtype=float)
    t0 = time.time()

    shared_est = None if cfg["per_fold"] else cfg["factory"]()

    for tr, te in loo.split(X):
        pre = make_preprocessor(num_cols)
        Xtr = pre.fit_transform(X.iloc[tr])
        Xte = pre.transform(X.iloc[te])
        Xtr = np.asarray(Xtr, dtype=np.float32)
        Xte = np.asarray(Xte, dtype=np.float32)
        ytr = y[tr]

        est = cfg["factory"]() if cfg["per_fold"] else shared_est

        if cfg.get("is_tabnet"):
            est.fit(Xtr, ytr, max_epochs=100, patience=0,
                    batch_size=min(16, len(ytr)), drop_last=False)
        else:
            est.fit(Xtr, ytr)

        proba = est.predict_proba(Xte)
        classes = list(est.classes_)
        pos_idx = classes.index(1) if 1 in classes else 1
        p1 = float(np.asarray(proba)[0, pos_idx])
        oof_proba[te[0]] = p1
        oof_pred[te[0]] = int(p1 >= 0.5)

    dt = time.time() - t0
    return dict(
        model=name, status="ok",
        accuracy=round(accuracy_score(y, oof_pred), 4),
        balanced_accuracy=round(balanced_accuracy_score(y, oof_pred), 4),
        roc_auc=round(roc_auc_score(y, oof_proba), 4),
        f1=round(f1_score(y, oof_pred), 4),
        confusion_matrix=confusion_matrix(y, oof_pred).tolist(),
        runtime_sec=round(dt, 1),
    )


def run_kfold(name, cfg, X, y, num_cols, n_splits=5):
    """Stratified k-fold CV. Reports per-fold metrics -> mean/std, plus pooled."""
    if cfg.get("factory") is None:
        return dict(model=name, status="unavailable", error=cfg.get("err"))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    shared_est = None if cfg["per_fold"] else cfg["factory"]()
    fold_auc, fold_acc, fold_bacc, fold_f1 = [], [], [], []
    pool_y, pool_p = [], []
    t0 = time.time()

    for tr, te in skf.split(X, y):
        pre = make_preprocessor(num_cols)
        Xtr = np.asarray(pre.fit_transform(X.iloc[tr]), dtype=np.float32)
        Xte = np.asarray(pre.transform(X.iloc[te]), dtype=np.float32)
        ytr, yte = y[tr], y[te]
        est = cfg["factory"]() if cfg["per_fold"] else shared_est
        if cfg.get("is_tabnet"):
            est.fit(Xtr, ytr, max_epochs=100, patience=0,
                    batch_size=min(16, len(ytr)), drop_last=False)
        else:
            est.fit(Xtr, ytr)
        proba = np.asarray(est.predict_proba(Xte))
        classes = list(est.classes_)
        pos = classes.index(1) if 1 in classes else 1
        p1 = proba[:, pos]
        pred = (p1 >= 0.5).astype(int)
        fold_auc.append(roc_auc_score(yte, p1))
        fold_acc.append(accuracy_score(yte, pred))
        fold_bacc.append(balanced_accuracy_score(yte, pred))
        fold_f1.append(f1_score(yte, pred, zero_division=0))
        pool_y.extend(yte.tolist()); pool_p.extend(p1.tolist())

    m = lambda a: round(float(np.mean(a)), 4)
    s = lambda a: round(float(np.std(a)), 4)
    return dict(
        model=name, status="ok", n_splits=n_splits,
        roc_auc_mean=m(fold_auc), roc_auc_std=s(fold_auc),
        accuracy_mean=m(fold_acc), accuracy_std=s(fold_acc),
        balanced_accuracy_mean=m(fold_bacc), balanced_accuracy_std=s(fold_bacc),
        f1_mean=m(fold_f1), f1_std=s(fold_f1),
        roc_auc_pooled=round(roc_auc_score(pool_y, pool_p), 4),
        per_fold_auc=[round(a, 4) for a in fold_auc],
        runtime_sec=round(time.time() - t0, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of model names to run")
    ap.add_argument("--cv", type=int, default=0,
                    help="k for stratified k-fold CV; 0 = LOOCV (default)")
    args = ap.parse_args()

    X, y, num_cols = load_data(args.data)
    print(f"[data] X={X.shape}  y balance={np.bincount(y).tolist()}  "
          f"n_numeric={len(num_cols)}", flush=True)

    models = get_models()
    if args.only:
        want = set(s.strip() for s in args.only.split(","))
        models = {k: v for k, v in models.items() if k in want}

    kfold = args.cv and args.cv > 0
    tag = f"{args.cv}-FOLD CV" if kfold else "LOOCV"
    results = []
    for name, cfg in models.items():
        print(f"\n[run] {name} ...", flush=True)
        try:
            r = (run_kfold(name, cfg, X, y, num_cols, args.cv) if kfold
                 else run_loocv(name, cfg, X, y, num_cols))
        except Exception as e:
            r = dict(model=name, status="error", error=repr(e))
        results.append(r)
        keys = (("roc_auc_mean", "roc_auc_std", "accuracy_mean", "runtime_sec")
                if kfold else
                ("accuracy", "balanced_accuracy", "roc_auc", "f1", "runtime_sec"))
        print(f"[done] {name}: "
              + (", ".join(f"{k}={r[k]}" for k in keys if k in r)
                 or r.get("error") or r.get("status")), flush=True)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fname = f"cv{args.cv}_results" if kfold else "loocv_results"
    with open(outdir / f"{fname}.json", "w") as f:
        json.dump(results, f, indent=2)
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        cols = (["model", "roc_auc_mean", "roc_auc_std", "roc_auc_pooled",
                 "accuracy_mean", "balanced_accuracy_mean", "f1_mean", "runtime_sec"]
                if kfold else
                ["model", "accuracy", "balanced_accuracy", "roc_auc", "f1", "runtime_sec"])
        sortcol = "roc_auc_mean" if kfold else "roc_auc"
        tab = pd.DataFrame(ok)[cols].sort_values(sortcol, ascending=False)
        tab.to_csv(outdir / f"{fname}.csv", index=False)
        print(f"\n===================== {tag} SUMMARY =====================")
        print(tab.to_string(index=False))
    print(f"\n[saved] {outdir/f'{fname}.json'}")


if __name__ == "__main__":
    main()
