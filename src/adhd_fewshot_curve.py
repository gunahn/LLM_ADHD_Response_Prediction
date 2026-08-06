#!/usr/bin/env python3
"""
N-shot learning curve on the ADHD responder data (n=36, balanced 18/18).

Question: how does ROC-AUC improve as each model is given more labeled training
examples?  shots N in {0, 2, 8, 16, 24}  (N = TOTAL training examples, stratified
N/2 per class; 24/class is impossible with only 18/class, so N is a total).

Protocol (repeated stratified subsampling learning curve):
  for each shot count N:
    for each of K repeats:
      - draw N stratified training examples (N/2 per class), seed = repeat
      - the remaining 36-N patients are the test set
      - fit ALL models on the SAME draw (fair), score AUROC on the test set
    aggregate AUROC over the K repeats -> mean, std, 95% percentile CI
  0-shot: TabFM only (classical models cannot fit on 0 examples).

Preprocessing (scale + one-hot race) is fit on the N training examples only.
Outputs JSON (per model x shot) + a tidy CSV for plotting.
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys; sys.path.insert(0, str(Path(__file__).parent))
from responder_loocv import load_data, make_preprocessor

warnings.filterwarnings("ignore")
SEED = 42
SHOTS = [0, 2, 8, 16, 24]

_TABFM = None
def tabfm_model():
    global _TABFM
    if _TABFM is None:
        import tabfm, torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _TABFM = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification", device=dev)
    return _TABFM


# models whose LOOCV is deterministic (repeats add no variance -> run once)
DETERMINISTIC = {"LR", "TabFM"}

def build(name, seed=SEED):
    """Fresh estimator each call (few-shot => refit on each draw)."""
    if name == "LR":
        return LogisticRegression(max_iter=5000, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             eval_metric="logloss", random_state=seed, verbosity=0)
    if name == "TabNet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        return TabNetClassifier(seed=seed, verbose=0, device_name="cpu")
    if name == "TabFM":
        import tabfm
        return tabfm.TabFMClassifier(model=tabfm_model())
    raise ValueError(name)


def loocv_auc(name, X, y, num_cols, seed=SEED):
    """Pooled leave-one-out AUROC for one model (= the 35-shot point, n-1 train)."""
    from sklearn.model_selection import LeaveOneOut
    oof = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        pre = make_preprocessor(num_cols)
        Xtr = np.asarray(pre.fit_transform(X.iloc[tr]), dtype=np.float32)
        Xte = np.asarray(pre.transform(X.iloc[te]), dtype=np.float32)
        est = build(name, seed)
        if name == "TabNet":
            est.fit(Xtr, y[tr], max_epochs=60, patience=0,
                    batch_size=min(16, len(tr)), drop_last=False)
        else:
            est.fit(Xtr, y[tr])
        proba = np.asarray(est.predict_proba(Xte))
        pos = list(est.classes_).index(1) if 1 in list(est.classes_) else 1
        oof[te[0]] = proba[0, pos]
    return roc_auc_score(y, oof)


def fit_predict_auc(name, X, y, tr, te, num_cols):
    """Fit on train subset, return test AUROC. NaN if it can't fit/score."""
    try:
        pre = make_preprocessor(num_cols)
        Xtr = np.asarray(pre.fit_transform(X.iloc[tr]), dtype=np.float32)
        Xte = np.asarray(pre.transform(X.iloc[te]), dtype=np.float32)
        ytr = y[tr]
        est = build(name)
        if name == "TabNet":
            est.fit(Xtr, ytr, max_epochs=60, patience=0,
                    batch_size=min(16, len(ytr)), drop_last=False)
        else:
            est.fit(Xtr, ytr)
        proba = np.asarray(est.predict_proba(Xte))
        classes = list(est.classes_)
        pos = classes.index(1) if 1 in classes else 1
        return roc_auc_score(y[te], proba[:, pos])
    except Exception as e:
        return np.nan


def tabfm_zero_shot_auc(X, y, num_cols):
    """0-shot TabFM: no training context. Try empty fit; if unsupported, the
    model has no task information => uninformative predictions (AUROC ~0.5)."""
    import tabfm
    pre = make_preprocessor(num_cols)
    Xall = np.asarray(pre.fit_transform(X), dtype=np.float32)
    est = tabfm.TabFMClassifier(model=tabfm_model())
    try:
        empty_X = Xall[:0]; empty_y = y[:0]
        est.fit(empty_X, empty_y)
        proba = np.asarray(est.predict_proba(Xall))
        pos = list(est.classes_).index(1) if 1 in getattr(est, "classes_", []) else 1
        return roc_auc_score(y, proba[:, pos]), "empty-context fit"
    except Exception:
        # true 0-shot not supported by the API -> chance-level reference
        return 0.5, "not supported (reported as chance=0.5)"


def stratified_train(y, n, rng):
    idx = np.arange(len(y))
    per = n // 2
    tr = np.concatenate([rng.choice(idx[y == c], per, replace=False) for c in (0, 1)])
    return np.sort(tr)


def summarize(aucs):
    a = np.array([v for v in aucs if not np.isnan(v)], dtype=float)
    if len(a) == 0:
        return dict(auc_mean=None, n_valid=0)
    return dict(
        auc_mean=round(float(a.mean()), 4),
        auc_std=round(float(a.std()), 4),
        auc_95ci=[round(float(np.percentile(a, 2.5)), 4),
                  round(float(np.percentile(a, 97.5)), 4)],
        n_valid=int(len(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/fewshot")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--loocv_repeats", type=int, default=10,
                    help="repeats for the 35-shot LOOCV point; 0 = skip (merge from --merge_loocv)")
    ap.add_argument("--merge_loocv", default=None,
                    help="path to a prior fewshot_curve.json to copy the 35-shot LOOCV point from")
    ap.add_argument("--models", default="LR,RF,XGBoost,TabNet,TabFM")
    args = ap.parse_args()

    X, y, num_cols = load_data(args.data)
    models = [m.strip() for m in args.models.split(",")]
    print(f"[data] n={len(y)} feats={X.shape[1]} balance={np.bincount(y).tolist()} "
          f"| shots={SHOTS} repeats={args.repeats} models={models}", flush=True)

    results = {m: {} for m in models}
    notes = {}
    t0 = time.time()

    # 0-shot: TabFM only
    if 0 in SHOTS and "TabFM" in models:
        auc0, note = tabfm_zero_shot_auc(X, y, num_cols)
        results["TabFM"]["0"] = dict(auc_mean=round(float(auc0), 4), auc_std=0.0,
                                     auc_95ci=[round(float(auc0), 4)] * 2, n_valid=1)
        notes["tabfm_0shot"] = note
        print(f"[0-shot] TabFM AUROC={auc0:.4f} ({note})", flush=True)

    for n in [s for s in SHOTS if s > 0]:
        rng = np.random.default_rng(SEED + n)
        per_model = {m: [] for m in models}
        for r in range(args.repeats):
            tr = stratified_train(y, n, rng)
            te = np.setdiff1d(np.arange(len(y)), tr)
            for m in models:
                per_model[m].append(fit_predict_auc(m, X, y, tr, te, num_cols))
        for m in models:
            results[m][str(n)] = {**summarize(per_model[m]),
                                  "raw_auc": [round(float(v), 6) for v in per_model[m]]}
        row = "  ".join(f"{m}={results[m][str(n)].get('auc_mean')}" for m in models)
        print(f"[{n}-shot] ({time.time()-t0:.0f}s) {row}", flush=True)

    # 35-shot == LOOCV (train on n-1). Either recompute (loocv_repeats>0) or
    # reuse a prior run's LOOCV point via --merge_loocv (cheaper; LR/TabFM exact).
    n_loo = len(y) - 1
    timing = {}
    have_loo = False
    if args.loocv_repeats > 0:
        for m in models:
            tm = time.time()
            if m in DETERMINISTIC:
                a = loocv_auc(m, X, y, num_cols, SEED)
                results[m][str(n_loo)] = dict(auc_mean=round(float(a), 4), auc_std=0.0,
                                              auc_95ci=[round(float(a), 4)] * 2, n_valid=1,
                                              note="deterministic: single LOOCV run")
            else:
                aucs = [loocv_auc(m, X, y, num_cols, SEED + r) for r in range(args.loocv_repeats)]
                results[m][str(n_loo)] = summarize(aucs)
            timing[m] = round(time.time() - tm, 1)
            print(f"[{n_loo}-shot/LOOCV] ({time.time()-t0:.0f}s) {m}="
                  f"{results[m][str(n_loo)].get('auc_mean')}  [{timing[m]}s]", flush=True)
        have_loo = True
    elif args.merge_loocv:
        prior = json.load(open(args.merge_loocv))["results"]
        for m in models:
            if m in prior and str(n_loo) in prior[m]:
                results[m][str(n_loo)] = {**prior[m][str(n_loo)], "note": "merged from prior run"}
        have_loo = True
        print(f"[LOOCV] merged 35-shot point from {args.merge_loocv}", flush=True)

    all_shots = SHOTS + ([n_loo] if have_loo else [])
    out = dict(dataset="ADHD responder", n=int(len(y)), n_features=int(X.shape[1]),
               shots=all_shots, repeats=args.repeats, loocv_timing_sec=timing,
               total_runtime_sec=round(time.time() - t0, 1),
               shot_definition="N = total training examples, stratified N/2 per class",
               protocol="repeated stratified subsampling; test = remaining patients; "
                        "AUROC averaged over repeats",
               notes=notes, results=results)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "fewshot_curve.json", "w") as f:
        json.dump(out, f, indent=2)
    # tidy CSV for plotting
    rows = []
    for m in models:
        for n in all_shots:
            r = results[m].get(str(n))
            if r and r.get("auc_mean") is not None:
                rows.append(dict(model=m, shots=n, auc_mean=r["auc_mean"],
                                 auc_std=r.get("auc_std"),
                                 ci_lo=r.get("auc_95ci", [None, None])[0],
                                 ci_hi=r.get("auc_95ci", [None, None])[1]))
    pd.DataFrame(rows).to_csv(outdir / "fewshot_curve.csv", index=False)

    # raw per-repeat AUROC (paired across models by (shot, repeat)) for sig tests
    raw_rows = []
    for m in models:
        for n in [s for s in SHOTS if s > 0]:
            ra = results[m].get(str(n), {}).get("raw_auc", [])
            for r, v in enumerate(ra):
                raw_rows.append(dict(shot=n, repeat=r, model=m, auc=round(float(v), 6)))
    pd.DataFrame(raw_rows).to_csv(outdir / "fewshot_raw.csv", index=False)

    # paired Wilcoxon: TabFM vs each other model, at each shot (same splits => paired)
    sig = {}
    if "TabFM" in models:
        try:
            from scipy.stats import wilcoxon
            for n in [s for s in SHOTS if s > 0]:
                ra = np.array(results["TabFM"].get(str(n), {}).get("raw_auc", []))
                sig[str(n)] = {}
                for m in models:
                    if m == "TabFM":
                        continue
                    rb = np.array(results[m].get(str(n), {}).get("raw_auc", []))
                    if len(ra) != len(rb) or len(ra) == 0:
                        continue
                    diff = ra - rb
                    try:
                        _, p = wilcoxon(ra, rb)
                    except Exception:
                        p = None
                    sig[str(n)][f"TabFM_vs_{m}"] = dict(
                        mean_auc_diff=round(float(diff.mean()), 4),
                        median_auc_diff=round(float(np.median(diff)), 4),
                        tabfm_wins=int((diff > 0).sum()), n_repeats=int(len(diff)),
                        wilcoxon_p=(round(float(p), 5) if p is not None else None))
            with open(outdir / "fewshot_significance.json", "w") as f:
                json.dump(dict(test="paired Wilcoxon signed-rank, TabFM vs model, "
                               "per shot (paired by repeat/split)", results=sig), f, indent=2)
        except Exception as e:
            print(f"[sig] skipped: {e}", flush=True)

    print("\n===== N-SHOT LEARNING CURVE (mean ROC-AUC) =====", flush=True)
    tab = pd.DataFrame(rows).pivot(index="model", columns="shots", values="auc_mean")
    print(tab.to_string(), flush=True)
    print(f"\n[saved] {outdir/'fewshot_curve.json'}", flush=True)


if __name__ == "__main__":
    main()
