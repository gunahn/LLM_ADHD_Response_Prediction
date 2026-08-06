#!/usr/bin/env python3
"""
S_feature_importance.py
Clinician-facing feature-importance analysis (manuscript Figure 2 + Table 3).

Two complementary views, appropriate for p (40 features) > n (36 patients) with
highly correlated questionnaire scales:

  (a) Univariate discrimination: each feature's single-feature ROC-AUC, direction
      of effect (higher -> responder vs higher -> non-responder), Mann-Whitney U
      p-value, Cliff's delta effect size, and Benjamini-Hochberg (FDR) q-value.

  (b) Bootstrap stability selection (Meinshausen & Buhlmann, 2010): refit an
      L1-penalized logistic regression over 1,000 bootstrap resamples and record
      each feature's selection frequency (fraction of resamples with non-zero
      coefficient). Robust importance under collinearity / small n.

Outputs:
  results/feature_importance_univariate.csv
  results/feature_importance_stability.csv
  results/feature_importance_clinician_table.csv
  results/figures/feature_importance_2panel.png

Usage:
  python src/supplementary/S_feature_importance.py --data "data/Prediction Data for GA (02.13.26).xls"
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

SEED = 42
ID = ["study_id", "clinic_id", "mit_id"]

INSTR = {"age":"Demographic","gender":"Demographic","adhd_rs_total":"ADHD-RS","adhd_rs_inatt":"ADHD-RS",
 "adhd_rs_hyper":"ADHD-RS","aggret":"ASR","anxdept":"ASR","attent":"ASR","intrut":"ASR","delnqt":"ASR",
 "somat":"ASR","thougt":"ASR","withdt":"ASR","extert":"ASR","intert":"ASR","totalt":"ASR","asr_aaa":"ASR",
 "inhibit_t":"BRIEF-A","shift_t":"BRIEF-A","emocont_t":"BRIEF-A","selfmon_t":"BRIEF-A","initiate_t":"BRIEF-A",
 "workmem_t":"BRIEF-A","planorg_t":"BRIEF-A","taskmon_t":"BRIEF-A","orgmat_t":"BRIEF-A","bri_t":"BRIEF-A",
 "mi_t":"BRIEF-A","gec_t":"BRIEF-A","aware_t":"SRS-2","cog_t":"SRS-2","comm_t":"SRS-2","motiv_t":"SRS-2",
 "rrb_t":"SRS-2","srs_total_t":"SRS-2","mwq_total":"MWQ","desr_total":"DERS","q_les_q_raw":"Q-LES-Q"}
LABEL = {"age":"Age","gender":"Sex (female)","adhd_rs_total":"ADHD-RS total","adhd_rs_inatt":"ADHD-RS inattention",
 "adhd_rs_hyper":"ADHD-RS hyperactivity","aggret":"Aggressive behavior","anxdept":"Anxious/depressed",
 "attent":"Attention problems","intrut":"Intrusive","delnqt":"Rule-breaking","somat":"Somatic complaints",
 "thougt":"Thought problems","withdt":"Withdrawn","extert":"Externalizing","intert":"Internalizing",
 "totalt":"Total problems","asr_aaa":"AAA composite","inhibit_t":"Inhibit","shift_t":"Shift",
 "emocont_t":"Emotional control","selfmon_t":"Self-monitor","initiate_t":"Initiate","workmem_t":"Working memory",
 "planorg_t":"Plan/organize","taskmon_t":"Task-monitor","orgmat_t":"Organization of materials",
 "bri_t":"Behavioral Regulation Index","mi_t":"Metacognition Index","gec_t":"Global Executive Composite",
 "aware_t":"Social awareness","cog_t":"Social cognition","comm_t":"Social communication",
 "motiv_t":"Social motivation","rrb_t":"Restricted/repetitive","srs_total_t":"SRS-2 total",
 "mwq_total":"Mind-wandering","desr_total":"Emotional dysregulation","q_les_q_raw":"Quality of life"}


def load(path):
    df = pd.read_excel(path, sheet_name="Data")
    y = df["responder"].astype(int).values
    drop = ID + ["responder", "race", "stim_type"]
    feats = [c for c in df.columns if c not in drop]
    X = df[feats].copy()
    X["gender"] = (X["gender"] == 2).astype(int)
    return X.astype(float), y, feats


def univariate(X, y, feats):
    rows = []
    for c in feats:
        v = X[c].values
        a = roc_auc_score(y, v)
        r, n = v[y == 1], v[y == 0]
        U, p = mannwhitneyu(r, n)
        delta = 2 * U / (len(r) * len(n)) - 1
        rows.append(dict(feature=c, label=LABEL[c], instrument=INSTR[c],
                         auc=round(max(a, 1 - a), 3),
                         direction=("responder" if a >= 0.5 else "non-responder"),
                         resp_mean=round(r.mean(), 1), resp_sd=round(r.std(), 1),
                         nonresp_mean=round(n.mean(), 1), nonresp_sd=round(n.std(), 1),
                         mwu_p=round(p, 4), cliffs_delta=round(delta, 3)))
    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    df["mwu_q"] = np.round(multipletests(df["mwu_p"].values, method="fdr_bh")[1], 3)
    return df


def stability(X, y, feats, B=1000, C=0.3):
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    sel = np.zeros(X.shape[1]); nfit = 0
    for _ in range(B):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000,
                               random_state=int(rng.integers(1e6))).fit(Xs[idx], y[idx])
        sel += (m.coef_[0] != 0).astype(int); nfit += 1
    full = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000,
                              random_state=SEED).fit(Xs, y)
    coef = pd.Series(full.coef_[0], index=X.columns)
    df = pd.DataFrame({"feature": X.columns, "label": [LABEL[c] for c in X.columns],
                       "instrument": [INSTR[c] for c in X.columns],
                       "selection_freq": np.round(sel / nfit, 3),
                       "direction": ["responder" if coef[c] > 0 else ("non-responder" if coef[c] < 0 else "-")
                                     for c in X.columns]})
    return df.sort_values("selection_freq", ascending=False).reset_index(drop=True)


def figure(uni, stab, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C_RESP, C_NON, GREY = "#2c7fb8", "#d95f0e", "#8a8f98"
    order = ["ASR","BRIEF-A","SRS-2","ADHD-RS","MWQ","DERS","Q-LES-Q","Demographic"]
    show = uni[uni.auc >= 0.58].copy()
    show["io"] = show.instrument.map({k: i for i, k in enumerate(order)})
    show = show.sort_values(["io", "auc"], ascending=[True, True]).reset_index(drop=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 6.6))
    for i, r in show.iterrows():
        col = C_RESP if r.direction == "responder" else C_NON
        axA.hlines(i, 0.5, r.auc, color=col, lw=1.6); axA.plot(r.auc, i, "o", color=col, ms=7, mec="white", mew=0.6)
    axA.set_yticks(range(len(show))); axA.set_yticklabels(show.label, fontsize=7.5)
    axA.axvline(0.5, color=GREY, lw=0.8); axA.set_xlim(0.5, 0.80)
    axA.set_xlabel("Single-feature ROC AUC"); axA.set_title("a  Univariate discrimination by scale", loc="left", fontweight="bold")
    axA.legend(handles=[Line2D([0],[0],color=C_RESP,marker="o",lw=1.6,label="higher to responder"),
                        Line2D([0],[0],color=C_NON,marker="o",lw=1.6,label="higher to non-responder")],
               loc="lower right", frameon=False, fontsize=7)
    top = stab.head(14).iloc[::-1].reset_index(drop=True)
    cm = {"responder": C_RESP, "non-responder": C_NON}
    for i, r in top.iterrows():
        axB.barh(i, r.selection_freq, color=cm.get(r.direction, GREY), edgecolor="black", lw=0.4, height=0.66)
    axB.set_yticks(range(len(top))); axB.set_yticklabels(top.label, fontsize=7.5)
    axB.axvline(0.5, color="#c23b3b", ls="--", lw=1); axB.set_xlim(0, 0.72)
    axB.set_xlabel("Bootstrap L1 selection frequency (1000 resamples)")
    axB.set_title("b  Multivariate stability selection", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    X, y, feats = load(args.data)
    uni = univariate(X, y, feats)
    stab = stability(X, y, feats)
    Path(args.out + "/figures").mkdir(parents=True, exist_ok=True)
    uni.to_csv(f"{args.out}/feature_importance_univariate.csv", index=False)
    stab.to_csv(f"{args.out}/feature_importance_stability.csv", index=False)
    # combined clinician table
    ub = uni.set_index("label")
    rows = []
    for _, r in stab.head(12).iterrows():
        u = ub.loc[r.label]
        rows.append(dict(Scale=r.label, Instrument=r.instrument,
                         Direction=("higher -> responder" if r.direction == "responder" else "higher -> non-responder"),
                         Univariate_AUC=u.auc, Cliffs_delta=u.cliffs_delta, MWU_p=u.mwu_p, FDR_q=u.mwu_q,
                         Bootstrap_selection_pct=f"{int(round(r.selection_freq*100))}%"))
    pd.DataFrame(rows).to_csv(f"{args.out}/feature_importance_clinician_table.csv", index=False)
    figure(uni, stab, f"{args.out}/figures/feature_importance_2panel.png")
    print("top univariate:\n", uni.head(4).to_string(index=False))
    print("\ntop stability:\n", stab.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
