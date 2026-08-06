#!/usr/bin/env python3
"""
04_analyze_and_figure.py
Build the main two-panel learning-curve figure (manuscript Figure 1):
  (a) Tabular models: ROC-AUC vs number of labeled training examples (shots);
      the 35-shot point is leave-one-out cross-validation (max context, n-1).
  (b) LLM: ROC-AUC (mean +/- SEM) vs number of in-context examples, for the
      simple and clinician-guided prompts.

Inputs:
  results/fewshot_curve.csv         (tabular ML learning curve; from adhd_fewshot_curve.py)
  results/llm_fewshot_results.json  (LLM sweep; from the LLM arm)
Output:
  results/figures/fig_learning_curves.png

Usage:
  python src/04_analyze_and_figure.py
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ML_MODELS = ["TabFM", "XGBoost", "RF", "LR", "TabNet"]
COL = {"TabFM":"#d62728","XGBoost":"#1f77b4","RF":"#2ca02c","LR":"#ff7f0e","TabNet":"#9467bd"}
MK  = {"TabFM":"o","XGBoost":"s","RF":"^","LR":"D","TabNet":"v"}
GREY = "#8a8f98"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--ml_repeats", type=int, default=100)
    args = ap.parse_args()
    mlc = pd.read_csv(f"{args.results}/fewshot_curve.csv")
    llm = json.load(open(f"{args.results}/llm_fewshot_results.json"))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 5.4), gridspec_kw={"width_ratios":[1.15,1]})
    for m in ML_MODELS:
        d = mlc[mlc.model == m].sort_values("shots")
        sem = d.auc_std.values / np.sqrt(args.ml_repeats)
        lw = 2.6 if m == "TabFM" else 1.3
        axA.errorbar(d.shots, d.auc_mean, yerr=sem, marker=MK[m], color=COL[m], lw=lw,
                     ms=7.5 if m == "TabFM" else 5, label=m, capsize=2, elinewidth=0.9,
                     markeredgecolor="white", markeredgewidth=0.6, zorder=6 if m=="TabFM" else 3)
    axA.axhline(0.5, color=GREY, ls=":", lw=1)
    axA.axvline(35, color="#cccccc", ls="--", lw=0.8)
    axA.annotate("35 = leave-one-out\n(max context, n-1)", xy=(35, 0.765), xytext=(25, 0.805),
                 fontsize=7, color="#d62728", ha="center", arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    axA.set_xticks([0,2,8,16,24,35]); axA.set_ylim(0.44, 0.84)
    axA.set_xlabel("Training examples (shots)"); axA.set_ylabel("Test ROC-AUC")
    axA.set_title("a  Tabular models: learning curves", loc="left", fontweight="bold")
    axA.legend(loc="upper left", frameon=False, fontsize=8)

    LP = {"simple":"#5b8db8","clinician_guided":"#e0902a"}
    LAB = {"simple":"Simple prompt","clinician_guided":"Clinician-guided prompt"}
    shots = llm["shots"]
    for p in ["simple", "clinician_guided"]:
        ys = np.array([llm["resampled"][p][str(s)]["auc_mean"] for s in shots])
        ns = np.array([llm["resampled"][p][str(s)]["n_seeds"] for s in shots])
        sem = np.array([llm["resampled"][p][str(s)]["auc_std"] for s in shots]) / np.sqrt(ns)
        axB.errorbar(shots, ys, yerr=sem, marker="o", color=LP[p], lw=1.9, ms=5.5, capsize=2,
                     elinewidth=0.9, label=LAB[p], markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    axB.axhline(0.5, color=GREY, ls=":", lw=1)
    axB.set_xticks(shots); axB.set_ylim(0.44, 0.84)
    axB.set_xlabel("In-context examples (shots)"); axB.set_ylabel("Test ROC-AUC")
    axB.set_title("b  LLM: prompt x shots", loc="left", fontweight="bold")
    axB.legend(loc="upper left", frameon=False, fontsize=8)

    Path(f"{args.results}/figures").mkdir(parents=True, exist_ok=True)
    out = f"{args.results}/figures/fig_learning_curves.png"
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
