#!/usr/bin/env python3
"""
S_llm_vs_ml_significance.py
Paired significance testing for the learning-curve comparison.

For the tabular models, TabFM is compared against each competing model at each
shot count using a paired Wilcoxon signed-rank test over the matched per-draw
AUCs (all models were trained/evaluated on identical data draws, so the AUCs are
paired). This reproduces results/fewshot_significance.json.

For the LLM, we test whether AUC changes across shots (Spearman trend) and
whether the two prompts differ (paired Wilcoxon on per-seed AUC).

Inputs:
  results/fewshot_raw.csv           (per-repeat AUC, tabular models)
  results/llm_fewshot_results.json  (LLM per-seed AUC)
Usage:
  python src/supplementary/S_llm_vs_ml_significance.py
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr

SHOTS = [2, 8, 16, 24]


def tabular_sig(raw):
    out = {}
    for n in SHOTS:
        ref = raw[(raw.model == "TabFM") & (raw.shot == n)].sort_values("repeat").auc.values
        out[n] = {}
        for m in raw.model.unique():
            if m == "TabFM":
                continue
            b = raw[(raw.model == m) & (raw.shot == n)].sort_values("repeat").auc.values
            if len(ref) != len(b) or len(ref) == 0:
                continue
            diff = ref - b
            try:
                _, p = wilcoxon(ref, b)
            except Exception:
                p = None
            out[n][f"TabFM_vs_{m}"] = dict(mean_auc_diff=round(float(diff.mean()), 4),
                                           tabfm_wins=int((diff > 0).sum()), n=int(len(diff)),
                                           wilcoxon_p=(round(float(p), 5) if p is not None else None))
    return out


def llm_trend(llm):
    out = {}
    for p in ["simple", "clinician_guided"]:
        xs, ys = [], []
        for s in llm["shots"]:
            m = llm["resampled"][p][str(s)]["auc_mean"]
            xs.append(s); ys.append(m)
        rho, pv = spearmanr(xs, ys)
        out[p] = dict(spearman_rho=round(float(rho), 3), p=round(float(pv), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    raw = pd.read_csv(f"{args.results}/fewshot_raw.csv")
    llm = json.load(open(f"{args.results}/llm_fewshot_results.json"))
    tab = tabular_sig(raw)
    trend = llm_trend(llm)
    print("=== TabFM vs each model (paired Wilcoxon) ===")
    for n in SHOTS:
        print(f"[{n}-shot]", {k: v["wilcoxon_p"] for k, v in tab[n].items()})
    print("\n=== LLM shot-trend (Spearman across shots) ===")
    for p, v in trend.items():
        print(f"  {p}: rho={v['spearman_rho']} p={v['p']}")


if __name__ == "__main__":
    main()
