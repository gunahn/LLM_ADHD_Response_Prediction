# Supplementary analyses

| Script | Produces |
|---|---|
| `S_feature_importance.py` | Figure 2 (univariate lollipops + bootstrap L1 stability selection) and Table 3 (clinician-facing feature table). Outputs `results/feature_importance_*.csv` and `results/figures/feature_importance_2panel.png`. |
| `S_lr_top10.py` | Logistic-regression top-10 |standardized coefficient| ranking (LOOCV-averaged), illustrative interpretability. |
| `S_llm_vs_ml_significance.py` | Paired Wilcoxon (TabFM vs each model, per shot) and LLM shot-trend tests. Reproduces `results/fewshot_significance.json`. |

**Feature-importance method (Figure 2).** Two complementary views for a setting
where features (40) outnumber patients (36) and are highly correlated:
- *Univariate:* single-feature ROC-AUC + direction of effect; group differences
  by Mann–Whitney U; effect size by Cliff's delta; Benjamini–Hochberg (FDR)
  correction across all 40 features.
- *Multivariate stability selection* (Meinshausen & Bühlmann, 2010): selection
  frequency of each feature across 1,000 bootstrap L1-logistic refits.

Both views converge on quality of life (Q-LES-Q) and ASR intrusive/thought
problems, matching the predictors reported for this cohort by DiSalvo et al.
(2026).
