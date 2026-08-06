# Data-Efficient Prediction of Stimulant Response in Adult ADHD

Reproducible code for the study benchmarking **tabular foundation models,
conventional supervised learning, and large language models (LLMs)** at
predicting stimulant-medication response in medication-naive adults with ADHD,
with a focus on **data efficiency** — how each model's accuracy scales with the
number of labeled examples.

Companion repository to
[LLM_Bipolar_Prediction](https://github.com/gunahn/LLM_Bipolar_Prediction).

## Clinical motivation

Stimulants are first-line for adult ADHD, yet roughly 40% of patients do not
respond adequately, and clinicians have no validated way to predict response
before a trial. We ask how accurately, and how data-efficiently, modern
predictive systems can forecast response from routine baseline questionnaires.

## Cohort

- **n = 36** medication-naive adults with ADHD, prospectively followed after
  starting a stimulant (25 methylphenidate, 11 amphetamine).
- **Outcome:** binary treatment response at final follow-up, endpoint
  **CGI-I ≤ 2** (responder) vs > 2 (non-responder). Exactly balanced (18/18).
- **Features:** 40 baseline features from validated self-report instruments
  (ADHD-RS, ASR, BRIEF-A, SRS-2, Barkley DESR, MWQ, Q-LES-Q) plus demographics.
- This is the same cohort analyzed with classical statistics by
  **DiSalvo et al. (2026)**, *Ther Adv Psychopharmacol* 16:20451253261428807.
  The present work is a distinct, prediction-focused analysis (foundation-model
  ML + LLM benchmark, learning curves); feature importance is reported as
  convergent validation of DiSalvo's predictors, not as new discovery.

## Pipeline

```bash
# 0. install
pip install -r requirements.txt

# 1. full-training-set baselines (LOOCV / 5-fold): LR, RF, XGBoost, TabNet, TabPFN, TabFM
python src/responder_loocv.py --data "data/Prediction Data for GA (02.13.26).xls"          # LOOCV
python src/responder_loocv.py --data "data/...xls" --cv 5                                    # 5-fold

# 2. N-shot learning curve across all tabular models (0/2/8/16/24/35-shot; 35 = LOOCV)
python src/adhd_fewshot_curve.py --data "data/...xls" --repeats 100 --loocv_repeats 10

# 3. few-shot LLM sweep (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
python src/llm_responder.py --data "data/...xls" --model claude-opus-4-8

# 4. main learning-curve figure (Figure 1)
python src/04_analyze_and_figure.py

# 5. feature importance (Figure 2 + Table 3)
python src/supplementary/S_feature_importance.py --data "data/...xls"
```

## Models

| Role | Identifier |
|---|---|
| TabFM (tabular foundation model, in-context) | `google-research/tabfm` |
| TabPFN (tabular foundation model, in-context) | `tabpfn` (Hollmann et al.) |
| Supervised baselines | Logistic Regression, Random Forest, XGBoost, TabNet |
| Few-shot LLM predictor | Claude Opus 4 (`claude-opus-4-8`, Anthropic; accessed 2026) |

The model string recorded for the run is `claude-opus-4-8`; aliases roll over time and the exact dated snapshot served is returned in each
API response's `model` field.

## Headline results

- **Tabular models improve with more labeled examples**; TabFM has the steepest
  learning curve and the best endpoint, reaching **LOOCV AUC 0.77**
  (35-shot = leave-one-out, maximal context n−1), above XGBoost (0.64),
  LR (0.61), RF (0.59) and TabNet (0.53).
- **The LLM does not benefit from more shots.** It predicts above chance zero-shot
  (AUC ≈ 0.66) and stays flat across 0–24 in-context examples; the
  simple and clinician-guided prompts perform comparably.
- **Crossover:** the LLM leads when labels are scarce (≤ 8 examples); tabular
  foundation models overtake it once a few dozen labeled cases exist.
- **Feature importance** (bootstrap stability selection) converges on quality of
  life (Q-LES-Q) and ASR intrusive/thought problems — the predictors reported by
  DiSalvo et al. (2026), providing methodologically independent confirmation.

## Layout

```
src/responder_loocv.py           shared data loader + preprocessing; LOOCV/5-fold baselines (6 models)
src/serialize.py                 patient row -> clinical narrative (LLM input)
src/prompts.json                 exact simple / clinician-guided prompts (loaded by llm_responder.py)
src/adhd_fewshot_curve.py        N-shot learning curve across all tabular models
src/llm_responder.py             few-shot LLM predictor (Claude); 0/2/4/8/16/24-shot, 2 prompts, 15 seeds
src/04_analyze_and_figure.py     main learning-curve figure (Fig 1)
src/models/                      model-baseline notes (see src/models/README.md)
src/supplementary/               feature importance (Fig 2 / Table 3), LR top-10,
                                 LLM-vs-ML significance (see src/supplementary/README.md)
src/*.sbatch                     SLURM submission scripts (cluster runs)
data/                            raw workbook (not committed — human-subjects data)
results/                         summary CSV/JSON + figures (committed for reference)
results/figures/                 manuscript figures
APPENDIX_prompt_and_exemplars.md exact LLM prompts + serialization (verbatim)
```

## Notes / caveats

- Single-site sample, n = 36; all estimates carry wide confidence intervals.
  Prospective, multi-site validation is required before any clinical use.
  **Research code — not a medical device.**
- The TabFM results were obtained from a pretrained checkpoint applied without
  task-specific fine-tuning; foundation-model performance can vary with checkpoint
  and preprocessing.
- A prospective **clinician** prediction arm (human experts on the identical task)
  is part of the study design and will be added separately.
- The raw clinical data are **not** included in this repository (human-subjects
  data); place the workbook under `data/` to run the pipeline.

## Prompt & serialization specification

The exact system prompts (simple and clinician-guided), output instruction, and
patient-record serialization are reproduced verbatim in
[APPENDIX_prompt_and_exemplars.md](APPENDIX_prompt_and_exemplars.md).
