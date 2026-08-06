# Model baselines

All six tabular predictors are defined in two shared modules rather than as
separate files, because they share the same preprocessing and evaluation harness:

| Model | Where it is defined | Notes |
|---|---|---|
| Logistic Regression (LR) | `src/responder_loocv.py` `get_models()` / `src/adhd_fewshot_curve.py` `build()` | standardized features |
| Random Forest (RF) | same | 400–500 trees |
| XGBoost | same | depth-3, lr 0.05 |
| TabNet | same | deep tabular net, 60–100 epochs, CPU |
| TabPFN | `src/responder_loocv.py` `get_models()` | in-context tabular foundation model (Hollmann et al.) |
| TabFM | `src/adhd_fewshot_curve.py` `tabfm_model()` / `src/responder_loocv.py` | Google tabular foundation model; in-context, single forward pass |

**Foundation-model weights.** TabFM and TabPFN load pretrained weights. On an
offline compute node, prefetch them on a login node first (see the SLURM scripts
and the bipolar-repo prefetch pattern). TabFM: `google-research/tabfm`
(`tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")`). TabPFN:
`pip install tabpfn`.

The N-shot **learning curve** across all models is produced by
`src/adhd_fewshot_curve.py`; the full-training-set LOOCV / 5-fold baselines by
`src/responder_loocv.py`.
