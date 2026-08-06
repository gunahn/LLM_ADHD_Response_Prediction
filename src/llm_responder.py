#!/usr/bin/env python3
"""
llm_responder.py
Few-shot LLM prediction of stimulant response (Responder vs Non-responder) in
adult ADHD, using Claude (Anthropic).

This reproduces the exact experiment behind results/llm_fewshot_results.json:

  * Model:   Claude Opus 4. The exact model string recorded for the run that
             produced results/llm_fewshot_results.json is `claude-opus-4-8`
             (this is the default below, for faithful reproduction). Model
             aliases roll over time; the dated snapshot actually served is
             returned in each API response's `model` field.
  * Prompts: two designs compared head-to-head -- "simple" (concise task
             description) and "clinician_guided" (structured clinical briefing).
             Both are reproduced verbatim in APPENDIX_prompt_and_exemplars.md.
  * Serialization: each patient row -> full-instrument clinical narrative
             (src/serialize.py).
  * Shots:   0, 2, 4, 8, 16, 24 in-context labeled exemplars.
  * Design:  15 balanced resampled draws (seeds) per (prompt, shot). For each
             seed, a balanced test set of 8 patients (4 responders + 4
             non-responders) is held out; exemplars are drawn from the
             remaining patients as NESTED sets (the 4-shot set extends the
             2-shot set, etc.) so the shot effect is paired within a seed.
             Exemplars and test cases are always disjoint.
  * Scoring: the model returns an integer response probability per test patient;
             ROC-AUC is computed per seed and averaged across the 15 seeds
             (mean +/- SD).

Usage:
  export ANTHROPIC_API_KEY=sk-...
  python src/llm_responder.py --data "data/Prediction Data for GA (02.13.26).xls" \
      --model claude-opus-4-8 --out results

Requires: anthropic, pandas, numpy, scikit-learn, and src/serialize.py.
"""
import os, sys, json, argparse, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from serialize import verbalize

SHOTS = [0, 2, 4, 8, 16, 24]
N_SEEDS = 15
TEST_PER_CLASS = 4           # balanced test set = 8 (4 responder + 4 non-responder)
PROMPTS = ["simple", "clinician_guided"]

# --------------------------------------------------------------------------- #
#  Prompts: loaded verbatim from src/prompts.json (the exact strings used in    #
#  the study). Also reproduced in APPENDIX_prompt_and_exemplars.md.             #
# --------------------------------------------------------------------------- #
_PROMPTS = json.load(open(Path(__file__).parent / "prompts.json"))
PROMPT_TEXT = {"simple": _PROMPTS["simple"], "clinician_guided": _PROMPTS["clinician_guided"]}
OUTPUT_INSTRUCTION = _PROMPTS["output_instruction"]


# --------------------------------------------------------------------------- #
#  Data + resampling design                                                    #
# --------------------------------------------------------------------------- #
def load(path):
    df = pd.read_excel(path, sheet_name="Data")
    y = df["responder"].astype(int).values
    return df.reset_index(drop=True), y


def build_seed_design(y, n_seeds=N_SEEDS, seed=0):
    """For each seed: a balanced held-out test set (4R+4N) and NESTED exemplar
    draws for each shot count from the remaining patients."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    designs = []
    for s in range(n_seeds):
        te = np.concatenate([rng.choice(pos, TEST_PER_CLASS, replace=False),
                             rng.choice(neg, TEST_PER_CLASS, replace=False)])
        pool_pos = [i for i in pos if i not in te]
        pool_neg = [i for i in neg if i not in te]
        rng.shuffle(pool_pos); rng.shuffle(pool_neg)
        ex = {}
        for shot in SHOTS:
            k = shot // 2                       # k per class (nested)
            ex[shot] = _interleave(pool_pos[:k], pool_neg[:k])
        designs.append({"test": te.tolist(), "ex": ex})
    return designs


def _interleave(a, b):
    out = []
    for x, y_ in zip(a, b):
        out += [x, y_]
    return out


def build_user_message(df, target_idx, exemplar_idx, y):
    """Assemble the user message exactly as in the study: optional labeled
    exemplars, then the query patient under a 'NOW CLASSIFY THIS PATIENT:'
    header (present at every shot count, including 0-shot), then the output
    instruction. Verified byte-for-byte against the recorded calls."""
    parts = []
    if exemplar_idx:
        parts.append("LABELED EXAMPLE PATIENTS:\n\n")
        for i in exemplar_idx:
            outcome = "Responder" if y[i] == 1 else "Non-responder"
            parts.append("PATIENT:\n" + verbalize(df.iloc[i]) + f"\nOUTCOME: {outcome}\n\n")
    parts.append("NOW CLASSIFY THIS PATIENT:\n\n" + verbalize(df.iloc[target_idx]))
    return "".join(parts) + "\n\n" + OUTPUT_INSTRUCTION


# --------------------------------------------------------------------------- #
#  LLM call + parsing                                                          #
# --------------------------------------------------------------------------- #
def call_llm(client, model, system, user, max_retries=4):
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=100, system=system,
                messages=[{"role": "user", "content": user}])
            txt = msg.content[0].text
            m = re.search(r'"response_probability"\s*:\s*(\d+)', txt)
            if m:
                return int(m.group(1)) / 100.0
        except Exception:
            time.sleep(2 ** attempt)
    return None


def run(df, y, client, model, out, workers=8):
    designs = build_seed_design(y)
    results = {p: {} for p in PROMPTS}
    for prompt in PROMPTS:
        system = PROMPT_TEXT[prompt]   # output instruction is appended to the user message
        for shot in SHOTS:
            per_seed_auc = []
            for d in designs:
                jobs = [(t, build_user_message(df, t, d["ex"][shot], y)) for t in d["test"]]
                probs = {}
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(call_llm, client, model, system, u): t for t, u in jobs}
                    for f in as_completed(futs):
                        probs[futs[f]] = f.result()
                yt = [y[t] for t in d["test"] if probs.get(t) is not None]
                pp = [probs[t] for t in d["test"] if probs.get(t) is not None]
                if len(set(yt)) == 2:
                    per_seed_auc.append(roc_auc_score(yt, pp))
            results[prompt][str(shot)] = dict(
                auc_mean=float(np.mean(per_seed_auc)) if per_seed_auc else None,
                auc_std=float(np.std(per_seed_auc)) if per_seed_auc else None,
                n_seeds=len(per_seed_auc), per_seed_auc=[round(a, 4) for a in per_seed_auc])
            print(f"[{prompt} {shot}-shot] AUC="
                  f"{results[prompt][str(shot)]['auc_mean']}", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    payload = dict(model=model, shots=SHOTS, prompts=PROMPTS,
                   design=f"{N_SEEDS} balanced resampled draws per (prompt,shot); "
                          f"test={TEST_PER_CLASS*2} ({TEST_PER_CLASS}R+{TEST_PER_CLASS}N), "
                          "nested exemplar draws",
                   resampled=results)
    with open(f"{out}/llm_fewshot_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    rows = [dict(prompt=p, shots=int(s), auc_mean=results[p][s]["auc_mean"],
                 auc_std=results[p][s]["auc_std"], n_seeds=results[p][s]["n_seeds"])
            for p in PROMPTS for s in results[p]]
    pd.DataFrame(rows).to_csv(f"{out}/llm_fewshot_results.csv", index=False)
    print(f"[saved] {out}/llm_fewshot_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="Anthropic model id; default reproduces the recorded run "
                         "(the dated snapshot served is returned in each response's model field)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    df, y = load(args.data)
    print(f"[data] n={len(y)} balance={np.bincount(y).tolist()} "
          f"shots={SHOTS} seeds={N_SEEDS} prompts={PROMPTS}", flush=True)
    run(df, y, client, args.model, args.out, args.workers)


if __name__ == "__main__":
    main()
