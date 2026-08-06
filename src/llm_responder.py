#!/usr/bin/env python3
"""
LLM zero-shot & few-shot prediction of stimulant response (Responder vs
Non-responder) in adult ADHD, using Claude (claude-opus-4-8).

Each patient row is verbalized into a clinical vignette (matching the format
supplied by the user), then Claude returns a structured JSON with:
  - response_probability : integer 0-100 (P[Responder])
  - label                : "Responder" | "Non-responder"
The probability is used to compute pooled ROC-AUC.

Evaluation:
  - zero-shot : no exemplars, evaluated on all 36 patients
  - few-shot  : 10 labeled exemplars (5 Responder + 5 Non-responder) in context,
                evaluated on the 26 patients NOT used as exemplars (leakage-safe)
  - zero-shot AUC is also reported on those same 26 for a matched comparison
"""
import os, json, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, f1_score, confusion_matrix)
import anthropic

MODEL = "claude-opus-4-8"

GENDER = {1: "Male", 2: "Female"}
RACE = {1: "White/Caucasian", 2: "Black/African American",
        3: "American Indian/Alaskan Native", 4: "Asian",
        5: "More than one race", 6: "Unknown/Not reported"}

# ---- banding rules inferred from the user's example verbalizations ----
def tband(t):
    if t < 40:  return "below average"
    if t < 65:  return "within normal limits"
    if t < 70:  return "mildly impaired"
    return "significantly impaired"

def rs_sev(x):
    if x < 24:  return "mild"
    if x < 40:  return "moderate"
    return "severe"

def qles_band(x):
    if x < 40:  return "low"
    if x < 60:  return "moderate"
    return "high"

def verbalize(r):
    return (
        f"This {int(r.age)}-year-old {GENDER.get(int(r.gender),'Unknown')} patient "
        f"({RACE.get(int(r.race),'Unknown/Not reported')}) was referred for ADHD "
        f"evaluation and stimulant treatment.\n\n"
        f"Core ADHD symptoms are {rs_sev(r.adhd_rs_total)} (ADHD-RS Total={int(r.adhd_rs_total)}).\n\n"
        f"Behavioral and emotional self-report (ASR): externalizing problems is "
        f"{tband(r.extert)} (T={int(r.extert)}), aggressive behavior is {tband(r.aggret)} "
        f"(T={int(r.aggret)}), intrusive behavior is {tband(r.intrut)} (T={int(r.intrut)}), "
        f"rule-breaking behavior is {tband(r.delnqt)} (T={int(r.delnqt)}), "
        f"AAA-composite sum={int(r.asr_aaa)}.\n\n"
        f"Executive function (BRIEF-A): global composite (GEC) is {tband(r.gec_t)} "
        f"(T={int(r.gec_t)}), task monitoring is {tband(r.taskmon_t)} (T={int(r.taskmon_t)}), "
        f"organization of materials is {tband(r.orgmat_t)} (T={int(r.orgmat_t)}).\n\n"
        f"Additional measures: quality of life is {qles_band(r.q_les_q_raw)} "
        f"(Q-LES-Q={int(r.q_les_q_raw)})."
    )

SYSTEM_PROMPT = """\
You are a clinical expert predicting stimulant medication response in adult ADHD patients.

Study context:
Thirty-six medication-naive adults meeting DSM-5 criteria for ADHD were drawn from a previously conducted neuroimaging and clinical study (Hung et al., 2024). Participants with mild comorbid anxiety or depression were not excluded provided these did not require urgent clinical attention. Following baseline assessment, participants received methylphenidate (n=25) or amphetamine (n=9) at therapeutic doses and were followed naturalistically for a mean of 116.1 +/- 77.9 days. Treatment response was defined by the Clinical Global Impression-Improvement scale (CGI-I) at final follow-up: responders were classified as CGI-I <= 2 (much or very much improved) and non-responders as CGI-I > 2.

Scale reference:
- ADHD-RS: core ADHD symptom severity (inattention + hyperactivity/impulsivity)
- BRIEF-A: executive function deficits (GEC = global composite; higher = worse)
- ASR: behavioral and emotional problems (T-scores; mean=50, SD=10)
- SRS: social responsiveness and autism-spectrum traits
- MWQ: mind wandering tendency
- DESR: emotional dysregulation
- Q-LES-Q: quality of life (higher = better)
- T-score interpretation: >=65 = borderline elevated [borderline], >=70 = clinically elevated [high]

Study the labeled examples carefully, then classify the final patient.

Output a JSON object with two fields:
- "response_probability": an integer from 0 to 100 giving your probability that this
  patient is a Responder (CGI-I <= 2) to stimulant treatment.
- "label": "Responder" if response_probability >= 50, otherwise "Non-responder"."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "response_probability": {"type": "integer"},
        "label": {"type": "string", "enum": ["Responder", "Non-responder"]},
    },
    "required": ["response_probability", "label"],
    "additionalProperties": False,
}

# exemplar identity keys: (age, gender, adhd_rs_total, gec_t, taskmon_t, orgmat_t)
EXEMPLARS = {  # -> label
    (33, 2, 38, 62, 74, 63): "Responder",
    (40, 2, 41, 76, 85, 86): "Responder",
    (24, 2, 30, 62, 72, 58): "Responder",
    (31, 1, 67, 84, 93, 74): "Responder",
    (30, 1, 62, 77, 93, 83): "Responder",
    (22, 1, 57, 71, 68, 75): "Non-responder",
    (30, 2, 38, 62, 51, 45): "Non-responder",
    (38, 2, 14, 42, 51, 37): "Non-responder",
    (27, 1, 48, 68, 90, 39): "Non-responder",
    (26, 1, 40, 56, 59, 56): "Non-responder",
}

def load(path):
    df = pd.read_excel(path, sheet_name="Data")
    return df

def match_exemplars(df):
    idx = {}
    for _, r in df.iterrows():
        key = (int(r.age), int(r.gender), int(r.adhd_rs_total),
               int(r.gec_t), int(r.taskmon_t), int(r.orgmat_t))
        if key in EXEMPLARS:
            idx[r.name] = EXEMPLARS[key]
    return idx  # dataframe-index -> label


def build_fewshot_block(df, exemplar_idx):
    resp, nonresp = [], []
    for i, lab in exemplar_idx.items():
        (resp if lab == "Responder" else nonresp).append(verbalize(df.loc[i]))
    parts = ["Here are labeled example patients.\n", "=== RESPONDERS ==="]
    for v in resp:
        parts.append(v + "\nLabel: Responder\n")
    parts.append("=== NON-RESPONDERS ===")
    for v in nonresp:
        parts.append(v + "\nLabel: Non-responder\n")
    return "\n".join(parts)


def classify(client, target_text, fewshot_block=None):
    content = []
    if fewshot_block:
        content.append({"type": "text", "text": fewshot_block,
                        "cache_control": {"type": "ephemeral"}})  # cache exemplars
    content.append({"type": "text",
                    "text": "Now classify the following patient:\n\n" + target_text})
    resp = client.messages.create(
        model=MODEL, max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
    )
    txt = next(b.text for b in resp.content if b.type == "text")
    obj = json.loads(txt)
    p = max(0, min(100, int(obj["response_probability"]))) / 100.0
    return p, obj["label"]


def run(client, df, test_idx, fewshot_block, tag):
    def one(i):
        for attempt in range(4):
            try:
                p, lab = classify(client, verbalize(df.loc[i]), fewshot_block)
                return i, p, lab
            except Exception as e:
                if attempt == 3:
                    return i, None, f"ERROR:{type(e).__name__}"
                time.sleep(2 * (attempt + 1))
    results = {}
    # warm the cache with one call first, then parallelize the rest
    first = test_idx[0]
    results[first] = one(first)[1:]
    print(f"[{tag}] warmed cache on idx {first}", flush=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(one, i): i for i in test_idx[1:]}
        for k, f in enumerate(as_completed(futs), 2):
            i, p, lab = f.result()
            results[i] = (p, lab)
            print(f"[{tag}] {k}/{len(test_idx)} idx={i} p={p} {lab}", flush=True)
    return results


def score(df, results, tag):
    idx = [i for i in results if results[i][0] is not None]
    y = df.loc[idx, "responder"].astype(int).values
    proba = np.array([results[i][0] for i in idx])
    pred = (proba >= 0.5).astype(int)
    out = dict(
        eval=tag, n=len(idx),
        roc_auc=round(roc_auc_score(y, proba), 4),
        accuracy=round(accuracy_score(y, pred), 4),
        balanced_accuracy=round(balanced_accuracy_score(y, pred), 4),
        f1=round(f1_score(y, pred), 4),
        confusion_matrix=confusion_matrix(y, pred).tolist(),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/llm")
    args = ap.parse_args()

    df = load(args.data)
    exemplar_idx = match_exemplars(df)
    # sanity: 10 exemplars, labels agree with ground truth
    assert len(exemplar_idx) == 10, f"matched {len(exemplar_idx)} exemplars, expected 10"
    for i, lab in exemplar_idx.items():
        truth = "Responder" if int(df.loc[i, "responder"]) == 1 else "Non-responder"
        assert truth == lab, f"exemplar idx {i}: label {lab} != ground truth {truth}"
    print(f"[ok] matched 10 exemplars, all labels consistent with ground truth", flush=True)

    all_idx = list(df.index)
    test_fewshot = [i for i in all_idx if i not in exemplar_idx]  # 26 held-out
    fewshot_block = build_fewshot_block(df, exemplar_idx)

    client = anthropic.Anthropic()

    print("\n=== ZERO-SHOT (all 36) ===", flush=True)
    zs = run(client, df, all_idx, None, "zero-shot")

    print("\n=== FEW-SHOT (26 held-out) ===", flush=True)
    fs = run(client, df, test_fewshot, fewshot_block, "few-shot")

    zs_all = score(df, zs, "zero_shot_all36")
    zs_held = score(df, {i: zs[i] for i in test_fewshot}, "zero_shot_held26")
    fs_held = score(df, fs, "few_shot_held26")

    os.makedirs(args.out, exist_ok=True)
    summary = [zs_all, zs_held, fs_held]
    with open(os.path.join(args.out, "llm_results.json"), "w") as f:
        json.dump({"summary": summary,
                   "zero_shot": {int(i): zs[i] for i in zs},
                   "few_shot": {int(i): fs[i] for i in fs}}, f, indent=2)

    print("\n===================== LLM SUMMARY =====================")
    print(pd.DataFrame(summary)[["eval", "n", "roc_auc", "accuracy",
                                 "balanced_accuracy", "f1"]].to_string(index=False))
    print(f"\n[saved] {os.path.join(args.out, 'llm_results.json')}")


if __name__ == "__main__":
    main()
