"""
serialize.py
Render an ADHD patient's baseline feature row into a natural-language clinical
description (the "full-data serialization" used for the LLM experiments).

Every instrument in the battery is named with its scores and brief severity
qualifiers. This is the text that is shown to the LLM (see llm_responder.py and
adhd_fewshot_curve LLM arm). The exact prompts are in
APPENDIX_prompt_and_exemplars.md.
"""
import pandas as pd
import numpy as np

GENDER = {1: "male", 2: "female"}
RACE = {1: "White/Caucasian", 2: "Black/African American",
        3: "American Indian/Alaskan Native", 4: "Asian/Pacific Islander",
        5: "more than one race", 6: "unknown/not reported"}
STIM = {1: "methylphenidate", 0: "amphetamine"}

# ASR syndrome/composite scales (T-scores; higher = more impaired)
ASR = {"attent": "attention problems", "aggret": "aggressive behavior",
       "anxdept": "anxious/depressed", "intrut": "intrusive",
       "delnqt": "rule-breaking", "somat": "somatic complaints",
       "thougt": "thought problems", "withdt": "withdrawn",
       "extert": "externalizing", "intert": "internalizing",
       "totalt": "total problems", "asr_aaa": "attention-aggression-antisocial composite"}
# BRIEF-A executive function (T-scores; higher = worse)
BRIEF = {"inhibit_t": "inhibit", "shift_t": "shift", "emocont_t": "emotional control",
         "selfmon_t": "self-monitor", "initiate_t": "initiate", "workmem_t": "working memory",
         "planorg_t": "plan/organize", "taskmon_t": "task-monitor",
         "orgmat_t": "organization of materials", "bri_t": "Behavioral Regulation Index",
         "mi_t": "Metacognition Index", "gec_t": "Global Executive Composite"}
# SRS-2 (T-scores)
SRS = {"aware_t": "social awareness", "cog_t": "social cognition",
       "comm_t": "social communication", "motiv_t": "social motivation",
       "rrb_t": "restricted/repetitive behavior", "srs_total_t": "total"}


def _tscore_band(v):
    """Unified T-score banding used for all ASR / BRIEF-A / SRS-2 scales
    (higher = more impaired). Thresholds verified against the generated
    serialization: <40 below average, 40-64 within normal limits,
    65-69 mildly elevated, >=70 clinically elevated."""
    if v >= 70:
        return "clinically elevated"
    if v >= 65:
        return "mildly elevated"
    if v < 40:
        return "below average"
    return "within normal limits"

_asr_band = _tscore_band
_brief_band = _tscore_band
_srs_band = _tscore_band

def _adhd_band(total):
    return "severe" if total >= 40 else ("moderate" if total >= 24 else "mild")


def verbalize(row):
    """row: a pandas Series with the baseline columns. Returns the vignette string."""
    L = []
    age = int(row["age"]); sex = GENDER.get(int(row.get("gender", 1)), "unknown")
    race = RACE.get(int(row.get("race", 6)), "Unknown/Not reported")
    stim = STIM.get(int(row.get("stim_type", 1)), "a stimulant")
    L.append(f"Demographics: {age}-year-old {sex} patient, {race}; assigned {stim}.")
    if "adhd_rs_total" in row:
        L.append(f"ADHD Rating Scale (ADHD-RS): total {int(row['adhd_rs_total'])} "
                 f"({_adhd_band(row['adhd_rs_total'])}), inattention {int(row['adhd_rs_inatt'])}, "
                 f"hyperactivity/impulsivity {int(row['adhd_rs_hyper'])}.")
    # AAA is a raw aggregate composite (not a T-score) -> reported without a band
    asr = ", ".join(
        (f"{lbl} {int(row[c])}" if c == "asr_aaa" else f"{lbl} {int(row[c])} ({_asr_band(row[c])})")
        for c, lbl in ASR.items() if c in row and pd.notna(row[c]))
    if asr: L.append(f"Adult Self-Report (ASR) T-scores: {asr}.")
    # BRI and MI are aggregate index scores reported without a band; GEC keeps its band
    _brief_noband = {"bri_t", "mi_t"}
    brief = ", ".join(
        (f"{lbl} {int(row[c])}" if c in _brief_noband else f"{lbl} {int(row[c])} ({_brief_band(row[c])})")
        for c, lbl in BRIEF.items() if c in row and pd.notna(row[c]))
    if brief: L.append(f"BRIEF-A executive function T-scores (higher = worse): {brief}.")
    srs = ", ".join(f"{lbl} {int(row[c])} ({_srs_band(row[c])})" for c, lbl in SRS.items() if c in row and pd.notna(row[c]))
    if srs: L.append(f"Social Responsiveness Scale (SRS-2) T-scores: {srs}.")
    other = []
    if "mwq_total" in row and pd.notna(row["mwq_total"]): other.append(f"Mind-Wandering Questionnaire (MWQ) {int(row['mwq_total'])}")
    if "desr_total" in row and pd.notna(row["desr_total"]): other.append(f"emotional dysregulation (DERS-derived) {int(row['desr_total'])}")
    if "q_les_q_raw" in row and pd.notna(row["q_les_q_raw"]): other.append(f"Quality of Life Enjoyment and Satisfaction (Q-LES-Q) {int(row['q_les_q_raw'])} (higher = better)")
    if other: L.append("Other measures: " + "; ".join(other) + ".")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    df = pd.read_excel(sys.argv[1], sheet_name="Data")
    print(verbalize(df.iloc[0]))
