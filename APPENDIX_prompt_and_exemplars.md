# Appendix: LLM Prompt and Serialization Specification

This appendix reproduces the exact system prompts, output instruction, and
patient-record serialization used for the large language model (LLM) experiments,
verbatim. It is the authoritative prompt reference for `src/llm_responder.py`,
`src/serialize.py`, and the LLM arm of `src/adhd_fewshot_curve.py`.

**Model.** All LLM results use **Claude Opus 4** (Anthropic; API model
`claude-opus-4`). `claude-opus-4` is a rolling alias; the exact dated snapshot is
returned in each API response's `model` field. Accessed 2026.

**Two prompt designs are compared** (manuscript Figure 1b): a **Simple** prompt
(A.1, concise task description) and a **Clinician-guided** prompt (A.2, structured
clinical briefing). Both performed comparably.

## A.1 Simple system prompt

```text
You are predicting whether an adult ADHD patient will respond to stimulant medication (Responder vs Non-responder) from a pre-treatment clinical description.

Return a JSON object with exactly two fields:
- "response_probability": an integer 0-100 (your probability the patient is a Responder).
- "label": "Responder" if response_probability >= 50, otherwise "Non-responder".
```

## A.2 Clinician-guided system prompt

```text
You are an expert psychiatrist specializing in adult ADHD pharmacotherapy. Your task is to predict whether a medication-naive adult with ADHD will respond to stimulant treatment, based on a comprehensive pre-treatment clinical assessment.

STUDY CONTEXT
Thirty-six medication-naive adults meeting DSM-5 criteria for ADHD were drawn from a prospective neuroimaging and clinical study (Hung et al., 2024) at the Massachusetts General Hospital Adult ADHD Program. Participants with mild comorbid anxiety or depression not requiring urgent clinical attention were not excluded. After baseline assessment, participants received methylphenidate (n=25) or amphetamine (n=11) at therapeutic doses and were followed naturalistically for a mean of 116.1 +/- 77.9 days. Treatment response was defined by the Clinician-rated Clinical Global Impression-Improvement scale (CGI-I) at final follow-up: a Responder is CGI-I <= 2 (much or very much improved); a Non-responder is CGI-I > 2. The cohort is balanced (18 responders, 18 non-responders).

INSTRUMENTS AND HOW TO READ THEM
- ADHD Rating Scale (ADHD-RS): core ADHD symptom severity. Raw totals; higher = more severe. Total ~< 24 mild, 24-39 moderate, >= 40 severe. Reported with inattention and hyperactivity/impulsivity subscales.
- Adult Self-Report (ASR, Achenbach): dimensional behavioral/emotional problems as T-scores (population mean 50, SD 10). Broadband scales include attention problems, aggressive behavior, anxious/depressed, intrusive, rule-breaking, somatic complaints, thought problems, withdrawn, plus Externalizing, Internalizing, and Total Problems composites, and an attention-aggression-antisocial (AAA) raw composite. Higher = more problems.
- BRIEF-A: everyday executive function. T-scores where HIGHER = WORSE (more dysfunction). Subscales: inhibit, shift, emotional control, self-monitor, initiate, working memory, plan/organize, task-monitor, organization of materials; indices: Behavioral Regulation Index (BRI), Metacognition Index (MI), Global Executive Composite (GEC).
- Social Responsiveness Scale (SRS-2): social-communication and autism-spectrum traits as T-scores; higher = more difficulty. Subscales: social awareness, cognition, communication, motivation, restricted/repetitive behavior, and Total.
- Mind-Wandering Questionnaire (MWQ): trait mind-wandering; higher = more.
- Emotional dysregulation (DERS-derived index): higher = more emotion-regulation difficulty.
- Quality of Life Enjoyment and Satisfaction Questionnaire (Q-LES-Q): HIGHER = BETTER quality of life.

T-SCORE INTERPRETATION (ASR, BRIEF-A, SRS)
< 40 below average; 40-64 within normal limits; 65-69 mildly elevated (borderline); >= 70 clinically elevated.

CLINICAL REASONING GUIDANCE
Integrate the full profile rather than any single score. Consider symptom severity and its distribution across inattentive vs hyperactive domains, the degree of executive dysfunction, the presence and pattern of comorbid behavioral/emotional and social difficulties, emotion-regulation load, and baseline quality of life. Weigh how these features, in combination, bear on the likelihood of a favorable stimulant response. When labeled example patients are provided, study them carefully and use them to calibrate your judgment before classifying the final patient.

OUTPUT
Return a JSON object with exactly two fields:
- "response_probability": an integer 0-100 giving your probability that this patient is a Responder (CGI-I <= 2).
- "label": "Responder" if response_probability >= 50, otherwise "Non-responder".
```

## A.3 Output instruction (appended to every query)

```text
Respond with ONLY a JSON object: {"response_probability": <int 0-100>, "label": "Responder"|"Non-responder"}.
```

## A.4 Patient-record serialization

Each patient's tabular record is converted to text by naming every instrument
and its scores, with brief severity qualifiers ("clinically elevated", "within
normal limits") and directional notes ("higher = worse" for BRIEF-A; "higher =
better" for Q-LES-Q). See `src/serialize.py`. Example serialized patient:

```text
Demographics: 22-year-old male patient, White/Caucasian; assigned amphetamine.
ADHD Rating Scale (ADHD-RS): total 57 (severe), inattention 29, hyperactivity/impulsivity 28.
Adult Self-Report (ASR) T-scores: attention problems 70 (clinically elevated), aggressive behavior 60 (within normal limits), anxious/depressed 52 (within normal limits), intrusive 63 (within normal limits), rule-breaking 59 (within normal limits), somatic complaints 58 (within normal limits), thought problems 52 (within normal limits), withdrawn 57 (within normal limits), externalizing 63 (within normal limits), internalizing 55 (within normal limits), total problems 62 (within normal limits), attention-aggression-antisocial composite 182.
BRIEF-A executive function T-scores (higher = worse): inhibit 70 (clinically elevated), shift 47 (within normal limits), emotional control 49 (within normal limits), self-monitor 54 (within normal limits), initiate 63 (within normal limits), working memory 86 (clinically elevated), plan/organize 78 (clinically elevated), task-monitor 68 (mildly elevated), organization of materials 75 (clinically elevated), Behavioral Regulation Index 56, Metacognition Index 79, Global Executive Composite 71 (clinically elevated).
Social Responsiveness Scale (SRS-2) T-scores: social awareness 55 (within normal limits), social cognition 55 (within normal limits), social communication 54 (within normal limits), social motivation 51 (within normal limits), restricted/repetitive behavior 47 (within normal limits), total 52 (within normal limits).
Other measures: Mind-Wandering Questionnaire (MWQ) 25; emotional dysregulation (DERS-derived) 6; Quality of Life Enjoyment and Satisfaction (Q-LES-Q) 44 (higher = better).
```

## A.5 Few-shot structure

In N-shot conditions the user message lists N labeled example patients (each
ending in `OUTCOME: Responder` / `OUTCOME: Non-responder`), followed by the query
patient under `NOW CLASSIFY THIS PATIENT:` and the output instruction. Exemplars
and query cases are always disjoint. The LLM was evaluated at 0, 2, 4, 8, 16 and
24 in-context examples under each prompt, with multiple random exemplar/test
partitions (seeds) per condition; ROC-AUC was computed per seed and averaged.
