We have now completed and verified the Level 2.2 frozen-baseline failure analysis across all 112 manually annotated GT keyframes in:

* MATH-SESSION
* slides_AA
* linear_algebra_session
* CODING_SESSION

Verified baseline facts:

* E1 overall: P=0.280, R=0.545, F1=0.370
* Coding E1: P=0.050, R=0.160, F1=0.076
* 157 false positives
* 51 false negatives
* 157/157 FPs passed the raw information-gain floor
* 12 FNs stopped because coverage_target_reached
* 39 FNs stopped because raw_information_floor
* Important coding FNs can have strong Quality/Event/ViT/transition evidence while Raw Gain = 0 because the current OCR/spatial representation considers them already covered.

We are now beginning the DESIGN phase for Level 2.2 v2.

IMPORTANT:

* Do NOT modify production code yet.
* Do NOT implement anything yet.
* Do NOT train a model yet.
* Do NOT arbitrarily tune thresholds.
* Do NOT assume the previous taxonomy is complete.
* Do NOT simply suggest "add more features."
* Treat the verified baseline evidence as the starting point.

Your task is to DESIGN a defensible Level 2.2 v2 formulation.

The central research question is:

How should a video keyframe-selection objective identify pedagogically meaningful state changes while rejecting transient visual noise, especially when the new state occupies the same spatial region and therefore has little/no incremental OCR or spatial gain?

Analyze the problem from first principles.

### 1. Define the target

Precisely define what a "keyframe" represents in Level 2.2.

Distinguish:

* visual difference
* information novelty
* state change
* stable state
* transient event
* pedagogical milestone
* final/settled representation

Do not collapse these into one concept.

### 2. Diagnose the current mathematical objective

Write the current A–E2 objective mathematically.

Then identify exactly why:

* transient frames receive high scores
* meaningful state changes can receive zero score
* chronological greedy accumulation causes later frames to lose marginal gain
* multiplication by Raw Gain can suppress otherwise useful evidence

Do not merely describe these problems verbally.

### 3. Design candidate v2 objectives

Develop 2–4 mathematically distinct candidate formulations.

For each candidate specify:

* objective function
* variables/signals used
* what each term means
* whether it is additive, multiplicative, gated, or hierarchical
* how it handles already-covered information
* how it handles same-location state changes
* how it handles transient frames
* how it handles final/stable states
* how it handles scrolling
* how it handles coding
* how it handles mathematical derivations
* expected failure modes

Do NOT choose a winner yet.

### 4. Separate the dimensions

Investigate whether Level 2.2 should explicitly model separate dimensions such as:

Information Novelty
State Change
Persistence/Stability
Quality
Event Evidence
Redundancy
Temporal Context

Determine whether these should be:

* one unified scalar objective,
* hierarchical decision stages,
* constrained optimization,
* or another formulation.

Justify mathematically and using the observed failures.

### 5. Persistence / stability

The baseline is strongly vulnerable to transient typing, cursors, pen strokes, and slide transitions.

Investigate how "persistent state" could be defined from the existing temporal sequence.

For example, analyze concepts such as:

* persistence over subsequent frames
* recurrence
* temporal neighborhood agreement
* state stabilization
* replacement/subsumption
* delayed confirmation

Do NOT implement these. Formulate them mathematically and identify their assumptions.

### 6. Same-region state changes

The coding FN problem is especially important.

A new line of code or changed terminal output may occupy exactly the same bounding box as previous content.

Analyze what "information gain" should mean in this case.

Determine whether the current OCR representation should compare:

* token sets
* token sequences
* token frequencies
* spatially aligned text
* textual replacement
* semantic embeddings
* visual embeddings
* temporal differences

Do not assume one is correct. Compare them conceptually and against the actual failure evidence.

### 7. Global vs greedy selection

Analyze whether chronological greedy selection is fundamentally appropriate.

Compare:

A. chronological greedy
B. global candidate scoring
C. two-pass selection
D. candidate generation + replacement/substitution
E. dynamic programming / sequence optimization
F. other defensible approaches

For each, explain computational complexity and what failure mechanism it addresses.

### 8. Existing feature utilization

We already have:

* OCR information
* spatial information
* ViT similarity
* SSIM/transition information
* quality
* event evidence
* temporal position

Determine which should remain, which should change roles, and which currently have weak evidence.

Do NOT automatically remove ViT just because the baseline sometimes performs worse with it.

### 9. Research novelty

Be critical.

Determine what part of the proposed v2 formulation could constitute a legitimate research contribution versus ordinary engineering.

Identify what would need to be experimentally demonstrated to support the contribution.

### 10. Experimental design

Before implementation, define the experiments needed to distinguish the competing hypotheses.

Include:

* ablation experiments
* per-session evaluation
* per-content-type evaluation
* FP/FN analysis
* stability/persistence analysis
* same-region-change analysis
* threshold sensitivity
* computational cost

Use the existing four manually annotated datasets.

### 11. Final deliverable

Produce a research design document with:

1. Problem Definition
2. Current Objective Analysis
3. Design Requirements Derived From Evidence
4. Candidate V2 Formulations
5. Mathematical Definitions
6. Global-vs-Greedy Analysis
7. Persistence/State Modeling
8. Same-Region Information Modeling
9. Feature Role Analysis
10. Research Contribution Analysis
11. Experimental Plan
12. Recommended Next Research Step

Important:

Do NOT write production code.
Do NOT edit the Level 2.2 implementation.
Do NOT select arbitrary thresholds.
Do NOT claim that a proposed formulation will improve F1 until experimentally tested.

The goal is to derive a defensible Level 2.2 v2 research formulation from the verified evidence, not to immediately optimize the benchmark.
