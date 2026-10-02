We have now completed the frozen-baseline diagnosis and the pre-implementation validation across all 4 manually annotated GT datasets (112 GT keyframes total).

I do NOT want implementation yet.

Your task now is to design and justify the **mathematical formulation of Level 2.2 v2** based strictly on the validated evidence.

## 1. Evidence you MUST treat as established

The previous validation established:

1. Adjacent-frame SSIM is NOT a valid strict stability gate.

   * GT keyframes and FPs have similar adjacent volatility.
   * Many valid GT keyframes occur near state-change boundaries.

2. Same-region state changes are real and important, especially in CODING_SESSION.

   * Token-set difference can be zero/low even when the visible/code state meaningfully changes.
   * OCR sequence/edit/line-level differences can expose some of these changes.
   * This does NOT prove OCR edit distance should replace the existing representation. Treat it as an additional candidate signal.

3. The current multiplicative objective is structurally problematic:
   Score ≈ RawGain × (1 + ViT/Event terms) × Quality
   because RawGain = 0 can completely suppress other useful evidence.

4. Chronological greedy coverage causes premature suppression.

   * There are 194 FP→FN instances.
   * Average FP→FN temporal distance ≈ 28.9 seconds.
   * Some missed coding GT frames contain many independent tokens despite marginal RawGain = 0.

5. Simple "choose the highest-quality later frame" substitution is NOT sufficient.

   * Later FNs have higher Quality than the preceding FP only 55% of the time.

6. ViT is currently complementary but not sufficiently validated to be the primary state-change signal.

7. We still do NOT have a proven signal that alone distinguishes transient noise from meaningful keyframe transitions.

Do not contradict these findings without explicitly identifying new evidence.

---

# 2. Core research problem

Formulate Level 2.2 as:

**Given a Level-2.1 group containing temporally ordered candidate frames, identify meaningful visual/state transitions and select an appropriate representative frame for each meaningful state, while avoiding transient noise, redundant frames, and premature chronological coverage suppression.**

Do NOT assume that every visual change is a keyframe.

Do NOT assume that every keyframe must be the most stable frame.

Do NOT assume that the final frame is always the correct representative.

---

# 3. Separate the problem into TWO conceptual decisions

This is important.

Analyze Level 2.2 as two related but distinct problems:

### A. State-transition / novelty detection

Determine whether frame t represents a meaningful new state relative to the previously represented information.

Possible evidence:

* OCR token-set change
* OCR sequence/edit change
* line-level/code-region change
* spatial/layout change
* SSIM / visual difference
* ViT difference
* event evidence
* temporal context
* accumulated coverage

### B. Representative-frame selection

Once a meaningful state/change is detected, determine which frame best represents that state.

Possible evidence:

* image quality
* OCR quality/confidence
* completeness
* text visibility
* absence of transient artifacts
* persistence of the state
* temporal neighborhood
* similarity to surrounding frames

Explicitly explain why these should NOT simply be collapsed into one multiplicative score.

---

# 4. Develop 3–4 candidate mathematical formulations

Do not immediately choose one.

Develop several mathematically defensible alternatives.

At minimum consider:

### Candidate A — Multi-signal state-transition + representative selection

Separate:

StateChange(t)

from

RepresentativeQuality(t)

and select:

transition first → representative second.

Explain the mathematical formulation.

---

### Candidate B — Sequence-aware formulation

Use temporal context rather than only frame t vs t-1.

For example, investigate whether:

StateChange(t)

should compare the candidate against a temporal neighborhood/window or inferred previous state rather than only the immediately previous frame.

Do NOT assume a fixed window is correct. Just formulate and analyze it.

---

### Candidate C — Global/windowed optimization

Instead of chronological greedy selection, formulate keyframe selection as an optimization problem over the whole group or temporal windows.

Consider concepts such as:

* coverage
* novelty
* redundancy
* diversity
* representative quality
* temporal consistency

You may investigate submodular/facility-location-style objectives, dynamic programming, or other sequence-selection formulations.

Do not use these terms merely because they sound like research. Explain whether they actually fit this problem.

---

### Candidate D — Hybrid formulation

Consider whether the best design is:

1. detect meaningful state transitions,
2. form state segments/windows,
3. choose the best representative from each segment,
4. optionally apply a global redundancy constraint.

Explain mathematically how this differs from the current greedy approach.

---

# 5. Investigate the concept of "state"

This is critical.

We need to determine what a "state" actually means for our thesis.

Do NOT casually define state as:

* identical OCR,
* identical pixels,
* identical bounding boxes,
* or simply "stable for N frames."

Instead analyze whether a state should be represented as a feature representation such as:

S_t = f(OCR, layout, visual embedding, image structure, etc.)

Then determine how state similarity/change could be measured.

Discuss:

* set-based representation
* sequence-based representation
* spatial representation
* visual representation
* hybrid representation

Explain what each captures and what it misses.

---

# 6. Investigate temporal persistence properly

The previous hypothesis "stable in the next frame = meaningful" was rejected.

Do NOT bring back an adjacent-frame SSIM stability gate.

Instead investigate broader concepts:

* persistence over a temporal neighborhood
* recurrence of a state
* duration of a state
* before/after state comparison
* transition boundary vs settled representative
* whether the transition frame itself or a later settled frame should be selected

Important distinction:

A frame can be a valid **transition detector** without being the best **representative frame**.

Analyze this explicitly.

---

# 7. Investigate the coding-video problem

Use the verified coding failures as a concrete test case.

Explain mathematically how the formulation handles:

Example pattern:

Frame A:
old code

Frame B:
partial typing / transient state

Frame C:
completed code

The system should not automatically interpret B as a new keyframe merely because OCR changed.

Likewise:

Frame A:
old code

Frame B:
new code occupying the same spatial region

must not be considered redundant merely because spatial coverage has already been claimed.

This is one of the central weaknesses of the current formulation.

---

# 8. Analyze the existing features individually

For each existing feature:

* Raw OCR gain
* OCR Jaccard
* asymmetric information containment
* OCR sequence/edit difference
* layout/spatial overlap
* SSIM
* ViT cosine distance
* image quality
* event evidence
* temporal distance

Answer:

1. What phenomenon does it measure?
2. What does it fail to measure?
3. Is it evidence for state change, persistence, representative quality, redundancy, or something else?
4. Should it be:

   * primary evidence,
   * secondary evidence,
   * a gate,
   * a penalty,
   * or only diagnostic?

Do not force every feature into the final equation.

---

# 9. Critically analyze the current greedy coverage mechanism

Current behavior effectively allows earlier frames to "claim" information and make later frames appear redundant.

Formalize this problem.

Determine whether coverage should be:

* accumulated permanently,
* state-local,
* temporally decayed,
* reset when a new state is detected,
* represented as similarity to selected representatives,
* or handled through global optimization.

Do NOT choose arbitrarily.

Use the 194 FP→FN temporal evidence to motivate the analysis.

---

# 10. Define what "meaningful keyframe" mathematically means

We need an operational definition that can actually be evaluated.

Compare possible definitions:

* information novelty
* state transition
* persistent state
* representative of a new state
* visual distinctiveness
* pedagogical milestone

Do not assume "pedagogical milestone" unless the GT evidence actually supports it.

Identify which definition is observable from our available data and GT annotations.

---

# 11. Compare the candidate formulations

Create a table containing:

| Formulation | Handles same-region changes | Handles transient noise | Handles later better representatives | Handles greedy suppression | Computational cost | Interpretability | Research defensibility |
| ----------- | --------------------------- | ----------------------- | ------------------------------------ | -------------------------- | ------------------ | ---------------- | ---------------------- |

Do NOT assign arbitrary numerical scores.

Use qualitative evidence-based descriptions.

Then identify:

* what each formulation solves,
* what each fails to solve,
* what assumptions each introduces,
* what experiments would distinguish them.

---

# 12. Research contribution analysis

Determine whether the proposed v2 formulation is actually a research contribution or merely engineering.

Separate:

### Engineering contribution

Combining existing OCR/SSIM/ViT/quality features.

### Potential research contribution

A novel formulation for sequence-aware keyframe selection that explicitly separates:

* state transition detection,
* state representation,
* temporal persistence,
* representative selection,
* and redundancy/coverage.

Be critical.

If this is not sufficiently novel by itself, say so.

Do NOT artificially label ordinary feature engineering as a novel research contribution.

---

# 13. Required final output

Produce ONE research design report containing:

1. Problem reformulation
2. Why current E1 mathematically fails
3. Formal definition of state
4. Formal definition of state change
5. Formal definition of representative quality
6. Formal definition of redundancy/coverage
7. Role of temporal context
8. 3–4 candidate mathematical formulations
9. Equations for each candidate
10. Assumptions behind each
11. Strengths/weaknesses
12. Comparison table
13. Recommended formulation, IF the evidence supports one
14. What remains experimentally unproven
15. Experiments required before implementation
16. Research contribution assessment

## Strict constraints

* DO NOT modify production code.
* DO NOT implement v2.
* DO NOT change A–E2.
* DO NOT train any model.
* DO NOT tune thresholds.
* DO NOT invent empirical results.
* DO NOT assume OCR edit distance replaces token-set coverage.
* DO NOT use adjacent-frame SSIM as a stability gate.
* DO NOT assume highest image quality is always the correct representative.
* DO NOT assume global optimization is automatically superior to greedy selection.
* DO NOT force ViT into the formulation merely because it is available.
* Use the four GT datasets and the validated diagnostics as evidence.
* Clearly distinguish established evidence, hypothesis, and proposed mathematical design.

The goal is to leave us with a mathematically defensible Level 2.2 v2 formulation that we can experimentally test BEFORE writing the implementation.
