We now have the Level 2.2 v2 research design document.

DO NOT implement the v2 algorithm yet.

DO NOT modify the production Level 2.2 code.
DO NOT change A–E2.
DO NOT train a model.
DO NOT introduce new thresholds into the production pipeline.

Before implementation, we need to empirically validate whether the proposed v2 mechanisms are actually supported by our existing data.

Use the four manually annotated datasets:

* MATH-SESSION
* slides_AA
* linear_algebra_session
* CODING_SESSION

Use the existing GT annotations, cached features, diagnostic JSON, A–E2 outputs, and temporal frame sequences.

The goal is to test the hypotheses behind the v2 design, NOT to improve the benchmark yet.

### Experiment 1 — Stability / Persistence Hypothesis

Test whether temporal persistence can actually distinguish:

* human-selected GT keyframes
* transient false positives
* other non-keyframes

Using existing signals where possible, analyze temporal neighborhoods around candidate frames.

Measure things such as:

* forward SSIM agreement
* backward SSIM agreement
* ViT temporal similarity
* OCR/text stability
* persistence duration
* temporal change before and after the frame

Do NOT assume a particular stability threshold.

Report distributions for GT keyframes vs FPs vs ordinary frames.

Determine:

* whether GT frames are actually more stable
* whether transient FPs are actually less stable
* which sessions/content types violate this assumption
* whether some legitimate GT frames are themselves transient

If the evidence does not support the hypothesis, say so.

### Experiment 2 — Same-Region State-Change Hypothesis

Focus especially on the Coding session, but also inspect Math and Linear Algebra.

For GT frames that were missed because RawGain was near zero:

compare the frame against relevant previous frames using:

* OCR token-set difference
* OCR sequence/edit difference
* line-level text difference
* OCR spatial alignment
* ViT distance
* SSIM difference

Determine whether meaningful state changes are actually detectable in the existing data despite zero RawGain.

Classify each FN as:

A. existing signals detect the change but RawGain ignores it
B. OCR detects it
C. visual/ViT detects it
D. spatial alignment detects it
E. none of the current signals reliably detect it
F. insufficient evidence

Do not assume edit distance is the solution.

### Experiment 3 — Replacement / Later-Frame Hypothesis

For each important FP→FN sequence:

1. Identify the noisy/intermediate frame selected by E1.
2. Identify the later GT frame that was missed.
3. Examine the temporal interval between them.
4. Determine whether the later GT frame contains a more complete/stable representation of the same underlying information.
5. Measure whether the later frame has higher quality/stability/coverage.

Quantify how often this pattern actually occurs.

Do not assume local substitution would work.

Determine how far apart the noisy frame and GT frame typically are temporally.

### Experiment 4 — Coverage/Greedy Hypothesis

Determine whether chronological accumulation itself causes the failure.

For every FN, determine:

* what earlier frames consumed its information
* which feature caused the information to be considered covered
* whether the FN would still have zero marginal gain if evaluated independently
* whether the problem is truly chronological accumulation or simply an inadequate representation

Separate these two causes carefully.

### Experiment 5 — Feature Evidence Matrix

For every GT FN and representative FP, create a compact matrix:

| Frame | GT | RawGain | OCR change | Spatial change | ViT distance | SSIM change | Quality | Event | Persistence |
| ----- | -- | ------- | ---------- | -------------- | ------------ | ----------- | ------- | ----- | ----------- |

Use actual values from the repository.

The goal is to determine which existing signals contain useful evidence that the current objective fails to use.

### Experiment 6 — Test the V2 Design Assumptions

Evaluate these assumptions individually:

1. Stable frames are generally preferable to transient frames.
2. Meaningful state changes can occur without spatial gain.
3. OCR sequence/edit information captures some changes missed by token sets.
4. Later frames can sometimes be better representatives of information already detected earlier.
5. A single multiplicative RawGain objective is insufficient.
6. ViT provides useful complementary evidence.
7. A hierarchical/gated formulation is justified by the data.

For each assumption give:

* Supported
* Partially supported
* Not supported
* Insufficient evidence

And provide the evidence.

### Important constraints

Do not propose implementation fixes yet.

Do not change thresholds.

Do not run an optimization search.

Do not report an improved F1 by artificially modifying selection behavior.

Do not retrofit the analysis to prove the v2 design correct.

If an experiment contradicts the design document, explicitly report the contradiction.

### Final deliverable

Create:

`level2_v2_preimplementation_validation.md`

with:

1. Experimental Setup
2. Stability/Persistence Results
3. Same-Region State-Change Results
4. Replacement Hypothesis Results
5. Greedy/Coverage Results
6. Feature Evidence Matrix
7. V2 Assumption Validation
8. Contradictory Evidence
9. What the Data Actually Supports
10. What Remains Unproven
11. Requirements for the V2 Algorithm

At the end, give a clear distinction between:

**Established by evidence**
vs.
**Plausible but unproven**
vs.
**Rejected/unsupported**

Do not implement Level 2.2 v2 after producing this report.

Stop after the report.
