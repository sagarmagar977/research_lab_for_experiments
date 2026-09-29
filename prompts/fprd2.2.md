# Level 2.2 — Minimal Informative Keyframe Selection

## 1. Research Context

The thesis pipeline processes educational videos as a structured information source.

The current pipeline is:

Video
→ frame extraction
→ Level 1 candidate-frame selection
→ Level 2.1 temporal/information grouping
→ Level 2.2 keyframe selection
→ OCR/layout extraction
→ downstream representation/RAG/knowledge generation

### Level 1

Level 1 removes highly redundant frames and produces candidate frames.

Level 1 is already implemented.

### Level 2.1

Level 2.1 groups temporally related candidate frames into coherent information groups.

Level 2.1 already uses OCR, SSIM, layout information, and ViT.

ViT is already functioning correctly in the Level 2.1 workflow and must NOT be redesigned or removed as part of this task.

Level 2.2 receives the groups produced by Level 2.1.

### Level 2.2 Research Question

Level 2.1 asks:

> Which candidate frames belong to the same temporal/information unit?

Level 2.2 asks:

> Given a coherent group, what is the minimum useful subset of frames required to represent the important information contained in that group?

The objective is NOT simply to choose visually different frames.

The objective is information-preserving compression of a coherent frame group.

---

# 2. Important Research Constraint

There is currently NO human-verified Level 2.2 keyframe ground-truth dataset.

This is intentional.

The first implementation should produce an exploratory keyframe-selection system whose output can be inspected qualitatively.

Later, the researcher will manually annotate keyframes / necessary states and perform formal evaluation.

Therefore:

* Do not claim the algorithm is already "correct".
* Do not require human ground truth before implementing the exploratory selector.
* Do not fabricate labels.
* Do not create synthetic human annotations.
* Formal human-ground-truth evaluation is a later research stage.

The current goal is to implement the proposed algorithm cleanly and make its decisions inspectable.

---

# 3. Input to Level 2.2

Level 2.2 must accept Level 2.1 group output.

It should NOT assume a particular Level 2.1 grouping algorithm beyond the group structure.

Each group contains ordered candidate frames.

Available information may include:

* frame filename
* original frame ID
* candidate sequence index
* timestamp
* OCR tokens
* OCR full text
* OCR bounding boxes
* OCR confidence
* image dimensions
* cached ViT embedding
* existing pairwise Level 2.1 features
* Level 2.1 group metadata

The implementation must inspect the existing repository and reuse existing schemas/functions where possible.

Do not duplicate existing feature extraction logic unnecessarily.

---

# 4. No New Neural Model

Level 2.2 must NOT introduce a new neural model.

Reuse the cached ViT embeddings already produced by Level 2.1 where ViT is required.

Do not re-run expensive ViT inference if the embedding is already cached.

Classical image processing is allowed, including:

* Laplacian variance
* image statistics
* OCR bounding-box geometry
* SSIM where needed
* other deterministic image measurements

Do not add a new neural dependency merely for Level 2.2.

---

# 5. Core Concept

For a Level 2.1 group:

G = {F1, F2, ..., Fn}

we want:

S ⊆ G

such that:

1. S preserves important information in G.
2. S is substantially smaller than G.
3. Selected frames remain chronologically ordered.
4. Redundant frames are not repeatedly selected.
5. Important transient information can survive.
6. The system does not assume that the final frame is always sufficient.
7. The system does not assume that the first frame is always necessary.

Examples of important information:

* progressive mathematical derivation
* equation before it is erased
* final mathematical result
* code before an important modification
* traceback/error output
* execution result
* newly introduced code
* distinct slide regions
* meaningful visual/layout changes

---

# 6. Six Experimental Formulations

Level 2.2 should support these formulations as separate experimental modes.

## A — Unique Lexical Set

Only unique OCR tokens matter.

For group G:

U_A(G) = union of unique tokens across all frames.

A token is covered once it appears in at least one selected frame.

This is the simplest lexical baseline.

Purpose:

Determine how much the naive unique-token representation can achieve.

---

## B — Lexical Multiset

Use token multiplicity rather than only unique presence.

For each token w:

C_G(w) = max count of w in any single frame in G.

Create anonymous multiplicity slots:

U_M(G) = {(w,k) | 1 <= k <= C_G(w)}

A frame containing c occurrences of w covers:

(w,1), ..., (w,c)

The slots represent multiplicity only.

They must NOT be interpreted as semantic identity or OCR position.

Example:

Frame A:
item item item

Frame B:
item

A unique-token representation sees both as the same token.

The multiset representation distinguishes their multiplicity.

A frame F2 subsumes F1 lexically only if:

count(w,F2) >= count(w,F1)

for every token w.

This formulation is intended to test whether token multiplicity provides useful information beyond a simple set.

---

# 7. C — Multiset + Spatial Layout

Extend B with low-level spatial OCR geometry.

Represent normalized OCR bounding-box occupancy using a 16×16 grid.

The spatial representation is:

Cells(F) = occupied normalized grid cells derived from OCR bounding boxes.

The group spatial universe is:

U_sp(G) = union Cells(F), for all F in G.

Coverage combines:

* lexical multiset coverage
* spatial occupancy coverage

with configurable weight:

Coverage_C =
(1 - w_spatial) * lexical_coverage

* w_spatial * spatial_coverage

Default:

w_spatial = 0.15

but this is experimental and must be configurable.

Spatial occupancy is a low-level geometric feature.

Do NOT claim that it semantically understands:

* diagrams
* equations
* arrows
* factor trees
* handwriting structure
* semantic columns

The experiment determines whether the spatial feature actually improves keyframe selection.

If the entire group has no OCR bounding boxes, spatial weighting must automatically become zero.

---

# 8. D — Multiset + Spatial + Quality

Extend C with image-quality awareness.

Use classical image quality information, primarily Laplacian variance/sharpness.

Compute a normalized quality score:

Q(F) ∈ [0,1]

using session-level percentile normalization.

Proposed normalization:

P5 = 5th percentile of Laplacian variance.

P95 = 95th percentile.

For nonblank frames:

Q(F) =
Q_min +
(1-Q_min) *
clip((L(F)-P5)/(P95-P5+epsilon), 0, 1)

Default:

Q_min = 0.20

Blank frames may receive Q=0.

The purpose is NOT to discard blurry frames blindly.

The purpose is to prefer a sharper representation when the same information can be represented by multiple frames.

Important:

D should be understood as:

> quality-aware information coverage

rather than pure information coverage.

If two frames represent the same information and one is substantially sharper, the sharper frame can provide greater coverage of that information.

However, unique information must not disappear merely because its frame is slightly blurry.

---

# 9. E1 — D + ViT Diversity

E1 adds the existing cached ViT representation to D.

ViT is an auxiliary diversity signal.

It must NOT independently select a frame that has zero base information gain.

For candidate F and selected set S:

If S is empty, ViT diversity contribution is zero.

Otherwise:

Div_vit(F|S) =
min over s∈S of:

(1 - cosine_similarity(v_F,v_s)) / 2

The ViT signal modifies the ranking of candidates that already have positive base information gain.

Conceptually:

base_gain =
quality-aware lexical/spatial information gain

Then:

E1_gain =
base_gain *
(1 + w_vit * Div_vit)

Default:

w_vit = 0.20

This is experimental.

Important invariant:

If:

base_gain = 0

then:

E1_gain = 0

Therefore pure presenter movement, camera movement, cursor movement, etc. cannot independently trigger selection merely because ViT sees visual difference.

ViT is treated as an auxiliary visual-diversity signal, not as ground-truth semantic understanding.

---

# 10. E2 — D + ViT + Transition Evidence

E2 extends E1 with auxiliary transition evidence.

The purpose is to test whether explicit transition evidence helps preserve important frames around changes such as:

* OCR destruction
* canvas replacement
* major visual change

Possible transition evidence may use existing pairwise features such as:

* OCR loss
* reverse OCR/layout preservation
* SSIM distance
* ViT distance

Event evidence must remain a heuristic auxiliary signal.

Do NOT claim it is a general semantic event detector.

A candidate receives event evidence only when measurable transition signals support it.

The event signal is also gated by positive base information gain.

Therefore:

base_gain = 0
→ E2_gain = 0

Event evidence cannot independently select a redundant frame.

Default:

w_event = 0.20

This is experimental.

---

# 11. Selection Objective

Use greedy forward selection.

Start:

S = empty

At every iteration:

1. Compute current information coverage.
2. Compute marginal gain for every unselected candidate.
3. Select the candidate with the highest gain.
4. Update the covered information state.
5. Continue until a stopping condition is reached.

The selector must preserve chronological ordering in its final output.

Selection itself may evaluate candidates globally within the group.

---

# 12. Lexical Coverage

For A:

Coverage_A(S) =
number of unique tokens represented by S
/
number of unique tokens in G

For B/C/D/E1/E2:

Coverage_B(S) =
sum over token multiplicity slots covered by S
/
total multiplicity slots in G

For quality-aware D/E1/E2, information elements should track the best quality representation observed for each element.

This allows a sharper frame to improve the representation of information already selected.

---

# 13. Spatial Coverage

For C/D/E1/E2:

Coverage_spatial(S) =
number of group spatial cells represented by S
/
number of spatial cells in the group

Combined coverage:

Coverage =
(1-w_spatial) * lexical_coverage
+
w_spatial * spatial_coverage

For A/B:

Coverage = lexical coverage only.

Do NOT apply spatial weighting to A/B.

---

# 14. Stopping Conditions

Selection stops when either:

### Condition 1 — Target coverage

Coverage(S) >= tau_coverage

or:

### Condition 2 — Marginal gain floor

maximum candidate gain < epsilon_gain

Default:

tau_coverage = 0.95

epsilon_gain = 0.02

These are experimental hyperparameters, not theoretically established constants.

Suggested future sensitivity analysis:

tau_coverage:
0.80, 0.85, 0.90, 0.95, 0.98

epsilon_gain:
0.005, 0.01, 0.02, 0.05

The algorithm must record why it stopped:

* coverage_target
* marginal_gain_floor
* no_valid_candidate
* empty_group

---

# 15. Minimum Frame Constraint

Every non-empty group should return at least one nonblank frame whenever possible.

If normal greedy selection selects nothing but the group contains valid frames, select a deterministic valid frame as fallback.

The final reported coverage MUST correspond to the actual returned selected frame set.

Do not report stale coverage from before the fallback selection.

---

# 16. Tie Breaking

Do not encode semantic assumptions such as:

* always prefer the last frame
* always prefer the first frame

If gains are numerically equal:

1. higher base information gain
2. higher quality
3. earlier candidate sequence index

This is only deterministic tie-breaking.

---

# 17. Ground Truth Status

There is currently NO Level 2.2 human ground truth.

The initial implementation must therefore support:

### Exploratory evaluation

Show:

* all frames in group
* selected frames
* timestamps
* OCR
* quality
* selection gain
* coverage
* ViT contribution
* event contribution
* stopping reason

This allows the researcher to inspect whether the algorithm behaves sensibly.

### Future formal evaluation

Later, the researcher will manually annotate which frames/states are necessary.

At that stage the system should support:

* exact frame matching
* temporal-tolerance matching
* potentially state/interval-equivalent matching
* precision
* recall
* F1
* compression ratio
* information-preservation metrics

Do NOT create fake human annotations now.

Do NOT make human agreement a prerequisite for implementation.

---

# 18. Experimental Ablation Structure

The six modes are intended to isolate contributions:

A:
Unique lexical set

B:
Multiset lexical representation

C:
B + spatial geometry

D:
C + image quality

E1:
D + ViT diversity

E2:
E1 + transition evidence

The comparison must keep the rest of the experimental protocol as constant as possible.

The goal is to determine empirically whether each additional representation improves useful keyframe selection.

Do not assume beforehand that a later formulation is better.

---

# 19. Required Outputs

For every group, return at minimum:

* group ID
* mode
* selected frame filenames
* selected timestamps
* selected count
* final lexical coverage
* final spatial coverage
* final combined coverage
* compression ratio
* stopping reason

For each selected frame, record:

* candidate sequence index
* timestamp
* quality score
* base information gain
* ViT diversity contribution if applicable
* event evidence if applicable
* final selection gain

This information is required for reproducibility and debugging.

---

# 20. UI Requirements

Create or extend a Level 2.2 Keyframe Selection Lab.

The UI should allow the researcher to:

1. Select session.
2. Select Level 2.1 grouping run.
3. Select group.
4. Select formulation A/B/C/D/E1/E2.
5. Run selection.
6. View all frames chronologically.
7. Clearly identify selected frames.
8. Inspect why each selected frame was selected.
9. Compare selected frames against the entire group.
10. View coverage and compression.
11. View stopping reason.
12. Compare formulations on the same group.

Do not require human annotation functionality in the first implementation unless it already fits naturally into the existing architecture.

---

# 21. Reproducibility

The implementation must:

* use deterministic selection
* store configuration used for each run
* store formulation/mode
* store selected frame IDs
* store scores
* avoid hidden hardcoded thresholds
* preserve existing Level 2.1 outputs
* avoid modifying Level 2.1 cached features unnecessarily

Each run should be reproducible from:

session
+
Level 2.1 grouping run
+
Level 2.2 configuration.

---

# 22. Implementation Constraints

Before coding:

1. Inspect the complete relevant Level 2.1 implementation.
2. Inspect actual cache schemas.
3. Inspect actual grouping-run schemas.
4. Inspect candidate-frame directory resolution.
5. Reuse existing utilities.
6. Identify any mismatch between this PRD and the current codebase.
7. Report conflicts before implementation.

Do NOT silently change the research design to fit the existing code.

If the existing code contradicts the PRD, report the contradiction.

Do NOT modify code during the review phase.

---

# 23. Research Philosophy

The selector is an experimental research system.

The implementation must make it possible to answer:

> Does progressively richer information representation allow a system to select fewer frames while preserving more important information?

The system should therefore make each contribution measurable:

A → lexical set

B → multiplicity

C → spatial structure

D → quality

E1 → ViT visual diversity

E2 → transition evidence

No formulation should be assumed to be superior before evaluation.

---

# 24. Current Task

IMPORTANT:

This document is the Level 2.2 research specification.

For the first response, DO NOT IMPLEMENT ANYTHING.

Instead:

1. Read this PRD completely.
2. Inspect the relevant existing codebase.
3. Identify reusable components.
4. Identify schema mismatches.
5. Identify mathematical inconsistencies.
6. Identify implementation risks.
7. Identify anything in this PRD that conflicts with the existing Level 2.1 implementation.
8. Check whether the proposed six formulations can actually be implemented using the currently available cached data.
9. Check whether the proposed greedy objective and pseudocode are internally consistent.
10. Check whether any assumptions are unsupported by the existing data.

Return a detailed technical review.

Do not modify files.

Do not implement Level 2.2 yet.

Do not redesign Level 2.1.

Treat Level 2.1 as an existing working component, including its current ViT integration.
