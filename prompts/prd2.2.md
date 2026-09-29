You have full access to the thesis codebase.

We are now implementing **Level 2.2: Minimal Information-Preserving Keyframe Selection**.

Do NOT redesign Level 2.1. Treat the existing Level 2.1 groups as the input to Level 2.2.

## 1. Core objective

Level 2.2 receives one temporal group from Level 2.1:

F1 → F2 → ... → Fn

and selects the **minimum useful subset of frames** that preserves the important instructional information contained in that group.

The output must remain chronologically ordered.

The conceptual optimization is:

```
minimize |S|
```

subject to:

```
InformationCoverage(S) >= threshold
```

where S is the selected keyframe subset.

Do NOT interpret this as simply selecting the frame with the most OCR tokens.

---

# 2. Strict architectural boundary

Reuse existing cached data wherever possible.

Level 2.2 MAY use:

* features_cache.json
* vit_embeddings.npz
* existing Level 2.1 transition metrics
* candidate-frame images already stored on disk
* OpenCV / NumPy / scikit-image classical image processing

Level 2.2 MUST NOT:

* rerun OCR
* rerun ViT
* introduce another neural model
* download another model
* modify Level 2.1 grouping logic

The purpose is to test how much better keyframe selection can become using the information already available after Level 2.1.

---

# 3. Build a clean Level 2.2 module

First inspect the existing codebase and determine the best module structure.

Prefer something conceptually like:

modules/l2_keyframe_selection.py

or a dedicated:

modules/l2_2/

Do not duplicate existing feature extraction unnecessarily.

Before coding, report:

1. existing relevant files
2. existing reusable functions
3. existing data structures
4. proposed files to add/change
5. exact Level 2.2 input/output schema

Then implement.

---

# 4. Frame representation

For each frame construct a Level 2.2 representation containing:

### A. Lexical information

Use the raw ordered token list from features_cache.json.

Construct:

* token set
* token Counter / multiset
* token count
* unique-token count
* repeated-token information

Do NOT discard multiplicity by converting everything to set.

The Counter representation should be the primary lexical representation.

---

### B. Spatial information

Use existing OCR bounding boxes.

Derive lightweight geometric features:

* normalized box coordinates
* box area
* spatial occupancy
* vertical ordering
* approximate line grouping
* approximate column grouping
* left/right distribution
* layout density

Do not claim this reconstructs true semantic structure.

It is only geometric structural information.

---

### C. Visual information

Use cached ViT embeddings.

Compute cosine distance between arbitrary frames in a group when required.

Do NOT introduce a fixed universal rule such as:

```
ViT distance >= 0.04 means important
```

unless it is explicitly treated as an experimental threshold.

---

### D. Image quality

From existing raw images, compute lightweight classical metrics:

* Laplacian variance / sharpness
* blank/near-blank detection
* optionally basic image validity

Cache these results so images are not repeatedly loaded.

Do not use image quality as a semantic measure.

Its purpose is to prevent obviously degraded frames from being preferred merely because they contain OCR noise.

---

### E. Temporal information

Preserve:

* timestamp
* candidate sequence index
* temporal distance

---

# 5. Information-event analysis

Before selecting keyframes, analyze changes between frames.

For Fi → Fi+1 calculate or reuse:

### Lexical change

Use multiset-aware comparison.

Measure:

* additions
* removals
* frequency changes

Do not rely exclusively on Jaccard set similarity.

### Spatial change

Measure:

* layout occupancy change
* bounding-box movement
* approximate region changes

### Visual change

Use cached ViT cosine distance.

### Quality change

Compare sharpness/validity where useful.

The purpose is to detect **information events**, not merely visual transitions.

Examples:

1. accumulation
2. replacement
3. erasure/destruction
4. persistent state
5. transient state
6. visual-only movement/noise

Do NOT hardcode the assumption that every transient is irrelevant.

For example:

* a context-menu popup may be irrelevant
* a traceback/error may be educationally important
* a temporary mathematical derivation may be essential

The event analysis should therefore produce evidence/features, not blindly delete frames.

---

# 6. Candidate scoring / information contribution

Implement a frame information contribution function.

Conceptually:

```
marginal_gain(Fi | S)
```

should represent how much information Fi adds beyond the already selected frames S.

The contribution should be composed from explicitly separated components:

* lexical information
* spatial information
* visual information
* temporal/event information
* quality suitability

Do NOT immediately invent arbitrary weights and claim they are optimal.

Make the weights/configuration explicit and configurable.

---

# 7. Multiset lexical coverage

The first coverage component should preserve token multiplicity.

For example:

Frame A:

```
x x x y
```

Frame B:

```
x y z
```

The system must not treat these as equivalent to:

```
{x,y}
```

Use Counter-based coverage/intersection.

Also preserve the possibility of frequency changes being informative.

This should be experimentally comparable with a simpler set-based baseline.

---

# 8. Spatial coverage

Implement a lightweight spatial coverage measure based on bounding boxes / rasterized occupancy.

The purpose is to distinguish situations such as:

```
information on left side
information on right side
```

even when the OCR vocabulary is similar.

Keep this computationally simple.

Do not introduce a learned layout model.

---

# 9. Visual diversity

Use cached ViT embeddings as an optional visual-diversity component.

Do NOT use a universal fixed ViT threshold.

Instead, expose the threshold as a configuration parameter and make it possible to disable the ViT component.

This is necessary for the ablation study.

Remember that existing evidence shows:

* small code edits can have low ViT distance
* presenter movement can have larger ViT distance
* therefore ViT distance alone is not semantic importance

ViT is supporting evidence, not the sole selector.

---

# 10. Quality-aware selection

A frame with more OCR tokens should not automatically win if it is severely blurred or otherwise degraded.

Use sharpness as a quality/suitability factor.

Do NOT allow sharpness to dominate semantic information.

A clean frame with slightly less OCR content may be preferable to a blurry frame containing OCR garbage.

Keep this component configurable so it can be ablated.

---

# 11. Minimality / greedy selection

Implement a greedy information-preserving selector.

Initial:

```
S = {}
```

At each iteration:

```
select the frame with the highest marginal information gain
```

subject to any configured temporal/quality constraints.

After each selection, recompute the remaining marginal gains.

Stop when:

```
coverage >= target
```

or when additional frames provide negligible information gain.

Always keep at least one frame per non-empty Level 2.1 group.

Finally sort selected frames chronologically.

Do NOT call the entire algorithm mathematically "submodular" unless the implemented objective actually satisfies the required assumptions.

If the implementation uses a submodular component, document that component precisely.

---

# 12. Important protection against bad selections

The selector must avoid blindly selecting:

* OCR garbage
* heavily blurred frames
* blank frames
* transitional partially-erased frames

BUT:

Do not automatically discard a frame simply because:

* it has few tokens
* it is transient
* its ViT distance is low
* its OCR confidence is low

These can still contain important information.

Frame rejection should require evidence from multiple signals.

---

# 13. No rigid role taxonomy

Do not force every group into:

```
initial_setup
pre_erasure_peak
final_culmination
```

Those are examples only.

If useful, expose descriptive event metadata such as:

* accumulation
* replacement
* erasure
* transient
* persistent
* visual_change

but keep the system extensible.

Do not fabricate probabilistic confidence values.

Use deterministic scores/features unless an actual calibrated model exists.

---

# 14. Implement baselines for ablation

Level 2.2 should support at least these modes:

### L2.2-A — Lexical Set Baseline

Set-based information coverage.

### L2.2-B — Lexical Multiset

Counter/frequency-aware coverage.

### L2.2-C — Multiset + Spatial

Add bounding-box/layout information.

### L2.2-D — Multiset + Spatial + Quality

Add sharpness/blank-frame information.

### L2.2-E — Full Multimodal

Add cached ViT + temporal/event information.

All modes must use the same groups and evaluation procedure.

The purpose is to determine which information sources actually improve keyframe selection.

---

# 15. Human ground truth

Add support for a Level 2.2 annotation dataset.

Do NOT assume ground truth exists.

Create a clear annotation format, for example:

```
group_id
frame_id
selected_by_human
necessary
redundant
notes
```

If a group requires multiple frames, humans should be able to mark all necessary frames.

The annotation UI should make it easy to browse:

* all frames in chronological order
* timestamps
* OCR text
* selected frames
* group boundaries

Start with approximately 20–30 carefully chosen groups covering:

* slides
* progressive bullets
* mathematics
* erasure
* coding
* terminal output
* scrolling
* diagrams/mixed layouts
* transient UI
* errors/tracebacks

Do not claim the benchmark is statistically representative. It is an exploratory human-annotated benchmark.

---

# 16. Evaluation metrics

Once annotations exist, evaluate each L2.2 method using:

### Selection efficiency

How many frames were selected?

```
selected_count
```

### Human recall

Of the human-required frames, how many did the method preserve?

```
recall = TP / (TP + FN)
```

### Precision

How many selected frames were actually considered necessary?

```
precision = TP / (TP + FP)
```

### F1

Report F1 where appropriate.

### Compression

Compare:

```
selected frames / original group frames
```

### Information preservation

Where possible, compare the selected subset's information coverage against the complete group.

Do not report only compression.

A method that selects one frame for everything may have excellent compression but terrible information preservation.

---

# 17. Evaluation should include difficult cases

Do not only evaluate average performance.

Break results down by group/event type:

* accumulation
* replacement
* erasure
* coding edits
* scrolling
* transient UI
* mathematical derivation
* multi-column layout

This will expose where particular representations fail.

---

# 18. Required outputs

For every group, produce machine-readable output similar to:

```
{
    "group_id": "...",
    "method": "...",
    "input_frame_count": N,
    "selected_frames": [...],
    "selected_count": K,
    "compression_ratio": ...,
    "coverage": ...,
    "events": [...],
    "selection_scores": [...]
}
```

Do not put fake confidence values in this output.

---

# 19. UI

Extend the existing Level 2 lab rather than creating an unrelated application.

The UI should allow:

1. selecting a Level 2.1 group
2. viewing all frames chronologically
3. seeing selected keyframes
4. comparing L2.2-A/B/C/D/E
5. seeing why each frame was selected
6. manually marking ground-truth keyframes
7. comparing algorithmic selection with human selection

For debugging, show the component contributions behind each selected frame.

---

# 20. Important implementation principle

Separate:

### Representation

"What information do we extract?"

from:

### Selection

"Which frames do we choose?"

from:

### Evaluation

"Did the selection preserve what humans consider important?"

Do not mix these three layers together.

---

# 21. Do NOT over-engineer the first implementation

First implement the smallest end-to-end version:

```
cached features
   ↓
multiset representation
   ↓
spatial representation
   ↓
quality features
   ↓
information contribution
   ↓
greedy minimal selection
   ↓
chronological keyframes
   ↓
JSON output
```

Then test it on:

* MATH-SESSION
* CODING_SESSION
* slides_AA

Only after this works should additional heuristics be added.

---

# 22. Scientific discipline

For every threshold or heuristic introduced:

1. make it configurable
2. document why it exists
3. distinguish empirical observation from assumption
4. avoid claiming it is optimal without evaluation

Do NOT hardcode previously proposed values such as:

```
final/max = 0.85
token loss = 0.40
ViT distance = 0.04
popup = +8 tokens / 2 frames
```

unless they are introduced as experimental parameters and evaluated.

---

# 23. Before implementation

First inspect the codebase and produce a short implementation report answering:

1. Which existing functions can be reused?
2. What exact schema does Level 2.1 currently provide?
3. Where are candidate images stored?
4. Where are ViT embeddings stored?
5. How should Level 2.2 integrate with the current UI?
6. What files will be created/modified?
7. What parts of the proposed architecture conflict with the existing code?

Do NOT implement until this inspection/report is complete.

After the report, implement incrementally and test each stage.

The final Level 2.2 should be reproducible, ablation-friendly, and experimentally measurable rather than a collection of unexplained heuristics.
