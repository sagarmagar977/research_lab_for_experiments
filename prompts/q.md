I do NOT want you to propose or implement a Level 2.2 plan yet.

First, I want you to inspect the actual thesis codebase and answer the questions below. You have access to the complete codebase, so answer based on the **actual current implementation and files**, not assumptions.

Do not modify any files.

The goal is to collect enough information for an external reviewer to design Level 2.2 correctly afterward.

---

# A. Current Level 2.1 implementation

### 1. What exactly does Level 2.1 currently do?

Trace the real execution flow from:

```text
input candidate frames
→ feature extraction
→ similarity calculation
→ grouping
→ output
```

Give me the actual files/functions/classes involved and explain what each does.

---

### 2. What information does each Level 2.1 frame object currently contain?

List the actual fields available for each frame.

For example, determine whether we already have:

* frame ID
* timestamp
* OCR text
* OCR tokens
* OCR confidence
* bounding boxes
* font height
* layout features
* SSIM
* histogram
* edge features
* ViT embedding
* ViT cosine similarity
* image path
* quality/sharpness
* other features

Give the actual field names from the code/schema.

---

### 3. Where does Level 2.1 persist its output?

Inspect:

```text
grouping_runs.json
features_cache.json
```

and any related files.

Explain the actual JSON structure.

Show one representative example structure, but don't dump huge files.

---

# B. Current OCR implementation

### 4. How exactly is OCR currently performed?

Identify:

* OCR library/model
* preprocessing
* tokenization
* normalization
* confidence handling
* minimum token rules
* whether OCR is run once per frame or cached
* where OCR results are stored

---

### 5. Does the current OCR representation allow directional containment?

For:

$$
P(A\rightarrow B)
=
\frac{|T_A\cap T_B|}{|T_A|}
$$

determine whether the current representation can reliably support this.

If not, explain exactly what is missing.

Do NOT redesign it yet.

---

### 6. How does the current code handle OCR failure?

Find what happens when:

* OCR returns empty text
* OCR returns very few tokens
* OCR confidence is low
* OCR produces garbage
* OCR detects text in only one frame

Explain the current behavior.

---

# C. Current ViT implementation

### 7. Where and how is ViT currently used?

This is very important.

Trace the actual ViT workflow:

```text
image
→ preprocessing
→ ViT model
→ embedding
→ normalization
→ cosine similarity/distance
→ grouping decision
```

Give the exact model name and implementation.

---

### 8. Is ViT currently used for pairwise similarity or something else?

Determine whether the current system calculates:

```text
cosine(F1,F2)
```

or compares frames against a reference/centroid/group representation.

Explain the actual implementation.

---

### 9. Is the ViT embedding cached?

If yes:

* where?
* what format?
* how large?
* how many dimensions?
* how expensive is extraction?

If no, explain the current computational cost.

---

### 10. Does current ViT output capture meaningful visual changes that OCR cannot?

Inspect existing test outputs if possible.

Look specifically at examples where:

```text
OCR ≈ same
but visual content changes
```

and:

```text
OCR changes
but visual content is essentially the same
```

Report concrete examples from the existing sessions if available.

Do not invent examples.

---

# D. Existing datasets/sessions

### 11. Inspect the current Level 2 test sessions.

At minimum inspect:

```text
slides_AA
MATH-SESSION
CODING_SESSION
```

and identify:

* number of candidate frames
* number of Level 2.1 groups
* typical group sizes
* timestamps/frame IDs
* OCR availability
* ViT availability

Give a concise summary.

---

### 12. Find actual progressive-information examples.

From the existing sessions, identify real examples of:

```text
A
A + B
A + B + C
```

where information accumulates.

Give the actual frame IDs and describe what changes.

---

### 13. Find actual destructive-information examples.

Identify real cases where:

```text
information appears
→ later disappears/replaced
```

For example:

```text
math derivation
→ erased
→ final answer
```

or:

```text
valid code
→ modified code
→ error
```

Give actual frame IDs.

---

### 14. Find cases where the final frame is NOT the best representative.

Use the actual datasets.

Identify examples where:

* final frame is occluded
* final frame is blurry
* final frame loses important information
* final frame contains only a conclusion
* earlier frame is more informative

Give actual frame IDs.

---

### 15. Find cases where one frame is clearly insufficient.

Identify real groups where representing the group requires more than one frame.

Give actual examples.

Do not assume K=2.

Determine what the data actually shows.

---

# E. Scrolling / terminal / dynamic content

### 16. Do the current datasets contain scrolling content?

Check coding/terminal/whiteboard sessions.

If yes, identify examples.

Explain what happens to OCR tokens as the viewport moves.

---

### 17. Does the current code use spatial layout?

Determine whether OCR bounding boxes/layout are currently available and whether the current grouping algorithm uses them.

Explain whether layout coordinates are absolute image coordinates or normalized coordinates.

---

# F. Current quality signals

### 18. What quality measures already exist?

Search the codebase for:

* blur
* Laplacian variance
* sharpness
* occlusion
* frame quality
* black/blank frame detection
* OCR confidence
* visual stability
* duplicate detection

List what already exists.

Do not propose new measures yet.

---

# G. Current Level 2.1 behavior

### 19. What are the current A1, A2 and B2 formulations in the actual code?

Do not rely on the previous discussion.

Inspect the implementation and report:

* exact formula
* weights
* thresholds
* normalization
* fallback behavior
* OCR-invalid behavior
* ViT-invalid behavior
* layout behavior

---

### 20. What is currently controlled by sliders/configuration?

List every relevant Level 2.1 parameter.

For example:

```text
tau_A
tau_B
OCR weight
ViT weight
layout weight
minimum OCR tokens
etc.
```

Give actual defaults.

---

# H. Existing evaluation

### 21. How are Level 2.1 results currently evaluated?

Inspect the code and existing reports.

Determine whether there is already:

* manual ground truth
* expected grouping
* precision
* recall
* F1
* pairwise accuracy
* group accuracy
* human annotation
* visual inspection only

---

### 22. What are the current known failure cases?

Search existing reports, comments, experiment outputs, and cached runs.

Give the actual observed failure modes.

Separate:

```text
OCR failures
ViT failures
grouping failures
data problems
UI problems
```

---

# I. Computational constraints

### 23. What is the actual computational cost of the current Level 2 pipeline?

Estimate or measure:

* number of frames
* OCR cost
* ViT embedding cost
* pairwise similarity cost
* memory usage
* runtime for each test session

The goal is to make Level 2.2 practical on the current machine.

---

### 24. What components can Level 2.2 reuse without recomputation?

Identify reusable caches/features.

Especially:

* OCR
* ViT embeddings
* layout
* similarity matrices
* frame metadata

---

# J. Existing architecture boundaries

### 25. What should Level 2.2 receive as input?

Based strictly on the current architecture, identify the cleanest interface between:

```text
Level 2.1
        ↓
Level 2.2
```

Specify the actual data structure.

---

### 26. What should Level 2.2 output?

Do NOT design the algorithm yet.

Just determine what information the output would need to contain for Level 3.

For example:

```text
group_id
selected frame IDs
timestamps
selection reason
coverage information
confidence
```

Only identify requirements; do not propose the selection method yet.

---

# K. Critical empirical questions

### 27. Does the existing data actually justify a "culmination" assumption?

Inspect the sessions.

Determine whether most groups really have:

```text
progressive accumulation → final complete state
```

or whether there are many other patterns.

Categorize the actual observed patterns.

---

### 28. How many groups appear to be:

* monotonic accumulation
* destructive replacement
* alternating states
* scrolling
* mostly static
* visual-only change
* OCR-only change
* mixed multimodal change

Use the current datasets if possible.

---

### 29. Does "persistent information" actually correlate with useful keyframe information?

Do an empirical inspection.

Look for counterexamples where:

* important information is transient
* unimportant information persists
* OCR noise persists
* UI elements persist

Report what you observe.

---

### 30. What information cannot be represented by OCR alone?

Using the actual sessions, identify cases involving:

* diagrams
* equations
* formatting
* code indentation
* visual annotations
* arrows
* highlighting
* handwritten structure
* charts

where ViT or visual information is necessary.

---

# L. Most important final questions

### 31. Based ONLY on the existing code/data, what does Level 2.2 actually need to solve?

Do not propose the solution.

State the problem in concrete terms derived from the evidence.

---

### 32. What assumptions from our previous Level 2.2 proposal are contradicted by the actual code/data?

List them explicitly.

---

### 33. What information is already available that we were previously planning to recompute?

List it.

---

### 34. What important information is currently missing that Level 2.2 would need?

List it.

---

### 35. What are the 5–10 most important empirical observations from the codebase that an external researcher should know before designing Level 2.2?

Give concise evidence-based answers.

---

# STRICT INSTRUCTIONS

Do NOT:

* implement code
* modify files
* propose the final Level 2.2 algorithm
* choose thresholds
* invent ground truth
* claim novelty
* redesign the architecture

At this stage I ONLY want a **codebase/data audit and factual answers**.

Use the actual code and actual cached/session data wherever possible.

If something cannot be determined from the codebase, explicitly say:

> "Not available from current codebase."

Do not guess.

The purpose is to give another researcher enough factual information to design Level 2.2 correctly afterward.
