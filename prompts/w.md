Do NOT design or implement Level 2.2 yet.

The previous audit established that the cache contains more information than the current Level 2.1 comparison functions use:

* ordered OCR token lists
* duplicate tokens
* token confidences
* normalized bounding boxes
* consecutive SSIM
* cached ViT embeddings
* timestamps

Before deciding on an algorithm, I want one final audit focused specifically on what representations can be constructed from EXISTING cached information.

Do not propose thresholds or a final method.

---

# 1. Enumerate all usable representations from the cache

Starting ONLY from:

* tokens
* confidences
* bboxes
* timestamps
* consecutive SSIM
* ViT embeddings
* raw frame files

list the representations Level 2.2 could theoretically construct WITHOUT OCR or ViT re-inference.

For example, determine whether we can construct:

* token set
* token multiset / frequency vector
* ordered token sequence
* confidence-weighted token representation
* token + spatial position
* line-level representation
* spatial region representation
* text-density representation
* frame-level OCR information density
* temporal change representation
* ViT visual similarity
* combinations of the above

Do not choose which one is best.

---

# 2. Ordered OCR representation

The cache preserves ordered tokens.

Determine exactly how reliable that order is.

Is the token list order:

A. true reading order from RapidOCR,

B. detection order,

C. arbitrary model output order,

or something else?

Inspect the OCR implementation and determine this.

This matters because "ordered token sequence" is only useful if the order is meaningful.

---

# 3. Bounding-box reconstruction

Determine exactly what can be reconstructed from:

```
token + bbox
```

without new OCR.

Specifically test conceptually whether we can derive:

* line grouping
* vertical ordering
* horizontal ordering
* approximate paragraphs
* indentation
* spatial clusters
* left/right columns
* text regions
* overlapping text regions

Also identify what cannot be reconstructed.

Do NOT implement these features.

---

# 4. Confidence information

The cache contains OCR confidence.

Determine:

* Is confidence per token?
* Is it aligned positionally with the token list?
* Are confidence values reliable enough to identify garbage tokens?
* Are there examples where garbage tokens have high confidence?
* Are there examples where legitimate handwritten/math tokens have low confidence?

Use actual data where possible.

Do NOT propose a confidence threshold.

---

# 5. Token multiplicity

We now know duplicates are preserved.

Determine how much information would be lost by:

```
set(tokens)
```

versus:

```
Counter(tokens)
```

using actual examples.

Find at least 2–3 cases where multiplicity materially differs.

Do not propose which representation to use.

---

# 6. Token order + spatial order

There may be two different notions of order:

1. OCR detection sequence
2. spatial reading order reconstructed from bounding boxes

Determine whether they agree in existing examples.

If they disagree, identify why.

This will tell us whether token sequence modeling is reliable.

---

# 7. Structural information available from bboxes

Determine whether bounding boxes can distinguish the following hypothetical transformations using existing information:

### A

x
x + 1

### B

x + 1
x

### C

x       x + 1

### D

x + 1       x

Explain which distinctions are observable from the cache and which are not.

Again, no implementation.

---

# 8. Important correction: image pixels

The previous audit said image pixels are "not in cache."

Clarify the architecture:

Are the original candidate frame image files still available at the Level 2.2 stage?

If yes:

* Can Level 2.2 access them without re-running OCR/ViT?
* What information could theoretically be extracted from the raw image?
* Which of those would count as "new inference" versus ordinary image processing?

Do NOT propose which image features to compute.

---

# 9. Non-adjacent comparisons

The cache only stores consecutive pairwise metrics.

But ViT embeddings are stored per frame.

Determine exactly which non-adjacent comparisons can be reconstructed without re-inference.

For example:

* ViT cosine
* OCR set similarity
* OCR multiset similarity
* spatial similarity

State which are directly possible from cached data.

---

# 10. What is actually missing?

After examining all available information, divide missing information into:

### A. Recoverable from existing cache

Information that can be derived algebraically from current fields.

### B. Recoverable from raw frame images without OCR/ViT

Information requiring ordinary image processing.

### C. Requires new model inference

Information that genuinely requires another learned model or new OCR/ViT inference.

This distinction is critical.

---

# 11. Re-examine "zero re-inference"

Do NOT simply accept or reject zero re-inference.

Instead determine:

What can Level 2.2 accomplish using only cached features?

What would require raw-image processing?

What would require neural inference?

Give a capability boundary, not a recommendation.

---

# 12. Verify the proposed five experiments

The previous answer proposed:

1. human ground truth
2. token-set failure rate
3. ViT code threshold distribution
4. bbox structural reconstruction
5. blur frequency

For each one, classify:

* genuinely necessary before algorithm design
* useful but optional
* premature / implementation-dependent

Do NOT replace them with your own experimental plan.

---

# 13. Final evidence inventory

Produce one final table:

| Evidence | Available now? | Source | Reliability | Can Level 2.2 derive more from it? |
| -------- | -------------- | ------ | ----------- | ---------------------------------- |

Include:

* OCR text
* token list
* token multiplicity
* token order
* confidence
* bbox
* timestamp
* consecutive SSIM
* ViT embedding
* consecutive ViT distance
* raw images
* sharpness
* layout hierarchy
* semantic labels
* ground truth

---

# 14. Final boundary statement

Answer this exact question:

"Given the current codebase, what is the maximum amount of information Level 2.2 can obtain WITHOUT running another OCR model, another ViT model, or another learned model?"

Give a factual answer only.

Do not design the algorithm.

---

STRICT RULES:

NO CODE CHANGES.

NO IMPLEMENTATION.

NO FINAL LEVEL 2.2 ALGORITHM.

NO THRESHOLD SELECTION.

NO CLAIM OF NOVELTY.

NO "BEST" REPRESENTATION.

NO "OPTIMAL" INPUT.

NO ASSUMPTION THAT B2 IS GROUND TRUTH.

If a previous answer was incorrect, correct it explicitly.
