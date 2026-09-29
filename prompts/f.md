You have access to the complete codebase.

I have given you the Level 2.2 PRD above.

Your task right now is ONLY to review the PRD against the actual repository.

DO NOT IMPLEMENT ANYTHING.
DO NOT MODIFY ANY FILES.
DO NOT CREATE NEW CODE.

I want you to act as a technical/research implementation reviewer.

Specifically inspect:

1. The existing Level 2.1 pipeline.
2. The actual Level 2.1 grouping output schema.
3. Existing feature caches.
4. Existing OCR token representation.
5. Existing OCR bounding-box representation.
6. Existing ViT embedding cache and how it is generated.
7. Existing candidate-frame directory structure.
8. Existing reporting/UI infrastructure.
9. Existing reusable functions that Level 2.2 should use.
10. Existing dependencies available in the environment.

Then review every part of the Level 2.2 PRD.

For each issue you find, classify it as:

* BLOCKER — mathematically or technically impossible/inconsistent
* MAJOR — likely to cause incorrect research results
* MINOR — implementation detail that should be clarified
* OK — consistent with the current codebase

Pay particular attention to:

### A. Mathematical consistency

Check that:

* A/B lexical coverage is correct.
* Multiset anonymous-slot representation is correctly defined.
* C spatial coverage is mathematically consistent.
* D quality-aware coverage and marginal gain are consistent with each other.
* E1 ViT modulation cannot independently select zero-information frames.
* E2 event evidence cannot independently select zero-information frames.
* stopping conditions agree with the objective.
* fallback selection agrees with reported final coverage.
* tie-breaking is deterministic.
* A/B do not accidentally receive spatial weighting.

### B. Data availability

Verify whether the current caches actually contain everything required for:

* tokens
* token multiplicity
* bounding boxes
* image dimensions
* ViT embeddings
* timestamps
* candidate sequence indices
* pairwise OCR/SSIM/ViT information
* image quality computation

If something is missing, say exactly what is missing.

### C. Level 2.1 boundary

Do NOT redesign or critique the Level 2.1 research approach unless something directly prevents Level 2.2 from functioning.

Level 2.1 is already working and ViT is already part of that workflow.

Treat its output as the input to Level 2.2.

### D. Research validity

Identify places where the PRD accidentally assumes that a feature is semantically meaningful when it is actually only a heuristic.

Examples include:

* ViT diversity
* spatial occupancy
* OCR transition evidence
* quality

The implementation can still use these features; I only want such assumptions clearly identified.

### E. Current evaluation status

Remember:

There is currently NO human-verified Level 2.2 keyframe ground truth.

That is intentional.

The first implementation is exploratory.

Do not demand human annotations before implementation.

Instead, identify what the implementation should expose so that I can later create ground truth and formally evaluate the selector.

### F. Final response format

Return:

1. Executive assessment
2. Codebase components that can be reused
3. PRD/codebase compatibility
4. Mathematical issues
5. Data/schema issues
6. Research-validity concerns
7. Implementation risks
8. Exact changes you recommend before implementation
9. Any remaining decisions that require the researcher (me)
10. Final verdict:

* READY FOR IMPLEMENTATION
* READY AFTER SPECIFIC FIXES
* NOT READY

Do not implement anything until I explicitly approve it.
