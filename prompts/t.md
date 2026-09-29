Do NOT implement Level 2.2.

Do NOT propose the final algorithm.

The previous response identified many weaknesses in the original proposal. Before we design anything, I want you to verify the strongest claims you just made against the actual code and data.

This round is about EVIDENCE VERIFICATION, not solution design.

For every answer, clearly label:

* VERIFIED FROM CODE
* VERIFIED FROM DATA
* INFERENCE
* UNVERIFIED CLAIM

If you previously made a claim without sufficient evidence, correct it explicitly.

---

# 1. VERIFY THE ACTUAL GROUP CONTENT

You stated:

"MATH Group 6 contains a full problem presentation, derivation, board wipe, and subsequent root calculation."

Verify this directly.

Give:

* exact group ID
* exact start/end frame
* exact number of frames
* chronological phases inside the group
* actual important transitions
* whether these are truly one pedagogical unit or whether this is your interpretation

Do not call it a "lesson" unless the data/code actually establishes that.

---

# 2. VERIFY THE "FRAME 0260 IS BETTER THAN FRAME 0286" CLAIM

You claimed:

frame_0260 has fewer tokens but better visual/legibility quality than frame_0286.

Verify this from the actual images/data.

If there is no quantitative sharpness/quality measurement currently available, say so.

Do NOT infer "better" purely from visual intuition unless you explicitly label it as visual inspection.

---

# 3. VERIFY THE CODING SCROLLING CLAIM

You claimed:

CODING_SESSION Group 8 frame_0357 → 0433 demonstrates that K=1 loses earlier code.

Verify:

* group ID
* frame IDs
* what text appears at the beginning
* what disappears from the viewport
* what appears later
* whether the information is actually lost or merely outside the viewport
* whether Level 2.2 has access to those earlier frames

Do not equate "not visible in final frame" with "semantically lost" without evidence.

---

# 4. VERIFY THE OCR-GARBAGE CLAIM

You claimed noisy OCR can make a frame appear more informative.

Find several actual examples.

For each:

* frame ID
* OCR tokens
* confidence values if available
* visual context
* whether the extra tokens are actually garbage
* whether those tokens are unique to that frame

If you only have one example, say so.

---

# 5. VERIFY THE POPUP EXAMPLE

You claimed:

frame_0076 has 44 → 68 tokens and frame_0077 drops to 30.

Verify these exact values from the cache.

Also verify:

* whether the popup is actually an autocomplete/UI popup
* whether it is pedagogically irrelevant
* whether the current Level 2.1 bridge already handles it
* whether Level 2.2 actually needs another popup detector

This is important because we should not duplicate functionality already present in Level 2.1.

---

# 6. VERIFY THE "TRANSIENT INFORMATION" EXAMPLES

You claimed that:

* blackboard scratch calculations
* terminal traceback
* temporary annotations

may be pedagogically important.

Separate:

A. directly observed in the current dataset

B. plausible but not actually observed

For each observed example, give exact frame IDs.

---

# 7. VERIFY THE ViT CLAIMS

You stated that ViT:

* succeeds on slide transitions
* fails on code edits
* reacts to irrelevant camera/light changes

Verify each with actual frame pairs.

For every example provide:

frame A
frame B
d_ssim
d_ocr_jaccard
ocr_preservation if available
d_layout
d_vit

Then explain ONLY what these numbers demonstrate.

Do not infer "semantic" meaning from one pair.

---

# 8. VERIFY THE CLAIM THAT ViT IS BAD FOR CODE

You said:

"ViT should not be used as a hard gate for code screens."

This may be reasonable, but distinguish:

1. What the current data proves
2. What the current data suggests
3. What would require a larger experiment

Do not turn three examples into a general claim about ViT.

---

# 9. VERIFY THE "SUBMODULAR" CLAIM

Be mathematically precise.

Separate:

A. pure token union objective

B. token union + cardinality constraint

C. token union + pairwise ViT constraints

D. token union + transient vetoes

E. token union + temporal interval constraints

For each, determine:

* Is the objective submodular?
* Is the feasible set a standard constraint class?
* Does the classic greedy guarantee apply?
* If not, what exactly prevents the guarantee?

Do not merely say "not submodular."

---

# 10. VERIFY THE "SEMANTIC INFORMATION" PROBLEM

We agree that OCR token union is not semantic information.

But now identify the precise observable information currently available in the cache.

Create a table:

| Information type      | Currently available? | Exact source |
| --------------------- | -------------------- | ------------ |
| OCR lexical content   |                      |              |
| OCR confidence        |                      |              |
| token multiplicity    |                      |              |
| token order           |                      |              |
| line structure        |                      |              |
| bounding boxes        |                      |              |
| spatial relationships |                      |              |
| SSIM                  |                      |              |
| ViT embedding         |                      |              |
| visual sharpness      |                      |              |
| temporal position     |                      |              |
| frame-to-frame change |                      |              |
| image pixels          |                      |              |

Do not propose new features.

---

# 11. VERIFY WHETHER TOKEN ORDER IS REALLY LOST

There is an important distinction:

The original OCR extraction may contain ordered tokens even if later comparison converts them to sets.

Determine:

* Does `features_cache.json` preserve the original ordered token list?
* Or is order already destroyed during OCR extraction?
* At exactly which stage does the conversion to set occur?

This matters greatly for Level 2.2.

---

# 12. VERIFY WHETHER TOKEN MULTIPLICITY IS AVAILABLE

Similarly:

* Does the cached token list preserve duplicates?
* Can Level 2.2 reconstruct token frequencies from the cache without re-running OCR?
* Is `tokens` a list or set in the persisted JSON?

Answer from the actual schema.

---

# 13. VERIFY WHETHER SPATIAL INFORMATION CAN ALREADY SUPPORT STRUCTURE

The previous response said token sets lose spatial structure.

But the cache contains bounding boxes.

Determine exactly what can already be recovered from:

* token text
* bbox
* confidence

without new OCR inference.

For example, can the existing data distinguish:

A:
x
x + 1

from:

B:
x + 1
x

?

Do not implement anything. Just determine what information exists.

---

# 14. VERIFY THE CLAIM THAT LEVEL 2.1 B2 IS "OPTIMAL INPUT"

You previously wrote:

"B2 = Optimal Input."

Remove the word "optimal" unless there is actual experimental evidence supporting that conclusion.

Instead compare A1/A2/B2 using only:

* current group counts
* current grouping behavior
* current known failures
* existing evaluation status

Also state whether there is currently a ground-truth benchmark proving B2 is superior.

---

# 15. VERIFY THE CLAIM THAT A2/B2 GROUPS ARE "PEDAGOGICAL UNITS"

Determine whether "pedagogical unit" is actually defined or annotated anywhere in the codebase.

If not, use a more precise description such as:

"contiguous groups produced by the Level 2.1 grouping algorithm."

Do not attribute semantic correctness to the grouping algorithm without ground truth.

---

# 16. VERIFY THE CLAIM THAT ZERO RE-INFERENCE IS SUFFICIENT

The proposal says Level 2.2 can operate entirely from cached information.

Determine:

Which proposed decisions can be made from the cache?

Which potentially important decisions CANNOT currently be made reliably because the cache lacks:

* sharpness
* image-level quality
* structural OCR information
* visual-region information
* non-adjacent comparisons

Do not solve the missing-information problem.

---

# 17. VERIFY THE EXISTING EVALUATION STATE

Earlier you said there is a Ground Truth Studio but no saved ground-truth CSV.

Verify the exact current state.

Answer:

* Does `ground_truth_transitions.csv` exist?
* Does it contain annotations?
* How many annotations?
* Are they used in any current report?
* Are A1/A2/B2 actually evaluated against ground truth?
* Are current group counts merely outputs, rather than accuracy evidence?

---

# 18. VERIFY THE DATASET SIZE

Give exact current numbers for:

slides_AA
MATH-SESSION
CODING_SESSION

For each:

* candidate frames
* A1 groups
* A2 groups
* B2 groups
* total duration
* number of transitions

Then calculate the total number of groups across all sessions.

Do not describe the dataset as statistically sufficient unless you can justify that.

---

# 19. Separate FACTS FROM HYPOTHESES

Create three lists.

### Established facts

Only things directly demonstrated by the current code/data.

### Strong hypotheses

Things supported by multiple observations but not yet experimentally validated.

### Speculative assumptions

Things we should NOT build into architecture yet.

This distinction is extremely important.

---

# 20. Final question

After performing this verification, answer:

"What is the smallest set of experimentally testable facts we need to establish before choosing the Level 2.2 representation and selection algorithm?"

Do NOT give the algorithm.

Do NOT choose thresholds.

Do NOT propose implementation.

The purpose is to establish the evidence base first.

Again:

NO CODE CHANGES.
NO IMPLEMENTATION.
NO FINAL LEVEL 2.2 DESIGN.
NO DEFENDING PREVIOUS ASSUMPTIONS.

Correct yourself wherever necessary.
