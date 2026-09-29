I have now given you the complete Level 2.1 codebase audit and your proposed Level 2.2 architectural/methodological specification.

DO NOT IMPLEMENT ANYTHING.

DO NOT MODIFY FILES.

DO NOT finalize or defend the current Level 2.2 proposal.

Instead, act as a skeptical research/code reviewer and stress-test your own proposal against the ACTUAL current codebase and datasets.

I will give your response to an external reviewer afterward, so be precise and evidence-based.

For every question, distinguish:

* directly supported by current code/data
* reasonable inference
* unsupported assumption
* contradicted by existing evidence

If something cannot be determined, explicitly say:

"Not available from current codebase/data."

---

# A. Stress-test the fundamental objective

### 1. What exactly does "semantic information coverage" mean in the current system?

Your proposal uses:

```
Coverage(S) =
|union of OCR tokens in selected frames|
/
|union of OCR tokens in the entire group|
```

Is this actually a valid measure of semantic information for the observed datasets?

Test this against the actual examples you found:

* math factor trees
* equations
* diagrams
* charts
* code indentation
* syntax highlighting
* arrows
* visual annotations
* transient UI elements

Identify concrete cases where OCR-token coverage would give the WRONG answer.

Do not redesign the metric yet.

---

### 2. Is the claim that the proposed objective is "submodular" mathematically justified?

Check whether the proposed set function actually satisfies the properties required for submodularity.

Do not call something submodular merely because it uses set union.

Explain precisely what is and is not submodular here.

---

# B. Stress-test dynamic K

### 3. Is "monotonic information accumulation -> K=1" actually supported by the data?

Use the observed groups.

Find counterexamples if any.

In particular ask:

Could information be cumulative in OCR tokens while visual/structural information is lost?

Could an intermediate frame contain important information that the final frame no longer preserves?

Could a frame contain more OCR tokens but be less useful because of an artifact, popup, overlap, or bad OCR?

Report actual examples.

---

### 4. Is token-count monotonicity a reliable way to classify groups?

The proposal uses:

```
R_group = |T_end| / max |T_i|
```

and thresholds such as 0.85.

Determine whether token count can increase/decrease for reasons unrelated to semantic information.

Use actual examples from:

* slides
* math
* coding
* scrolling
* UI popups

---

### 5. Does the current dataset justify the proposed 0.85 threshold?

Do NOT tune it.

Instead answer:

What evidence currently supports 0.85?

Was it derived statistically?

Was it derived from annotated data?

Was it chosen heuristically?

Does the current dataset contain enough groups to justify it?

If unsupported, explicitly say so.

---

### 6. Same question for every hard-coded threshold.

For each of these:

* boilerplate >= 70%
* popup token surge > 8
* popup disappearance within 2 frames
* previous/next overlap >= 0.50
* final/max ratio >= 0.85
* token disappearance >= 0.40
* ViT distance >= 0.04
* visual duplicate/near-duplicate thresholds

state:

1. Where did the value come from?
2. Is it present in existing code?
3. Is it supported by empirical data?
4. Is it merely a proposed heuristic?

Do not defend arbitrary numbers.

---

# C. Stress-test OCR as the main representation

### 7. Can OCR token sets represent information preservation adequately?

Analyze:

* duplicate tokens
* repeated tokens
* equations
* superscripts/subscripts
* mathematical operators
* ordering
* indentation
* line structure
* spatial relationships
* diagrams
* handwriting
* charts

Identify exactly what information is lost by:

```
set(Tokens)
```

---

### 8. Does the current OCR representation preserve multiplicity and ordering?

Your audit said the current implementation uses sets.

Verify this directly.

Explain how this affects:

```
token union
token coverage
information loss
frame comparison
```

---

### 9. Does the proposal accidentally reward OCR garbage?

For example:

If OCR incorrectly detects additional tokens in a transient or corrupted frame, could:

```
|T_i|
```

increase

and make that frame look more informative?

Find whether the current code/data contains examples of this.

---

# D. Stress-test transient suppression

### 10. Is the proposed popup detector actually supported by observed data?

The proposal says:

```
token surge > 8
AND disappears within 2 frames
AND surrounding overlap >= 0.50
```

Check actual autocomplete/tooltips/popup examples.

Determine whether:

* they satisfy these conditions
* legitimate short-lived information could also satisfy them
* some observed transient artifacts would NOT satisfy them

Do not redesign the detector yet.

---

### 11. What happens to transient information that is actually pedagogically important?

For example:

* temporary equation
* intermediate calculation
* temporary annotation
* error message
* debugging output

Could the proposed "transient suppression" remove useful information?

Use actual dataset evidence where possible.

---

# E. Stress-test ViT usage

### 12. Does ViT actually provide information useful for Level 2.2 selection?

Do NOT assume it does merely because it exists.

Use the existing examples and answer:

* where ViT distinguishes meaningful visual states
* where ViT fails
* where ViT reacts to irrelevant visual changes
* where OCR and ViT disagree

---

### 13. Is the proposed ViT distance >= 0.04 constraint justified?

Determine whether 0.04 comes from:

* code
* experiments
* distribution analysis
* annotation
* heuristic choice

If it is unsupported, say so.

---

### 14. Is pairwise ViT diversity sufficient for selecting useful keyframes?

The proposal only checks:

```
D_vit(Fa, Fb) >= threshold
```

Determine whether this can distinguish:

* meaningful visual change
* cursor movement
* popup
* scrolling
* typing
* layout shift
* actual diagram/structure change

Use actual evidence.

---

# F. Stress-test the "submodular selection" idea

### 15. What exactly is the optimization problem?

Write down:

* ground set
* objective function
* constraints
* optimization target
* stopping condition
* whether K is fixed or variable

Then determine whether the current proposal actually defines a complete optimization problem.

---

### 16. Can the proposed greedy procedure produce a globally useful subset?

Find possible counterexamples.

For example:

```
Frame A contains information X
Frame B contains information Y
Frame C contains X + Y but is visually poor
```

Would the proposed method select the right frames?

Construct examples using patterns actually present in the dataset where possible.

---

# G. Stress-test Level 2.1 → 2.2 boundary

### 17. Is it actually safe to treat Level 2.1 groups as immutable?

Your proposal says:

"Preserve group boundaries."

But if Level 2.1 accidentally merges two different pedagogical states, Level 2.2 cannot recover from that.

Check the actual A2/B2 outputs.

Identify whether any existing group appears internally heterogeneous.

Do NOT redesign Level 2.1.

Just report whether this is a real dependency/risk.

---

### 18. Which grouping output should Level 2.2 actually consume?

Compare:

* A1
* A2
* B2

using actual results.

Do not simply assume A2/B2.

Explain the consequences of using each as the Level 2.2 input.

---

# H. Stress-test the proposed semantic roles

### 19. Are roles such as:

* initial_setup
* pre_erasure_peak
* final_culmination

actually general enough for all observed domains?

Check:

* slides
* math
* coding
* scrolling
* debugging
* UI interactions

Identify actual patterns that do not fit these roles.

---

# I. Stress-test the confidence score

### 20. The proposed output contains:

```
confidence: 0.95
confidence: 0.98
```

Where would these numbers come from?

Does the current system contain any calibrated confidence model for keyframe selection?

If not, explicitly state that these values would currently be fabricated/heuristic rather than statistically calibrated.

---

# J. Identify the weakest assumptions

### 21. List the 10 weakest assumptions in the current Level 2.2 proposal.

Rank them ONLY by methodological risk.

Do not rank the overall quality of the proposal.

For each:

* assumption
* evidence for it
* evidence against it
* consequence if false

---

# K. Identify what must be experimentally validated

### 22. Before implementing Level 2.2, what hypotheses MUST be tested experimentally?

Separate them into:

1. representation hypotheses
2. dynamic-pattern hypotheses
3. selection hypotheses
4. ViT hypotheses
5. transient-filtering hypotheses
6. evaluation hypotheses

Do not propose the final algorithm.

---

# L. Most important question

### 23. Based on the actual codebase and observed datasets:

What parts of the current Level 2.2 proposal are sufficiently supported to keep?

What parts are premature assumptions?

What parts are directly contradicted by the evidence?

Do NOT redesign them yet.

The goal is to tell an external researcher exactly which parts are evidence-backed and which parts require experiments before becoming architectural decisions.

---

## STRICT RULES

* No code changes.
* No implementation.
* No final Level 2.2 algorithm.
* No invented experiments.
* No invented ground truth.
* No invented thresholds.
* No claims of novelty.
* No "this should work" without evidence.
* Do not defend your previous proposal simply because you proposed it.
* If your previous proposal is wrong, say so explicitly.
* Prefer falsification/counterexamples over confirmation.
* Use actual frame IDs/session names whenever available.

The purpose of this round is NOT to produce a better Level 2.2 plan.

The purpose is to determine which assumptions survive scrutiny before the external reviewer designs the final methodology.
