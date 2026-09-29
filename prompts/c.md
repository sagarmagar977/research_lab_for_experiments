The specification is close, but do NOT implement yet.

I found several issues that need to be resolved first.

### 1. Remove unsupported hardcoded semantic thresholds

You reintroduced:

```
OCR loss >= 0.40
reverse spatial containment >= 0.60
```

for detecting erasure.

These may be experimental configurable thresholds, but they must NOT be treated as established semantic truth.

Explain how these will be parameterized and evaluated.

---

### 2. Fix the L2.2-D objective inconsistency

Currently you say:

```
Objective = Cov_C(S)
```

but selection uses:

```
Delta_C(F|S) * Q(F)
```

Therefore the optimizer and reported objective are different.

Define a mathematically consistent L2.2-D objective.

The reported coverage and the quantity optimized by greedy selection must correspond.

---

### 3. Reconsider spatial coverage

The 16x16 occupancy grid is acceptable as an experimental representation, but the group-wide union can incorrectly treat presenter movement or transient spatial occupation as instructional information.

Explain how spatial coverage will avoid or at least expose this limitation.

Do not call spatial occupancy "semantic information."

---

### 4. Constrain ViT's role

Keep cached ViT as an optional visual-diversity signal, but do not treat visual difference automatically as information.

Explain:

* exact ViT contribution
* why the chosen formulation is appropriate
* how it is prevented from overpowering lexical/spatial evidence
* how its weight is evaluated experimentally

Do not introduce a universal distance threshold.

---

### 5. Ground-truth evaluation must handle equivalent nearby frames

Exact filename matching can incorrectly mark:

```
human: frame_0408
algorithm: frame_0409
```

as a failure even when both frames represent the same instructional state.

Keep exact-frame precision/recall, but design an additional temporal/state-equivalent evaluation rule.

For example, define a configurable temporal tolerance or equivalent-state matching mechanism.

Explain how this will work without assuming that all nearby frames are equivalent.

---

### 6. Treat 0.95 as an experimental default only

`target_coverage = 0.95` must be explicitly described as an initial configurable experimental value, not a scientifically established threshold.

The evaluation should support sensitivity testing across multiple coverage targets.

---

### 7. Clarify the ablation interpretation

Current progression:

A = set
B = multiset
C = multiset + spatial
D = multiset + spatial + quality
E = multiset + spatial + quality + ViT + event

is acceptable.

However, E-D measures the combined effect of:

```
ViT + event features
```

not ViT alone.

State this correctly.

If you think independent ViT/event ablations are necessary, propose them separately rather than silently changing the five primary modes.

---

### 8. Reconsider event handling

Do not make:

```
OCR loss >= threshold → semantic erasure
```

a definitive classification.

Instead distinguish:

* measurable transition evidence
* heuristic event candidate
* actual human-annotated semantic interpretation

The selector should use event evidence as a supporting feature, not claim that the heuristic has discovered the true semantic event.

---

### 9. Clarify quality handling

The current quality multiplier is acceptable as a starting idea, but explain:

* how Vmin/Vmax are determined
* whether normalization is per-group or per-session
* how extreme blur values behave
* why quality cannot overpower unique information

Also clarify the blank-frame rule separately from blur.

---

### 10. Final deliverable

After making these corrections, provide:

1. Final mathematical definition of A-E.
2. Exact greedy selection algorithm.
3. Exact stopping rule.
4. Exact role of quality.
5. Exact role of spatial information.
6. Exact role of ViT.
7. Exact role of event evidence.
8. Exact ground-truth matching/evaluation method.
9. Final configuration parameters and which are experimental.
10. A short pseudocode implementation of the complete selector.

Do NOT write implementation code yet.

Once this specification is reviewed and confirmed, implementation can begin.
