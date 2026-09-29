# Level 2.2 Architecture: Representative Keyframe Selection

---

## 1. System Context & Layer Hierarchy

* **Level 1 (Candidate Filtering):** Filters raw video to remove static duplicates, yielding candidate frames with visual/text delta.
* **Level 2.1 (Pedagogical State Grouping):** Partitions candidate frames into coarse, discrete lesson/topic boundaries.
  * **Rule:** 1 Group = 1 Complete Pedagogical Unit (1 Slide, 1 Math Problem, or 1 Coding Program).
  * **Boundary Trigger:** Canvas reset / blank state (full board wipe, empty editor, slide transition).
  * **Intra-Group Continuity:** Scratch-work erasures, code edits, and transient tooltips remain inside the *same* group.
* **Level 2.2 (Representative Keyframe Selection):** Selects the minimal set of culmination keyframes within each group that maximally preserve all information taught during that lesson.

```text
Level 1: Candidate Frames (N frames)
           │
           ▼
Level 2.1: Topic / Lesson Groups (G groups, 1 group = 1 lesson)
           │
           ▼
Level 2.2: Intra-Group Keyframe Selection (1 to K culmination frames per group)
           │
           ▼
Final Educational Keyframe Summary (Zero information loss)
```

---

## 2. Core Principle of Level 2.2: Maximal Successor Subsumption

Within any given group `G = {F_1, F_2, ..., F_m}`:

### A. The Cumulative Monotonic Case (Additive Builds)
In standard slides or continuous code typing, information accumulates monotonically:
* `Info(F_1) ⊆ Info(F_2) ⊆ ... ⊆ Info(F_m)`.
* **Selection Rule:** The final stable successor frame `F_m` subsumes all preceding frames:
  `Subsumption(F_m, G) = 1.0`.
* **Output:** Exactly **1 keyframe** (`F_m`), representing the complete final state of the lesson.

### B. The Destructive / Multi-Stage Case (Scratch-Work Erasures & Code Overwrites)
When an instructor erases scratch work (e.g. Math Q6 factor trees) or replaces code lines to test an alternative:
* `Info(F_early)` contains information that was destroyed before `F_m`.
* Therefore, `F_m` does **not** fully subsume the group: `Subsumption(F_m, G) < 1.0`.
* **Multi-Keyframe Selection Rule:**
  1. Identify the **Culmination Frame of Stage 1** (`F_k`), immediately before the erasure.
  2. Identify the **Culmination Frame of Stage 2** (`F_m`), containing the final answer.
  3. Joint coverage satisfies: `Info(F_k) ∪ Info(F_m) ≈ Info(G)`.
* **Output:** Exactly **2 keyframes** for that group. Zero pedagogical steps are lost.

---

## 3. Mathematical Formulation for Level 2.2

### 3.1 Pairwise Information Subsumption Matrix
For any two frames `F_a` and `F_b` within group `G`:

```text
Subsumes(F_b, F_a) = |Tokens(F_a) ∩ Tokens(F_b)| / |Tokens(F_a)|
```
`Subsumes(F_b, F_a) = 1.0` means frame `F_b` contains everything present in `F_a`.

### 3.2 Visual & Text Stability Score
A frame `F_i` is a valid culmination candidate if it is free of transient artifacts:

```python
Stability(F_i) = 1.0 - (w_motion * MotionBlur(F_i) + w_popup * IsTransientPopup(F_i))
```

### 3.3 Greedy Submodular Keyframe Coverage Algorithm

```python
def select_group_keyframes(group_frames, tau_coverage=0.90):
    """
    Selects minimal keyframes from group_frames to cover >= tau_coverage of group tokens.
    """
    all_group_tokens = set().union(*[f["tokens"] for f in group_frames])
    selected_keyframes = []
    covered_tokens = set()

    # Iterate until all pedagogical tokens taught in this group are preserved
    while len(covered_tokens) / max(len(all_group_tokens), 1) < tau_coverage:
        best_candidate = None
        best_marginal_gain = -1

        for f in reversed(group_frames):  # Prefer later culmination frames
            if f in selected_keyframes or f["is_transient"]:
                continue
            marginal_gain = len(f["tokens"] - covered_tokens)
            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_candidate = f

        if best_candidate is None or best_marginal_gain == 0:
            break

        selected_keyframes.append(best_candidate)
        covered_tokens.update(best_candidate["tokens"])

    # Return keyframes in chronological order
    return sorted(selected_keyframes, key=lambda x: x["timestamp_sec"])
```

---

## 4. Domain-Specific Behavior in Level 2.2

| Domain | Group Event in Level 2.1 | Level 2.2 Keyframe Decision | Output Keyframes |
|---|---|---|---|
| **Slides** | Progressive bullet reveals across 1 slide | Final frame contains all bullets | **1 Keyframe** (Complete Slide) |
| **Math** | Standard problem derivation | Final frame contains problem + boxed answer | **1 Keyframe** (Full Solution) |
| **Math** | Problem with erased scratch-tree (Q6) | Stage 1 culmination + Stage 2 final answer | **2 Keyframes** (Step 1 + Final Step) |
| **Coding** | Typing variable definitions | Final frame contains fully typed program + terminal run | **1 Keyframe** (Complete Script & Output) |
| **Coding** | Program written, then modified to test error (`176 -> 178`) | Frame 176 (working code) + Frame 178 (error test) | **2 Keyframes** (Working Code + Error Output) |

---

## 5. Verification Checklist for Implementation

* [ ] Candidate group input loaded strictly from Level 2.1 partitions.
* [ ] Monotonic additive groups yield exactly 1 representative frame.
* [ ] Groups with destructive intra-group edits dynamically yield K >= 2 keyframes.
* [ ] Transient UI popups (autocomplete, tooltips) are strictly vetoed from selection.
* [ ] Selected keyframes are tagged with audit metadata: `group_id`, `culmination_stage`, `token_coverage_ratio`.
