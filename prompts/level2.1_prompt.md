Implement **Level 2.1 as a parallel A/B experiment**.

### Objective

For the exact same Level-1 candidate-frame input, run **two independent grouping approaches in parallel**:

**Approach A — Baseline**

* OCR text similarity
* OCR bounding-box/layout similarity
* SSIM visual similarity
* Temporal adjacency/order

**Approach B — Baseline + ViT**

* OCR text similarity
* OCR bounding-box/layout similarity
* SSIM visual similarity
* Temporal adjacency/order
* Pretrained ViT visual embedding similarity

Both approaches must produce their **own independent frame groups**.

Example:

```text
Level 1 Candidate Frames
          |
          +--------------------+
          |                    |
          v                    v
   Approach A             Approach B
 OCR + Layout + SSIM   OCR + Layout + SSIM
                       + ViT
          |                    |
          v                    v
      Groups A              Groups B
```

### Manual evaluation

After both approaches finish, create a comparison interface where I can manually inspect:

```text
Approach A:
Group 1 → F1, F2, F3
Group 2 → F4, F5
Group 3 → F6, F7

Approach B:
Group 1 → F1, F2
Group 2 → F3, F4, F5
Group 3 → F6, F7
```

For every group, show:

* frame images
* original frame filename
* parsed video timestamp
* group ID
* transition/boundary decision
* relevant similarity scores

Allow me to manually mark whether the grouping is:

* Correct
* Incorrect
* Boundary should be earlier
* Boundary should be later
* Frames should belong to the same group
* Frames should be separated

The purpose is to let me **visually compare Approach A vs Approach B on real thesis videos and determine which grouping behavior is more useful for Level 2.1**.

### IMPORTANT TIMESTAMP RULE

Level 1 frame filenames already encode the video timestamp.

Examples:

```text
frame_0029.jpg → 00:29
frame_0122.jpg → 02:02
frame_0135.jpg → 02:15
```

Therefore:

**DO NOT calculate timestamps using FPS.**

Parse the numeric portion of the filename as `MM:SS`.

Preserve:

* original frame filename
* candidate-frame sequence/index
* parsed timestamp

Do not modify Level 1.

### ViT requirements

ViT must be **optional**, not mandatory.

Use a pretrained ViT only for Approach B.

Do not train ViT.

If PyTorch/Transformers/ViT dependencies are unavailable, Approach A must still work normally. Make ViT loading modular so it can be enabled/disabled through configuration.

Cache ViT embeddings so they are not recomputed unnecessarily.

### Feature caching

Extract/cache reusable features once:

```text
OCR text
OCR bounding boxes
OCR confidence
SSIM-related features
ViT embedding (Approach B only)
```

Both approaches must use the **same candidate-frame dataset**.

### Ground truth

Do NOT train XGBoost, Random Forest, Extra Trees, Decision Tree, etc. yet.

First allow manual evaluation of the two approaches.

However, save my manual decisions in structured form so they can later become a ground-truth dataset for supervised experiments.

For transition labeling, use:

```text
0 = same group
1 = new group starts after this frame
```

Example:

```text
F1 → F2 = 0
F2 → F3 = 0
F3 → F4 = 1
F4 → F5 = 0
```

### Outputs

For each approach, save:

```text
groups.json
transition_scores.json
features/cache
```

Also create a human-readable report such as:

```text
comparison_report.md
```

The report should clearly separate:

```text
Approach A results
Approach B results
Manual observations
Differences between A and B
```

Do not automatically declare either approach superior.

The system should simply give me **both grouping results side-by-side so I can manually determine which approach performs better on the thesis dataset**.

### Stage boundary

Implement **only Level 2.1** for now.

Do NOT implement:

* Level 2.2 information coverage
* Level 2.3 key-frame selection
* final RAG
* final note generation

First complete the parallel A/B grouping experiment and manual comparison interface.

### Main research purpose

The experiment should answer:

> **Does adding ViT visual representation to the OCR + layout + SSIM baseline produce more useful grouping of candidate frames?**

The answer must come from comparison on the actual thesis videos, not from an assumed preference for ViT.
