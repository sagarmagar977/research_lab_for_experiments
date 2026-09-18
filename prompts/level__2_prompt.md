# Level 2 Implementation — Key-Frame Selection from Level-1 Candidate Frames

## Project Context

We are developing an MCA thesis titled:

**“An Intelligent Video Processing Architecture for Multimodal Information Extraction and Generation.”**

The system processes educational/informational videos.

### Level 1 is already completed

Level 1 takes the original video and performs candidate-frame selection/redundancy reduction.

Its output is a sequence of **unique candidate frames** that are already considered visually non-redundant.

Do **not** redesign or modify Level 1.

Level 2 must operate only on the candidate frames produced by Level 1.

---

# Main Objective of Level 2

The purpose of Level 2 is:

> **Select the minimum set of key frames from the Level-1 candidate frames while preserving the maximum amount of useful information.**

Important distinction:

* Level 1 removes **visual redundancy**.
* Level 2 removes **informational redundancy**.

Do NOT assume that the last frame of a sequence is always the best key frame.

For example:

```text
F1 = Title + Bullet A
F2 = Title + Bullet A + Bullet B
F3 = Title + Bullet A + Bullet B + Bullet C
F4 = Title + Bullet A + Bullet B + Bullet C + Bullet D
```

F4 may replace F1–F3 because it contains their information.

But:

```text
F1 = A + B
F2 = A + B + Diagram X
F3 = A + B + Diagram Y
```

F3 cannot replace F2 because Diagram X disappeared.

Therefore Level 2 must detect **information accumulation and information loss**, rather than simply selecting the final frame.

---

# Level 2 Architecture

Implement Level 2 as three sub-stages:

```text
Level-1 Candidate Frames
        ↓
2.1 Candidate Grouping
        ↓
2.2 Information Coverage Analysis
        ↓
2.3 Key-Frame Selection
        ↓
Final Key Frames
```

Keep each stage modular so that different algorithms/models can be tested independently.

---

# LEVEL 2.1 — Candidate Frame Grouping

## Objective

Group consecutive candidate frames that belong to the same underlying:

* slide
* presentation page
* coding screen
* mathematical page
* document/page
* visual scene
* evolving screen

Example:

```text
F1 F2 F3 F4 F5 → Group 1
F6 F7 F8       → Group 2
F9 F10        → Group 3
```

A new group should begin when there is sufficient evidence that the underlying visual context has changed.

## Features

Initially calculate pairwise features for adjacent candidate frames:

### Visual features

Use a pretrained Vision Transformer (ViT) to extract visual embeddings.

Preferred initial model:

`google/vit-base-patch16-224`

Do NOT train the ViT from scratch.

Calculate a visual distance/similarity between:

```text
ViT(Fi)
ViT(Fi+1)
```

### Image similarity

Calculate:

* SSIM
* optional normalized pixel/image similarity

### OCR/layout features

Use PaddleOCR.

Extract:

* recognized text
* bounding boxes
* confidence
* text-region positions
* number of text regions

Calculate between adjacent frames:

* OCR text similarity
* bounding-box/layout similarity
* number of changed text regions
* newly appearing text
* disappearing text

### Temporal features

Preserve:

* original frame number
* timestamp
* temporal distance between candidate frames

---

# Important

Do not automatically assume ViT is necessary.

Implement the feature system so that we can compare:

### Experiment A

```text
SSIM + OCR/layout features
```

### Experiment B

```text
SSIM + OCR/layout features + ViT features
```

This allows us to test whether ViT actually improves grouping.

---

# Grouping Algorithm

Initially implement a deterministic baseline.

For example:

```text
pairwise similarity
        ↓
threshold
        ↓
same group / new group
```

Do not hard-code arbitrary thresholds without documenting them.

Make all thresholds configurable.

Example configuration:

```yaml
grouping:
  visual_similarity_threshold: ...
  ocr_similarity_threshold: ...
  layout_similarity_threshold: ...
  temporal_constraints: ...
```

Also implement an optional ML classifier interface.

Potential models:

* Decision Tree
* Random Forest
* Extra Trees
* XGBoost

The classification target should be:

```text
0 = same group
1 = new group
```

IMPORTANT:

The label belongs to the **transition between two adjacent frames**, not to the frame itself.

Example:

```text
F1 → F2 = 0
F2 → F3 = 0
F3 → F4 = 1
F4 → F5 = 0
```

Meaning:

```text
F1-F2-F3 = Group 1
F4-F5    = Group 2
```

Do not use alternating labels such as:

```text
Group 1 = 1
Group 2 = 0
Group 3 = 1
```

That is not the correct formulation.

---

# LEVEL 2.2 — Information Coverage Analysis

After grouping, analyze the frames **inside each group**.

The question is:

> Does a new candidate frame add information that is not already sufficiently represented by previously selected frames?

Example:

```text
F1 → A
F2 → A+B
F3 → A+B+C
F4 → A+B+C+D
F5 → A+B+C+D
```

Information gain:

```text
F1 → A
F2 → +B
F3 → +C
F4 → +D
F5 → +nothing
```

F5 is therefore informationally redundant.

But:

```text
F1 → A+B
F2 → A+B+Diagram X
F3 → A+B+Diagram Y
```

requires more than one key frame.

---

# Information Representation

Represent each candidate frame using:

```text
Frame
 ├── OCR text
 ├── OCR bounding boxes
 ├── OCR confidence
 ├── visual embedding
 ├── layout information
 ├── timestamp
 └── frame metadata
```

Use these representations to estimate information overlap.

## OCR comparison

Calculate:

* text overlap
* newly appearing text
* disappearing text
* text-region overlap
* spatial/layout similarity

Do not compare OCR text only as raw strings.

For example:

```text
F1:
Newton's Second Law

F2:
Newton's Second Law
F = ma
```

F2 contains the information from F1 plus new information.

---

# Visual information

Use ViT embeddings to determine visual similarity.

However:

**ViT similarity alone must not determine information coverage.**

A frame can have a similar layout while containing a completely different equation or diagram.

Therefore combine:

```text
visual similarity
+
OCR similarity
+
layout similarity
+
temporal order
```

---

# LEVEL 2.3 — Key-Frame Selection

For every group:

```text
Group
 ↓
candidate frames
 ↓
calculate information contribution
 ↓
select minimum sufficient set
```

The algorithm should prefer a later frame when that later frame genuinely contains the information of earlier frames.

Example:

```text
F1 = A
F2 = A+B
F3 = A+B+C
F4 = A+B+C+D
```

Output:

```text
Key frame = F4
```

But:

```text
F1 = A+B
F2 = A+B+X
F3 = A+B+Y
```

Output could be:

```text
Key frames = F2, F3
```

because neither completely subsumes the other.

---

# Do NOT use a simple “last frame wins” rule

The system must verify information coverage.

A later frame can:

* add information
* preserve information
* remove information
* replace information
* modify information

The algorithm must distinguish these cases.

---

# Output Requirements

Level 2 must produce machine-readable and human-readable outputs.

## 1. JSON

Create structured metadata for every group and selected key frame.

Example:

```json
{
  "group_id": 1,
  "candidate_frames": [
    {
      "frame_id": 1,
      "timestamp": 1.0
    },
    {
      "frame_id": 2,
      "timestamp": 2.0
    }
  ],
  "selected_key_frames": [
    {
      "frame_id": 2,
      "reason": "Contains information accumulated across previous frames"
    }
  ]
}
```

Use a richer schema in the actual implementation.

---

# 2. Markdown

Generate a human-readable `.md` report.

Example:

```markdown
# Level 2 Report

## Group 1

Candidate Frames:
F1, F2, F3, F4

Selected Key Frame:
F4

Reason:
F4 contains the information introduced progressively
in F1-F3.

## Group 2

Candidate Frames:
F5, F6, F7

Selected Key Frames:
F6, F7

Reason:
F6 contains information that is not preserved in F7.
```

This report is for inspection/debugging/research analysis.

---

# 3. Embeddings

Store ViT embeddings in a structured format so they can later be used by Level 3.

Do not immediately require Qdrant.

For Level 2, a local NumPy/JSON/Parquet representation is sufficient.

Qdrant can be introduced later when the Level-3 retrieval architecture is implemented.

---

# Folder Structure

Keep Level 2 separate from Level 1.

Suggested structure:

```text
project/
│
├── level1/
│   ├── ...
│
├── level2/
│   ├── grouping/
│   ├── information_analysis/
│   ├── keyframe_selection/
│   ├── models/
│   ├── features/
│   ├── outputs/
│   └── evaluation/
│
├── data/
│   ├── candidate_frames/
│   └── level2_outputs/
│
└── config/
```

Adapt this to the existing project rather than unnecessarily restructuring the whole repository.

---

# Model Requirements

Do not train deep neural networks from scratch.

Use:

```text
PaddleOCR
+
pretrained ViT
+
traditional ML / algorithms
```

Potential ML classifiers:

```text
Decision Tree
Random Forest
Extra Trees
XGBoost
```

Use them only where classification is actually necessary.

The primary research question is not:

> “Which model is biggest?”

It is:

> “Can multimodal visual/textual features identify informationally redundant candidate frames and reduce them to a smaller key-frame representation without losing important information?”

---

# Evaluation

Build an evaluation dataset from manually labeled candidate frames.

Manual annotation should identify:

1. group boundaries
2. same-group transitions
3. information-preserving frames
4. information-adding frames
5. final key-frame set

For grouping evaluate:

* Accuracy
* Precision
* Recall
* F1
* confusion matrix

For key-frame selection evaluate:

* number of candidate frames
* number of selected key frames
* compression/reduction ratio
* information coverage
* missed information
* redundant selected frames

If possible, manually verify whether the selected key frames preserve the information present across the original candidate-frame sequence.

---

# Experimental Design

Do not assume that ViT improves the system.

Compare:

### Baseline

```text
SSIM + OCR/layout
```

### Proposed visual-enhanced approach

```text
SSIM + OCR/layout + ViT
```

Then compare grouping and key-frame-selection performance.

If ViT provides little improvement, report that honestly rather than forcing it into the final architecture.

---

# Important Engineering Rules

1. Do not modify Level 1.
2. Do not overwrite Level-1 outputs.
3. Preserve frame IDs and timestamps.
4. Preserve chronological ordering.
5. Make every threshold configurable.
6. Save intermediate features.
7. Make the pipeline reproducible.
8. Avoid training models unnecessarily.
9. Keep algorithms modular so features/models can be swapped.
10. Log why every frame was removed or selected.
11. Never silently discard information.
12. Every selected/rejected frame should be traceable to its group and decision.

---

# Final Level-2 Pipeline

Implement this architecture:

```text
LEVEL 1
Unique Candidate Frames
        │
        ▼
┌──────────────────────────────┐
│ LEVEL 2.1                    │
│ Candidate Grouping            │
│                              │
│ ViT + OCR + Layout + SSIM    │
│ + Temporal Features          │
└──────────────┬───────────────┘
               ▼
          Frame Groups
               │
               ▼
┌──────────────────────────────┐
│ LEVEL 2.2                    │
│ Information Coverage          │
│                              │
│ What information does each   │
│ frame add/remove/preserve?   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ LEVEL 2.3                    │
│ Key-Frame Selection           │
│                              │
│ Minimum frames               │
│ Maximum information coverage │
└──────────────┬───────────────┘
               ▼
        FINAL KEY FRAMES
               │
               ▼
     Level 2 JSON + MD +
     visual embeddings
               │
               ▼
             LEVEL 3
```

Before coding, first inspect the **existing Level-1 implementation, output files, frame naming, metadata, and directory structure**. Do not invent a new input format if Level 1 already provides the required information.

First produce a short implementation plan and identify any conflicts between the existing Level-1 outputs and this Level-2 design. Then implement Level 2 incrementally, starting with **2.1 grouping**, test it on a small sample, and only then implement 2.2 and 2.3.
