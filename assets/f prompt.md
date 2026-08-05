The current implementation plan is almost complete. Before implementation, I would like to make one final architectural refinement.

My goal is for the Pairwise Feature Vector Lab to become the permanent feature extraction engine for the entire thesis, not just a Streamlit module.

Please update the architecture with the following improvements.

---

# 1. Introduce a Domain Layer

Separate the project into clear layers.

```
UI Layer (Streamlit)

↓

Application Layer

↓

Feature Extraction Engine

↓

Computer Vision Utilities

↓

Persistence (CSV / JSON)
```

The Pairwise Feature Extractor should never know whether it is being called from Streamlit, a batch script, or a future API.

---

# 2. Separate Computation from Visualization

Feature extraction should never call plotting functions.

Instead,

```
Frame A
Frame B

↓

Feature Engine

↓

FrameFeatures
PairwiseFeatures
VisualArtifacts

↓

Visualization Module
```

Visualizers only receive computed results.

They never compute anything themselves.

---

# 3. Separate Data Export

Instead of the dataclasses creating CSV rows directly,

create a dedicated exporter.

Example:

```python
CSVExporter.export(...)
MarkdownExporter.export(...)
JSONExporter.export(...)
```

This keeps the feature classes focused only on representing data.

---

# 4. Introduce a VisualArtifacts Dataclass

Instead of returning a plain visuals dictionary,

create

```python
@dataclass
class VisualArtifacts:

    edges_a

    edges_b

    text_mask_a

    text_mask_b

    histogram_curves

    ssim_map

    difference_map

    grid_scores
```

Visualizers consume this object.

---

# 5. Add Feature Versioning

Every feature extraction run should contain

```
Feature Engine Version

Feature Schema Version

Experiment Version
```

This allows future datasets to remain compatible even after features evolve.

---

# 6. Validation Framework

Create a validator that automatically checks feature correctness.

Example:

```
FeatureValidator.validate(
    frame_features,
    pairwise_features
)
```

It should detect impossible values.

Examples:

SSIM > 1

Negative histogram distance

Negative edge density

Text occupancy > 100%

etc.

---

# 7. Add Unit-Testable Components

Each extractor should be independently testable.

Example

tests/

test_histogram.py

test_ssim.py

test_edges.py

test_text.py

test_export.py

The PairwiseFeatureExtractor should be an orchestration layer, not the place where algorithms are tested.

---

# 8. Introduce Feature Manifest

Automatically generate a feature manifest.

Example:

```
Feature Name

Description

Formula

Range

Type

Dependencies

Added Version
```

This manifest becomes documentation for both the thesis and future experiments.

---

# 9. Future Compatibility

Design the architecture so that future extractors can be added without changing existing code.

Examples:

OCR Extractor

CNN Extractor

Vision Transformer Extractor

Vision Language Model Extractor

Audio Extractor

Transcript Extractor

Embedding Extractor

The PairwiseFeatureExtractor should simply register them.

---

# 10. Final Goal

The Pairwise Feature Engine should become the core backend used by

• Pairwise Feature Lab

• Batch Dataset Generator

• Candidate Frame Selector

• Model Training Pipeline

• Evaluation Pipeline

• Future Video NotebookLM System

without modifying its public API.

This engine should become the single source of truth for feature extraction throughout the thesis.




## Proposed Long-Term Module Structure

I would also like to slightly reorganize the project so that the Pairwise Comparison system becomes an independent reusable package rather than a single large module.

```
pairwise_feature_lab/
│
├── feature_engine/
│   ├── extractor.py          # PairwiseFeatureExtractor
│   ├── config.py             # PairwiseFeatureConfig
│   ├── models.py             # FrameFeatures, PairwiseFeatures, VisualArtifacts
│   ├── registry.py           # Extractor registration
│   ├── validators.py         # Feature validation rules
│   └── interpretation.py     # Rule-based explanation engine
│
├── extractors/
│   ├── histogram.py
│   ├── edge.py
│   ├── ssim.py
│   └── morphology.py
│
├── visualizers/
│   ├── histogram_view.py
│   ├── edge_view.py
│   ├── ssim_view.py
│   ├── heatmap_view.py
│   └── dashboard.py
│
├── exporters/
│   ├── csv_exporter.py
│   ├── markdown_exporter.py
│   └── json_exporter.py
│
├── utils/
│   ├── image_utils.py
│   ├── cache.py
│   ├── timing.py
│   └── file_utils.py
│
└── pairwise_feature_lab.py
```

### Why I prefer this

The Pairwise Comparison system is becoming the foundation for the thesis.

Later modules should be able to import it directly, for example:

```
candidate_frame_selector/
    ↓
imports PairwiseFeatureExtractor

batch_dataset_generator/
    ↓
imports PairwiseFeatureExtractor

model_training/
    ↓
reads exported CSVs

ocr_experiments/
    ↓
uses selected candidate frames

vit_experiments/
    ↓
uses selected candidate frames

vlm_experiments/
    ↓
uses selected candidate frames
```

The Pairwise Comparison package should therefore become a reusable library that can be imported anywhere in the project rather than being tightly coupled to a Streamlit page.

### Important

This is intended as the long-term architecture only.

There is **no need to move every file immediately**.

The priority should still be completing Version 2 first.

Once the implementation is stable and tested, we can gradually refactor the code into this package structure without changing the public API.

The public API of `PairwiseFeatureExtractor` should remain unchanged during the refactor so that future modules continue to work without modification.