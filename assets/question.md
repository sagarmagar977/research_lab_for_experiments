# Pairwise Feature Vector Lab - Architecture Review (Before Version 2)

## Purpose

Before improving the Pairwise Feature Vector Lab, I want to fully understand the current implementation.

The goal is **not** to change anything yet.

I first want to understand:

- current architecture
- feature extraction pipeline
- visualization pipeline
- CSV export pipeline
- limitations
- implementation decisions

Please answer the following questions in as much detail as possible.

---

# 1. Overall Architecture

Please explain the complete execution pipeline when I upload Frame A and Frame B.

Example:

Frame Upload
↓

Image Loading
↓

Preprocessing

↓

Histogram

↓

Edge Detection

↓

Morphology

↓

SSIM

↓

Feature Aggregation

↓

CSV Export

↓

Visualization

Please explain the actual pipeline currently implemented in the project.

---

# 2. Feature Extraction Pipeline

For every feature below, please explain:

- how it is computed
- which function computes it
- what inputs it uses
- whether it depends on another feature

Current features:

- Global Histogram
- Grid Histogram
- Whole Edge Density
- Grid Edge Density
- SSIM
- Morphological Text Mask
- Text Occupancy

---

# 3. Histogram Module

Currently the UI shows

- Global Histogram Comparison
- Grid Histogram Comparison

but both return N/A.

Please explain:

- Why are they N/A?
- Is histogram comparison implemented?
- Is only visualization implemented?
- Is the comparison function returning None?
- Is this a bug or unfinished feature?

Also explain how histogram comparison is intended to work.

---

# 4. Grid Features

How are grid-based features computed?

For example:

Does the code

Split image into N×N cells

↓

Compute feature for every cell

↓

Compare corresponding cells

↓

Average them

or something else?

Please explain.

---

# 5. Edge Features

Currently I see:

Whole Edge Density

Difference

Questions:

- Is this computed on the whole image?
- Is there also a grid-wise edge comparison?
- If yes, how?
- If not, why does the UI mention Grid Edge?

Please clarify.

---

# 6. SSIM

Please explain:

- Whole image or grid?
- Window size
- Gaussian weighting?
- Which library?
- Which parameters?

---

# 7. Morphological Text Mask

Please explain:

How is the text mask generated?

Pipeline?

Example:

Grayscale

↓

Threshold

↓

Morphology

↓

Connected Components

↓

Mask

What measurements are extracted from it?

---

# 8. Text Occupancy

How exactly is Text Occupancy calculated?

Example:

White Pixels

----------------

Total Pixels

?

Or something else?

---

# 9. CSV Export

Please explain exactly how the CSV row is generated.

Questions:

- Which metrics are exported?
- Which metrics are visualization only?
- Which values are computed but not exported?
- Which exported values are currently placeholders?

---

# 10. Visualizations

For every visualization please explain

- what it represents
- what data it uses
- whether it is global or grid based
- whether it affects CSV export

Current visualizations:

- Grid Overlay
- RGB Histogram
- Edge Map
- Morphological Mask
- Difference Image
- SSIM Heatmap
- Feature Table

---

# 11. Hyperparameter Flow

When I change

- Histogram bins
- Grid size
- Blur kernel
- Canny thresholds
- SSIM window size

Which computations are re-run?

Which visualizations update?

Which exported values change?

---

# 12. Module Separation

How is the code organized?

Please describe the project structure.

Example:

feature_engine.py

histogram.py

edge.py

ssim.py

morphology.py

visualization.py

export.py

etc.

---

# 13. Current Limitations

What parts of the current implementation do you think are unfinished?

Examples:

- Placeholder values
- Temporary implementation
- Missing metrics
- Known bugs
- Future TODOs

---

# 14. Future Compatibility

The future research pipeline will be

Video

↓

Text Region Cropping

↓

Candidate Frame Selection

↓

Pairwise Feature Extraction

↓

Dataset CSV

↓

Logistic Regression

Questions:

Does the current Pairwise Feature Vector Lab already produce features compatible with this future dataset?

Or is it only a visualization/debugging prototype?

---

# 15. Design Review

Finally, if you were redesigning this module from scratch, what would you improve?

Please discuss:

- Architecture
- Maintainability
- Performance
- Visualization
- Feature Engineering
- CSV Design
- Extensibility

No implementation yet.

I only want your design thoughts before building Version 2.