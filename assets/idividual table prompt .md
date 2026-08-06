
Right now our UI jumps from:

**Frame A features**
→ **Pairwise Difference**

without showing the actual source values that produced the pairwise metrics.

A reviewer or even you, six months later, cannot verify where a value like:

* Global RGB Histogram Distance = 3.8220
* Whole Edge Difference = 0.0181

came from.

The pairwise table should always be derivable from the raw measurements.

---

## Prompt

> **Expand the "Individual Frame Scalar Features" table so it becomes the complete raw feature table for each frame instead of containing only five summary statistics.**
>
> The purpose is to expose every raw feature that is later used to compute the pairwise comparison metrics, allowing complete traceability from Frame A and Frame B to every value shown in the Pairwise Metric Statistics table.
>
> ### Requirements
>
> Replace the current limited table with a comprehensive raw feature table.
>
> The table should contain three columns:
>
> ```
> Feature
> Frame A
> Frame B
> ```
>
> Include all raw per-frame features used by the comparison pipeline.
>
> ---
>
> ### Histogram Features
>
> Show the raw histogram descriptors for each frame:
>
> * Global RGB Histogram
>
>   * Mean
>   * Max
>   * Min
>   * Variance
>   * Standard Deviation
> * Global Gray Histogram
>
>   * Mean
>   * Max
>   * Min
>   * Variance
>   * Standard Deviation
> * Grid RGB Histogram
>
>   * Mean
>   * Max
>   * Min
>   * Variance
>   * Standard Deviation
> * Grid Gray Histogram
>
>   * Mean
>   * Max
>   * Min
>   * Variance
>   * Standard Deviation
>
> ---
>
> ### Edge Features
>
> Show the raw edge measurements for each frame.
>
> Include:
>
> * Whole Edge Density
> * Grid Edge Mean
> * Grid Edge Max
> * Grid Edge Min
> * Grid Edge Variance
> * Grid Edge Standard Deviation
>
> ---
>
> ### SSIM Features
>
> Since SSIM is inherently a pairwise metric, do **not** attempt to display per-frame SSIM values.
>
> Leave SSIM statistics exclusively inside the Pairwise Metric Statistics table.
>
> ---
>
> ### Existing Scalar Features
>
> Keep the existing scalar measurements:
>
> * Brightness (Mean)
> * Contrast (Std)
> * Shannon Entropy
> * Edge Density
> * Text Occupancy
>
> ---
>
> ### Pairwise Table
>
> Keep the Pairwise Metric Statistics table exactly as it is.
>
> The values in that table should now be interpretable because every underlying raw measurement is visible in the Individual Frame table.
>
> ---
>
> ### Design Goal
>
> The interface should clearly separate:
>
> **Raw per-frame measurements**
>
> →
>
> **Derived pairwise comparison metrics**
>
> so that every pairwise value can be understood as a transformation or comparison of the raw measurements shown above, improving interpretability and research reproducibility.

---

I would make one small correction to our assumption, though.

> "Everything in pairwise must exist in the individual frame table."

Not quite.

There are **three kinds of features**:

1. **Per-frame features** ✅ (should appear in the Individual table)

   * Brightness
   * Entropy
   * Edge Density
   * Histogram statistics
   * Grid histogram statistics
   * Grid edge statistics
   * Text occupancy

2. **Pair-derived features** ❌ (cannot exist for a single frame)

   * Histogram distance
   * Absolute histogram difference
   * Mean Absolute Difference (MAD)
   * Edge difference
   * Text Occupancy Difference

3. **Intrinsic pair metrics** ❌

   * SSIM
   * SSIM variance
   * SSIM minimum

These are defined only by comparing **two images simultaneously**, so there is no meaningful "Frame A SSIM" or "Frame B SSIM."

So the architecture should be:

```
Frame A
│
├── Raw Features
│
Frame B
│
├── Raw Features
│
───────────────
        │
        ▼
Pairwise Engine
│
├── Histogram Distance
├── Grid Difference
├── MAD
├── Text Difference
├── SSIM
│
▼
Pairwise Metrics
```

That separation is cleaner, mathematically correct, and will make our later batch-processing dataset much easier to understand and maintain.
