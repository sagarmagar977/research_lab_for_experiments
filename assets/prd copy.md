# Product Requirement Document (PRD)

## Project Title

**Single-Frame Educational Content Region Extractor & Classifier Test Harness**

---

## 1. Executive Summary & Objective

The goal of this project is to build an interactive single-image test bench using **Streamlit** and a **Deep Learning Vision Engine** (CNN-based DBNet / PaddleOCR Detection) to analyze, crop, and evaluate educational video frames.

Instead of processing entire videos blindly, this tool acts as an experimental harness to test frame taxonomy, fine-tune spatial bounds on individual frames (slides, whiteboards, handwritten notebooks, presenter occlusions, and intro screens), and gracefully handle non-text/empty scenes.

---

## 2. Target Hardware & Execution Constraints

* **Platform Target:** Intel Core i3 (8th Gen CPU) | 20 GB RAM | Integrated Graphics (No Dedicated GPU).
* **Execution Mode:** `Detection-Only` (bypasses Text Recognition/OCR to minimize CPU load).
* **Latency Budget:** $< 150 \text{ ms}$ CPU inference per frame.
* **Memory Footprint:** $< 500 \text{ MB}$ RAM overhead.

---

## 3. Frame Taxonomy & System Behavior

The system classifies and handles five distinct input scenarios:

| Category | Visual Characteristics | Expected Engine Action |
| --- | --- | --- |
| **A. Pure Digital Slide** | High-contrast, clean layout, no human presence. | Tightly crop slide body, removing outer margins. |
| **B. Board + Presenter** | Digital board/whiteboard with presenter in foreground. | Crop text bounds while ignoring human body contours. |
| **C. Physical Notebook / Paper** | Ambient lighting, shadows, handwriting, pen/fingers present. | Isolate written area on page, filtering out desk background. |
| **D. Whiteboard + Glare** | Marker text with glare, fading, or eraser ghosting. | Extract active writing block using local adaptive contrast bounds. |
| **E. Non-Educational / Empty** | YouTube intros, title logos, blank boards, speaker-only shots. | Trigger **`NO_TEXT_DETECTED`** routine and discard/skip frame. |

---

## 4. Operational Workflow & System Architecture

```
                       ┌─────────────────────────┐
                       │   Upload Frame Image    │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Internal Downscale      │ (e.g., max 960px width for fast CPU processing)
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ CNN Detection Backbone  │ (DBNet / MobileNetV3)
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Probability Heatmap     │
                       └────────────┬────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │ Text Regions Found? │
                         └─────┬─────────┬─────┘
                               │         │
                     YES ──────┘         └────── NO
                      │                          │
                      ▼                          ▼
       ┌─────────────────────────┐   ┌───────────────────────────┐
       │ Compute Bounding Box    │   │ Trigger Empty Frame       │
       │ (x_min, y_min, etc.)    │   │ Handling Routine          │
       └────────────┬────────────┘   └───────────┬───────────────┘
                    │                            │
                    ▼                            ▼
       ┌─────────────────────────┐   ┌───────────────────────────┐
       │ Map Coordinates to      │   │ Render UI Warning Banner  │
       │ Original High-Res Image │   │ & Set Status Flag         │
       └────────────┬────────────┘   └───────────────────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Slice Array & Output    │
       │ High-Res Cropped ROI    │
       └─────────────────────────┘

```

---

## 5. Detailed Functional Requirements

### 5.1 Input Module

* **Single-File Uploader:** Drag-and-drop interface accepting `.jpg`, `.jpeg`, and `.png` files.
* **Resolution Support:** Handles 720p, 1080p, and 4K input frames without manual pre-formatting.

### 5.2 Deep Learning Processing Pipeline

1. **Multi-Scale Processing:**
* Downscale input image to an internal max width (e.g., 960px) for neural network inference.
* Calculate scale factors:

$$S_x = \frac{\text{Width}_{\text{original}}}{\text{Width}_{\text{downscaled}}}, \quad S_y = \frac{\text{Height}_{\text{original}}}{\text{Height}_{\text{downscaled}}}$$




2. **Feature Extraction:** Pass image through CNN backbone to produce text probability scores.
3. **Bounding Polygon Calculation:** Group pixels exceeding confidence threshold into bounding coordinates.
4. **Coordinate Slicing & Padding:**
* Extract global outer boundaries:

$$x_{\text{min}} = \min(X) \cdot S_x, \quad x_{\text{max}} = \max(X) \cdot S_x$$


$$y_{\text{min}} = \min(Y) \cdot S_y, \quad y_{\text{max}} = \max(Y) \cdot S_y$$


* Add pixel padding $P$:

$$X_1 = \max(0, x_{\text{min}} - P), \quad Y_1 = \max(0, y_{\text{min}} - P)$$


$$X_2 = \min(\text{Width}_{\text{orig}}, x_{\text{max}} + P), \quad Y_2 = \min(\text{Height}_{\text{orig}}, y_{\text{max}} + P)$$


* Slice high-resolution frame array: `original_image[Y1:Y2, X1:X2]`.



### 5.3 Exception & Empty Frame Handling Routine

If the model detects **zero text regions** (or all detected regions fall below the minimum area threshold):

1. Intercept array slicing to prevent code failure or $0 \times 0$ pixel crop errors.
2. Set application status flag to **`NO_TEXT_REGION_DETECTED`**.
3. Display a warning notification banner in the output column.
4. Provide a configurable UI toggle choice for batch processing mode:
* **Option A (Skip):** Exclude frame from output pipeline (ideal for removing intros/outros).
* **Option B (Pass-Through):** Retain and output the unmodified full original frame.



---

## 6. User Interface Requirements

### 6.1 Interactive Sidebar Controls

| Control Name | Control Type | Default | Purpose |
| --- | --- | --- | --- |
| **Detection Score Threshold** | Slider | `0.30` (`0.10–0.90`) | Filters out low-confidence predictions (shadows, noise). |
| **Box Unclip Ratio** | Slider | `1.50` (`1.0–3.0`) | Controls how aggressively detected text boundaries expand to group adjacent lines. |
| **Padding (px)** | Slider | `15` (`0–100`) | Adds breathing room around the cropped content. |
| **Min Area Filter (% of Frame)** | Slider | `1.0%` (`0.1%–10.0%`) | Filters out tiny artifacts, logos, or isolated text noise. |
| **Empty Frame Strategy** | Dropdown | `Skip Frame` | Defines action when no text is detected (`Skip Frame` vs `Pass-Through Original`). |

### 6.2 Main Workspace Layout (2-Column View)

* **Left Column — Input & Overlay Visualizer:**
* Displays original frame with a high-contrast bounding box drawn around detected content regions.
* Shows intermediate binary mask / heatmap preview for debugging.


* **Right Column — Extracted Output & Analytics:**
* Renders isolated high-resolution cropped image (or warning banner if empty).
* **Diagnostics JSON Panel:** Renders execution metrics:
```json
{
  "Detection Status": "CONTENT_DETECTED",
  "Original Resolution": "3840x2160",
  "Cropped Resolution": "2420x1750",
  "Frame Area Retained": "51.3%",
  "Inference Latency": "82ms",
  "Bounding Coordinates": {"X1": 120, "Y1": 80, "X2": 2540, "Y2": 1830}
}

```





---

## 7. Success Criteria & Edge-Case Benchmark

| Edge Case | Criteria for Success |
| --- | --- |
| **Slide Margins** | Removes $>80\%$ of outer empty whitespace while preserving $100\%$ of text/diagrams. |
| **Presenter Occlusion** | Encloses text on screen/board without expanding bounding box to cover presenter's body. |
| **Notebook Paper** | Successfully isolates handwritten text block, discarding desk surface and binders. |
| **Intro / Blank Frame** | Correctly identifies absence of content, returning `NO_TEXT_REGION_DETECTED` within $<100\text{ms}$. |