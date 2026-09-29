

### A. Current Level 2.1 Implementation

#### 1. Real Execution Flow & Architecture
```text
Candidate Frames (L1 Sessions / Folders)
  │
  ▼
Stage 0: Feature Extraction & Caching [modules/l2/feature_extractor.py]
  ├─ extract_single_frame_features(): RapidOCR ONNX -> tokens, bboxes, confidences
  ├─ extract_single_frame_vit_embedding(): ViT-B/16 -> 768-d unit vector
  └─ compute_pairwise_transition_features(): Consecutive pair metrics (SSIM, OCR Jaccard, Asymmetric Containment, Layout IoU, ViT Cosine)
  │
  ▼
Persistence Layer [sessions/<name>/level2/]
  ├─ features_cache.json (frame metadata + transition records)
  ├─ vit_embeddings.npz (NumPy compressed embeddings)
  └─ l2_pairwise_features.csv (flat transition export)
  │
  ▼
Stage 2.1: Parallel Grouping Engine [modules/l2/grouping_engine.py]
  ├─ run_approach_a1(): Frozen baseline (SSIM + Symmetric OCR Jaccard + Symmetric Layout IoU)
  ├─ run_approach_a2(): Asymmetric containment + Dynamic weight renormalization + SSIM gate + Lookahead bridge + Lesson boundary constraints
  ├─ run_approach_b2(): A2 formulation + ViT cosine distance
  ├─ partition_groups(): Partitions sequence into contiguous group dictionaries
  └─ find_divergences_3way(): Computes pairwise disagreement transitions
  │
  ▼
Output & UI Presentation [modules/l2_grouping_lab.py]
  ├─ grouping_runs.json (full 3-way partition payloads)
  └─ Streamlit UI: Group browser, 3-way divergence studio, annotation studio
```

#### 2. Frame Object Schema
From [modules/l2/feature_extractor.py:141-155](file:///f:/THESIS/prototype%20for%20thesis/candidate%20frame%20selction%20lab/modules/l2/feature_extractor.py#L141-L155) and `features_cache.json`:

| Feature Query | Present in Frame Object? | Code / Schema Field Name |
| :--- | :--- | :--- |
| **Frame ID** | Yes | `filename` (str), `original_frame_id` (int), `candidate_seq_idx` (int) |
| **Timestamp** | Yes | `timestamp_sec` (int), `timestamp_str` (str, formatted `"MM:SS"`) |
| **OCR Text** | Yes | `full_text` (str) |
| **OCR Tokens** | Yes | `tokens` (list of str) |
| **OCR Confidence** | Yes | `confidences` (list of float, per detected box) |
| **Bounding Boxes** | Yes | `bboxes` (list of 4-point normalized coords `[[x, y], ...]`) |
| **Font Height** | **No** | *Not available from current codebase.* |
| **Layout Features** | Yes | `bboxes` (list), `num_boxes` (int), `img_h` (int), `img_w` (int) |
| **SSIM** | Transition only | Stored in transition record as `d_ssim` (`1.0 - SSIM`), not on single frame |
| **Histogram** | **No** | *Not available from current codebase.* |
| **Edge Features** | **No** | *Not available from current codebase in Level 2.1* (existed only in legacy L1) |
| **ViT Embedding** | Separate file | Stored in `vit_embeddings.npz` as `(768,) float32` array keyed by `filename` |
| **ViT Cosine Similarity** | Transition only | Stored in transition record as `d_vit_cosine` |
| **Image Path** | Reconstructed | Built dynamically from `session_path + candidate_folder + filename` |
| **Quality / Sharpness** | **No** | *Not available from current codebase.* |

#### 3. Output Persistence & JSON Structure
Output directory: `sessions/<session_name>/level2/`

Files generated:
1. `features_cache.json` (~1.0–3.3 MB):
```json
{
  "num_frames": 505,
  "num_transitions": 504,
  "frames": [
    {
      "filename": "frame_0008.jpg",
      "candidate_seq_idx": 0,
      "original_frame_id": 8,
      "timestamp_sec": 8,
      "timestamp_str": "00:08",
      "tokens": ["x"],
      "full_text": "X",
      "bboxes": [[[0.0828, 0.0833], [0.1266, 0.0833], [0.1266, 0.1583], [0.0828, 0.1583]]],
      "confidences": [0.5616],
      "num_tokens": 1,
      "num_boxes": 1,
      "img_h": 360,
      "img_w": 640
    }
  ],
  "transitions": [
    {
      "pair_idx": 1,
      "frame_a": "frame_0008.jpg",
      "frame_b": "frame_0009.jpg",
      "original_frame_id_a": 8,
      "original_frame_id_b": 9,
      "timestamp_a": "00:08",
      "timestamp_b": "00:09",
      "delta_time_sec": 1,
      "token_count_a": 1,
      "token_count_b": 1,
      "token_count_diff": 0,
      "d_ssim": 0.0076,
      "d_ocr_jaccard": 0.0,
      "ocr_preservation": 1.0,
      "ocr_loss": 0.0,
      "ocr_valid_raw": true,
      "d_layout_iou": 0.0412,
      "layout_preservation": 0.985,
      "layout_loss": 0.015,
      "layout_valid": true,
      "d_vit_cosine": 0.0031
    }
  ]
}
```

2. `grouping_runs.json` (~4.4 MB for 505 frames):
```json
{
  "session_name": "MATH-SESSION",
  "updated_at": "2026-09-24T03:58:53.340780",
  "approach_a1": { "config": { "ssim": 0.4, "ocr": 0.4, "layout": 0.2, "threshold": 0.35 }, "num_groups": 80, "groups": [...], "scores": [...], "preds": [...] },
  "approach_a2": { "config": { "ssim": 0.4, "ocr": 0.4, "layout": 0.2, "threshold": 0.35, "min_tokens": 3, "use_layout": true, "ssim_gate": 0.025, "use_boilerplate": true, "use_transient_bridge": true }, "num_groups": 8, "groups": [...], "scores": [...], "preds": [...] },
  "approach_b2": { "config": { "ssim": 1.0, "ocr": 1.0, "layout": 1.0, "vit": 0.5, "threshold": 0.35, "min_tokens": 3, "use_layout": true, "ssim_gate": 0.025, "use_boilerplate": true, "use_transient_bridge": true }, "num_groups": 8, "groups": [...], "scores": [...], "preds": [...] },
  "divergences_count": 72
}
```

---

### B. Current OCR Implementation

#### 4. OCR Execution Details
* **Library / Model:** `rapidocr_onnxruntime.RapidOCR(det_limit_side_len=960, det_limit_type="max")` (ONNX runtime wrapping PaddleOCR models).
* **Preprocessing:** `cv2.imread(img_path)` -> converted to RGB via `cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)`. Maximum side dimension resized internally to 960px by RapidOCR.
* **Tokenization:** Regex extraction: `re.findall(r'\b[a-zA-Z0-9_\-\$]+\b', text.lower())`.
* **Coordinate Normalization:** Normalized to `[0.0, 1.0]` by dividing pixel coordinates by `img_w` and `img_h`, rounded to 4 decimal places.
* **Confidence Handling:** Float score per detected box stored in `confidences`. No minimum confidence thresholding or filtering is applied.
* **Minimum Token Rule:** Configured via `min_tokens` parameter (default = 3). If `token_count_a < min_tokens`, OCR is flagged `ocr_valid = False` and excluded during dynamic weight renormalization.
* **Execution & Caching:** Run **strictly once** per candidate frame in Stage 0 and persisted to `features_cache.json`. Subsequent grouping runs take 0 ms.
* **Storage Location:** `sessions/<session_name>/level2/features_cache.json`.

#### 5. Directional Containment Support
* **Status:** Supported and actively implemented.
* **Implementation:**
  * In [modules/l2/feature_extractor.py:284-287](file:///f:/THESIS/prototype%20for%20thesis/candidate%20frame%20selction%20lab/modules/l2/feature_extractor.py#L284-L287):
    `ocr_preservation = min(1.0, max(0.0, len(tokens_a & tokens_b) / len(tokens_a)))`
    `ocr_loss = 1.0 - ocr_preservation`
  * In [modules/l2/grouping_engine.py:318-321](file:///f:/THESIS/prototype%20for%20thesis/candidate%20frame%20selction%20lab/modules/l2/grouping_engine.py#L318-L321):
    Reverse spatial mask containment: `C_rev = Area(M_a & M_b) / Area(M_b) >= 0.70`.
* **Missing Elements:**
  * Multiset frequencies: Uses `set()`, losing duplicate word counts.
  * Line and layout hierarchy: Reading order and vertical structure are lost.
  * Fuzzy matching: Exact string match only; OCR character recognition noise reduces intersection.

#### 6. Handling of OCR Failures
* **Empty text (`num_tokens == 0`):** `ocr_valid_raw` set to `False`; `ocr_preservation` and `ocr_loss` set to `None`. Dynamic weight renormalization drops OCR weight to 0.0 and divides composite score by remaining valid weights (`w_ssim + w_layout`).
* **Very few tokens (`num_tokens < min_tokens`):** Marked `ocr_valid = False` in `grouping_engine.py`. OCR weight is omitted; denominator drops to `w_ssim (+ w_layout)`.
* **Low confidence:** Recorded in `confidences` array, but no discarding occurs. Low-confidence text is treated identically to high-confidence text.
* **Garbage OCR:** If random symbols match `\b[a-zA-Z0-9_\-\$]+\b`, they enter the token set. If garbage differs between frames, `ocr_loss` surges to 1.0, but is vetoed if visual canvas is static via SSIM gate (`d_ssim < 0.025`).
* **Text in only one frame:**
  * `Frame A empty, Frame B has text`: `tokens_a == 0` -> OCR invalid -> split decided purely by SSIM / Layout.
  * `Frame A has text, Frame B empty`: `ocr_preservation = 0.0`, `ocr_loss = 1.0` -> triggers boundary split unless vetoed by SSIM gate.

---

### C. Current ViT Implementation

#### 7. ViT Workflow & Model Details
```text
Image Path
  │
  ▼
PIL.Image.open(img_path).convert("RGB")
  │
  ▼
ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
  │
  ▼
ViTModel.from_pretrained("google/vit-base-patch16-224") [torch.no_grad()]
  │
  ▼
Embedding Extraction: outputs.pooler_output[0] (or outputs.last_hidden_state[0, 0])
  │
  ▼
L2 Normalization: emb = emb / np.linalg.norm(emb) -> (768,) float32
  │
  ▼
Pairwise Distance: d_vit = (1.0 - np.dot(u, v)) / 2.0
  │
  ▼
Grouping Decision: B2 composite score: (w_ssim * d_s + w_ocr * ocr_loss + w_layout * layout_loss + w_vit * d_v) / total_w
```
* **Exact Model Name:** `google/vit-base-patch16-224`.

#### 8. Usage Mode: Pairwise vs Group Representation
* **Mode:** Strictly **pairwise consecutive distance** between adjacent frames:
  `d_vit = (1.0 - cos_sim(F_i, F_{i+1})) / 2.0`
* It does **not** compute centroid, group-level pooling, or comparison against a keyframe reference.

#### 9. ViT Embedding Caching
* **Storage Location:** `sessions/<session_name>/level2/vit_embeddings.npz`.
* **Format:** Compressed NumPy archive (`np.savez_compressed`), storing unit-normalized vectors keyed by filename.
* **Dimensions:** 768 dimensions per frame (`float32`).
* **Disk Size:**
  * `slides_AA` (225 frames): 679.8 KB (~3.0 KB / frame)
  * `CODING_SESSION` (334 frames): 1,008.3 KB (~3.0 KB / frame)
  * `MATH-SESSION` (505 frames): 1,525.1 KB (~3.0 KB / frame)
* **Extraction Latency:** ~20–40 ms/frame on GPU (NVIDIA CUDA), ~150–250 ms/frame on CPU. Done once in Stage 0. Subsequent grouping queries execute in 0 ms.

#### 10. ViT vs. OCR Discriminative Capacity
Empirical examples from session caches:
1. **OCR changes, but visual content is essentially identical (ViT correctly invariant):**
   * Session: `slides_AA`, transition `frame_0091.png -> frame_0092.png`.
   * Event: One new line of bullet text added to slide.
   * Metrics: `d_ocr_jaccard = 0.4324`, but `d_vit_cosine = 0.0085` and `d_ssim = 0.0278`.
   * Result: ViT correctly reflects that global visual scene did not change.
2. **OCR ≈ same, but visual content changes (ViT detects visual shift):**
   * Session: `slides_AA`, transition `frame_0704.png -> frame_0733.png`.
   * Event: Slide switches from text summary to full-screen architecture diagram.
   * Metrics: `d_vit_cosine = 0.2126`, `d_ssim = 0.6549`. ViT captures the graphic shift.
3. **Limitation of ViT (Blind to code edits):**
   * Session: `CODING_SESSION`, transition `frame_0176.jpg -> frame_0178.jpg`.
   * Event: Major code modification in editor (variable definition replaced with print operation).
   * Metrics: `d_ocr_jaccard = 0.5278`, but `d_vit_cosine = 0.0384`.
   * Result: ViT is largely blind to code syntax changes because the dark IDE background constitutes >90% of pixel area.

---

### D. Existing Datasets / Sessions

#### 11. Test Sessions Summary
| Session Name | Candidate Frames | A1 Groups | A2 Groups | B2 Groups | Group Size (A2) | Timestamp Span | OCR Available | ViT Available |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **slides_AA** | 225 | 33 | 33 | 33 | Min: 1, Max: 10, Mean: 6.8 | 01:28 -> 12:13 | Yes | Yes (679.8 KB) |
| **MATH-SESSION** | 505 | 80 | 8 | 8 | Min: 34, Max: 124, Mean: 63.1 | 00:08 -> 12:02 | Yes | Yes (1,525.1 KB) |
| **CODING_SESSION** | 334 | 24 | 11 | 11 | Min: 7, Max: 49, Mean: 30.4 | 00:01 -> 10:13 | Yes | Yes (1,008.3 KB) |

#### 12. Progressive-Information Examples
1. `slides_AA` Group 8 (`03:19` -> `03:32`):
   * `frame_0199.png`: Slide title + bullet point 1.
   * `frame_0205.png`: Slide title + bullet point 1 + bullet point 2.
   * `frame_0212.png`: Slide title + bullet points 1, 2, and 3 (culminating slide state).
2. `MATH-SESSION` Group 6 (`04:47` -> `06:48`):
   * `frame_0287.jpg`: Question 5 header written.
   * `frame_0350.jpg`: Quadratic equation expansion.
   * `frame_0408.jpg`: Quadratic equation + factor tree branches (`-120`) written in full.

#### 13. Destructive-Information Examples
1. `MATH-SESSION` Group 6 (`frame_0408.jpg -> frame_0413.jpg`):
   * `frame_0408.jpg` (20 tokens): Contains full factor tree (`-120`, factor pairs).
   * `frame_0413.jpg` (4 tokens): Instructor erases top half of blackboard to create space to write roots `(x+...)(4x-5)=0`. Intermediate factor tree disappears permanently.
2. `CODING_SESSION` Group 3 (`frame_0176.jpg -> frame_0178.jpg -> frame_0196.jpg`):
   * `frame_0176.jpg` (53 tokens): Clean, working script (`integer = 2021`, `name = '...'`).
   * `frame_0178.jpg` (80 tokens): Code edited to introduce a type error for teaching purposes.
   * `frame_0196.jpg` (93 tokens): Terminal displays traceback `TypeError: can only concatenate str (not "int") to str`. The original working code is overwritten.

#### 14. Cases Where Final Frame is NOT the Best Representative
1. `MATH-SESSION` Group 6:
   * Final frame: `frame_0459.jpg` (`07:39`).
   * Why flawed: Displays only final root solutions `X=-3`. The factor tree from `frame_0408.jpg` was wiped away 50 seconds earlier.
2. `CODING_SESSION` Group 3:
   * Final frame: `frame_0196.jpg` (`03:16`).
   * Why flawed: Terminal pane fills the lower screen with red error traceback; working implementation from `frame_0176.jpg` is obscured/replaced.
3. `CODING_SESSION` Group 1:
   * `frame_0076.jpg` (`01:16`): Contains an autocomplete popup menu obscuring editor text. `frame_0094.jpg` (`01:34`) has cursor artifacts.

#### 15. Cases Where One Frame is Clearly Insufficient
* **`MATH-SESSION` Group 6 (124 frames):** Requires at least 2 frames:
  1. `frame_0408.jpg` (`06:48`): Peak pre-erasure information (equation setup + factor tree).
  2. `frame_0459.jpg` (`07:39`): Final post-erasure solution (roots of the equation).
* **`CODING_SESSION` Group 3 (38 frames):** Requires at least 2 frames:
  1. `frame_0176.jpg` (`02:56`): Functional source code.
  2. `frame_0196.jpg` (`03:16`): Terminal execution result / exception demonstration.

---

### E. Scrolling / Terminal / Dynamic Content

#### 16. Scrolling Content in Existing Datasets
* **Presence:** Yes. Observed in `CODING_SESSION` during terminal execution (`02:50 - 03:16`, `07:15 - 08:10`) where terminal logs scroll upwards.
* **Effect on OCR:**
  * Top lines leave the terminal viewport, dropping out of `tokens` (`token_count_diff > 15`, `ocr_loss` increases).
  * Bottom lines enter the terminal viewport.
  * Code editor pane at the top remains static, resulting in ~50–60% token overlap but spatial layout degradation (`d_layout > 0.40`). In baseline A1, this caused 24 fragmented groups.

#### 17. Spatial Layout Representation
* **Availability:** Yes, bounding boxes are extracted by RapidOCR and stored in `features_cache.json`.
* **Coordinate Space:** Strictly **normalized coordinates** in `[0.0, 1.0]` (floats rounded to 4 decimal places):
  `norm_box = [[round(x / img_w, 4), round(y / img_h, 4)], ...]`.
* **Algorithm Utilization:**
  * A1 uses symmetric polygon rasterization IoU: `compute_layout_mask_iou(bboxes_a, bboxes_b)`.
  * A2 and B2 compute directional layout preservation: `P_layout = Area(M_a & M_b) / Area(M_a)`.
  * Lesson boundary engine uses bounding box coordinates to filter editor content: `0.08 <= x <= 0.95`, `0.08 <= y <= 0.62`.

---

### F. Current Quality Signals

#### 18. Inventory of Existing Quality Measures
* **Blur / Laplacian Variance:** *Not available from current codebase in Level 2.1.* (Gaussian blur was used solely as a pre-filter for Canny edge detection in L1).
* **Sharpness:** *Not available from current codebase.*
* **Occlusion Metrics:** *Not available from current codebase.*
* **Black / Blank Frame Detection:** *Not explicitly implemented.* Only checked via empty token counts (`num_tokens == 0`) or empty boxes (`num_boxes == 0`).
* **OCR Confidence:** Extracted per detected box and stored in `confidences` list in `features_cache.json`.
* **Visual Stability:** Measured via consecutive pairwise SSIM distance (`d_ssim = 1.0 - SSIM`). GroupingEngine includes `ssim_gate_thresh` veto (default 0.025) to suppress splits when canvas is visually static.
* **Duplicate Detection:** Handled implicitly via `d_ssim == 0.0` or near-zero distance transitions.

---

### G. Current Level 2.1 Behavior

#### 19. Exact Formulations in Code
##### A1 (Frozen Baseline)
* Formula: `comp_score = (w_ssim * d_s + w_ocr * d_o + w_layout * d_l) / (w_ssim + w_ocr + w_layout)`
* Distance definitions:
  * `d_s = 1.0 - SSIM`
  * `d_o = 1.0 - |T_a & T_b| / |T_a | T_b|` (Symmetric Jaccard)
  * `d_l = 1.0 - IoU(M_a, M_b)` (Symmetric Layout IoU)
* Decision: `pred = 1` if `comp_score >= threshold` else `0`.
* Fallbacks: None. OCR and layout failures default to 0.0 or 1.0.

##### A2 (Improved Asymmetric)
* Formula: `comp_score = (w_ssim * d_s + w_ocr * ocr_loss + w_layout * layout_loss) / denom`
* Distance definitions:
  * `ocr_loss = 1.0 - |T_a & T_b| / |T_a|` (Directional text loss)
  * `layout_loss = 1.0 - Area(M_a & M_b) / Area(M_a)` (Directional layout loss)
* Dynamic Renormalization:
  * If `token_count_a < min_tokens` or `ocr_loss is None`, OCR is excluded from numerator and denominator (`denom` drops `w_ocr`).
  * If layout is disabled or invalid, layout is excluded (`denom` drops `w_layout`).
* SSIM Consistency Gate: If `d_s < ssim_gate_thresh` (default 0.025), split is vetoed (`pred = 0`).
* Lookahead Bridge: If tokens surge in `F_i` and collapse in `F_{i+1}` (`len(T_i) > len(T_{i-1}) + 8` and `len(T_{i-1} & T_{i+1}) / len(T_{i-1}) >= 0.50`), split is vetoed (`pred = 0`).
* Lesson Boundary Constraints:
  * Blackboard Reverse Spatial Containment: If `len(bp_set) <= 2` and `Area(M_a & M_b) / Area(M_b) >= 0.70`, veto split (`pred = 0`).
  * IDE Canvas Reset: If `is_ide` and editor bounding box lines drop to `<= 1` after `>= 4`, trigger group split (`pred = 1`).

##### B2 (Multimodal Asymmetric)
* Formula: `comp_score = (w_ssim * d_s + w_ocr * ocr_loss + w_layout * layout_loss + w_vit * d_v) / denom`
* ViT Distance: `d_v = (1.0 - cos_sim(emb_a, emb_b)) / 2.0`
* Dynamic Renormalization: If ViT is unavailable or `d_v is None`, ViT is excluded from numerator and denominator.
* Applies identical SSIM gate, lookahead bridge, and lesson boundary constraints as A2.

#### 20. Sliders and Configuration Parameters
From [modules/l2_grouping_lab.py:270-322](file:///f:/THESIS/prototype%20for%20thesis/candidate%20frame%20selction%20lab/modules/l2_grouping_lab.py#L270-L322):

| Parameter | Formulation | Default Value | Tunable Range | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `w_ssim_a1` | A1 | 0.40 | 0.0 – 2.0 | Grayscale SSIM distance weight |
| `w_ocr_a1` | A1 | 0.40 | 0.0 – 2.0 | Symmetric OCR Jaccard weight |
| `w_layout_a1` | A1 | 0.20 | 0.0 – 2.0 | Symmetric layout IoU weight |
| `tau_a1` | A1 | 0.35 | 0.05 – 0.95 | Binary split threshold |
| `w_ssim_a2` | A2 | 0.40 | 0.0 – 2.0 | SSIM distance weight |
| `w_ocr_a2` | A2 | 0.40 | 0.0 – 2.0 | Asymmetric OCR containment weight |
| `w_layout_a2` | A2 | 0.20 | 0.0 – 2.0 | Directional layout loss weight |
| `min_tok_a2` | A2 | 3 | 1 – 20 | Minimum token validity guard |
| `use_layout_a2` | A2 | True | Boolean | Enable/disable directional layout |
| `tau_a2` | A2 | 0.35 | 0.05 – 0.95 | Binary split threshold |
| `ssim_gate_a2` | A2 | 0.025 | 0.000 – 0.100 | Visual static canvas split veto |
| `use_bp_a2` | A2 | True | Boolean | Filter tokens appearing in >=70% frames |
| `use_bridge_a2` | A2 | True | Boolean | 3-frame lookahead for transient popups |
| `w_ssim_b2` | B2 | 1.00 | 0.0 – 5.0 | SSIM distance weight |
| `w_ocr_b2` | B2 | 1.00 | 0.0 – 5.0 | Asymmetric OCR containment weight |
| `w_layout_b2` | B2 | 1.00 | 0.0 – 5.0 | Directional layout loss weight |
| `w_vit_b2` | B2 | 0.50 | 0.0 – 5.0 | Pretrained ViT cosine distance weight |
| `tau_b2` | B2 | 0.35 | 0.05 – 0.95 | Binary split threshold |
| `ssim_gate_b2` | B2 | 0.025 | 0.000 – 0.100 | Visual static canvas split veto |
| `use_bp_b2` | B2 | True | Boolean | Boilerplate filter |
| `use_bridge_b2` | B2 | True | Boolean | 3-frame lookahead for transient popups |

---

### H. Existing Evaluation

#### 21. Current Evaluation Mechanism
* **Manual Ground Truth Studio:** Built into `modules/l2_grouping_lab.py` Section 6. Allows researchers to annotate transitions with `0 = Same Group` or `1 = New Group` along with notes.
* **Persistence:** Saves annotations and feature matrices to `sessions/<name>/level2/ground_truth_transitions.csv` via `save_ground_truth_dataset()`.
* **Metrics:** `GroupingEngine.compute_metrics()` calculates Precision, Recall, F1, Accuracy, TP, FP, FN, TN when ground truth labels are provided.
* **Academic Reporting:** Generates `comparison_report.md` comparing A1, A2, and B2.
* **Current Database State:** No `ground_truth_transitions.csv` files have been saved to disk yet by the user. Current evaluation is conducted via visual inspection in the Triple Group Browser and Divergence Studio.

#### 22. Known Failure Cases
* **OCR Failures:**
  * Handwritten math superscripts, subscripts, and square roots are frequently omitted or merged into noisy tokens (e.g. `x^2` recognized as `x~` or `X`).
  * Chalk smudges and partial erasures generate spurious single-character tokens.
* **ViT Failures:**
  * Dark IDE themes dominate image feature representations; ViT cosine distance between different code blocks remains negligible (`d_vit < 0.04`).
* **Grouping Failures:**
  * Baseline A1 over-splits on monotonic text additions (80 groups on `MATH-SESSION` vs. 8 true pedagogical units).
* **Data Problems:**
  * 1 fps frame extraction includes duplicate frames during instructor pauses.
  * Autocomplete menus and tooltips appear for 1–2 frames, temporarily altering token counts.
* **UI Problems:**
  * Rendering hundreds of full-resolution candidate frames in Streamlit causes browser tab freezing. Resolved by caching JPEG Base64 thumbnails at 220px max dimension.

---

### I. Computational Constraints

#### 23. Pipeline Computational Cost
* **Candidate Frame Counts:** `slides_AA` = 225 frames; `CODING_SESSION` = 334 frames; `MATH-SESSION` = 505 frames.
* **OCR Runtime (RapidOCR ONNX):** ~30–70 ms/frame on CPU (~15–30s total per session).
* **ViT Embedding Runtime (`vit-base-patch16-224`):** ~25 ms/frame on GPU (~5–10s total), ~200 ms/frame on CPU (~60–90s total).
* **Pairwise Feature Computation:** < 10 ms for all 500 transitions combined.
* **Memory Footprint:** Peak RAM < 1.5 GB during ViT extraction; < 400 MB during Streamlit interactive grouping.
* **Interactive Latency:** Recalculating grouping upon slider adjustments takes < 25 milliseconds because all features are precomputed and cached.

#### 24. Reusable Components for Level 2.2
Level 2.2 can execute instantly with zero neural network inference by reusing:
1. `features_cache.json`: All OCR tokens, full text, normalized bounding boxes, confidences, and transition metrics (`d_ssim`, `ocr_preservation`, `ocr_loss`, `layout_preservation`).
2. `vit_embeddings.npz`: All 768-d unit-normalized feature vectors.
3. `grouping_runs.json`: Partitioned group structures, start/end frames, and frame sequences.

---

### J. Existing Architecture Boundaries

#### 25. Input Interface for Level 2.2
Level 2.2 should receive the active partition payload from Level 2.1:
```python
groups: list[dict] = [
    {
        "group_id": int,
        "frame_count": int,
        "start_frame": str,
        "end_frame": str,
        "start_timestamp": str,
        "end_timestamp": str,
        "duration_sec": int,
        "frames": [
            {
                "filename": str,
                "candidate_seq_idx": int,
                "original_frame_id": int,
                "timestamp_sec": int,
                "timestamp_str": str,
                "tokens": list[str],
                "full_text": str,
                "bboxes": list[list[list[float]]],  # Normalized [0, 1]
                "confidences": list[float],
                "num_tokens": int,
                "num_boxes": int,
                "img_h": int,
                "img_w": int
            }, ...
        ]
    }, ...
]
# Supplementary precomputed embeddings:
vit_dict: dict[str, np.ndarray]  # {filename: (768,) float32}
```

#### 26. Required Output Requirements for Level 2.2
For downstream consumption by Level 3:
* `group_id`: Integer identifier of the pedagogical group.
* `selected_frames`: List of selected keyframe filenames (supports variable K >= 1).
* `frame_roles`: Semantic classification for each selected frame (e.g. `initial_setup`, `intermediate_derivation`, `final_culmination`).
* `selection_reason`: Categorical explanation (e.g. `maximum_information_coverage`, `pre_erasure_peak`, `syntax_error_outcome`).
* `coverage_ratio`: Fraction of the group's total unique information preserved by the selected subset.
* `confidence_score`: Selection quality confidence score.

---

### K. Critical Empirical Questions

#### 27. Validity of the "Culmination" Assumption
* **Finding:** The culmination assumption holds **strictly for static slide presentations** (`slides_AA`).
* **Counter-Evidence:** In blackboard math and live programming, content evolves non-monotonically:
  * Intermediate steps are erased to make room on the board.
  * Code is modified, broken, or scrolled out of view.
* **Observed Evolution Patterns:**
  1. Monotonic accumulation (bullet points, incremental code addition).
  2. Destructive replacement (blackboard erasure, variable overwriting).
  3. Transient UI popups (autocomplete dropdowns, context menus).
  4. Dynamic vertical scrolling (terminal command output).

#### 28. Empirical Distribution Across 52 Groups
Across `slides_AA` (33 groups), `CODING_SESSION` (11 groups), and `MATH-SESSION` (8 groups):

| Dynamic Pattern | Group Count | Datasets Observed |
| :--- | :--- | :--- |
| **Monotonic Accumulation** | 35 | 33 in `slides_AA`, 1 in `CODING_SESSION`, 1 in `MATH-SESSION` |
| **Destructive Replacement** | 9 | 5 in `MATH-SESSION`, 4 in `CODING_SESSION` |
| **Alternating / Debugging States** | 3 | 3 in `CODING_SESSION` |
| **Scrolling Content** | 3 | 3 in `CODING_SESSION` (terminal panes) |
| **Mostly Static** | 2 | 1 in `MATH-SESSION`, 1 in `CODING_SESSION` |
| **Visual-Only Change** | 0 | Diagram changes were accompanied by text changes |
| **OCR-Only Change** | 4 | Code typing on static IDE canvas |
| **Mixed Multimodal Change** | 48 | Present across all sessions |

#### 29. Correlation Between Persistence and Information Value
* **Empirical Finding:** Persistent information does **not** consistently correlate with pedagogical importance:
  * **Static chrome persists:** IDE menu bars (`learn_python_5`, `main.py`, status bar) appear in 100% of frames with zero instructional value.
  * **Critical information is transient:** Blackboard factor trees (`MATH-SESSION` `frame_0408`) and functional code before debugging (`CODING_SESSION` `frame_0176`) persist for only 15–30 seconds before being erased or edited.
  * **OCR artifacts persist:** Board smudges and static chalk marks can persist across dozens of frames without containing meaningful text.

#### 30. Information Beyond OCR Representation
1. **Mathematical Factor Trees & Graphs:** In `MATH-SESSION` `frame_0408.jpg`, factor branches link numbers visually. OCR captures only disjoint numbers (`-120`, `4`, `30`), losing mathematical structure.
2. **Code Indentation & Block Scope:** Python indentation is stripped during tokenization, losing nested control flow structure.
3. **Slide Diagrams & Visual Charts:** In `slides_AA` `frame_0704 -> 0733`, an architectural diagram contains boxes, arrows, and spatial relationships that OCR reduces to unordered text fragments.
4. **UI State & Highlights:** Syntax error squiggles, active terminal cursors, and breakpoint highlights are invisible to OCR.

---

### L. Critical Final Questions

#### 31. Problem Statement for Level 2.2 (Derived Strictly from Data)
Given a partitioned group of candidate frames containing progressive, destructive, or scrolling transitions, Level 2.2 must select the minimal set of representative keyframes (variable $K \ge 1$) that maximizes preserved semantic information, captures transient pre-erasure/pre-modification states, and rejects ephemeral UI popups and visual artifacts.

#### 32. Contradicted Assumptions from Prior Proposals
1. *Assumption: The final frame of a group is always the optimal keyframe.* Contradicted by blackboard erasures (`MATH-SESSION` Group 6) and terminal errors (`CODING_SESSION` Group 3).
2. *Assumption: Exactly 1 or 2 frames are sufficient for all groups.* Contradicted by complex blackboard derivations containing multiple distinct wiped stages.
3. *Assumption: ViT distance can distinguish code logic.* Contradicted by dark IDE background dominance, where ViT distance between different code blocks remains $< 0.04$.
4. *Assumption: OCR token set union captures complete information.* Contradicted by diagrams, factor trees, and syntax indentation.

#### 33. Pre-existing Features (No Need to Recompute)
* Single-frame OCR tokens, text strings, bounding boxes, and confidence scores (`features_cache.json`).
* 768-d unit-normalized ViT feature vectors (`vit_embeddings.npz`).
* Adjacent pairwise SSIM, OCR Jaccard, asymmetric token containment, and directional layout loss (`features_cache.json`, `l2_pairwise_features.csv`).
* Contiguous group boundaries and frame lists (`grouping_runs.json`).

#### 34. Currently Missing Signals
* Frame sharpness / blur metrics (e.g. Laplacian variance) to detect camera motion or hand sweep blur.
* Non-adjacent pairwise / cumulative similarity matrices (current cache only stores consecutive pairs $F_i \to F_{i+1}$).
* Code line indentation and text hierarchy.
* Presenter / occlusion segmentation (detecting when an instructor covers board content).

#### 35. Core Empirical Takeaways for Designing Level 2.2
1. **Level 2.1 Grouping is Sound:** A2/B2 grouping produces clean pedagogical units (8 math lessons, 11 coding lessons, 33 slides); Level 2.2 should operate strictly within these boundaries.
2. **Slides are Monotonic, Physical Media is Not:** Slide decks culminate at the final frame; blackboard and live-coding groups frequently destroy information mid-group.
3. **Final Frames Often Miss Crucial Steps:** Relying on the final frame loses pre-erasure derivations and clean working code.
4. **ViT is Modality-Specific:** ViT distinguishes slide graphics and natural scenes, but is ineffective for code editors.
5. **Transient Popups Require Filtering:** 1-frame autocomplete and context menus must be excluded from keyframe selection.
6. **Zero Re-Inference Needed:** RapidOCR and ViT are fully precomputed; Level 2.2 can execute via fast algebraic selection over existing JSON and NPZ artifacts.
7. **Adaptive $K$ is Required:** $K = 1$ is sufficient for monotonic slides, but $K \ge 2$ is necessary for groups with destructive modifications.