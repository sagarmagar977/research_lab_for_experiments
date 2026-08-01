Viewed app.py:1-800
Viewed app.py:800-1599

# Prototype Current State Summary

### Core Architecture & Stack
*   **Framework**: Streamlit web interface.
*   **Libraries**: OpenCV (`cv2`), Numpy (`np`), Pillow (`PIL.Image`), RapidOCR (ONNX runtime for PaddleOCR v3 and v4).
*   **Persistence**: Config values auto-saved to and loaded from [user_settings.json](file:///x:/prototype%20for%20thesis/candidate%20frame%20selction%20lab/user_settings.json).

---

### Key Modules & Tabs

#### 1. Single Frame Analyzer
*   **Image Preprocessing**: Color conversions (RGB, Grayscale, Adaptive Thresholding) and noise reduction via Gaussian Blur.
*   **Detection Pipeline**:
    *   ONNX detection model scores thresholding (`det_db_thresh`) and box unclip expansion (`det_db_unclip_ratio`).
    *   **Morphological Dilation**: Expands and merges adjacent text boxes (using adjustable Kernel Width/Height) to unify mathematical structures.
    *   **Area Filtering & Boundary Bounds**: Ignores noise segments below frame-area percentage (`min_area_filter`). Extracts either global bounding box union or largest single region.
*   **Layout-Ordered DLA / OCR Engine**: Runs on-demand text recognition on cropped regions. Classifies segments into standard `text` or `formula` using Unicode Sm block math categories.

#### 2. Batch Session Manager
*   **Inputs**: Supports batch frame uploads or absolute local directory path inputs.
*   **Processing**: Runs both PP-OCRv3 and PP-OCRv4 detection pipelines concurrently, outputs crops to local directories under `sessions/session_<timestamp>_<prefix>/` (`original_frames`, `v3_crops`, `v4_crops`).
*   **Session Browser**: Side-by-side comparison pane displaying original frame alongside output crops.

#### 3. Candidate Selector
*   **Gallery Grid**: Displays paginated crop frame cards with HTML styled container frames (solid green borders, tinted overlays, and drop shadows for selected candidates).
*   **Showcase Viewer**: Paginated view of current starred candidate frames (`v3_candidate_frames` and `v4_candidate_frames`).
*   **Lightbox Mode**: Early-return fullscreen image preview. Installs parent-frame keyboard listeners (`ArrowLeft`/`ArrowRight` for browsing, `S` to mark/unmark, `Escape` to close).

---

### Core Algorithms & Layout Logic

#### 1. Advanced 2D Document Layout Analyzer (`extract_ocr_metadata`)
*   **Paragraph Grouping**: Groups lines into logical text blocks:
    *   Merges blocks with horizontal overlap ($> 40\%$) and vertical gaps $\le \max(10, 0.5 \times \text{font\_height})$.
    *   Requires font heights to align within $\le 6$px difference. Prevents header text (e.g. `"Key Features"`) from merging with body text.
*   **Bands Separation**: Distinguishes row elements (no vertical overlaps, e.g. Titles) from column blocks (overlapping side-by-side neighbors).
*   **Horizontal Column Clustering**: Groups vertical blocks into separate columns based on X-coordinates (alignment tolerance $< 120$px).
*   **Natural Reading order**: Reconstructs reading text column-by-column, from left to right, rather than horizontal slicing. Adds layout metadata keys: `column_id`, `paragraph_id`, and `line_id` inside outputs.

#### 2. State & UX Synchronizations
*   **Callbacks (`on_click`)**:
    *   *Safe Pagination*: Button clicks call page callbacks (`change_g_page` / `change_c_page`) to update indices before selectboxes render, preventing `StreamlitAPIException` state-mutation locks.
    *   *Zero-Lag File Actions*: File copy (`Mark`) and deletion (`Selected` / `Remove`) actions run directly inside callback triggers. Eliminates delayed rendering and double-click lags.
*   **Lightbox Page Index Persistence**: Stores gallery pages inside non-widget keys (`backup_gallery_page_...` / `backup_cand_page_...`) before early returns in lightbox mode. Prevents Streamlit's widget state garbage collector from resetting active gallery views back to page 1 on exit.
*   **RGBA Discarding**: Converts uploaded alpha-channel images to RGB prior to numpy array conversion to prevent engine broadcast crashes.