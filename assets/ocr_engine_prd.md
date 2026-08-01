# PRD — Prototype 2: OCR Metadata Extraction Engine

## Objective

Build an OCR engine that extracts both textual content and geometric layout information from a candidate frame.

The goal of this prototype is **not** to reconstruct the document.

Instead, the objective is to produce a rich JSON representation that preserves every piece of information required for later layout reconstruction.

---

# Tech Stack

## Frontend

* Streamlit

## Backend

* FastAPI
* Python 3.12+

## OCR Engine

* PaddleOCR

## Libraries

* OpenCV
* NumPy

---

# User Workflow

1. Launch the application.
2. Upload a single candidate frame.
3. Click **Extract OCR**.
4. Run PaddleOCR.
5. Generate OCR metadata.
6. Save the metadata as JSON.
7. Display the OCR result.

Only single-image processing is required in this prototype.

---

# Input

Example

```text
candidate_frame.png
```

---

# OCR Engine

Use PaddleOCR for text detection and recognition.

For every detected text region extract:

* Detected text
* Confidence score
* Bounding box coordinates

No layout reconstruction should occur in this stage.

---

# Geometry Extraction

For every detected text block compute:

### Bounding Box

```text
x_min
y_min
x_max
y_max
```

---

### Width

```text
width = x_max - x_min
```

---

### Height

```text
height = y_max - y_min
```

---

### Center Point

```text
center_x = (x_min + x_max) / 2

center_y = (y_min + y_max) / 2
```

---

### Relative Font Height

```text
font_height = height
```

This is a relative estimate only.

It will later be used to distinguish headings from normal text.

---

# Line Assignment

The OCR engine should assign every detected text block to a line.

Suggested approach:

* Compute the vertical center (`center_y`) of every text block.
* Group text blocks whose vertical centers are within a configurable tolerance.
* Assign a unique `line_id` to each group.

Example

```text
CNN      uses      filters
```

↓

```text
line_id = 1
```

---

# Reading Order

After line assignment:

1. Sort lines from top to bottom.
2. Sort text blocks inside each line from left to right.

Store an `order_index` for every block.

---

# JSON Structure

Each detected text block should contain:

```json
{
    "id": 1,

    "text": "Convolution Layer",

    "confidence": 0.99,

    "bbox": {
        "x_min": 52,
        "y_min": 18,
        "x_max": 364,
        "y_max": 62
    },

    "width": 312,

    "height": 44,

    "center_x": 208,

    "center_y": 40,

    "font_height": 44,

    "line_id": 1,

    "order_index": 1
}
```

---

# Complete JSON Structure

```json
{
    "image": "frame_000027.png",

    "width": 1280,

    "height": 720,

    "ocr_engine": "PaddleOCR",

    "blocks": [
        ...
    ]
}
```

---

# Output

Save the JSON file.

Example

```text
frame_000027.json
```

---

# Streamlit Interface

The interface should contain only:

* Image uploader
* Image preview
* Extract OCR button
* OCR progress indicator
* JSON preview
* Download JSON button

No additional functionality is required.

---

# Out of Scope

The following should NOT be implemented:

* Heading detection
* Paragraph detection
* Table detection
* Bullet detection
* Code detection
* Equation detection
* Markdown generation
* Sequence merging
* Embedding generation
* Candidate frame detection

This prototype is responsible only for extracting OCR metadata.

---

# Success Criteria

The prototype is complete when it can:

* Accept a single candidate frame.
* Run PaddleOCR successfully.
* Extract text and confidence scores.
* Compute geometric metadata for every detected text block.
* Assign line IDs.
* Preserve reading order.
* Display the JSON in the Streamlit interface.
