import os
import time
import urllib.request
import cv2
import numpy as np
import streamlit as st
from rapidocr_onnxruntime import RapidOCR
import unicodedata

MODEL_DIR = "models"
V4_MODEL_PATH = os.path.join(MODEL_DIR, "ch_PP-OCRv4_det_infer.onnx")
V4_MODEL_URL = "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx"

EN_REC_V3_PATH = os.path.join(MODEL_DIR, "en_PP-OCRv3_rec_infer.onnx")
EN_REC_V3_URL = "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx"

EN_REC_V4_PATH = os.path.join(MODEL_DIR, "en_PP-OCRv4_rec_infer.onnx")
EN_REC_V4_URL = "https://huggingface.co/breezedeus/cnocr-ppocr-en_PP-OCRv4/resolve/main/en_PP-OCRv4_rec_infer.onnx"

@st.cache_resource
def load_detection_engines(use_english_ocr=False):
    """Initializes and caches the PP-OCRv3 and PP-OCRv4 engines."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    if not os.path.exists(V4_MODEL_PATH):
        with st.spinner("Downloading PP-OCRv4 detection model from Hugging Face... Please wait."):
            urllib.request.urlretrieve(V4_MODEL_URL, V4_MODEL_PATH)
            
    if use_english_ocr:
        if not os.path.exists(EN_REC_V3_PATH):
            with st.spinner("Downloading English PP-OCRv3 recognition model..."):
                urllib.request.urlretrieve(EN_REC_V3_URL, EN_REC_V3_PATH)
        if not os.path.exists(EN_REC_V4_PATH):
            with st.spinner("Downloading English PP-OCRv4 recognition model..."):
                urllib.request.urlretrieve(EN_REC_V4_URL, EN_REC_V4_PATH)
                
        engine_v3 = RapidOCR(
            det_model_path=None,
            rec_model_path=EN_REC_V3_PATH,
            det_limit_side_len=960,
            det_limit_type="max"
        )
        engine_v4 = RapidOCR(
            det_model_path=V4_MODEL_PATH,
            rec_model_path=EN_REC_V4_PATH,
            det_limit_side_len=960,
            det_limit_type="max"
        )
    else:
        engine_v3 = RapidOCR(
            det_model_path=None,
            det_limit_side_len=960,
            det_limit_type="max"
        )
        engine_v4 = RapidOCR(
            det_model_path=V4_MODEL_PATH,
            det_limit_side_len=960,
            det_limit_type="max"
        )
    return engine_v3, engine_v4

def run_detection_pipeline(
    img_orig, down_img, scale_x, scale_y, engine, thresh, unclip_ratio, padding_px, min_area_filter, empty_strategy, preprocess_mode, use_blur, blur_kernel_size, use_dilation, dilation_w, dilation_h, crop_mode
):
    """Runs text detection, filters coordinates, crops region, and computes diagnostics."""
    engine.text_detector.postprocess_op.thresh = float(thresh)
    engine.text_detector.postprocess_op.unclip_ratio = float(unclip_ratio)
    
    orig_h, orig_w = img_orig.shape[:2]
    frame_area = orig_w * orig_h
    
    if preprocess_mode == "Grayscale (Monochrome)":
        down_gray = cv2.cvtColor(down_img, cv2.COLOR_RGB2GRAY)
        if use_blur:
            down_gray = cv2.GaussianBlur(down_gray, (blur_kernel_size, blur_kernel_size), 0)
        down_img_pre = cv2.cvtColor(down_gray, cv2.COLOR_GRAY2RGB)
    elif preprocess_mode == "Adaptive Thresholding":
        down_gray = cv2.cvtColor(down_img, cv2.COLOR_RGB2GRAY)
        if use_blur:
            down_gray = cv2.GaussianBlur(down_gray, (blur_kernel_size, blur_kernel_size), 0)
        down_bin = cv2.adaptiveThreshold(
            down_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        down_img_pre = cv2.cvtColor(down_bin, cv2.COLOR_GRAY2RGB)
    else:
        if use_blur:
            down_img_pre = cv2.GaussianBlur(down_img, (blur_kernel_size, blur_kernel_size), 0)
        else:
            down_img_pre = down_img
        
    t0 = time.perf_counter()
    dt_boxes, _ = engine.text_detector(down_img_pre)
    latency_ms = (time.perf_counter() - t0) * 1000
    
    raw_mask = np.zeros(down_img.shape[:2], dtype=np.uint8)
    if dt_boxes is not None:
        for box in dt_boxes:
            cv2.fillPoly(raw_mask, [box.astype(np.int32)], 255)
            
    if use_dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_w, dilation_h))
        mask_dilated = cv2.dilate(raw_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        processed_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            processed_boxes.append(box)
    else:
        mask_dilated = raw_mask.copy()
        processed_boxes = dt_boxes if dt_boxes is not None else []
        
    kept_boxes_orig = []
    skipped_count = 0
    
    for box in processed_boxes:
        box_orig = (box * [scale_x, scale_y]).astype(np.int32)
        box_area = cv2.contourArea(box_orig.astype(np.float32))
        area_pct = (box_area / frame_area) * 100.0
        
        if area_pct >= min_area_filter:
            kept_boxes_orig.append(box_orig)
        else:
            skipped_count += 1
            
    if crop_mode == "Largest Region Only" and kept_boxes_orig:
        areas = [cv2.contourArea(b.astype(np.float32)) for b in kept_boxes_orig]
        largest_idx = np.argmax(areas)
        kept_boxes_orig = [kept_boxes_orig[largest_idx]]
                
    visualizer_img = img_orig.copy()
    
    if not kept_boxes_orig:
        status = "NO_TEXT_REGION_DETECTED"
        if empty_strategy == "Pass-Through Original":
            cropped_img = img_orig
            X1, Y1, X2, Y2 = 0, 0, orig_w, orig_h
        else:
            cropped_img = None
            X1, Y1, X2, Y2 = 0, 0, 0, 0
    else:
        status = "CONTENT_DETECTED"
        for box in kept_boxes_orig:
            cv2.polylines(visualizer_img, [box], isClosed=True, color=(16, 185, 129), thickness=3)
            
        all_x = [pt[0] for box in kept_boxes_orig for pt in box]
        all_y = [pt[1] for box in kept_boxes_orig for pt in box]
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        X1 = int(max(0, x_min - padding_px))
        Y1 = int(max(0, y_min - padding_px))
        X2 = int(min(orig_w, x_max + padding_px))
        Y2 = int(min(orig_h, y_max + padding_px))
        
        cropped_img = img_orig[Y1:Y2, X1:X2]
        cv2.rectangle(visualizer_img, (X1, Y1), (X2, Y2), (239, 68, 68), 4)
        
    if cropped_img is not None:
        cropped_h, cropped_w = cropped_img.shape[:2]
        area_retained = f"{((cropped_w * cropped_h) / frame_area) * 100.0:.1f}%"
    else:
        cropped_h, cropped_w = 0, 0
        area_retained = "0.0%"
        
    diagnostics = {
        "Detection Status": status,
        "Original Resolution": f"{orig_w}x{orig_h}",
        "Cropped Resolution": f"{cropped_w}x{cropped_h}" if status == "CONTENT_DETECTED" or empty_strategy == "Pass-Through Original" else "N/A",
        "Frame Area Retained": area_retained,
        "Inference Latency": f"{latency_ms:.1f}ms",
        "Bounding Coordinates": {"X1": X1, "Y1": Y1, "X2": X2, "Y2": Y2}
    }
    
    return {
        "status": status,
        "visualizer_img": visualizer_img,
        "mask_down": raw_mask,
        "mask_dilated": mask_dilated,
        "down_img_pre": down_img_pre,
        "cropped_img": cropped_img,
        "latency_ms": latency_ms,
        "diagnostics": diagnostics,
        "total_detected": len(dt_boxes) if dt_boxes is not None else 0,
        "kept_count": len(kept_boxes_orig),
        "skipped_count": skipped_count
    }

def classify_block_type(text):
    """Classifies a block as either a math 'formula' or regular 'text' using Unicode categories."""
    clean = text.replace(" ", "")
    if not clean:
        return "text"
        
    math_char_count = 0
    letter_count = 0
    digit_count = 0
    
    for char in clean:
        if unicodedata.category(char) == "Sm":
            math_char_count += 1
        elif "\u0370" <= char <= "\u03ff":
            math_char_count += 1
        elif char.isalpha():
            letter_count += 1
        elif char.isdigit():
            digit_count += 1
            
    if "=" in clean:
        return "formula"
    if (digit_count + math_char_count) > letter_count:
        return "formula"
    if math_char_count > 0 and letter_count < 5:
        return "formula"
    if len(clean) <= 6 and digit_count > 0 and letter_count <= 2:
        return "formula"
        
    return "text"

def extract_ocr_metadata(
    cropped_img, engine, tolerance_px, ocr_preprocess_mode, ocr_use_blur, ocr_blur_kernel, ocr_det_db_thresh, ocr_det_db_unclip_ratio
):
    """Runs PaddleOCR on the cropped image, groups lines with tolerance, and generates metadata JSON."""
    if cropped_img is None:
        return None
        
    if ocr_preprocess_mode == "Grayscale (Monochrome)":
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
        if ocr_use_blur:
            gray = cv2.GaussianBlur(gray, (ocr_blur_kernel, ocr_blur_kernel), 0)
        det_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif ocr_preprocess_mode == "Adaptive Thresholding":
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
        if ocr_use_blur:
            gray = cv2.GaussianBlur(gray, (ocr_blur_kernel, ocr_blur_kernel), 0)
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        det_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
    else:
        if ocr_use_blur:
            det_img = cv2.GaussianBlur(cropped_img, (ocr_blur_kernel, ocr_blur_kernel), 0)
        else:
            det_img = cropped_img.copy()
            
    engine.text_detector.postprocess_op.thresh = float(ocr_det_db_thresh)
    engine.text_detector.postprocess_op.unclip_ratio = float(ocr_det_db_unclip_ratio)
    
    t0 = time.perf_counter()
    dt_boxes, det_elapse = engine.text_detector(det_img)
    
    if dt_boxes is None or len(dt_boxes) < 1:
        return {
            "ocr_visualizer_img": cropped_img.copy(),
            "json_output": {
                "image": "cropped_frame.png",
                "width": cropped_img.shape[1],
                "height": cropped_img.shape[0],
                "ocr_engine": "PaddleOCR (ONNX)",
                "blocks": [],
                "reconstructed_text": []
            },
            "latency_ms": det_elapse * 1000 if isinstance(det_elapse, (int, float)) else 0.0
        }
        
    dt_boxes = engine.sorted_boxes(dt_boxes)
    img_crop_list = engine.get_crop_img_list(cropped_img, dt_boxes)
    rec_res, rec_elapse = engine.text_recognizer(img_crop_list)
    filter_boxes, filter_rec_res = engine.filter_boxes_rec_by_score(dt_boxes, rec_res)
    total_elapse_ms = (time.perf_counter() - t0) * 1000
    
    blocks = []
    for idx, (box, rec_item) in enumerate(zip(filter_boxes, filter_rec_res)):
        text, score = rec_item
        all_x = [pt[0] for pt in box]
        all_y = [pt[1] for pt in box]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        blocks.append({
            "text": text,
            "confidence": float(score),
            "bbox": {
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max)
            },
            "width": int(x_max - x_min),
            "height": int(y_max - y_min),
            "center_x": float((x_min + x_max) / 2),
            "center_y": float((y_min + y_max) / 2),
            "font_height": int(y_max - y_min),
            "box_polygon": box
        })
        
    ocr_visualizer_img = cropped_img.copy()
    for b in blocks:
        b_type = classify_block_type(b["text"])
        box_pts = np.array(b["box_polygon"], dtype=np.int32)
        color = (21, 204, 250) if b_type == "formula" else (129, 185, 16)
        cv2.polylines(ocr_visualizer_img, [box_pts], isClosed=True, color=color, thickness=2)
        
    blocks_sorted = sorted(blocks, key=lambda x: (x["center_y"], x["center_x"]))
    
    paragraphs = []
    for b in blocks_sorted:
        merged = False
        for p in paragraphs:
            p_x_min = min(item["bbox"]["x_min"] for item in p)
            p_x_max = max(item["bbox"]["x_max"] for item in p)
            p_w = p_x_max - p_x_min
            b_w = b["width"]
            
            overlap = min(b["bbox"]["x_max"], p_x_max) - max(b["bbox"]["x_min"], p_x_min)
            min_w = min(p_w, b_w)
            
            if overlap > 0 and (overlap / min_w) > 0.4:
                p_sorted_y = sorted(p, key=lambda x: x["center_y"])
                last_item = p_sorted_y[-1]
                v_gap = b["bbox"]["y_min"] - last_item["bbox"]["y_max"]
                max_fh = max(b["font_height"], last_item["font_height"])
                
                if v_gap <= max(10, 0.5 * max_fh) and abs(b["font_height"] - last_item["font_height"]) <= 6:
                    p.append(b)
                    merged = True
                    break
        if not merged:
            paragraphs.append([b])
            
    sorted_paras = sorted(paragraphs, key=lambda p: min(item["bbox"]["y_min"] for item in p))
    
    bands = []
    current_col_paras = []
    
    for p in sorted_paras:
        p_y_min = min(item["bbox"]["y_min"] for item in p)
        p_y_max = max(item["bbox"]["y_max"] for item in p)
        
        is_row = True
        for other in paragraphs:
            if other == p:
                continue
            o_y_min = min(item["bbox"]["y_min"] for item in other)
            o_y_max = max(item["bbox"]["y_max"] for item in other)
            
            overlap = min(p_y_max, o_y_max) - max(p_y_min, o_y_min)
            if overlap > 0:
                h_min = min(p_y_max - p_y_min, o_y_max - o_y_min)
                if (overlap / h_min) > 0.15:
                    is_row = False
                    break
                    
        if is_row:
            if current_col_paras:
                bands.append({"type": "columns", "paras": current_col_paras})
                current_col_paras = []
            bands.append({"type": "row", "para": p})
        else:
            current_col_paras.append(p)
            
    if current_col_paras:
        bands.append({"type": "columns", "paras": current_col_paras})
        
    final_blocks = []
    reconstructed_lines = []
    global_order_idx = 1
    global_line_id = 1
    global_para_id = 1
    
    for band in bands:
        if band["type"] == "row":
            para = band["para"]
            para_lines = []
            para_sorted_y = sorted(para, key=lambda x: x["center_y"])
            for b in para_sorted_y:
                assigned = False
                for line in para_lines:
                    avg_y = sum(item["center_y"] for item in line) / len(line)
                    if abs(b["center_y"] - avg_y) <= tolerance_px:
                        line.append(b)
                        assigned = True
                        break
                if not assigned:
                    para_lines.append([b])
                    
            para_lines = sorted(para_lines, key=lambda line: sum(item["center_y"] for item in line) / len(line))
            
            para_text_parts = []
            for line in para_lines:
                line_sorted_x = sorted(line, key=lambda x: x["center_x"])
                line_text = " ".join(item["text"] for item in line_sorted_x)
                para_text_parts.append(line_text)
                
                for b in line_sorted_x:
                    b_type = classify_block_type(b["text"])
                    out_block = {
                        "id": global_order_idx,
                        "type": b_type,
                        "text": b["text"],
                        "confidence": round(b["confidence"], 2),
                        "bbox": b["bbox"],
                        "width": b["width"],
                        "height": b["height"],
                        "center_x": int(round(b["center_x"])),
                        "center_y": int(round(b["center_y"])),
                        "font_height": b["font_height"],
                        "line_id": global_line_id,
                        "paragraph_id": global_para_id,
                        "column_id": None,
                        "order_index": global_order_idx
                    }
                    final_blocks.append(out_block)
                    global_order_idx += 1
                global_line_id += 1
                
            reconstructed_lines.append(" ".join(para_text_parts))
            global_para_id += 1
            
        elif band["type"] == "columns":
            paras = band["paras"]
            band_cols = []
            paras_sorted_x = sorted(paras, key=lambda p: min(item["bbox"]["x_min"] for item in p))
            
            for p in paras_sorted_x:
                p_x_min = min(item["bbox"]["x_min"] for item in p)
                p_x_max = max(item["bbox"]["x_max"] for item in p)
                p_x_center = (p_x_min + p_x_max) / 2
                
                matched_col = None
                for col in band_cols:
                    col_x_min = min(min(item["bbox"]["x_min"] for item in cp) for cp in col)
                    col_x_max = max(max(item["bbox"]["x_max"] for item in cp) for cp in col)
                    col_x_center = (col_x_min + col_x_max) / 2
                    
                    overlap = min(p_x_max, col_x_max) - max(p_x_min, col_x_min)
                    col_w = col_x_max - col_x_min
                    p_w = p_x_max - p_x_min
                    
                    if (overlap > 0 and (overlap / min(col_w, p_w)) > 0.4) or abs(p_x_center - col_x_center) < 120:
                        matched_col = col
                        break
                if matched_col is not None:
                    matched_col.append(p)
                else:
                    band_cols.append([p])
                    
            band_cols = sorted(band_cols, key=lambda col: sum(sum(item["center_x"] for item in cp) / len(cp) for cp in col) / len(col))
            
            for col_idx, col in enumerate(band_cols, start=1):
                col_paras_sorted_y = sorted(col, key=lambda p: min(item["bbox"]["y_min"] for item in p))
                
                for p in col_paras_sorted_y:
                    para_lines = []
                    para_sorted_y = sorted(p, key=lambda x: x["center_y"])
                    for b in para_sorted_y:
                        assigned = False
                        for line in para_lines:
                            avg_y = sum(item["center_y"] for item in line) / len(line)
                            if abs(b["center_y"] - avg_y) <= tolerance_px:
                                line.append(b)
                                assigned = True
                                break
                        if not assigned:
                            para_lines.append([b])
                            
                    para_lines = sorted(para_lines, key=lambda line: sum(item["center_y"] for item in line) / len(line))
                    
                    para_text_parts = []
                    for line in para_lines:
                        line_sorted_x = sorted(line, key=lambda x: x["center_x"])
                        line_text = " ".join(item["text"] for item in line_sorted_x)
                        para_text_parts.append(line_text)
                        
                        for b in line_sorted_x:
                            b_type = classify_block_type(b["text"])
                            out_block = {
                                "id": global_order_idx,
                                "type": b_type,
                                "text": b["text"],
                                "confidence": round(b["confidence"], 2),
                                "bbox": b["bbox"],
                                "width": b["width"],
                                "height": b["height"],
                                "center_x": int(round(b["center_x"])),
                                "center_y": int(round(b["center_y"])),
                                "font_height": b["font_height"],
                                "line_id": global_line_id,
                                "paragraph_id": global_para_id,
                                "column_id": col_idx,
                                "order_index": global_order_idx
                            }
                            final_blocks.append(out_block)
                            global_order_idx += 1
                        global_line_id += 1
                        
                    reconstructed_lines.append(" ".join(para_text_parts))
                    global_para_id += 1
            
    h_crop, w_crop = cropped_img.shape[:2]
    json_output = {
        "image": "cropped_frame.png",
        "width": w_crop,
        "height": h_crop,
        "ocr_engine": "PaddleOCR (ONNX)",
        "blocks": final_blocks,
        "reconstructed_text": reconstructed_lines
    }
    
    return {
        "ocr_visualizer_img": ocr_visualizer_img,
        "json_output": json_output,
        "latency_ms": total_elapse_ms
    }
