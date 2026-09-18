import os
import re
import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from rapidocr_onnxruntime import RapidOCR

# Lazy singleton for RapidOCR engine
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR(det_limit_side_len=960, det_limit_type="max")
    return _ocr_engine

# --- ViT Environment Check & Model Loader ---
def check_vit_availability():
    """Checks if PyTorch and HuggingFace Transformers are installed for ViT."""
    try:
        import torch
        import transformers
        return True, "PyTorch & Transformers available"
    except ImportError as e:
        return False, f"Missing dependencies ({str(e)}). ViT will remain disabled."

_vit_model = None
_vit_processor = None

def get_vit_components():
    """Loads pretrained ViT model and processor on demand if dependencies exist."""
    global _vit_model, _vit_processor
    avail, _ = check_vit_availability()
    if not avail:
        return None, None
    if _vit_model is None:
        import torch
        from transformers import ViTImageProcessor, ViTModel
        model_name = "google/vit-base-patch16-224"
        _vit_processor = ViTImageProcessor.from_pretrained(model_name)
        _vit_model = ViTModel.from_pretrained(model_name)
        _vit_model.eval()
    return _vit_processor, _vit_model

# --- Timestamp Parser (Strict Rule: Total Seconds -> MM:SS) ---
def parse_timestamp_from_filename(filename, fallback_idx=0):
    """
    Parses total seconds from frame filename (e.g. frame_0029.jpg -> 29s -> '00:29',
    frame_0122.jpg -> 122s -> '02:02').
    Strictly avoids FPS derivation.
    """
    match = re.search(r'frame_(\d+)', filename, re.IGNORECASE)
    if match:
        total_sec = int(match.group(1))
    else:
        # Fallback to pure numeric search if available
        num_match = re.search(r'(\d+)', filename)
        if num_match:
            total_sec = int(num_match.group(1))
        else:
            total_sec = int(fallback_idx)
            
    minutes = total_sec // 60
    seconds = total_sec % 60
    timestamp_str = f"{minutes:02d}:{seconds:02d}"
    return total_sec, timestamp_str

# --- Tokenization Helper ---
def tokenize_text(text):
    """Extracts lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r'\b[a-zA-Z0-9_\-\$]+\b', text.lower())

# --- Single Frame Feature Extraction ---
def extract_single_frame_features(img_path, candidate_idx, ocr_engine=None):
    """
    Extracts text tokens, bounding boxes, confidence scores, and metadata for a single frame.
    """
    if ocr_engine is None:
        ocr_engine = get_ocr_engine()
        
    filename = os.path.basename(img_path)
    total_sec, timestamp_str = parse_timestamp_from_filename(filename, fallback_idx=candidate_idx)
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return {
            "filename": filename,
            "candidate_seq_idx": candidate_idx,
            "original_frame_id": total_sec,
            "timestamp_sec": total_sec,
            "timestamp_str": timestamp_str,
            "tokens": [],
            "full_text": "",
            "bboxes": [],
            "confidences": [],
            "num_tokens": 0,
            "num_boxes": 0,
            "img_h": 0,
            "img_w": 0
        }
        
    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    ocr_result, _ = ocr_engine(img_rgb)
    
    tokens = []
    text_lines = []
    bboxes = []
    confidences = []
    
    if ocr_result:
        for item in ocr_result:
            # item = [box, text, score]
            box = item[0]
            txt = str(item[1]).strip()
            score = float(item[2])
            
            if txt:
                text_lines.append(txt)
                tokens.extend(tokenize_text(txt))
                # Store box as normalized coordinates [[x/w, y/h], ...] for resolution-invariant layout matching
                norm_box = [[round(float(pt[0]) / max(img_w, 1), 4), round(float(pt[1]) / max(img_h, 1), 4)] for pt in box]
                bboxes.append(norm_box)
                confidences.append(round(score, 4))
                
    full_text = " ".join(text_lines)
    
    return {
        "filename": filename,
        "candidate_seq_idx": candidate_idx,
        "original_frame_id": total_sec,
        "timestamp_sec": total_sec,
        "timestamp_str": timestamp_str,
        "tokens": tokens,
        "full_text": full_text,
        "bboxes": bboxes,
        "confidences": confidences,
        "num_tokens": len(tokens),
        "num_boxes": len(bboxes),
        "img_h": img_h,
        "img_w": img_w
    }

# --- Optional ViT Single Frame Embedding ---
def extract_single_frame_vit_embedding(img_path):
    """
    Computes a 768-d unit-normalized ViT feature vector if available.
    Returns None if ViT is disabled or unavailable.
    """
    processor, model = get_vit_components()
    if processor is None or model is None:
        return None
        
    try:
        import torch
        from PIL import Image
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # Use pooler output or CLS token embedding
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output[0].cpu().numpy()
            else:
                emb = outputs.last_hidden_state[0, 0].cpu().numpy()
                
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm
        return emb.astype(np.float32)
    except Exception:
        return None

# --- Pairwise / Transition Feature Computation ---
def compute_layout_mask_iou(bboxes_a, bboxes_b, mask_size=256):
    """
    Renders bounding boxes onto a normalized binary canvas and computes spatial IoU.
    """
    if not bboxes_a and not bboxes_b:
        return 1.0  # Both empty = identical layout
    if not bboxes_a or not bboxes_b:
        return 0.0  # One has text, the other does not = complete mismatch
        
    mask_a = np.zeros((mask_size, mask_size), dtype=np.uint8)
    mask_b = np.zeros((mask_size, mask_size), dtype=np.uint8)
    
    for box in bboxes_a:
        pts = np.array([[int(pt[0] * mask_size), int(pt[1] * mask_size)] for pt in box], dtype=np.int32)
        cv2.fillPoly(mask_a, [pts], 1)
        
    for box in bboxes_b:
        pts = np.array([[int(pt[0] * mask_size), int(pt[1] * mask_size)] for pt in box], dtype=np.int32)
        cv2.fillPoly(mask_b, [pts], 1)
        
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    
    if union == 0:
        return 1.0
    return float(intersection / union)

def compute_pairwise_transition_features(frame_a_data, frame_b_data, img_path_a, img_path_b, vit_emb_a=None, vit_emb_b=None):
    """
    Computes all distance and differential transition features between consecutive frames Fi -> Fi+1.
    """
    # 1. Temporal distance
    delta_time = frame_b_data["timestamp_sec"] - frame_a_data["timestamp_sec"]
    
    # 2. Text Jaccard distance
    tokens_a = set(frame_a_data["tokens"])
    tokens_b = set(frame_b_data["tokens"])
    if not tokens_a and not tokens_b:
        d_ocr = 0.0  # Both have no text
    elif not tokens_a or not tokens_b:
        d_ocr = 1.0  # One has text, one doesn't
    else:
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = inter / union if union > 0 else 1.0
        d_ocr = round(1.0 - jaccard, 4)
        
    # 3. Layout IoU distance
    layout_iou = compute_layout_mask_iou(frame_a_data["bboxes"], frame_b_data["bboxes"])
    d_layout = round(1.0 - layout_iou, 4)
    
    # 4. SSIM distance
    d_ssim = 1.0
    try:
        img_a = cv2.imread(img_path_a, cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
        if img_a is not None and img_b is not None:
            # Resize img_b to match img_a dimensions for pairwise comparison if needed
            if img_a.shape != img_b.shape:
                img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)
            
            min_dim = min(img_a.shape[0], img_a.shape[1])
            win_size = min(11, min_dim)
            if win_size % 2 == 0:
                win_size = max(3, win_size - 1)
            win_size = max(3, win_size)
            
            ssim_val = structural_similarity(img_a, img_b, win_size=win_size)
            d_ssim = round(max(0.0, min(1.0, 1.0 - float(ssim_val))), 4)
    except Exception:
        d_ssim = 1.0
        
    # 5. ViT Cosine distance
    d_vit = None
    if vit_emb_a is not None and vit_emb_b is not None:
        try:
            cos_sim = float(np.dot(vit_emb_a, vit_emb_b) / (np.linalg.norm(vit_emb_a) * np.linalg.norm(vit_emb_b) + 1e-9))
            d_vit = round(max(0.0, min(1.0, (1.0 - cos_sim) / 2.0)), 4)
        except Exception:
            d_vit = None
            
    # Token count differentials
    n_a = frame_a_data["num_tokens"]
    n_b = frame_b_data["num_tokens"]
    
    return {
        "pair_idx": frame_b_data["candidate_seq_idx"],  # 1-indexed transition target
        "frame_a": frame_a_data["filename"],
        "frame_b": frame_b_data["filename"],
        "original_frame_id_a": frame_a_data["original_frame_id"],
        "original_frame_id_b": frame_b_data["original_frame_id"],
        "timestamp_a": frame_a_data["timestamp_str"],
        "timestamp_b": frame_b_data["timestamp_str"],
        "timestamp_sec_a": frame_a_data["timestamp_sec"],
        "timestamp_sec_b": frame_b_data["timestamp_sec"],
        "delta_time_sec": delta_time,
        "token_count_a": n_a,
        "token_count_b": n_b,
        "token_count_diff": abs(n_b - n_a),
        "d_ssim": d_ssim,
        "d_ocr_jaccard": d_ocr,
        "d_layout_iou": d_layout,
        "d_vit_cosine": d_vit
    }

# --- Cache Serialization ---
def get_session_level2_dir(session_dir):
    l2_dir = os.path.join(session_dir, "level2")
    os.makedirs(l2_dir, exist_ok=True)
    return l2_dir

def load_cached_features(session_dir):
    """Loads features_cache.json and vit_embeddings.npz if present."""
    l2_dir = os.path.join(session_dir, "level2")
    cache_path = os.path.join(l2_dir, "features_cache.json")
    vit_path = os.path.join(l2_dir, "vit_embeddings.npz")
    
    if not os.path.exists(cache_path):
        return None, None
        
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            
        vit_dict = {}
        if os.path.exists(vit_path):
            loaded = np.load(vit_path)
            for k in loaded.files:
                vit_dict[k] = loaded[k]
                
        return cache_data, vit_dict
    except Exception:
        return None, None

def save_features_to_cache(session_dir, frame_records, transition_records, vit_dict=None):
    """Saves extracted frame and transition features to disk."""
    l2_dir = get_session_level2_dir(session_dir)
    cache_path = os.path.join(l2_dir, "features_cache.json")
    
    payload = {
        "num_frames": len(frame_records),
        "num_transitions": len(transition_records),
        "frames": frame_records,
        "transitions": transition_records
    }
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    if vit_dict and len(vit_dict) > 0:
        vit_path = os.path.join(l2_dir, "vit_embeddings.npz")
        np.savez_compressed(vit_path, **vit_dict)
