import os
import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List

def resolve_candidate_image_dir(session_dir: str) -> str:
    """
    Deterministically resolves candidate frames directory for a given session.
    Checks standard candidate subdirectories in order of priority.
    """
    candidates = ["v3_candidate_frames", "v4_candidate_frames", "selected_frames", "v3_crops", "v4_crops"]
    for sub in candidates:
        p = os.path.join(session_dir, sub)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                return p
    raise FileNotFoundError(f"No candidate image directory found in session: {session_dir}")

def compute_frame_laplacian_variance(img_path: str) -> float:
    """
    Computes Laplacian variance (sharpness metric) on a grayscale image.
    Returns 0.0 if image cannot be read.
    """
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return 0.0
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

def load_or_compute_quality_cache(
    session_dir: str,
    frame_filenames: List[str],
    q_min: float = 0.20,
    force_recompute: bool = False
) -> Dict:
    """
    Loads or lazily computes session-level Laplacian variance and normalized quality scores.
    Persists to sessions/<session>/level2/quality_cache.json.
    
    Q(F) = q_min + (1 - q_min) * clip((L(F) - P5) / (P95 - P5 + epsilon), 0, 1)
    """
    l2_dir = os.path.join(session_dir, "level2")
    os.makedirs(l2_dir, exist_ok=True)
    cache_path = os.path.join(l2_dir, "quality_cache.json")

    if not force_recompute and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verify cache covers requested frames
            cached_frames = data.get("frames", {})
            if all(fn in cached_frames for fn in frame_filenames):
                # Update q_min on the fly if requested q_min differs
                p5 = data.get("p5", 0.0)
                p95 = data.get("p95", 1.0)
                denom = max(p95 - p5, 1e-6)
                for fn, record in cached_frames.items():
                    lap = record.get("laplacian_variance", 0.0)
                    norm_val = float(np.clip((lap - p5) / denom, 0.0, 1.0))
                    record["quality_score"] = round(q_min + (1.0 - q_min) * norm_val, 4)
                data["q_min"] = q_min
                return data
        except Exception:
            pass  # Recompute on corruption

    cand_dir = resolve_candidate_image_dir(session_dir)
    raw_variances: Dict[str, float] = {}

    for fn in frame_filenames:
        img_path = os.path.join(cand_dir, fn)
        if os.path.exists(img_path):
            raw_variances[fn] = compute_frame_laplacian_variance(img_path)
        else:
            raw_variances[fn] = 0.0

    vals = list(raw_variances.values())
    if vals:
        p5 = float(np.percentile(vals, 5))
        p95 = float(np.percentile(vals, 95))
    else:
        p5, p95 = 0.0, 1.0

    denom = max(p95 - p5, 1e-6)
    frame_records: Dict[str, Dict] = {}

    for fn, lap in raw_variances.items():
        norm_val = float(np.clip((lap - p5) / denom, 0.0, 1.0))
        q_score = round(q_min + (1.0 - q_min) * norm_val, 4)
        frame_records[fn] = {
            "laplacian_variance": round(lap, 4),
            "quality_score": q_score
        }

    payload = {
        "session_dir": session_dir,
        "candidate_dir": cand_dir,
        "num_frames": len(frame_records),
        "q_min": q_min,
        "p5": round(p5, 4),
        "p95": round(p95, 4),
        "frames": frame_records
    }

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    return payload
