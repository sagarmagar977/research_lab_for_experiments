import os
import json
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import numpy as np

def get_spatial_cells_16x16(bboxes: List) -> Set[Tuple[int, int]]:
    """
    Renders normalized OCR bounding boxes onto a 16x16 normalized grid.
    Handles 4-point polygon formats [[x,y]*4] by computing axis-aligned bounds.
    Returns set of integer cell coordinates (gx, gy) in [0..15] x [0..15].
    """
    cells = set()
    if not bboxes:
        return cells

    for box in bboxes:
        try:
            # Handle [[x, y], [x, y], [x, y], [x, y]] or flat [x1, y1, x2, y2]
            if len(box) >= 4 and isinstance(box[0], (list, tuple)):
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
            elif len(box) == 4:
                x_min, y_min, x_max, y_max = box
            else:
                continue

            # Clip to [0.0, 1.0]
            x_min = max(0.0, min(1.0, float(x_min)))
            x_max = max(0.0, min(1.0, float(x_max)))
            y_min = max(0.0, min(1.0, float(y_min)))
            y_max = max(0.0, min(1.0, float(y_max)))

            g_x0 = min(15, max(0, int(np.floor(x_min * 16))))
            g_x1 = min(15, max(0, int(np.floor(x_max * 16))))
            g_y0 = min(15, max(0, int(np.floor(y_min * 16))))
            g_y1 = min(15, max(0, int(np.floor(y_max * 16))))

            for gx in range(g_x0, g_x1 + 1):
                for gy in range(g_y0, g_y1 + 1):
                    cells.add((gx, gy))
        except Exception:
            continue

    return cells

def compute_vit_cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes normalized cosine distance in [0.0, 1.0] between two unit vectors.
    """
    try:
        cos_sim = float(np.dot(vec_a, vec_b))
        return float(np.clip((1.0 - cos_sim) / 2.0, 0.0, 1.0))
    except Exception:
        return 0.0

class KeyframeSelectorL2_2:
    """
    Level 2.2 — Minimal Informative Keyframe Selector.
    Executes controlled ablation formulations A, B, C, D, E1, E2 on Level 2.1 groups.
    """

    @staticmethod
    def select_group_keyframes(
        group_dict: Dict,
        mode: str = "D",  # "A", "B", "C", "D", "E1", "E2"
        tau_coverage: float = 0.95,
        epsilon_info: float = 0.02,
        w_spatial: float = 0.15,
        w_vit: float = 0.20,
        w_event: float = 0.20,
        tau_event_ocr: float = 0.30,
        tau_event_ssim: float = 0.025,
        event_direction: str = "pre",  # "pre" (default) or "post"
        vit_embeddings: Optional[Dict[str, np.ndarray]] = None,
        cached_transitions_map: Optional[Dict[Tuple[str, str], Dict]] = None,
        quality_map: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Runs greedy keyframe selection on a single Level 2.1 group.
        """
        group_frames = group_dict.get("frames", [])
        group_id = group_dict.get("group_id", 1)

        if not group_frames:
            return {
                "group_id": group_id,
                "mode": mode,
                "input_frame_count": 0,
                "selected_frames": [],
                "selected_count": 0,
                "final_coverage": 0.0,
                "final_lexical_coverage": 0.0,
                "final_spatial_coverage": 0.0,
                "compression_ratio": 0.0,
                "stopping_reason": "empty_group",
                "audit_trail": []
            }

        mode = mode.upper()
        use_multiset = (mode != "A")
        use_spatial = (mode in ("C", "D", "E1", "E2"))
        use_quality = (mode in ("D", "E1", "E2"))
        vit_embeddings = vit_embeddings or {}
        cached_transitions_map = cached_transitions_map or {}
        quality_map = quality_map or {}

        # -------------------------------------------------------------
        # 1. UNIVERSE DEFINITION
        # -------------------------------------------------------------
        if mode == "A":
            universe_lex = set()
            for f in group_frames:
                universe_lex.update(f.get("tokens", []))
            total_lex = len(universe_lex)
        else:
            max_counts = Counter()
            for f in group_frames:
                f_counts = Counter(f.get("tokens", []))
                for tok, cnt in f_counts.items():
                    if cnt > max_counts[tok]:
                        max_counts[tok] = cnt
            universe_lex = {(tok, k) for tok, max_c in max_counts.items() for k in range(1, max_c + 1)}
            total_lex = len(universe_lex)

        universe_sp: Set[Tuple[int, int]] = set()
        if use_spatial:
            for f in group_frames:
                universe_sp.update(get_spatial_cells_16x16(f.get("bboxes", [])))
        total_sp = len(universe_sp)

        eff_w_spatial = float(w_spatial) if (use_spatial and total_sp > 0) else 0.0

        # -------------------------------------------------------------
        # 2. STATE TRACKERS
        # -------------------------------------------------------------
        covered_lex_counts = Counter()
        covered_lex_set: Set[str] = set()
        covered_sp: Set[Tuple[int, int]] = set()

        selected_frames: List[Dict] = []
        audit_trail: List[Dict] = []
        stopping_reason = "unknown"

        group_frame_names = [f["filename"] for f in group_frames]
        n_frames = len(group_frames)

        # -------------------------------------------------------------
        # 3. GREEDY SELECTION LOOP
        # -------------------------------------------------------------
        while True:
            # Unweighted pedagogical coverage calculation
            if mode == "A":
                cov_lex = len(covered_lex_set) / max(total_lex, 1)
            else:
                cov_lex = sum(covered_lex_counts.values()) / max(total_lex, 1)
            cov_sp = len(covered_sp) / max(total_sp, 1) if total_sp > 0 else 1.0

            cur_coverage = (1.0 - eff_w_spatial) * cov_lex + eff_w_spatial * cov_sp

            # Stopping Condition 1: Global Information Target Reached
            if cur_coverage >= tau_coverage:
                stopping_reason = "coverage_target_reached"
                break

            best_candidate = None
            best_score = -1.0
            best_raw_gain = -1.0
            best_quality = -1.0
            best_audit = {}

            for idx, f in enumerate(group_frames):
                if f in selected_frames or f.get("is_blank", False):
                    continue

                fn = f["filename"]

                # A. Raw unweighted lexical gain
                if mode == "A":
                    new_toks = set(f.get("tokens", [])) - covered_lex_set
                    norm_lex_gain = len(new_toks) / max(total_lex, 1)
                else:
                    f_counts = Counter(f.get("tokens", []))
                    new_slots = sum(max(0, f_counts[t] - covered_lex_counts[t]) for t in f_counts)
                    norm_lex_gain = new_slots / max(total_lex, 1)

                # B. Raw unweighted spatial gain
                norm_sp_gain = 0.0
                if eff_w_spatial > 0.0:
                    f_cells = get_spatial_cells_16x16(f.get("bboxes", []))
                    new_cells = f_cells - covered_sp
                    norm_sp_gain = len(new_cells) / max(total_sp, 1)

                raw_info_gain = (1.0 - eff_w_spatial) * norm_lex_gain + eff_w_spatial * norm_sp_gain

                # Raw Information Eligibility Gate
                if raw_info_gain < epsilon_info:
                    continue

                # C. Quality Steering (Mode D/E1/E2)
                q_factor = quality_map.get(fn, f.get("quality_score", 1.0)) if use_quality else 1.0

                # D. Gated ViT Visual Diversity Modulation (Mode E1/E2)
                vit_multiplier = 0.0
                vit_dist = 0.0
                if mode in ("E1", "E2") and selected_frames:
                    v_curr = vit_embeddings.get(fn)
                    if v_curr is not None:
                        sel_vecs = [vit_embeddings.get(s["filename"]) for s in selected_frames if s["filename"] in vit_embeddings]
                        if sel_vecs:
                            vit_dist = min(compute_vit_cosine_distance(v_curr, sv) for sv in sel_vecs)
                            vit_multiplier = float(w_vit) * vit_dist

                # E. Gated Intra-Group Event Evidence Modulation (Mode E2)
                event_multiplier = 0.0
                event_score = 0.0
                if mode == "E2":
                    if event_direction == "pre" and idx < n_frames - 1:
                        nxt_fn = group_frame_names[idx + 1]
                        trans_data = cached_transitions_map.get((fn, nxt_fn), {})
                        ocr_loss = trans_data.get("ocr_loss", 0.0)
                        d_ssim = trans_data.get("d_ssim", 0.0)
                        if ocr_loss is not None and ocr_loss >= tau_event_ocr and d_ssim >= tau_event_ssim:
                            event_score = float(ocr_loss)
                            event_multiplier = float(w_event) * event_score
                    elif event_direction == "post" and idx > 0:
                        prev_fn = group_frame_names[idx - 1]
                        trans_data = cached_transitions_map.get((prev_fn, fn), {})
                        ocr_loss = trans_data.get("ocr_loss", 0.0)
                        d_ssim = trans_data.get("d_ssim", 0.0)
                        if ocr_loss is not None and ocr_loss >= tau_event_ocr and d_ssim >= tau_event_ssim:
                            event_score = float(ocr_loss)
                            event_multiplier = float(w_event) * event_score

                # Multiplicative candidate utility score
                candidate_score = raw_info_gain * q_factor * (1.0 + vit_multiplier + event_multiplier)

                # Deterministic Neutral Tie-Breaking:
                # 1. Total Score -> 2. Raw Info Gain -> 3. Quality -> 4. Earlier seq index
                is_better = False
                if candidate_score > best_score + 1e-7:
                    is_better = True
                elif abs(candidate_score - best_score) <= 1e-7:
                    if raw_info_gain > best_raw_gain + 1e-7:
                        is_better = True
                    elif abs(raw_info_gain - best_raw_gain) <= 1e-7:
                        if q_factor > best_quality + 1e-7:
                            is_better = True
                        elif abs(q_factor - best_quality) <= 1e-7:
                            if best_candidate is None or f["candidate_seq_idx"] < best_candidate["candidate_seq_idx"]:
                                is_better = True

                if is_better:
                    best_score = candidate_score
                    best_raw_gain = raw_info_gain
                    best_quality = q_factor
                    best_candidate = f
                    best_audit = {
                        "filename": fn,
                        "timestamp_str": f.get("timestamp_str", ""),
                        "timestamp_sec": f.get("timestamp_sec", 0),
                        "candidate_seq_idx": f.get("candidate_seq_idx", 0),
                        "raw_info_gain": round(raw_info_gain, 4),
                        "quality_score": round(q_factor, 4),
                        "vit_dist": round(vit_dist, 4),
                        "vit_multiplier": round(vit_multiplier, 4),
                        "event_score": round(event_score, 4),
                        "event_multiplier": round(event_multiplier, 4),
                        "selection_score": round(candidate_score, 4)
                    }

            # Stopping Condition 2: No candidate meets the raw information eligibility floor
            if best_candidate is None:
                stopping_reason = "raw_information_floor"
                break

            # Commit selection
            selected_frames.append(best_candidate)
            audit_trail.append(best_audit)

            if mode == "A":
                covered_lex_set.update(best_candidate.get("tokens", []))
            else:
                cand_counts = Counter(best_candidate.get("tokens", []))
                for t, cnt in cand_counts.items():
                    if cnt > covered_lex_counts[t]:
                        covered_lex_counts[t] = cnt

            if eff_w_spatial > 0.0:
                covered_sp.update(get_spatial_cells_16x16(best_candidate.get("bboxes", [])))

        # -------------------------------------------------------------
        # 4. NEUTRAL INFORMATION-BASED FALLBACK RULE
        # -------------------------------------------------------------
        if not selected_frames and group_frames:
            valid_frames = [f for f in group_frames if not f.get("is_blank", False)]
            fallback_pool = valid_frames if valid_frames else group_frames

            best_fb = None
            best_fb_info = -1.0
            best_fb_quality = -1.0

            for f in fallback_pool:
                fn = f["filename"]
                if mode == "A":
                    s_lex = len(set(f.get("tokens", []))) / max(total_lex, 1)
                else:
                    s_lex = sum(Counter(f.get("tokens", [])).values()) / max(total_lex, 1)
                s_sp = len(get_spatial_cells_16x16(f.get("bboxes", []))) / max(total_sp, 1) if total_sp > 0 else 1.0

                standalone_info = (1.0 - eff_w_spatial) * s_lex + eff_w_spatial * s_sp
                q_val = quality_map.get(fn, f.get("quality_score", 1.0)) if use_quality else 1.0

                is_better = False
                if standalone_info > best_fb_info + 1e-7:
                    is_better = True
                elif abs(standalone_info - best_fb_info) <= 1e-7:
                    if q_val > best_fb_quality + 1e-7:
                        is_better = True
                    elif abs(q_val - best_fb_quality) <= 1e-7:
                        if best_fb is None or f["candidate_seq_idx"] < best_fb["candidate_seq_idx"]:
                            is_better = True

                if is_better:
                    best_fb = f
                    best_fb_info = standalone_info
                    best_fb_quality = q_val

            selected_frames.append(best_fb)
            stopping_reason = "fallback_max_standalone_info"
            audit_trail.append({
                "filename": best_fb["filename"],
                "timestamp_str": best_fb.get("timestamp_str", ""),
                "timestamp_sec": best_fb.get("timestamp_sec", 0),
                "candidate_seq_idx": best_fb.get("candidate_seq_idx", 0),
                "raw_info_gain": round(best_fb_info, 4),
                "quality_score": round(best_fb_quality, 4),
                "selection_score": round(best_fb_info * best_fb_quality, 4),
                "note": "Fallback candidate maximizing standalone pedagogical content"
            })

            # Update covered state for accurate final reporting
            if mode == "A":
                covered_lex_set.update(best_fb.get("tokens", []))
            else:
                for t, cnt in Counter(best_fb.get("tokens", [])).items():
                    covered_lex_counts[t] = cnt
            if eff_w_spatial > 0.0:
                covered_sp.update(get_spatial_cells_16x16(best_fb.get("bboxes", [])))

        # -------------------------------------------------------------
        # 5. FINAL EXACT COVERAGE CALCULATION & RETURN
        # -------------------------------------------------------------
        if mode == "A":
            fin_cov_lex = len(covered_lex_set) / max(total_lex, 1)
        else:
            fin_cov_lex = sum(covered_lex_counts.values()) / max(total_lex, 1)
        fin_cov_sp = len(covered_sp) / max(total_sp, 1) if total_sp > 0 else 1.0
        final_total_coverage = (1.0 - eff_w_spatial) * fin_cov_lex + eff_w_spatial * fin_cov_sp

        # Chronological sort of selected frames
        selected_frames.sort(key=lambda x: x["timestamp_sec"])
        comp_ratio = round(len(selected_frames) / max(n_frames, 1), 4)

        return {
            "group_id": group_id,
            "mode": mode,
            "input_frame_count": n_frames,
            "selected_frames": selected_frames,
            "selected_filenames": [f["filename"] for f in selected_frames],
            "selected_timestamps": [f["timestamp_str"] for f in selected_frames],
            "selected_count": len(selected_frames),
            "final_coverage": round(final_total_coverage, 4),
            "final_lexical_coverage": round(fin_cov_lex, 4),
            "final_spatial_coverage": round(fin_cov_sp, 4),
            "compression_ratio": comp_ratio,
            "stopping_reason": stopping_reason,
            "audit_trail": audit_trail
        }

    @staticmethod
    def run_all_groups(
        groups: List[Dict],
        mode: str = "D",
        tau_coverage: float = 0.95,
        epsilon_info: float = 0.02,
        w_spatial: float = 0.15,
        w_vit: float = 0.20,
        w_event: float = 0.20,
        tau_event_ocr: float = 0.30,
        tau_event_ssim: float = 0.025,
        event_direction: str = "pre",
        vit_embeddings: Optional[Dict[str, np.ndarray]] = None,
        cached_transitions_map: Optional[Dict[Tuple[str, str], Dict]] = None,
        quality_map: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Runs keyframe selection across all groups in a partition.
        """
        results = []
        for g in groups:
            res = KeyframeSelectorL2_2.select_group_keyframes(
                group_dict=g,
                mode=mode,
                tau_coverage=tau_coverage,
                epsilon_info=epsilon_info,
                w_spatial=w_spatial,
                w_vit=w_vit,
                w_event=w_event,
                tau_event_ocr=tau_event_ocr,
                tau_event_ssim=tau_event_ssim,
                event_direction=event_direction,
                vit_embeddings=vit_embeddings,
                cached_transitions_map=cached_transitions_map,
                quality_map=quality_map
            )
            results.append(res)
        return results
