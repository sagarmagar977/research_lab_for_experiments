import numpy as np

class GroupingEngine:
    """
    Executes 3-way candidate grouping:
      - A1: Original Frozen Baseline (SSIM + Symmetric OCR Jaccard + Symmetric Layout IoU)
      - A2: Improved Asymmetric Preservation (SSIM + Asymmetric OCR Containment + Modular Directional Layout + Dynamic Renormalization)
      - B2: Multimodal Asymmetric (A2 + Pretrained ViT Cosine Distance)
    Partitions frames into discrete clusters and analyzes 3-way divergence.
    """
    
    @staticmethod
    def run_approach_a1(transitions, weights, threshold):
        """
        A1 — Original Frozen Baseline:
        Composite = (w_ssim * D_ssim + w_ocr * D_ocr_jaccard + w_layout * D_layout_iou) / sum(w)
        """
        w_ssim = float(weights.get("ssim", 1.0))
        w_ocr = float(weights.get("ocr", 1.0))
        w_layout = float(weights.get("layout", 1.0))
        total_w = w_ssim + w_ocr + w_layout + 1e-9
        
        scores = []
        preds = []
        
        for t in transitions:
            d_s = t.get("d_ssim", 0.0)
            d_o = t.get("d_ocr_jaccard", 0.0)
            d_l = t.get("d_layout_iou", 0.0)
            
            comp_score = (w_ssim * d_s + w_ocr * d_o + w_layout * d_l) / total_w
            comp_score = round(float(comp_score), 4)
            pred = 1 if comp_score >= threshold else 0
            
            scores.append(comp_score)
            preds.append(pred)
            
        return scores, preds

    # Alias for backwards compatibility
    run_approach_a = run_approach_a1

    @staticmethod
    def run_approach_a2(transitions, weights, threshold, min_tokens=3, use_layout=True):
        """
        A2 — Improved Asymmetric Preservation:
        Replaces symmetric OCR Jaccard with directional OCR Information Preservation / Containment.
        Uses dynamic weight renormalization when OCR or layout signals are invalid or disabled.
        """
        w_ssim = float(weights.get("ssim", 1.0))
        w_ocr = float(weights.get("ocr", 1.0))
        w_layout = float(weights.get("layout", 1.0)) if use_layout else 0.0
        
        scores = []
        preds = []
        meta_list = []
        
        for t in transitions:
            d_s = t.get("d_ssim", 0.0)
            
            # OCR information preservation check
            tokens_a = t.get("token_count_a", 0)
            ocr_loss = t.get("ocr_loss")
            ocr_valid_raw = t.get("ocr_valid_raw", True)
            ocr_valid = (tokens_a >= min_tokens) and (ocr_loss is not None) and (ocr_valid_raw is True)
            
            # Directional layout loss check
            layout_loss = t.get("layout_loss")
            layout_valid_raw = t.get("layout_valid", True)
            layout_valid = use_layout and (layout_loss is not None) and (layout_valid_raw is True)
            
            # Dynamic Renormalization
            num = w_ssim * d_s
            denom = w_ssim
            
            if ocr_valid:
                num += w_ocr * float(ocr_loss)
                denom += w_ocr
                
            if layout_valid:
                num += w_layout * float(layout_loss)
                denom += w_layout
                
            comp_score = round(float(num / max(denom, 1e-9)), 4)
            pred = 1 if comp_score >= threshold else 0
            
            scores.append(comp_score)
            preds.append(pred)
            meta_list.append({
                "ocr_valid": ocr_valid,
                "layout_valid": layout_valid
            })
            
        return scores, preds, meta_list

    @staticmethod
    def run_approach_b2(transitions, weights, threshold, min_tokens=3, use_layout=True, vit_available=True):
        """
        B2 — Multimodal Asymmetric Method:
        A2 (Asymmetric OCR + Modular Directional Layout + SSIM) + Pretrained ViT Cosine Distance.
        Uses dynamic weight renormalization across all valid multimodal signals.
        """
        w_ssim = float(weights.get("ssim", 1.0))
        w_ocr = float(weights.get("ocr", 1.0))
        w_layout = float(weights.get("layout", 1.0)) if use_layout else 0.0
        w_vit = float(weights.get("vit", 1.0)) if vit_available else 0.0
        
        scores = []
        preds = []
        meta_list = []
        
        for t in transitions:
            d_s = t.get("d_ssim", 0.0)
            
            # OCR information preservation check
            tokens_a = t.get("token_count_a", 0)
            ocr_loss = t.get("ocr_loss")
            ocr_valid_raw = t.get("ocr_valid_raw", True)
            ocr_valid = (tokens_a >= min_tokens) and (ocr_loss is not None) and (ocr_valid_raw is True)
            
            # Directional layout loss check
            layout_loss = t.get("layout_loss")
            layout_valid_raw = t.get("layout_valid", True)
            layout_valid = use_layout and (layout_loss is not None) and (layout_valid_raw is True)
            
            # ViT distance check
            d_v = t.get("d_vit_cosine")
            vit_valid = vit_available and (d_v is not None)
            
            # Dynamic Renormalization
            num = w_ssim * d_s
            denom = w_ssim
            
            if ocr_valid:
                num += w_ocr * float(ocr_loss)
                denom += w_ocr
                
            if layout_valid:
                num += w_layout * float(layout_loss)
                denom += w_layout
                
            if vit_valid:
                num += w_vit * float(d_v)
                denom += w_vit
                
            comp_score = round(float(num / max(denom, 1e-9)), 4)
            pred = 1 if comp_score >= threshold else 0
            
            scores.append(comp_score)
            preds.append(pred)
            meta_list.append({
                "ocr_valid": ocr_valid,
                "layout_valid": layout_valid,
                "vit_valid": vit_valid
            })
            
        return scores, preds, meta_list

    # Alias for legacy compatibility
    @staticmethod
    def run_approach_b(transitions, weights, threshold, vit_available=True):
        scores, preds, _ = GroupingEngine.run_approach_b2(
            transitions, weights, threshold, min_tokens=3, use_layout=True, vit_available=vit_available
        )
        return scores, preds

    @staticmethod
    def partition_groups(frames, preds):
        """
        Partitions frame sequence into discrete contiguous groups based on binary transitions.
        preds[i] corresponds to transition from frames[i] to frames[i+1].
        """
        if not frames:
            return []
            
        groups = []
        current_frames = [frames[0]]
        
        for i, pred in enumerate(preds):
            next_frame = frames[i + 1]
            if pred == 1:
                # Transition point: finalize current group and open new one
                groups.append(GroupingEngine._build_group_dict(len(groups) + 1, current_frames))
                current_frames = [next_frame]
            else:
                current_frames.append(next_frame)
                
        if current_frames:
            groups.append(GroupingEngine._build_group_dict(len(groups) + 1, current_frames))
            
        return groups

    @staticmethod
    def _build_group_dict(group_id, frame_list):
        first_f = frame_list[0]
        last_f = frame_list[-1]
        duration = last_f["timestamp_sec"] - first_f["timestamp_sec"]
        
        return {
            "group_id": group_id,
            "frame_count": len(frame_list),
            "start_frame": first_f["filename"],
            "end_frame": last_f["filename"],
            "start_timestamp": first_f["timestamp_str"],
            "end_timestamp": last_f["timestamp_str"],
            "duration_sec": max(0, duration),
            "frames": frame_list
        }

    @staticmethod
    def find_divergences_3way(
        transitions,
        scores_a1, preds_a1, thresh_a1,
        scores_a2, preds_a2, thresh_a2,
        scores_b2, preds_b2, thresh_b2
    ):
        """
        Tracks 3-way disagreements across A1 (frozen baseline), A2 (asymmetric), and B2 (multimodal ViT).
        """
        divergences = []
        for i, t in enumerate(transitions):
            p_a1 = preds_a1[i]
            p_a2 = preds_a2[i]
            p_b2 = preds_b2[i]
            
            # Any pair disagreement
            if not (p_a1 == p_a2 == p_b2):
                s_a1 = scores_a1[i]
                s_a2 = scores_a2[i]
                s_b2 = scores_b2[i]
                
                diff_types = []
                if p_a1 != p_a2:
                    if p_a1 == 1 and p_a2 == 0:
                        diff_types.append("A1 Split -> A2 Merged (Asymmetric Preservation absorbed build)")
                    else:
                        diff_types.append("A1 Merged -> A2 Split")
                if p_a2 != p_b2:
                    if p_a2 == 0 and p_b2 == 1:
                        diff_types.append("A2 Merged -> B2 Split (ViT visual deviation triggered boundary)")
                    else:
                        diff_types.append("A2 Split -> B2 Merged (ViT similarity suppressed boundary)")
                if p_a1 != p_b2 and p_a1 == p_a2:
                    diff_types.append("A1 & A2 agreed, B2 disagreed")
                    
                divergences.append({
                    "pair_idx": t.get("pair_idx", i + 1),
                    "frame_a": t["frame_a"],
                    "frame_b": t["frame_b"],
                    "timestamp_a": t["timestamp_a"],
                    "timestamp_b": t["timestamp_b"],
                    "delta_time_sec": t.get("delta_time_sec", 0),
                    "score_a1": s_a1,
                    "pred_a1": p_a1,
                    "thresh_a1": thresh_a1,
                    "score_a2": s_a2,
                    "pred_a2": p_a2,
                    "thresh_a2": thresh_a2,
                    "score_b2": s_b2,
                    "pred_b2": p_b2,
                    "thresh_b2": thresh_b2,
                    "d_ssim": t.get("d_ssim", 0.0),
                    "d_ocr_jaccard": t.get("d_ocr_jaccard", 0.0),
                    "ocr_preservation": t.get("ocr_preservation"),
                    "ocr_loss": t.get("ocr_loss"),
                    "ocr_valid": t.get("ocr_valid_raw", True),
                    "d_layout_iou": t.get("d_layout_iou", 0.0),
                    "layout_preservation": t.get("layout_preservation"),
                    "layout_loss": t.get("layout_loss"),
                    "layout_valid": t.get("layout_valid", True),
                    "d_vit": t.get("d_vit_cosine"),
                    "diff_types": diff_types,
                    "diff_summary": " | ".join(diff_types)
                })
        return divergences

    @staticmethod
    def find_divergences(transitions, scores_a, preds_a, thresh_a, scores_b, preds_b, thresh_b):
        """
        Legacy 2-way divergence helper.
        """
        divergences = []
        for i, t in enumerate(transitions):
            pa = preds_a[i]
            pb = preds_b[i]
            if pa != pb:
                sa = scores_a[i]
                sb = scores_b[i]
                diff_type = "B triggered New Group (A kept Same)" if (pa == 0 and pb == 1) else "A triggered New Group (B kept Same)"
                divergences.append({
                    "pair_idx": t.get("pair_idx", i + 1),
                    "frame_a": t["frame_a"],
                    "frame_b": t["frame_b"],
                    "timestamp_a": t["timestamp_a"],
                    "timestamp_b": t["timestamp_b"],
                    "delta_time_sec": t.get("delta_time_sec", 0),
                    "score_a": sa,
                    "pred_a": pa,
                    "thresh_a": thresh_a,
                    "score_b": sb,
                    "pred_b": pb,
                    "thresh_b": thresh_b,
                    "d_ssim": t.get("d_ssim", 0.0),
                    "d_ocr": t.get("d_ocr_jaccard", 0.0),
                    "d_layout": t.get("d_layout_iou", 0.0),
                    "d_vit": t.get("d_vit_cosine"),
                    "diff_type": diff_type
                })
        return divergences

    @staticmethod
    def compute_metrics(preds, ground_truth_dict):
        """
        Computes precision, recall, f1, and accuracy given researcher ground-truth labels.
        ground_truth_dict: {pair_idx: label (0 or 1)}
        """
        labeled_pairs = [idx for idx in ground_truth_dict if ground_truth_dict[idx] in (0, 1)]
        if not labeled_pairs:
            return None
            
        y_true = []
        y_pred = []
        
        for idx in labeled_pairs:
            # pair_idx is 1-indexed (idx 1 maps to transition index 0)
            t_idx = idx - 1
            if 0 <= t_idx < len(preds):
                y_true.append(ground_truth_dict[idx])
                y_pred.append(preds[t_idx])
                
        if not y_true:
            return None
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        
        return {
            "total_labeled": len(y_true),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
