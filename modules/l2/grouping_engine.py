import numpy as np

class GroupingEngine:
    """
    Executes parallel A/B candidate grouping, partitions frames into discrete slide clusters,
    and analyzes divergence between Approach A and Approach B.
    """
    
    @staticmethod
    def run_approach_a(transitions, weights, threshold):
        """
        Approach A (Baseline): OCR Text + Layout IoU + SSIM.
        weights: dict with 'ssim', 'ocr', 'layout'
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

    @staticmethod
    def run_approach_b(transitions, weights, threshold, vit_available=True):
        """
        Approach B (Experimental): OCR Text + Layout IoU + SSIM + ViT Cosine Distance.
        weights: dict with 'ssim', 'ocr', 'layout', 'vit'
        """
        w_ssim = float(weights.get("ssim", 1.0))
        w_ocr = float(weights.get("ocr", 1.0))
        w_layout = float(weights.get("layout", 1.0))
        w_vit = float(weights.get("vit", 1.0)) if vit_available else 0.0
        
        total_w = w_ssim + w_ocr + w_layout + w_vit + 1e-9
        
        scores = []
        preds = []
        
        for t in transitions:
            d_s = t.get("d_ssim", 0.0)
            d_o = t.get("d_ocr_jaccard", 0.0)
            d_l = t.get("d_layout_iou", 0.0)
            d_v = t.get("d_vit_cosine")
            
            if d_v is None or not vit_available:
                # Fallback to 3-feature normalization if ViT missing for this pair
                cur_total = w_ssim + w_ocr + w_layout + 1e-9
                comp_score = (w_ssim * d_s + w_ocr * d_o + w_layout * d_l) / cur_total
            else:
                comp_score = (w_ssim * d_s + w_ocr * d_o + w_layout * d_l + w_vit * float(d_v)) / total_w
                
            comp_score = round(float(comp_score), 4)
            pred = 1 if comp_score >= threshold else 0
            
            scores.append(comp_score)
            preds.append(pred)
            
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
    def find_divergences(transitions, scores_a, preds_a, thresh_a, scores_b, preds_b, thresh_b):
        """
        Identifies frame transitions where Approach A and Approach B make opposite decisions.
        """
        divergences = []
        for i, t in enumerate(transitions):
            pa = preds_a[i]
            pb = preds_b[i]
            if pa != pb:
                sa = scores_a[i]
                sb = scores_b[i]
                if pa == 0 and pb == 1:
                    diff_type = "Approach B triggered New Group (Approach A kept Same Group)"
                else:
                    diff_type = "Approach A triggered New Group (Approach B kept Same Group)"
                    
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
