import os
import csv
import json
import datetime
import pandas as pd

def get_level2_dir(session_dir):
    l2_dir = os.path.join(session_dir, "level2")
    os.makedirs(l2_dir, exist_ok=True)
    return l2_dir

def load_ground_truth_dataset(session_dir):
    """
    Loads saved ground truth annotations from ground_truth_transitions.csv.
    Returns (labels_dict, notes_dict).
    """
    l2_dir = get_level2_dir(session_dir)
    csv_path = os.path.join(l2_dir, "ground_truth_transitions.csv")
    labels = {}
    notes = {}
    
    if not os.path.exists(csv_path):
        return labels, notes
        
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            p_idx = int(row["pair_idx"])
            val = row.get("label")
            if pd.notna(val) and val != "":
                labels[p_idx] = int(val)
            note = row.get("researcher_note")
            if pd.notna(note) and str(note).strip():
                notes[p_idx] = str(note).strip()
    except Exception:
        pass
        
    return labels, notes

def save_ground_truth_dataset(session_dir, transitions, ground_truth_dict, notes_dict=None, preds_a=None, preds_b=None):
    """
    Persists pairwise features and manual 0/1 transition labels to CSV for future ML training.
    """
    l2_dir = get_level2_dir(session_dir)
    csv_path = os.path.join(l2_dir, "ground_truth_transitions.csv")
    notes = notes_dict or {}
    
    headers = [
        "pair_idx", "frame_a", "frame_b", "original_frame_id_a", "original_frame_id_b",
        "timestamp_a", "timestamp_b", "delta_time_sec",
        "token_count_a", "token_count_b", "token_count_diff",
        "d_ssim", "d_ocr_jaccard", "d_layout_iou", "d_vit_cosine",
        "pred_approach_a", "pred_approach_b", "researcher_note", "label"
    ]
    
    rows = []
    for i, t in enumerate(transitions):
        p_idx = t.get("pair_idx", i + 1)
        lbl = ground_truth_dict.get(p_idx, "")
        nt = notes.get(p_idx, "")
        
        pa = preds_a[i] if preds_a and i < len(preds_a) else ""
        pb = preds_b[i] if preds_b and i < len(preds_b) else ""
        
        d_vit = t.get("d_vit_cosine")
        d_vit_val = "" if d_vit is None else d_vit
        
        rows.append([
            p_idx,
            t["frame_a"],
            t["frame_b"],
            t["original_frame_id_a"],
            t["original_frame_id_b"],
            t["timestamp_a"],
            t["timestamp_b"],
            t.get("delta_time_sec", 0),
            t.get("token_count_a", 0),
            t.get("token_count_b", 0),
            t.get("token_count_diff", 0),
            t.get("d_ssim", 0.0),
            t.get("d_ocr_jaccard", 0.0),
            t.get("d_layout_iou", 0.0),
            d_vit_val,
            pa,
            pb,
            nt,
            lbl
        ])
        
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    return csv_path

def save_grouping_runs_json(session_dir, run_data):
    """
    Saves partition outputs, scores, and threshold metadata to grouping_runs.json.
    """
    l2_dir = get_level2_dir(session_dir)
    json_path = os.path.join(l2_dir, "grouping_runs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2)
    return json_path

def generate_comparison_markdown_report(
    session_name,
    total_frames,
    config_a,
    groups_a,
    config_b,
    groups_b,
    divergences,
    metrics_a,
    metrics_b,
    ground_truth_dict
):
    """
    Constructs a comprehensive academic Markdown evaluation report for Level 2.1 A/B comparison.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate group duration stats
    dur_a = [g["duration_sec"] for g in groups_a] if groups_a else [0]
    dur_b = [g["duration_sec"] for g in groups_b] if groups_b else [0]
    avg_dur_a = sum(dur_a) / max(len(dur_a), 1)
    avg_dur_b = sum(dur_b) / max(len(dur_b), 1)
    
    total_transitions = max(0, total_frames - 1)
    divergence_count = len(divergences)
    agreement_count = total_transitions - divergence_count
    agreement_rate = (agreement_count / total_transitions * 100.0) if total_transitions > 0 else 100.0
    
    md = []
    md.append(f"# Level 2.1 A/B Grouping Comparison Report")
    md.append(f"**Session:** `{session_name}`  ")
    md.append(f"**Generated:** {now_str}  ")
    md.append(f"**Input Candidate Frames:** {total_frames} (Total Transitions: {total_transitions})\n")
    md.append("---\n")
    
    # 1. Executive Summary Table
    md.append("## 1. Experimental Setup & High-Level Summary\n")
    md.append("| Metric / Dimension | Approach A (Baseline) | Approach B (Baseline + ViT) | Delta / Shift |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Active Features** | OCR + Layout + SSIM | OCR + Layout + SSIM + ViT | + Pretrained ViT-B/16 |")
    md.append(f"| **Threshold ($\\tau$)** | `{config_a.get('threshold', 0.40):.2f}` | `{config_b.get('threshold', 0.40):.2f}` | `{config_b.get('threshold', 0.40) - config_a.get('threshold', 0.40):+.2f}` |")
    md.append(f"| **Total Groups Formed** | **{len(groups_a)}** groups | **{len(groups_b)}** groups | `{len(groups_b) - len(groups_a):+d}` |")
    md.append(f"| **Average Group Duration** | {avg_dur_a:.1f} sec | {avg_dur_b:.1f} sec | `{avg_dur_b - avg_dur_a:+.1f}s` |")
    md.append(f"| **Total New Group Triggers** | {sum(1 for g in groups_a if g['group_id'] > 1)} | {sum(1 for g in groups_b if g['group_id'] > 1)} | `{sum(1 for g in groups_b if g['group_id'] > 1) - sum(1 for g in groups_a if g['group_id'] > 1):+d}` |")
    md.append(f"| **Transition Agreement Rate** | {agreement_rate:.1f}% ({agreement_count}/{total_transitions}) | {agreement_rate:.1f}% ({agreement_count}/{total_transitions}) | {divergence_count} Disagreements |\n")
    
    # 2. Ground Truth & Quantitative Validation
    md.append("## 2. Quantitative Ground-Truth Evaluation\n")
    total_labeled = sum(1 for v in ground_truth_dict.values() if v in (0, 1))
    
    if total_labeled == 0:
        md.append("> [!NOTE]")
        md.append("> No transitions have been manually labeled by the researcher yet.")
        md.append("> To establish definitive Precision, Recall, and F1 scores, use the Manual Ground-Truth Annotation tool in the lab UI.\n")
    else:
        md.append(f"Evaluated against **{total_labeled}** manually verified ground-truth transitions:\n")
        md.append("| Performance Metric | Approach A (Baseline) | Approach B (Baseline + ViT) | Performance Verdict |")
        md.append("| :--- | :--- | :--- | :--- |")
        
        acc_a = f"{metrics_a['accuracy']:.4f}" if metrics_a else "N/A"
        acc_b = f"{metrics_b['accuracy']:.4f}" if metrics_b else "N/A"
        p_a = f"{metrics_a['precision']:.4f}" if metrics_a else "N/A"
        p_b = f"{metrics_b['precision']:.4f}" if metrics_b else "N/A"
        r_a = f"{metrics_a['recall']:.4f}" if metrics_a else "N/A"
        r_b = f"{metrics_b['recall']:.4f}" if metrics_b else "N/A"
        f1_a = f"{metrics_a['f1']:.4f}" if metrics_a else "N/A"
        f1_b = f"{metrics_b['f1']:.4f}" if metrics_b else "N/A"
        
        verdict = "Equal"
        if metrics_a and metrics_b:
            if metrics_b['f1'] > metrics_a['f1']:
                verdict = f"ViT Outperformed (+{metrics_b['f1'] - metrics_a['f1']:.4f} F1)"
            elif metrics_b['f1'] < metrics_a['f1']:
                verdict = f"Baseline Outperformed (+{metrics_a['f1'] - metrics_b['f1']:.4f} F1)"
            else:
                verdict = "Tied F1 Score"
                
        md.append(f"| **F1-Score** | **{f1_a}** | **{f1_b}** | **{verdict}** |")
        md.append(f"| **Precision** | {p_a} | {p_b} | - |")
        md.append(f"| **Recall** | {r_a} | {r_b} | - |")
        md.append(f"| **Overall Accuracy** | {acc_a} | {acc_b} | - |")
        if metrics_a and metrics_b:
            md.append(f"| **Confusion Matrix (TP/FP/FN/TN)** | `{metrics_a['tp']}/{metrics_a['fp']}/{metrics_a['fn']}/{metrics_a['tn']}` | `{metrics_b['tp']}/{metrics_b['fp']}/{metrics_b['fn']}/{metrics_b['tn']}` | - |\n")
        else:
            md.append("")
            
    # 3. Divergence Analysis (Disagreements)
    md.append("## 3. Disagreement & Divergence Analysis\n")
    if not divergences:
        md.append("Both Approach A and Approach B produced 100% identical group partitions under the current thresholds.\n")
    else:
        md.append(f"Found **{len(divergences)}** transitions where Approach A and Approach B produced opposing decisions:\n")
        md.append("| Transition (Fi -> Fi+1) | Timestamps | Score A | Score B | ViT Distance | Disagreement Mechanism |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for div in divergences:
            pair_str = f"`{div['frame_a']}` -> `{div['frame_b']}`"
            time_str = f"`{div['timestamp_a']}` -> `{div['timestamp_b']}`"
            sa_str = f"`{div['score_a']:.3f}` ($\\ge {div['thresh_a']:.2f}$)" if div['pred_a'] == 1 else f"`{div['score_a']:.3f}` (< {div['thresh_a']:.2f})"
            sb_str = f"`{div['score_b']:.3f}` ($\\ge {div['thresh_b']:.2f}$)" if div['pred_b'] == 1 else f"`{div['score_b']:.3f}` (< {div['thresh_b']:.2f})"
            vit_str = f"`{div['d_vit']:.3f}`" if div['d_vit'] is not None else "N/A"
            md.append(f"| {pair_str} | {time_str} | {sa_str} | {sb_str} | {vit_str} | {div['diff_type']} |")
        md.append("")
        
    # 4. Group Partitions Breakdown
    md.append("## 4. Group Partition Inspection\n")
    md.append("### Approach A (Baseline Groups)\n")
    for g in groups_a:
        f_names = ", ".join([f["filename"] for f in g["frames"][:6]])
        if len(g["frames"]) > 6:
            f_names += f", ... (+{len(g['frames']) - 6} more)"
        md.append(f"* **Group {g['group_id']}** (`{g['start_timestamp']}` -> `{g['end_timestamp']}`, {g['duration_sec']}s, {g['frame_count']} frames): {f_names}")
        
    md.append("\n### Approach B (Baseline + ViT Groups)\n")
    for g in groups_b:
        f_names = ", ".join([f["filename"] for f in g["frames"][:6]])
        if len(g["frames"]) > 6:
            f_names += f", ... (+{len(g['frames']) - 6} more)"
        md.append(f"* **Group {g['group_id']}** (`{g['start_timestamp']}` -> `{g['end_timestamp']}`, {g['duration_sec']}s, {g['frame_count']} frames): {f_names}")
        
    md.append("\n---\n")
    md.append("## 5. Thesis Conclusion on Level 2.1 ViT Utility\n")
    if total_labeled > 0 and metrics_a and metrics_b:
        if metrics_b['f1'] > metrics_a['f1']:
            md.append("> **Experimental Finding:** Pretrained ViT features improved transition boundary detection (+%.4f F1). ViT embeddings successfully captured semantic visual scene changes not fully reflected by OCR text and bounding-box overlap alone." % (metrics_b['f1'] - metrics_a['f1']))
        elif metrics_b['f1'] < metrics_a['f1']:
            md.append("> **Experimental Finding:** The addition of ViT visual embeddings did not improve grouping quality (-%.4f F1). ViT introduced false positive transitions due to subtle non-informational visual variations (e.g. cursor movement, font anti-aliasing) where underlying slide text remained identical." % (metrics_a['f1'] - metrics_b['f1']))
        else:
            md.append("> **Experimental Finding:** ViT visual embeddings produced an identical F1 score to the baseline OCR+Layout+SSIM approach.")
    else:
        md.append("> **Note:** Complete manual ground-truth annotation across thesis test sessions is required to finalize the academic conclusion.")
        
    report_text = "\n".join(md)
    l2_dir = get_level2_dir(os.path.join("sessions", session_name))
    report_path = os.path.join(l2_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return report_text, report_path
