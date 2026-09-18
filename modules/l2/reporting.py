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

def save_ground_truth_dataset(
    session_dir,
    transitions,
    ground_truth_dict,
    notes_dict=None,
    scores_a1=None,
    preds_a1=None,
    scores_a2=None,
    preds_a2=None,
    scores_b2=None,
    preds_b2=None,
    preds_a=None,  # Legacy fallback
    preds_b=None   # Legacy fallback
):
    """
    Persists pairwise features, algorithm scores, and manual 0/1 transition labels to CSV for supervised ML training.
    """
    l2_dir = get_level2_dir(session_dir)
    csv_path = os.path.join(l2_dir, "ground_truth_transitions.csv")
    notes = notes_dict or {}
    
    # Fallback to legacy parameters if provided
    p_a1 = preds_a1 if preds_a1 is not None else preds_a
    p_b2 = preds_b2 if preds_b2 is not None else preds_b
    
    headers = [
        "pair_idx", "frame_a", "frame_b", "original_frame_id_a", "original_frame_id_b",
        "timestamp_a", "timestamp_b", "delta_time_sec",
        "token_count_a", "token_count_b", "token_count_diff",
        "d_ssim",
        "d_ocr_jaccard", "ocr_preservation", "ocr_loss", "ocr_valid",
        "d_layout_iou", "layout_preservation", "layout_loss", "layout_valid",
        "d_vit_cosine",
        "score_a1", "pred_a1",
        "score_a2", "pred_a2",
        "score_b2", "pred_b2",
        "researcher_note", "label"
    ]
    
    rows = []
    for i, t in enumerate(transitions):
        p_idx = t.get("pair_idx", i + 1)
        lbl = ground_truth_dict.get(p_idx, "")
        nt = notes.get(p_idx, "")
        
        sa1 = scores_a1[i] if scores_a1 and i < len(scores_a1) else ""
        pa1 = p_a1[i] if p_a1 and i < len(p_a1) else ""
        
        sa2 = scores_a2[i] if scores_a2 and i < len(scores_a2) else ""
        pa2 = preds_a2[i] if preds_a2 and i < len(preds_a2) else ""
        
        sb2 = scores_b2[i] if scores_b2 and i < len(scores_b2) else ""
        pb2 = p_b2[i] if p_b2 and i < len(p_b2) else ""
        
        d_vit = t.get("d_vit_cosine")
        d_vit_val = "" if d_vit is None else d_vit
        
        ocr_pres = t.get("ocr_preservation")
        ocr_loss = t.get("ocr_loss")
        layout_pres = t.get("layout_preservation")
        layout_loss = t.get("layout_loss")
        
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
            t.get("d_ssim", ""),
            t.get("d_ocr_jaccard", ""),
            "" if ocr_pres is None else ocr_pres,
            "" if ocr_loss is None else ocr_loss,
            t.get("ocr_valid_raw", True),
            t.get("d_layout_iou", ""),
            "" if layout_pres is None else layout_pres,
            "" if layout_loss is None else layout_loss,
            t.get("layout_valid", True),
            d_vit_val,
            sa1, pa1,
            sa2, pa2,
            sb2, pb2,
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
    Saves latest partition outputs, scores, and threshold metadata to grouping_runs.json.
    """
    l2_dir = get_level2_dir(session_dir)
    json_path = os.path.join(l2_dir, "grouping_runs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2)
    return json_path

def get_history_dir(session_dir):
    """Returns and creates the level2/history directory for saved experiment snapshots."""
    hist_dir = os.path.join(get_level2_dir(session_dir), "history")
    os.makedirs(hist_dir, exist_ok=True)
    return hist_dir

def save_run_snapshot(session_dir, run_payload, tag=""):
    """
    Saves an immutable timestamped snapshot of the current 3-way grouping run to history.
    """
    hist_dir = get_history_dir(session_dir)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_id = f"run_{ts_str}"
    
    snapshot_data = run_payload.copy()
    snapshot_data["snapshot_id"] = snapshot_id
    snapshot_data["saved_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_data["tag"] = tag.strip() if tag and tag.strip() else f"Snapshot {ts_str}"
    
    file_path = os.path.join(hist_dir, f"{snapshot_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2)
        
    return file_path, snapshot_data["tag"]

def list_run_snapshots(session_dir):
    """
    Returns a list of saved snapshot metadata sorted from newest to oldest.
    """
    hist_dir = get_history_dir(session_dir)
    if not os.path.exists(hist_dir):
        return []
        
    snapshots = []
    for f in os.listdir(hist_dir):
        if f.startswith("run_") and f.endswith(".json"):
            fp = os.path.join(hist_dir, f)
            try:
                with open(fp, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    snapshots.append({
                        "filename": f,
                        "filepath": fp,
                        "snapshot_id": data.get("snapshot_id", f.replace(".json", "")),
                        "saved_at": data.get("saved_at", "Unknown"),
                        "tag": data.get("tag", "Unnamed Run"),
                        "num_groups_a1": data.get("approach_a1", {}).get("num_groups", 0),
                        "num_groups_a2": data.get("approach_a2", {}).get("num_groups", 0),
                        "num_groups_b2": data.get("approach_b2", {}).get("num_groups", 0),
                        "divergences_count": data.get("divergences_count", 0),
                        "config_a1": data.get("approach_a1", {}).get("config", {}),
                        "config_a2": data.get("approach_a2", {}).get("config", {}),
                        "config_b2": data.get("approach_b2", {}).get("config", {})
                    })
            except Exception:
                pass
                
    snapshots.sort(key=lambda x: x["saved_at"], reverse=True)
    return snapshots

def load_run_snapshot(filepath):
    """Loads a full snapshot file from disk."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def delete_run_snapshot(filepath):
    """Deletes a saved snapshot file."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return True
        except Exception:
            return False
    return False

def generate_comparison_markdown_report(
    session_name,
    total_frames,
    config_a1,
    groups_a1,
    config_a2,
    groups_a2,
    config_b2,
    groups_b2,
    divergences_3way,
    metrics_a1,
    metrics_a2,
    metrics_b2,
    ground_truth_dict
):
    """
    Constructs an academic 3-way evaluation report comparing:
      - A1: Original Frozen Baseline (Symmetric Jaccard + Symmetric Layout IoU + SSIM)
      - A2: Asymmetric OCR Containment + Directional Layout + SSIM
      - B2: A2 + Pretrained ViT Cosine Distance
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate group duration statistics
    def calc_dur(groups):
        dur = [g["duration_sec"] for g in groups] if groups else [0]
        return sum(dur) / max(len(dur), 1)
        
    avg_dur_a1 = calc_dur(groups_a1)
    avg_dur_a2 = calc_dur(groups_a2)
    avg_dur_b2 = calc_dur(groups_b2)
    
    total_transitions = max(0, total_frames - 1)
    total_labeled = sum(1 for v in ground_truth_dict.values() if v in (0, 1))
    
    md = []
    md.append(f"# Level 2.1 Three-Way Grouping Comparison Report")
    md.append(f"**Session:** `{session_name}`  ")
    md.append(f"**Generated:** {now_str}  ")
    md.append(f"**Input Candidate Frames:** {total_frames} (Total Consecutive Transitions: {total_transitions})\n")
    md.append("---\n")
    
    # 1. Executive Summary Table
    md.append("## 1. Experimental Architecture & Summary\n")
    md.append("| Metric / Dimension | A1: Frozen Baseline | A2: Improved Asymmetric | B2: Multimodal (A2 + ViT) |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append("| **OCR Representation** | Symmetric Jaccard Distance | Asymmetric Information Preservation ($P_{ocr}$) | Asymmetric Information Preservation ($P_{ocr}$) |")
    md.append("| **Layout Representation** | Symmetric Layout IoU | Directional Layout Preservation (Modular) | Directional Layout Preservation (Modular) |")
    md.append("| **Visual Embedding** | SSIM only | SSIM only | SSIM + Pretrained ViT-B/16 Cosine Distance |")
    md.append(f"| **Decision Threshold ($\\tau$)** | `{config_a1.get('threshold', 0.35):.2f}` | `{config_a2.get('threshold', 0.35):.2f}` | `{config_b2.get('threshold', 0.35):.2f}` |")
    md.append(f"| **Total Groups Formed** | **{len(groups_a1)}** groups | **{len(groups_a2)}** groups | **{len(groups_b2)}** groups |")
    md.append(f"| **Average Group Duration** | {avg_dur_a1:.1f} sec | {avg_dur_a2:.1f} sec | {avg_dur_b2:.1f} sec |")
    md.append(f"| **New Group Triggers** | {sum(1 for g in groups_a1 if g['group_id'] > 1)} | {sum(1 for g in groups_a2 if g['group_id'] > 1)} | {sum(1 for g in groups_b2 if g['group_id'] > 1)} |\n")
    
    # 2. Ground Truth & Quantitative Validation
    md.append("## 2. Quantitative Ground-Truth Evaluation\n")
    if total_labeled == 0:
        md.append("> [!NOTE]")
        md.append("> No transitions have been manually labeled by the researcher yet.")
        md.append("> Annotate transitions in the lab UI to compute quantitative Accuracy, Precision, Recall, and F1 scores.\n")
    else:
        md.append(f"Evaluated against **{total_labeled}** researcher-verified ground-truth transitions:\n")
        md.append("| Performance Metric | A1 (Baseline) | A2 (Asymmetric) | B2 (A2 + ViT) |")
        md.append("| :--- | :--- | :--- | :--- |")
        
        def fmt(m, key, is_float=True):
            if not m or key not in m:
                return "N/A"
            return f"{m[key]:.4f}" if is_float else str(m[key])
            
        md.append(f"| **F1-Score** | **{fmt(metrics_a1, 'f1')}** | **{fmt(metrics_a2, 'f1')}** | **{fmt(metrics_b2, 'f1')}** |")
        md.append(f"| **Precision** | {fmt(metrics_a1, 'precision')} | {fmt(metrics_a2, 'precision')} | {fmt(metrics_b2, 'precision')} |")
        md.append(f"| **Recall** | {fmt(metrics_a1, 'recall')} | {fmt(metrics_a2, 'recall')} | {fmt(metrics_b2, 'recall')} |")
        md.append(f"| **Overall Accuracy** | {fmt(metrics_a1, 'accuracy')} | {fmt(metrics_a2, 'accuracy')} | {fmt(metrics_b2, 'accuracy')} |")
        
        cm_a1 = f"{metrics_a1['tp']}/{metrics_a1['fp']}/{metrics_a1['fn']}/{metrics_a1['tn']}" if metrics_a1 else "N/A"
        cm_a2 = f"{metrics_a2['tp']}/{metrics_a2['fp']}/{metrics_a2['fn']}/{metrics_a2['tn']}" if metrics_a2 else "N/A"
        cm_b2 = f"{metrics_b2['tp']}/{metrics_b2['fp']}/{metrics_b2['fn']}/{metrics_b2['tn']}" if metrics_b2 else "N/A"
        md.append(f"| **Confusion Matrix (TP/FP/FN/TN)** | `{cm_a1}` | `{cm_a2}` | `{cm_b2}` |\n")
        
    # 3. Controlled Ablation Disagreements
    md.append("## 3. Disagreement & Controlled Ablation Analysis\n")
    div_a1_a2 = [d for d in divergences_3way if d['pred_a1'] != d['pred_a2']]
    div_a2_b2 = [d for d in divergences_3way if d['pred_a2'] != d['pred_b2']]
    
    md.append(f"### Controlled Comparison 1: A1 → A2 (Effect of Asymmetric OCR Preservation)\n")
    if not div_a1_a2:
        md.append("No boundary differences observed between A1 and A2 under current settings.\n")
    else:
        md.append(f"Found **{len(div_a1_a2)}** transitions altered by replacing symmetric Jaccard with asymmetric containment:\n")
        md.append("| Transition (Fi -> Fi+1) | Timestamps | Score A1 | Score A2 | Asymm Preservation | Asymm Loss | Effect |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for d in div_a1_a2:
            pair = f"`{d['frame_a']}` -> `{d['frame_b']}`"
            t_span = f"`{d['timestamp_a']}` -> `{d['timestamp_b']}`"
            s1 = f"`{d['score_a1']:.3f}` (P={d['pred_a1']})"
            s2 = f"`{d['score_a2']:.3f}` (P={d['pred_a2']})"
            pres = f"`{d['ocr_preservation']:.2f}`" if d['ocr_preservation'] is not None else "N/A"
            loss = f"`{d['ocr_loss']:.2f}`" if d['ocr_loss'] is not None else "N/A"
            eff = "Absorbed Build (Split -> Merged)" if (d['pred_a1'] == 1 and d['pred_a2'] == 0) else "Triggered Split (Merged -> Split)"
            md.append(f"| {pair} | {t_span} | {s1} | {s2} | {pres} | {loss} | {eff} |")
        md.append("")
        
    md.append(f"### Controlled Comparison 2: A2 → B2 (Contribution of ViT Embeddings)\n")
    if not div_a2_b2:
        md.append("No boundary differences observed between A2 and B2 under current settings.\n")
    else:
        md.append(f"Found **{len(div_a2_b2)}** transitions altered by adding ViT cosine distance to A2:\n")
        md.append("| Transition (Fi -> Fi+1) | Timestamps | Score A2 | Score B2 | ViT Distance | Effect |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for d in div_a2_b2:
            pair = f"`{d['frame_a']}` -> `{d['frame_b']}`"
            t_span = f"`{d['timestamp_a']}` -> `{d['timestamp_b']}`"
            s2 = f"`{d['score_a2']:.3f}` (P={d['pred_a2']})"
            sb = f"`{d['score_b2']:.3f}` (P={d['pred_b2']})"
            vit_d = f"`{d['d_vit']:.3f}`" if d['d_vit'] is not None else "N/A"
            eff = "ViT Triggered Split (Merged -> Split)" if (d['pred_a2'] == 0 and d['pred_b2'] == 1) else "ViT Suppressed Split (Split -> Merged)"
            md.append(f"| {pair} | {t_span} | {s2} | {sb} | {vit_d} | {eff} |")
        md.append("")
        
    # 4. Group Partitions Breakdown
    md.append("## 4. Group Partition Breakdown\n")
    md.append("### A1: Original Baseline Groups\n")
    for g in groups_a1:
        f_names = ", ".join([f["filename"] for f in g["frames"][:5]])
        if len(g["frames"]) > 5:
            f_names += f", ... (+{len(g['frames']) - 5} more)"
        md.append(f"* **Group {g['group_id']}** (`{g['start_timestamp']}` -> `{g['end_timestamp']}`, {g['duration_sec']}s, {g['frame_count']} frames): {f_names}")
        
    md.append("\n### A2: Asymmetric Groups\n")
    for g in groups_a2:
        f_names = ", ".join([f["filename"] for f in g["frames"][:5]])
        if len(g["frames"]) > 5:
            f_names += f", ... (+{len(g['frames']) - 5} more)"
        md.append(f"* **Group {g['group_id']}** (`{g['start_timestamp']}` -> `{g['end_timestamp']}`, {g['duration_sec']}s, {g['frame_count']} frames): {f_names}")
        
    md.append("\n### B2: Multimodal (A2 + ViT) Groups\n")
    for g in groups_b2:
        f_names = ", ".join([f["filename"] for f in g["frames"][:5]])
        if len(g["frames"]) > 5:
            f_names += f", ... (+{len(g['frames']) - 5} more)"
        md.append(f"* **Group {g['group_id']}** (`{g['start_timestamp']}` -> `{g['end_timestamp']}`, {g['duration_sec']}s, {g['frame_count']} frames): {f_names}")
        
    md.append("\n---\n")
    md.append("## 5. Scientific Findings & Observations\n")
    if total_labeled > 0 and metrics_a1 and metrics_a2 and metrics_b2:
        md.append(f"* **A1 to A2 Shift:** Transitioning from symmetric Jaccard to asymmetric OCR containment altered {len(div_a1_a2)} transitions, resulting in an F1 shift of {metrics_a2['f1'] - metrics_a1['f1']:+.4f} (A1: {metrics_a1['f1']:.4f} vs A2: {metrics_a2['f1']:.4f}).")
        md.append(f"* **A2 to B2 Shift:** Incorporating ViT cosine distance onto the asymmetric formulation altered {len(div_a2_b2)} transitions, resulting in an F1 shift of {metrics_b2['f1'] - metrics_a2['f1']:+.4f} (A2: {metrics_a2['f1']:.4f} vs B2: {metrics_b2['f1']:.4f}).")
    else:
        md.append("* Evaluation requires researcher ground-truth annotations across test sessions to compute definitive precision, recall, and F1 comparisons.")
        
    report_text = "\n".join(md)
    l2_dir = get_level2_dir(os.path.join("sessions", session_name))
    report_path = os.path.join(l2_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return report_text, report_path
