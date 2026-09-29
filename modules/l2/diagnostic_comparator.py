import os
import json
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd

class KeyframeDiagnosticComparator:
    """
    Compares Human-Curated Ground Truth Keyframes against Level 2.2 Algorithm Ablations (A through E2).
    Quantifies Precision, Recall, F1 (broken down by role 'final' vs 'state') and
    diagnoses why algorithms missed human choices (False Negatives) or picked spurious frames (False Positives).
    """

    @staticmethod
    def evaluate_mode(
        gt_dict: Dict[str, Dict],  # {filename: {"role_type": "final"|"state", "group_id": int, ...}}
        algo_selected_map: Dict[int, List[str]],  # {group_id: [selected_filenames]}
        algo_audit_map: Optional[Dict[int, List[Dict]]] = None,  # {group_id: audit_trail}
        frames_features: Optional[Dict[str, Dict]] = None,
        quality_map: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Evaluates a single algorithmic mode against human ground truth.
        """
        # Collect all human keyframe filenames
        human_keyframes = set(gt_dict.keys())
        human_finals = {fn for fn, rec in gt_dict.items() if rec.get("role_type") == "final"}
        human_states = {fn for fn, rec in gt_dict.items() if rec.get("role_type") == "state"}

        # Collect all algorithm keyframes
        algo_keyframes = set()
        for gid, fnames in algo_selected_map.items():
            for fn in fnames:
                algo_keyframes.add(fn)

        tp_set = human_keyframes.intersection(algo_keyframes)
        fp_set = algo_keyframes.difference(human_keyframes)
        fn_set = human_keyframes.difference(algo_keyframes)

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Role-specific recall
        tp_finals = len(human_finals.intersection(algo_keyframes))
        tp_states = len(human_states.intersection(algo_keyframes))

        final_recall = tp_finals / len(human_finals) if human_finals else 0.0
        state_recall = tp_states / len(human_states) if human_states else 0.0

        # Build detailed False Negative diagnosis
        fn_diagnostics = []
        for fn_name in sorted(list(fn_set)):
            gt_rec = gt_dict.get(fn_name, {})
            role = gt_rec.get("role_type", "unknown")
            gid = gt_rec.get("group_id", 0)
            
            f_feat = frames_features.get(fn_name, {}) if frames_features else {}
            q_score = quality_map.get(fn_name, 0.0) if quality_map else 0.0
            
            # Check audit trail if available
            audit_records = algo_audit_map.get(gid, []) if algo_audit_map else []
            step_rec = next((r for r in audit_records if r.get("chosen_frame") == fn_name), None)
            
            diag_reason = "Excluded by algorithm"
            if audit_records:
                last_step = audit_records[-1] if audit_records else {}
                stopping_reason = last_step.get("stopping_reason", "unknown")
                if stopping_reason == "coverage_reached":
                    diag_reason = "Pruned: Algorithm terminated early upon hitting target pedagogical coverage (tau_cov)"
                elif stopping_reason == "eligibility_floor":
                    diag_reason = "Pruned: Frame unweighted marginal information gain fell below floor (epsilon_info)"
                elif stopping_reason == "budget_exhausted":
                    diag_reason = "Pruned: Maximum frame budget exhausted"

            fn_diagnostics.append({
                "Group ID": gid,
                "Filename": fn_name,
                "Human Role": role.upper(),
                "Timestamp": gt_rec.get("timestamp_str", ""),
                "Tokens Count": len(f_feat.get("tokens", [])),
                "Quality Score": round(float(q_score), 3),
                "Algorithmic Discard Diagnosis": diag_reason
            })

        # Build detailed False Positive diagnosis
        fp_diagnostics = []
        for fp_name in sorted(list(fp_set)):
            # Find group
            gid = None
            for g, fnames in algo_selected_map.items():
                if fp_name in fnames:
                    gid = g
                    break
            
            f_feat = frames_features.get(fp_name, {}) if frames_features else {}
            q_score = quality_map.get(fp_name, 0.0) if quality_map else 0.0

            fp_diagnostics.append({
                "Group ID": gid if gid is not None else "-",
                "Filename": fp_name,
                "Algorithm Selection": "Spurious / Over-selected",
                "Timestamp": f_feat.get("timestamp_str", ""),
                "Tokens Count": len(f_feat.get("tokens", [])),
                "Quality Score": round(float(q_score), 3),
                "Algorithmic Selection Cause": "Selected due to residual information gain or quality bonus unneeded by human"
            })

        return {
            "metrics": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "final_recall": round(float(final_recall), 4),
                "state_recall": round(float(state_recall), 4),
                "human_total": len(human_keyframes),
                "human_finals": len(human_finals),
                "human_states": len(human_states),
                "algo_total": len(algo_keyframes)
            },
            "fn_diagnostics": fn_diagnostics,
            "fp_diagnostics": fp_diagnostics,
            "tp_frames": sorted(list(tp_set)),
            "fp_frames": sorted(list(fp_set)),
            "fn_frames": sorted(list(fn_set))
        }

    @staticmethod
    def compare_all_modes(
        gt_dict: Dict[str, Dict],
        algo_runs_by_mode: Dict[str, Dict[int, Dict]], # {mode: {gid: result_dict}}
        frames_features: Optional[Dict[str, Dict]] = None,
        quality_map: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Creates a high-level comparison table across all ablation modes (A through E2).
        """
        rows = []
        mode_descriptions = {
            "A": "A: Unique Lexical Set",
            "B": "B: Lexical Multiset",
            "C": "C: Multiset + Spatial (16x16)",
            "D": "D: Multiset + Spatial + Quality",
            "E1": "E1: D + Gated ViT Diversity",
            "E2": "E2: D + Gated ViT + Event Evidence"
        }

        for mode, g_dict in algo_runs_by_mode.items():
            algo_selected_map = {gid: res["selected_filenames"] for gid, res in g_dict.items()}
            algo_audit_map = {gid: res.get("audit_trail", []) for gid, res in g_dict.items()}

            eval_res = KeyframeDiagnosticComparator.evaluate_mode(
                gt_dict=gt_dict,
                algo_selected_map=algo_selected_map,
                algo_audit_map=algo_audit_map,
                frames_features=frames_features,
                quality_map=quality_map
            )
            m = eval_res["metrics"]
            rows.append({
                "Ablation Mode": mode_descriptions.get(mode, mode),
                "Precision": f"{m['precision']*100:.1f}%",
                "Recall": f"{m['recall']*100:.1f}%",
                "F1-Score": f"{m['f1']*100:.1f}%",
                "Final Recall": f"{m['final_recall']*100:.1f}% ({m['human_finals']} finals)",
                "State Recall": f"{m['state_recall']*100:.1f}% ({m['human_states']} states)",
                "True Positives": m["tp"],
                "False Positives": m["fp"],
                "False Negatives": m["fn"],
                "Total Selected": m["algo_total"]
            })

        return pd.DataFrame(rows)
