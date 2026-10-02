import os, sys, json
sys.path.insert(0, os.getcwd())
from collections import defaultdict
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.keyframe_selector import KeyframeSelectorL2_2, get_spatial_cells_16x16
from modules.l2.feature_extractor import load_cached_features
from modules.l2.quality_cache import load_or_compute_quality_cache
from modules.l2.diagnostic_comparator import KeyframeDiagnosticComparator

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']
modes = ['A', 'B', 'C', 'D', 'E1', 'E2']

metrics = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))

fp_reasons = defaultdict(int)
fn_reasons = defaultdict(int)

# To challenge the categories
fp_raw_gains = []
fn_marginal_gains = []

for s in sessions:
    s_path = os.path.join('sessions', s)
    runs_file = os.path.join(s_path, 'level2', 'grouping_runs.json')
    if not os.path.exists(runs_file): continue
    
    run_data = json.load(open(runs_file, 'r', encoding='utf-8'))
    part_key = 'approach_a2' if 'approach_a2' in run_data else list(run_data.keys())[2]
    groups = run_data[part_key]['groups']
    gt_dict = load_keyframe_ground_truth(s_path)
    if not gt_dict: continue

    cached_data, cached_vit = load_cached_features(s_path)
    frames_feat = {fr['filename']: fr for fr in cached_data['frames']}
    trans_map = {(t['frame_a'], t['frame_b']): t for t in cached_data.get('transitions', [])}
    q_cache = load_or_compute_quality_cache(s_path, list(frames_feat.keys()))
    q_map = {fn: rec['quality_score'] for fn, rec in q_cache.get('frames', {}).items()}
    vit_map = cached_vit or {}

    for m in modes:
        all_selected = []
        for g in groups:
            res = KeyframeSelectorL2_2.select_group_keyframes(
                group_dict=g, mode=m, tau_coverage=0.95, epsilon_info=0.02,
                w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings=vit_map,
                cached_transitions_map=trans_map, quality_map=q_map
            )
            selected = res['selected_filenames']
            all_selected.extend(selected)
            
            # Analyze FPs and FNs for Mode E1 (the baseline we are auditing heavily)
            if m == 'E1':
                gt_in_group = [fr['filename'] for fr in g.get('frames', []) if fr['filename'] in gt_dict]
                # FPs
                for fn in selected:
                    if fn not in gt_dict:
                        audit_item = next((x for x in res.get('audit_trail', []) if x['filename'] == fn), None)
                        if audit_item:
                            fp_reasons['Selected via Raw Gain >= 0.02'] += 1
                            fp_raw_gains.append(audit_item.get('raw_info_gain', 0))
                        else:
                            fp_reasons['Unknown'] += 1
                
                # FNs
                for fn in gt_in_group:
                    if fn not in selected:
                        stop_reason = res.get('stopping_reason', 'unknown')
                        fn_reasons[stop_reason] += 1
                        
        # Calculate metrics for this group
        eval_res = KeyframeDiagnosticComparator.evaluate_mode(
            gt_dict=gt_dict,
            algo_selected_map={0: all_selected},
            algo_audit_map=None,
            frames_features=frames_feat,
            quality_map=q_map
        )
        
        metrics[s][m]['tp'] += eval_res['metrics']['tp']
        metrics[s][m]['fp'] += eval_res['metrics']['fp']
        metrics[s][m]['fn'] += eval_res['metrics']['fn']
        
        metrics['AGGREGATED'][m]['tp'] += eval_res['metrics']['tp']
        metrics['AGGREGATED'][m]['fp'] += eval_res['metrics']['fp']
        metrics['AGGREGATED'][m]['fn'] += eval_res['metrics']['fn']

print("========================================")
print("PER-SESSION METRICS")
print("========================================")
for s in sessions:
    print(f"\\n--- {s} ---")
    for m in modes:
        tp = metrics[s][m]['tp']
        fp = metrics[s][m]['fp']
        fn = metrics[s][m]['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"Mode {m:2s} -> P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}  (TP:{tp}, FP:{fp}, FN:{fn})")

print("\\n========================================")
print("AGGREGATED METRICS (All 4 Sessions)")
print("========================================")
for m in modes:
    tp = metrics['AGGREGATED'][m]['tp']
    fp = metrics['AGGREGATED'][m]['fp']
    fn = metrics['AGGREGATED'][m]['fn']
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"Mode {m:2s} -> P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}  (TP:{tp}, FP:{fp}, FN:{fn})")

print("\\n========================================")
print("VERIFICATION OF ROOT CAUSES (Mode E1)")
print("========================================")
print("False Positive Reasons:")
for k, v in fp_reasons.items():
    print(f"  - {k}: {v} cases")
if fp_raw_gains:
    avg_gain = sum(fp_raw_gains)/len(fp_raw_gains)
    print(f"  -> Average Raw Gain of FPs: {avg_gain:.4f}")

print("\\nFalse Negative (Omission) Stopping Reasons:")
for k, v in fn_reasons.items():
    print(f"  - {k}: {v} cases")
