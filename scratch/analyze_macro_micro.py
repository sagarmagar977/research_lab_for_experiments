import os, sys, json, difflib
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.getcwd())
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.keyframe_selector import KeyframeSelectorL2_2
from modules.l2.feature_extractor import load_cached_features
from modules.l2.quality_cache import load_or_compute_quality_cache

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']

# Stats to gather
local_changes = {'FP': [], 'GT': []}
cumulative_changes = {'FP': [], 'GT': []}

# Penalty simulation
# We will simulate whether Novelty - Penalty > 0 keeps GTs and drops FPs
lambdas = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
sim_results = {s: {l: {'FP_kept': 0, 'FP_dropped': 0, 'GT_kept': 0, 'GT_dropped': 0} for l in lambdas} for s in sessions}

def get_edit_dist(f1_toks, f2_toks):
    if not f1_toks and not f2_toks: return 0.0
    return 1.0 - difflib.SequenceMatcher(None, f1_toks, f2_toks).ratio()

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
    
    for g in groups:
        res = KeyframeSelectorL2_2.select_group_keyframes(
            group_dict=g, mode='E1', tau_coverage=0.95, epsilon_info=0.02,
            w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings={},
            cached_transitions_map={}, quality_map={}
        )
        selected = res['selected_filenames']
        group_frames = g.get('frames', [])
        fps = [fn for fn in selected if fn not in gt_dict]
        gts = [fr['filename'] for fr in group_frames if fr['filename'] in gt_dict]
        
        # We simulate a sequence of events. 
        # For simplicity, calculate local change (vs immediate prior frame) 
        # and cumulative change (vs frame at index 0 of the group, assuming it's the last anchor)
        
        if len(group_frames) == 0: continue
        anchor_toks = frames_feat.get(group_frames[0]['filename'], {}).get('tokens', [])
        
        for i, fr in enumerate(group_frames):
            fn = fr['filename']
            is_gt = fn in gts
            is_fp = fn in fps
            
            if not is_gt and not is_fp: continue
            
            f_toks = frames_feat.get(fn, {}).get('tokens', [])
            prev_toks = frames_feat.get(group_frames[i-1]['filename'], {}).get('tokens', []) if i > 0 else anchor_toks
            
            loc_dist = get_edit_dist(prev_toks, f_toks)
            cum_dist = get_edit_dist(anchor_toks, f_toks)
            
            if is_fp:
                local_changes['FP'].append(loc_dist)
                cumulative_changes['FP'].append(cum_dist)
                for l in lambdas:
                    if cum_dist - l > 0: sim_results[s][l]['FP_kept'] += 1
                    else: sim_results[s][l]['FP_dropped'] += 1
            if is_gt:
                local_changes['GT'].append(loc_dist)
                cumulative_changes['GT'].append(cum_dist)
                for l in lambdas:
                    if cum_dist - l > 0: sim_results[s][l]['GT_kept'] += 1
                    else: sim_results[s][l]['GT_dropped'] += 1

print("--- Local vs Cumulative Change (Edit Distance) ---")
print(f"FP Local: {np.mean(local_changes['FP']):.4f} | FP Cumulative: {np.mean(cumulative_changes['FP']):.4f}")
print(f"GT Local: {np.mean(local_changes['GT']):.4f} | GT Cumulative: {np.mean(cumulative_changes['GT']):.4f}")

print("\\n--- Penalty Simulation (Novelty - Lambda > 0) ---")
for s in sessions:
    print(f"Session: {s}")
    for l in lambdas:
        fp_kept = sim_results[s][l]['FP_kept']
        fp_dropped = sim_results[s][l]['FP_dropped']
        gt_kept = sim_results[s][l]['GT_kept']
        gt_dropped = sim_results[s][l]['GT_dropped']
        print(f"  Lambda={l:.2f} -> FP kept:{fp_kept:3d}, dropped:{fp_dropped:3d} | GT kept:{gt_kept:3d}, dropped:{gt_dropped:3d}")
