import os, sys, json, difflib
import numpy as np
sys.path.insert(0, os.getcwd())
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.keyframe_selector import KeyframeSelectorL2_2
from modules.l2.feature_extractor import load_cached_features

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']
windows = [1, 3, 5, 10, 20]

edit_stats = {w: {'GT': [], 'FP': [], 'ORD': []} for w in windows}
gt_types = {'Transition': 0, 'Settled Rep': 0, 'Pre-transition': 0, 'Continuous': 0}
fp_patterns = {'Transient (A->B->A)': 0, 'Persistent (A->B->B)': 0, 'Evolving (A->B->C->D)': 0, 'Continuous': 0}

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
    trans_map = {(t['frame_a'], t['frame_b']): t for t in cached_data.get('transitions', [])}
    
    for g in groups:
        res = KeyframeSelectorL2_2.select_group_keyframes(
            group_dict=g, mode='E1', tau_coverage=0.95, epsilon_info=0.02,
            w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings={},
            cached_transitions_map=trans_map, quality_map={}
        )
        selected = res['selected_filenames']
        group_frames = g.get('frames', [])
        fps = [fn for fn in selected if fn not in gt_dict]
        
        for i, fr in enumerate(group_frames):
            fn = fr['filename']
            is_gt = fn in gt_dict
            is_fp = fn in fps
            cat = 'GT' if is_gt else ('FP' if is_fp else 'ORD')
            f_toks = frames_feat.get(fn, {}).get('tokens', [])
            
            # Forward window persistence
            for w in windows:
                if i + w < len(group_frames):
                    f_fwd_toks = frames_feat.get(group_frames[i+w]['filename'], {}).get('tokens', [])
                    ed_fwd = get_edit_dist(f_toks, f_fwd_toks)
                    edit_stats[w][cat].append(ed_fwd)
            
            # FP Temporal pattern
            if is_fp and i >= 5 and i + 5 < len(group_frames):
                t_m5 = frames_feat.get(group_frames[i-5]['filename'], {}).get('tokens', [])
                t_p5 = frames_feat.get(group_frames[i+5]['filename'], {}).get('tokens', [])
                
                d_m5_p5 = get_edit_dist(t_m5, t_p5)
                d_m5_t = get_edit_dist(t_m5, f_toks)
                d_t_p5 = get_edit_dist(f_toks, t_p5)
                
                if d_m5_p5 < 0.05 and d_m5_t > 0.05:
                    fp_patterns['Transient (A->B->A)'] += 1
                elif d_t_p5 < 0.05 and d_m5_t > 0.05:
                    fp_patterns['Persistent (A->B->B)'] += 1
                elif d_m5_t > 0.05 and d_t_p5 > 0.05:
                    fp_patterns['Evolving (A->B->C->D)'] += 1
                else:
                    fp_patterns['Continuous'] += 1
            
            # GT transition vs representative
            if is_gt and i >= 3 and i + 3 < len(group_frames):
                t_m3 = frames_feat.get(group_frames[i-3]['filename'], {}).get('tokens', [])
                t_p3 = frames_feat.get(group_frames[i+3]['filename'], {}).get('tokens', [])
                
                d_m3_t = get_edit_dist(t_m3, f_toks)
                d_t_p3 = get_edit_dist(f_toks, t_p3)
                
                if d_t_p3 < 0.05 and d_m3_t > 0.05:
                    gt_types['Settled Rep'] += 1
                elif d_t_p3 > 0.05 and d_m3_t > 0.05:
                    gt_types['Transition'] += 1
                elif d_t_p3 > 0.05 and d_m3_t < 0.05:
                    gt_types['Pre-transition'] += 1
                else:
                    gt_types['Continuous'] += 1

print("--- Persistence (OCR Edit Dist) ---")
for w in windows:
    print(f"Window +{w} frames:")
    for cat in ['GT', 'FP', 'ORD']:
        vals = edit_stats[w][cat]
        if vals:
            print(f"  {cat}: mean={np.mean(vals):.3f}")

print("\n--- GT Transition vs Representative ---")
for k, v in gt_types.items():
    print(f"  {k}: {v}")

print("\n--- FP Temporal Patterns ---")
for k, v in fp_patterns.items():
    print(f"  {k}: {v}")
