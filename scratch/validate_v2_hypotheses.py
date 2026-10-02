import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np
import difflib
from collections import defaultdict
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.keyframe_selector import KeyframeSelectorL2_2, get_spatial_cells_16x16
from modules.l2.feature_extractor import load_cached_features
from modules.l2.quality_cache import load_or_compute_quality_cache

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']
mode = 'E1'

# Stats for Exp 1
gt_ssim_diffs = []
fp_ssim_diffs = []
ord_ssim_diffs = []

# Stats for Exp 2 (Coding FNs)
coding_fns_data = []

# Stats for Exp 3
fp_to_fn_replacements = []

# Stats for Exp 4
independent_gains = []

# Exp 5 Matrix
matrix_data = []

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
    
    # transitions are stored as dict mapping (frame_a, frame_b) -> dict
    trans_map = {(t['frame_a'], t['frame_b']): t for t in cached_data.get('transitions', [])}
    
    q_cache = load_or_compute_quality_cache(s_path, list(frames_feat.keys()))
    q_map = {fn: rec['quality_score'] for fn, rec in q_cache.get('frames', {}).items()}
    vit_map = cached_vit or {}

    for g in groups:
        res = KeyframeSelectorL2_2.select_group_keyframes(
            group_dict=g, mode=mode, tau_coverage=0.95, epsilon_info=0.02,
            w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings=vit_map,
            cached_transitions_map=trans_map, quality_map=q_map
        )
        selected = res['selected_filenames']
        audit_trail = res.get('audit_trail', [])
        audit_map = {item['chosen_frame'] if 'chosen_frame' in item else item['filename']: item for item in audit_trail}
        
        group_frames = g.get('frames', [])
        gt_in_group = [fr['filename'] for fr in group_frames if fr['filename'] in gt_dict]
        
        # Determine FPs and FNs
        fps = [fn for fn in selected if fn not in gt_dict]
        fns = [fn for fn in gt_in_group if fn not in selected]
        
        # Exp 1: Stability (using d_ssim with previous frame as backward, and next frame as forward if possible)
        for i, fr in enumerate(group_frames):
            fn = fr['filename']
            prev_fn = group_frames[i-1]['filename'] if i > 0 else None
            next_fn = group_frames[i+1]['filename'] if i < len(group_frames)-1 else None
            
            d_ssim_back = trans_map.get((prev_fn, fn), {}).get('d_ssim', 0.0) if prev_fn else 0.0
            d_ssim_fwd = trans_map.get((fn, next_fn), {}).get('d_ssim', 0.0) if next_fn else 0.0
            avg_d_ssim = (d_ssim_back + d_ssim_fwd) / 2.0
            
            if fn in gt_dict:
                gt_ssim_diffs.append(avg_d_ssim)
            elif fn in fps:
                fp_ssim_diffs.append(avg_d_ssim)
            else:
                ord_ssim_diffs.append(avg_d_ssim)
                
        # Exp 2 & 4: Coding FNs
        if s == 'CODING_SESSION':
            for fn in fns:
                idx = next(i for i, fr in enumerate(group_frames) if fr['filename'] == fn)
                f_feat = frames_feat.get(fn, {})
                prev_fn = group_frames[idx-1]['filename'] if idx > 0 else None
                prev_feat = frames_feat.get(prev_fn, {}) if prev_fn else {}
                
                tok_f = f_feat.get('tokens', [])
                tok_p = prev_feat.get('tokens', [])
                
                set_diff = len(set(tok_f) - set(tok_p))
                edit_dist = 1.0 - difflib.SequenceMatcher(None, tok_f, tok_p).ratio()
                
                # independent gain: if evaluated alone against empty
                indep_gain = len(set(tok_f))
                
                coding_fns_data.append({
                    'frame': fn,
                    'set_diff': set_diff,
                    'edit_dist': edit_dist,
                    'indep_gain': indep_gain
                })
        
        # Exp 3: Replacements (noisy FP followed by FN)
        if fps and fns:
            for fp in fps:
                idx_fp = next(i for i, fr in enumerate(group_frames) if fr['filename'] == fp)
                for fn in fns:
                    idx_fn = next(i for i, fr in enumerate(group_frames) if fr['filename'] == fn)
                    if idx_fn > idx_fp:
                        t_dist = group_frames[idx_fn]['timestamp_sec'] - group_frames[idx_fp]['timestamp_sec']
                        fp_to_fn_replacements.append({
                            'session': s,
                            'fp': fp,
                            'fn': fn,
                            'temporal_dist': t_dist,
                            'fp_qual': q_map.get(fp, 0),
                            'fn_qual': q_map.get(fn, 0)
                        })
                        
        # Exp 5: Matrix Data
        for fp in fps:
            audit = audit_map.get(fp, {})
            matrix_data.append({
                'Frame': fp, 'GT': 'FP', 
                'RawGain': audit.get('raw_info_gain', 0.0),
                'Quality': q_map.get(fp, 0),
                'Event': audit.get('event_score', 0.0)
            })
        for fn in fns:
            matrix_data.append({
                'Frame': fn, 'GT': 'FN', 
                'RawGain': 0.0, # Typically 0 or near 0 when skipped
                'Quality': q_map.get(fn, 0),
                'Event': 0.0 # Would need to recalculate
            })

print("EXPERIMENT 1: STABILITY")
print(f"GT Keyframes avg d_ssim: {np.mean(gt_ssim_diffs):.4f} (lower is more stable)")
print(f"FP Keyframes avg d_ssim: {np.mean(fp_ssim_diffs):.4f}")
print(f"Ordinary frames avg d_ssim: {np.mean(ord_ssim_diffs):.4f}")

print("\\nEXPERIMENT 2: SAME-REGION (CODING)")
for data in coding_fns_data[:5]:
    print(f"FN: {data['frame']} -> Set diff: {data['set_diff']}, Edit Dist: {data['edit_dist']}, Indep Gain: {data['indep_gain']}")

print("\\nEXPERIMENT 3: REPLACEMENTS")
print(f"Found {len(fp_to_fn_replacements)} instances where an FP was selected before a missed FN in the same group.")
if fp_to_fn_replacements:
    avg_t_dist = np.mean([r['temporal_dist'] for r in fp_to_fn_replacements])
    better_qual = sum(1 for r in fp_to_fn_replacements if r['fn_qual'] > r['fp_qual'])
    print(f"Average temporal distance: {avg_t_dist:.1f} seconds")
    print(f"FN has better quality than FP in {better_qual}/{len(fp_to_fn_replacements)} cases.")

print("\\nEXPERIMENT 5: MATRIX (Sample)")
print(f"Collected {len(matrix_data)} rows for matrix.")
