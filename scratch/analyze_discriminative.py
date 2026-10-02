import os, sys, json, difflib
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.getcwd())
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.keyframe_selector import KeyframeSelectorL2_2
from modules.l2.feature_extractor import load_cached_features
from modules.l2.quality_cache import load_or_compute_quality_cache

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']

# Store features
features = {'GT': defaultdict(list), 'FP': defaultdict(list)}
seq_anatomy = []

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
    
    # Q-cache
    q_cache = load_or_compute_quality_cache(s_path, list(frames_feat.keys()))
    q_map = {fn: rec['quality_score'] for fn, rec in q_cache.get('frames', {}).items()}

    for g in groups:
        res = KeyframeSelectorL2_2.select_group_keyframes(
            group_dict=g, mode='E1', tau_coverage=0.95, epsilon_info=0.02,
            w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings={},
            cached_transitions_map=trans_map, quality_map=q_map
        )
        selected = res['selected_filenames']
        audit_trail = res.get('audit_trail', [])
        audit_map = {item['chosen_frame'] if 'chosen_frame' in item else item['filename']: item for item in audit_trail}
        
        group_frames = g.get('frames', [])
        if not group_frames: continue
        
        fps = [fn for fn in selected if fn not in gt_dict]
        gts = [fr['filename'] for fr in group_frames if fr['filename'] in gt_dict]
        
        anchor_toks = frames_feat.get(group_frames[0]['filename'], {}).get('tokens', [])
        group_start_t = group_frames[0]['timestamp_sec']
        group_dur = group_frames[-1]['timestamp_sec'] - group_start_t
        
        # Seq anatomy FP -> GT
        if gts and fps:
            for gt in gts:
                gt_idx = next(i for i, fr in enumerate(group_frames) if fr['filename'] == gt)
                gt_time = group_frames[gt_idx]['timestamp_sec']
                
                prev_gt_idx = next((i for i, fr in enumerate(group_frames) if fr['filename'] in gts and i < gt_idx), -1)
                
                fps_in_seq = [fp for fp in fps if prev_gt_idx < next(i for i, fr in enumerate(group_frames) if fr['filename'] == fp) < gt_idx]
                if fps_in_seq:
                    first_fp_time = next(fr['timestamp_sec'] for fr in group_frames if fr['filename'] == fps_in_seq[0])
                    seq_dur = gt_time - first_fp_time
                    seq_anatomy.append({
                        'session': s,
                        'num_intermediate_fps': len(fps_in_seq),
                        'duration': seq_dur
                    })
        
        for i, fr in enumerate(group_frames):
            fn = fr['filename']
            is_gt = fn in gts
            is_fp = fn in fps
            if not is_gt and not is_fp: continue
            
            cat = 'GT' if is_gt else 'FP'
            f_toks = frames_feat.get(fn, {}).get('tokens', [])
            
            # Find previous E1 selected frame (or group anchor if none)
            prev_sel_idx = next((j for j in range(i-1, -1, -1) if group_frames[j]['filename'] in selected), 0)
            prev_sel_fn = group_frames[prev_sel_idx]['filename']
            prev_toks = frames_feat.get(prev_sel_fn, {}).get('tokens', [])
            
            loc_ed = get_edit_dist(prev_toks, f_toks)
            cum_ed = get_edit_dist(anchor_toks, f_toks)
            
            audit = audit_map.get(fn, {})
            raw_gain = audit.get('raw_info_gain', 0.0) if is_fp else 0.0 # GTs often skipped, so raw gain isn't in audit map if it wasn't selected
            
            t_dist_prev = fr['timestamp_sec'] - group_frames[prev_sel_idx]['timestamp_sec']
            pos_ratio = i / len(group_frames)
            
            features[cat]['ocr_token_count'].append(len(f_toks))
            features[cat]['quality'].append(q_map.get(fn, 0.0))
            features[cat]['edit_dist_local'].append(loc_ed)
            features[cat]['edit_dist_cum'].append(cum_ed)
            features[cat]['t_dist_prev'].append(t_dist_prev)
            features[cat]['pos_ratio'].append(pos_ratio)

print("--- Feature Distributions ---")
for feat in ['ocr_token_count', 'quality', 'edit_dist_local', 'edit_dist_cum', 't_dist_prev', 'pos_ratio']:
    gt_vals = features['GT'][feat]
    fp_vals = features['FP'][feat]
    print(f"Feature: {feat}")
    print(f"  GT: mean={np.mean(gt_vals):.4f}, median={np.median(gt_vals):.4f}, std={np.std(gt_vals):.4f}, min={np.min(gt_vals):.4f}, max={np.max(gt_vals):.4f}")
    print(f"  FP: mean={np.mean(fp_vals):.4f}, median={np.median(fp_vals):.4f}, std={np.std(fp_vals):.4f}, min={np.min(fp_vals):.4f}, max={np.max(fp_vals):.4f}")

print("\\n--- Sequence Anatomy ---")
print(f"Total sequences analyzed: {len(seq_anatomy)}")
durations = [x['duration'] for x in seq_anatomy]
fp_counts = [x['num_intermediate_fps'] for x in seq_anatomy]
print(f"Sequence Duration: mean={np.mean(durations):.1f}s, max={np.max(durations):.1f}s")
print(f"Num intermediate FPs: mean={np.mean(fp_counts):.1f}, max={np.max(fp_counts)}")
