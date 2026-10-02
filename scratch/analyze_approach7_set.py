import os, sys, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.getcwd())
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.keyframe_selector import KeyframeSelectorL2_2
from modules.l2.feature_extractor import load_cached_features

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']
set_features = {'GT_SET': defaultdict(list), 'FP_SET': defaultdict(list)}

def jaccard(set1, set2):
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def cosine_sim(v1, v2):
    if len(v1) == 0 or len(v2) == 0: return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def calc_set_stats(K_indices, vit_sim, ocr_jaccard, group_frames, frames_feat):
    if not K_indices:
        return None
    
    # Intra-set diversity
    k_vit_sims = []
    k_ocr_jacc = []
    for i in range(len(K_indices)):
        for j in range(i+1, len(K_indices)):
            k_vit_sims.append(vit_sim[K_indices[i], K_indices[j]])
            k_ocr_jacc.append(ocr_jaccard[K_indices[i], K_indices[j]])
            
    mean_intra_vit_sim = np.mean(k_vit_sims) if k_vit_sims else 1.0
    mean_intra_ocr_jacc = np.mean(k_ocr_jacc) if k_ocr_jacc else 1.0
    
    # Facility Location Coverage
    N = vit_sim.shape[0]
    fac_loc_vit = 0.0
    for i in range(N):
        max_sim = max([vit_sim[i, k] for k in K_indices])
        fac_loc_vit += max_sim
    fac_loc_vit /= N
    
    # Complementarity (Tokens)
    union_toks = set()
    sum_toks = 0
    for k in K_indices:
        fn = group_frames[k]['filename']
        toks = set(frames_feat.get(fn, {}).get('tokens', []))
        union_toks.update(toks)
        sum_toks += len(toks)
        
    complementarity = len(union_toks) / sum_toks if sum_toks > 0 else 1.0
    
    return {
        'size': len(K_indices),
        'mean_intra_vit_sim': mean_intra_vit_sim,
        'mean_intra_ocr_jacc': mean_intra_ocr_jacc,
        'fac_loc_vit': fac_loc_vit,
        'complementarity': complementarity
    }

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
    vit_map = cached_vit or {}

    for g in groups:
        res = KeyframeSelectorL2_2.select_group_keyframes(
            group_dict=g, mode='E1', tau_coverage=0.95, epsilon_info=0.02,
            w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings={},
            cached_transitions_map={}, quality_map={}
        )
        selected = res['selected_filenames']
        group_frames = g.get('frames', [])
        N = len(group_frames)
        if N < 2: continue
        
        # Build pairwise
        vit_sim = np.zeros((N, N))
        ocr_jaccard = np.zeros((N, N))
        
        for i in range(N):
            fn_i = group_frames[i]['filename']
            toks_i = frames_feat.get(fn_i, {}).get('tokens', [])
            vit_i = vit_map.get(fn_i, [])
            for j in range(N):
                fn_j = group_frames[j]['filename']
                toks_j = frames_feat.get(fn_j, {}).get('tokens', [])
                vit_j = vit_map.get(fn_j, [])
                
                if len(vit_i) > 0 and len(vit_j) > 0:
                    vit_sim[i, j] = cosine_sim(vit_i, vit_j)
                else:
                    vit_sim[i, j] = 1.0 if i == j else 0.0
                    
                ocr_jaccard[i, j] = jaccard(set(toks_i), set(toks_j))

        # Find sets
        fps = [fn for fn in selected if fn not in gt_dict]
        gts = [fr['filename'] for fr in group_frames if fr['filename'] in gt_dict]
        
        gt_indices = [i for i, fr in enumerate(group_frames) if fr['filename'] in gts]
        fp_indices = [i for i, fr in enumerate(group_frames) if fr['filename'] in fps]
        
        gt_stats = calc_set_stats(gt_indices, vit_sim, ocr_jaccard, group_frames, frames_feat)
        fp_stats = calc_set_stats(fp_indices, vit_sim, ocr_jaccard, group_frames, frames_feat)
        
        if gt_stats:
            for k, v in gt_stats.items(): set_features['GT_SET'][k].append(v)
        if fp_stats:
            for k, v in fp_stats.items(): set_features['FP_SET'][k].append(v)

print("--- Set-Level Feature Distributions ---")
for feat in ['size', 'mean_intra_vit_sim', 'mean_intra_ocr_jacc', 'fac_loc_vit', 'complementarity']:
    gt_vals = set_features['GT_SET'].get(feat, [])
    fp_vals = set_features['FP_SET'].get(feat, [])
    if not gt_vals or not fp_vals: continue
    print(f"Feature: {feat}")
    print(f"  GT SET: mean={np.mean(gt_vals):.4f}, median={np.median(gt_vals):.4f}, std={np.std(gt_vals):.4f}")
    print(f"  FP SET: mean={np.mean(fp_vals):.4f}, median={np.median(fp_vals):.4f}, std={np.std(fp_vals):.4f}")
