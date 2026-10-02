import os, sys, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.getcwd())
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.keyframe_selector import KeyframeSelectorL2_2
from modules.l2.feature_extractor import load_cached_features

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']

features = {'GT': defaultdict(list), 'FP': defaultdict(list)}
coding_cases = []

def jaccard(set1, set2):
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def asym_containment(target, source):
    if not target: return 1.0
    return len(set(target).intersection(set(source))) / len(set(target))

def cosine_sim(v1, v2):
    if len(v1) == 0 or len(v2) == 0: return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

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
        fps = [fn for fn in selected if fn not in gt_dict]
        gts = [fr['filename'] for fr in group_frames if fr['filename'] in gt_dict]
        
        N = len(group_frames)
        if N < 2: continue
        
        vit_sim = np.zeros((N, N))
        ocr_jaccard = np.zeros((N, N))
        ocr_asym = np.zeros((N, N))
        
        for i in range(N):
            fn_i = group_frames[i]['filename']
            toks_i = frames_feat.get(fn_i, {}).get('tokens', [])
            vit_i = vit_map.get(fn_i, [])
            for j in range(N):
                if i == j: continue
                fn_j = group_frames[j]['filename']
                toks_j = frames_feat.get(fn_j, {}).get('tokens', [])
                vit_j = vit_map.get(fn_j, [])
                
                if len(vit_i) > 0 and len(vit_j) > 0:
                    vit_sim[i, j] = cosine_sim(vit_i, vit_j)
                ocr_jaccard[i, j] = jaccard(set(toks_i), set(toks_j))
                ocr_asym[i, j] = asym_containment(toks_i, toks_j)
                
        for i in range(N):
            fn = group_frames[i]['filename']
            is_gt = fn in gts
            is_fp = fn in fps
            if not is_gt and not is_fp: continue
            
            cat = 'GT' if is_gt else 'FP'
            
            v_sims = np.delete(vit_sim[i], i)
            o_jacc = np.delete(ocr_jaccard[i], i)
            o_asym = np.delete(ocr_asym[i], i)
            o_asym_rev = np.delete(ocr_asym[:, i], i)
            
            features[cat]['mean_vit'].append(np.mean(v_sims))
            features[cat]['mean_ocr_jaccard'].append(np.mean(o_jacc))
            features[cat]['mean_ocr_contains_others'].append(np.mean(o_asym_rev))
            features[cat]['mean_ocr_contained_in_others'].append(np.mean(o_asym))
            features[cat]['max_vit'].append(np.max(v_sims))
            
            centrality = np.mean(v_sims) * 0.5 + np.mean(o_jacc) * 0.5
            features[cat]['centrality'].append(centrality)
            
            if s == 'CODING_SESSION':
                coding_cases.append({
                    'frame': fn, 'cat': cat,
                    'mean_vit': np.mean(v_sims), 'centrality': centrality,
                    'mean_ocr_contains_others': np.mean(o_asym_rev)
                })

print("--- Relational Feature Distributions ---")
for feat in ['mean_vit', 'mean_ocr_jaccard', 'mean_ocr_contains_others', 'mean_ocr_contained_in_others', 'max_vit', 'centrality']:
    gt_vals = features['GT'][feat]
    fp_vals = features['FP'][feat]
    if not gt_vals or not fp_vals: continue
    print(f"Feature: {feat}")
    print(f"  GT: mean={np.mean(gt_vals):.4f}, median={np.median(gt_vals):.4f}, std={np.std(gt_vals):.4f}")
    print(f"  FP: mean={np.mean(fp_vals):.4f}, median={np.median(fp_vals):.4f}, std={np.std(fp_vals):.4f}")
