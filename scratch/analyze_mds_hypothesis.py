import os, sys, json
import numpy as np
sys.path.insert(0, os.getcwd())
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.keyframe_selector import KeyframeSelectorL2_2
from modules.l2.feature_extractor import load_cached_features

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']
results = {s: {'explain_fp_by_gt': [], 'explain_gt_by_fp': [], 'counterexamples': []} for s in sessions}

def get_explainability(f_target, f_rep):
    if not f_target: return 1.0
    intersection = len(set(f_target).intersection(set(f_rep)))
    return intersection / len(set(f_target))

for s in sessions:
    s_path = os.path.join('sessions', s)
    runs_file = os.path.join(s_path, 'level2', 'grouping_runs.json')
    if not os.path.exists(runs_file): continue
    
    run_data = json.load(open(runs_file, 'r', encoding='utf-8'))
    part_key = 'approach_a2' if 'approach_a2' in run_data else list(run_data.keys())[2]
    groups = run_data[part_key]['groups']
    gt_dict = load_keyframe_ground_truth(s_path)
    if not gt_dict: continue

    cached_data, _ = load_cached_features(s_path)
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
        
        for fp in fps:
            fp_idx = next((i for i, fr in enumerate(group_frames) if fr['filename'] == fp), -1)
            subsequent_gts = [fn for fn in gts if next((i for i, fr in enumerate(group_frames) if fr['filename'] == fn), -1) > fp_idx]
            
            if subsequent_gts:
                nearest_gt = subsequent_gts[0]
                fp_toks = frames_feat.get(fp, {}).get('tokens', [])
                gt_toks = frames_feat.get(nearest_gt, {}).get('tokens', [])
                
                ex_fp_by_gt = get_explainability(fp_toks, gt_toks)
                ex_gt_by_fp = get_explainability(gt_toks, fp_toks)
                
                results[s]['explain_fp_by_gt'].append(ex_fp_by_gt)
                results[s]['explain_gt_by_fp'].append(ex_gt_by_fp)
                
                if ex_fp_by_gt < 0.8:
                    results[s]['counterexamples'].append({'fp': fp, 'gt': nearest_gt, 'ex_fp_by_gt': ex_fp_by_gt})

for s in sessions:
    print(f"Session: {s}")
    fp_by_gt = results[s]['explain_fp_by_gt']
    gt_by_fp = results[s]['explain_gt_by_fp']
    if fp_by_gt:
        print(f"  Explain(FP | GT): mean={np.mean(fp_by_gt):.3f} (How much FP info is retained in GT)")
        print(f"  Explain(GT | FP): mean={np.mean(gt_by_fp):.3f} (How much GT info was already in FP)")
        print(f"  Total FP->GT pairs: {len(fp_by_gt)}")
        print(f"  Counterexamples (Explain(FP|GT) < 80%): {len(results[s]['counterexamples'])}")
        for ce in results[s]['counterexamples'][:3]:
            print(f"    - {ce['fp']} -> {ce['gt']} (Retained: {ce['ex_fp_by_gt']:.3f})")
