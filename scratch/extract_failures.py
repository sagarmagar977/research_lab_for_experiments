import os, sys, json
sys.path.insert(0, os.getcwd())
from collections import Counter
import pandas as pd
from modules.l2_manual_keyframe_curator import load_keyframe_ground_truth
from modules.l2.diagnostic_comparator import KeyframeDiagnosticComparator
from modules.l2.keyframe_selector import (
    KeyframeSelectorL2_2,
    get_spatial_cells_16x16,
    compute_vit_cosine_distance
)
from modules.l2.feature_extractor import load_cached_features
from modules.l2.quality_cache import load_or_compute_quality_cache

sessions = ['MATH-SESSION', 'slides_AA', 'linear_algebra_session', 'CODING_SESSION']
modes = ['A', 'B', 'C', 'D', 'E1', 'E2']

failures = []

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
        use_multiset = (m != 'A')
        use_spatial = (m in ('C', 'D', 'E1', 'E2'))
        use_quality = (m in ('D', 'E1', 'E2'))

        # Track first FP and first FN for each mode in each session
        found_fp = None
        found_fn = None

        for g in groups:
            group_frames = g.get('frames', [])
            res = KeyframeSelectorL2_2.select_group_keyframes(
                group_dict=g, mode=m, tau_coverage=0.95, epsilon_info=0.02,
                w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings=vit_map,
                cached_transitions_map=trans_map, quality_map=q_map
            )
            selected_names = res['selected_filenames']
            audit_trail = res.get('audit_trail', [])
            stopping_reason = res.get('stopping_reason', 'unknown')

            # Build universe for unweighted gains calculation
            if m == 'A':
                u_lex = set()
                for fr in group_frames: u_lex.update(fr.get('tokens', []))
                tot_lex = len(u_lex)
            else:
                max_c = Counter()
                for fr in group_frames:
                    fc = Counter(fr.get('tokens', []))
                    for tok, cnt in fc.items():
                        if cnt > max_c[tok]: max_c[tok] = cnt
                tot_lex = sum(max_c.values())
            
            u_sp = set()
            if use_spatial:
                for fr in group_frames: u_sp.update(get_spatial_cells_16x16(fr.get('bboxes', [])))
            tot_sp = len(u_sp)
            eff_w_sp = 0.15 if (use_spatial and tot_sp > 0) else 0.0

            # Map audit info for selected frames
            audit_map = {item['filename']: item for item in audit_trail}

            # Check False Positives: in selected_names, but NOT in gt_dict
            if not found_fp:
                for fn in selected_names:
                    if fn not in gt_dict:
                        f_obj = frames_feat.get(fn, {})
                        a_info = audit_map.get(fn, {})
                        
                        # Find adjacent transition for SSIM
                        idx_in_g = next((i for i, fr in enumerate(group_frames) if fr['filename'] == fn), 0)
                        prev_fn = group_frames[idx_in_g - 1]['filename'] if idx_in_g > 0 else None
                        t_data = trans_map.get((prev_fn, fn), {}) if prev_fn else {}
                        
                        found_fp = {
                            'session': s,
                            'mode': m,
                            'type': 'False Positive',
                            'frame': fn,
                            'group_id': g['group_id'],
                            'human_label': 'None (Spurious)',
                            'status': 'Selected by Algorithm',
                            'temporal_pos': f"{f_obj.get('timestamp_str', '')} (t={f_obj.get('timestamp_sec', 0):.1f}s, seq={f_obj.get('candidate_seq_idx', idx_in_g)})",
                            'ocr_gain': a_info.get('raw_info_gain', 0.0),
                            'spatial_gain': 'N/A' if not use_spatial else round((a_info.get('raw_info_gain', 0) - (1-eff_w_sp)*(len(set(f_obj.get('tokens',[])))/max(tot_lex,1)))/max(eff_w_sp,1e-6), 4),
                            'ssim_transition': t_data.get('d_ssim', 0.0),
                            'vit_distance': a_info.get('vit_dist', 0.0),
                            'quality_score': q_map.get(fn, f_obj.get('quality_score', 1.0)),
                            'event_evidence': a_info.get('event_score', 0.0),
                            'stopping_reason': f"Selected at Round {audit_trail.index(a_info)+1 if a_info in audit_trail else 1} (Score: {a_info.get('selection_score', 0):.4f}, Raw Gain: {a_info.get('raw_info_gain', 0):.4f} >= 0.02). Group stopping reason: '{stopping_reason}'."
                        }
                        break

            # Check False Negatives: in gt_dict and in this group, but NOT in selected_names
            if not found_fn:
                for fr in group_frames:
                    fn = fr['filename']
                    if fn in gt_dict and fn not in selected_names:
                        f_obj = frames_feat.get(fn, {})
                        idx_in_g = next((i for i, f_item in enumerate(group_frames) if f_item['filename'] == fn), 0)
                        prev_fn = group_frames[idx_in_g - 1]['filename'] if idx_in_g > 0 else None
                        t_data = trans_map.get((prev_fn, fn), {}) if prev_fn else {}
                        
                        # Calculate what raw gain this frame would have offered against initial or final state
                        # Check why it was skipped
                        v_curr = vit_map.get(fn)
                        v_d = 0.0
                        if v_curr is not None and selected_names:
                            sel_v = [vit_map.get(sn) for sn in selected_names if sn in vit_map]
                            if sel_v:
                                v_d = min(compute_vit_cosine_distance(v_curr, sv) for sv in sel_v)
                        
                        # Marginal gain at end of loop
                        # Tokens already covered
                        cov_tokens = set()
                        for sn in selected_names:
                            cov_tokens.update(frames_feat.get(sn, {}).get('tokens', []))
                        new_t = set(f_obj.get('tokens', [])) - cov_tokens
                        marginal_lex_gain = len(new_t) / max(tot_lex, 1)

                        found_fn = {
                            'session': s,
                            'mode': m,
                            'type': 'False Negative',
                            'frame': fn,
                            'group_id': g['group_id'],
                            'human_label': f"{gt_dict[fn].get('role_type', '').upper()} Keyframe",
                            'status': 'Not Selected (Omitted)',
                            'temporal_pos': f"{f_obj.get('timestamp_str', '')} (t={f_obj.get('timestamp_sec', 0):.1f}s, seq={f_obj.get('candidate_seq_idx', idx_in_g)})",
                            'ocr_gain': round(marginal_lex_gain, 4),
                            'spatial_gain': 0.0 if not use_spatial else 'Subsumed',
                            'ssim_transition': t_data.get('d_ssim', 0.0),
                            'vit_distance': round(v_d, 4),
                            'quality_score': q_map.get(fn, f_obj.get('quality_score', 1.0)),
                            'event_evidence': t_data.get('ocr_loss', 0.0),
                            'stopping_reason': f"Premature group termination via '{stopping_reason}' (Target coverage {res.get('final_coverage',0):.2%} reached before inspecting this frame; or marginal gain {marginal_lex_gain:.4f} < floor 0.02)."
                        }
                        break

            if found_fp and found_fn:
                break
        
        if found_fp: failures.append(found_fp)
        if found_fn: failures.append(found_fn)

with open('scratch/failures_diagnosis.json', 'w') as f:
    json.dump(failures, f, indent=2)

print(f'Harvested {len(failures)} representative failure cases across all sessions and modes.')
