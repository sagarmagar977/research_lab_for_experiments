import os, sys, json
sys.path.insert(0, os.getcwd())
from modules.l2.keyframe_selector import KeyframeSelectorL2_2
from modules.l2.feature_extractor import load_cached_features

s_path = os.path.join('sessions', 'MATH-SESSION')
runs_file = os.path.join(s_path, 'level2', 'grouping_runs.json')
run_data = json.load(open(runs_file, 'r', encoding='utf-8'))
part_key = 'approach_a2' if 'approach_a2' in run_data else list(run_data.keys())[2]
groups = run_data[part_key]['groups']

cached_data, cached_vit = load_cached_features(s_path)
vit_map = cached_vit or {}

print("Testing Group 0 Mode F...")
g = groups[0]
res = KeyframeSelectorL2_2.select_group_keyframes(
    group_dict=g, mode='F', tau_coverage=0.95, epsilon_info=0.02,
    w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings=vit_map,
    cached_transitions_map={}, quality_map={}
)
print("Finished Group 0 Mode F. Selected:", res['selected_count'])

print("Testing Group 1 Mode F...")
g = groups[1]
res = KeyframeSelectorL2_2.select_group_keyframes(
    group_dict=g, mode='F', tau_coverage=0.95, epsilon_info=0.02,
    w_spatial=0.15, w_vit=0.20, w_event=0.20, vit_embeddings=vit_map,
    cached_transitions_map={}, quality_map={}
)
print("Finished Group 1 Mode F. Selected:", res['selected_count'])
print("Success!")
