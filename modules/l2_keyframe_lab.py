import os
import json
import time
import datetime
import traceback
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

from modules.image_cache import get_cached_thumbnail_b64
from modules.l2.feature_extractor import load_cached_features, parse_timestamp_from_filename
from modules.l2.quality_cache import load_or_compute_quality_cache, resolve_candidate_image_dir
from modules.l2.keyframe_selector import KeyframeSelectorL2_2

def get_image_b64_src(path, max_dim=400):
    b64 = get_cached_thumbnail_b64(path, max_dim=max_dim)
    if b64:
        return f"data:image/jpeg;base64,{b64}"
    return None

def render_l2_keyframe_lab():
    col_t_title, col_t_jump = st.columns([2.5, 1.5])
    with col_t_title:
        st.markdown("### Level 2.2 — Minimal Informative Keyframe Selection Lab")
    with col_t_jump:
        def goto_manual_curator():
            st.session_state["_redirect_module"] = "Manual Keyframe Selector"
            st.session_state["selected_module"] = "Manual Keyframe Selector"
            st.session_state["_module_selector_widget"] = "Manual Keyframe Selector"
        st.button("🎯 Open Manual Keyframe Selector", type="primary", use_container_width=True, on_click=goto_manual_curator)


    st.write(
        "Dedicated research interface for **Level 2.2 Keyframe Selection**. "
        "Consumes temporal lesson groups produced by **Level 2.1** and selects the minimal informative subset "
        "of culmination keyframes across six controlled ablation formulations (**A** through **E2**). "
        "All visual candidate frames, thumbnail galleries, and decision audit logs can be inspected across groups."
    )

    sessions_root = "sessions"
    os.makedirs(sessions_root, exist_ok=True)

    # -------------------------------------------------------------
    # 1. SESSION & LEVEL 2.1 RUN INGESTION
    # -------------------------------------------------------------
    st.markdown("#### 1. Ingest Level 2.1 Session & Partition")
    all_raw_sessions = sorted([d for d in os.listdir(sessions_root) if os.path.isdir(os.path.join(sessions_root, d))], reverse=True)
    if not all_raw_sessions:
        st.info("No sessions found in `sessions/`. Please run Level 1 or batch cropping first.")
        return

    # Prioritize sessions with completed Level 2 runs
    valid_l2_sessions = [d for d in all_raw_sessions if os.path.exists(os.path.join(sessions_root, d, "level2", "grouping_runs.json"))]
    all_sessions = valid_l2_sessions + [d for d in all_raw_sessions if d not in valid_l2_sessions]

    col_s1, col_s2, col_s3 = st.columns([1.5, 1.5, 1])

    with col_s1:
        chosen_session = st.selectbox("Select Session", all_sessions, key="l2_2_standalone_sess_pick")

    session_path = os.path.join(sessions_root, chosen_session)
    print(f"[L2.2 DEBUG] Ingesting session: '{chosen_session}'", flush=True)
    l2_dir = os.path.join(session_path, "level2")
    runs_file = os.path.join(l2_dir, "grouping_runs.json")

    # Verify features and Level 2.1 runs exist
    cached_data, cached_vit = load_cached_features(session_path)
    has_l2_features = cached_data is not None
    has_l2_runs = os.path.exists(runs_file)

    if not has_l2_features:
        print(f"[L2.2 WARN] Session '{chosen_session}' missing L2 features.", flush=True)
        st.warning(f"Session `{chosen_session}` does not have Level 2 precomputed features. Please compute features in Level 2 Grouping Lab first.")
        return

    if not has_l2_runs:
        print(f"[L2.2 WARN] Session '{chosen_session}' missing grouping_runs.json.", flush=True)
        st.warning(f"Session `{chosen_session}` does not have saved Level 2.1 grouping runs (`grouping_runs.json`). Please execute Level 2.1 grouping first.")
        return

    with open(runs_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    # Extract available Level 2.1 partitions
    available_partitions = {}
    if "approach_a2" in run_data and "groups" in run_data["approach_a2"]:
        available_partitions["A2: Improved Asymmetric (Default)"] = run_data["approach_a2"]["groups"]
    if "approach_b2" in run_data and "groups" in run_data["approach_b2"]:
        available_partitions["B2: Multimodal (A2 + ViT)"] = run_data["approach_b2"]["groups"]
    if "approach_a1" in run_data and "groups" in run_data["approach_a1"]:
        available_partitions["A1: Frozen Baseline"] = run_data["approach_a1"]["groups"]

    if not available_partitions:
        print(f"[L2.2 ERROR] Session '{chosen_session}' has no valid partitions in grouping_runs.json.", flush=True)
        st.warning("No valid partitions found in `grouping_runs.json`.")
        return

    with col_s2:
        part_key = f"l2_2_part_{chosen_session}"
        chosen_part_label = st.selectbox("Level 2.1 Group Partition", list(available_partitions.keys()), index=0, key=part_key)
        active_groups = available_partitions[chosen_part_label]

    # Resolve candidate image directory
    try:
        cand_img_dir = resolve_candidate_image_dir(session_path)
    except FileNotFoundError as fnf_err:
        print(f"[L2.2 ERROR] Could not locate candidate images directory in '{session_path}': {fnf_err}", flush=True)
        st.error(f"Could not locate candidate images directory in `{session_path}`.")
        return

    frames_list = cached_data["frames"]
    transitions_list = cached_data["transitions"]
    vit_dict = cached_vit or {}

    with col_s3:
        st.metric("Total Candidate Frames", len(frames_list))
        st.caption(f"Groups in partition: **{len(active_groups)}**")

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. SELECTION CONTROLS & HYPERPARAMETERS
    # -------------------------------------------------------------
    st.markdown("#### 2. Configure Keyframe Selection Formulation")

    col_m1, col_m2 = st.columns([1.5, 1.5])
    with col_m1:
        chosen_mode_raw = st.selectbox(
            "Evaluation Formulation (Ablation Mode)",
            [
                "D: Multiset + Spatial + Quality (Recommended Default)",
                "A: Unique Lexical Set",
                "B: Lexical Multiset",
                "C: Multiset + Spatial (16x16 Grid)",
                "E1: D + Gated ViT Diversity",
                "E2: D + Gated ViT + Event Evidence",
                "F: Approach 7 (Set Selection / Facility Location)",
                "Compare All Modes Side-by-Side (A-F)"
            ],
            index=0,
            key="l2_2_standalone_mode_pick"
        )
        mode_code = chosen_mode_raw.split(":")[0].strip()

    with col_m2:
        group_opts = ["Batch All Groups"] + [
            f"Group {g['group_id']} ({g['frame_count']} frames | {g['start_timestamp']} -> {g['end_timestamp']})"
            for g in active_groups
        ]
        scope_key = f"l2_2_scope_{chosen_session}_{chosen_part_label.split(':')[0]}"
        selected_scope = st.selectbox("Processing Scope", group_opts, index=0, key=scope_key)


    with st.expander("⚙️ Advanced Experimental Hyperparameters", expanded=False):
        hp_c1, hp_c2, hp_c3 = st.columns(3)
        with hp_c1:
            tau_cov = st.slider("Target Pedagogical Coverage ($\\tau_{cov}$)", 0.70, 0.99, 0.95, 0.01, key="l2_2_s_tau_cov", help="Stopping target for unweighted pedagogical coverage.")
            eps_info = st.slider("Information Eligibility Floor ($\\epsilon_{info}$)", 0.005, 0.050, 0.020, 0.005, key="l2_2_s_eps_info", help="Minimum raw unweighted information gain required for candidate eligibility.")
            q_min = st.slider("Quality Lower Bound ($q_{min}$)", 0.10, 0.40, 0.20, 0.05, key="l2_2_s_q_min", help="Session-normalized quality floor. Steers selection order without disqualifying unique content.")
        with hp_c2:
            w_spatial = st.slider("Spatial Weight ($w_{spatial}$)", 0.00, 0.30, 0.15, 0.05, key="l2_2_s_w_sp", help="Weight of 16x16 bounding box occupancy in C/D/E1/E2. Reverts to 0 in A/B.")
            w_vit = st.slider("ViT Diversity Bonus ($w_{vit}$)", 0.00, 0.50, 0.20, 0.05, key="l2_2_s_w_vit", help="Multiplicative visual diversity factor in E1/E2. Strictly gated by positive raw gain.")
            w_event = st.slider("Event Evidence Bonus ($w_{event}$)", 0.00, 0.50, 0.20, 0.05, key="l2_2_s_w_ev", help="Multiplicative intra-group transition bonus in E2. Strictly gated by positive raw gain.")
        with hp_c3:
            tau_ev_ocr = st.slider("Event OCR Threshold ($\\tau_{ocr}$)", 0.10, 0.50, 0.30, 0.05, key="l2_2_s_tau_ev_ocr", help="Minimum cached OCR loss to trigger intra-group event.")
            tau_ev_ssim = st.slider("Event SSIM Gate ($\\tau_{ssim}$)", 0.005, 0.050, 0.025, 0.005, key="l2_2_s_tau_ev_ssim", help="Minimum visual difference to confirm non-flicker canvas change.")
            ev_dir = st.radio("Event Direction", ["pre", "post"], index=0, horizontal=True, key="l2_2_s_ev_dir", help="'pre' = culmination before wipe; 'post' = inception after wipe.")

    # Lazy-load session quality cache
    all_frame_names = [f["filename"] for f in frames_list]
    q_cache_data = load_or_compute_quality_cache(session_path, all_frame_names, q_min=q_min)
    quality_lookup = {fn: rec["quality_score"] for fn, rec in q_cache_data.get("frames", {}).items()}

    # Cached transitions map for fast E2 lookup
    trans_lookup = {(t["frame_a"], t["frame_b"]): t for t in transitions_list}

    st.markdown("<br/>", unsafe_allow_html=True)
    run_btn = st.button("🔄 Re-run Keyframe Selection", type="primary", use_container_width=True)

    if True:
        st.session_state["_l2_2_standalone_auto"] = True

        st.markdown("---")
        st.markdown("#### 3. Keyframe Selection Results & Visual Inspection")

        inspect_key = f"l2_2_inspect_frame_{chosen_session}"
        if inspect_key not in st.session_state:
            st.session_state[inspect_key] = None

        selected_inspect = st.session_state[inspect_key]
        if selected_inspect:
            inspect_path = os.path.join(cand_img_dir, selected_inspect)
            if os.path.exists(inspect_path):
                st.markdown(
                    f"<div style='background-color:#18181b; padding:10px 16px; border-radius:10px; border:2px solid #10b981; margin-top:8px; margin-bottom:12px;'>"
                    f"<b style='font-size:1.05rem; color:#10b981;'>Full Resolution Native Inspection: <code>{selected_inspect}</code></b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.image(Image.open(inspect_path), use_container_width=True)
                if st.button("✕ Close Full Preview", key="btn_close_l2_2_inspect", use_container_width=True):
                    st.session_state[inspect_key] = None
                    st.rerun()
                st.markdown("---")

        col_layout = st.radio("Gallery Columns per Row", [3, 4, 6], index=1, horizontal=True, key=f"l2_2_col_layout_{chosen_session}")

        # -------------------------------------------------------------
        # SCENARIO A: COMPARE ALL MODES SIDE-BY-SIDE (SINGLE GROUP)
        # -------------------------------------------------------------
        if "Compare All Modes" in chosen_mode_raw:
            if selected_scope == "Batch All Groups":
                target_group = active_groups[0]
                st.info(f"Comparing all modes on Group 1 ({target_group['frame_count']} frames).")
            else:
                gid = int(selected_scope.split()[1])
                target_group = next((g for g in active_groups if g["group_id"] == gid), active_groups[0])

            modes_list = ["A", "B", "C", "D", "E1", "E2", "F"]
            comp_rows = []
            res_dict = {}

            for m in modes_list:
                res_m = KeyframeSelectorL2_2.select_group_keyframes(
                    group_dict=target_group,
                    mode=m,
                    tau_coverage=tau_cov,
                    epsilon_info=eps_info,
                    w_spatial=w_spatial,
                    w_vit=w_vit,
                    w_event=w_event,
                    tau_event_ocr=tau_ev_ocr,
                    tau_event_ssim=tau_ev_ssim,
                    event_direction=ev_dir,
                    vit_embeddings=vit_dict,
                    cached_transitions_map=trans_lookup,
                    quality_map=quality_lookup
                )
                res_dict[m] = res_m
                comp_rows.append({
                    "Mode": m,
                    "Description": {
                        "A": "Unique Lexical Set",
                        "B": "Lexical Multiset",
                        "C": "Multiset + Spatial (16x16)",
                        "D": "Quality-Steered",
                        "E1": "D + Gated ViT Diversity",
                        "E2": "D + Gated ViT + Event Evidence",
                        "F": "Approach 7 (Set Selection / Facility Location)"
                    }[m],
                    "Selected Count": res_m["selected_count"],
                    "Input Count": res_m["input_frame_count"],
                    "Compression": f"{(1.0 - res_m['compression_ratio']) * 100:.1f}%",
                    "Final Coverage": f"{res_m['final_coverage'] * 100:.1f}%",
                    "Stopping Reason": res_m["stopping_reason"],
                    "Selected Keyframes": ", ".join(res_m["selected_filenames"])
                })

            st.markdown(f"##### Comparative Mode Ablation on Group {target_group['group_id']} ({target_group['frame_count']} frames)")
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

            # Visual comparison tabs across modes
            tab_names = [f"Mode {m}" for m in modes_list]
            mode_tabs = st.tabs(tab_names)
            for idx, m in enumerate(modes_list):
                with mode_tabs[idx]:
                    r_m = res_dict[m]
                    st.caption(f"**Mode {m}:** Selected **{r_m['selected_count']}** keyframes | Coverage: **{r_m['final_coverage']*100:.1f}%** | Reason: `{r_m['stopping_reason']}`")
                    render_group_frame_gallery(target_group, r_m["selected_filenames"], cand_img_dir, cols_per_row=col_layout, inspect_key=inspect_key)

        # -------------------------------------------------------------
        # SCENARIO B: BATCH ALL GROUPS (FULL SESSION VISUAL GALLERY)
        # -------------------------------------------------------------
        elif selected_scope == "Batch All Groups":
            print(f"[L2.2 DEBUG] Running Batch All Groups (Session: '{chosen_session}', Mode: '{mode_code}', Groups: {len(active_groups)})...", flush=True)
            t_batch_start = time.time()
            try:
                batch_results = KeyframeSelectorL2_2.run_all_groups(
                    groups=active_groups,
                    mode=mode_code,
                    tau_coverage=tau_cov,
                    epsilon_info=eps_info,
                    w_spatial=w_spatial,
                    w_vit=w_vit,
                    w_event=w_event,
                    tau_event_ocr=tau_ev_ocr,
                    tau_event_ssim=tau_ev_ssim,
                    event_direction=ev_dir,
                    vit_embeddings=vit_dict,
                    cached_transitions_map=trans_lookup,
                    quality_map=quality_lookup
                )
            except Exception as e:
                print(f"[L2.2 ERROR] Batch selection execution failed: {e}", flush=True)
                traceback.print_exc()
                st.error(f"Batch selection failed: {e}")
                return
            tot_input = sum(r["input_frame_count"] for r in batch_results)
            tot_selected = sum(r["selected_count"] for r in batch_results)
            print(f"[L2.2 DEBUG] Batch selection finished in {time.time() - t_batch_start:.3f}s. Selected: {tot_selected}/{tot_input} frames.", flush=True)

            avg_cov = np.mean([r["final_coverage"] for r in batch_results]) if batch_results else 0.0
            overall_comp = (1.0 - (tot_selected / max(tot_input, 1))) * 100

            bm1, bm2, bm3, bm4 = st.columns(4)
            with bm1:
                st.metric("Total Candidate Frames", tot_input)
            with bm2:
                st.metric("Total Selected Keyframes", tot_selected, delta=f"-{tot_input - tot_selected} pruned")
            with bm3:
                st.metric("Overall Compression", f"{overall_comp:.1f}%")
            with bm4:
                st.metric("Mean Pedagogical Coverage", f"{avg_cov * 100:.1f}%")

            # Collect all selected keyframes in chronological sequence across all groups
            all_chosen_keyframes = []
            for r in batch_results:
                for sf in r["selected_frames"]:
                    all_chosen_keyframes.append({
                        "group_id": r["group_id"],
                        "filename": sf["filename"],
                        "timestamp_str": sf.get("timestamp_str", ""),
                        "timestamp_sec": sf.get("timestamp_sec", 0),
                        "candidate_seq_idx": sf.get("candidate_seq_idx", 0)
                    })

            # Sort strictly chronologically
            all_chosen_keyframes.sort(key=lambda x: x["timestamp_sec"])

            st.markdown("<br/>", unsafe_allow_html=True)
            batch_view_mode = st.radio(
                "Batch Output Display Format",
                [
                    f"Visual Keyframe Gallery (All {len(all_chosen_keyframes)} Chosen Keyframes Chronologically)",
                    "Group-by-Group Visual Sequence Cards",
                    "Summary Data Table"
                ],
                horizontal=True,
                key="l2_2_batch_view_mode"
            )

            # View 1: Complete Chronological Keyframe Gallery
            if "Visual Keyframe Gallery" in batch_view_mode:
                st.markdown(f"##### Full Chronological Keyframe Sequence ({len(all_chosen_keyframes)} Culmination Frames)")
                st.caption("Displays every culmination keyframe selected across all groups in chronological sequence, representing the zero-information-loss video summary.")

                ROW_SIZE = col_layout
                for row_start in range(0, len(all_chosen_keyframes), ROW_SIZE):
                    chunk = all_chosen_keyframes[row_start : row_start + ROW_SIZE]
                    cols = st.columns(ROW_SIZE)
                    for k_idx, kf in enumerate(chunk):
                        global_idx = row_start + k_idx + 1
                        with cols[k_idx]:
                            fr_name = kf["filename"]
                            fr_path = os.path.join(cand_img_dir, fr_name)
                            b64 = get_image_b64_src(fr_path)

                            if b64:
                                st.markdown(
                                    f"<div style='border: 2px solid #10b981; border-radius: 8px; padding: 6px; background-color: rgba(16, 185, 129, 0.05); margin-bottom: 6px;'>"
                                    f"<div style='display:flex; justify-content:space-between; margin-bottom:4px;'>"
                                    f"<span style='background-color:#059669; color:#ffffff; font-size:0.65rem; font-weight:700; padding:1px 6px; border-radius:3px;'>KEY #{global_idx}</span>"
                                    f"<span style='background-color:#27272a; color:#a1a1aa; font-size:0.65rem; padding:1px 5px; border-radius:3px;'>Group {kf['group_id']}</span>"
                                    f"</div>"
                                    f"<img src='{b64}' style='width:100%; border-radius:6px; display:block;' loading='lazy'>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                st.caption(f"**{kf['timestamp_str']}** | `{fr_name}`")
                                if st.button("Open", key=f"btn_open_l22_k_{global_idx}_{fr_name}", use_container_width=True):
                                    st.session_state[inspect_key] = fr_name
                                    st.rerun()

            # View 2: Group-by-Group Visual Sequence Cards
            elif "Group-by-Group" in batch_view_mode:
                st.markdown("##### Group-by-Group Visual Partitioning")
                for r in batch_results:
                    gid = r["group_id"]
                    orig_group = next((g for g in active_groups if g["group_id"] == gid), None)
                    if not orig_group:
                        continue

                    with st.container():
                        st.markdown(
                            f"<div style='background-color:#141414; padding:8px 14px; border-radius:6px; border:1px solid #27272a; border-left:4px solid #38bdf8; margin-top:12px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>"
                            f"<span><b style='color:#38bdf8; font-size:1.0rem;'>Group {gid}</b> &nbsp;|&nbsp; "
                            f"<span>Span: <code>{orig_group['start_timestamp']}</code> -> <code>{orig_group['end_timestamp']}</code></span> &nbsp;|&nbsp; "
                            f"<span><b>{r['selected_count']} / {r['input_frame_count']}</b> keyframes ({(1.0 - r['compression_ratio'])*100:.1f}% compressed)</span></span>"
                            f"<span style='background-color:#27272a; padding:2px 8px; border-radius:10px; font-size:0.75rem;'>Coverage: <b>{r['final_coverage']*100:.1f}%</b> ({r['stopping_reason']})</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        render_group_frame_gallery(orig_group, r["selected_filenames"], cand_img_dir, cols_per_row=col_layout, inspect_key=inspect_key)

            # View 3: Summary Data Table
            else:
                batch_table = [
                    {
                        "Group ID": r["group_id"],
                        "Input Frames": r["input_frame_count"],
                        "Selected Count": r["selected_count"],
                        "Compression": f"{(1.0 - r['compression_ratio']) * 100:.1f}%",
                        "Coverage": f"{r['final_coverage'] * 100:.1f}%",
                        "Stopping Reason": r["stopping_reason"],
                        "Selected Keyframes": ", ".join(r["selected_filenames"]),
                        "Timestamps": ", ".join(r["selected_timestamps"])
                    }
                    for r in batch_results
                ]
                st.dataframe(pd.DataFrame(batch_table), use_container_width=True)

            # JSON export
            export_payload = {
                "session_name": chosen_session,
                "partition": chosen_part_label,
                "mode": mode_code,
                "parameters": {
                    "tau_coverage": tau_cov,
                    "epsilon_info": eps_info,
                    "w_spatial": w_spatial,
                    "w_vit": w_vit,
                    "w_event": w_event,
                    "tau_event_ocr": tau_ev_ocr,
                    "tau_event_ssim": tau_ev_ssim,
                    "event_direction": ev_dir,
                    "q_min": q_min
                },
                "summary": {
                    "total_candidate_frames": tot_input,
                    "total_selected_keyframes": tot_selected,
                    "compression_percent": round(overall_comp, 2),
                    "mean_pedagogical_coverage": round(float(avg_cov), 4)
                },
                "groups": batch_results
            }
            st.download_button(
                "📥 Download Keyframe Selection JSON (l2_2_selection.json)",
                data=json.dumps(export_payload, indent=2),
                file_name=f"{chosen_session}_l2_2_selection_{mode_code}.json",
                mime="application/json",
                use_container_width=True
            )

        # -------------------------------------------------------------
        # SCENARIO C: SINGLE GROUP DETAILED INSPECTION
        # -------------------------------------------------------------
        else:
            gid = int(selected_scope.split()[1])
            target_group = next((g for g in active_groups if g["group_id"] == gid), active_groups[0])

            print(f"[L2.2 DEBUG] Running single group selection on Group {target_group['group_id']} (Mode: '{mode_code}', Frames: {target_group['frame_count']})...", flush=True)
            t_single_start = time.time()
            try:
                res = KeyframeSelectorL2_2.select_group_keyframes(
                    group_dict=target_group,
                    mode=mode_code,
                    tau_coverage=tau_cov,
                    epsilon_info=eps_info,
                    w_spatial=w_spatial,
                    w_vit=w_vit,
                    w_event=w_event,
                    tau_event_ocr=tau_ev_ocr,
                    tau_event_ssim=tau_ev_ssim,
                    event_direction=ev_dir,
                    vit_embeddings=vit_dict,
                    cached_transitions_map=trans_lookup,
                    quality_map=quality_lookup
                )
            except Exception as e:
                print(f"[L2.2 ERROR] Single group selection failed: {e}", flush=True)
                traceback.print_exc()
                st.error(f"Group keyframe selection failed: {e}")
                return
            print(f"[L2.2 DEBUG] Single group selection finished in {time.time() - t_single_start:.3f}s. Selected: {res['selected_count']}/{res['input_frame_count']} keyframes.", flush=True)


            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric("Input Frames", res["input_frame_count"])
            with sm2:
                st.metric("Selected Keyframes", res["selected_count"], delta=f"-{res['input_frame_count'] - res['selected_count']} pruned")
            with sm3:
                st.metric("Compression Ratio", f"{(1.0 - res['compression_ratio']) * 100:.1f}%")
            with sm4:
                st.metric("Pedagogical Coverage", f"{res['final_coverage'] * 100:.1f}%", help=f"Lexical: {res['final_lexical_coverage']*100:.1f}%, Spatial: {res['final_spatial_coverage']*100:.1f}%")

            st.caption(f"**Stopping Reason:** `{res['stopping_reason']}` | **Mode:** `{res['mode']}`")

            st.markdown("##### Chronological Group Sequence (Culmination Keyframes Highlighted)")
            render_group_frame_gallery(target_group, res["selected_filenames"], cand_img_dir, cols_per_row=col_layout, inspect_key=inspect_key)

            if res.get("audit_trail"):
                with st.expander("🔍 Selection Decision Audit Trail (Component Scores)", expanded=True):
                    st.dataframe(pd.DataFrame(res["audit_trail"]), use_container_width=True)

def render_group_frame_gallery(group_dict: Dict, selected_filenames: List[str], cand_img_dir: str, cols_per_row: int = 4, inspect_key: Optional[str] = None):
    """
    Renders all frames in a group in order, highlighting selected keyframes with green borders and badges.
    """
    ROW_SIZE = cols_per_row
    all_frames = group_dict.get("frames", [])
    selected_names_set = set(selected_filenames)

    for row_start in range(0, len(all_frames), ROW_SIZE):
        chunk = all_frames[row_start : row_start + ROW_SIZE]
        cols = st.columns(ROW_SIZE)
        for f_idx, fr in enumerate(chunk):
            with cols[f_idx]:
                fr_name = fr["filename"]
                fr_path = os.path.join(cand_img_dir, fr_name)
                b64 = get_image_b64_src(fr_path)
                is_sel = fr_name in selected_names_set

                if is_sel:
                    sel_idx = selected_filenames.index(fr_name) + 1
                    border_style = "2px solid #10b981"
                    bg_color = "rgba(16, 185, 129, 0.05)"
                    badge_html = f"<div style='background-color:#059669; color:#ffffff; font-size:0.65rem; font-weight:700; padding:2px 6px; border-radius:4px; display:inline-block; margin-bottom:3px;'>KEYFRAME #{sel_idx}</div>"
                else:
                    border_style = "1px solid #27272a"
                    bg_color = "transparent"
                    badge_html = "<div style='color:#71717a; font-size:0.65rem; padding:2px 6px; display:inline-block; margin-bottom:3px;'>Pruned</div>"

                if b64:
                    st.markdown(
                        f"<div style='border: {border_style}; border-radius: 8px; padding: 6px; background-color: {bg_color}; margin-bottom: 6px;'>"
                        f"{badge_html}"
                        f"<img src='{b64}' style='width: 100%; border-radius: 6px; display: block;' loading='lazy'>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"**{fr.get('timestamp_str', '')}** | `{fr_name}`")
                    if inspect_key:
                        if st.button("Open", key=f"btn_open_l22_g_{group_dict.get('group_id', 0)}_{fr_name}_{row_start}_{f_idx}", use_container_width=True):
                            st.session_state[inspect_key] = fr_name
                            st.rerun()
