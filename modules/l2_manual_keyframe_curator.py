import os
import json
import io
import zipfile
import datetime
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from modules.image_cache import (
    get_cached_thumbnail_b64,
    trigger_background_session_prefetch,
    get_cache_stats
)
from modules.l2.feature_extractor import load_cached_features, parse_timestamp_from_filename
from modules.l2.quality_cache import load_or_compute_quality_cache, resolve_candidate_image_dir
from modules.l2.keyframe_selector import KeyframeSelectorL2_2
from modules.l2.diagnostic_comparator import KeyframeDiagnosticComparator

def get_image_base64(path: str, max_dim: Optional[int] = None) -> Optional[str]:
    """Retrieves thumbnail from fast in-memory cache or computes it immediately (defaults to full native resolution)."""
    b64 = get_cached_thumbnail_b64(path, max_dim=max_dim)
    if b64 and not b64.startswith("data:"):
        return f"data:image/jpeg;base64,{b64}"
    return b64

def get_ground_truth_csv_path(session_path: str) -> str:
    """Returns path to Level 2.2 ground truth CSV file."""
    l2_dir = os.path.join(session_path, "level2")
    os.makedirs(l2_dir, exist_ok=True)
    return os.path.join(l2_dir, "l2_2_ground_truth.csv")

def load_keyframe_ground_truth(session_path: str, partition_name: Optional[str] = None) -> Dict[str, Dict]:
    """
    Loads human annotations from sessions/<session>/level2/l2_2_ground_truth.csv.
    Returns: {filename: {"role_type": "final"|"state", "group_id": int, "timestamp_str": str, "timestamp_sec": float, "is_keyframe": 1}}
    """
    csv_path = get_ground_truth_csv_path(session_path)
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
        required_cols = {"frame_filename", "is_keyframe", "role_type"}
        if not required_cols.issubset(set(df.columns)):
            return {}

        gt_dict = {}
        for _, row in df.iterrows():
            fname = str(row["frame_filename"]).strip()
            is_kf = int(row.get("is_keyframe", 0))
            role = str(row.get("role_type", "")).strip().lower()

            if is_kf == 1 and role in ("final", "state"):
                part = str(row.get("partition_name", "")).strip() if "partition_name" in df.columns else ""
                # If partition filter specified, check match or generic
                if partition_name and part and part != partition_name:
                    continue

                gt_dict[fname] = {
                    "role_type": role,
                    "group_id": int(row.get("group_id", 1)) if pd.notnull(row.get("group_id")) else 1,
                    "timestamp_str": str(row.get("timestamp_str", "")) if pd.notnull(row.get("timestamp_str")) else "",
                    "timestamp_sec": float(row.get("timestamp_sec", 0.0)) if pd.notnull(row.get("timestamp_sec")) else 0.0,
                    "is_keyframe": 1,
                    "partition_name": part,
                    "annotated_at": str(row.get("annotated_at", ""))
                }
        return gt_dict
    except Exception as e:
        st.warning(f"Could not load existing ground truth CSV: {e}")
        return {}

def save_keyframe_ground_truth(
    session_path: str,
    session_name: str,
    partition_name: str,
    gt_dict: Dict[str, Dict]
) -> str:
    """
    Persists human annotations to sessions/<session>/level2/l2_2_ground_truth.csv.
    """
    csv_path = get_ground_truth_csv_path(session_path)
    rows = []
    for fname, rec in gt_dict.items():
        role = rec.get("role_type", "")
        is_kf = 1 if role in ("final", "state") else 0
        rows.append({
            "session_name": session_name,
            "partition_name": partition_name,
            "group_id": rec.get("group_id", 1),
            "frame_filename": fname,
            "timestamp_str": rec.get("timestamp_str", ""),
            "timestamp_sec": rec.get("timestamp_sec", 0.0),
            "is_keyframe": is_kf,
            "role_type": role if is_kf else "",
            "annotated_at": rec.get("annotated_at", datetime.datetime.now().isoformat())
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["group_id", "timestamp_sec", "frame_filename"])
    df.to_csv(csv_path, index=False)
    return csv_path

def create_curated_keyframe_zip(
    cand_dir: str,
    gt_dict: Dict[str, Dict],
    session_name: str,
    partition_name: str
) -> io.BytesIO:
    """Creates in-memory ZIP containing curated final & state keyframes with manifest and ground truth CSV."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        finals = [fn for fn, r in gt_dict.items() if r.get("role_type") == "final"]
        states = [fn for fn, r in gt_dict.items() if r.get("role_type") == "state"]

        for fn in finals:
            fpath = os.path.join(cand_dir, fn)
            if os.path.exists(fpath):
                zipf.write(fpath, arcname=f"keyframes_final/{fn}")

        for fn in states:
            fpath = os.path.join(cand_dir, fn)
            if os.path.exists(fpath):
                zipf.write(fpath, arcname=f"keyframes_state/{fn}")

        manifest = {
            "session_name": session_name,
            "partition_name": partition_name,
            "exported_at": datetime.datetime.now().isoformat(),
            "total_keyframes": len(finals) + len(states),
            "final_keyframes_count": len(finals),
            "state_keyframes_count": len(states),
            "final_keyframes": sorted(finals),
            "state_keyframes": sorted(states)
        }
        zipf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Include CSV in ZIP
        csv_rows = []
        for fn, r in gt_dict.items():
            csv_rows.append({
                "session_name": session_name,
                "partition_name": partition_name,
                "group_id": r.get("group_id", 1),
                "frame_filename": fn,
                "timestamp_str": r.get("timestamp_str", ""),
                "timestamp_sec": r.get("timestamp_sec", 0.0),
                "is_keyframe": 1,
                "role_type": r.get("role_type", ""),
                "annotated_at": r.get("annotated_at", "")
            })
        if csv_rows:
            csv_content = pd.DataFrame(csv_rows).to_csv(index=False)
            zipf.writestr("l2_2_ground_truth.csv", csv_content)

    buffer.seek(0)
    return buffer

def render_manual_keyframe_curator():
    """
    Renders the Level 2.2 Manual Keyframe Selector and Diagnostic Gap Studio.
    Provides fast single-click tagging for strictly two roles:
    1. 'final' : The culmination / terminal keyframe of the pedagogical group (exactly 1 per group).
    2. 'state' : Intermediate milestone keyframe(s) within the group (0 or more per group).
    """
    st.markdown("### Manual Keyframe Selector & Human Ground Truth Curator")
    st.write(
        "Human curation module for **Level 2.2 Keyframes**. Consumes temporal groups produced by **Level 2.1** "
        "and enables rapid single-click tagging into strictly two fundamental roles: **`🏆 Final`** (culmination keyframe) "
        "and **`📌 State`** (intermediate milestone). Downstream, compares human selections against algorithm ablations (A-E2)."
    )

    sessions_root = "sessions"
    if not os.path.exists(sessions_root) or not os.path.isdir(sessions_root):
        st.info("No active sessions found in `sessions/`.")
        return

    # Ingest sessions that have Level 2.1 grouping runs
    valid_sessions = []
    for d in sorted(os.listdir(sessions_root), reverse=True):
        full_d = os.path.join(sessions_root, d)
        if os.path.isdir(full_d):
            runs_p = os.path.join(full_d, "level2", "grouping_runs.json")
            if os.path.exists(runs_p):
                valid_sessions.append(d)

    if not valid_sessions:
        st.warning("No sessions with completed Level 2.1 grouping runs found. Please run Level 2 Grouping Lab first.")
        return

    # -------------------------------------------------------------
    # 1. SESSION & LEVEL 2.1 PARTITION SELECTION
    # -------------------------------------------------------------
    col_s1, col_s2, col_s3 = st.columns([1.5, 1.5, 1])
    with col_s1:
        chosen_session = st.selectbox("Select Session", valid_sessions, key="l2_2_curator_sess")

    session_path = os.path.join(sessions_root, chosen_session)
    runs_file = os.path.join(session_path, "level2", "grouping_runs.json")

    with open(runs_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    available_partitions = {}
    if "approach_a2" in run_data and "groups" in run_data["approach_a2"]:
        available_partitions["A2: Improved Asymmetric (Default)"] = run_data["approach_a2"]["groups"]
    if "approach_b2" in run_data and "groups" in run_data["approach_b2"]:
        available_partitions["B2: Multimodal (A2 + ViT)"] = run_data["approach_b2"]["groups"]
    if "approach_a1" in run_data and "groups" in run_data["approach_a1"]:
        available_partitions["A1: Frozen Baseline"] = run_data["approach_a1"]["groups"]

    if not available_partitions:
        st.error("No valid partitions found in grouping_runs.json.")
        return

    with col_s2:
        part_keys = list(available_partitions.keys())
        chosen_part_label = st.selectbox("Level 2.1 Group Partition", part_keys, index=0, key="l2_2_curator_part")
        active_groups = available_partitions[chosen_part_label]
        partition_short_id = chosen_part_label.split(":")[0].strip()

    try:
        cand_img_dir = resolve_candidate_image_dir(session_path)
    except FileNotFoundError:
        st.error(f"Could not locate candidate images directory in `{session_path}`.")
        return

    # Load feature cache & quality cache
    cached_data, cached_vit = load_cached_features(session_path)
    all_frames_features = {f["filename"]: f for f in cached_data.get("frames", [])} if cached_data else {}
    transitions_list = cached_data.get("transitions", []) if cached_data else []
    vit_dict = cached_vit or {}

    all_frame_names = list(all_frames_features.keys())
    q_cache_data = load_or_compute_quality_cache(session_path, all_frame_names, q_min=0.20)
    quality_lookup = {fn: rec["quality_score"] for fn, rec in q_cache_data.get("frames", {}).items()}

    with col_s3:
        st.metric("Partition Groups", len(active_groups))
        total_p_frames = sum(g.get("frame_count", len(g.get("frames", []))) for g in active_groups)
        st.caption(f"Frames in partition: **{total_p_frames}**")

    # -------------------------------------------------------------
    # 2. GROUND TRUTH STATE SYNCHRONIZATION
    # -------------------------------------------------------------
    gt_state_key = f"l2_2_gt_{chosen_session}_{partition_short_id}"
    if gt_state_key not in st.session_state:
        st.session_state[gt_state_key] = load_keyframe_ground_truth(session_path, chosen_part_label)

    gt_dict = st.session_state[gt_state_key]

    # Inspection key for lightbox
    inspect_key = f"l2_2_curator_inspect_{chosen_session}"
    if inspect_key not in st.session_state:
        st.session_state[inspect_key] = None
    selected_inspect = st.session_state[inspect_key]

    # Main Tabs: Curator, Diagnostics, Showcase
    tab_curator, tab_diagnostic, tab_showcase = st.tabs([
        "🎯 Manual Keyframe Curator",
        "⚖️ Human vs. Algorithm Diagnostic Gap Studio",
        "🖼️ Keyframe Showcase & Export"
    ])

    # =========================================================================
    # TAB 1: MANUAL KEYFRAME CURATOR
    # =========================================================================
    with tab_curator:
        # Full Preview / Lightbox Mode
        if selected_inspect:
            render_lightbox_view(
                inspect_fname=selected_inspect,
                cand_img_dir=cand_img_dir,
                session_path=session_path,
                session_name=chosen_session,
                partition_name=chosen_part_label,
                active_groups=active_groups,
                gt_dict=gt_dict,
                gt_state_key=gt_state_key,
                inspect_key=inspect_key,
                frames_features=all_frames_features,
                quality_lookup=quality_lookup
            )
            return

        # Top Summary Metrics Bar
        final_count = sum(1 for r in gt_dict.values() if r.get("role_type") == "final")
        state_count = sum(1 for r in gt_dict.values() if r.get("role_type") == "state")
        total_curated = final_count + state_count

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f'<div style="background-color:#18181b; border:1px solid #27272a; border-radius:8px; padding:10px; text-align:center;">'
                f'<div style="font-size:0.8rem; color:#a1a1aa;">Total Partition Frames</div>'
                f'<div style="font-size:1.4rem; font-weight:700; color:#f4f4f5;">{total_p_frames}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_m2:
            st.markdown(
                f'<div style="background-color:#18181b; border:1px solid #059669; border-radius:8px; padding:10px; text-align:center;">'
                f'<div style="font-size:0.8rem; color:#a1a1aa;">Culmination Keyframes</div>'
                f'<div style="font-size:1.4rem; font-weight:700; color:#10b981;">{final_count} <span style="font-size:0.85rem; font-weight:normal;">[🏆 Final]</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_m3:
            st.markdown(
                f'<div style="background-color:#18181b; border:1px solid #0891b2; border-radius:8px; padding:10px; text-align:center;">'
                f'<div style="font-size:0.8rem; color:#a1a1aa;">Intermediate Milestones</div>'
                f'<div style="font-size:1.4rem; font-weight:700; color:#06b6d4;">{state_count} <span style="font-size:0.85rem; font-weight:normal;">[📌 State]</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_m4:
            comp_rate = (1.0 - (total_curated / total_p_frames)) * 100 if total_p_frames > 0 else 0.0
            st.markdown(
                f'<div style="background-color:#18181b; border:1px solid #4f46e5; border-radius:8px; padding:10px; text-align:center;">'
                f'<div style="font-size:0.8rem; color:#a1a1aa;">Human Compression</div>'
                f'<div style="font-size:1.4rem; font-weight:700; color:#818cf8;">{comp_rate:.1f}% <span style="font-size:0.85rem; font-weight:normal;">({total_curated} kfs)</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.write("")

        # Group Scope Navigation
        col_g1, col_g2, col_g3, col_g4 = st.columns([1, 2.5, 1, 1.5])
        group_opts = [
            f"Group {g['group_id']} ({g['frame_count']} frames | {g['start_timestamp']} -> {g['end_timestamp']})"
            for g in active_groups
        ] + ["All Groups (Flat Gallery)"]

        scope_key = f"curator_scope_{chosen_session}_{partition_short_id}"
        sel_widget_key = f"scope_sel_{chosen_session}_{partition_short_id}"

        if scope_key not in st.session_state:
            st.session_state[scope_key] = 0

        curr_scope_idx = max(0, min(len(group_opts) - 1, st.session_state[scope_key]))
        st.session_state[scope_key] = curr_scope_idx

        if sel_widget_key not in st.session_state or st.session_state[sel_widget_key] not in group_opts:
            st.session_state[sel_widget_key] = group_opts[curr_scope_idx]

        def set_group_scope(new_idx: int):
            idx_clamped = max(0, min(len(group_opts) - 1, new_idx))
            st.session_state[scope_key] = idx_clamped
            st.session_state[sel_widget_key] = group_opts[idx_clamped]

        def on_scope_select_change():
            sel_val = st.session_state[sel_widget_key]
            if sel_val in group_opts:
                st.session_state[scope_key] = group_opts.index(sel_val)

        with col_g1:
            st.button(
                "◀ Prev Group",
                key="btn_prev_g_nav",
                disabled=(curr_scope_idx == 0),
                on_click=set_group_scope,
                args=(curr_scope_idx - 1,),
                use_container_width=True
            )

        with col_g2:
            st.selectbox(
                "Active Group Container",
                group_opts,
                key=sel_widget_key,
                on_change=on_scope_select_change
            )
            selected_scope_str = st.session_state[sel_widget_key]

        with col_g3:
            can_next_g = curr_scope_idx < (len(group_opts) - 1)
            st.button(
                "Next Group ▶",
                key="btn_next_g_nav",
                disabled=not can_next_g,
                on_click=set_group_scope,
                args=(curr_scope_idx + 1,),
                use_container_width=True
            )

        with col_g4:
            col_layout = st.radio("Columns per Row", [3, 4, 6], index=1, horizontal=True, key=f"curator_cols_{chosen_session}")

        # Filter target frames for gallery
        if "All Groups" in selected_scope_str:
            display_frames = []
            for g in active_groups:
                for fr in g.get("frames", []):
                    display_frames.append({**fr, "group_id": g["group_id"]})
        else:
            gid = active_groups[curr_scope_idx]["group_id"]
            active_g_obj = next((g for g in active_groups if g["group_id"] == gid), active_groups[0])
            display_frames = [{**fr, "group_id": gid} for fr in active_g_obj.get("frames", [])]

        if not display_frames:
            st.info("No frames found in selected group container.")
            return

        # Pagination logic
        page_size = 24
        total_items = len(display_frames)
        total_pages = max(1, (total_items + page_size - 1) // page_size)

        page_key = f"curator_page_{chosen_session}_{partition_short_id}_{curr_scope_idx}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1

        curr_page = st.session_state[page_key]
        if curr_page > total_pages:
            curr_page = total_pages
            st.session_state[page_key] = total_pages
        elif curr_page < 1:
            curr_page = 1
            st.session_state[page_key] = 1

        start_idx = (curr_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        page_frames = display_frames[start_idx:end_idx]

        # Trigger background prefetching for fast rendering
        full_fnames = [fr["filename"] for fr in display_frames]
        trigger_background_session_prefetch(
            [os.path.join(cand_img_dir, fn) for fn in full_fnames],
            current_page=curr_page,
            page_size=page_size,
            session_key=f"{cand_img_dir}_{curr_scope_idx}"
        )

        full_paths = [os.path.join(cand_img_dir, fn) for fn in full_fnames]
        cached_cnt, total_cnt = get_cache_stats(full_paths)
        if cached_cnt >= total_cnt:
            cache_info = f"<span style='color: #10b981; font-size: 0.8rem;'><b>{cached_cnt:,} / {total_cnt:,}</b> frames pre-warmed in RAM</span>"
        else:
            cache_info = f"<span style='color: #94a3b8; font-size: 0.8rem;'>RAM Cache: <b>{cached_cnt:,} / {total_cnt:,}</b> ready (prefetching in background...)</span>"

        # Pagination Toolbar
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        with col_p1:
            if st.button("Previous Page", key=f"btn_p_prev_{chosen_session}", disabled=(curr_page <= 1), use_container_width=True):
                st.session_state[page_key] -= 1
                st.rerun()
        with col_p2:
            st.markdown(
                f"<div style='text-align: center; padding-top: 4px;'>"
                f"<div style='font-size: 0.95rem;'>Page <b>{curr_page}</b> of <b>{total_pages}</b> &nbsp;|&nbsp; "
                f"Showing <b>{start_idx + 1}-{end_idx}</b> of <b>{total_items:,}</b> frames</div>"
                f"<div style='margin-top: 2px;'>{cache_info}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_p3:
            if st.button("Next Page", key=f"btn_p_next_{chosen_session}", disabled=(curr_page >= total_pages), use_container_width=True):
                st.session_state[page_key] += 1
                st.rerun()

        st.markdown("---")

        # -------------------------------------------------------------
        # 3. INTERACTIVE FRAME CARD GALLERY
        # -------------------------------------------------------------
        def toggle_keyframe_role(fname: str, target_role: str, gid: int, ts_str: str, ts_sec: float):
            rec = gt_dict.get(fname)
            if rec and rec.get("role_type") == target_role:
                # Toggle off (unmark)
                del gt_dict[fname]
            else:
                # Set target role
                gt_dict[fname] = {
                    "role_type": target_role,
                    "group_id": gid,
                    "timestamp_str": ts_str,
                    "timestamp_sec": ts_sec,
                    "is_keyframe": 1,
                    "partition_name": chosen_part_label,
                    "annotated_at": datetime.datetime.now().isoformat()
                }
            # Auto-save immediately to disk
            save_keyframe_ground_truth(session_path, chosen_session, chosen_part_label, gt_dict)

        cols_per_row = col_layout
        for r_start in range(0, len(page_frames), cols_per_row):
            chunk = page_frames[r_start : r_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for c_idx, fr in enumerate(chunk):
                with cols[c_idx]:
                    fname = fr["filename"]
                    fpath = os.path.join(cand_img_dir, fname)
                    img_b64 = get_image_base64(fpath)
                    
                    gt_rec = gt_dict.get(fname)
                    cur_role = gt_rec.get("role_type") if gt_rec else None

                    # Card border and badge styling
                    if cur_role == "final":
                        border_color = "#10b981"
                        bg_color = "rgba(16, 185, 129, 0.08)"
                        shadow = "0 0 14px rgba(16, 185, 129, 0.35)"
                        badge_html = "<div style='background-color:#059669; color:#ffffff; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:4px; display:inline-block; margin-bottom:4px;'>🏆 FINAL KEYFRAME</div>"
                    elif cur_role == "state":
                        border_color = "#06b6d4"
                        bg_color = "rgba(6, 182, 212, 0.08)"
                        shadow = "0 0 14px rgba(6, 182, 212, 0.35)"
                        badge_html = "<div style='background-color:#0891b2; color:#ffffff; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:4px; display:inline-block; margin-bottom:4px;'>📌 STATE KEYFRAME</div>"
                    else:
                        border_color = "#27272a"
                        bg_color = "transparent"
                        shadow = "none"
                        badge_html = "<div style='color:#71717a; font-size:0.7rem; padding:2px 8px; display:inline-block; margin-bottom:4px;'>Unmarked</div>"

                    st.markdown(
                        f'<div style="border: 2px solid {border_color}; border-radius: 10px; padding: 6px; background-color: {bg_color}; box-shadow: {shadow}; margin-bottom: 6px; transition: all 0.2s ease;">'
                        f'{badge_html}'
                        f'<img src="{img_b64}" style="width: 100%; border-radius: 6px; display: block;"/>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    q_val = quality_lookup.get(fname, 0.0)
                    st.caption(f"**{fr.get('timestamp_str', '')}** | `{fname}` | Q: `{q_val:.2f}`")

                    # Fast One-Click Action Buttons
                    b_col1, b_col2, b_col3 = st.columns([1, 1.2, 1.2])
                    with b_col1:
                        if st.button("Open", key=f"btn_open_{chosen_session}_{fname}", use_container_width=True):
                            st.session_state[inspect_key] = fname
                            st.rerun()

                    with b_col2:
                        final_btn_label = "✓ Final" if cur_role == "final" else "🏆 Final"
                        final_btn_type = "primary" if cur_role == "final" else "secondary"
                        st.button(
                            final_btn_label,
                            key=f"btn_final_{chosen_session}_{fname}",
                            type=final_btn_type,
                            on_click=toggle_keyframe_role,
                            args=(fname, "final", fr["group_id"], fr.get("timestamp_str", ""), fr.get("timestamp_sec", 0.0)),
                            use_container_width=True
                        )

                    with b_col3:
                        state_btn_label = "✓ State" if cur_role == "state" else "📌 State"
                        state_btn_type = "primary" if cur_role == "state" else "secondary"
                        st.button(
                            state_btn_label,
                            key=f"btn_state_{chosen_session}_{fname}",
                            type=state_btn_type,
                            on_click=toggle_keyframe_role,
                            args=(fname, "state", fr["group_id"], fr.get("timestamp_str", ""), fr.get("timestamp_sec", 0.0)),
                            use_container_width=True
                        )

        # Bottom Pagination Controls
        if total_pages > 1:
            st.write("")
            col_b_prev, col_b_sel, col_b_next = st.columns([1, 2, 1])
            with col_b_prev:
                if st.button("Prev Page", key=f"btn_b_prev_{chosen_session}", disabled=(curr_page <= 1), use_container_width=True):
                    st.session_state[page_key] -= 1
                    st.rerun()
            with col_b_sel:
                st.selectbox(
                    "Page:",
                    options=list(range(1, total_pages + 1)),
                    index=(curr_page - 1),
                    key=f"jump_p_{chosen_session}_{partition_short_id}_{curr_scope_idx}_{curr_page}",
                    on_change=lambda: st.session_state.update({page_key: st.session_state[f"jump_p_{chosen_session}_{partition_short_id}_{curr_scope_idx}_{curr_page}"]})
                )
            with col_b_next:
                if st.button("Next Page", key=f"btn_b_next_{chosen_session}", disabled=(curr_page >= total_pages), use_container_width=True):
                    st.session_state[page_key] += 1
                    st.rerun()

        # Bottom Group Navigation Bar (Quick Jump to Next/Previous Group)
        st.write("")
        st.markdown("<div style='border-top: 1px solid #27272a; margin-top: 14px; margin-bottom: 14px;'></div>", unsafe_allow_html=True)
        col_bot_g1, col_bot_info, col_bot_g2 = st.columns([1.5, 2.5, 1.5])
        with col_bot_g1:
            st.button(
                "◀ Prev Group",
                key=f"btn_bot_prev_g_{chosen_session}_{partition_short_id}",
                disabled=(curr_scope_idx == 0),
                on_click=set_group_scope,
                args=(curr_scope_idx - 1,),
                use_container_width=True
            )
        with col_bot_info:
            active_label = group_opts[curr_scope_idx].split('(')[0].strip()
            total_g_num = len(active_groups)
            st.markdown(
                f"<div style='text-align: center; padding-top: 6px; font-size: 0.95rem; color: #a1a1aa;'>"
                f"Currently viewing <b style='color: #10b981;'>{active_label}</b> ({curr_scope_idx + 1} of {total_g_num} groups)"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_bot_g2:
            can_bot_next = curr_scope_idx < (len(group_opts) - 1)
            next_g_name = f"Next Group (Group {curr_scope_idx + 2}) ▶" if curr_scope_idx < len(active_groups) - 1 else ("All Groups ▶" if curr_scope_idx == len(active_groups) - 1 else "Next Group ▶")
            st.button(
                next_g_name,
                key=f"btn_bot_next_g_{chosen_session}_{partition_short_id}",
                disabled=not can_bot_next,
                type="primary",
                on_click=set_group_scope,
                args=(curr_scope_idx + 1,),
                use_container_width=True
            )

    # =========================================================================
    # TAB 2: HUMAN VS ALGORITHM DIAGNOSTIC GAP STUDIO
    # =========================================================================
    with tab_diagnostic:
        render_diagnostic_gap_studio(
            session_path=session_path,
            session_name=chosen_session,
            partition_name=chosen_part_label,
            active_groups=active_groups,
            gt_dict=gt_dict,
            cand_img_dir=cand_img_dir,
            frames_features=all_frames_features,
            transitions_list=transitions_list,
            quality_lookup=quality_lookup,
            vit_dict=vit_dict
        )

    # =========================================================================
    # TAB 3: KEYFRAME SHOWCASE & EXPORT
    # =========================================================================
    with tab_showcase:
        render_keyframe_showcase_and_export(
            session_path=session_path,
            session_name=chosen_session,
            partition_name=chosen_part_label,
            gt_dict=gt_dict,
            cand_img_dir=cand_img_dir,
            inspect_key=inspect_key
        )

def render_lightbox_view(
    inspect_fname: str,
    cand_img_dir: str,
    session_path: str,
    session_name: str,
    partition_name: str,
    active_groups: List[Dict],
    gt_dict: Dict[str, Dict],
    gt_state_key: str,
    inspect_key: str,
    frames_features: Dict[str, Dict],
    quality_lookup: Dict[str, float]
):
    """
    Renders high-resolution full preview lightbox with keyboard hotkey controls.
    """
    fpath = os.path.join(cand_img_dir, inspect_fname)
    all_fnames = [fr["filename"] for g in active_groups for fr in g.get("frames", [])]
    if inspect_fname not in all_fnames:
        st.session_state[inspect_key] = None
        st.rerun()

    curr_idx = all_fnames.index(inspect_fname)
    prev_idx = (curr_idx - 1) % len(all_fnames)
    next_idx = (curr_idx + 1) % len(all_fnames)

    gt_rec = gt_dict.get(inspect_fname)
    cur_role = gt_rec.get("role_type") if gt_rec else None

    # Determine frame metadata
    feat = frames_features.get(inspect_fname, {})
    q_score = quality_lookup.get(inspect_fname, 0.0)
    gid = gt_rec.get("group_id") if gt_rec else 1
    ts_str = feat.get("timestamp_str", parse_timestamp_from_filename(inspect_fname))
    ts_sec = feat.get("timestamp_sec", 0.0)

    st.markdown("---")
    badge_label = f"🏆 FINAL KEYFRAME" if cur_role == "final" else (f"📌 STATE KEYFRAME" if cur_role == "state" else "UNMARKED")
    badge_color = "#10b981" if cur_role == "final" else ("#06b6d4" if cur_role == "state" else "#71717a")

    st.markdown(
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>"
        f"<h3 style='margin:0;'>Full Preview: <code>{inspect_fname}</code> &nbsp; "
        f"<span style='background-color:{badge_color}; color:#fff; font-size:0.75rem; padding:3px 10px; border-radius:6px; font-weight:700;'>{badge_label}</span></h3>"
        f"<span style='color:#a1a1aa;'>Frame {curr_idx + 1} of {len(all_fnames)} | Group {gid} | Timestamp: <b>{ts_str}</b> | Quality: <b>{q_score:.3f}</b></span>"
        f"</div>",
        unsafe_allow_html=True
    )

    if os.path.exists(fpath):
        st.image(Image.open(fpath), use_container_width=True)

    # Lightbox Action Controls
    st.write("")
    c_prev, c_final, c_state, c_unmark, c_next, c_close = st.columns(6)

    def set_lightbox_role(role_to_set: Optional[str]):
        if role_to_set is None:
            if inspect_fname in gt_dict:
                del gt_dict[inspect_fname]
        else:
            gt_dict[inspect_fname] = {
                "role_type": role_to_set,
                "group_id": gid,
                "timestamp_str": ts_str,
                "timestamp_sec": ts_sec,
                "is_keyframe": 1,
                "partition_name": partition_name,
                "annotated_at": datetime.datetime.now().isoformat()
            }
        save_keyframe_ground_truth(session_path, session_name, partition_name, gt_dict)

    with c_prev:
        if st.button("◀ Prev (←)", key="btn_lb_prev", use_container_width=True):
            st.session_state[inspect_key] = all_fnames[prev_idx]
            st.rerun()

    with c_final:
        final_type = "primary" if cur_role == "final" else "secondary"
        if st.button("🏆 Final (F)", key="btn_lb_final", type=final_type, use_container_width=True):
            set_lightbox_role("final" if cur_role != "final" else None)
            st.rerun()

    with c_state:
        state_type = "primary" if cur_role == "state" else "secondary"
        if st.button("📌 State (S)", key="btn_lb_state", type=state_type, use_container_width=True):
            set_lightbox_role("state" if cur_role != "state" else None)
            st.rerun()

    with c_unmark:
        if st.button("✖ Unmark (U)", key="btn_lb_unmark", use_container_width=True):
            set_lightbox_role(None)
            st.rerun()

    with c_next:
        if st.button("Next (→) ▶", key="btn_lb_next", use_container_width=True):
            st.session_state[inspect_key] = all_fnames[next_idx]
            st.rerun()

    with c_close:
        if st.button("✕ Close (Esc)", key="btn_lb_close", use_container_width=True):
            st.session_state[inspect_key] = None
            st.rerun()

    # Keyboard shortcut injection
    st.components.v1.html("""
    <script>
    function handleKeyDown(e) {
        const doc = window.parent.document;
        const buttons = Array.from(doc.querySelectorAll('button'));
        const key = e.key.toLowerCase();

        if (e.key === 'ArrowLeft') {
            const btn = buttons.find(b => b.textContent.includes('Prev'));
            if (btn) { btn.click(); e.preventDefault(); }
        } else if (e.key === 'ArrowRight') {
            const btn = buttons.find(b => b.textContent.includes('Next'));
            if (btn) { btn.click(); e.preventDefault(); }
        } else if (key === 'f') {
            const btn = buttons.find(b => b.textContent.includes('Final'));
            if (btn) { btn.click(); e.preventDefault(); }
        } else if (key === 's') {
            const btn = buttons.find(b => b.textContent.includes('State'));
            if (btn) { btn.click(); e.preventDefault(); }
        } else if (key === 'u') {
            const btn = buttons.find(b => b.textContent.includes('Unmark'));
            if (btn) { btn.click(); e.preventDefault(); }
        } else if (e.key === 'Escape') {
            const btn = buttons.find(b => b.textContent.includes('Close'));
            if (btn) { btn.click(); e.preventDefault(); }
        }
    }
    if (window.parent) {
        if (window.parent._l2CuratorHandler) {
            window.parent.removeEventListener('keydown', window.parent._l2CuratorHandler);
        }
        window.parent._l2CuratorHandler = handleKeyDown;
        window.parent.addEventListener('keydown', handleKeyDown);
    }
    </script>
    """, height=0, width=0)

def render_diagnostic_gap_studio(
    session_path: str,
    session_name: str,
    partition_name: str,
    active_groups: List[Dict],
    gt_dict: Dict[str, Dict],
    cand_img_dir: str,
    frames_features: Dict[str, Dict],
    transitions_list: List[Dict],
    quality_lookup: Dict[str, float],
    vit_dict: Dict[str, np.ndarray]
):
    """
    Renders the quantitative Human vs Algorithm Diagnostic Gap Studio.
    """
    st.markdown("#### Human vs. Algorithm Diagnostic Gap Analysis")
    st.write(
        "Directly quantifies the performance of Level 2.2 algorithms (Modes **A** through **E2**) against your "
        "human ground truth keyframes, highlighting exact **False Negatives** (keyframes the human prioritized but the algorithm pruned) "
        "and **False Positives** (spurious selections made by the algorithm)."
    )

    if not gt_dict:
        st.warning("No ground truth keyframes have been annotated yet. Please curate keyframes in Tab 1 first.")
        return

    # Execute all 6 modes across all groups in partition
    trans_lookup = {(t["frame_a"], t["frame_b"]): t for t in transitions_list}
    modes = ["A", "B", "C", "D", "E1", "E2"]

    with st.spinner("Computing algorithmic keyframe selections across all modes (A through E2)..."):
        algo_results_by_mode = {}
        for m in modes:
            m_dict = {}
            for g in active_groups:
                res_g = KeyframeSelectorL2_2.select_group_keyframes(
                    group_dict=g,
                    mode=m,
                    tau_coverage=0.95,
                    epsilon_info=0.02,
                    w_spatial=0.15,
                    w_vit=0.20,
                    w_event=0.20,
                    vit_embeddings=vit_dict,
                    cached_transitions_map=trans_lookup,
                    quality_map=quality_lookup
                )
                m_dict[g["group_id"]] = res_g
            algo_results_by_mode[m] = m_dict

    # Comparison Table across Modes
    st.markdown("##### 1. Benchmark Metrics Summary across Ablation Formulations")
    summary_df = KeyframeDiagnosticComparator.compare_all_modes(
        gt_dict=gt_dict,
        algo_runs_by_mode=algo_results_by_mode,
        frames_features=frames_features,
        quality_map=quality_lookup
    )
    st.dataframe(summary_df, use_container_width=True)

    # Detailed Mode Inspector
    st.markdown("---")
    st.markdown("##### 2. Deep-Dive False Negative & False Positive Diagnostics")
    c_dm1, c_dm2 = st.columns([1.5, 2.5])
    with c_dm1:
        chosen_inspect_mode = st.selectbox(
            "Select Ablation Mode for Gap Diagnosis",
            [
                "D: Multiset + Spatial + Quality (Recommended Default)",
                "A: Unique Lexical Set",
                "B: Lexical Multiset",
                "C: Multiset + Spatial (16x16 Grid)",
                "E1: D + Gated ViT Diversity",
                "E2: D + Gated ViT + Event Evidence"
            ],
            index=0,
            key="diag_mode_inspect_sel"
        )
        mode_code = chosen_inspect_mode.split(":")[0].strip()

    m_group_res = algo_results_by_mode[mode_code]
    algo_selected_map = {gid: res["selected_filenames"] for gid, res in m_group_res.items()}
    algo_audit_map = {gid: res.get("audit_trail", []) for gid, res in m_group_res.items()}

    eval_detail = KeyframeDiagnosticComparator.evaluate_mode(
        gt_dict=gt_dict,
        algo_selected_map=algo_selected_map,
        algo_audit_map=algo_audit_map,
        frames_features=frames_features,
        quality_map=quality_lookup
    )
    metrics = eval_detail["metrics"]

    with c_dm2:
        st.write("")
        c_p, c_r, c_f1, c_rf, c_rs = st.columns(5)
        c_p.metric("Precision", f"{metrics['precision']*100:.1f}%")
        c_r.metric("Recall", f"{metrics['recall']*100:.1f}%")
        c_f1.metric("F1-Score", f"{metrics['f1']*100:.1f}%")
        c_rf.metric("Final Recall", f"{metrics['final_recall']*100:.1f}%")
        c_rs.metric("State Recall", f"{metrics['state_recall']*100:.1f}%")

    col_gap1, col_gap2 = st.columns(2)
    with col_gap1:
        st.markdown(f"###### ❌ False Negatives ({len(eval_detail['fn_diagnostics'])} missed by Algorithm)")
        if eval_detail["fn_diagnostics"]:
            st.dataframe(pd.DataFrame(eval_detail["fn_diagnostics"]), use_container_width=True)
        else:
            st.success("Zero False Negatives! Algorithm captured all human keyframes.")

    with col_gap2:
        st.markdown(f"###### ⚠️ False Positives ({len(eval_detail['fp_diagnostics'])} spurious selections)")
        if eval_detail["fp_diagnostics"]:
            st.dataframe(pd.DataFrame(eval_detail["fp_diagnostics"]), use_container_width=True)
        else:
            st.success("Zero False Positives! Algorithm made no superfluous selections.")

    # Side-by-side Visual Comparison for a Chosen Group
    st.markdown("---")
    st.markdown("##### 3. Visual Group-Level Alignment Gallery")
    group_choices = [f"Group {g['group_id']} ({g['frame_count']} frames)" for g in active_groups]
    sel_g_str = st.selectbox("Select Group to Visually Compare", group_choices, index=0, key="diag_vis_g_sel")
    target_gid = int(sel_g_str.split()[1])
    target_group = next((g for g in active_groups if g["group_id"] == target_gid), active_groups[0])

    col_h, col_a = st.columns(2)
    with col_h:
        st.markdown(f"###### 👤 Human Ground Truth (Group {target_gid})")
        human_group_kfs = [
            fn for fn, r in gt_dict.items()
            if r.get("group_id") == target_gid and r.get("role_type") in ("final", "state")
        ]
        if not human_group_kfs:
            st.info("No human keyframes tagged in this group.")
        else:
            for kf in human_group_kfs:
                r_type = gt_dict[kf].get("role_type", "")
                badge = "🏆 FINAL" if r_type == "final" else "📌 STATE"
                b_color = "#10b981" if r_type == "final" else "#06b6d4"
                fpath = os.path.join(cand_img_dir, kf)
                b64 = get_image_base64(fpath)
                st.markdown(
                    f"<div style='border:2px solid {b_color}; border-radius:8px; padding:6px; margin-bottom:8px;'>"
                    f"<div style='color:{b_color}; font-size:0.75rem; font-weight:700;'>{badge} — {kf}</div>"
                    f"<img src='{b64}' style='width:100%; border-radius:6px; display:block;'/>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    with col_a:
        st.markdown(f"###### 🤖 Mode {mode_code} Algorithmic Selection (Group {target_gid})")
        algo_group_kfs = m_group_res[target_gid]["selected_filenames"]
        if not algo_group_kfs:
            st.info("No keyframes selected by algorithm.")
        else:
            for kf in algo_group_kfs:
                is_hit = kf in human_group_kfs
                hit_badge = "✅ MATCHED HUMAN" if is_hit else "⚠️ SPURIOUS (FP)"
                b_color = "#10b981" if is_hit else "#ef4444"
                fpath = os.path.join(cand_img_dir, kf)
                b64 = get_image_base64(fpath)
                st.markdown(
                    f"<div style='border:2px solid {b_color}; border-radius:8px; padding:6px; margin-bottom:8px;'>"
                    f"<div style='color:{b_color}; font-size:0.75rem; font-weight:700;'>{hit_badge} — {kf}</div>"
                    f"<img src='{b64}' style='width:100%; border-radius:6px; display:block;'/>"
                    f"</div>",
                    unsafe_allow_html=True
                )

def render_keyframe_showcase_and_export(
    session_path: str,
    session_name: str,
    partition_name: str,
    gt_dict: Dict[str, Dict],
    cand_img_dir: str,
    inspect_key: str
):
    """
    Renders curated final & state keyframe showcase and export download utilities.
    """
    st.markdown("#### Curated Keyframes Showcase & Dataset Export")

    finals = sorted([fn for fn, r in gt_dict.items() if r.get("role_type") == "final"])
    states = sorted([fn for fn, r in gt_dict.items() if r.get("role_type") == "state"])
    total = len(finals) + len(states)

    if total == 0:
        st.info("No keyframes marked yet. Please tag frames in Tab 1 first.")
        return

    # Export toolbar
    c_ex1, c_ex2 = st.columns(2)
    with c_ex1:
        # Download Ground Truth CSV
        csv_path = get_ground_truth_csv_path(session_path)
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_bytes = f.read().encode("utf-8")
            st.download_button(
                label=f"📥 Download Ground Truth CSV ({total} annotations)",
                data=csv_bytes,
                file_name=f"{session_name}_l2_2_ground_truth.csv",
                mime="text/csv",
                use_container_width=True
            )

    with c_ex2:
        zip_key = f"l2_2_zip_{session_name}"
        if st.button(f"📦 Prepare Curated Keyframe ZIP ({total} frames)", use_container_width=True, key="btn_prep_l2_2_zip"):
            with st.spinner("Compressing curated keyframes archive..."):
                zip_buf = create_curated_keyframe_zip(cand_img_dir, gt_dict, session_name, partition_name)
                st.session_state[zip_key] = zip_buf.getvalue()
            st.rerun()

        if zip_key in st.session_state:
            st.download_button(
                label=f"⬇️ Download Curated ZIP ({total} frames)",
                data=st.session_state[zip_key],
                file_name=f"{session_name}_level2_2_curated.zip",
                mime="application/zip",
                use_container_width=True
            )

    st.markdown("---")

    tab_all, sub_f, sub_s = st.tabs([
        f"🖼️ All Keyframes ({total})",
        f"🏆 Final Keyframes ({len(finals)})",
        f"📌 State Keyframes ({len(states)})"
    ])

    with tab_all:
        all_kfs = sorted(
            list(gt_dict.keys()),
            key=lambda fn: (gt_dict[fn].get("group_id", 0), gt_dict[fn].get("timestamp_sec", 0.0), fn)
        )
        cols_per_row = 4
        for r in range(0, len(all_kfs), cols_per_row):
            chunk = all_kfs[r : r + cols_per_row]
            cols = st.columns(cols_per_row)
            for i, fn in enumerate(chunk):
                with cols[i]:
                    r_type = gt_dict[fn].get("role_type", "")
                    is_final = (r_type == "final")
                    badge_label = "🏆 FINAL" if is_final else "📌 STATE"
                    b_color = "#10b981" if is_final else "#06b6d4"
                    bg_color = "rgba(16, 185, 129, 0.08)" if is_final else "rgba(6, 182, 212, 0.08)"

                    fpath = os.path.join(cand_img_dir, fn)
                    b64 = get_image_base64(fpath)
                    st.markdown(
                        f"<div style='border:2px solid {b_color}; border-radius:8px; padding:6px; background-color:{bg_color}; margin-bottom:6px;'>"
                        f"<div style='color:{b_color}; font-size:0.7rem; font-weight:700;'>{badge_label}</div>"
                        f"<img src='{b64}' style='width:100%; border-radius:6px; display:block;'/>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"`{fn}` | Group {gt_dict[fn].get('group_id')} | {gt_dict[fn].get('timestamp_str', '')}")

    with sub_f:
        if not finals:
            st.info("No final keyframes tagged.")
        else:
            cols_per_row = 4
            for r in range(0, len(finals), cols_per_row):
                chunk = finals[r : r + cols_per_row]
                cols = st.columns(cols_per_row)
                for i, fn in enumerate(chunk):
                    with cols[i]:
                        fpath = os.path.join(cand_img_dir, fn)
                        b64 = get_image_base64(fpath)
                        st.markdown(
                            f"<div style='border:2px solid #10b981; border-radius:8px; padding:6px; margin-bottom:6px;'>"
                            f"<div style='color:#10b981; font-size:0.7rem; font-weight:700;'>🏆 FINAL</div>"
                            f"<img src='{b64}' style='width:100%; border-radius:6px; display:block;'/>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"`{fn}` | Group {gt_dict[fn].get('group_id')} | {gt_dict[fn].get('timestamp_str', '')}")

    with sub_s:
        if not states:
            st.info("No state keyframes tagged.")
        else:
            cols_per_row = 4
            for r in range(0, len(states), cols_per_row):
                chunk = states[r : r + cols_per_row]
                cols = st.columns(cols_per_row)
                for i, fn in enumerate(chunk):
                    with cols[i]:
                        fpath = os.path.join(cand_img_dir, fn)
                        b64 = get_image_base64(fpath)
                        st.markdown(
                            f"<div style='border:2px solid #06b6d4; border-radius:8px; padding:6px; margin-bottom:6px;'>"
                            f"<div style='color:#06b6d4; font-size:0.7rem; font-weight:700;'>📌 STATE</div>"
                            f"<img src='{b64}' style='width:100%; border-radius:6px; display:block;'/>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"`{fn}` | Group {gt_dict[fn].get('group_id')} | {gt_dict[fn].get('timestamp_str', '')}")
