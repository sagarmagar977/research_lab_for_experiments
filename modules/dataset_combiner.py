import os
import io
import datetime
import pandas as pd
import streamlit as st

def scan_existing_datasets():
    """Discovers all available CSV datasets in the workspace."""
    found_files = []
    
    # 1. generated_datasets
    gen_root = "generated_datasets"
    if os.path.exists(gen_root):
        for root, _, files in os.walk(gen_root):
            for f in files:
                if f.lower().endswith(".csv"):
                    fpath = os.path.join(root, f)
                    found_files.append(fpath)
                    
    # 2. mixed_dataset
    mixed_root = "mixed_dataset"
    if os.path.exists(mixed_root):
        for root, _, files in os.walk(mixed_root):
            for f in files:
                if f.lower().endswith(".csv"):
                    fpath = os.path.join(root, f)
                    found_files.append(fpath)
                    
    # 3. Root directory CSVs
    for f in os.listdir("."):
        if os.path.isfile(f) and f.lower().endswith(".csv"):
            found_files.append(f)
            
    # Deduplicate and sort by modification time descending
    unique_paths = list(set(os.path.normpath(p) for p in found_files))
    unique_paths.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return unique_paths

def format_dataset_label(path):
    """Formats dataset path for display with timestamp and file size."""
    try:
        mtime = os.path.getmtime(path)
        dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        size_kb = os.path.getsize(path) / 1024.0
        if size_kb > 1024:
            size_str = f"{size_kb / 1024.0:.1f} MB"
        else:
            size_str = f"{size_kb:.1f} KB"
        return f"{os.path.basename(path)} ({size_str}, {dt_str}) — [{os.path.dirname(path)}]"
    except Exception:
        return path

def render_dataset_combiner():
    """Main rendering entrypoint for Dataset Combiner & Multi-Session Merger module."""
    st.markdown("### 🔀 Dataset Combiner & Multi-Session Merger")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;'>"
        "Select and merge multiple pairwise transition datasets into a single unified training corpus. "
        "Inspect schema alignment, check class distributions, remove duplicates, and manage combined dataset archives."
        "</p>",
        unsafe_allow_html=True
    )

    # Output directory configuration
    default_output_dir = os.path.normpath("mixed_dataset/mixed")
    os.makedirs(default_output_dir, exist_ok=True)

    tab_combine, tab_history = st.tabs(["⚡ Merge & Combine Datasets", "📂 Combined Datasets History (Latest First)"])

    # =========================================================================
    # TAB 1: MERGE & COMBINE DATASETS
    # =========================================================================
    with tab_combine:
        st.markdown("#### 1. Select or Upload Datasets to Combine")
        
        col_src_local, col_src_upload = st.columns(2)
        
        with col_src_local:
            st.markdown("**Option A: Select from Local Lab Datasets**")
            available_datasets = scan_existing_datasets()
            selected_local_paths = st.multiselect(
                "Choose datasets from project sessions",
                available_datasets,
                format_func=format_dataset_label,
                key="comb_selected_local_paths",
                help="Select 2 or more datasets from generated_datasets or mixed_dataset folders."
            )
            
        with col_src_upload:
            st.markdown("**Option B: Upload Custom CSV Files**")
            uploaded_csv_files = st.file_uploader(
                "Drag & drop external CSV datasets",
                type=["csv"],
                accept_multiple_files=True,
                key="comb_uploaded_csv_files",
                help="Upload one or multiple CSV files to merge."
            )

        # Ingest selected datasets into a unified list of DataFrames
        datasets_to_merge = []
        
        # Load local selected files
        for p in selected_local_paths:
            try:
                df = pd.read_csv(p)
                datasets_to_merge.append({
                    "name": os.path.basename(p),
                    "source": p,
                    "df": df
                })
            except Exception as e:
                st.error(f"Error loading `{p}`: {e}")
                
        # Load uploaded files
        if uploaded_csv_files:
            for uf in uploaded_csv_files:
                try:
                    df = pd.read_csv(uf)
                    datasets_to_merge.append({
                        "name": uf.name,
                        "source": f"Upload: {uf.name}",
                        "df": df
                    })
                except Exception as e:
                    st.error(f"Error reading uploaded file `{uf.name}`: {e}")

        total_selected = len(datasets_to_merge)
        if total_selected < 2:
            st.info(f"💡 Currently {total_selected} dataset(s) selected. Please select or upload at least **2 datasets** to begin merging.")
            st.markdown("---")
            return

        st.success(f"Selected **{total_selected} datasets** ready for inspection and merging.")

        # =====================================================================
        # 2. PRE-MERGE SCHEMA & CLASS DISTRIBUTION INSPECTION
        # =====================================================================
        st.markdown("#### 2. Pre-Merge Inspection & Feature Compatibility")
        
        inspect_cols = st.columns(min(total_selected, 4))
        for idx, item in enumerate(datasets_to_merge):
            c_idx = idx % min(total_selected, 4)
            df = item["df"]
            rows = len(df)
            cols = len(df.columns)
            
            # Check GroundTruth / label column
            gt_col = None
            for candidate in ["GroundTruth", "label", "prediction", "target"]:
                if candidate in df.columns:
                    gt_col = candidate
                    break
                    
            with inspect_cols[c_idx]:
                with st.container(border=True):
                    st.markdown(f"**📄 {item['name']}**")
                    st.markdown(f"**Rows:** `{rows:,}` | **Cols:** `{cols}`")
                    if gt_col:
                        counts = df[gt_col].value_counts()
                        k_count = counts.get(1, 0)
                        d_count = counts.get(0, 0)
                        k_ratio = (k_count / rows * 100.0) if rows > 0 else 0
                        st.markdown(f"• **Keep (1):** `{k_count:,}` ({k_ratio:.1f}%)")
                        st.markdown(f"• **Discard (0):** `{d_count:,}` ({100.0 - k_ratio:.1f}%)")
                    else:
                        st.caption("No GroundTruth / label column found.")

        # Column schema compatibility check
        col_sets = [set(item["df"].columns) for item in datasets_to_merge]
        common_cols = set.intersection(*col_sets)
        all_cols = set.union(*col_sets)
        all_matched = (len(common_cols) == len(all_cols))
        
        if all_matched:
            st.markdown(
                f"<div style='background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; "
                f"border-radius: 8px; padding: 10px; margin-top: 10px; color: #10b981; font-weight: 600;'>"
                f"✅ Perfect Schema Alignment: All {len(common_cols)} columns match identically across all selected datasets."
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            diff_cols = all_cols - common_cols
            st.warning(
                f"⚠️ Schema Discrepancy: Datasets have {len(common_cols)} common columns and "
                f"{len(diff_cols)} divergent columns."
            )
            with st.expander("🔍 View Divergent Columns Details"):
                st.write("**Common Columns:**", sorted(list(common_cols)))
                st.write("**Divergent Columns:**", sorted(list(diff_cols)))

        st.markdown("---")

        # =====================================================================
        # 3. MERGE & OUTPUT SETTINGS
        # =====================================================================
        st.markdown("#### 3. Merge & Output Configuration")
        
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            schema_mode = st.radio(
                "Feature Schema Alignment Strategy",
                ["Intersection (Common Columns Only — Recommended)", "Union (All Columns — Zero-Fill Missing)"],
                index=0,
                help="Intersection keeps only features present in all datasets, ensuring maximum model training compatibility."
            )
            drop_dups = st.checkbox(
                "Remove Duplicate Rows (`drop_duplicates()`)",
                value=True,
                help="Automatically drops duplicate transition vectors across merged datasets (matches notebook behavior)."
            )
            tag_source = st.checkbox(
                "Add 'Source_Dataset' Column",
                value=False,
                help="Appends a column tagging the origin dataset for each row."
            )
            
        with cfg_col2:
            custom_out_dir = st.text_input("Output Directory", value=default_output_dir)
            auto_name = f"combined_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            custom_filename = st.text_input("Combined Dataset Filename", value=auto_name)
            
        st.markdown("")
        
        # Action button to trigger merge
        if st.button("🚀 Merge & Save Combined Dataset", type="primary", use_container_width=True):
            with st.spinner("Merging datasets and standardizing features..."):
                processed_dfs = []
                for item in datasets_to_merge:
                    curr_df = item["df"].copy()
                    if schema_mode.startswith("Intersection"):
                        curr_df = curr_df[list(common_cols)]
                    if tag_source:
                        curr_df["Source_Dataset"] = item["name"]
                    processed_dfs.append(curr_df)
                    
                combined_df = pd.concat(processed_dfs, axis=0, ignore_index=True)
                initial_count = len(combined_df)
                
                if drop_dups:
                    # Drop duplicates excluding Source_Dataset if tagged
                    subset_cols = [c for c in combined_df.columns if c != "Source_Dataset"]
                    combined_df = combined_df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
                    
                final_count = len(combined_df)
                dups_removed = initial_count - final_count
                
                # Re-index Pair_Index if present
                if "Pair_Index" in combined_df.columns:
                    combined_df["Pair_Index"] = range(1, final_count + 1)

                # Save file
                os.makedirs(custom_out_dir, exist_ok=True)
                target_path = os.path.join(custom_out_dir, custom_filename)
                combined_df.to_csv(target_path, index=False)
                
                st.success(
                    f"🎉 Successfully created **{custom_filename}**! "
                    f"Merged `{initial_count:,}` rows into `{final_count:,}` rows "
                    f"({dups_removed:,} duplicates removed). Saved to `{target_path}`."
                )
                
                # Store in session state for instant preview & download
                st.session_state["last_combined_df"] = combined_df
                st.session_state["last_combined_path"] = target_path
                st.session_state["last_combined_name"] = custom_filename

        # Display result preview and download button if available
        if "last_combined_df" in st.session_state and "last_combined_name" in st.session_state:
            res_df = st.session_state["last_combined_df"]
            res_name = st.session_state["last_combined_name"]
            
            st.markdown("##### Combined Dataset Preview")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Rows", f"{len(res_df):,}")
            with col_m2:
                st.metric("Feature Columns", f"{len(res_df.columns)}")
            with col_m3:
                gt_col = next((c for c in ["GroundTruth", "label", "prediction"] if c in res_df.columns), None)
                if gt_col:
                    k_ratio = (res_df[gt_col] == 1).mean() * 100.0
                    st.metric("Keep Ratio (Class 1)", f"{k_ratio:.1f}%")
                else:
                    st.metric("Keep Ratio", "N/A")

            # CSV Download Button
            csv_buf = io.StringIO()
            res_df.to_csv(csv_buf, index=False)
            st.download_button(
                label=f"📥 Download {res_name}",
                data=csv_buf.getvalue(),
                file_name=res_name,
                mime="text/csv",
                use_container_width=True
            )
            
            st.dataframe(res_df.head(50), use_container_width=True)

    # =========================================================================
    # TAB 2: COMBINED DATASETS HISTORY (SORTED BY DATE/TIME DESCENDING)
    # =========================================================================
    with tab_history:
        st.markdown("#### 📂 Combined Datasets Archive")
        st.caption(
            "Lists all merged and combined datasets stored in `mixed_dataset/mixed/`. "
            "Sorted chronologically with the **latest / newest at the top** and oldest at the bottom."
        )

        history_dir = default_output_dir
        if not os.path.exists(history_dir):
            st.info("No combined datasets found yet. Create one in the first tab!")
            return

        all_files = [
            f for f in os.listdir(history_dir)
            if os.path.isfile(os.path.join(history_dir, f)) and f.lower().endswith(".csv")
        ]

        if not all_files:
            st.info("No combined dataset CSV files found in directory.")
            return

        # Sort by modification time in descending order (newest / latest at top)
        all_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(history_dir, f)),
            reverse=True
        )

        st.markdown(f"Found **{len(all_files)} combined datasets** in `{history_dir}`:")

        for fname in all_files:
            fpath = os.path.join(history_dir, fname)
            mtime = os.path.getmtime(fpath)
            dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_kb = os.path.getsize(fpath) / 1024.0
            size_str = f"{size_kb / 1024.0:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"

            with st.container(border=True):
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"##### 📊 `{fname}`")
                    st.markdown(f"**Modified:** `{dt_str}` &nbsp;|&nbsp; **Size:** `{size_str}` &nbsp;|&nbsp; **Path:** `{fpath}`")
                    
                    # Read sample summary
                    try:
                        sample_df = pd.read_csv(fpath, nrows=5000)
                        total_rows = len(sample_df)
                        total_cols = len(sample_df.columns)
                        gt_col = next((c for c in ["GroundTruth", "label", "prediction"] if c in sample_df.columns), None)
                        
                        badge_html = f"<span style='color: #a78bfa; font-size: 0.85rem;'>Rows: <b>{total_rows:,}</b> | Cols: <b>{total_cols}</b></span>"
                        if gt_col:
                            c1 = (sample_df[gt_col] == 1).sum()
                            c0 = (sample_df[gt_col] == 0).sum()
                            pct = (c1 / total_rows * 100.0) if total_rows > 0 else 0
                            badge_html += f" &nbsp;|&nbsp; <span style='color: #10b981; font-size: 0.85rem;'>Keep (1): <b>{c1:,}</b> ({pct:.1f}%)</span> &nbsp;|&nbsp; <span style='color: #ef4444; font-size: 0.85rem;'>Discard (0): <b>{c0:,}</b></span>"
                            
                        st.markdown(badge_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.caption(f"Could not read metadata: {e}")

                with col_actions:
                    # Download button
                    try:
                        with open(fpath, "rb") as f_data:
                            st.download_button(
                                label="📥 Download",
                                data=f_data.read(),
                                file_name=fname,
                                mime="text/csv",
                                key=f"dl_hist_{fname}",
                                use_container_width=True
                            )
                    except Exception:
                        pass
                        
                    # Preview button
                    if st.button("👁️ Preview", key=f"prev_hist_{fname}", use_container_width=True):
                        st.session_state[f"show_preview_{fname}"] = not st.session_state.get(f"show_preview_{fname}", False)
                        st.rerun()

                # Expandable preview
                if st.session_state.get(f"show_preview_{fname}", False):
                    try:
                        full_df = pd.read_csv(fpath)
                        st.markdown(f"**Previewing `{fname}` (first 50 rows):**")
                        st.dataframe(full_df.head(50), use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to preview `{fname}`: {e}")
