"""
Trials Analysis Web App (Streamlit)
-----------------------------------
Upload one or more Excel files containing assessment sheets (each sheet = a date).
For each sheet, the app will:
  • Auto-detect the header/data start row (heuristic) and standardize columns
  • Identify the grouping column (Treatment) and all numeric metrics
  • Compute summary statistics (N, Mean, Median, Q1, Q3, Std Dev, Min, Max, Whiskers, Outliers)
  • (Optional) Run one-way ANOVA across treatments for each metric
  • Generate box-and-whisker plots per metric grouped by Treatment
  • Export: Stats_Summary.xlsx, Boxplots.pdf, Individual_Plots.zip

Quick start (local):
  pip install streamlit pandas numpy matplotlib scipy openpyxl xlsxwriter
  streamlit run app.py

Notes:
  • This script is robust to a few leading metadata rows above the data table.
  • It uses a heuristic to find the header row; you can override defaults in the sidebar.
"""

import io
import os
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# ----------------------------
# Utility: Header row detection
# ----------------------------

def detect_header_row(xls_path_or_buf, sheet_name, max_check_rows: int = 25) -> int:
    """Heuristically detect the header row index (0-based) where the actual table starts.

    Strategy:
      1) Try each candidate row r in [0, max_check_rows):
         - Read a small sample with header=r
         - If first two columns look like integers/IDs and we have at least 4-5 numeric columns
           in the first 8 data rows, accept r.
      2) Fallback to 9 (common in the provided example) if nothing matches.
    """
    for r in range(0, max_check_rows):
        try:
            sample = pd.read_excel(xls_path_or_buf, sheet_name=sheet_name, header=r, nrows=12)
        except Exception:
            continue
        if sample.empty:
            continue
        # Drop fully empty columns
        sample = sample.dropna(axis=1, how="all")
        if sample.shape[1] < 3:
            continue
        # Check that first row is header-like (strings or mixed) and next rows contain numbers
        # Heuristic: first two data rows should have at least 3 numeric columns
        data_part = sample.iloc[0:8]
        numeric_counts = data_part.apply(lambda row: pd.to_numeric(row, errors="coerce").notna().sum(), axis=1)
        if (numeric_counts >= 3).sum() >= 3:
            # Additional hint: left-most column often looks like small integers (rep/block)
            first_col = pd.to_numeric(sample.iloc[:, 0], errors="coerce")
            if first_col.notna().sum() >= 3:
                return r
    return 9  # sensible default for this workbook

# ----------------------------
# Utility: Column standardization
# ----------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to a consistent schema where possible.

    We expect something akin to:
      Rep, Plot, Treatment, Score1, Score2, Percent, Count, Group, Index
    We won't force exact count; we map by position when headers are messy/Unnamed.
    """
    df = df.copy()
    # Drop fully empty rows/columns
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # If column names look like 'Unnamed: x' or metadata spillover, replace with positional labels
    cleaned_cols = []
    default_labels = [
        "Rep", "Plot", "Treatment", "Score1", "Score2", "Percent", "Count", "Group", "Index"
    ]
    for i, col in enumerate(df.columns):
        name = str(col)
        if name.startswith("Unnamed") or re.search(r"Trial Code|Trial Name|Date|Area|Assessor|STRI", name, re.I):
            cleaned_cols.append(default_labels[i] if i < len(default_labels) else f"Col{i+1}")
        else:
            # Keep human-friendly names, but normalize spaces
            cleaned_cols.append(re.sub(r"\s+", " ", name).strip())
    df.columns = cleaned_cols

    return df

# ----------------------------
# Stats computation
# ----------------------------

def compute_group_stats(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    """Compute summary stats per group for a given metric."""
    rows = []
    grouped = df.groupby(group_col, dropna=True)[metric]
    for treatment, values in grouped:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) == 0:
            continue
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_whisker = values[values >= (q1 - 1.5 * iqr)].min()
        upper_whisker = values[values <= (q3 + 1.5 * iqr)].max()
        outliers = values[(values < (q1 - 1.5 * iqr)) | (values > (q3 + 1.5 * iqr))]
        rows.append({
            "Treatment": treatment,
            "N": int(values.size),
            "Mean": float(values.mean()),
            "Median": float(values.median()),
            "Q1": float(q1),
            "Q3": float(q3),
            "Std Dev": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "Min": float(values.min()),
            "Max": float(values.max()),
            "Lower Whisker": float(lower_whisker),
            "Upper Whisker": float(upper_whisker),
            "Outliers": int(outliers.size),
        })
    return pd.DataFrame(rows)


def try_anova(df: pd.DataFrame, group_col: str, metric: str) -> Optional[float]:
    """Run one-way ANOVA across groups if there are >=2 groups with >=2 observations each.
       Returns p-value, or None if not applicable."""
    groups = []
    for _, g in df.groupby(group_col):
        vals = pd.to_numeric(g[metric], errors="coerce").dropna()
        if vals.size >= 2:
            groups.append(vals.values)
    if len(groups) >= 2:
        try:
            f, p = stats.f_oneway(*groups)
            return float(p)
        except Exception:
            return None
    return None

# ----------------------------
# Plotting
# ----------------------------

def make_boxplot(df: pd.DataFrame, group_col: str, metric: str, title: str, ylabel: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    # Use pandas' built-in boxplot grouped by group_col
    df.boxplot(column=metric, by=group_col)
    plt.title(title)
    plt.suptitle("")
    plt.xlabel(group_col)
    plt.ylabel(ylabel or metric)
    plt.tight_layout()

# ----------------------------
# Processing a single sheet
# ----------------------------

def process_sheet(xls_path_or_buf, sheet_name: str, opts: Dict) -> Tuple[pd.DataFrame, List[Tuple[str, io.BytesIO]]]:
    """Return (stats_df, list_of_plot_images[(metric, png_bytes)])."""
    # Detect header row
    if opts.get("override_header") is not None:
        header_row = opts["override_header"]
    else:
        header_row = detect_header_row(xls_path_or_buf, sheet_name)

    df = pd.read_excel(xls_path_or_buf, sheet_name=sheet_name, header=header_row)
    df = standardize_columns(df)

    # Choose grouping column
    group_col = opts.get("group_col") or ("Treatment" if "Treatment" in df.columns else df.columns[2])

    # Select numeric metrics (exclude the group col and obviously non-numeric columns)
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    # Include columns that are numeric-like but object due to mixed types
    for c in df.columns:
        if c not in numeric_candidates and c != group_col:
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().sum() >= 3:
                numeric_candidates.append(c)

    metrics = [c for c in numeric_candidates if c != group_col]

    # Stats & plots
    stats_rows = []
    plot_images: List[Tuple[str, io.BytesIO]] = []

    for metric in metrics:
        # Build per-group stats
        stats_df = compute_group_stats(df, group_col, metric)
        if not stats_df.empty:
            stats_df.insert(0, "Metric", metric)
            stats_df.insert(0, "Sheet", sheet_name)
            stats_rows.append(stats_df)

        # Plot
        title = f"{opts.get('title_prefix', 'Trial Results')} – {sheet_name} – {metric}"
        make_boxplot(df, group_col, metric, title, ylabel=metric)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=160)
        plt.close()
        buf.seek(0)
        plot_images.append((metric, buf))

        # Optional ANOVA
        if opts.get("run_anova", False):
            p = try_anova(df, group_col, metric)
            if p is not None and not stats_df.empty:
                stats_rows[-1]["ANOVA p-value"] = p

    all_stats = pd.concat(stats_rows, ignore_index=True) if stats_rows else pd.DataFrame()
    return all_stats, plot_images

# ----------------------------
# Streamlit UI
# ----------------------------

def main():
    st.set_page_config(page_title="Trials Analysis Tool", layout="wide")
    st.title("📊 Trials Analysis Tool")
    st.write("Upload Excel files; the app will analyze every sheet and produce stats + boxplots for each numeric metric.")

    with st.sidebar:
        st.header("Options")
        run_anova = st.checkbox("Include ANOVA", value=True)
        save_pdf = st.checkbox("Create Boxplots.pdf", value=True)
        save_pngs = st.checkbox("Create PNGs (zipped)", value=True)
        export_stats = st.checkbox("Export Stats to Excel", value=True)
        title_prefix = st.text_input("Plot title prefix", value="Dollar Spot Curative")
        override_header = st.text_input("Override header row (optional)", value="")
        override_header_val = int(override_header) if override_header.strip().isdigit() else None

    uploaded_files = st.file_uploader(
        "Drag & drop Excel files (.xlsx) or click to browse",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more Excel files to begin.")
        return

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Processing…"):
            all_stats_frames = []

            # Prepare collectors for outputs
            pdf_bytes = io.BytesIO()
            pdf = PdfPages(pdf_bytes) if save_pdf else None
            zip_buf = io.BytesIO()
            png_zip = zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) if save_pngs else None

            # Process each uploaded file
            for upl in uploaded_files:
                file_label = upl.name
                st.write(f"**Reading:** {file_label}")
                try:
                    xls = pd.ExcelFile(upl)
                except Exception as e:
                    st.error(f"Failed to read {file_label}: {e}")
                    continue

                # Iterate sheets
                for sheet_name in xls.sheet_names:
                    st.write(f"↳ Processing sheet: `{sheet_name}` …")
                    try:
                        stats_df, plot_images = process_sheet(
                            upl,
                            sheet_name,
                            {
                                "run_anova": run_anova,
                                "title_prefix": title_prefix,
                                "override_header": override_header_val,
                            },
                        )
                    except Exception as e:
                        st.warning(f"  Skipped sheet `{sheet_name}` due to error: {e}")
                        continue

                    # Collect stats
                    if not stats_df.empty:
                        stats_df.insert(0, "File", file_label)
                        all_stats_frames.append(stats_df)

                    # Write plots
                    for metric, png_bytes in plot_images:
                        # To PDF
                        if pdf is not None:
                            img = plt.imread(png_bytes)
                            plt.figure(figsize=(8, 6))
                            plt.imshow(img)
                            plt.axis('off')
                            pdf.savefig()  # saves the current figure to the PDF
                            plt.close()
                        # To ZIP (PNGs)
                        if png_zip is not None:
                            arcname = f"{os.path.splitext(file_label)[0]}/{sheet_name}/{metric}.png"
                            png_zip.writestr(arcname, png_bytes.getvalue())

            # Close outputs
            if pdf is not None:
                pdf.close()
                pdf_bytes.seek(0)
            if png_zip is not None:
                png_zip.close()
                zip_buf.seek(0)

            # Combine & export stats
            stats_xlsx_bytes = None
            if export_stats and all_stats_frames:
                combined_stats = pd.concat(all_stats_frames, ignore_index=True)
                out_xlsx = io.BytesIO()
                with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
                    combined_stats.to_excel(writer, index=False, sheet_name="Summary")
                out_xlsx.seek(0)
                stats_xlsx_bytes = out_xlsx.read()

        # Show download buttons
        st.success("Analysis complete.")
        cols = st.columns(3)
        if export_stats and stats_xlsx_bytes is not None:
            cols[0].download_button(
                label="📊 Download Stats_Summary.xlsx",
                data=stats_xlsx_bytes,
                file_name=f"Stats_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        if save_pdf and pdf_bytes.getbuffer().nbytes > 0:
            cols[1].download_button(
                label="📈 Download Boxplots.pdf",
                data=pdf_bytes.getvalue(),
                file_name=f"Boxplots_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
        if save_pngs and zip_buf.getbuffer().nbytes > 0:
            cols[2].download_button(
                label="🖼️ Download Individual_Plots.zip",
                data=zip_buf.getvalue(),
                file_name=f"Individual_Plots_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
