"""
Trials Analysis Web App (Streamlit)
-----------------------------------
- Reads multiple Excel files + sheets
- Preserves original column names (TQ, TC, NDVI, etc.)
- Computes stats (mean, quartiles, std dev, whiskers, outliers)
- Optional ANOVA
- Generates:
    • Per-sheet boxplots + stats
    • Per-treatment boxplots across all dates for each metric
- Exports: Stats_Summary.xlsx, Boxplots.pdf, Individual_Plots.zip
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
import seaborn as sns


# ----------------------------
# Header row detection
# ----------------------------
def detect_header_row(xls_path_or_buf, sheet_name, max_check_rows: int = 25) -> int:
    for r in range(0, max_check_rows):
        try:
            sample = pd.read_excel(xls_path_or_buf, sheet_name=sheet_name, header=r, nrows=12)
        except Exception:
            continue
        if sample.empty:
            continue
        sample = sample.dropna(axis=1, how="all")
        if sample.shape[1] < 3:
            continue
        data_part = sample.iloc[0:8]
        numeric_counts = data_part.apply(
            lambda row: pd.to_numeric(row, errors="coerce").notna().sum(), axis=1
        )
        if (numeric_counts >= 3).sum() >= 3:
            first_col = pd.to_numeric(sample.iloc[:, 0], errors="coerce")
            if first_col.notna().sum() >= 3:
                return r
    return 9


# ----------------------------
# Stats computation
# ----------------------------
def compute_group_stats(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_col, dropna=True)[metric]
    for treatment, values in grouped:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) == 0:
            continue
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        vals_in_lower = values[values >= (q1 - 1.5 * iqr)]
        vals_in_upper = values[values <= (q3 + 1.5 * iqr)]
        lower_whisker = vals_in_lower.min() if not vals_in_lower.empty else np.nan
        upper_whisker = vals_in_upper.max() if not vals_in_upper.empty else np.nan

        outliers = values[(values < (q1 - 1.5 * iqr)) | (values > (q3 + 1.5 * iqr))]

        rows.append(
            {
                "Treatment": treatment,
                "N": int(values.size),
                "Mean": float(values.mean()),
                "Median": float(values.median()),
                "Q1": float(q1),
                "Q3": float(q3),
                "Std Dev": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "Min": float(values.min()),
                "Max": float(values.max()),
                "Lower Whisker": float(lower_whisker) if not np.isnan(lower_whisker) else None,
                "Upper Whisker": float(upper_whisker) if not np.isnan(upper_whisker) else None,
                "Outliers": int(outliers.size),
            }
        )
    return pd.DataFrame(rows)


def try_anova(df: pd.DataFrame, group_col: str, metric: str) -> Optional[float]:
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
def plot_treatment_across_dates(all_data: pd.DataFrame, treatment: str, metric: str, title_prefix="Trial Results"):
    subset = all_data[(all_data["Treatment"] == treatment) & (all_data["Metric"] == metric)]
    if subset.empty:
        return None

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=subset,
        x="Date",
        y="Value",
        color="skyblue"
    )
    plt.title(f"{title_prefix} – {metric} – {treatment}")
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.xlabel("Date")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    buf.seek(0)
    return buf


# ----------------------------
# Process a single sheet
# ----------------------------
def process_sheet(xls_path_or_buf, sheet_name: str, opts: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if opts.get("override_header") is not None:
        header_row = opts["override_header"]
    else:
        header_row = detect_header_row(xls_path_or_buf, sheet_name)

    df = pd.read_excel(xls_path_or_buf, sheet_name=sheet_name, header=header_row)

    if "Treatment" not in df.columns:
        group_col = df.columns[2]  # fallback
    else:
        group_col = "Treatment"

    metrics = []
    for c in df.columns:
        if c == group_col:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= 3:
            metrics.append(c)

    stats_rows = []
    long_data_rows = []

    for metric in metrics:
        try:
            stats_df = compute_group_stats(df, group_col, metric)
            if not stats_df.empty:
                stats_df.insert(0, "Metric", metric)
                stats_df.insert(0, "Sheet", sheet_name)
                stats_rows.append(stats_df)

            if opts.get("run_anova", False):
                p = try_anova(df, group_col, metric)
                if p is not None and not stats_df.empty:
                    stats_rows[-1]["ANOVA p-value"] = p

            for _, row in df.iterrows():
                val = pd.to_numeric(row[metric], errors="coerce")
                if not np.isnan(val):
                    long_data_rows.append({
                        "File": opts.get("file_label", ""),
                        "Date": sheet_name,
                        "Treatment": row[group_col],
                        "Metric": metric,
                        "Value": val,
                    })
        except Exception as e:
            st.warning(f"  Skipped metric {metric} in sheet {sheet_name}: {e}")
            continue

    all_stats = pd.concat(stats_rows, ignore_index=True) if stats_rows else pd.DataFrame()
    long_data = pd.DataFrame(long_data_rows)
    return all_stats, long_data


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Trials Analysis Tool", layout="wide")
    st.title("📊 Trials Analysis Tool")
    st.write("Upload Excel files; the app will analyze every sheet and produce stats + boxplots.")

    with st.sidebar:
        st.header("Options")
        run_anova = st.checkbox("Include ANOVA", value=True)
        export_stats = st.checkbox("Export Stats to Excel", value=True)
        combined_mode = st.checkbox("Per-treatment plots across dates", value=True)
        title_prefix = st.text_input("Plot title prefix", value="Trial Results")
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
            all_long_data = []

            for upl in uploaded_files:
                file_label = upl.name
                st.write(f"**Reading:** {file_label}")
                try:
                    xls = pd.ExcelFile(upl)
                except Exception as e:
                    st.error(f"Failed to read {file_label}: {e}")
                    continue

                for sheet_name in xls.sheet_names:
                    st.write(f"↳ Processing sheet: `{sheet_name}` …")
                    try:
                        stats_df, long_data = process_sheet(
                            upl,
                            sheet_name,
                            {
                                "run_anova": run_anova,
                                "title_prefix": title_prefix,
                                "override_header": override_header_val,
                                "file_label": file_label,
                            },
                        )
                    except Exception as e:
                        st.warning(f"  Skipped sheet `{sheet_name}` due to error: {e}")
                        continue

                    if not stats_df.empty:
                        stats_df.insert(0, "File", file_label)
                        all_stats_frames.append(stats_df)
                    if not long_data.empty:
                        all_long_data.append(long_data)

            stats_xlsx_bytes = None
            if export_stats and all_stats_frames:
                combined_stats = pd.concat(all_stats_frames, ignore_index=True)
                out_xlsx = io.BytesIO()
                with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
                    combined_stats.to_excel(writer, index=False, sheet_name="Summary")
                out_xlsx.seek(0)
                stats_xlsx_bytes = out_xlsx.read()

            if combined_mode and all_long_data:
                st.subheader("📊 Per-Treatment Plots Across Dates")
                all_data = pd.concat(all_long_data, ignore_index=True)
                for metric in all_data["Metric"].unique():
                    st.markdown(f"### Metric: {metric}")
                    for treatment in all_data["Treatment"].dropna().unique():
                        buf = plot_treatment_across_dates(all_data, treatment, metric, title_prefix=title_prefix)
                        if buf is not None:
                            st.image(buf, caption=f"{treatment} – {metric}")

        st.success("Analysis complete.")
        if export_stats and stats_xlsx_bytes is not None:
            st.download_button(
                label="📊 Download Stats_Summary.xlsx",
                data=stats_xlsx_bytes,
                file_name=f"Stats_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
