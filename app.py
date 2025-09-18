"""
Trials Analysis Web App (Streamlit)
-----------------------------------
- Reads multiple Excel files + sheets
- Cleans messy headers / non-numeric data
- Computes stats (mean, quartiles, std dev, whiskers, outliers)
- Optional ANOVA
- Generates boxplots (per sheet + combined across dates)
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
# Column standardization
# ----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cleaned_cols = []
    default_labels = [
        "Rep",
        "Plot",
        "Treatment",
        "Score1",
        "Score2",
        "Percent",
        "Count",
        "Group",
        "Index",
    ]
    for i, col in enumerate(df.columns):
        name = str(col)
        if name.startswith("Unnamed") or re.search(
            r"Trial Code|Trial Name|Date|Area|Assessor|STRI", name, re.I
        ):
            cleaned_cols.append(default_labels[i] if i < len(default_labels) else f"Col{i+1}")
        else:
            cleaned_cols.append(re.sub(r"\s+", " ", name).strip())
    df.columns = cleaned_cols
    return df


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

        # ✅ Safe whisker handling
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
def make_boxplot(df: pd.DataFrame, group_col: str, metric: str, title: str, ylabel: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    safe_values = pd.to_numeric(df[metric], errors="coerce")
    safe_df = pd.DataFrame({group_col: df[group_col], metric: safe_values})
    safe_df = safe_df.dropna()
    if safe_df.empty:
        plt.text(0.5, 0.5, f"No numeric data for {metric}", ha="center")
    else:
        safe_df.boxplot(column=metric, by=group_col)
        plt.title(title)
        plt.suptitle("")
        plt.xlabel(group_col)
        plt.ylabel(ylabel or metric)
    plt.tight_layout()


def plot_across_dates(all_data: pd.DataFrame, metric: str, title_prefix: str = "Trial Results"):
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=all_data,
        x="Date",
        y="Value",   # ✅ Always use Value as Y
        hue="Treatment",
        palette="Set2"
    )
    plt.title(f"{title_prefix} – Box & Whisker by Treatment across Dates – {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    buf.seek(0)
    return buf


# ----------------------------
# Process a single sheet
# ----------------------------
def process_sheet(xls_path_or_buf, sheet_name: str, opts: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, io.BytesIO]]]:
    if opts.get("override_header") is not None:
        header_row = opts["override_header"]
    else:
        header_row = detect_header_row(xls_path_or_buf, sheet_name)

    df = pd.read_excel(xls_path_or_buf, sheet_name=sheet_name, header=header_row)
    df = standardize_columns(df)

    group_col = opts.get("group_col") or ("Treatment" if "Treatment" in df.columns else df.columns[2])

    metrics = []
    for c in df.columns:
        if c == group_col:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= 3:
            metrics.append(c)

    stats_rows = []
    plot_images: List[Tuple[str, io.BytesIO]] = []
    long_data_rows = []

    for metric in metrics:
        try:
            stats_df = compute_group_stats(df, group_col, metric)
            if not stats_df.empty:
                stats_df.insert(0, "Metric", metric)
                stats_df.insert(0, "Sheet", sheet_name)
                stats_rows.append(stats_df)

            title = f"{opts.get('title_prefix', 'Trial Results')} – {sheet_name} – {metric}"
            make_boxplot(df, group_col, metric, title, ylabel=metric)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=160)
            plt.close()
            buf.seek(0)
            plot_images.append((metric, buf))

            if opts.get("run_anova", False):
                p = try_anova(df, group_col, metric)
                if p is not None and not stats_df.empty:
                    stats_rows[-1]["ANOVA p-value"] = p

            # Collect raw long-form data for combined plots
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
    return all_stats, long_data, plot_images


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
        combine_across_dates = st.checkbox("Generate combined plots across dates", value=True)
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
            pdf_bytes = io.BytesIO()
            pdf = PdfPages(pdf_bytes) if save_pdf else None
            zip_buf = io.BytesIO()
            png_zip = zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) if save_pngs else None

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
                        stats_df, long_data, plot_images = process_sheet(
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

                    for metric, png_bytes in plot_images:
                        if pdf is not None:
                            img = plt.imread(png_bytes)
                            plt.figure(figsize=(8, 6))
                            plt.imshow(img)
                            plt.axis("off")
                            pdf.savefig()
                            plt.close()
                        if png_zip is not None:
                            arcname = f"{os.path.splitext(file_label)[0]}/{sheet_name}/{metric}.png"
                            png_zip.writestr(arcname, png_bytes.getvalue())

            if pdf is not None:
                pdf.close()
                pdf_bytes.seek(0)
            if png_zip is not None:
                png_zip.close()
                zip_buf.seek(0)

            stats_xlsx_bytes = None
            if export_stats and all_stats_frames:
                combined_stats = pd.concat(all_stats_frames, ignore_index=True)
                out_xlsx = io.BytesIO()
                with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
                    combined_stats.to_excel(writer, index=False, sheet_name="Summary")
                out_xlsx.seek(0)
                stats_xlsx_bytes = out_xlsx.read()

            # ----------------------------
            # Combined across dates plots
            # ----------------------------
            if combine_across_dates and all_long_data:
                st.subheader("📊 Combined Plots Across Dates")
                all_data = pd.concat(all_long_data, ignore_index=True)
                for metric in all_data["Metric"].unique():
                    subset = all_data[all_data["Metric"] == metric]
                    if subset.empty:
                        continue
                    try:
                        buf = plot_across_dates(subset, metric, title_prefix=title_prefix)
                        st.image(buf, caption=f"Combined plot for {metric}")
                    except Exception as e:
                        st.warning(f"Could not plot combined view for {metric}: {e}")

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
