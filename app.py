"""
Trials Analysis Web App (Streamlit)
-----------------------------------
- Reads multiple Excel files + sheets
- Preserves original metric names (TQ, TC, NDVI, etc.)
- Computes stats (mean, quartiles, std dev, whiskers, outliers)
- Optional ANOVA
- Generates:
    • One boxplot per metric (each showing all treatments across all dates)
- Exports: Stats_Summary.xlsx
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats


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
            return r
    return 9


# ----------------------------
# Stats computation
# ----------------------------
def compute_group_stats(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    rows = []
    if group_col not in df.columns or metric not in df.columns:
        return pd.DataFrame()

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


def try_anova(df: pd.DataFrame, group_col: str, metric: str):
    if group_col not in df.columns or metric not in df.columns:
        return None
    groups = []
    for _, g in df.groupby(group_col):
        vals = pd.to_numeric(g[metric], errors="coerce").dropna()
        if vals.size >= 2:
            groups.append(vals.values)
    if len(groups) >= 2:
        try:
            _, p = stats.f_oneway(*groups)
            return float(p)
        except Exception:
            return None
    return None


# ----------------------------
# Plotting
# ----------------------------
def plot_metric_across_dates(df: pd.DataFrame, metric: str, colors: dict, title_prefix="Trial Results"):
    # Ensure required columns exist
    if "Date" not in df.columns or "Treatment" not in df.columns or metric not in df.columns:
        return None

    subset = df[["Date", "Treatment", metric]].dropna()
    if subset.empty:
        return None
    subset["Treatment"] = subset["Treatment"].astype(str)

    plt.figure(figsize=(14, 6))

    # Loop over treatments
    for i, treatment in enumerate(sorted(subset["Treatment"].unique()), start=1):
        t_data = subset[subset["Treatment"] == treatment]
        data = [t_data.loc[t_data["Date"] == d, metric].values
                for d in sorted(t_data["Date"].unique())]
        if len(data) == 0:
            continue
        plt.boxplot(
            data,
            positions=[j + (i-1)*(len(data)+1) for j in range(len(data))],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors.get(treatment, "lightgray"))
        )

    plt.title(f"{title_prefix} – {metric}")
    plt.xlabel("Dates (grouped per Treatment)")
    plt.ylabel(metric)
    plt.xticks([])  # hide crowded labels
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    buf.seek(0)
    return buf


# ----------------------------
# Process a single sheet
# ----------------------------
def process_sheet(xls_path_or_buf, sheet_name: str, opts: dict):
    header_row = detect_header_row(xls_path_or_buf, sheet_name)
    df = pd.read_excel(xls_path_or_buf, sheet_name=sheet_name, header=header_row)

    # Expect columns roughly: Date, ..., Treatment, ..., metric(s)
    if "Treatment" not in df.columns:
        group_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
    else:
        group_col = "Treatment"

    # If no Date column, use sheet name as Date
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(sheet_name, errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    metrics = []
    for c in df.columns:
        if c in [group_col, "Date"]:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= 3:
            metrics.append(c)

    stats_rows = []
    for metric in metrics:
        stats_df = compute_group_stats(df, group_col, metric)
        if not stats_df.empty:
            stats_df.insert(0, "Metric", metric)
            stats_df.insert(0, "Sheet", sheet_name)
            stats_rows.append(stats_df)
            if opts.get("run_anova", False):
                p = try_anova(df, group_col, metric)
                if p is not None:
                    stats_rows[-1]["ANOVA p-value"] = p

    all_stats = pd.concat(stats_rows, ignore_index=True) if stats_rows else pd.DataFrame()
    return df, all_stats, metrics


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Trials Analysis Tool", layout="wide")
    st.title("📊 Trials Analysis Tool")
    st.write("Upload Excel files; the app will analyze each metric and produce one boxplot per metric.")

    with st.sidebar:
        st.header("Options")
        run_anova = st.checkbox("Include ANOVA", value=True)
        export_stats = st.checkbox("Export Stats to Excel", value=True)
        title_prefix = st.text_input("Plot title prefix", value="Trial Results")

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
            all_data_frames = []
            metrics_found = set()

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
                        df, stats_df, metrics = process_sheet(
                            upl,
                            sheet_name,
                            {
                                "run_anova": run_anova,
                            },
                        )
                    except Exception as e:
                        st.warning(f"  Skipped sheet `{sheet_name}` due to error: {e}")
                        continue

                    if not stats_df.empty:
                        stats_df.insert(0, "File", file_label)
                        all_stats_frames.append(stats_df)
                    if not df.empty:
                        all_data_frames.append(df)
                    metrics_found.update(metrics)

            # Merge all sheets/files
            if all_data_frames:
                all_data = pd.concat(all_data_frames, ignore_index=True)
            else:
                all_data = pd.DataFrame()

            # Export stats
            stats_xlsx_bytes = None
            if export_stats and all_stats_frames:
                combined_stats = pd.concat(all_stats_frames, ignore_index=True)
                out_xlsx = io.BytesIO()
                with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
                    combined_stats.to_excel(writer, index=False, sheet_name="Summary")
                out_xlsx.seek(0)
                stats_xlsx_bytes = out_xlsx.read()

            # ✅ Preview combined data
            if not all_data.empty:
                st.write("### Preview of combined data")
                st.dataframe(all_data.head(20))
                st.write("Columns detected:", list(all_data.columns))

            # Colors for treatments
            colors = {
                "1": "black",
                "2": "orange",
                "3": "blue",
                "4": "green",
                "5": "red",
                "6": "purple",
                "7": "brown",
                "8": "pink",
                "9": "gray"
            }

            # Make one chart per metric
            if not all_data.empty:
                st.subheader("📊 Boxplots per Metric")
                for metric in sorted(metrics_found):
                    if metric not in all_data.columns:  # ✅ avoid KeyError
                        continue
                    buf = plot_metric_across_dates(all_data, metric, colors, title_prefix=title_prefix)
                    if buf is not None:
                        st.image(buf, caption=f"Box & Whisker across Dates – {metric}")

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
