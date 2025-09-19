import streamlit as st
import pandas as pd
import plotly.express as px
import re
from io import BytesIO
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp

st.title("Assessment Data Explorer")

# --- Significance level selector ---
alpha_choice = st.radio(
    "Select significance level:",
    {
        "Fungicide (0.005)": 0.005,
        "Biologicals in lab (0.010)": 0.010,
        "Biologicals in field (0.015)": 0.015
    }
)

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    all_data = []
    assessment_cols = set()

    for sheet in xls.sheet_names:
        try:
            # Look at the first 20 rows to find header row
            preview = pd.read_excel(xls, sheet_name=sheet, nrows=20)
            header_row = None
            for i, row in preview.iterrows():
                values = [str(v).lower() for v in row.values if pd.notna(v)]
                if any("block" in v for v in values) and any("treat" in v for v in values):
                    header_row = i
                    break

            if header_row is None:
                st.warning(f"No suitable header row found in {sheet}, skipping.")
                continue

            # Read full sheet, skipping up to header row
            df = pd.read_excel(xls, sheet_name=sheet, skiprows=header_row)

            # Promote first row to header
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])

            # Drop empty columns
            df = df.dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]

            # ---- Robust column detection ----
            col_map = {c: re.sub(r"\W+", "", c).lower() for c in df.columns}

            block_col = next((orig for orig, norm in col_map.items() if "block" in norm), None)
            plot_col = next((orig for orig, norm in col_map.items() if "plot" in norm), None)
            treat_col = next((orig for orig, norm in col_map.items() if "treat" in norm or "trt" in norm), None)

            if not (block_col and treat_col):
                st.warning(
                    f"Sheet {sheet}: Could not detect Block/Treatment columns. "
                    f"Detected columns: {list(df.columns)}"
                )
                continue

            # Everything after treatment column = assessments
            treat_idx = df.columns.get_loc(treat_col)
            assess_list = df.columns[treat_idx+1:].tolist()
            assessment_cols.update(assess_list)

            # Reshape into tidy format
            id_vars = [block_col, treat_col]
            if plot_col:
                id_vars.append(plot_col)

            df_long = df.melt(
                id_vars=id_vars,
                value_vars=assess_list,
                var_name="Assessment",
                value_name="Value"
            )
            df_long["Date"] = sheet
            all_data.append(df_long)

        except Exception as e:
            st.warning(f"Could not parse sheet {sheet}: {e}")

    if not all_data:
        st.error("No valid tables found in this file.")
    else:
        data = pd.concat(all_data, ignore_index=True)

        # Let user select which assessments to analyse
        selected_assessments = st.multiselect(
            "Select assessments to analyse:",
            sorted(assessment_cols)
        )

        # Flip toggle
        view_mode = st.radio(
            "How should boxplots be grouped?",
            ["By Date (treatments across dates)", "By Treatment (dates across treatments)"]
        )

        # Fixed color mapping for treatments
        treat_col = next((c for c in data.columns if re.search("treat|trt", c, re.I)))
        treatments = sorted(data[treat_col].dropna().unique(), key=lambda x: int(x))
        color_map = {
            t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, t in enumerate(treatments)
        }

        all_tables = {}
        all_figs = []

        # Show plots + stats tables
        for assess in selected_assessments:
            st.subheader(f"Assessment: {assess}")
            df_sub = data[data["Assessment"] == assess].dropna(subset=["Value"])

            # --- Boxplot ---
            if view_mode.startswith("By Date"):
                fig = px.box(
                    df_sub,
                    x="Date", y="Value", color=treat_col,
                    color_discrete_map=color_map,
                    category_orders={treat_col: treatments},
                    title=f"{assess} grouped by Date"
                )
            else:
                fig = px.box(
                    df_sub,
                    x=treat_col, y="Value", color="Date",
                    category_orders={treat_col: treatments},
                    title=f"{assess} grouped by Treatment (colored by Date)"
                )

            fig.update_traces(boxpoints=False)
            fig.update_layout(boxmode="group")
            st.plotly_chart(fig, use_container_width=True)
            all_figs.append(fig)

            # --- Stats Table ---
            wide_table = pd.DataFrame({ "Treatment": treatments })

            for date in sorted(df_sub["Date"].unique()):
                df_date = df_sub[df_sub["Date"] == date]

                # Means
                means = df_date.groupby(treat_col)["Value"].mean()

                # ANOVA
                model = ols("Value ~ C("+treat_col+")", data=df_date).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                p_val = anova_table["PR(>F)"][0]
                df_error = model.df_resid
                mse = anova_table["sum_sq"][-1] / df_error

                # LSD calc
                t_crit = stats.t.ppf(1 - alpha_choice/2, df_error)
                lsd = t_crit * np.sqrt(2*mse/df_date[treat_col].value_counts().min())

                # %CV
                cv = 100 * np.sqrt(mse) / means.mean()

                # Grouping letters
                letters = sp.posthoc_dunn(df_date, val_col="Value", group_col=treat_col, p_adjust="bonferroni")
                # Simplified placeholder: assign 'a' for now
                # TODO: implement LSD-based grouping

                mean_col = f"{date} Mean"
                group_col = f"{date} Group"

                wide_table[mean_col] = wide_table["Treatment"].map(means)
                wide_table[group_col] = "a"  # placeholder grouping

                # Add summary rows later
                summary = pd.DataFrame({
                    "Treatment": ["P", "LSD", "d.f.", "%CV"],
                    mean_col: [p_val, lsd, df_error, cv],
                    group_col: ["", "", "", ""]
                })

            st.dataframe(wide_table)

            all_tables[assess] = wide_table

        # ---- Export section ----
        st.subheader("Export Results")

        # Export tables
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for assess, table in all_tables.items():
                table.to_excel(writer, sheet_name=assess, index=False)
        st.download_button(
            "Download Tables (Excel)",
            data=buffer,
            file_name="assessment_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_data = "\n\n".join([df.to_csv(index=False) for df in all_tables.values()]).encode("utf-8")
        st.download_button(
            "Download Tables (CSV)",
            data=csv_data,
            file_name="assessment_tables.csv",
            mime="text/csv"
        )

        # Export charts (PDF)
        # TODO: export Plotly figs as PDF
