import streamlit as st
import pandas as pd
import plotly.express as px
import re
from io import BytesIO

st.title("Assessment Data Explorer")

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

            # Read full sheet with detected header
            df = pd.read_excel(xls, sheet_name=sheet, header=header_row)

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
        treatments = sorted(data[treat_col].dropna().unique())
        color_map = {
            t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, t in enumerate(treatments)
        }

        # Show plots + tables
        for assess in selected_assessments:
            st.subheader(f"Assessment: {assess}")
            df_sub = data[data["Assessment"] == assess].dropna(subset=["Value"])

            if view_mode.startswith("By Date"):
                fig = px.box(
                    df_sub,
                    x="Date", y="Value", color=treat_col,
                    color_discrete_map=color_map,
                    points="all",
                    title=f"{assess} grouped by Date"
                )
            else:
                fig = px.box(
                    df_sub,
                    x=treat_col, y="Value", color="Date",
                    points="all",
                    title=f"{assess} grouped by Treatment"
                )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_sub)

        # ---- Download section ----
        st.subheader("Download Tidy Dataset")
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", data=csv, file_name="tidy_assessments.csv", mime="text/csv")

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            data.to_excel(writer, index=False, sheet_name="Assessments")
        st.download_button(
            "Download as Excel",
            data=buffer,
            file_name="tidy_assessments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
