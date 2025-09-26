import streamlit as st
import pandas as pd
import re
from helpers import parse_sheet_label_to_date, chronological_labels

def load_data():
    """Upload and parse Excel into a clean long-format DataFrame."""

    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
    if not uploaded_file:
        return None, None, None

    xls = pd.ExcelFile(uploaded_file)
    all_data = []
    assessment_cols = set()

    for sheet in xls.sheet_names:
        try:
            # Look ahead for header row
            preview = pd.read_excel(xls, sheet_name=sheet, nrows=20)
            header_row = None
            for i, row in preview.iterrows():
                values = [str(v).lower() for v in row.values if pd.notna(v)]
                if any("block" in v for v in values) and any("treat" in v for v in values):
                    header_row = i
                    break
            if header_row is None:
                continue

            # Read full sheet
            df = pd.read_excel(xls, sheet_name=sheet, skiprows=header_row)
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df = df.dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]

            # Identify key columns
            col_map = {c: re.sub(r"\\W+", "", c).lower() for c in df.columns}
            block_col = next((o for o, n in col_map.items() if "block" in n), None)
            plot_col  = next((o for o, n in col_map.items() if "plot" in n), None)
            treat_col = next((o for o, n in col_map.items() if "treat" in n or "trt" in n), None)
            if not (block_col and treat_col):
                continue

            treat_idx = df.columns.get_loc(treat_col)
            assess_list = df.columns[treat_idx+1:].tolist()
            assessment_cols.update(assess_list)

            id_vars = [block_col, treat_col]
            if plot_col:
                id_vars.append(plot_col)

            # Melt to long format
            df_long = df.melt(
                id_vars=id_vars,
                value_vars=assess_list,
                var_name="Assessment",
                value_name="Value"
            )
            df_long = df_long.rename(columns={block_col: "Block", treat_col: "Treatment"})
            df_long["DateLabel"] = sheet
            df_long["DateParsed"] = parse_sheet_label_to_date(sheet)

            all_data.append(df_long)

        except Exception:
            continue

    if not all_data:
        st.error("No valid tables found in this file.")
        return None, None, None

    # Combine sheets
    data = pd.concat(all_data, ignore_index=True)

    # Treatment naming
    treatments = sorted(data["Treatment"].dropna().unique(), key=lambda x: str(x))
    st.sidebar.subheader("Treatment Names")
    names_input = st.sidebar.text_area("Paste treatment names (one per line)", height=200)
    if names_input.strip():
        pasted = [n.strip() for n in names_input.split("\\n") if n.strip()]
        if len(pasted) == len(treatments):
            mapping = dict(zip(treatments, pasted))
            data["Treatment"] = data["Treatment"].map(mapping).fillna(data["Treatment"])
            treatments = pasted
        else:
            st.sidebar.warning("Number of names pasted does not match detected treatments!")

    # Chronological ordering of sheets
    date_labels_all = data["DateLabel"].dropna().unique().tolist()
    date_labels_ordered = chronological_labels(date_labels_all)

    return data, treatments, date_labels_ordered
