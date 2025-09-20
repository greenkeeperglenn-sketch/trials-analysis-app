import streamlit as st
import pandas as pd
import plotly.express as px
import re
from io import BytesIO
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations

st.title("Assessment Data Explorer")

# --- Significance level selector ---
alpha_options = {
    "Fungicide (0.05)": 0.05,
    "Biologicals in lab (0.10)": 0.10,
    "Biologicals in field (0.15)": 0.15
}
alpha_label = st.radio("Select significance level:", list(alpha_options.keys()))
alpha_choice = alpha_options[alpha_label]

# --- Helper: Compact Letter Display (CLD) with overlaps ---
def generate_cld(means, mse, df_error, alpha, rep_counts):
    """
    Build overlapping letter groupings (a, ab, bc, ...) using LSD.
    - Lowest mean gets 'a', higher means progress to b, c, ...
    - A treatment can inherit multiple letters if it bridges groups.
    """
    # Critical t
    t_crit = stats.t.ppf(1 - alpha/2, df_error)

    treatments = list(means.index)
    # NSD matrix: True if two treatments are NOT significantly different by LSD
    nsd = pd.DataFrame(False, index=treatments, columns=treatments)
    for t in treatments:
        nsd.loc[t, t] = True  # diagonal

    for t1, t2 in combinations(treatments, 2):
        n1, n2 = rep_counts.get(t1, 1), rep_counts.get(t2, 1)
        # per-pair r as average reps (matches balanced design; adapts if blocks removed)
        r_pair = np.mean([n1, n2])
        lsd = t_crit * np.sqrt(2 * mse / r_pair)
        if abs(means[t1] - means[t2]) <= lsd:
            nsd.loc[t1, t2] = True
            nsd.loc[t2, t1] = True

    # Assign letters (lowest mean first)
    order = means.sort_values(ascending=True).index
    letters = {t: "" for t in treatments}
    groups = []  # each item: {"letter": "a", "members": [treatments...] }
    next_letter_code = ord("a")

    for t in order:
        joined_any = False
        # Try to join ALL compatible existing groups (enables overlaps)
        for g in groups:
            if all(nsd.loc[t, m] for m in g["members"]):
                letters[t] += g["letter"]
                g["members"].append(t)
                joined_any = True
        if not joined_any:
            new_letter = chr(next_letter_code)
            groups.append({"letter": new_letter, "members": [t]})
            letters[t] += new_letter
            next_letter_code += 1

    return letters

# --- Upload Excel file ---
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    all_data = []
    assessment_cols = set()
    block_col_global = None

    for sheet in xls.sheet_names:
        try:
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

            df = pd.read_excel(xls, sheet_name=sheet, skiprows=header_row)
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df = df.dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]

            col_map = {c: re.sub(r"\W+", "", c).lower() for c in df.columns}
            block_col = next((orig for orig, norm in col_map.items() if "block" in norm), None)
            plot_col = next((orig for orig, norm in col_map.items() if "plot" in norm), None)
            treat_col = next((orig for orig, norm in col_map.items() if "treat" in norm or "trt" in norm), None)

            if not (block_col and treat_col):
                st.warning(f"Sheet {sheet}: Could not detect Block/Treatment columns. Found {list(df.columns)}")
                continue

            block_col_global = block_col
            treat_idx = df.columns.get_loc(treat_col)
            assess_list = df.columns[treat_idx+1:].tolist()
            assessment_cols.update(assess_list)

            id_vars = [block_col, treat_col]
            if plot_col:
                id_vars.append(plot_col)

            df_long = df.melt(
                id_vars=id_vars,
                value_vars=assess_list,
                var_name="Assessment",
                value_name="Value"
            )
            df_long = df_long.rename(columns={block_col: "Block", treat_col: "Treatment"})
            df_long["Date"] = sheet
            all_data.append(df_long)

        except Exception as e:
            st.warning(f"Could not parse sheet {sheet}: {e}")

    if not all_data:
        st.error("No valid tables found in this file.")
    else:
        data = pd.concat(all_data, ignore_index=True)

        # --- Treatment naming box ---
        treatments = sorted(data["Treatment"].dropna().unique(), key=lambda x: str(x))
        st.subheader("Treatment Names")
        st.markdown(
            f"Detected **{len(treatments)} treatments**: {', '.join(map(str, treatments))}. "
            "Paste names below (one per line, in order)."
        )
        names_input = st.text_area("Treatment names", height=200, placeholder="Paste names here, one per line")
        if names_input.strip():
            pasted_names = [n.strip() for n in names_input.split("\n") if n.strip()]
            if len(pasted_names) == len(treatments):
                mapping = dict(zip(treatments, pasted_names))
                data["Treatment"] = data["Treatment"].map(mapping).fillna(data["Treatment"])
                treatments = pasted_names
            else:
                st.warning("Number of names pasted does not match number of treatments!")

        # --- Block selector ---
        if "Block" in data.columns:
            unique_blocks = sorted(data["Block"].dropna().unique())
            selected_blocks = st.multiselect(
                "Include Blocks", unique_blocks, default=unique_blocks
            )
            data = data[data["Block"].isin(selected_blocks)]

        # --- User selects assessments ---
        selected_assessments = st.multiselect(
            "Select assessments to analyse:",
            sorted(set(data["Assessment"].unique()))
        )

        view_mode = st.radio(
            "How should boxplots be grouped?",
            ["By Date (treatments across dates)", "By Treatment (dates across treatments)"]
        )

        color_map = {
            t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, t in enumerate(treatments)
        }

        all_tables = {}
        all_figs = []

        for assess in selected_assessments:
            st.subheader(f"Assessment: {assess}")
            df_sub = data[data["Assessment"] == assess].dropna(subset=["Value"])

            # --- Boxplot ---
            if view_mode.startswith("By Date"):
                fig = px.box(
                    df_sub,
                    x="Date", y="Value", color="Treatment",
                    color_discrete_map=color_map,
                    category_orders={"Treatment": treatments},
                    title=f"{assess} grouped by Date"
                )
            else:
                fig = px.box(
                    df_sub,
                    x="Treatment", y="Value", color="Date",
                    category_orders={"Treatment": treatments},
                    title=f"{assess} grouped by Treatment (colored by Date)"
                )
            fig.update_traces(boxpoints=False)
            st.plotly_chart(fig, use_container_width=True)
            all_figs.append(fig)

            # --- Stats Table ---
            wide_table = pd.DataFrame({"Treatment": treatments})
            summaries = {}

            for date in sorted(df_sub["Date"].unique()):
                df_date = df_sub[df_sub["Date"] == date].copy()
                df_date["Value"] = pd.to_numeric(df_date["Value"], errors="coerce")
                df_date = df_date.dropna(subset=["Value"])

                if df_date["Treatment"].nunique() > 1 and len(df_date) > 1:
                    means = df_date.groupby("Treatment")["Value"].mean()
                    rep_counts = df_date["Treatment"].value_counts().to_dict()

                    # Model with/without Block
                    if "Block" in df_date.columns:
                        model = ols("Value ~ C(Treatment) + C(Block)", data=df_date).fit()
                    else:
                        model = ols("Value ~ C(Treatment)", data=df_date).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    df_error = model.df_resid
                    mse = anova_table.loc["Residual", "sum_sq"] / df_error
                    p_val = anova_table.loc["C(Treatment)", "PR(>F)"]

                    # %CV
                    cv = 100 * np.sqrt(mse) / means.mean()

                    # Letters via CLD (overlapping)
                    letters = generate_cld(means, mse, df_error, alpha_choice, rep_counts)

                    # LSD (always)
                    t_crit = stats.t.ppf(1 - alpha_choice/2, df_error)
                    r = np.mean(list(rep_counts.values()))
                    lsd_val = t_crit * np.sqrt(2 * mse / r)

                    mean_col, group_col = f"{date} Mean", f"{date} Group"
                    wide_table[mean_col] = wide_table["Treatment"].map(means)
                    wide_table[group_col] = wide_table["Treatment"].map(letters)

                    summaries[date] = {"P": p_val, "LSD": lsd_val, "d.f.": df_error, "%CV": cv}
                else:
                    summaries[date] = {"P": np.nan, "LSD": np.nan, "d.f.": np.nan, "%CV": np.nan}

            # Add summary rows
            summary_rows = []
            for metric in ["P", "LSD", "d.f.", "%CV"]:
                row = {"Treatment": metric}
                for date in sorted(df_sub["Date"].unique()):
                    row[f"{date} Mean"] = summaries[date][metric]
                    row[f"{date} Group"] = ""
                summary_rows.append(row)
            wide_table = pd.concat([wide_table, pd.DataFrame(summary_rows)], ignore_index=True)

            st.dataframe(wide_table)
            all_tables[assess] = wide_table

        # --- Export Tables ---
        st.subheader("Export Results")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for assess, table in all_tables.items():
                table.to_excel(writer, sheet_name=assess[:30], index=False)
        st.download_button(
            "Download Tables (Excel)",
            data=buffer,
            file_name="assessment_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
