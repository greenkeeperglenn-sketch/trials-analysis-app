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

# Use full browser width
st.set_page_config(layout="wide")
st.title("Assessment Data Explorer")

# ======================
# UI: Significance level
# ======================
alpha_options = {
    "Fungicide (0.05)": 0.05,
    "Biologicals in lab (0.10)": 0.10,
    "Biologicals in field (0.15)": 0.15
}
alpha_label = st.radio("Select significance level:", list(alpha_options.keys()))
alpha_choice = alpha_options[alpha_label]

# ======================
# Helpers
# ======================
def parse_sheet_label_to_date(label: str):
    """Try to parse a sheet name into a datetime for sorting."""
    for dayfirst in (True, False):
        dt = pd.to_datetime(label, errors="coerce", dayfirst=dayfirst)
        if pd.notna(dt):
            return dt
    return None

def chronological_labels(labels):
    """Sort labels chronologically when possible."""
    pairs = []
    for lab in labels:
        dt = parse_sheet_label_to_date(lab)
        pairs.append((lab, dt))
    pairs_sorted = sorted(
        pairs,
        key=lambda x: (pd.isna(x[1]), x[1] if pd.notna(x[1]) else pd.Timestamp.max)
    )
    return [p[0] for p in pairs_sorted]

def generate_cld_overlap(means, mse, df_error, alpha, rep_counts, a_is_lowest=True):
    """
    Agricolae-style CLD with overlaps (ab, bc).
    Deduplicates letters so outputs are clean (no 'aa' or 'bb').
    """
    trts = list(means.index)
    letters = {t: set() for t in trts}

    # --- Build NSD matrix ---
    nsd = pd.DataFrame(False, index=trts, columns=trts)
    for t in trts:
        nsd.loc[t, t] = True
    t_crit = stats.t.ppf(1 - alpha/2, df_error) if df_error > 0 else np.nan
    for a, b in combinations(trts, 2):
        n1, n2 = rep_counts.get(a, 1), rep_counts.get(b, 1)
        if n1 > 0 and n2 > 0 and pd.notna(mse) and pd.notna(t_crit):
            lsd_pair = t_crit * np.sqrt(mse * (1/n1 + 1/n2))
            diff = abs(means[a] - means[b])
            if diff <= lsd_pair:
                nsd.loc[a, b] = True
                nsd.loc[b, a] = True

    # --- Assign groups with back-fill closure ---
    order = means.sort_values(ascending=a_is_lowest).index
    groups = []
    next_letter_code = ord("a")

    for t in order:
        joined_any = False
        for g in groups:
            if all(nsd.loc[t, m] for m in g["members"]):
                letters[t].add(g["letter"])
                g["members"].append(t)
                joined_any = True
        if not joined_any:
            new_letter = chr(next_letter_code)
            groups.append({"letter": new_letter, "members": [t]})
            letters[t].add(new_letter)
            next_letter_code += 1

        # --- Back-fill step ---
        changed = True
        while changed:
            changed = False
            for g in groups:
                for cand in trts:
                    if g["letter"] not in letters[cand]:
                        if all(nsd.loc[cand, m] for m in g["members"]):
                            letters[cand].add(g["letter"])
                            g["members"].append(cand)
                            changed = True

    letters = {t: "".join(sorted(v)) for t, v in letters.items()}
    return letters, nsd

def rotate_headers(df):
    """Apply header rotation + small font via Styler, with rounding."""
    return (df.style
              .set_table_styles(
                  [{"selector": "th.col_heading",
                    "props": [("transform", "rotate(-60deg)"),
                              ("text-align", "left"),
                              ("vertical-align", "bottom"),
                              ("font-size", "10px"),
                              ("white-space", "nowrap")]}]
              )
              .format(precision=1)  # Force 1 decimal place in display
           )

# ======================
# Upload & Parse
# ======================
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    all_data = []
    assessment_cols = set()

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
                continue

            df = pd.read_excel(xls, sheet_name=sheet, skiprows=header_row)
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df = df.dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]

            col_map = {c: re.sub(r"\W+", "", c).lower() for c in df.columns}
            block_col = next((o for o, n in col_map.items() if "block" in n), None)
            plot_col  = next((o for o, n in col_map.items() if "plot"  in n), None)
            treat_col = next((o for o, n in col_map.items() if "treat" in n or "trt" in n), None)
            if not (block_col and treat_col):
                continue

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
            df_long["DateLabel"] = sheet
            df_long["DateParsed"] = parse_sheet_label_to_date(sheet)
            all_data.append(df_long)
        except Exception:
            continue

    if not all_data:
        st.error("No valid tables found in this file.")
    else:
        data = pd.concat(all_data, ignore_index=True)

        # Treatment naming
        treatments = sorted(data["Treatment"].dropna().unique(), key=lambda x: str(x))
        st.subheader("Treatment Names")
        names_input = st.text_area("Paste treatment names (one per line)", height=200)
        if names_input.strip():
            pasted = [n.strip() for n in names_input.split("\n") if n.strip()]
            if len(pasted) == len(treatments):
                mapping = dict(zip(treatments, pasted))
                data["Treatment"] = data["Treatment"].map(mapping).fillna(data["Treatment"])
                treatments = pasted
            else:
                st.warning("Number of names pasted does not match detected treatments!")

        # Assessment selector
        selected_assessments = st.multiselect("Select assessments:", sorted(set(data["Assessment"].unique())))
        view_mode = st.radio("How should boxplots be grouped?", ["By Date", "By Treatment"])

        color_map = {t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, t in enumerate(treatments)}
        date_labels_all = data["DateLabel"].dropna().unique().tolist()
        date_labels_ordered = chronological_labels(date_labels_all)

        all_tables = {}
        for assess in selected_assessments:
            st.subheader(f"Assessment: {assess}")
            df_sub = data[data["Assessment"] == assess].copy()
            df_sub["Value"] = pd.to_numeric(df_sub["Value"], errors="coerce")
            df_sub = df_sub.dropna(subset=["Value"])

            # --- Block selector applied BEFORE plotting ---
            if "Block" in df_sub.columns:
                blocks = sorted(df_sub["Block"].dropna().unique())
                sel_blocks = st.multiselect(
                    f"Include Blocks for {assess}",
                    blocks,
                    default=blocks,
                    key=f"blocks_{assess.replace(' ', '_')}"
                )
                if not sel_blocks:
                    st.warning("No blocks selected. Please select at least one block to see results.")
                    continue
                df_sub = df_sub[df_sub["Block"].isin(sel_blocks)]

            # --- Boxplot ---
            if view_mode == "By Date":
                fig = px.box(df_sub, x="DateLabel", y="Value", color="Treatment",
                             color_discrete_map=color_map,
                             category_orders={"DateLabel": date_labels_ordered, "Treatment": treatments})
            else:
                fig = px.box(df_sub, x="Treatment", y="Value", color="DateLabel",
                             category_orders={"Treatment": treatments, "DateLabel": date_labels_ordered})
            fig.update_traces(boxpoints=False)
            st.plotly_chart(fig, use_container_width=True)

            # --- Stats table ---
            wide_table = pd.DataFrame({"Treatment": treatments})
            summaries = {}
            nsd_debug = {}

            for date_label in date_labels_ordered:
                df_date = df_sub[df_sub["DateLabel"] == date_label].copy()
                if df_date.empty:
                    wide_table[f"{date_label}"] = np.nan
                    wide_table[f"{date_label} S"] = ""
                    summaries[date_label] = {"P": np.nan, "LSD": np.nan, "d.f.": np.nan, "%CV": np.nan}
                    continue

                df_date["Value"] = pd.to_numeric(df_date["Value"], errors="coerce")
                df_date = df_date.dropna(subset=["Value"])
                if df_date["Treatment"].nunique() > 1 and len(df_date) > 1:
                    means = df_date.groupby("Treatment")["Value"].mean()
                    rep_counts = df_date["Treatment"].value_counts().to_dict()
                    try:
                        if "Block" in df_date.columns:
                            model = ols("Value ~ C(Treatment) + C(Block)", data=df_date).fit()
                        else:
                            model = ols("Value ~ C(Treatment)", data=df_date).fit()
                        anova = sm.stats.anova_lm(model, typ=2)
                        df_error = float(model.df_resid)
                        mse = float(anova.loc["Residual", "sum_sq"] / df_error)
                        p_val = float(anova.loc["C(Treatment)", "PR(>F)"])
                    except Exception:
                        df_error, mse, p_val = np.nan, np.nan, np.nan
                    cv = 100 * np.sqrt(mse) / means.mean() if pd.notna(mse) and means.mean() != 0 else np.nan

                    # --- Lettering toggle (unique key per assessment + date) ---
                    a_is_lowest = (
                        st.radio(
                            f"Lettering convention for {assess} ({date_label})",
                            ["Lowest = A", "Highest = A"],
                            index=0,
                            key=f"letters_{assess.replace(' ', '_')}_{date_label.replace(' ', '_')}"
                        ) == "Lowest = A"
                    )

                    letters, nsd = generate_cld_overlap(means, mse, df_error, alpha_choice, rep_counts, a_is_lowest=a_is_lowest)
                    nsd_debug[date_label] = nsd
                    n_avg = np.mean(list(rep_counts.values()))
                    lsd_val = stats.t.ppf(1 - alpha_choice/2, df_error) * np.sqrt(2*mse/n_avg) if pd.notna(mse) else np.nan
                    wide_table[f"{date_label}"] = wide_table["Treatment"].map(means)
                    wide_table[f"{date_label} S"] = wide_table["Treatment"].map(letters).fillna("")
                    summaries[date_label] = {"P": p_val, "LSD": lsd_val, "d.f.": df_error, "%CV": cv}
                else:
                    wide_table[f"{date_label}"] = np.nan
                    wide_table[f"{date_label} S"] = ""
                    summaries[date_label] = {"P": np.nan, "LSD": np.nan, "d.f.": np.nan, "%CV": np.nan}

            summary_rows = []
            for metric in ["P", "LSD", "d.f.", "%CV"]:
                row = {"Treatment": metric}
                for date_label in date_labels_ordered:
                    row[f"{date_label}"] = summaries[date_label][metric]
                    row[f"{date_label} S"] = ""
                summary_rows.append(row)
            wide_table = pd.concat([wide_table, pd.DataFrame(summary_rows)], ignore_index=True)

            wide_table = wide_table.round(1)

            st.dataframe(
                rotate_headers(wide_table),
                use_container_width=True,
                hide_index=True,
                height=wide_table.shape[0] * 35,
                column_config={"Treatment": st.column_config.Column(pinned=True)}
            )

            all_tables[assess] = wide_table

            with st.expander(f"NSD Matrix ({assess})"):
                for date_label, nsd in nsd_debug.items():
                    st.markdown(f"**{date_label}**")
                    nsd_display = nsd.replace({True: "✓", False: "×"})
                    st.dataframe(nsd_display)

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for assess, table in all_tables.items():
                table.round(1).to_excel(writer, sheet_name=assess[:30], index=False)
        st.download_button("Download Tables (Excel)", data=buffer,
                           file_name="assessment_tables.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
