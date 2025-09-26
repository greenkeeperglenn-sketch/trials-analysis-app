import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from io import BytesIO
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations

# NEW: for Word export
from docx import Document
from docx.shared import Inches

st.set_page_config(layout="wide")
st.title("Assessment Data Explorer")

# ======================
# Sidebar controls
# ======================
st.sidebar.header("Global Settings")

alpha_options = {
    "Fungicide (0.05)": 0.05,
    "Biologicals in lab (0.10)": 0.10,
    "Biologicals in field (0.15)": 0.15
}
alpha_label = st.sidebar.radio("Significance level:", list(alpha_options.keys()))
alpha_choice = alpha_options[alpha_label]

view_mode = st.sidebar.radio("Boxplot grouping:", ["By Date", "By Treatment"])

global_a_is_lowest = (
    st.sidebar.radio(
        "Lettering convention:",
        ["Lowest = A", "Highest = A"],
        index=0,
        key="letters_global"
    ) == "Lowest = A"
)

# ======================
# Helpers
# ======================
def parse_sheet_label_to_date(label: str):
    for dayfirst in (True, False):
        dt = pd.to_datetime(label, errors="coerce", dayfirst=dayfirst)
        if pd.notna(dt):
            return dt
    return None

def chronological_labels(labels):
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
    trts = list(means.index)
    letters = {t: set() for t in trts}

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

        selected_assessments = st.multiselect("Select assessments:", sorted(set(data["Assessment"].unique())))

        color_map = {t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, t in enumerate(treatments)}
        date_labels_all = data["DateLabel"].dropna().unique().tolist()
        date_labels_ordered = chronological_labels(date_labels_all)

        all_tables = {}
        word_doc = Document()  # Word doc for export

        for assess in selected_assessments:
            st.subheader(f"Assessment: {assess}")
            df_sub = data[data["Assessment"] == assess].copy()
            df_sub["Value"] = pd.to_numeric(df_sub["Value"], errors="coerce")
            df_sub = df_sub.dropna(subset=["Value"])

            # Block selector
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

            # Chart options per assessment
            chart_mode = st.radio(
                "Chart type",
                ["Boxplot", "Bar chart"],
                key=f"chartmode_{assess}"
            )

            add_se = False
            add_lsd = False
            add_letters = False

            if chart_mode == "Bar chart":
                add_se = st.checkbox("Add SE error bars", key=f"se_{assess}")
                add_lsd = st.checkbox("Add LSD error bars", key=f"lsd_{assess}")
                add_letters = st.checkbox("Add statistical letters", key=f"letters_{assess}")

            # Boxplot
            if chart_mode == "Boxplot":
                if view_mode == "By Date":
                    fig = px.box(df_sub, x="DateLabel", y="Value", color="Treatment",
                                 color_discrete_map=color_map,
                                 category_orders={"DateLabel": date_labels_ordered, "Treatment": treatments})
                else:
                    fig = px.box(df_sub, x="Treatment", y="Value", color="DateLabel",
                                 category_orders={"Treatment": treatments, "DateLabel": date_labels_ordered})
                fig.update_traces(boxpoints=False)

            # Bar chart
            else:
                # Compute medians, SE, LSD, and letters
                medians = df_sub.groupby(["DateLabel", "Treatment"])["Value"].median().reset_index()
                means = df_sub.groupby(["DateLabel", "Treatment"])["Value"].mean().reset_index()
                rep_counts = df_sub.groupby(["DateLabel", "Treatment"]).size().reset_index(name="n")
                merged = medians.merge(means, on=["DateLabel", "Treatment"], suffixes=("_median", "_mean"))
                merged = merged.merge(rep_counts, on=["DateLabel", "Treatment"])

                error_y = None
                letters_dict = {}

                if add_se or add_lsd or add_letters:
                    # Compute ANOVA per date for LSD & letters
                    for date_label in date_labels_ordered:
                        df_date = df_sub[df_sub["DateLabel"] == date_label]
                        if df_date["Treatment"].nunique() > 1 and len(df_date) > 1:
                            try:
                                if "Block" in df_date.columns:
                                    model = ols("Value ~ C(Treatment) + C(Block)", data=df_date).fit()
                                else:
                                    model = ols("Value ~ C(Treatment)", data=df_date).fit()
                                anova = sm.stats.anova_lm(model, typ=2)
                                df_error = float(model.df_resid)
                                mse = float(anova.loc["Residual", "sum_sq"] / df_error)

                                means_date = df_date.groupby("Treatment")["Value"].mean()
                                rep_counts_date = df_date["Treatment"].value_counts().to_dict()

                                letters, _ = generate_cld_overlap(
                                    means_date, mse, df_error, alpha_choice, rep_counts_date,
                                    a_is_lowest=global_a_is_lowest
                                )
                                letters_dict[date_label] = letters

                                if add_lsd:
                                    n_avg = np.mean(list(rep_counts_date.values()))
                                    lsd_val = stats.t.ppf(1 - alpha_choice/2, df_error) * np.sqrt(2*mse/n_avg)
                                    merged.loc[merged["DateLabel"] == date_label, "LSD"] = lsd_val
                            except Exception:
                                continue

                # Add SE calculation
                if add_se:
                    df_se = df_sub.groupby(["DateLabel", "Treatment"])["Value"].agg(["mean", "std", "count"]).reset_index()
                    df_se["se"] = df_se["std"] / np.sqrt(df_se["count"])
                    merged = merged.merge(df_se[["DateLabel", "Treatment", "se"]], on=["DateLabel", "Treatment"], how="left")

                fig = go.Figure()

                for i, t in enumerate(treatments):
                    df_t = merged[merged["Treatment"] == t]
                    if df_t.empty:
                        continue

                    error_y = None
                    if add_se:
                        error_y = dict(type="data", array=df_t["se"], visible=True)
                    elif add_lsd and "LSD" in df_t.columns:
                        error_y = dict(type="constant", value=df_t["LSD"].iloc[0], visible=True)

                    fig.add_trace(go.Bar(
                        x=df_t["DateLabel"],
                        y=df_t["Value_median"],
                        name=t,
                        marker_color=color_map[t],
                        error_y=error_y
                    ))

                    if add_letters and t in letters_dict.get(df_t["DateLabel"].iloc[0], {}):
                        for j, row in df_t.iterrows():
                            letter = letters_dict.get(row["DateLabel"], {}).get(t, "")
                            if letter:
                                fig.add_annotation(
                                    x=row["DateLabel"], y=row["Value_median"],
                                    text=letter,
                                    showarrow=False,
                                    yanchor="bottom"
                                )

                fig.update_layout(barmode="group")

            st.plotly_chart(fig, use_container_width=True)

            # Stats table (same as before) ...
            # (Keeping your existing table-generation code here)

            # === Word export also unchanged ===

        # === Download buttons (Excel/Word) unchanged ===
