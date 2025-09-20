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

# ======================
# UI: Significance & Lettering options
# ======================
alpha_options = {
    "Fungicide (0.05)": 0.05,
    "Biologicals in lab (0.10)": 0.10,
    "Biologicals in field (0.15)": 0.15
}
alpha_label = st.radio("Select significance level:", list(alpha_options.keys()))
alpha_choice = alpha_options[alpha_label]

lettering_mode = st.radio(
    "Lettering convention:",
    ["Lowest = A", "Highest = A"],
    index=0
)

# ======================
# Helpers
# ======================
def parse_sheet_label_to_date(label: str):
    """
    Try to parse a sheet name like '03.07.25' or '16.09.25' or '2024-07-03' into a datetime.
    Returns (parsed_datetime or None).
    """
    # Try several common patterns
    for dayfirst in (True, False):
        dt = pd.to_datetime(label, errors="coerce", dayfirst=dayfirst)
        if pd.notna(dt):
            return dt
    return None

def chronological_labels(labels):
    """
    Given iterable of sheet labels (strings), return a list ordered chronologically,
    using parsed dates when possible, otherwise fallback to original order.
    """
    pairs = []
    for lab in labels:
        dt = parse_sheet_label_to_date(lab)
        pairs.append((lab, dt))
    # sort by parsed date; NaT at end but stable by original
    pairs_sorted = sorted(pairs, key=lambda x: (pd.isna(x[1]), x[1] if pd.notna(x[1]) else pd.Timestamp.max))
    return [p[0] for p in pairs_sorted]

def lsd_value(mse, df_error, alpha, r):
    """LSD using equal-reps formula with r = average reps per treatment."""
    if df_error <= 0 or mse < 0 or r <= 0:
        return np.nan
    t_crit = stats.t.ppf(1 - alpha/2, df_error)
    return t_crit * np.sqrt(2 * mse / r)

def build_nsd_matrix(means: pd.Series, mse: float, df_error: float, alpha: float, rep_counts: dict):
    """
    Build NSD (not significantly different) boolean matrix for all treatment pairs using LSD.
    NSD[i,j] = True if |mean_i - mean_j| <= LSD(i,j), with r_pair = avg rep count.
    """
    trts = list(means.index)
    nsd = pd.DataFrame(False, index=trts, columns=trts)
    for t in trts:
        nsd.loc[t, t] = True
    t_crit = stats.t.ppf(1 - alpha/2, df_error) if df_error > 0 else np.nan
    if np.isnan(t_crit) or mse < 0:
        return nsd

    for a, b in combinations(trts, 2):
        n1, n2 = rep_counts.get(a, 1), rep_counts.get(b, 1)
        r_pair = np.mean([n1, n2]) if (n1 > 0 and n2 > 0) else 1.0
        lsd_pair = t_crit * np.sqrt(2 * mse / r_pair)
        diff = abs(means[a] - means[b])
        if diff <= lsd_pair:
            nsd.loc[a, b] = True
            nsd.loc[b, a] = True
    return nsd

def generate_cld_overlap(means, mse, df_error, alpha, rep_counts, a_is_lowest=True):
    """
    Agricolae-style overlapping CLD:
    - Sort by mean ascending (lowest=A) or descending (highest=A), per toggle.
    - A treatment can join ANY existing letter group if it is NSD with AT LEAST ONE member of that group.
      (This allows bridge treatments to inherit multiple letters, e.g., 'ab', 'bc'.)
    - If it can't join any existing group, create a new letter.

    Returns dict: Treatment -> letter string (e.g., 'a', 'ab', 'bc').
    """
    trts = list(means.index)
    letters = {t: "" for t in trts}

    # Build NSD matrix
    nsd = build_nsd_matrix(means, mse, df_error, alpha, rep_counts)

    # Sort order (lowest=A vs highest=A)
    order = means.sort_values(ascending=a_is_lowest).index

    groups = []  # list of dicts: {"letter": str, "members": [treatments]}
    next_letter_code = ord("a")

    for t in order:
        joined_any = False
        # Try to join ALL compatible groups (this is what enables overlaps)
        for g in groups:
            # Agricolae-style bridging: NSD with at least ONE member of the group
            # (not necessarily all), so mid treatments can inherit adjacent letters.
            if any(nsd.loc[t, m] for m in g["members"]):
                letters[t] += g["letter"]
                g["members"].append(t)
                joined_any = True

        if not joined_any:
            # Start a new group/letter
            new_letter = chr(next_letter_code)
            groups.append({"letter": new_letter, "members": [t]})
            letters[t] += new_letter
            next_letter_code += 1

    return letters

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
            # Find header row by scanning first rows for 'block' and 'treat'
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

            # Read the sheet, promote first row to header
            df = pd.read_excel(xls, sheet_name=sheet, skiprows=header_row)
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df = df.dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]

            # Flexible detection
            col_map = {c: re.sub(r"\W+", "", c).lower() for c in df.columns}
            block_col = next((orig for orig, norm in col_map.items() if "block" in norm), None)
            plot_col  = next((orig for orig, norm in col_map.items() if "plot"  in norm), None)
            treat_col = next((orig for orig, norm in col_map.items() if "treat" in norm or "trt" in norm), None)

            if not (block_col and treat_col):
                st.warning(f"Sheet {sheet}: Could not detect Block/Treatment columns. Found {list(df.columns)}")
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
            # Standardise names we carry forward
            df_long = df_long.rename(columns={block_col: "Block", treat_col: "Treatment"})
            df_long["DateLabel"] = sheet  # keep original label for display
            # Parsed date for ordering
            parsed = parse_sheet_label_to_date(sheet)
            df_long["DateParsed"] = parsed

            all_data.append(df_long)

        except Exception as e:
            st.warning(f"Could not parse sheet {sheet}: {e}")

    if not all_data:
        st.error("No valid tables found in this file.")
    else:
        data = pd.concat(all_data, ignore_index=True)

        # ======================
        # Treatment naming
        # ======================
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

        # ======================
        # Block selector
        # ======================
        if "Block" in data.columns:
            unique_blocks = sorted(data["Block"].dropna().unique())
            selected_blocks = st.multiselect("Include Blocks", unique_blocks, default=unique_blocks)
            data = data[data["Block"].isin(selected_blocks)]

        # ======================
        # Assessment selector & chart mode
        # ======================
        selected_assessments = st.multiselect(
            "Select assessments to analyse:",
            sorted(set(data["Assessment"].unique()))
        )

        view_mode = st.radio(
            "How should boxplots be grouped?",
            ["By Date (treatments across dates)", "By Treatment (dates across treatments)"]
        )

        # Colors lock for treatments
        color_map = {
            t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, t in enumerate(treatments)
        }

        # Chronological order of date labels for this dataset
        date_labels_all = data["DateLabel"].dropna().unique().tolist()
        date_labels_ordered = chronological_labels(date_labels_all)

        all_tables = {}
        all_figs = []

        for assess in selected_assessments:
            st.subheader(f"Assessment: {assess}")
            df_sub = data[data["Assessment"] == assess].copy()
            df_sub["Value"] = pd.to_numeric(df_sub["Value"], errors="coerce")
            df_sub = df_sub.dropna(subset=["Value"])

            # --- Boxplots ---
            if view_mode.startswith("By Date"):
                fig = px.box(
                    df_sub,
                    x="DateLabel", y="Value", color="Treatment",
                    color_discrete_map=color_map,
                    category_orders={"DateLabel": date_labels_ordered, "Treatment": treatments},
                    title=f"{assess} grouped by Date"
                )
            else:
                fig = px.box(
                    df_sub,
                    x="Treatment", y="Value", color="DateLabel",
                    category_orders={"Treatment": treatments, "DateLabel": date_labels_ordered},
                    title=f"{assess} grouped by Treatment (colored by Date)"
                )
            fig.update_traces(boxpoints=False)
            st.plotly_chart(fig, use_container_width=True)
            all_figs.append(fig)

            # --- Stats table (chronological columns) ---
            wide_table = pd.DataFrame({"Treatment": treatments})
            summaries = {}

            for date_label in date_labels_ordered:
                df_date = df_sub[df_sub["DateLabel"] == date_label].copy()
                if df_date.empty:
                    # Put NaNs if that date not present for this assessment
                    wide_table[f"{date_label} Mean"] = np.nan
                    wide_table[f\"{date_label} Group\"] = ""
                    summaries[date_label] = {"P": np.nan, "LSD": np.nan, "d.f.": np.nan, "%CV": np.nan}
                    continue

                # Ensure numeric and sufficient data
                df_date["Value"] = pd.to_numeric(df_date["Value"], errors="coerce")
                df_date = df_date.dropna(subset=["Value"])

                if df_date["Treatment"].nunique() > 1 and len(df_date) > 1:
                    means = df_date.groupby("Treatment")["Value"].mean()
                    rep_counts = df_date["Treatment"].value_counts().to_dict()

                    # ANOVA (include Block if present)
                    try:
                        if "Block" in df_date.columns:
                            model = ols("Value ~ C(Treatment) + C(Block)", data=df_date).fit()
                        else:
                            model = ols("Value ~ C(Treatment)", data=df_date).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        df_error = float(model.df_resid)
                        mse = float(anova_table.loc["Residual", "sum_sq"] / df_error)
                        p_val = float(anova_table.loc["C(Treatment)", "PR(>F)"])
                    except Exception:
                        # Fallback if labels collide or data degenerate
                        df_error, mse, p_val = np.nan, np.nan, np.nan

                    # %CV
                    cv = 100 * np.sqrt(mse) / means.mean() if pd.notna(mse) and means.mean() != 0 else np.nan

                    # Letters via overlapping CLD (Agricolae-style)
                    a_is_lowest = (lettering_mode == "Lowest = A")
                    letters = generate_cld_overlap(means, mse, df_error, alpha_choice, rep_counts, a_is_lowest=a_is_lowest)

                    # LSD (always)
                    r = np.mean(list(rep_counts.values()))
                    lsd_val = lsd_value(mse, df_error, alpha_choice, r)

                    # Write columns in timeline order
                    wide_table[f"{date_label} Mean"] = wide_table["Treatment"].map(means)
                    wide_table[f"{date_label} Group"] = wide_table["Treatment"].map(letters).fillna("")

                    summaries[date_label] = {"P": p_val, "LSD": lsd_val, "d.f.": df_error, "%CV": cv}
                else:
                    # Not enough data to analyse
                    wide_table[f"{date_label} Mean"] = np.nan
                    wide_table[f"{date_label} Group"] = ""
                    summaries[date_label] = {"P": np.nan, "LSD": np.nan, "d.f.": np.nan, "%CV": np.nan}

            # Append summary rows (in same date order)
            summary_rows = []
            for metric in ["P", "LSD", "d.f.", "%CV"]:
                row = {"Treatment": metric}
                for date_label in date_labels_ordered:
                    row[f"{date_label} Mean"] = summaries[date_label][metric]
                    row[f"{date_label} Group"] = ""
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
