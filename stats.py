import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from helpers import generate_cld_overlap

def build_stats_table(df_sub, treatments, date_labels_ordered, alpha_choice, a_is_lowest_chart):
    """Build wide stats table with means, letters, and summary rows."""

    wide_table = pd.DataFrame({"Treatment": treatments})
    summaries = {}

    for date_label in date_labels_ordered:
        df_date = df_sub[df_sub["DateLabel"] == date_label].copy()
        if df_date.empty:
            wide_table[f"{date_label}"] = np.nan
            wide_table[f"{date_label} S"] = ""
            summaries[date_label] = {"P": "", "LSD": "", "d.f.": "", "%CV": ""}
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

            # ✅ Conditional letters & stats
            if pd.notna(p_val) and p_val > alpha_choice:
                # Not significant → no letters
                letters = {t: "" for t in means.index}
                p_disp = "ns"
                lsd_disp = "-"
            else:
                # Significant → generate letters
                letters, _ = generate_cld_overlap(
                    means, mse, df_error, alpha_choice, rep_counts,
                    a_is_lowest=a_is_lowest_chart
                )
                p_disp = f"{p_val:.3f}" if pd.notna(p_val) else ""
                lsd_disp = f"{stats.t.ppf(1 - alpha_choice/2, df_error) * np.sqrt(2*mse/np.mean(list(rep_counts.values()))):.4f}" if pd.notna(mse) else "-"

            df_disp = f"{df_error:.0f}" if pd.notna(df_error) else ""
            cv_disp = f"{cv:.1f}" if pd.notna(cv) else ""

            wide_table[f"{date_label}"] = wide_table["Treatment"].map(means)
            wide_table[f"{date_label} S"] = wide_table["Treatment"].map(letters).fillna("")
            summaries[date_label] = {"P": p_disp, "LSD": lsd_disp, "d.f.": df_disp, "%CV": cv_disp}

        else:
            wide_table[f"{date_label}"] = np.nan
            wide_table[f"{date_label} S"] = ""
            summaries[date_label] = {"P": "", "LSD": "", "d.f.": "", "%CV": ""}

    # Add summary rows
    summary_rows = []
    for metric in ["P", "LSD", "d.f.", "%CV"]:
        row = {"Treatment": metric}
        for date_label in date_labels_ordered:
            row[f"{date_label}"] = summaries[date_label][metric]
            row[f"{date_label} S"] = ""
        summary_rows.append(row)

    wide_table = pd.concat([wide_table, pd.DataFrame(summary_rows)], ignore_index=True)

    # ✅ Styling (gray treatment col + summary rows)
    def style_table(df):
        styled = df.style.applymap(
            lambda _: "background-color: lightgray", subset=["Treatment"]
        )
        styled = styled.apply(
            lambda row: [
                "background-color: lightgray"
                if row["Treatment"] in ["P", "LSD", "d.f.", "%CV"] else ""
                for _ in row
            ],
            axis=1,
        )
        return styled

    return wide_table, style_table(wide_table)
