import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from helpers import generate_cld_overlap


def make_boxplot(df_sub, treatments, date_labels_ordered, view_mode_chart, visible_treatments, color_map):
    """Generate a boxplot grouped by Date or Treatment."""
    df_plot = df_sub[df_sub["Treatment"].isin(visible_treatments)].copy()

    if view_mode_chart == "By Date":
        fig = px.box(
            df_plot,
            x="DateLabel", y="Value",
            color="Treatment",
            color_discrete_map=color_map,
            category_orders={"DateLabel": date_labels_ordered, "Treatment": treatments}
        )
    else:  # By Treatment
        fig = px.box(
            df_plot,
            x="Treatment", y="Value",
            color="DateLabel",
            category_orders={"Treatment": treatments, "DateLabel": date_labels_ordered}
        )
        fig.update_layout(boxmode="group")  # ✅ prevents overlapping

    fig.update_traces(boxpoints=False)
    return fig


def make_barchart(df_sub, treatments, date_labels_ordered,
                  view_mode_chart, visible_treatments,
                  alpha_choice, a_is_lowest_chart,
                  color_map, add_se, add_lsd, add_letters):
    """Generate a bar chart with optional SE, LSD, and letters."""

    df_plot = df_sub[df_sub["Treatment"].isin(visible_treatments)].copy()

    # Compute summary stats
    means = df_plot.groupby(["DateLabel", "Treatment"])["Value"].mean().reset_index()
    rep_counts = df_plot.groupby(["DateLabel", "Treatment"]).size().reset_index(name="n")
    merged = means.merge(rep_counts, on=["DateLabel", "Treatment"])

    letters_dict = {}

    if add_se or add_lsd or add_letters:
        for date_label in date_labels_ordered:
            df_date = df_plot[df_plot["DateLabel"] == date_label]
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
                        a_is_lowest=a_is_lowest_chart
                    )
                    letters_dict[date_label] = letters

                    if add_lsd:
                        n_avg = np.mean(list(rep_counts_date.values()))
                        lsd_val = stats.t.ppf(1 - alpha_choice/2, df_error) * np.sqrt(2*mse/n_avg)
                        merged.loc[merged["DateLabel"] == date_label, "LSD"] = lsd_val
                except Exception:
                    continue

    if add_se:
        df_se = df_plot.groupby(["DateLabel", "Treatment"])["Value"].agg(["mean", "std", "count"]).reset_index()
        df_se["se"] = df_se["std"] / np.sqrt(df_se["count"])
        merged = merged.merge(df_se[["DateLabel", "Treatment", "se"]],
                              on=["DateLabel", "Treatment"], how="left")

    # Build the chart
    fig = go.Figure()

    if view_mode_chart == "By Date":
        for t in visible_treatments:
            df_t = merged[merged["Treatment"] == t]
            if df_t.empty:
                continue

            error_y = None
            if add_se and "se" in df_t.columns:
                error_y = dict(type="data", array=df_t["se"], visible=True)
            elif add_lsd and "LSD" in df_t.columns:
                error_y = dict(type="constant", value=df_t["LSD"].iloc[0], visible=True)

            # Offset for letters (5% of value or fixed 0.5 if bar is very small)
            text_vals = []
            for _, row in df_t.iterrows():
                letter = letters_dict.get(row["DateLabel"], {}).get(t, "") if add_letters else ""
                if letter:
                    offset_val = max(row["Value"] * 0.05, 0.5)
                    text_vals.append(letter)
                else:
                    text_vals.append("")
            
            fig.add_trace(go.Bar(
                x=df_t["DateLabel"],
                y=df_t["Value"],
                name=t,
                marker_color=color_map.get(t, None),
                error_y=error_y,
                text=text_vals,
                textposition="outside",
                textfont=dict(color="black", size=12)
            ))

    else:  # By Treatment
        for d in date_labels_ordered:
            df_d = merged[merged["DateLabel"] == d]
            if df_d.empty:
                continue

            error_y = None
            if add_se and "se" in df_d.columns:
                error_y = dict(type="data", array=df_d["se"], visible=True)
            elif add_lsd and "LSD" in df_d.columns:
                error_y = dict(type="constant", value=df_d["LSD"].iloc[0], visible=True)

            text_vals = []
            for _, row in df_d.iterrows():
                letter = letters_dict.get(d, {}).get(row["Treatment"], "") if add_letters else ""
                if letter:
                    offset_val = max(row["Value"] * 0.05, 0.5)
                    text_vals.append(letter)
                else:
                    text_vals.append("")

            fig.add_trace(go.Bar(
                x=df_d["Treatment"],
                y=df_d["Value"],
                name=d,
                error_y=error_y,
                text=text_vals,
                textposition="outside",
                textfont=dict(color="black", size=12)
            ))

    fig.update_layout(barmode="group")
    return fig
