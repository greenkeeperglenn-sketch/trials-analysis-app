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

    fig.update_traces(boxpoints=False)
    return fig


def make_barchart(df_sub, treatments, date_labels_ordered, view_mode_chart,
                  visible_treatments, alpha_choice, a_is_lowest_chart, color_map,
                  add_se=True, add_lsd=True, add_letters=True):
    """Generate a bar chart with optional SE/LSD error bars and statistical letters."""

    df_plot = df_sub[df_sub["Treatment"].isin(visible_treatments)].copy()
    df_plot["DateLabel"] = pd.Categorical(df_plot["DateLabel"], categories=date_labels_ordered, ordered=True)

    fig = go.Figure()
    letters_dict = {}

    if view_mode_chart == "By Date":
        grouping = ["DateLabel", "Treatment"]
    else:  # By Treatment
        grouping = ["Treatment", "DateLabel"]

    # Aggregate means
    means = df_plot.groupby(grouping)["Value"].mean().reset_index()
    rep_counts = df_plot.groupby(grouping).size().reset_index(name="n")

    merged = means.merge(rep_counts, on=grouping)

    # Add SE
    if add_se:
        df_se = df_plot.groupby(grouping)["Value"].agg(["mean", "std", "count"]).reset_index()
        df_se["se"] = df_se["std"] / np.sqrt(df_se["count"])
        merged = merged.merge(df_se[grouping + ["se"]], on=grouping, how="left")

    # LSD + letters
    if add_lsd or add_letters:
        for key in merged[grouping[0]].unique():
            if grouping[0] == "DateLabel":
                df_group = df_plot[df_plot["DateLabel"] == key]
            else:
                df_group = df_plot[df_plot["Treatment"] == key]

            if df_group["Treatment"].nunique() > 1 and len(df_group) > 1:
                try:
                    if "Block" in df_group.columns:
                        model = ols("Value ~ C(Treatment) + C(Block)", data=df_group).fit()
                    else:
                        model = ols("Value ~ C(Treatment)", data=df_group).fit()

                    anova = sm.stats.anova_lm(model, typ=2)
                    df_error = float(model.df_resid)
                    mse = float(anova.loc["Residual", "sum_sq"] / df_error)

                    means_group = df_group.groupby("Treatment")["Value"].mean()
                    rep_counts_group = df_group["Treatment"].value_counts().to_dict()

                    letters, _ = generate_cld_overlap(
                        means_group, mse, df_error, alpha_choice, rep_counts_group,
                        a_is_lowest=a_is_lowest_chart
                    )

                    if add_letters:
                        letters_dict[key] = letters

                    if add_lsd:
                        n_avg = np.mean(list(rep_counts_group.values()))
                        lsd_val = stats.t.ppf(1 - alpha_choice/2, df_error) * np.sqrt(2*mse/n_avg)
                        merged.loc[merged[grouping[0]] == key, "LSD"] = lsd_val
                except Exception:
                    continue

    # Plot bars
    y_max = df_plot["Value"].max()
    offset = 0.05 * y_max if pd.notna(y_max) else 1

    for t in treatments:
        df_t = merged[merged["Treatment"] == t]
        if df_t.empty:
            continue

        error_y = None
        if add_se and "se" in df_t.columns:
            error_y = dict(type="data", array=df_t["se"], visible=True)
        elif add_lsd and "LSD" in df_t.columns:
            error_y = dict(type="constant", value=df_t["LSD"].iloc[0], visible=True)

        fig.add_trace(go.Bar(
            x=df_t[grouping[0]],
            y=df_t["Value"],
            name=t,
            marker_color=color_map.get(t, None),
            error_y=error_y
        ))

        # Add letters
        if add_letters and grouping[0] in letters_dict:
            for _, row in df_t.iterrows():
                letter = letters_dict.get(row[grouping[0]], {}).get(t, "")
                if letter:
                    fig.add_annotation(
                        x=row[grouping[0]],
                        y=row["Value"] + offset,
                        text=letter,
                        showarrow=False,
                        yanchor="bottom",
                        xanchor="center",
                        font=dict(color="black", size=12)
                    )

    fig.update_layout(barmode="group")
    return fig
