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
