import streamlit as st
import pandas as pd
import plotly.express as px

from helpers import safe_key
from data_loader import load_data
from charts import make_boxplot, make_barchart
from stats import build_stats_table
from exports import export_tables_to_excel

st.set_page_config(layout="wide")
st.title("Assessment Data Explorer")

# ----------------------
# Sidebar global settings
# ----------------------
st.sidebar.header("Global Settings")
alpha_options = {
    "Fungicide (0.05)": 0.05,
    "Biologicals in lab (0.10)": 0.10,
    "Biologicals in field (0.15)": 0.15,
}
alpha_label = st.sidebar.radio("Significance level:", list(alpha_options.keys()))
alpha_choice = alpha_options[alpha_label]

# ----------------------
# Load data
# ----------------------
data, treatments, date_labels_ordered = load_data()

if data is not None:
    selected_assessments = st.sidebar.multiselect(
        "Select assessments",
        sorted(set(data["Assessment"].unique()))
    )

    color_map = {
        t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, t in enumerate(treatments)
    }

    all_tables = {}

    # ----------------------
    # Loop through assessments
    # ----------------------
    for assess in selected_assessments:
        with st.expander(f"Assessment: {assess}", expanded=False):
            st.markdown(
                f"<h2 style='text-align:center'>{assess}</h2>",
                unsafe_allow_html=True
            )

            # Per-chart settings
            chart_mode = st.radio(
                "Chart type",
                ["Boxplot", "Bar chart"],
                key=safe_key("chartmode", assess)
            )
            view_mode_chart = st.radio(
                "Grouping",
                ["By Date", "By Treatment"],
                key=safe_key("viewmode", assess)
            )
            a_is_lowest_chart = st.radio(
                "Lettering convention",
                ["Lowest = A", "Highest = A"],
                key=safe_key("letters", assess)
            ) == "Lowest = A"

            # Treatment filter
            visible_treatments = st.multiselect(
                "Show treatments",
                options=treatments,
                default=treatments,
                key=safe_key("visible_treatments", assess),
            )

            # Subset data
            df_sub = data[data["Assessment"] == assess].copy()
            df_sub["Value"] = pd.to_numeric(df_sub["Value"], errors="coerce")
            df_sub = df_sub.dropna(subset=["Value"])
            df_sub["DateLabel"] = pd.Categorical(
                df_sub["DateLabel"],
                categories=date_labels_ordered,
                ordered=True,
            )

            # Chart
            if chart_mode == "Boxplot":
                fig = make_boxplot(
                    df_sub, treatments, date_labels_ordered,
                    view_mode_chart, visible_treatments, color_map
                )
            else:
                fig = make_barchart(
                    df_sub, treatments, date_labels_ordered,
                    view_mode_chart, visible_treatments,
                    alpha_choice, a_is_lowest_chart, color_map
                )

            st.plotly_chart(fig, use_container_width=True)

            # Stats table
            wide_table = build_stats_table(
                df_sub, treatments, date_labels_ordered,
                alpha_choice, a_is_lowest_chart
            )
            st.dataframe(wide_table, use_container_width=True, hide_index=True)
            all_tables[assess] = wide_table

    # ----------------------
    # Exports
    # ----------------------
    export_tables_to_excel(all_tables)
