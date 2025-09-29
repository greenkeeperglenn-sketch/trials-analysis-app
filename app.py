# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Local modules
import charts
import stats
import data_loader
import helpers
import exports  # <-- our exports.py

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="DataSynthesis by STRI",
    page_icon="🧪",
    layout="wide"
)

# ------------------------------------------------
# Asset paths (safe)
# ------------------------------------------------
logo_path = os.path.join(os.path.dirname(__file__), "DataSynthesis logo.png")

# ------------------------------------------------
# Custom CSS Styling
# ------------------------------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap" rel="stylesheet">

    <style>
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    :root {
        --primary: #0B6580;
        --secondary: #59B37D;
        --accent: #40B5AB;
        --dark: #004754;
    }

    .stApp { background-color: #ffffff; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--dark);
        color: white;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Buttons + Download buttons */
    .stButton>button, .stDownloadButton>button {
        background:#fff !important;
        color:#000 !important;
        border:1px solid var(--accent) !important;
        border-radius:6px;
        font-weight:600;
        padding:.5em 1em;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background:var(--accent) !important;
        color:#fff !important;
    }
    .stButton>button:active, .stDownloadButton>button:active {
        background:var(--primary) !important;
        color:#fff !important;
        border-color:var(--primary) !important;
    }

    /* Inputs & uploader: white boxes, BLACK text */
    div[data-testid="stFileUploader"], div[data-baseweb="input"], div[data-baseweb="select"] {
        background:#fff !important;
        color:#000 !important;
        border:1px solid var(--accent);
        border-radius:6px;
        text-align:center !important;
    }
    input, textarea, select { color:#000 !important; }
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        color:#000 !important;
        background:#fff !important;
    }
    div[data-testid="stFileUploader"] span {
        color:#000 !important;
        text-align:center !important;
    }
    div[data-testid="stFileUploader"] button {
        color:#000 !important;
    }

    /* Dropdown options */
    div[data-baseweb="popover"] * { color:#000 !important; }

    /* Radio (kill red → teal) */
    [data-baseweb="radio"] div[role="radio"] {
        border-color: var(--accent) !important;
    }
    [data-baseweb="radio"] div[role="radio"][aria-checked="true"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    [data-baseweb="radio"] svg {
        fill: white !important;
    }

    /* Checkbox */
    div[role="checkbox"] { border-color: var(--accent) !important; }
    div[role="checkbox"][aria-checked="true"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    /* Switch */
    div[role="switch"] { border-color: var(--accent) !important; }
    div[role="switch"][aria-checked="true"] {
        background: var(--accent) !important;
    }

    /* Slider */
    [data-baseweb="slider"] [aria-valuenow] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    /* Multiselect chips */
    div[data-baseweb="tag"], div[data-baseweb="tag"] div {
        background-color: var(--accent) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="tag"] svg { fill: white !important; }

    /* Headings */
    h1,h2,h3,h4 { color: var(--accent); font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# Header with Logo + Version (stable, centered)
# ------------------------------------------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.image(logo_path, width=250)
st.markdown("<h5>Version 1.1</h5></div>", unsafe_allow_html=True)

# ------------------------------------------------
# Sidebar global settings
# ------------------------------------------------
st.sidebar.image(logo_path, use_container_width=True)
st.sidebar.markdown("<h5 style='text-align:center;'>Global Settings</h5>", unsafe_allow_html=True)

alpha_options = {
    "Fungicide (0.05)": 0.05,
    "Biologicals in lab (0.10)": 0.10,
    "Biologicals in field (0.15)": 0.15,
}
alpha_label = st.sidebar.radio("Significance level:", list(alpha_options.keys()))
alpha_choice = alpha_options[alpha_label]

# ------------------------------------------------
# Prepare containers
# ------------------------------------------------
all_tables = {}
all_figs = {}

# ------------------------------------------------
# Load data
# ------------------------------------------------
data, treatments, date_labels_ordered = data_loader.load_data()

if data is not None:
    selected_assessments = st.sidebar.multiselect(
        "Select assessments",
        sorted(set(data["Assessment"].unique()))
    )

    color_map = {
        t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, t in enumerate(treatments)
    }

    for assess in selected_assessments:
        with st.expander(f"Assessment: {assess}", expanded=False):
            st.markdown(
                f"<h2 style='text-align:center'>{assess}</h2>",
                unsafe_allow_html=True
            )

            # --- Chart Settings ---
            with st.expander("Chart Settings", expanded=True):
                chart_mode = st.radio(
                    "Chart type",
                    ["Boxplot", "Bar chart"],
                    key=helpers.safe_key("chartmode", assess)
                )
                view_mode_chart = st.radio(
                    "Grouping",
                    ["By Date", "By Treatment"],
                    key=helpers.safe_key("viewmode", assess)
                )
                a_is_lowest_chart = st.radio(
                    "Lettering convention",
                    ["Lowest = A", "Highest = A"],
                    key=helpers.safe_key("letters", assess)
                ) == "Lowest = A"

                add_se = add_lsd = add_letters = False
                if chart_mode == "Bar chart":
                    add_se = st.checkbox("Add SE error bars", value=True,
                                         key=helpers.safe_key("se", assess))
                    add_lsd = st.checkbox("Add LSD error bars", value=False,
                                          key=helpers.safe_key("lsd", assess))
                    add_letters = st.checkbox("Add statistical letters", value=True,
                                              key=helpers.safe_key("letters_check", assess))

                y_min = st.number_input("Y-axis minimum", value=0, step=1,
                                        key=helpers.safe_key("ymin", assess))
                y_max = st.number_input("Y-axis maximum", value=100, step=1,
                                        key=helpers.safe_key("ymax", assess))

            # --- Treatment filter ---
            visible_treatments = st.multiselect(
                "Show treatments",
                options=treatments,
                default=treatments,
                key=helpers.safe_key("visible_treatments", assess),
            )

            df_sub = data[data["Assessment"] == assess].copy()
            df_sub["Value"] = pd.to_numeric(df_sub["Value"], errors="coerce")
            df_sub = df_sub.dropna(subset=["Value"])
            df_sub["DateLabel"] = pd.Categorical(
                df_sub["DateLabel"],
                categories=date_labels_ordered,
                ordered=True,
            )

            # --- Chart ---
            with st.expander("Chart", expanded=True):
                if chart_mode == "Boxplot":
                    fig = charts.make_boxplot(
                        df_sub, treatments, date_labels_ordered,
                        view_mode_chart, visible_treatments, color_map
                    )
                else:
                    fig = charts.make_barchart(
                        df_sub, treatments, date_labels_ordered,
                        view_mode_chart, visible_treatments,
                        alpha_choice, a_is_lowest_chart,
                        color_map, add_se, add_lsd, add_letters
                    )

                fig.update_yaxes(range=[y_min, y_max])
                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": True})

                all_figs[assess] = fig

            # --- Statistics Table ---
            with st.expander("Statistics Table", expanded=False):
                df_stats = df_sub[df_sub["Treatment"].isin(visible_treatments)].copy()
                wide_table, styled_table = stats.build_stats_table(
                    df_stats, visible_treatments, date_labels_ordered,
                    alpha_choice, a_is_lowest_chart
                )

                st.dataframe(
                    styled_table,
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                    column_config={
                        "Treatment": st.column_config.Column("Treatment", pinned=True)
                    }
                )

                all_tables[assess] = wide_table

# ------------------------------------------------
# Global Exports (sidebar)
# ------------------------------------------------
if all_tables:
    exports.export_buttons(
        all_tables,
        all_figs,
        logo_path=logo_path,
        significance_label=alpha_label
    )
