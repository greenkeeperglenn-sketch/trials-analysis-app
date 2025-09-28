import streamlit as st

# Page config
st.set_page_config(
    page_title="DataSynthesis by STRI",
    page_icon="🧪",
    layout="wide"
)

# Inject Montserrat font + STRI colour scheme
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap" rel="stylesheet">

    <style>
    /* Apply Montserrat font globally */
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    /* STRI Color Palette */
    :root {
        --primary: #0B6580;
        --secondary: #59B37D;
        --accent: #40B5AB;
        --dark: #004754;
    }

    /* Backgrounds */
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--dark);
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
    div[data-testid="stFileUploader"], 
    div[data-baseweb="input"], 
    div[data-baseweb="select"] {
        background:#fff !important;
        color:#000 !important;
        border:1px solid var(--accent);
        border-radius:6px;
    }
    div[data-testid="stFileUploader"] * {
        color:#000 !important;
    }
    div[data-testid="stFileUploader"] svg {
        fill:#000 !important;
    }
    input, textarea, select {
        color:#000 !important;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        color:#000 !important;
        background:#fff !important;
    }

    /* Dropdown options */
    div[data-baseweb="popover"] * {
        color:#000 !important;
    }

    /* --- Radios: teal instead of red --- */
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

    /* --- Multiselect chips: teal background, white text --- */
    div[data-baseweb="tag"], div[data-baseweb="tag"] div {
        background-color: var(--accent) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="tag"] svg {
        fill: white !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: var(--accent);
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Header with Logo ---
st.image("DataSynthesis logo.png", width=300)
st.markdown("<h4 style='text-align:center;'>Version 1.1</h4>", unsafe_allow_html=True)

# --- Demo content ---
st.title("DataSynthesis by STRI")
st.subheader("Custom Streamlit Styling Demo")

st.write("This app uses Montserrat font and STRI’s brand colours.")

if st.button("Run Analysis"):
    st.success("Analysis started... 🚀")

col1, col2 = st.columns(2)
with col1:
    st.header("Primary Colour Example")
    st.markdown('<div style="background-color:#0B6580; padding:20px; border-radius:8px; color:white;">Primary</div>', unsafe_allow_html=True)

with col2:
    st.header("Secondary Colour Example")
    st.markdown('<div style="background-color:#59B37D; padding:20px; border-radius:8px; color:white;">Secondary</div>', unsafe_allow_html=True)
