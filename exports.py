import pandas as pd
from io import BytesIO
import streamlit as st

def export_tables_to_excel(all_tables: dict):
    """Create an Excel file from all tables and return a download button."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for assess, table in all_tables.items():
            table.to_excel(writer, sheet_name=assess[:30], index=False)

    st.sidebar.download_button(
        "Download Tables (Excel)",
        data=buffer.getvalue(),
        file_name="assessment_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
