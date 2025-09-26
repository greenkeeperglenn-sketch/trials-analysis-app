import pandas as pd
from io import BytesIO
import streamlit as st

def export_tables_to_excel(all_tables):
    """Export stats tables to Excel with formatting."""

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for assess, table in all_tables.items():
            sheet_name = assess[:30]  # Excel sheet name limit
            table.to_excel(writer, sheet_name=sheet_name, index=False)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # --- Formats ---
            fmt_grey = workbook.add_format({"bg_color": "#D9D9D9", "border": 1})
            fmt_center = workbook.add_format({"align": "center", "border": 1})
            fmt_num_2dp = workbook.add_format({"num_format": "0.00", "border": 1})
            fmt_num_3dp = workbook.add_format({"num_format": "0.000", "border": 1})
            fmt_num_4dp = workbook.add_format({"num_format": "0.0000", "border": 1})
            fmt_num_1dp = workbook.add_format({"num_format": "0.0", "border": 1})
            fmt_int = workbook.add_format({"num_format": "0", "border": 1})

            # Summary row format (grey + bold border + right align)
            fmt_summary = workbook.add_format({
                "bg_color": "#D9D9D9",
                "align": "right",
                "border": 2,   # thicker border
                "bold": True
            })

            rows, cols = table.shape

            # Header row grey with border
            worksheet.set_row(0, None, fmt_grey)

            # Grey background for Treatment column + autofit width
            worksheet.set_column(0, 0, max(15, max(table["Treatment"].astype(str).map(len))) + 2, fmt_grey)

            # --- Write values with proper formatting ---
            for j, col in enumerate(table.columns):
                if col == "Treatment":
                    continue
                if col.endswith(" S"):  # stats letters column
                    worksheet.set_column(j, j, 6, fmt_center)
                    continue

                for i in range(1, rows+1):  # +1 for header row
                    val = table.iloc[i-1, j]
                    treat_name = str(table.iloc[i-1, 0])

                    if treat_name == "P":
                        worksheet.write(i, j, None if val == "" else val, fmt_num_3dp)
                    elif treat_name == "LSD":
                        worksheet.write(i, j, None if val in ["", "-"] else val, fmt_num_4dp)
                    elif treat_name == "d.f.":
                        worksheet.write(i, j, None if val == "" else val, fmt_int)
                    elif treat_name == "%CV":
                        worksheet.write(i, j, None if val == "" else val, fmt_num_1dp)
                    else:  # Treatment rows
                        worksheet.write(i, j, None if val == "" else val, fmt_num_2dp)

            # --- Summary rows with bold border + right alignment ---
            for i in range(1, rows+1):
                treat_name = str(table.iloc[i-1, 0])
                if treat_name in ["P", "LSD", "d.f.", "%CV"]:
                    for j in range(cols):
                        worksheet.write(i, j, table.iloc[i-1, j], fmt_summary)

            # --- Auto-fit numeric/date columns (but not S columns) ---
            for j, col in enumerate(table.columns):
                if col == "Treatment" or col.endswith(" S"):
                    continue
                max_len = max(
                    [len(str(x)) for x in table[col].astype(str).tolist()] + [len(col)]
                )
                worksheet.set_column(j, j, max(max_len + 2, 10))  # at least width 10

    # Download button
    st.download_button(
        "Download Tables (Excel)",
        data=buffer.getvalue(),
        file_name="assessment_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
