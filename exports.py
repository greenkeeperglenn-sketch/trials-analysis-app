import pandas as pd
from io import BytesIO
import streamlit as st

def export_tables_to_excel(all_tables):
    """Export stats tables to Excel with formatting."""

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for assess, table in all_tables.items():
            sheet_name = assess[:30]  # Excel sheet name limit
            table.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # --- Formats ---
            fmt_header = workbook.add_format({"bg_color": "#D9D9D9", "border": 1, "bold": True})
            fmt_treat = workbook.add_format({"bg_color": "#D9D9D9", "border": 1})
            fmt_center = workbook.add_format({"align": "center", "border": 1})
            fmt_num_2dp = workbook.add_format({"num_format": "0.00", "border": 1})
            fmt_num_3dp = workbook.add_format({"num_format": "0.000", "border": 1})
            fmt_num_4dp = workbook.add_format({"num_format": "0.0000", "border": 1})
            fmt_num_1dp = workbook.add_format({"num_format": "0.0", "border": 1})
            fmt_int = workbook.add_format({"num_format": "0", "border": 1})

            fmt_summary = workbook.add_format({
                "bg_color": "#D9D9D9",
                "align": "right",
                "border": 2,
                "bold": True
            })

            rows, cols = table.shape

            # --- Grey header row ---
            for j, col in enumerate(table.columns):
                worksheet.write(0, j, col, fmt_header)

            # --- Apply formatting row by row ---
            for i in range(1, rows+1):  # +1 because header is row 0
                treat_name = str(table.iloc[i-1, 0])

                for j, col in enumerate(table.columns):
                    val = table.iloc[i-1, j]

                    if j == 0:  # Treatment column
                        fmt = fmt_treat
                        worksheet.write(i, j, val, fmt)
                        continue

                    if col.endswith(" S"):  # Letters column
                        fmt = fmt_center
                        worksheet.write(i, j, "" if val == "" else str(val), fmt)
                        continue

                    # Summary rows
                    if treat_name == "P":
                        worksheet.write(i, j, None if val in ["", "ns"] else float(val), fmt_num_3dp)
                    elif treat_name == "LSD":
                        worksheet.write(i, j, None if val in ["", "-"] else float(val), fmt_num_4dp)
                    elif treat_name == "d.f.":
                        worksheet.write(i, j, None if val == "" else int(float(val)), fmt_int)
                    elif treat_name == "%CV":
                        worksheet.write(i, j, None if val == "" else float(val), fmt_num_1dp)
                    else:
                        # Treatment means
                        worksheet.write(i, j, None if val == "" else float(val), fmt_num_2dp)

                # If it's a summary row, reapply bold grey format
                if treat_name in ["P", "LSD", "d.f.", "%CV"]:
                    for j in range(cols):
                        worksheet.write(i, j, table.iloc[i-1, j], fmt_summary)

            # --- AutoFit column widths ---
            for j, col in enumerate(table.columns):
                col_data = [str(x) for x in table.iloc[:, j]]
                max_len = max([len(str(col))] + [len(x) for x in col_data if x != ""])
                worksheet.set_column(j, j, max_len + 2)

    # Download button
    st.download_button(
        "Download Tables (Excel)",
        data=buffer.getvalue(),
        file_name="assessment_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
