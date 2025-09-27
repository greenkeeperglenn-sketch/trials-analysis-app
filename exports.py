import os
import pandas as pd
from io import BytesIO
import streamlit as st
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Image, Spacer, PageBreak
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from PIL import Image as PILImage

# -------------------------------------------------------------------
# FOOTER DRAWING
# -------------------------------------------------------------------
def add_footer(canvas, doc, logo_path=None):
    """Draw logo + page number in footer of each page."""
    width, height = landscape(A4)

    # Page number (bottom-right)
    page_num = canvas.getPageNumber()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(width - 40, 20, f"Page {page_num}")

    # Logo (bottom-left)
    if logo_path and os.path.exists(logo_path):
        try:
            canvas.drawImage(logo_path, 40, 10, width=60, height=25, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass

# -------------------------------------------------------------------
# EXCEL EXPORT
# -------------------------------------------------------------------
def export_tables_to_excel(all_tables, logo_path=None):
    """Return a BytesIO Excel file with STRI branding."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for assess, table in all_tables.items():
            sheet_name = assess[:30]
            table.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Formats
            fmt_header = workbook.add_format({
                "bg_color": "#1f77b4", "font_color": "white",
                "bold": True, "border": 1
            })
            fmt_center = workbook.add_format({"align": "center", "border": 1})
            fmt_num = workbook.add_format({"num_format": "0.00", "border": 1})

            rows, cols = table.shape

            # Insert logo at top-left
            if logo_path and os.path.exists(logo_path):
                worksheet.insert_image("A1", logo_path, {"x_scale": 0.3, "y_scale": 0.3})

            # Header row
            for j, col in enumerate(table.columns):
                worksheet.write(2, j, col, fmt_header)

            # Data rows
            for i in range(rows):
                for j in range(cols):
                    val = table.iloc[i, j]
                    if isinstance(val, (int, float)):
                        worksheet.write(i+3, j, val, fmt_num)
                    else:
                        worksheet.write(i+3, j, val, fmt_center)

            worksheet.set_column(0, cols-1, 15)

    return buffer

# -------------------------------------------------------------------
# PDF EXPORT
# -------------------------------------------------------------------
def export_report_to_pdf(all_tables, all_figs, logo_path=None):
    """Return a BytesIO PDF report with STRI branding and charts."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=30, rightMargin=30,
        topMargin=50, bottomMargin=50
    )

    elements = []

    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="STRIHeading1", fontName="Helvetica-Bold", fontSize=18,
        textColor=colors.HexColor("#1f77b4"), spaceAfter=12, leading=22
    ))
    styles.add(ParagraphStyle(
        name="STRIHeading2", fontName="Helvetica-Bold", fontSize=14,
        textColor=colors.HexColor("#1f77b4"), spaceAfter=8, leading=18
    ))
    styles.add(ParagraphStyle(
        name="STRINormal", fontName="Helvetica", fontSize=10, leading=14
    ))

    # --- Cover page ---
    if logo_path and os.path.exists(logo_path):
        elements.append(Image(logo_path, width=300, height=120))
    elements.append(Spacer(1, 80))
    elements.append(Paragraph("Trial Assessment Report 2025", styles["STRIHeading1"]))
    elements.append(Paragraph("Prepared by STRI Group", styles["STRINormal"]))
    elements.append(PageBreak())

    # --- Assessment sections ---
    for assess, table in all_tables.items():
        elements.append(Paragraph(f"{assess} Results", styles["STRIHeading2"]))

        # Chart (make letters larger, full width, border)
        if all_figs and assess in all_figs:
            fig_bytes = BytesIO()
            try:
                # Bump font size for annotations
                all_figs[assess].update_layout(font=dict(size=16))
                all_figs[assess].write_image(fig_bytes, format="png", scale=2)
                fig_bytes.seek(0)

                chart_img = Image(fig_bytes, width=720, height=400)

                chart_table = Table([[chart_img]], colWidths=[720])
                chart_table.setStyle(TableStyle([
                    ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#1f77b4")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))

                elements.append(chart_table)
                elements.append(PageBreak())  # chart gets full page
            except Exception as e:
                elements.append(Paragraph(
                    f"(Chart for {assess} could not be exported: {e})",
                    styles["STRINormal"]
                ))
                elements.append(PageBreak())

        # Table (fit to page, compact, one per page)
        data = [table.columns.tolist()] + table.astype(str).values.tolist()
        pdf_table = Table(data, hAlign="LEFT", colWidths="*")
        pdf_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.whitesmoke, colors.lightgrey]),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ]))
        elements.append(pdf_table)
        elements.append(PageBreak())

    # Build doc with footer callback
    doc.build(elements, onFirstPage=lambda c, d: add_footer(c, d, logo_path),
              onLaterPages=lambda c, d: add_footer(c, d, logo_path))
    return buffer

# -------------------------------------------------------------------
# SIDEBAR EXPORT BUTTONS
# -------------------------------------------------------------------
def export_buttons(all_tables, all_figs, logo_path="download.jpg"):
    """Render global export buttons in the sidebar."""
    with st.sidebar:
        st.subheader("Exports")

        excel_buffer = export_tables_to_excel(all_tables, logo_path)
        st.download_button(
            "⬇️ Download Excel",
            data=excel_buffer.getvalue(),
            file_name="assessment_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        pdf_buffer = export_report_to_pdf(all_tables, all_figs, logo_path)
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_buffer.getvalue(),
            file_name="assessment_report.pdf",
            mime="application/pdf"
        )
