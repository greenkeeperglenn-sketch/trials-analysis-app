import pandas as pd
from io import BytesIO
import streamlit as st
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Image, Spacer, PageBreak)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from PIL import Image as PILImage, ImageDraw

# --- Register Montserrat font (ensure Montserrat-Regular.ttf is in your project folder) ---
pdfmetrics.registerFont(TTFont("Montserrat", "Montserrat-Regular.ttf"))

# --- Helper: Round corners on chart images ---
def round_corners(img_bytes, radius=25, bg_color="#E6F0FA"):
    img = PILImage.open(img_bytes).convert("RGBA")
    mask = PILImage.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), img.size], radius=radius, fill=255)

    rounded = PILImage.new("RGBA", img.size, bg_color)
    rounded.paste(img, (0, 0), mask=mask)

    out_bytes = BytesIO()
    rounded.save(out_bytes, format="PNG")
    out_bytes.seek(0)
    return out_bytes

# --- Footer callback for PDF ---
def add_footer(canvas_obj, doc, logo_path):
    canvas_obj.saveState()
    if logo_path:
        canvas_obj.drawImage(logo_path, x=40, y=20, width=60, height=25, preserveAspectRatio=True, mask='auto')
    page_num = canvas_obj.getPageNumber()
    canvas_obj.setFont("Montserrat", 8)
    canvas_obj.setFillColor(colors.grey)
    canvas_obj.drawRightString(A4[0]-40, 30, f"Page {page_num}")
    canvas_obj.restoreState()

# --- Excel Export ---
def export_tables_to_excel(all_tables, logo_path=None):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for assess, table in all_tables.items():
            sheet_name = assess[:30]
            table.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # STRI light blue branding
            fmt_header = workbook.add_format({"bg_color": "#1f77b4", "font_color": "white", "bold": True, "border": 1})
            fmt_center = workbook.add_format({"align": "center", "border": 1})
            fmt_num = workbook.add_format({"num_format": "0.00", "border": 1})

            rows, cols = table.shape

            # Insert STRI logo at top if provided
            if logo_path:
                worksheet.insert_image("A1", logo_path, {"x_scale": 0.3, "y_scale": 0.3})

            # Header formatting
            for j, col in enumerate(table.columns):
                worksheet.write(2, j, col, fmt_header)

            # Data formatting
            for i in range(rows):
                for j in range(cols):
                    val = table.iloc[i, j]
                    if isinstance(val, (int, float)):
                        worksheet.write(i+3, j, val, fmt_num)
                    else:
                        worksheet.write(i+3, j, val, fmt_center)

            worksheet.set_column(0, cols-1, 15)

    return buffer

# --- PDF Export ---
def export_report_to_pdf(all_tables, all_figs, logo_path):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=50, bottomMargin=50)

    elements = []

    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Heading1",
                              fontName="Montserrat",
                              fontSize=18,
                              textColor=colors.HexColor("#1f77b4"),
                              spaceAfter=12,
                              leading=22))
    styles.add(ParagraphStyle(name="Heading2",
                              fontName="Montserrat",
                              fontSize=14,
                              textColor=colors.HexColor("#1f77b4"),
                              spaceAfter=8,
                              leading=18))
    styles.add(ParagraphStyle(name="Normal",
                              fontName="Montserrat",
                              fontSize=10,
                              leading=14))

    # --- Cover Page ---
    if logo_path:
        elements.append(Image(logo_path, width=250, height=100))
    elements.append(Spacer(1, 60))
    elements.append(Paragraph("Trial Assessment Report 2025", styles["Heading1"]))
    elements.append(Paragraph("Prepared by STRI Group", styles["Normal"]))
    elements.append(Spacer(1, 300))
    elements.append(Paragraph("Confidential – For internal use only", styles["Normal"]))
    elements.append(PageBreak())

    # --- Executive Summary ---
    elements.append(Paragraph("Executive Summary", styles["Heading1"]))
    elements.append(Paragraph(
        "This section provides a concise overview of key findings and insights. "
        "Replace this text with bullet points or summary narrative as appropriate.",
        styles["Normal"]
    ))
    elements.append(PageBreak())

    # --- Assessment Sections ---
    for assess, table in all_tables.items():
        elements.append(Paragraph(f"{assess} Results", styles["Heading2"]))

        # Chart
        if assess in all_figs:
            fig_bytes = BytesIO()
            all_figs[assess].savefig(fig_bytes, format="png", bbox_inches="tight")
            fig_bytes.seek(0)
            rounded_bytes = round_corners(fig_bytes, radius=25, bg_color="#E6F0FA")
            elements.append(Image(rounded_bytes, width=400, height=250))
            elements.append(Paragraph("Figure: Mean values with statistical groupings", styles["Normal"]))
            elements.append(Spacer(1, 12))

        # Table
        data = [table.columns.tolist()] + table.astype(str).values.tolist()
        pdf_table = Table(data, hAlign="LEFT")
        pdf_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Montserrat"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(pdf_table)
        elements.append(Spacer(1, 24))
        elements.append(PageBreak())

    # Build PDF
    doc.build(elements,
              onLaterPages=lambda c, d: add_footer(c, d, logo_path),
              onFirstPage=lambda c, d: add_footer(c, d, logo_path))

    return buffer

# --- Streamlit UI (put in app.py or main page) ---
def export_buttons(all_tables, all_figs, logo_path="stri_logo.png"):
    st.sidebar.header("Export Options")

    # Excel
    excel_buffer = export_tables_to_excel(all_tables, logo_path)
    st.sidebar.download_button(
        "⬇️ Download Tables (Excel)",
        data=excel_buffer.getvalue(),
        file_name="assessment_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # PDF
    pdf_buffer = export_report_to_pdf(all_tables, all_figs, logo_path)
    st.sidebar.download_button(
        "⬇️ Download Report (PDF)",
        data=pdf_buffer.getvalue(),
        file_name="assessment_report.pdf",
        mime="application/pdf"
    )
