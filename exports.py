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
from reportlab.pdfbase.pdfmetrics import stringWidth
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Brand Colours
# -------------------------------------------------------------------
PRIMARY = colors.HexColor("#0B6580")     # deep blue
SECONDARY = colors.HexColor("#59B37D")  # green
ACCENT = colors.HexColor("#40B5AB")     # aqua (border, grid)
DARK = colors.HexColor("#004754")       # dark teal
NEUTRAL = colors.HexColor("#EAEFF2")    # light neutral for background if needed

# -------------------------------------------------------------------
# EXCEL EXPORT
# -------------------------------------------------------------------
def export_tables_to_excel(all_tables, logo_path=None):
    """Return a BytesIO Excel file with DataSynthesis branding."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for assess, table in all_tables.items():
            sheet_name = assess[:30]
            table.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Formats
            fmt_header = workbook.add_format({
                "bg_color": "#0B6580", "font_color": "white",
                "bold": True, "border": 1, "align": "center"
            })
            fmt_center = workbook.add_format({"align": "center", "border": 1, "border_color": "#40B5AB"})
            fmt_num = workbook.add_format({"num_format": "0.00", "border": 1, "border_color": "#40B5AB"})
            fmt_special = workbook.add_format({"bg_color": "#59B37D", "font_color": "white", "border": 1, "align": "center"})

            rows, cols = table.shape

            # Insert logo at top-left
            if logo_path and os.path.exists(logo_path):
                worksheet.insert_image("A1", logo_path,
                                       {"x_scale": 0.3, "y_scale": 0.3})

            # Header row
            for j, col in enumerate(table.columns):
                worksheet.write(2, j, col, fmt_header)

            # Data rows
            for i in range(rows):
                row_label = str(table.iloc[i, 0])
                for j in range(cols):
                    val = table.iloc[i, j]
                    fmt = fmt_num if isinstance(val, (int, float)) else fmt_center
                    # Special row shading for PLSD, D.F., %CV
                    if any(key in row_label for key in ["PLSD", "D.F.", "%CV"]):
                        fmt = fmt_special
                    worksheet.write(i+3, j, val, fmt)

            worksheet.set_column(0, cols-1, 15)

    return buffer

# -------------------------------------------------------------------
# PDF EXPORT
# -------------------------------------------------------------------
def export_report_to_pdf(all_tables, all_figs, logo_path="DataSynthesis LOGO.png"):
    """Return a BytesIO PDF report with DataSynthesis v1.1 branding and charts."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=50, rightMargin=50,
        topMargin=70, bottomMargin=70
    )

    elements = []

    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="DSHeading1", fontName="Helvetica-Bold", fontSize=20,
        textColor=PRIMARY, spaceAfter=14, leading=24
    ))
    styles.add(ParagraphStyle(
        name="DSHeading2", fontName="Helvetica-Bold", fontSize=14,
        textColor=PRIMARY, spaceAfter=10, leading=18
    ))
    styles.add(ParagraphStyle(
        name="DSNormal", fontName="Helvetica", fontSize=10, leading=14
    ))

    # --- Cover Page ---
    if logo_path and os.path.exists(logo_path):
        elements.append(Image(logo_path, width=320, height=130))
    elements.append(Spacer(1, 50))
    elements.append(Paragraph("Trial Assessment Report 2025", styles["DSHeading1"]))
    elements.append(Paragraph("Prepared by STRI Group", styles["DSNormal"]))
    elements.append(Paragraph("DataSynthesis v1.1", styles["DSNormal"]))
    elements.append(PageBreak())

    # --- Index Page ---
    elements.append(Paragraph("Index", styles["DSHeading1"]))
    toc_data = [["Assessment", "Chart Page", "Table Page"]]
    page_counter = 2  # cover=1, index=2
    for assess in all_tables.keys():
        toc_data.append([assess, str(page_counter+1), str(page_counter+2)])
        page_counter += 2
    toc_table = Table(toc_data, hAlign="LEFT")
    toc_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, ACCENT),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(toc_table)
    elements.append(PageBreak())

    # --- Helpers for tables ---
    def _merge_stat_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        def base_of_stat_col(col: str):
            if col.endswith(" S"): return col[:-2]
            if col.endswith("_S"): return col[:-2]
            if col.endswith(" (S)"): return col[:-4]
            if col.endswith("(S)"): return col[:-3]
            if col.endswith("S") and col[:-1] in df.columns: return col[:-1]
            return None
        pairs = {}
        for c in df.columns:
            base = base_of_stat_col(c)
            if base and base in df.columns:
                pairs[base] = c
        for base, scol in pairs.items():
            for i in range(len(df)):
                val = df.at[i, base]
                letter = df.at[i, scol]
                if pd.notna(letter) and str(letter).strip():
                    if isinstance(val, (int, float)):
                        df.at[i, base] = f"{val:.2f} {letter}"
                    else:
                        df.at[i, base] = f"{val} {letter}"
        df = df.drop(columns=list(pairs.values()), errors="ignore")
        return df

    def _first_col_width(df: pd.DataFrame) -> float:
        texts = [str(df.columns[0])] + [str(x) for x in df.iloc[:, 0].tolist()]
        width_pts = max(stringWidth(t, "Helvetica", 8) for t in texts) + 18
        return max(140, min(width_pts, 320))

    # --- Assessments ---
    for assess, table in all_tables.items():
        # Chart Page
        elements.append(Paragraph(f"{assess} Chart", styles["DSHeading2"]))
        if all_figs and assess in all_figs:
            fig_bytes = BytesIO()
            try:
                fig = go.Figure(all_figs[assess])  # keep colours
                # Remove "Date" axis title
                fig.update_xaxes(title_text=None)
                # Axis text in dark teal
                fig.update_xaxes(tickfont=dict(color="#004754"))
                fig.update_yaxes(tickfont=dict(color="#004754"))
                # Legend below
                fig.update_layout(
                    margin=dict(l=90, r=50, t=80, b=160),
                    legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center"),
                    font=dict(size=16)
                )
                # Stat letters bump
                if fig.layout.annotations:
                    for ann in fig.layout.annotations:
                        ann.update(
                            font=dict(size=max(getattr(ann.font, "size", 12), 20), color="black"),
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#0B6580",
                            yshift=12
                        )
                fig.write_image(fig_bytes, format="png", scale=2)
                fig_bytes.seek(0)
                chart_img = Image(fig_bytes, width=720, height=420)
                chart_table = Table([[chart_img]], colWidths=[720])
                chart_table.setStyle(TableStyle([
                    ("BOX", (0, 0), (-1, -1), 2, ACCENT),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                elements.append(chart_table)
                elements.append(PageBreak())
            except Exception as e:
                elements.append(Paragraph(f"(Chart error: {e})", styles["DSNormal"]))
                elements.append(PageBreak())

        # Table Page
        elements.append(Paragraph(f"{assess} Table", styles["DSHeading2"]))
        df = _merge_stat_columns(table)
        data = [df.columns.tolist()] + df.astype(str).values.tolist()
        first_w = _first_col_width(df)
        col_widths = [first_w] + [45] * (len(df.columns) - 1)
        pdf_table = Table(data, hAlign="LEFT", colWidths=col_widths)
        table_style = [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, ACCENT),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("ROTATE", (1, 0), (-1, 0), 90),
        ]
        # Special row shading
        for i, label in enumerate(df.iloc[:, 0], start=1):
            if any(key in str(label) for key in ["PLSD", "D.F.", "%CV"]):
                table_style += [
                    ("BACKGROUND", (0, i), (-1, i), SECONDARY),
                    ("TEXTCOLOR", (0, i), (-1, i), colors.white),
                ]
        pdf_table.setStyle(TableStyle(table_style))
        elements.append(pdf_table)
        elements.append(PageBreak())

    # --- Footer with border ---
    def _footer(c, d, logo_path_inner):
        width, height = landscape(A4)
        # Accent border
        c.setStrokeColor(ACCENT)
        c.setLineWidth(3.5)
        c.roundRect(20, 20, width-40, height-40, radius=18, stroke=1, fill=0)
        # Footer text
        c.setFont("Helvetica", 9)
        c.setFillColor(DARK)
        page_num = c.getPageNumber()
        c.drawRightString(width - 50, 28, f"DataSynthesis v1.1 – Page {page_num}")
        # Logo
        if logo_path_inner and os.path.exists(logo_path_inner):
            try:
                c.drawImage(logo_path_inner, 35, 18, width=60, height=25,
                            preserveAspectRatio=True, mask="auto")
            except Exception:
                pass

    doc.build(elements,
              onFirstPage=lambda c, d: _footer(c, d, logo_path),
              onLaterPages=lambda c, d: _footer(c, d, logo_path))
    return buffer

# -------------------------------------------------------------------
# SIDEBAR EXPORT BUTTONS
# -------------------------------------------------------------------
def export_buttons(all_tables, all_figs, logo_path="DataSynthesis LOGO.png"):
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
