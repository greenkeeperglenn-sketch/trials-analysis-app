import os
import re
from io import BytesIO
from datetime import datetime
import pandas as pd
import numpy as np

import streamlit as st

from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer, PageBreak
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth

import plotly.graph_objects as go


# =========================================
# Brand Colours
# =========================================
PRIMARY   = colors.HexColor("#0B6580")
SECONDARY = colors.HexColor("#59B37D")
ACCENT    = colors.HexColor("#40B5AB")
DARK      = colors.HexColor("#004754")


# =========================================
# Helpers
# =========================================
def _is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating, np.number)) and pd.notna(x)

def _detect_stat_base(col_name: str, existing_cols: list[str]) -> str | None:
    if col_name.endswith(" S"):
        base = col_name[:-2]
    elif col_name.endswith("_S"):
        base = col_name[:-2]
    elif col_name.endswith(" (S)"):
        base = col_name[:-4]
    elif col_name.endswith("(S)"):
        base = col_name[:-3]
    elif col_name.endswith("S") and col_name[:-1] in existing_cols:
        base = col_name[:-1]
    else:
        base = None
    return base if base in existing_cols else None

def _set_fixed_widths_xlsxwriter(df, worksheet):
    for j, col in enumerate(df.columns):
        if j == 0:
            worksheet.set_column(j, j, 20)   # Treatment names
        elif col.endswith("S") or col.endswith("_S") or "(S)" in col:
            worksheet.set_column(j, j, 4)    # stats letters
        else:
            worksheet.set_column(j, j, 10)   # numbers


# =========================================
# EXCEL EXPORT
# =========================================
def export_tables_to_excel(all_tables: dict[str, pd.DataFrame], logo_path=None) -> BytesIO:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for assess, table in all_tables.items():
            df = table.copy()

            # Drop completely empty columns
            df = df.dropna(axis=1, how="all")

            cols = list(df.columns)
            ordered, used = [], set()

            # Order columns (base + stats col)
            for c in cols:
                if c in used:
                    continue
                base = _detect_stat_base(c, cols)
                if base:
                    continue
                ordered.append(c); used.add(c)
                s_candidates = [sc for sc in cols if _detect_stat_base(sc, cols) == c]
                if s_candidates:
                    s_col = s_candidates[0]
                    col_series = df[s_col].astype(str).str.strip().fillna("")
                    if col_series.replace("nan", "").str.strip().eq("").all():
                        df = df.drop(columns=[s_col])
                        continue
                    ordered.append(s_col); used.add(s_col)

            for c in cols:
                if c not in used:
                    ordered.append(c); used.add(c)

            df = df.reindex(columns=ordered)

            sheet_name = str(assess)[:30] if assess else "Sheet1"
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
            workbook  = writer.book
            worksheet = writer.sheets[sheet_name]

            # Formats
            fmt_header = workbook.add_format({
                "bg_color": "#0B6580", "font_color": "white",
                "bold": True, "border": 1, "align": "center", "valign": "vcenter",
                "border_color": "#40B5AB", "rotation": 90
            })
            fmt_text = workbook.add_format({
                "align": "center", "valign": "vcenter",
                "border": 1, "border_color": "#40B5AB"
            })
            fmt_num = workbook.add_format({
                "border": 1, "border_color": "#40B5AB"
            })
            fmt_special = workbook.add_format({
                "bg_color": "#59B37D", "font_color": "white",
                "align": "center", "valign": "vcenter",
                "border": 1, "border_color": "white"
            })

            rows, cols_n = df.shape

            # Header row
            for j, col in enumerate(df.columns):
                worksheet.write(2, j, col, fmt_header)

            # Data rows
            for i in range(rows):
                is_bottom_4 = (rows - i) <= 4
                for j in range(cols_n):
                    val = df.iat[i, j]
                    fmt = fmt_special if is_bottom_4 else (fmt_num if _is_number(val) else fmt_text)
                    if _is_number(val):
                        worksheet.write_number(i + 3, j, float(val), fmt)
                    elif pd.isna(val):
                        worksheet.write_blank(i + 3, j, None, fmt)
                    else:
                        worksheet.write(i + 3, j, str(val), fmt)

            _set_fixed_widths_xlsxwriter(df, worksheet)

    return buffer


# =========================================
# PDF EXPORT
# =========================================
def export_report_to_pdf(all_tables, all_figs, logo_path="DataSynthesis logo.png",
                         significance_label=None, experiment_title=None) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=50, rightMargin=50, topMargin=70, bottomMargin=70
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("DSHeading1", fontName="Helvetica-Bold", fontSize=20,
                              textColor=PRIMARY, spaceAfter=14, leading=24))
    styles.add(ParagraphStyle("DSHeading2", fontName="Helvetica-Bold", fontSize=14,
                              textColor=PRIMARY, spaceAfter=10, leading=18))
    styles.add(ParagraphStyle("DSNormal",   fontName="Helvetica", fontSize=10, leading=14))

    elements = []

    # Cover
    if logo_path and os.path.exists(logo_path):
        elements.append(Image(logo_path, width=320, height=180))
    elements.append(Spacer(1, 50))
    elements.append(Paragraph("Trial Assessment Report 2025", styles["DSHeading1"]))

    if experiment_title:
        elements.append(Paragraph(experiment_title, styles["DSHeading2"]))

    today_str = datetime.today().strftime("%d %B %Y")
    elements.append(Paragraph(f"Report generated: {today_str}", styles["DSNormal"]))

    if significance_label:
        elements.append(Paragraph(f"Significance level used in this report: {significance_label}", styles["DSNormal"]))
    elements.append(Paragraph("Prepared by STRI Group", styles["DSNormal"]))
    elements.append(Paragraph("DataSynthesis v1.1", styles["DSNormal"]))
    elements.append(PageBreak())

    # Index
    index_rows = []
    page_no = 1
    index_rows.append([str(page_no), "Cover"]); page_no += 1
    index_rows.append([str(page_no), "Index"]); page_no += 1
    for assess in all_tables.keys():
        index_rows.append([str(page_no), f"{assess} – Chart"]); page_no += 1
        index_rows.append([str(page_no), f"{assess} – Table"]); page_no += 1
    idx_table = Table([["Page", "Description"]] + index_rows,
                      hAlign="CENTER", colWidths=[60, 540])
    idx_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, ACCENT),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
    ]))
    elements.append(Paragraph("Index", styles["DSHeading1"]))
    elements.append(idx_table)
    elements.append(PageBreak())

    # Helpers
    def _merge_stats_for_pdf(df):
        df = df.copy()
        # Drop completely empty columns
        df = df.dropna(axis=1, how="all")
        cols = list(df.columns)
        pairs = {}
        for c in cols:
            base = _detect_stat_base(c, cols)
            if base:
                pairs[base] = c
        for base, scol in pairs.items():
            for i in range(len(df)):
                val = df.at[i, base]
                letter = df.at[i, scol]
                if pd.notna(letter) and str(letter).strip():
                    if _is_number(val):
                        df.at[i, base] = f"{float(val):.2f} {str(letter).strip()}"
                    else:
                        df.at[i, base] = f"{str(val).strip()} {str(letter).strip()}"
        df = df.drop(columns=list(pairs.values()), errors="ignore")
        return df

    def _first_col_width(df):
        texts = [str(df.columns[0])] + [str(x) for x in df.iloc[:, 0].tolist()]
        width_pts = max(stringWidth(t, "Helvetica", 8) for t in texts) + 18
        return max(140, min(width_pts, 320))

    # =========================================
    # Charts + Tables for each assessment
    # =========================================
    for assess, table in all_tables.items():
        # Chart page (use fully styled figure from all_figs)
        if assess in all_figs:
            fig = all_figs[assess]
            img_bytes = fig.to_image(format="png", scale=2)  # keeps colors + axis
            img = Image(BytesIO(img_bytes), width=720, height=400)
            elements.append(Paragraph(f"{assess} – Chart", styles["DSHeading1"]))
            elements.append(img)
            elements.append(PageBreak())

        # Table page
        df = _merge_stats_for_pdf(table)
        first_col_w = _first_col_width(df)
        col_widths = [first_col_w] + [60] * (len(df.columns) - 1)

        data = [list(df.columns)] + df.astype(str).values.tolist()
        tbl = Table(data, colWidths=col_widths, hAlign="CENTER")
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, ACCENT),
            ("BACKGROUND", (0,0), (-1,0), PRIMARY),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))

        elements.append(Paragraph(f"{assess} – Table", styles["DSHeading1"]))
        elements.append(tbl)
        elements.append(PageBreak())

    # Footer
    def _footer(c, d, logo_path_inner, experiment_title_inner):
        width, height = landscape(A4)
        c.setStrokeColor(ACCENT)
        c.setLineWidth(3.5)
        c.roundRect(20, 20, width - 40, height - 40, radius=18, stroke=1, fill=0)
        c.setFont("Helvetica", 9)
        c.setFillColor(DARK)
        page_num = c.getPageNumber()
        footer_text = f"DataSynthesis v1.1 – Page {page_num}"
        # Left: experiment title
        if experiment_title_inner:
            c.drawString(40, 32, experiment_title_inner)
        # Right: version + page
        c.drawRightString(width - 50, 32, footer_text)
        # Logo
        if logo_path_inner and os.path.exists(logo_path_inner):
            try:
                logo_w, logo_h = 130, 53
                c.drawImage(
                    logo_path_inner,
                    width - 200, 44,
                    width=logo_w, height=logo_h,
                    preserveAspectRatio=True, mask="auto"
                )
            except Exception:
                pass

    doc.build(elements,
              onFirstPage=lambda c, d: _footer(c, d, logo_path, experiment_title),
              onLaterPages=lambda c, d: _footer(c, d, logo_path, experiment_title))
    return buffer


# =========================================
# SIDEBAR EXPORT BUTTONS
# =========================================
def export_buttons(all_tables, all_figs, logo_path="DataSynthesis logo.png",
                   significance_label=None, experiment_title=None):
    with st.sidebar:
        st.subheader("Exports")

        # Make safe filename using project title + today's date
        today_str = datetime.today().strftime("%Y-%m-%d")
        if experiment_title:
            safe_title = re.sub(r'[^A-Za-z0-9_-]+', '_', experiment_title.strip())
        else:
            safe_title = "assessment"
        base_filename = f"{safe_title}_{today_str}"

        # Excel export
        excel_buffer = export_tables_to_excel(all_tables, logo_path=None)
        st.download_button(
            "⬇️ Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF export
        pdf_buffer = export_report_to_pdf(
            all_tables, all_figs,
            logo_path=logo_path,
            significance_label=significance_label,
            experiment_title=experiment_title
        )
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_buffer.getvalue(),
            file_name=f"{base_filename}.pdf",
            mime="application/pdf"
        )
