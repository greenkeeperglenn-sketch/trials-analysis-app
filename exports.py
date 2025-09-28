# exports.py

import os
from io import BytesIO
import pandas as pd
import numpy as np

import streamlit as st

from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer, PageBreak, KeepTogether
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
def export_report_to_pdf(all_tables, all_figs, logo_path="DataSynthesis logo.png", significance_label=None) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=50, rightMargin=50, topMargin=70, bottomMargin=70
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("DSHeading1", fontName="Helvetica-Bold", fontSize=20, textColor=PRIMARY, spaceAfter=14, leading=24))
    styles.add(ParagraphStyle("DSHeading2", fontName="Helvetica-Bold", fontSize=14, textColor=PRIMARY, spaceAfter=10, leading=18))
    styles.add(ParagraphStyle("DSNormal",   fontName="Helvetica",      fontSize=10, leading=14))

    elements = []

    # Cover
    if logo_path and os.path.exists(logo_path):
        elements.append(Image(logo_path, width=320, height=180))  # larger logo
    elements.append(Spacer(1, 50))
    elements.append(Paragraph("Trial Assessment Report 2025", styles["DSHeading1"]))
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
    idx_table = Table([["Page", "Description"]] + index_rows, hAlign="CENTER", colWidths=[60, 540])
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

    # Content
    for assess, table in all_tables.items():
        # ------- Chart page (heading + chart kept together to avoid blank heading pages)
        if all_figs and assess in all_figs:
            try:
                orig = all_figs[assess]
                fig = go.Figure(orig)  # clone

                # Preserve y-range from the original chart if present
                try:
                    y_rng = getattr(getattr(orig.layout, "yaxis", None), "range", None)
                except Exception:
                    y_rng = None
                if y_rng is not None:
                    fig.update_yaxes(range=list(y_rng), autorange=False)

                # Layout: legend inside chart, space reserved below, rotated x labels, small ticks
                fig.update_layout(
                    margin=dict(l=90, r=50, t=70, b=220),   # reserve space for legend
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        y=-0.22, yanchor="top",
                        x=0.5, xanchor="center",
                        font=dict(size=8),
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#0B6580",
                        borderwidth=0.5,
                    ),
                    font=dict(size=15),
                    template="plotly",
                )
                fig.update_xaxes(tickangle=90, tickfont=dict(size=9, color="#004754"))
                fig.update_yaxes(tickfont=dict(size=9, color="#004754"))

                # Make annotation letters bigger & tidy "Date" label if present
                if fig.layout.annotations:
                    updated = []
                    for ann in fig.layout.annotations:
                        txt = str(getattr(ann, "text", "")).strip()
                        if txt.lower() == "date":
                            continue
                        ann.update(
                            font=dict(size=max(getattr(ann.font, "size", 12), 22), color="black"),
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="#0B6580",
                            yshift=12,
                        )
                        updated.append(ann)
                    fig.update_layout(annotations=updated)

                # Render image
                fig_bytes = BytesIO()
                fig.write_image(fig_bytes, format="png", scale=2)
                fig_bytes.seek(0)
                chart_img = Image(fig_bytes, width=720, height=400)

                # Keep heading + chart together to prevent blank pages
                chart_heading = Paragraph(f"{assess} Chart", styles["DSHeading2"])
                chart_table = Table([[chart_img]], colWidths=[720], hAlign="CENTER")
                chart_table.setStyle(TableStyle([
                    ("BOX", (0, 0), (-1, -1), 2, ACCENT),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                elements.append(KeepTogether([chart_heading, chart_table]))
                elements.append(PageBreak())  # move to the table page next

            except Exception as e:
                elements.append(Paragraph(f"(Chart error: {e})", styles["DSNormal"]))
                elements.append(PageBreak())

        # ------- Table page
        elements.append(Paragraph(f"{assess} Table", styles["DSHeading2"]))
        df = _merge_stats_for_pdf(table)
        data = [df.columns.tolist()] + df.astype(str).values.tolist()
        first_w = _first_col_width(df)

        # widths: first col auto, stats cols = 25, numeric cols = 45
        col_widths = []
        for c in df.columns:
            if c == df.columns[0]:
                col_widths.append(first_w)
            elif c.endswith("S") or c.endswith("_S") or "(S)" in c:
                col_widths.append(25)
            else:
                col_widths.append(45)

        pdf_table = Table(data, hAlign="CENTER", colWidths=col_widths)
        table_style = [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, ACCENT),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("ROTATE", (0, 0), (-1, 0), 90),
        ]
        total_rows = len(df)
        for i in range(total_rows):
            if (total_rows - i) <= 4:
                table_style += [
                    ("BACKGROUND", (0, i + 1), (-1, i + 1), SECONDARY),
                    ("TEXTCOLOR",  (0, i + 1), (-1, i + 1), colors.white),
                    ("LINEABOVE", (0, i + 1), (-1, i + 1), 0.5, colors.white),
                    ("LINEBELOW", (0, i + 1), (-1, i + 1), 0.5, colors.white),
                    ("LINEBEFORE",(0, i + 1), (-1, i + 1), 0.5, colors.white),
                    ("LINEAFTER", (0, i + 1), (-1, i + 1), 0.5, colors.white),
                ]
        pdf_table.setStyle(TableStyle(table_style))
        elements.append(pdf_table)
        elements.append(PageBreak())

    # Footer
    def _footer(c, d, logo_path_inner):
        width, height = landscape(A4)
        c.setStrokeColor(ACCENT)
        c.setLineWidth(3.5)
        c.roundRect(20, 20, width - 40, height - 40, radius=18, stroke=1, fill=0)
        c.setFont("Helvetica", 9)
        c.setFillColor(DARK)
        page_num = c.getPageNumber()
        footer_text = f"DataSynthesis v1.1 – Page {page_num}"
        if logo_path_inner and os.path.exists(logo_path_inner):
            try:
                logo_w, logo_h = 130, 53  # bigger and a bit taller
                c.drawImage(
                    logo_path_inner,
                    width - 200, 44,   # nudge left & up
                    width=logo_w, height=logo_h,
                    preserveAspectRatio=True, mask="auto"
                )
            except Exception:
                pass
        c.drawRightString(width - 50, 32, footer_text)

    doc.build(elements,
              onFirstPage=lambda c, d: _footer(c, d, logo_path),
              onLaterPages=lambda c, d: _footer(c, d, logo_path))
    return buffer


# =========================================
# SIDEBAR EXPORT BUTTONS
# =========================================
def export_buttons(all_tables, all_figs, logo_path="DataSynthesis logo.png", significance_label=None):
    with st.sidebar:
        st.subheader("Exports")

        excel_buffer = export_tables_to_excel(all_tables, logo_path=None)
        st.download_button(
            "⬇️ Download Excel",
            data=excel_buffer.getvalue(),
            file_name="assessment_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        pdf_buffer = export_report_to_pdf(all_tables, all_figs, logo_path=logo_path, significance_label=significance_label)
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_buffer.getvalue(),
            file_name="assessment_report.pdf",
            mime="application/pdf"
        )
