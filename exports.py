# exports.py

import os
from io import BytesIO
import pandas as pd
import numpy as np

# Streamlit is only used for the buttons function
import streamlit as st

# ReportLab / PDF
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth

# Charts
import plotly.graph_objects as go


# =========================================
# Brand Colours
# =========================================
PRIMARY   = colors.HexColor("#0B6580")   # headings / header rows
SECONDARY = colors.HexColor("#59B37D")   # highlights (PLSD, D.F., %CV rows)
ACCENT    = colors.HexColor("#40B5AB")   # borders, gridlines, page frame
DARK      = colors.HexColor("#004754")   # axis text, footer text


# =========================================
# Helpers (shared)
# =========================================
def _is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and pd.notna(x)

def _detect_stat_base(col_name: str, existing_cols: list[str]) -> str | None:
    """Detect if col_name is a stats letter column and return its base column name if so."""
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

def _autofit_widths_xlsxwriter(df: pd.DataFrame, min_width=8, max_width=50):
    """Estimate Excel column widths from content length."""
    widths = []
    for j, col in enumerate(df.columns):
        max_len = len(str(col))
        for val in df[col].astype(str).tolist():
            if len(val) > max_len:
                max_len = len(val)
        w = min(max_width, max(min_width, int(max_len * 1.15)))
        widths.append(w)
    return widths


# =========================================
# EXCEL EXPORT
# =========================================
def export_tables_to_excel(all_tables: dict[str, pd.DataFrame], logo_path=None) -> BytesIO:
    """
    Return a BytesIO Excel file with DataSynthesis branding.
    - No logo
    - Numbers written as numbers (preserve source precision)
    - Autofit columns
    - If a matching '<base> S' column is present:
        * Drop it if all empty
        * Otherwise keep it immediately to the right of its base numeric column
    - Bottom 4 rows shaded secondary green with white text
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for assess, table in all_tables.items():
            df = table.copy()
            cols = list(df.columns)
            ordered = []
            used = set()

            # Build ordered column list
            for c in cols:
                if c in used:
                    continue
                base = _detect_stat_base(c, cols)
                if base:
                    continue
                ordered.append(c)
                used.add(c)
                s_candidates = [sc for sc in cols if _detect_stat_base(sc, cols) == c]
                if s_candidates:
                    s_col = s_candidates[0]
                    col_series = df[s_col].astype(str).str.strip().fillna("")
                    # ✅ FIX: safer check for empty stats column
                    if col_series.replace("nan", "").str.strip().eq("").all():
                        df = df.drop(columns=[s_col])
                        continue
                    ordered.append(s_col)
                    used.add(s_col)

            for c in cols:
                if c not in used:
                    ordered.append(c)
                    used.add(c)

            df = df.reindex(columns=ordered)

            sheet_name = str(assess)[:30] if assess else "Sheet1"
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
            workbook  = writer.book
            worksheet = writer.sheets[sheet_name]

            # Formats
            fmt_header   = workbook.add_format({
                "bg_color": "#0B6580", "font_color": "white",
                "bold": True, "border": 1, "align": "center", "valign": "vcenter",
                "border_color": "#40B5AB"
            })
            fmt_text     = workbook.add_format({"align": "center", "valign": "vcenter",
                                                "border": 1, "border_color": "#40B5AB"})
            fmt_num      = workbook.add_format({"border": 1, "border_color": "#40B5AB"})
            fmt_special  = workbook.add_format({"bg_color": "#59B37D", "font_color": "white",
                                                "align": "center", "valign": "vcenter",
                                                "border": 1, "border_color": "#40B5AB"})

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
                    else:
                        worksheet.write(i + 3, j, val, fmt)

            # Autofit widths
            widths = _autofit_widths_xlsxwriter(df)
            for j, w in enumerate(widths):
                worksheet.set_column(j, j, w)

    return buffer


# =========================================
# PDF EXPORT
# =========================================
def export_report_to_pdf(
    all_tables: dict[str, pd.DataFrame],
    all_figs: dict[str, go.Figure],
    logo_path: str = "DataSynthesis logo.png",
    significance_label: str | None = None
) -> BytesIO:
    """Return a BytesIO PDF report with DataSynthesis v1.1 branding."""
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

    # ---------- Cover ----------
    if logo_path and os.path.exists(logo_path):
        elements.append(Image(logo_path, width=320, height=130))
    elements.append(Spacer(1, 50))
    elements.append(Paragraph("Trial Assessment Report 2025", styles["DSHeading1"]))
    if significance_label:
        elements.append(Paragraph(f"Significance level used in this report: {significance_label}", styles["DSNormal"]))
    elements.append(Paragraph("Prepared by STRI Group", styles["DSNormal"]))
    elements.append(Paragraph("DataSynthesis v1.1", styles["DSNormal"]))
    elements.append(PageBreak())

    # ---------- Index ----------
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

    # ---------- Helpers ----------
    def _merge_stats_for_pdf(df: pd.DataFrame) -> pd.DataFrame:
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

    def _first_col_width(df: pd.DataFrame) -> float:
        texts = [str(df.columns[0])] + [str(x) for x in df.iloc[:, 0].tolist()]
        width_pts = max(stringWidth(t, "Helvetica", 8) for t in texts) + 18
        return max(140, min(width_pts, 320))

    # ---------- Content ----------
    for assess, table in all_tables.items():
        # Chart page
        elements.append(Paragraph(f"{assess} Chart", styles["DSHeading2"]))
        if all_figs and assess in all_figs:
            fig_bytes = BytesIO()
            try:
                fig = go.Figure(all_figs[assess])
                fig.update_xaxes(title_text=None, tickangle=90, tickfont=dict(size=10, color="#004754"))
                fig.update_yaxes(tickfont=dict(size=11, color="#004754"))
                fig.update_layout(
                    margin=dict(l=90, r=50, t=70, b=170),
                    legend=dict(orientation="h", y=-0.32, x=0.5, xanchor="center", font=dict(size=10)),
                    font=dict(size=14),
                    template="plotly"
                )
                if fig.layout.annotations:
                    updated = []
                    for ann in fig.layout.annotations:
                        txt = str(getattr(ann, "text", "")).strip()
                        if txt.lower() == "date":
                            continue
                        ann.update(
                            font=dict(size=max(getattr(ann.font, "size", 12), 20), color="black"),
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="#0B6580",
                            yshift=12
                        )
                        updated.append(ann)
                    fig.update_layout(annotations=updated)

                fig.write_image(fig_bytes, format="png", scale=2)
                fig_bytes.seek(0)
                chart_img = Image(fig_bytes, width=720, height=420)
                chart_table = Table([[chart_img]], colWidths=[720], hAlign="CENTER")
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

        # Table page
        elements.append(Paragraph(f"{assess} Table", styles["DSHeading2"]))
        df = _merge_stats_for_pdf(table)
        data = [df.columns.tolist()] + df.astype(str).values.tolist()
        first_w = _first_col_width(df)
        col_widths = [first_w] + [45] * (len(df.columns) - 1)
        pdf_table = Table(data, hAlign="CENTER", colWidths=col_widths)
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
        total_rows = len(df)
        for i in range(total_rows):
            if (total_rows - i) <= 4:
                table_style += [
                    ("BACKGROUND", (0, i + 1), (-1, i + 1), SECONDARY),
                    ("TEXTCOLOR",  (0, i + 1), (-1, i + 1), colors.white),
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
        c.drawRightString(width - 50, 22, footer_text)
        if logo_path_inner and os.path.exists(logo_path_inner):
            try:
                logo_w, logo_h = 70, 28
                c.drawImage(
                    logo_path_inner,
                    width - 50 - logo_w,
                    22 + 6,
                    width=logo_w, height=logo_h,
                    preserveAspectRatio=True, mask="auto"
                )
            except Exception:
                pass

    doc.build(elements,
              onFirstPage=lambda c, d: _footer(c, d, logo_path),
              onLaterPages=lambda c, d: _footer(c, d, logo_path))
    return buffer


# =========================================
# SIDEBAR EXPORT BUTTONS
# =========================================
def export_buttons(
    all_tables: dict[str, pd.DataFrame],
    all_figs: dict[str, go.Figure],
    logo_path: str = "DataSynthesis logo.png",
    significance_label: str | None = None
):
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
