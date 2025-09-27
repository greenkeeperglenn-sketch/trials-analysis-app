# -------------------------------------------------------------------
# PDF EXPORT
# -------------------------------------------------------------------
def export_report_to_pdf(all_tables, all_figs, logo_path=None):
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

    # --- Index page ---
    elements.append(Paragraph("Index", styles["STRIHeading1"]))
    toc_data = [["Assessment", "Chart Page", "Table Page"]]
    page_counter = 2  # cover = 1, index = 2, start after that
    for assess in all_tables.keys():
        toc_data.append([assess, str(page_counter+1), str(page_counter+2)])
        page_counter += 2
    toc_table = Table(toc_data, hAlign="LEFT")
    toc_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(toc_table)
    elements.append(PageBreak())

    # --- Assessment sections ---
    for assess, table in all_tables.items():
        # Chart Page
        elements.append(Paragraph(f"{assess} Chart", styles["STRIHeading2"]))

        if all_figs and assess in all_figs:
            fig_bytes = BytesIO()
            try:
                # Clone fig & bump annotation font
                fig = all_figs[assess].to_dict()
                import plotly.graph_objects as go
                fig = go.Figure(fig)
                for ann in fig.layout.annotations or []:
                    ann.font.size = 18

                fig.write_image(fig_bytes, format="png", scale=2)
                fig_bytes.seek(0)

                chart_img = Image(fig_bytes, width=720, height=400)

                chart_table = Table([[chart_img]], colWidths=[720])
                chart_table.setStyle(TableStyle([
                    ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#1f77b4")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                elements.append(chart_table)
                elements.append(PageBreak())
            except Exception as e:
                elements.append(Paragraph(f"(Chart error: {e})", styles["STRINormal"]))
                elements.append(PageBreak())

        # Table Page
        elements.append(Paragraph(f"{assess} Table", styles["STRIHeading2"]))

        data = [table.columns.tolist()] + table.astype(str).values.tolist()
        col_widths = [180] + [40]*(len(table.columns)-1)  # wide first col
        pdf_table = Table(data, hAlign="LEFT", colWidths=col_widths)

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
            ("ROTATE", (1, 0), (-1, 0), 90),  # rotate headers after 1st col
        ]))
        elements.append(pdf_table)
        elements.append(PageBreak())

    # Build with footer
    doc.build(elements, onFirstPage=lambda c, d: add_footer(c, d, logo_path),
              onLaterPages=lambda c, d: add_footer(c, d, logo_path))
    return buffer
