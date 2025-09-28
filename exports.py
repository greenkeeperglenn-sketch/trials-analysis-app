def export_report_to_pdf(all_tables, all_figs, logo_path="DataSynthesis LOGO.png"):
    """Return a BytesIO PDF report with DataSynthesis v1.1 branding and charts."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=50, rightMargin=50,
        topMargin=70, bottomMargin=70
    )

    # --- Brand Colours ---
    BRAND_PRIMARY = colors.HexColor("#0B6580")   # main headers / titles
    BRAND_SECONDARY = colors.HexColor("#59B37D") # accents if needed
    BRAND_DARK = colors.HexColor("#004754")      # footer / emphasis

    elements = []

    # --- Styles ---
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="DSHeading1", fontName="Helvetica-Bold", fontSize=20,
        textColor=BRAND_PRIMARY, spaceAfter=14, leading=24
    ))
    styles.add(ParagraphStyle(
        name="DSHeading2", fontName="Helvetica-Bold", fontSize=14,
        textColor=BRAND_PRIMARY, spaceAfter=10, leading=18
    ))
    styles.add(ParagraphStyle(
        name="DSNormal", fontName="Helvetica", fontSize=10, leading=14
    ))

    # --- Cover Page ---
    if logo_path and os.path.exists(logo_path):
        elements.append(Image(logo_path, width=320, height=130))
    elements.append(Spacer(1, 60))
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
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(toc_table)
    elements.append(PageBreak())

    # --- Assessment Sections ---
    for assess, table in all_tables.items():
        # --- Chart Page ---
        elements.append(Paragraph(f"{assess} Chart", styles["DSHeading2"]))
        if all_figs and assess in all_figs:
            fig_bytes = BytesIO()
            try:
                fig = go.Figure(all_figs[assess])
                fig.update_layout(
                    font=dict(size=16),
                    margin=dict(l=80, r=40, t=80, b=120),
                    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
                )
                # bump statistical letters
                fig.update_annotations(font_size=22, yshift=10)

                fig.write_image(fig_bytes, format="png", scale=2)
                fig_bytes.seek(0)
                chart_img = Image(fig_bytes, width=720, height=420)
                chart_table = Table([[chart_img]], colWidths=[720])
                chart_table.setStyle(TableStyle([
                    ("BOX", (0, 0), (-1, -1), 1.5, BRAND_PRIMARY),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                elements.append(chart_table)
                elements.append(PageBreak())
            except Exception as e:
                elements.append(Paragraph(f"(Chart error: {e})", styles["DSNormal"]))
                elements.append(PageBreak())

        # --- Table Page ---
        elements.append(Paragraph(f"{assess} Table", styles["DSHeading2"]))

        df = table.copy()

        # Merge stats letters into numeric cells if 'Stats' col exists
        if "Stats" in df.columns:
            for i in range(len(df)):
                num = str(df.iloc[i, 1])  # assumes 2nd col is numeric
                letter = str(df.iloc[i]["Stats"]) if pd.notna(df.iloc[i]["Stats"]) else ""
                if letter and letter != "nan":
                    df.iloc[i, 1] = f"{num} {letter}"
            df = df.drop(columns=["Stats"], errors="ignore")

        data = [df.columns.tolist()] + df.astype(str).values.tolist()

        # Auto-fit first col width
        max_first = max(len(str(x)) for x in df.iloc[:, 0]) if not df.empty else 20
        col_widths = [max_first * 6] + [45] * (len(df.columns) - 1)

        pdf_table = Table(data, hAlign="LEFT", colWidths=col_widths)
        table_style = [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("ROTATE", (1, 0), (-1, 0), 90),  # rotate headers after first col
        ]

        # Grey shading only for PLSD, D.F., %CV rows
        for i, row in enumerate(df.iloc[:, 0]):
            if any(key in str(row) for key in ["PLSD", "D.F.", "%CV"]):
                table_style.append(("BACKGROUND", (0, i+1), (-1, i+1), BRAND_DARK))
                table_style.append(("TEXTCOLOR", (0, i+1), (-1, i+1), colors.white))

        pdf_table.setStyle(TableStyle(table_style))
        elements.append(pdf_table)
        elements.append(PageBreak())

    # --- Footer with logo + version + page number ---
    def add_footer(c, d, logo_path):
        width, height = landscape(A4)
        page_num = c.getPageNumber()
        c.setFont("Helvetica", 9)
        c.setFillColor(BRAND_DARK)
        c.drawRightString(width - 40, 20, f"DataSynthesis v1.1 – Page {page_num}")
        if logo_path and os.path.exists(logo_path):
            try:
                c.drawImage(logo_path, 40, 10, width=60, height=25,
                            preserveAspectRatio=True, mask="auto")
            except Exception:
                pass

    doc.build(elements,
              onFirstPage=lambda c, d: add_footer(c, d, logo_path),
              onLaterPages=lambda c, d: add_footer(c, d, logo_path))
    return buffer
