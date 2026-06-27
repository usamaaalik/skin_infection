"""
DermoScan – PDF Report Generator
Uses ReportLab to produce a patient scan history report.
"""

import base64
import io
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Brand colours ─────────────────────────────────────────────────────────────
PURPLE  = colors.HexColor("#6c5ce7")
TEAL    = colors.HexColor("#23d5ab")
LIGHT   = colors.HexColor("#f4f6f9")
MUTED   = colors.HexColor("#888888")
DARK    = colors.HexColor("#333333")
WHITE   = colors.white
RED     = colors.HexColor("#dc3545")
YELLOW  = colors.HexColor("#ffc107")
GREEN   = colors.HexColor("#28a745")

BADGE_COLOURS = {
    "normal":    GREEN,
    "acne":      YELLOW,
    "eczema":    TEAL,
    "psoriasis": RED,
    "ringworm":  colors.HexColor("#6c757d"),
}

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", fontSize=22, textColor=WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontSize=10, textColor=colors.HexColor("#e0f7f2"),
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "section", fontSize=13, textColor=PURPLE, fontName="Helvetica-Bold",
            spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", fontSize=9, textColor=DARK, fontName="Helvetica",
            leading=14,
        ),
        "muted": ParagraphStyle(
            "muted", fontSize=8, textColor=MUTED, fontName="Helvetica",
        ),
        "bold": ParagraphStyle(
            "bold", fontSize=9, textColor=DARK, fontName="Helvetica-Bold",
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", fontSize=8, textColor=colors.HexColor("#c2410c"),
            fontName="Helvetica-Oblique", leading=12,
            backColor=colors.HexColor("#fff7ed"), borderPadding=(6, 8, 6, 8),
        ),
    }


def _header_table(user, total_scans, styles):
    """Purple banner with user info."""
    name_para  = Paragraph(f"DermoScan – Patient Report", styles["title"])
    sub_para   = Paragraph(
        f"Patient: {user.full_name}  ·  {user.email}  ·  "
        f"Generated: {datetime.utcnow().strftime('%d %b %Y, %H:%M UTC')}",
        styles["subtitle"],
    )
    tbl = Table([[name_para], [sub_para]], colWidths=[PAGE_W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PURPLE),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("ROUNDEDCORNERS", [8]),
    ]))
    return tbl


def _summary_table(user, scans, styles):
    """4-cell summary stats row."""
    tier = "Premium ★" if user.is_premium else "Free"
    member_since = user.created_at.strftime("%b %Y")

    cells = [
        ["Total Scans", str(len(scans))],
        ["Account Tier", tier],
        ["Member Since", member_since],
        ["Report Date", datetime.utcnow().strftime("%d %b %Y")],
    ]
    col_w = (PAGE_W - 2 * MARGIN) / 4

    data = [[
        Table([[Paragraph(c[0], styles["muted"])], [Paragraph(c[1], styles["bold"])]],
              colWidths=[col_w - 8])
        for c in cells
    ]]

    tbl = Table(data, colWidths=[col_w] * 4)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [6]),
    ]))
    return tbl


def _scans_table(scans, upload_dir, styles):
    """Main scan history table with thumbnail images."""
    col_widths = [10*mm, 20*mm, 30*mm, 28*mm, 32*mm, 32*mm, 28*mm]
    header = [
        Paragraph("#",           styles["muted"]),
        Paragraph("Image",       styles["muted"]),
        Paragraph("Detection",   styles["muted"]),
        Paragraph("Confidence",  styles["muted"]),
        Paragraph("Date",        styles["muted"]),
        Paragraph("Description", styles["muted"]),
        Paragraph("Severity",    styles["muted"]),
    ]

    rows = [header]

    DISEASE_INFO = {
        "acne":      ("Acne Vulgaris – inflamed pilosebaceous units.",       "Moderate"),
        "eczema":    ("Atopic dermatitis – dry, itchy skin patches.",         "Moderate"),
        "normal":    ("No signs of infection detected.",                      "None"),
        "psoriasis": ("Autoimmune – rapid skin-cell buildup, scaling.",       "High"),
        "ringworm":  ("Tinea corporis – ring-shaped fungal infection.",       "Moderate"),
    }

    for i, scan in enumerate(scans, start=1):
        cls = scan.predicted_class.lower()

        # Thumbnail
        image_bytes = getattr(scan, "image_bytes", None)
        if image_bytes:
            try:
                image_data = base64.b64decode(image_bytes)
                image_stream = io.BytesIO(image_data)
                thumb = RLImage(image_stream, width=16*mm, height=16*mm)
                thumb.hAlign = "CENTER"
            except Exception:
                thumb = Paragraph("–", styles["muted"])
        else:
            img_path = os.path.join(upload_dir, scan.image_filename)
            if os.path.exists(img_path):
                try:
                    thumb = RLImage(img_path, width=16*mm, height=16*mm)
                    thumb.hAlign = "CENTER"
                except Exception:
                    thumb = Paragraph("–", styles["muted"])
            else:
                thumb = Paragraph("–", styles["muted"])

        # Badge colour cell
        badge_color = BADGE_COLOURS.get(cls, PURPLE)
        cls_para = Paragraph(
            f'<font color="white"><b> {cls.capitalize()} </b></font>',
            ParagraphStyle("badge", fontSize=8, fontName="Helvetica-Bold",
                           alignment=TA_CENTER, backColor=badge_color,
                           borderPadding=(3, 6, 3, 6), leading=14),
        )

        desc, severity = DISEASE_INFO.get(cls, ("–", "–"))

        rows.append([
            Paragraph(str(i),                              styles["muted"]),
            thumb,
            cls_para,
            Paragraph(f"{scan.confidence_pct}%",          styles["bold"]),
            Paragraph(scan.created_at.strftime("%d %b %Y\n%I:%M %p"), styles["muted"]),
            Paragraph(desc,                                styles["body"]),
            Paragraph(severity,                            styles["body"]),
        ])

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0),  LIGHT),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
        ("TOPPADDING",    (0, 0), (-1, 0),  8),
        # Data rows
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("TOPPADDING",    (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, colors.HexColor("#fafafa")]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#e0e0e0")),
        ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
        ("ALIGN",         (2, 0), (2, -1),  "CENTER"),
        ("ALIGN",         (3, 0), (3, -1),  "CENTER"),
    ]))
    return tbl


def generate_history_pdf(user, scans, upload_dir: str,
                         date_from=None, date_to=None) -> bytes:
    """
    Build a full history PDF report and return it as bytes.
    date_from / date_to are optional date objects for the range label.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title=f"DermoScan Report – {user.full_name}",
        author="DermoScan",
    )

    styles = _styles()
    story  = []

    # ── Header banner ────────────────────────────────────────────────────────
    story.append(_header_table(user, len(scans), styles))
    story.append(Spacer(1, 6*mm))

    # ── Date range note ───────────────────────────────────────────────────────
    if date_from or date_to:
        from_label = date_from.strftime("%d %b %Y") if date_from else "All time"
        to_label   = date_to.strftime("%d %b %Y")   if date_to   else "Today"
        range_para = Paragraph(
            f"<b>Date range:</b>  {from_label}  →  {to_label}",
            ParagraphStyle("range", fontSize=9, textColor=PURPLE,
                           fontName="Helvetica-Bold", spaceAfter=4),
        )
        story.append(range_para)

    # ── Summary stats ────────────────────────────────────────────────────────
    story.append(_summary_table(user, scans, styles))
    story.append(Spacer(1, 6*mm))

    # ── Scan history table ────────────────────────────────────────────────────
    story.append(Paragraph("Scan History", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=4))

    if scans:
        story.append(_scans_table(scans, upload_dir, styles))
    else:
        story.append(Paragraph("No scans recorded yet.", styles["muted"]))

    story.append(Spacer(1, 8*mm))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=4))
    story.append(Paragraph(
        "⚠  Medical Disclaimer: This report is generated by an AI model for informational "
        "purposes only. It does NOT constitute a medical diagnosis. Please consult a "
        "certified dermatologist for professional evaluation and treatment.",
        styles["disclaimer"],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"Generated by DermoScan  ·  {datetime.utcnow().strftime('%d %B %Y')}",
        styles["muted"],
    ))

    doc.build(story)
    return buf.getvalue()


def generate_single_pdf(user, scan, upload_dir: str) -> bytes:
    """
    Build a single-scan PDF report and return it as bytes.
    """
    import json

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title=f"DermoScan Scan #{scan.id} – {user.full_name}",
        author="DermoScan",
    )

    styles = _styles()
    story  = []

    # Header
    story.append(_header_table(user, 1, styles))
    story.append(Spacer(1, 6*mm))

    # Scan meta
    story.append(Paragraph(f"Scan Report  ·  #{scan.id}", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=6))

    # Image + result side by side
    image_bytes = getattr(scan, "image_bytes", None)
    img_cell = "–"
    if image_bytes:
        try:
            image_data = base64.b64decode(image_bytes)
            image_stream = io.BytesIO(image_data)
            img_cell = RLImage(image_stream, width=55*mm, height=55*mm)
        except Exception:
            pass
    else:
        img_path = os.path.join(upload_dir, scan.image_filename)
        if os.path.exists(img_path):
            try:
                img_cell = RLImage(img_path, width=55*mm, height=55*mm)
            except Exception:
                pass

    cls   = scan.predicted_class.lower()
    badge = BADGE_COLOURS.get(cls, PURPLE)

    DISEASE_INFO = {
        "acne":      ("Acne Vulgaris",  "Acne vulgaris is a chronic inflammatory condition of the pilosebaceous units.",
                      "Topical retinoids, benzoyl peroxide, antibiotics, or oral isotretinoin for severe cases.", "Moderate"),
        "eczema":    ("Atopic Dermatitis", "Eczema causes dry, red, and intensely itchy skin patches.",
                      "Moisturisers, topical corticosteroids, and avoiding known triggers.", "Moderate"),
        "normal":    ("Normal / Healthy", "No signs of skin infection were detected.",
                      "Maintain good skincare hygiene, stay hydrated, and use sunscreen.", "None"),
        "psoriasis": ("Psoriasis", "Chronic autoimmune condition causing rapid skin-cell buildup and scaling.",
                      "Topical treatments, phototherapy, or systemic medications. Consult a dermatologist.", "High"),
        "ringworm":  ("Tinea Corporis", "Contagious fungal infection with a ring-shaped rash.",
                      "Topical antifungal creams (clotrimazole, terbinafine). Oral antifungals for extensive cases.", "Moderate"),
    }

    full_name, desc, treatment, severity = DISEASE_INFO.get(
        cls, (cls.capitalize(), "–", "–", "–")
    )

    # Parse all_scores
    try:
        all_scores = json.loads(scan.all_scores) if scan.all_scores else {}
    except Exception:
        all_scores = {}

    scores_text = "  ·  ".join(
        f"{k.capitalize()}: {v}%" for k, v in
        sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    ) if all_scores else "–"

    result_data = [
        [Paragraph("<b>Detection</b>",   styles["body"]),
         Paragraph(f'<font color="white"><b> {full_name} </b></font>',
                   ParagraphStyle("b2", fontSize=10, fontName="Helvetica-Bold",
                                  backColor=badge, borderPadding=(4,8,4,8), leading=16))],
        [Paragraph("<b>Confidence</b>",  styles["body"]), Paragraph(f"{scan.confidence_pct}%", styles["bold"])],
        [Paragraph("<b>Severity</b>",    styles["body"]), Paragraph(severity,                  styles["body"])],
        [Paragraph("<b>Date</b>",        styles["body"]), Paragraph(scan.created_at.strftime("%d %B %Y, %I:%M %p"), styles["body"])],
        [Paragraph("<b>All Scores</b>",  styles["body"]), Paragraph(scores_text,               styles["muted"])],
        [Paragraph("<b>Description</b>", styles["body"]), Paragraph(desc,                      styles["body"])],
        [Paragraph("<b>Treatment</b>",   styles["body"]), Paragraph(treatment,                 styles["body"])],
    ]

    result_tbl = Table(result_data, colWidths=[35*mm, 95*mm])
    result_tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, LIGHT]),
    ]))

    combined = Table([[img_cell, result_tbl]],
                     colWidths=[60*mm, PAGE_W - 2*MARGIN - 60*mm])
    combined.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
    ]))
    story.append(combined)
    story.append(Spacer(1, 8*mm))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=4))
    story.append(Paragraph(
        "⚠  Medical Disclaimer: This report is generated by an AI model for informational "
        "purposes only. It does NOT constitute a medical diagnosis. Please consult a "
        "certified dermatologist for professional evaluation and treatment.",
        styles["disclaimer"],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"Generated by DermoScan  ·  {datetime.utcnow().strftime('%d %B %Y')}",
        styles["muted"],
    ))

    doc.build(story)
    return buf.getvalue()
