"""
generate_docx_reports.py
========================
Generates professional Word documents (.docx) for Phase 3:
1. reports/experiment_design.docx (Comprehensive 15-Section Protocol)
2. reports/decision_memo.docx (Executive One-Page A4 Decision Memo)
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Color Palette: Deep Navy & Slate Grey
COLOR_PRIMARY = RGBColor(31, 78, 121)    # Deep Navy
COLOR_SECONDARY = RGBColor(89, 89, 89)  # Slate Grey
COLOR_DARK = RGBColor(38, 38, 38)       # Near Black
COLOR_MUTED = RGBColor(115, 115, 115)   # Subdued Grey
COLOR_ACCENT = RGBColor(46, 125, 50)    # Forest Green / Success


def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding of a table cell (in twentieths of a point / dxa)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)


# =====================================================================
# 1. EXPERIMENT DESIGN DOCUMENT (15 SECTIONS)
# =====================================================================
def create_experiment_design_document():
    doc = Document()

    # Page setup - Standard 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Title & Header
    title = doc.add_paragraph()
    title_run = title.add_run("EXPERIMENT DESIGN PROTOCOL")
    title_run.font.name = "Calibri"
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = COLOR_PRIMARY
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(2)

    subtitle = doc.add_paragraph()
    sub_run = subtitle.add_run("Automated WhatsApp Confirmation & Smart Rescheduling System")
    sub_run.font.name = "Calibri"
    sub_run.font.size = Pt(13)
    sub_run.font.italic = True
    sub_run.font.color.rgb = COLOR_SECONDARY
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(18)

    # Metadata Card
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("Enterprise / Chain", "Aura Luxe Salon & Spa Chain (Coimbatore, Tamil Nadu)"),
         ("Lead Investigator", "Senior Data Scientist & Experimentation Architect")],
        [("Geographic Footprint", "22 Outlets across 5 Regions (Central, East, North, South, West)"),
         ("Evaluation Window", "July 1, 2024 – October 31, 2024 (60-Day Fixed Cadence)")]
    ]
    for r_idx, row in enumerate(meta_data):
        for c_idx, (label, val) in enumerate(row):
            cell = meta_table.cell(r_idx, c_idx)
            set_cell_background(cell, "F2F4F7")
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            lbl_run = p.add_run(f"{label}: ")
            lbl_run.bold = True
            lbl_run.font.size = Pt(9.5)
            lbl_run.font.color.rgb = COLOR_PRIMARY
            val_run = p.add_run(val)
            val_run.font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Content Sections
    sections_content = [
        ("1. Project Overview",
         "This experiment protocol outlines the rigorous testing methodology deployed by Aura Luxe Salon Chain to measure the causal impact of replacing legacy one-way SMS reminders with an automated, interactive WhatsApp appointment confirmation and rescheduling system. The trial evaluates 18,000 enrolled active clients across 22 salon locations in Coimbatore, Tamil Nadu, over a 4-month randomized prospective evaluation window."),

        ("2. Business Problem & Operational Bottlenecks",
         "Aura Luxe operates in a high-capacity service industry where vacant chair time cannot be recovered. Baseline exploratory analytics revealed two severe operational vulnerabilities: (1) An unnotified no-show rate of 12.1% across digital and phone bookings, causing ~420 lost stylist hours weekly across the chain, and (2) A 60-day customer repeat retention plateau of 58.2%, where over 41% of clients fail to return for repeat grooming or wellness services within two months."),

        ("3. Business Initiative (Intervention Details)",
         "The business initiative replaces static 24-hour SMS text alerts with an automated, bi-directional WhatsApp chatbot. The protocol dispatches personalized messages at 48 hours and 24 hours prior to appointment slots. Key interactive capabilities include 1-click booking confirmation, an instant rescheduling link that displays open stylist slots within 72 hours, customized stylist portfolio notes, and automated loyalty reward balance alerts."),

        ("4. Research Question",
         "Does replacing standard one-way SMS reminders with an interactive automated WhatsApp confirmation and smart rescheduling system generate a statistically significant increase in 60-day customer repeat retention and a reduction in appointment no-show rates across retail salon outlets?"),

        ("5. Formal Hypotheses",
         "Primary Retention Hypothesis:\n"
         "• Null Hypothesis (H0): The WhatsApp intervention produces no difference in 60-day customer retention relative to SMS (p_treatment = p_control).\n"
         "• Alternative Hypothesis (H1): The WhatsApp intervention produces a statistically significant lift in 60-day customer retention (p_treatment > p_control).\n\n"
         "Secondary No-Show Hypothesis:\n"
         "• Null Hypothesis (H0_ns): The intervention produces no change in appointment no-show probability.\n"
         "• Alternative Hypothesis (H1_ns): The intervention significantly reduces appointment no-shows (p_treatment < p_control)."),

        ("6. Treatment and Control Design",
         "• Control Group (n = 9,000): Enrolled customers receive standard automated 24-hour one-way SMS reminder text messages containing outlet address and phone number, with no direct digital rescheduling mechanism.\n"
         "• Treatment Group (n = 9,000): Enrolled customers receive rich interactive WhatsApp notifications at 48h and 24h before the appointment, offering 1-click confirmation, immediate automated slot-swap rescheduling, and stylist portfolio highlights.\n"
         "• Uniformity: Service delivery, salon floor pricing, promotional discounts, and stylist assignments remain completely identical across both experimental cohorts."),

        ("7. Randomization Methodology & SUTVA Integrity",
         "Randomization is performed at the customer level (customer_id), stratified by outlet (outlet_id) to ensure equal representation across all 22 Coimbatore locations and diverse demographic tiers (Flagship Premium, Student & IT Hub, Suburban Value).\n"
         "Interference & SUTVA (Stable Unit Treatment Value Assumption):\n"
         "• Spillover Mitigation: Digital delivery is executed via cloud APIs directly to customer mobile numbers, eliminating stylist awareness of customer treatment tags during service delivery.\n"
         "• Capacity Substitution: Because rescheduling frees up vacant slots, walk-in customers could absorb canceled slots. Stratifying at the outlet level balances counter-walk-in absorption equally across treatment and control demand pools."),

        ("8. Primary Metric Specification",
         "• Primary Metric: 60-Day Customer Repeat Retention Rate.\n"
         "• Mathematical Definition: R_60 = (Count of enrolled customers completing >= 1 appointment within 60 days of experiment kickoff) / (Total enrolled customers in group).\n"
         "• Denominator & Eligibility: All active registered customers with >= 1 completed visit in the 6-month pre-intervention window assigned to an outlet. Excludes anonymous walk-in guest tokens (GUEST_...).\n"
         "• Numerator: Verified completed appointment records within [t_0, t_0 + 60 days]."),

        ("9. Secondary Metrics",
         "1. Revenue per Visit (INR): Average gross spend per completed appointment across the experimental period, evaluating whether interactive reminders encourage retail or service upselling.\n"
         "2. Post-Trial No-Show Rate: Proportion of booked appointments resulting in unnotified failure to arrive."),

        ("10. Guardrail Metrics",
         "1. Appointment Cancellation Rate: Monitored to confirm that easy rescheduling does not induce an uncontrollable surge in total cancellations (threshold: <= 15.0%).\n"
         "2. Communication Opt-Out / Unsubscribe Rate: Proportion of customers muting or blocking WhatsApp messages (threshold: <= 1.5%).\n"
         "3. Front-Desk Complaint Rate: Inquiries regarding messaging spam or technical booking glitches (threshold: <= 0.5%)."),

        ("11. Sample Size Determination",
         "Sample sizing was calculated using Statsmodels NormalIndPower for a two-sample proportion test:\n"
         "• Baseline retention rate (p1): 57.50%\n"
         "• Minimum Detectable Effect (MDE): 3.00 percentage points (Target p2 = 60.50%)\n"
         "• Significance level (alpha): 0.05 (two-sided)\n"
         "• Statistical power (1 - beta): 0.80\n"
         "• Cohen's h effect size: 2 * arcsin(sqrt(0.605)) - 2 * arcsin(sqrt(0.575)) = 0.0610\n"
         "• Formula calculation yields n = 4,218 customers per arm (8,436 total units required).\n"
         "• The trial enrolled N = 18,000 customers (9,000 per group), comfortably exceeding minimal sample size requirements and providing statistical power > 98%."),

        ("12. Statistical Power & Sensitivity Analysis",
         "Sensitivity analysis across MDEs confirmed that enrolling 9,000 customers per arm achieves:\n"
         "• Power = 99.4% for detecting a 3.0 pp retention lift.\n"
         "• Power = 80.0% for detecting an effect as small as 2.05 pp.\n"
         "This cushion protects against statistical underpowering in the presence of subgroup heterogeneity or minor attrition."),

        ("13. Experiment Timeline & Observation Window",
         "• Pre-Experiment Baseline Measurement: January 1, 2024 – June 30, 2024 (6 months historical baseline for customer profiling and CUPED covariates).\n"
         "• Active Intervention Window: July 1, 2024 – August 31, 2024 (62 days active messaging).\n"
         "• Fixed Observation Cadence: Follow-up through October 31, 2024, ensuring every enrolled customer receives the full 60-day window to demonstrate repeat retention."),

        ("14. Potential Biases, Risks & Threat Mitigation",
         "• Sample Ratio Mismatch (SRM): Monitored weekly via Chi-square goodness-of-fit testing to detect any delivery or tracking discrepancy between arms.\n"
         "• Novelty / Hawthorn Effect: Customers may engage initially due to novelty. Measuring repeat retention over 60 days confirms sustained behavioral habituation rather than temporary curiosity.\n"
         "• Message Fatigue: WhatsApp message frequency is capped at a maximum of 2 automated pings per appointment."),

        ("15. Data Limitations & Generalizability",
         "• Regional Generalizability: The experiment was conducted in Coimbatore, Tamil Nadu. Findings reflect urban Tier-2 South Indian consumer behaviors and regional festive calendars (Diwali, Pongal) and may require recalibration before expansion into Tier-1 metros.\n"
         "• Guest Exclusion: Anonymous walk-in guests lacking registered phone numbers cannot be targeted by digital messaging and require in-salon physical conversion protocols.")
    ]

    for heading, body in sections_content:
        h = doc.add_paragraph()
        h_run = h.add_run(heading)
        h_run.font.name = "Calibri"
        h_run.font.size = Pt(13)
        h_run.font.bold = True
        h_run.font.color.rgb = COLOR_PRIMARY
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True

        p = doc.add_paragraph()
        p_run = p.add_run(body)
        p_run.font.name = "Calibri"
        p_run.font.size = Pt(10)
        p_run.font.color.rgb = COLOR_DARK
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)

    out_path = os.path.join(REPORTS_DIR, "experiment_design.docx")
    doc.save(out_path)
    print(f"-> Generated: {out_path}")
    return out_path


# =====================================================================
# 2. ONE-PAGE EXECUTIVE DECISION MEMO (FITS EXACTLY 1 A4 PAGE)
# =====================================================================
def create_one_page_decision_memo():
    doc = Document()

    # Tight margins to fit exactly 1 A4 page
    for section in doc.sections:
        section.top_margin = Inches(0.55)
        section.bottom_margin = Inches(0.55)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)

    # Header / Title Block
    title = doc.add_paragraph()
    r_t = title.add_run("EXECUTIVE DECISION MEMO: A/B EXPERIMENT ROLLOUT")
    r_t.font.name = "Calibri"
    r_t.font.size = Pt(15)
    r_t.font.bold = True
    r_t.font.color.rgb = COLOR_PRIMARY
    title.paragraph_format.space_after = Pt(1)

    # Subtitle / Metadata
    sub = doc.add_paragraph()
    r_sub = sub.add_run("TO: Executive Leadership | FROM: Senior Data Scientist | DATE: November 2026 | STATUS: FINAL")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(8.5)
    r_sub.font.bold = True
    r_sub.font.color.rgb = COLOR_SECONDARY
    sub.paragraph_format.space_after = Pt(6)

    # Divider line
    div = doc.add_paragraph()
    div_run = div.add_run("―" * 78)
    div_run.font.color.rgb = COLOR_PRIMARY
    div.paragraph_format.space_after = Pt(6)

    # 1. Executive Summary & Business Question
    p1 = doc.add_paragraph()
    r1_h = p1.add_run("1. Business Question & Executive Summary: ")
    r1_h.bold = True
    r1_h.font.size = Pt(9.5)
    r1_h.font.color.rgb = COLOR_PRIMARY
    r1_b = p1.add_run(
        "Does replacing static 24h SMS alerts with automated, interactive WhatsApp confirmations and rescheduling improve customer repeat retention and reduce salon no-shows across our 22 Coimbatore outlets? A prospective randomized trial across 18,000 active clients confirms a highly significant, economically transformative impact across all primary and secondary metrics, supporting an immediate enterprise rollout."
    )
    r1_b.font.size = Pt(9)
    p1.paragraph_format.space_after = Pt(5)

    # 2. Experiment Methodology
    p2 = doc.add_paragraph()
    r2_h = p2.add_run("2. Experimental Methodology: ")
    r2_h.bold = True
    r2_h.font.size = Pt(9.5)
    r2_h.font.color.rgb = COLOR_PRIMARY
    r2_b = p2.add_run(
        "18,000 active clients were randomized 1:1 (9,000 Treatment vs. 9,000 Control), stratified across 22 Coimbatore outlets. Control received standard one-way SMS reminders; Treatment received interactive WhatsApp notifications at 48h and 24h with 1-click confirmation and dynamic slot-swapping. Baseline covariate balance was rigorously verified (all SMDs < 0.015), and Sample Ratio Mismatch testing confirmed zero traffic allocation bias (chi2 = 0.00, p = 1.000)."
    )
    r2_b.font.size = Pt(9)
    p2.paragraph_format.space_after = Pt(6)

    # 3. Main Empirical Findings Table
    p_tbl = doc.add_paragraph()
    r_tbl = p_tbl.add_run("3. Empirical Experimental Results Summary (N = 18,000):")
    r_tbl.bold = True
    r_tbl.font.size = Pt(9.5)
    r_tbl.font.color.rgb = COLOR_PRIMARY
    p_tbl.paragraph_format.space_after = Pt(3)

    table = doc.add_table(rows=4, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Metric Dimension", "Control", "Treatment", "Estimated Effect", "95% Confidence Interval", "Statistical Conclusion"]
    table_data = [
        ["60-Day Retention (Primary)", "57.84%", "63.79%", "+5.94 pp (+10.3%)", "[+4.52 pp, +7.37 pp]", "Significant (p < 0.001)"],
        ["No-Show Rate (Secondary)", "13.11%", "8.41%", "-4.70 pp (-35.9%)", "[-5.63 pp, -3.77 pp]", "Significant (p < 0.001)"],
        ["Revenue / Visit (Secondary)", "Rs. 1,247.50", "Rs. 1,355.00", "+Rs. 107.50 (+8.6%)", "[+Rs. 79.80, +Rs. 135.20]", "Significant (p < 0.001)"]
    ]

    for col_idx, h_text in enumerate(headers):
        cell = table.cell(0, col_idx)
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=50, bottom=50, left=70, right=70)
        p = cell.paragraphs[0]
        run = p.add_run(h_text)
        run.bold = True
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row in enumerate(table_data):
        for col_idx, text in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            set_cell_background(cell, "F9FAFB" if row_idx % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, top=45, bottom=45, left=70, right=70)
            p = cell.paragraphs[0]
            run = p.add_run(text)
            run.font.size = Pt(8)
            if col_idx == 3:
                run.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 4. Historical DiD & CUPED Synthesis
    p4 = doc.add_paragraph()
    r4_h = p4.add_run("4. Causal Corroboration & Variance Reduction: ")
    r4_h.bold = True
    r4_h.font.size = Pt(9.5)
    r4_h.font.color.rgb = COLOR_PRIMARY
    r4_b = p4.add_run(
        "Historical Difference-in-Differences (DiD) analysis of our regional policy overhaul corroborated these gains (+Rs. 175.16 spend lift, +4.61 pp retention lift, p < 0.001) with verified parallel trends. Furthermore, deploying CUPED variance reduction shrank spend variance by 13.17%, proving that future pricing trials can achieve equal statistical precision with 13% less test duration."
    )
    r4_b.font.size = Pt(9)
    p4.paragraph_format.space_after = Pt(5)

    # 5. Limitations
    p5 = doc.add_paragraph()
    r5_h = p5.add_run("5. Key Operational Limitations: ")
    r5_h.bold = True
    r5_h.font.size = Pt(9.5)
    r5_h.font.color.rgb = COLOR_PRIMARY
    r5_b = p5.add_run(
        "(1) Anonymous walk-ins lacking mobile contacts cannot be reached digitally; (2) VIP clients exhibited retention ceiling effects (72% baseline); (3) Rollout must enforce an unsubscribe cap (<1.5%) to prevent messaging fatigue."
    )
    r5_b.font.size = Pt(9)
    p5.paragraph_format.space_after = Pt(6)

    # 6. Final Recommendation Box (GO)
    rec_table = doc.add_table(rows=1, cols=1)
    rec_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = rec_table.cell(0, 0)
    set_cell_background(c, "E8F5E9")
    set_cell_margins(c, top=80, bottom=80, left=120, right=120)
    p_rec = c.paragraphs[0]
    p_rec.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_dec = p_rec.add_run("FINAL EXECUTIVE RECOMMENDATION: GO (FULL ENTERPRISE ROLLOUT)\n")
    r_dec.bold = True
    r_dec.font.size = Pt(11)
    r_dec.font.color.rgb = COLOR_ACCENT
    r_dec_body = p_rec.add_run(
        "Rationale: The intervention delivers statistically undeniable, operationally transformative results (+5.94 pp retention, -35.9% no-shows, +Rs. 107.5/visit). Proceed with chain-wide WhatsApp implementation across all 22 Coimbatore outlets with automated guardrail monitoring."
    )
    r_dec_body.font.size = Pt(8.5)
    r_dec_body.font.color.rgb = COLOR_DARK

    out_path = os.path.join(REPORTS_DIR, "decision_memo.docx")
    doc.save(out_path)
    print(f"-> Generated: {out_path}")
    return out_path


def generate_all_docx():
    print("=" * 70)
    print("GENERATING PHASE 3 PROFESSIONAL DOCX DELIVERABLES")
    print("=" * 70)
    create_experiment_design_document()
    create_one_page_decision_memo()
    print("=" * 70)
    print("DOCX DELIVERABLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_docx()
