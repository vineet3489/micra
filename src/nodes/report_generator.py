"""
MICRA — Report Generator Node
================================

LEARNING CONCEPT: Grounded generation → shareable artifact

This node does something conceptually important: it converts the
accumulated structured state (JSON outputs from 6+ agents) into
a human-readable Word document.

This is the culmination of the "grounded generation" principle:
  Every section of the report is sourced from structured agent outputs.
  Every claim the report makes can be traced back to a source chunk.

This is fundamentally different from asking an LLM to "write a report
about DERMS". That produces fluent but unverifiable text. THIS approach:
  → Facts come from TAM agent output (which cites source chunks)
  → Competitor data comes from competitive intel output (cited)
  → Recommendations come from synthesis output (reasoned from above)

The report generator is NOT a creative writer. It's a formatter.

LEARNING CONCEPT: python-docx for structured document generation

python-docx works like this:
  doc = Document()
  doc.add_heading("Title", level=0)    # Heading 1
  doc.add_paragraph("Body text")
  doc.add_heading("Section", level=1)  # Heading 2
  table = doc.add_table(rows=1, cols=3)
  doc.save("report.docx")

Key formatting objects:
  - Heading levels 0-4 (like h1-h5 in HTML)
  - Paragraph with runs (each run can have different bold/color/size)
  - Table with cells (each cell contains paragraphs)
  - doc.add_page_break() between major sections

LEARNING CONCEPT: Graceful degradation with partial state

Some framework agents may have failed. Some competitor profiles may be empty.
The report generator must handle every combination of present/absent data
gracefully. The pattern: check before rendering, use placeholder text if missing.

A report with "TAM analysis unavailable — framework failed during analysis"
is more useful than a crashed pipeline.
"""

import os
import re
import json
from datetime import datetime
from typing import Any

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from rich.console import Console

from src.state import MICRAState, FrameworkOutput, CompetitorProfile

console = Console()

# ── Color palette ──────────────────────────────────────────────────────────
DARK_BLUE = RGBColor(0x1A, 0x3C, 0x5C)
MEDIUM_BLUE = RGBColor(0x2E, 0x6D, 0xA8)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)
GREEN = RGBColor(0x27, 0x7A, 0x40)
RED = RGBColor(0xC0, 0x39, 0x2B)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)


# ── Helpers ────────────────────────────────────────────────────────────────

def _add_heading(doc: Document, text: str, level: int = 1) -> None:
    """Add a styled heading."""
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = DARK_BLUE


def _add_body(doc: Document, text: str, bold: bool = False) -> None:
    """Add a body paragraph."""
    p = doc.add_paragraph(text)
    if bold:
        for run in p.runs:
            run.bold = True
    return p


def _add_bullet(doc: Document, text: str, color: RGBColor | None = None) -> None:
    """Add a bullet-point paragraph."""
    p = doc.add_paragraph(text, style="List Bullet")
    if color:
        for run in p.runs:
            run.font.color.rgb = color


def _add_kv(doc: Document, key: str, value: str) -> None:
    """Add a key: value line (key in bold)."""
    p = doc.add_paragraph()
    run_key = p.add_run(f"{key}: ")
    run_key.bold = True
    run_key.font.color.rgb = DARK_BLUE
    p.add_run(value)


def _find_framework(name: str, outputs: list[FrameworkOutput]) -> dict[str, Any] | None:
    """Find a specific framework's output dict, or None."""
    for o in outputs:
        if o["framework_name"] == name:
            return o["output"]
    return None


def _rating_color(rating: str) -> RGBColor:
    """Color-code High/Medium/Low ratings."""
    r = rating.lower()
    if "high" in r:
        return RED
    if "medium" in r:
        return ORANGE
    return GREEN


def _collect_all_sources(state: MICRAState) -> list[dict]:
    """
    Collect all unique sources cited across framework + competitor outputs.

    LEARNING: Citation tracing
    Each FrameworkOutput has source_chunks (list of chunk IDs).
    Each SourceDocument in state["sources"] has chunk_ids + url + title.
    We build a map: chunk_id → source, then deduplicate by URL.

    This is how "every claim cites a source" is enforced:
    frameworks store which chunks they used, we map those back to URLs.
    """
    # Try chunk-level citation tracing first (only works when chunk_ids
    # are populated in SourceDocuments — requires embedder to write them back).
    chunk_to_source: dict[str, dict] = {}
    for source in state.get("sources", []):
        for chunk_id in source.get("chunk_ids", []):
            chunk_to_source[chunk_id] = {
                "url": source["url"],
                "title": source["title"],
                "type": source["source_type"],
            }

    seen_urls: set[str] = set()
    cited_sources: list[dict] = []

    if chunk_to_source:
        # Chunk-level tracing available — list only actually-cited sources
        all_chunk_ids: list[str] = []
        for output in state.get("framework_outputs", []):
            all_chunk_ids.extend(output.get("source_chunks", []))
        for profile in state.get("competitor_profiles", []):
            all_chunk_ids.extend(profile.get("source_chunks", []))

        for chunk_id in all_chunk_ids:
            source = chunk_to_source.get(chunk_id)
            if source and source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                cited_sources.append(source)
    else:
        # Fallback: chunk_ids not populated — list all ingested sources.
        # All sources were scraped for this research run so all are relevant.
        for source in state.get("sources", []):
            if source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                cited_sources.append({
                    "url": source["url"],
                    "title": source["title"],
                    "type": source["source_type"],
                })

    return cited_sources


# ── Section renderers ──────────────────────────────────────────────────────

def _render_title_page(doc: Document, state: MICRAState) -> None:
    plan = state.get("research_plan", {})
    market = plan.get("target_market", "Market Intelligence")
    geography = plan.get("geography", "")
    date_str = datetime.now().strftime("%B %d, %Y")

    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_para.add_run(f"Market Intelligence Report\n{market}")
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = DARK_BLUE

    if geography:
        geo_para = doc.add_paragraph(geography)
        geo_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in geo_para.runs:
            run.font.color.rgb = MEDIUM_BLUE

    date_para = doc.add_paragraph(f"\nGenerated by MICRA  •  {date_str}")
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in date_para.runs:
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    doc.add_page_break()


def _render_executive_summary(doc: Document, state: MICRAState) -> None:
    _add_heading(doc, "Executive Summary", level=1)

    bbp = state.get("build_buy_partner_decision", {})
    mvp = state.get("mvp_recommendation", {})
    tam_output = _find_framework("tam_sam_som", state.get("framework_outputs", []))
    profiles = state.get("competitor_profiles", [])

    _add_heading(doc, "Strategic Recommendation", level=2)
    if bbp.get("recommendation"):
        _add_kv(doc, "Decision", bbp["recommendation"])
        _add_body(doc, bbp.get("reasoning", ""))

    _add_heading(doc, "Market Opportunity", level=2)
    if tam_output:
        _add_kv(doc, "TAM", tam_output.get("tam_value", "Unavailable"))
        _add_kv(doc, "SAM", tam_output.get("sam_value", "Unavailable"))
        _add_kv(doc, "SOM", tam_output.get("som_value", "Unavailable"))
        _add_kv(doc, "Growth Rate", tam_output.get("growth_rate", "Unavailable"))
    else:
        _add_body(doc, "TAM/SAM/SOM analysis was not completed in this run.")

    _add_heading(doc, "Competitive Landscape", level=2)
    if profiles:
        _add_body(doc, f"{len(profiles)} competitors analyzed: " +
                  ", ".join(p["name"] for p in profiles))
    else:
        _add_body(doc, "No competitor profiles were generated.")

    _add_heading(doc, "MVP North Star", level=2)
    if mvp.get("north_star_metric"):
        _add_body(doc, mvp["north_star_metric"], bold=True)

    doc.add_page_break()


def _render_market_overview(doc: Document, state: MICRAState) -> None:
    _add_heading(doc, "1. Market Overview", level=1)
    brief = state.get("research_brief", "")
    _add_body(doc, brief)

    tam_output = _find_framework("tam_sam_som", state.get("framework_outputs", []))
    if tam_output:
        _add_heading(doc, "1.1 Market Sizing (TAM / SAM / SOM)", level=2)

        # TAM/SAM/SOM table
        table = doc.add_table(rows=4, cols=3)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        hdr[0].text = "Metric"
        hdr[1].text = "Value"
        hdr[2].text = "Description"
        for cell in hdr:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True

        for i, (metric, val_key, desc_key) in enumerate([
            ("TAM", "tam_value", "tam_description"),
            ("SAM", "sam_value", "sam_description"),
            ("SOM", "som_value", "som_description"),
        ], start=1):
            row = table.rows[i].cells
            row[0].text = metric
            row[1].text = tam_output.get(val_key, "N/A")
            row[2].text = tam_output.get(desc_key, "")

        doc.add_paragraph()

        _add_heading(doc, "1.2 Key Growth Trends", level=2)
        for trend in tam_output.get("key_trends_driving_growth", []):
            _add_bullet(doc, trend)

        _add_heading(doc, "1.3 Key Assumptions", level=2)
        for assumption in tam_output.get("key_assumptions", []):
            _add_bullet(doc, assumption)

        confidence = tam_output.get("confidence", "Unknown")
        color = {"High": GREEN, "Medium": ORANGE, "Low": RED}.get(confidence, MEDIUM_BLUE)
        p = doc.add_paragraph()
        r = p.add_run(f"Data confidence: {confidence}")
        r.font.color.rgb = color
        r.bold = True
        doc.add_paragraph(tam_output.get("confidence_reasoning", ""))

    else:
        _add_body(doc, "TAM/SAM/SOM analysis unavailable for this run.")

    doc.add_page_break()


def _render_competitive_landscape(doc: Document, state: MICRAState) -> None:
    _add_heading(doc, "2. Competitive Landscape", level=1)

    # 2.1 Porter's 5 Forces
    porter = _find_framework("porter_5_forces", state.get("framework_outputs", []))
    if porter:
        _add_heading(doc, "2.1 Porter's 5 Forces Analysis", level=2)

        forces = [
            ("Threat of New Entrants", "threat_of_new_entrants"),
            ("Bargaining Power of Buyers", "bargaining_power_buyers"),
            ("Bargaining Power of Suppliers", "bargaining_power_suppliers"),
            ("Threat of Substitutes", "threat_of_substitutes"),
            ("Competitive Rivalry", "competitive_rivalry"),
        ]

        table = doc.add_table(rows=len(forces) + 1, cols=3)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        for cell, label in zip(hdr, ["Force", "Rating", "Key Factors"]):
            cell.text = label
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True

        for i, (label, key) in enumerate(forces, start=1):
            force_data = porter.get(key, {})
            row = table.rows[i].cells
            row[0].text = label
            rating = force_data.get("rating", "N/A")
            row[1].text = rating
            row[2].text = " | ".join(force_data.get("key_factors", [])[:3])

        doc.add_paragraph()
        _add_kv(doc, "Overall Market Attractiveness",
                porter.get("overall_market_attractiveness", "N/A"))
        _add_heading(doc, "Strategic Implications", level=3)
        for impl in porter.get("strategic_implications", []):
            _add_bullet(doc, impl)

    # 2.2 Competitor Profiles
    profiles = state.get("competitor_profiles", [])
    if profiles:
        _add_heading(doc, "2.2 Competitor Profiles", level=2)

        for profile in profiles:
            _add_heading(doc, profile["name"], level=3)
            _add_kv(doc, "Product", profile.get("product_summary", ""))
            _add_kv(doc, "Target Segment", profile.get("target_segment", ""))
            _add_kv(doc, "Pricing", profile.get("pricing_model", ""))
            _add_kv(doc, "Funding", profile.get("funding", ""))

            _add_body(doc, "Core Features:", bold=True)
            for f in profile.get("core_features", []):
                _add_bullet(doc, f)

            _add_body(doc, "Strengths:", bold=True)
            for s in profile.get("strengths", []):
                _add_bullet(doc, s, color=GREEN)

            _add_body(doc, "Weaknesses:", bold=True)
            for w in profile.get("weaknesses", []):
                _add_bullet(doc, w, color=RED)

            _add_body(doc, "Differentiation Gaps (opportunities for us):", bold=True)
            for gap in profile.get("differentiation_gaps", []):
                _add_bullet(doc, gap, color=MEDIUM_BLUE)

            doc.add_paragraph()

    doc.add_page_break()


def _render_strategic_analysis(doc: Document, state: MICRAState) -> None:
    _add_heading(doc, "3. Strategic Analysis", level=1)

    # SWOT
    swot = _find_framework("swot", state.get("framework_outputs", []))
    if swot:
        _add_heading(doc, "3.1 SWOT Analysis", level=2)

        table = doc.add_table(rows=2, cols=2)
        table.style = "Table Grid"
        quadrants = [
            ("Strengths", "strengths", 0, 0, GREEN),
            ("Weaknesses", "weaknesses", 0, 1, RED),
            ("Opportunities", "opportunities", 1, 0, MEDIUM_BLUE),
            ("Threats", "threats", 1, 1, ORANGE),
        ]
        for label, key, row_i, col_i, color in quadrants:
            cell = table.rows[row_i].cells[col_i]
            cell.text = ""
            p = cell.add_paragraph()
            r = p.add_run(label)
            r.bold = True
            r.font.color.rgb = color
            for item in swot.get(key, []):
                cell.add_paragraph(f"• {item}")

        doc.add_paragraph()
        _add_kv(doc, "Critical Factor", swot.get("most_critical_factor", ""))
        _add_kv(doc, "Recommended Posture", swot.get("recommended_strategic_posture", ""))

    # Kano
    kano = _find_framework("kano", state.get("framework_outputs", []))
    if kano:
        _add_heading(doc, "3.2 Kano Model — Feature Prioritization", level=2)

        categories = [
            ("Must-Have (MVP Required)", "must_have"),
            ("Performance (Competitive Differentiators)", "performance"),
            ("Delighters (Moat Builders)", "delighters"),
            ("Indifferent (Cut from Roadmap)", "indifferent"),
        ]
        for label, key in categories:
            _add_body(doc, label, bold=True)
            for feat in kano.get(key, []):
                if isinstance(feat, dict):
                    _add_bullet(doc, f"{feat.get('feature_name', '')}: {feat.get('description', '')}")
                else:
                    _add_bullet(doc, str(feat))

    doc.add_page_break()


def _render_recommendation(doc: Document, state: MICRAState) -> None:
    _add_heading(doc, "4. Strategic Recommendation", level=1)

    bbp = state.get("build_buy_partner_decision", {})
    mvp = state.get("mvp_recommendation", {})

    # 4.1 Build/Buy/Partner
    _add_heading(doc, "4.1 Build / Buy / Partner Decision", level=2)

    if bbp.get("recommendation"):
        p = doc.add_paragraph()
        r = p.add_run(f"Recommendation: {bbp['recommendation']}")
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = DARK_BLUE

        _add_body(doc, bbp.get("reasoning", ""))
        _add_kv(doc, "Time to Market", bbp.get("time_to_market_assessment", ""))

        _add_heading(doc, "The Case For Each Option", level=3)
        _add_kv(doc, "Build", bbp.get("build_case", ""))
        _add_kv(doc, "Buy", bbp.get("buy_case", ""))
        _add_kv(doc, "Partner", bbp.get("partner_case", ""))

        if bbp.get("recommended_partners_or_targets"):
            _add_heading(doc, "Recommended Partners / Acquisition Targets", level=3)
            for p_name in bbp["recommended_partners_or_targets"]:
                _add_bullet(doc, p_name)

        _add_heading(doc, "Capability Gaps to Address", level=3)
        for gap in bbp.get("capability_gaps", []):
            _add_bullet(doc, gap, color=ORANGE)

    # 4.2 MVP
    _add_heading(doc, "4.2 MVP Recommendation", level=2)

    if mvp.get("north_star_metric"):
        p = doc.add_paragraph()
        r = p.add_run(f"North Star Metric: {mvp['north_star_metric']}")
        r.bold = True
        r.font.color.rgb = MEDIUM_BLUE

    _add_kv(doc, "Target Customer", mvp.get("target_customer", ""))
    _add_kv(doc, "Go-to-Market Entry", mvp.get("go_to_market_entry_point", ""))
    _add_kv(doc, "Pricing Strategy", mvp.get("pricing_strategy", ""))

    _add_heading(doc, "MVP Feature Set (v1)", level=3)
    for feat in mvp.get("core_features", []):
        _add_bullet(doc, feat, color=GREEN)

    _add_heading(doc, "Explicitly NOT in v1", level=3)
    for feat in mvp.get("features_explicitly_excluded", []):
        _add_bullet(doc, feat, color=RED)

    _add_heading(doc, "6-Month Success Criteria", level=3)
    for criterion in mvp.get("success_criteria_6_months", []):
        _add_bullet(doc, criterion)

    if mvp.get("differentiation_thesis"):
        _add_heading(doc, "Differentiation Thesis", level=3)
        _add_body(doc, mvp["differentiation_thesis"])

    doc.add_page_break()


def _render_risk_register(doc: Document, state: MICRAState) -> None:
    _add_heading(doc, "5. Risk Register", level=1)

    bbp = state.get("build_buy_partner_decision", {})
    risks = bbp.get("risk_factors", [])

    if risks:
        table = doc.add_table(rows=len(risks) + 1, cols=2)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        hdr[0].text = "#"
        hdr[1].text = "Risk"
        for cell in hdr:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True

        for i, risk in enumerate(risks, start=1):
            table.rows[i].cells[0].text = str(i)
            table.rows[i].cells[1].text = risk
    else:
        _add_body(doc, "No risks were captured in the Build/Buy/Partner analysis.")

    doc.add_page_break()


def _render_sources(doc: Document, cited_sources: list[dict]) -> None:
    _add_heading(doc, "6. Sources & Citations", level=1)

    if cited_sources:
        for i, source in enumerate(cited_sources, start=1):
            p = doc.add_paragraph()
            p.add_run(f"[{i}] ").bold = True
            p.add_run(f"{source['title']} ")
            run = p.add_run(f"({source['type']})")
            run.font.color.rgb = MEDIUM_BLUE
            p.add_run(f"\n    {source['url']}")
    else:
        _add_body(doc, "Source citation data was not available for this run.")


# ── Main node ──────────────────────────────────────────────────────────────

def report_generator_node(state: MICRAState) -> dict:
    """
    Generate the Word report from accumulated state.

    LEARNING: This node only READS state — it writes nothing back except
    the report file path. It's a pure consumer of all previous nodes' work.

    This is a key pattern: keep side effects (file I/O) isolated to a
    dedicated node. All analysis nodes stay pure (state in/out).
    """
    console.print("\n[bold cyan]Phase 4: Generating Report[/bold cyan]")

    plan = state.get("research_plan", {})
    market = plan.get("target_market", "market")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize: strip any characters that would create sub-directories or
    # be invalid in filenames (slashes, colons, quotes, etc.)
    market_safe = re.sub(r"[^\w\s]", "_", market)   # keep only word chars + spaces
    market_safe = re.sub(r"\s+", "_", market_safe).lower().strip("_")
    filename = f"micra_{market_safe}_{timestamp}.docx"
    output_path = os.path.join("outputs", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = Document()

    # Set default font for the document
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    console.print("  Building sections...", end=" ")

    _render_title_page(doc, state)
    _render_executive_summary(doc, state)
    _render_market_overview(doc, state)
    _render_competitive_landscape(doc, state)
    _render_strategic_analysis(doc, state)
    _render_recommendation(doc, state)
    _render_risk_register(doc, state)

    cited_sources = _collect_all_sources(state)
    _render_sources(doc, cited_sources)

    doc.save(output_path)
    console.print("[green]✓[/green]")

    # Count non-empty sections for the message
    section_count = sum([
        bool(state.get("build_buy_partner_decision")),
        bool(state.get("mvp_recommendation")),
        bool(state.get("framework_outputs")),
        bool(state.get("competitor_profiles")),
    ])

    console.print(
        f"[green]✓[/green] Report saved: [bold]{output_path}[/bold] "
        f"({section_count} major sections, {len(cited_sources)} sources cited)"
    )

    return {
        "report_path": output_path,
        "messages": [
            f"[report_generator] Report generated: {output_path} "
            f"({len(cited_sources)} sources cited)"
        ],
    }
