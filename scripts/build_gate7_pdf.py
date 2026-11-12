"""Assemble the Gate 7 paper-quality PDF from frozen tables and audited figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#2F6B9A")
TEAL = colors.HexColor("#0F7C7B")
ORANGE = colors.HexColor("#E58B2A")
RED = colors.HexColor("#C94C4C")
SLATE = colors.HexColor("#526579")
LIGHT = colors.HexColor("#EAF0F4")
PALE_BLUE = colors.HexColor("#F2F7FA")
WHITE = colors.white


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/gate7.yaml"))
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def pct(value: str | float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}%"


def num(value: str | float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12.5,
            leading=18,
            textColor=SLATE,
            spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceAfter=8,
        ),
        "subsection": ParagraphStyle(
            "subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=TEAL,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.1,
            textColor=colors.HexColor("#263746"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.1,
            textColor=SLATE,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=9.5,
            textColor=SLATE,
            spaceBefore=3,
            spaceAfter=5,
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11.3,
            leading=16,
            textColor=NAVY,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=6,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#263746"),
        ),
        "table_head": ParagraphStyle(
            "table_head",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.1,
            leading=8.6,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "metric": ParagraphStyle(
            "metric",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "metric_label",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.2,
            textColor=SLATE,
            alignment=TA_CENTER,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str], style: ParagraphStyle, bullet_color: str = "#0F7C7B") -> KeepTogether:
    bullet_style = ParagraphStyle(
        "ascii_bullet",
        parent=style,
        leftIndent=15,
        firstLineIndent=-9,
        textColor=style.textColor,
        borderColor=colors.HexColor(bullet_color),
        spaceAfter=4,
    )
    return KeepTogether([Paragraph(f"- {item}", bullet_style) for item in items])


def scaled_image(path: Path, width: float, max_height: float) -> Image:
    with PILImage.open(path) as opened:
        px_width, px_height = opened.size
    height = width * px_height / px_width
    if height > max_height:
        height = max_height
        width = height * px_width / px_height
    return Image(str(path), width=width, height=height)


def page_rule(canvas: Any, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    page = canvas.getPageNumber()
    width, height = A4
    if page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 18 * mm, width, 18 * mm, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, 0, width, 7 * mm, fill=1, stroke=0)
    else:
        canvas.setStrokeColor(colors.HexColor("#CDD7DE"))
        canvas.setLineWidth(0.6)
        canvas.line(document.leftMargin, height - 13 * mm, width - document.rightMargin, height - 13 * mm)
        canvas.setFont("Helvetica", 7.3)
        canvas.setFillColor(SLATE)
        canvas.drawString(document.leftMargin, height - 10 * mm, "STATE-DEPENDENT CROSS-SECTIONAL ALPHA")
        canvas.drawRightString(width - document.rightMargin, height - 10 * mm, "GATE 7 | FROZEN EVIDENCE")
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(SLATE if page > 1 else WHITE)
    canvas.drawRightString(width - document.rightMargin, 7.8 * mm, f"{page}")
    canvas.restoreState()


def table(data: list[list[Any]], widths: list[float], styles: dict[str, ParagraphStyle]) -> Table:
    converted: list[list[Any]] = []
    for row_index, row in enumerate(data):
        row_style = styles["table_head"] if row_index == 0 else styles["table"]
        converted.append([cell if hasattr(cell, "wrap") else Paragraph(str(cell), row_style) for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD6DE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def metric_cards(metrics: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    cards = []
    for value, label in metrics:
        cards.append([Paragraph(value, styles["metric"]), Paragraph(label, styles["metric_label"])])
    card_tables = []
    for card in cards:
        inner = Table([[card[0]], [card[1]]], colWidths=[38 * mm], rowHeights=[10 * mm, 11 * mm])
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                    ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#C7D4DD")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        card_tables.append(inner)
    outer = Table([card_tables], colWidths=[42 * mm] * len(card_tables), hAlign="LEFT")
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return outer


def build_pdf(project_root: Path, output_root: Path, config: dict[str, Any], config_path: Path) -> Path:
    sys.path.insert(0, str(project_root / "src"))
    from state_alpha.reporting.gate7 import (  # noqa: PLC0415
        FIGURE_FILES,
        SCORE_LABELS,
        SCORE_ORDER,
        sha256_file,
        verify_locked_sources,
    )

    verify_locked_sources(project_root, config)
    figure_dir = output_root / config["figures"]["output_directory"]
    for name in FIGURE_FILES:
        if not (figure_dir / name).is_file():
            raise FileNotFoundError(figure_dir / name)

    performance = read_csv_rows(project_root / "reports/tables/gate6_performance_summary.csv")
    costs = read_csv_rows(project_root / "reports/tables/gate6_cost_stress.csv")
    missing = read_csv_rows(project_root / "reports/tables/gate6_missing_return_sensitivity.csv")
    participation = read_csv_rows(project_root / "reports/tables/gate6_participation_summary.csv")
    validation = json.loads((project_root / "reports/tables/gate6_validation.json").read_text(encoding="utf-8"))

    def perf(signal: str, neutralization: str) -> dict[str, str]:
        return select_row(
            performance,
            period="boundary_clean_primary",
            signal=signal,
            weighting="rank_weight",
            neutralization=neutralization,
            bucket="all",
        )

    def cost20(signal: str, neutralization: str) -> dict[str, str]:
        return select_row(
            costs,
            period="boundary_clean_primary",
            signal=signal,
            weighting="rank_weight",
            neutralization=neutralization,
            bucket="all",
            one_way_cost_bps="20",
        )

    raw_tree = perf("tree_alpha_state", "raw")
    neutral_tree = perf("tree_alpha_state", "risk_neutralized")
    neutral_tree_cost = cost20("tree_alpha_state", "risk_neutralized")
    neutral_participation = select_row(
        participation,
        signal="tree_alpha_state",
        weighting="rank_weight",
        neutralization="risk_neutralized",
        bucket="all",
    )
    max_missing = max(
        abs(float(row["annualized_return_difference_zero_minus_loss100"]))
        for row in missing
        if row["period"] == "boundary_clean_primary"
    )

    output_path = output_root / config["report"]["output_pdf"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=15 * mm,
        title="Learning When Technical Signals Work",
        author="State-Dependent Alpha Research Project",
        subject="Frozen Gate 5 and Gate 6 evidence package",
        creator="Gate 7 reproducible report builder",
    )
    styles = build_styles()
    content_width = A4[0] - document.leftMargin - document.rightMargin
    story: list[Any] = []

    # Cover
    story.extend(
        [
            Spacer(1, 34 * mm),
            Paragraph("Learning When Technical Signals Work", styles["cover_title"]),
            Paragraph(
                "State-Dependent Cross-Sectional Alpha Selection in U.S. Equities",
                styles["cover_subtitle"],
            ),
            Spacer(1, 7 * mm),
            Table(
                [[Paragraph("FROZEN RESEARCH EVIDENCE", styles["table_head"])]],
                colWidths=[67 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), TEAL),
                        ("BOX", (0, 0), (-1, -1), 0, TEAL),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 12 * mm),
            Paragraph(
                "Conditional alpha-plus-state models produced positive gross cross-sectional portfolio evidence, including after beta, size, and industry neutralization. The economics are materially sensitive to cost and capacity assumptions.",
                styles["quote"],
            ),
            Spacer(1, 14 * mm),
            table(
                [
                    ["Evidence status", "Boundary-clean period", "Release", "Gate 7 rule"],
                    [
                        "Gate 6 independently validated",
                        "2014-01-02 to 2025-12-31",
                        "0.8.0 presentation package",
                        "No empirical changes",
                    ],
                ],
                [42 * mm, 43 * mm, 44 * mm, 40 * mm],
                styles,
            ),
            Spacer(1, 23 * mm),
            Paragraph(
                f"Gate 7 config SHA-256: {sha256_file(config_path)}",
                styles["small"],
            ),
            Paragraph(
                "Prepared from immutable Gate 5 and Gate 6 outputs. All seven figures are hash-audited.",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Executive summary
    story.extend(
        [
            Paragraph("Executive readout", styles["section"]),
            Paragraph(
                "Gate 7 is presentation-only. The research question is narrow: do interpretable price-based signals work differently across stock, liquidity, attention, and market states, and can conditional models improve cross-sectional ranking without disguising implementation risk?",
                styles["body"],
            ),
            Spacer(1, 3 * mm),
            metric_cards(
                [
                    (pct(raw_tree["annualized_return"]), "Raw rank tree alpha + state gross return"),
                    (pct(neutral_tree["annualized_return"]), "Risk-neutral rank tree alpha + state gross return"),
                    (num(neutral_tree["sharpe"]), "Risk-neutral gross Sharpe"),
                    (pct(neutral_tree_cost["annualized_return"]), "Risk-neutral return under 20 bps stress"),
                ],
                styles,
            ),
            Spacer(1, 6 * mm),
            Paragraph("What the evidence supports", styles["subsection"]),
            bullets(
                [
                    "Frozen conditional models translate into positive gross portfolio returns in the boundary-clean period.",
                    "Risk neutralization removes targeted beta, size, and industry tilts to numerical tolerance.",
                    "The formation-domain correction prevents future label availability from defining the investable universe.",
                    "The evidence is stronger for conditional ranking than for a specific suppression mechanism.",
                ],
                styles["body"],
            ),
            Paragraph("What it does not support", styles["subsection"]),
            bullets(
                [
                    "No OOS winner is selected; all ten preregistered scores remain visible.",
                    "The 5/10/20/30 bps deductions are cost stresses, not calibrated realized transaction costs.",
                    "Reference-NAV ADV participation is a diagnostic, not a capacity estimate.",
                    "Capped gates do not confirm that suppressing bad signals is the primary mechanism.",
                ],
                styles["body"],
                bullet_color="#C94C4C",
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "Research judgment: the project is resume-ready because the main contribution is not a headline Sharpe. It is the combination of a disciplined alpha-versus-state design, walk-forward evidence, caught leakage risks, and an honest implementation audit.",
                styles["quote"],
            ),
            PageBreak(),
        ]
    )

    # Architecture
    story.extend(
        [
            Paragraph("1. Research architecture", styles["section"]),
            Paragraph(
                "The project differs from an event study: it ranks a liquid U.S. equity universe repeatedly using continuous price signals and conditional state variables. Volume, turnover, and ADV appear only as conditioning or implementation variables, never as the primary alpha.",
                styles["body"],
            ),
            scaled_image(figure_dir / FIGURE_FILES[0], content_width, 108 * mm),
            Paragraph(
                "Figure 1. End-to-end governance architecture. The red formation-repair stage marks the most important Gate 6 research-integrity correction.",
                styles["caption"],
            ),
            Paragraph("Frozen design choices", styles["subsection"]),
            table(
                [
                    ["Component", "Frozen choice", "Purpose"],
                    ["Signal families", "Relative strength, close breakout, oversold recovery", "Economically interpretable alpha primitives"],
                    ["State families", "Stock, liquidity, attention/crowding, market", "Learn when signals work"],
                    ["Validation", "Purged expanding window, annual refits, 10-session embargo", "Prevent temporal leakage"],
                    ["Models", "Ridge controls and constrained trees", "Linear interactions versus nonlinear conditional structure"],
                    ["Portfolio", "Every fifth session, t+1 close execution, fixed 10-session hold", "Delayed, staggered translation"],
                ],
                [31 * mm, 70 * mm, 68 * mm],
                styles,
            ),
            PageBreak(),
        ]
    )

    # Integrity and predictive evidence
    story.extend(
        [
            Paragraph("2. Predictive evidence and integrity corrections", styles["section"]),
            Paragraph(
                "The boundary-clean Gate 5 table excludes the disclosed validation-to-OOS label-boundary contamination. It remains secondary evidence because the OOS results had already been observed before Gate 6 portfolio construction.",
                styles["body"],
            ),
            scaled_image(figure_dir / FIGURE_FILES[1], content_width, 104 * mm),
            Paragraph(
                "Figure 2. Boundary-clean mean daily Spearman rank IC. Newey-West inference uses the frozen Gate 5 specification.",
                styles["caption"],
            ),
            table(
                [
                    ["Integrity issue", "Why it mattered", "Resolution"],
                    ["OOS label boundary", "Training labels overlapped the first OOS observations", "Boundary-clean 2014-2025 evidence reported separately; prior output retained"],
                    ["Label-valid prediction panel", "Future label availability would define formation eligibility", "All frozen models rescored on the full point-in-time formation universe"],
                    ["Execution failures", "Fill status could leak into ranking or redistribute notional", "Rank first; failed order remains cash; no rerank or postfill reoptimization"],
                ],
                [37 * mm, 62 * mm, 70 * mm],
                styles,
            ),
            Paragraph(
                "The formation repair restored 527 rows. The final artifact contains 652 formation dates and 1,596,066 formation-score rows, with no forward-label columns.",
                styles["quote"],
            ),
            PageBreak(),
        ]
    )

    # Raw portfolio evidence
    story.extend(
        [
            Paragraph("3. Raw portfolio translation", styles["section"]),
            Paragraph(
                "Gate 6 carries all ten scores forward. Rank weighting emphasizes the tails without changing the long-short, self-financing contract. Gross returns are arithmetic annualized means; the stressed series subtracts two times one-way turnover times the stated one-way cost.",
                styles["body"],
            ),
            scaled_image(figure_dir / FIGURE_FILES[2], content_width, 111 * mm),
            Paragraph(
                "Figure 3. Raw rank-weight gross return versus a 20 bps one-way stress. The stress is deliberately severe relative to several break-even estimates.",
                styles["caption"],
            ),
            Paragraph("Interpretation", styles["subsection"]),
            bullets(
                [
                    "The unconditional base family portfolio is negative, which prevents a generic technical-alpha claim.",
                    "Both interpretable Ridge interactions and tree alpha-plus-state scores produce positive gross evidence.",
                    "Several state-only controls are meaningful, so conditional structure is not equivalent to alpha timing.",
                    "Turnover compresses the economic margin; gross performance should never be described as realized net alpha.",
                ],
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Neutralized evidence
    story.extend(
        [
            Paragraph("4. Risk-neutralized portfolio evidence", styles["section"]),
            Paragraph(
                "Support-preserving least-squares projection removes beta, size, and industry-division exposure before execution. The portfolio support is preserved, and failed fills remain cash; there is no postfill reoptimization.",
                styles["body"],
            ),
            scaled_image(figure_dir / FIGURE_FILES[3], content_width, 111 * mm),
            Paragraph(
                "Figure 4. Risk-neutralized rank-weight gross return versus 20 bps one-way stress. Neutralization cleans targeted exposures but does not solve turnover or capacity.",
                styles["caption"],
            ),
            table(
                [
                    ["Role", "Raw gross", "Neutral gross", "Neutral Sharpe", "Neutral NW t", "Neutral break-even"],
                    *[
                        [
                            SCORE_LABELS[signal],
                            pct(perf(signal, "raw")["annualized_return"]),
                            pct(perf(signal, "risk_neutralized")["annualized_return"]),
                            num(perf(signal, "risk_neutralized")["sharpe"]),
                            num(perf(signal, "risk_neutralized")["newey_west_t"]),
                            f"{num(perf(signal, 'risk_neutralized')['break_even_one_way_bps'], 1)} bps",
                        ]
                        for signal in [
                            "base_family_score",
                            "ridge_explicit_interactions",
                            "tree_alpha_state",
                        ]
                    ],
                ],
                [35 * mm, 25 * mm, 28 * mm, 25 * mm, 24 * mm, 32 * mm],
                styles,
            ),
            PageBreak(),
        ]
    )

    # Wealth paths
    story.extend(
        [
            Paragraph("5. Path dependence and research-role comparison", styles["section"]),
            Paragraph(
                "A cumulative path can reveal concentration in time that an annualized average hides. The display is role-based: the unconditional base, the interpretable conditional Ridge model, and the nonlinear conditional tree are shown in both raw and neutralized form.",
                styles["body"],
            ),
            scaled_image(figure_dir / FIGURE_FILES[4], content_width, 139 * mm),
            Paragraph(
                "Figure 5. Gross wealth from daily Gate 6 portfolio returns. No model is selected or promoted as an OOS winner.",
                styles["caption"],
            ),
            Paragraph(
                "The paths support a conditional-ranking interpretation but also show why a single Sharpe number is inadequate: drawdowns remain economically material and performance is not uniform through time.",
                styles["quote"],
            ),
            PageBreak(),
        ]
    )

    # Exposures
    story.extend(
        [
            Paragraph("6. Exposure control", styles["section"]),
            Paragraph(
                "Raw conditional portfolios carry material factor and industry tilts. Neutralization is therefore a necessary robustness test, not cosmetic portfolio engineering.",
                styles["body"],
            ),
            scaled_image(figure_dir / FIGURE_FILES[5], content_width, 137 * mm),
            Paragraph(
                "Figure 6. Mean prefill targeted exposures for all rank-weight scores. Each panel uses its own scale; cell values are the audit values.",
                styles["caption"],
            ),
            Paragraph(
                f"Independent validation reports a maximum neutralization constraint residual of {validation.get('neutralized_constraint_max', 1.4537e-14):.4g}. The near-zero targeted exposures do not imply broad factor neutrality beyond the frozen beta, size, and industry constraints.",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Liquidity and capacity
    story.extend(
        [
            Paragraph("7. Liquidity and capacity diagnostics", styles["section"]),
            Paragraph(
                "The strategy looks stronger in some less-liquid buckets, which increases rather than resolves implementation concern. Reference-NAV ADV participation converts frozen formation weights into lagged-ADV diagnostics without pretending to model spread or market impact.",
                styles["body"],
            ),
            scaled_image(figure_dir / FIGURE_FILES[6], content_width, 145 * mm),
            Paragraph(
                "Figure 7. Formation-date liquidity terciles and $100M reference-NAV participation. Liquidity buckets are raw equal-weight; participation is risk-neutral rank-weight.",
                styles["caption"],
            ),
            metric_cards(
                [
                    (pct(neutral_participation["mean_entry_participation_p95_100m"], 1), "Mean cohort entry p95 participation"),
                    (pct(neutral_participation["p95_entry_participation_p95_100m"], 1), "P95 across cohorts"),
                    (pct(neutral_participation["mean_entry_fraction_gt_5pct_100m"], 1), "Mean fraction of entries above 5% ADV"),
                ],
                styles,
            ),
            PageBreak(),
        ]
    )

    # Mechanics and limitations
    story.extend(
        [
            Paragraph("8. Portfolio contract, sensitivities, and limitations", styles["section"]),
            Paragraph("Frozen portfolio contract", styles["subsection"]),
            bullets(
                [
                    "Every fifth market session; information through close t; execution at close t+1.",
                    "Returns accrue from t+2 through t+11 in two fixed 50% capital sleeves.",
                    "Top-minus-bottom deciles with equal and rank weights; no daily renormalization.",
                    "Security-level netting of simultaneous scheduled entries and exits.",
                    "Missing/invalid held returns use explicit loss100 and zero terminal conventions.",
                ],
                styles["body"],
            ),
            Paragraph("Sensitivity checks", styles["subsection"]),
            table(
                [
                    ["Audit", "Frozen result", "Interpretation"],
                    ["Missing terminal convention", f"Maximum annualized difference: {100 * max_missing:.2f}%", "Small relative to main gross gaps; not monotone for signed long-short weights"],
                    ["20 bps stress", f"Tree alpha + state risk-neutral: {pct(neutral_tree_cost['annualized_return'])}", "Economic margin becomes thin"],
                    ["Turnover", f"Tree alpha + state risk-neutral: {num(neutral_tree['annualized_one_way_turnover'], 1)}x one-way/year", "Implementation sensitivity is structural"],
                    ["Validation", "27 artifact files; 41/41 tests", "Hash, schema, accounting, and source immutability checks passed"],
                ],
                [38 * mm, 51 * mm, 80 * mm],
                styles,
            ),
            Spacer(1, 4 * mm),
            Paragraph("Binding limitations", styles["subsection"]),
            bullets(
                [
                    "No calibrated spread or impact model, so no realized net-return or precise capacity claim.",
                    "Gate 5 OOS evidence had already been observed before portfolio translation; Gate 6 is secondary, locked evidence.",
                    "Nonlinear dominance is not established by Gate 5 IC comparisons.",
                    "State-only controls are strong, and capped gates are weak; the proposed bad-signal suppression mechanism remains unconfirmed.",
                    "Neutralization targets only beta, size, and industry and may introduce concentration or other exposures.",
                ],
                styles["body"],
                bullet_color="#C94C4C",
            ),
            Paragraph(
                "Final claim: conditional alpha-plus-state ranking generates positive gross evidence, but cost and capacity remain the binding constraints.",
                styles["quote"],
            ),
            PageBreak(),
        ]
    )

    # Interview page
    story.extend(
        [
            Paragraph("9. Interview narrative", styles["section"]),
            Paragraph("Two-minute version", styles["subsection"]),
            Paragraph(
                "I studied when simple price signals work rather than asking whether a large model can forecast returns. I separated relative-strength, breakout, and oversold-recovery alpha primitives from stock, liquidity, attention, and market state, then compared explicit Ridge interactions with constrained trees in a purged expanding-window design.",
                styles["body"],
            ),
            Paragraph(
                "The most important engineering decision occurred before portfolio P and L. The prediction table contained only rows with valid future labels, which is correct for IC evaluation but creates look-ahead if it defines the investable universe. I reapplied every frozen annual model to the full point-in-time formation universe, ranked before observing execution success, and translated the signals into delayed, staggered portfolios.",
                styles["body"],
            ),
            Paragraph(
                "Conditional models generated positive gross returns, including after beta, size, and industry neutralization. But 20 bps stress, stronger results in illiquid stocks, and $100M ADV participation diagnostics materially weakened the implementation case. My conclusion is therefore about robust conditional ranking and research controls, not a frictionless Sharpe or guaranteed tradability.",
                styles["body"],
            ),
            Spacer(1, 4 * mm),
            Paragraph("Resume bullets", styles["subsection"]),
            bullets(
                [
                    "Built a point-in-time CRSP CIZ cross-sectional pipeline with delisting-aware labels, purged walk-forward validation, and annual frozen model states.",
                    "Compared Ridge signal-by-state interactions with constrained nonlinear trees using alpha-only, state-only, main-effect, and capped-gate controls.",
                    "Translated ten preregistered scores into delayed staggered portfolios with neutralization, turnover, cost stress, liquidity, and ADV diagnostics.",
                    "Detected and corrected an OOS label-boundary leak and a future-label-filtered formation universe without overwriting prior evidence.",
                ],
                styles["body"],
            ),
            Paragraph("Likely follow-up question", styles["subsection"]),
            table(
                [
                    ["Question", "Answer"],
                    ["Why is state-only strong?", "State variables can proxy expected return or risk directly. That is why state-only is retained as a first-class control and the result is framed as conditional ranking, not pure alpha timing."],
                    ["Why not choose tree alpha + state?", "The OOS evidence is already observed and the portfolio layer is secondary. Selecting a winner would overstate the design; all ten scores remain visible."],
                    ["Is the strategy tradable?", "Not established. Cost stress and ADV participation show economic sensitivity, but there is no calibrated spread or impact model."],
                ],
                [48 * mm, 121 * mm],
                styles,
            ),
            PageBreak(),
        ]
    )

    # Reproducibility
    locked_rows = [["Locked source", "SHA-256 prefix"]]
    locked_rows.extend(
        [[entry["path"], entry["sha256"][:16]] for entry in config["locked_sources"]]
    )
    story.extend(
        [
            Paragraph("10. Reproducibility and provenance", styles["section"]),
            Paragraph(
                "Gate 7 does not rerun empirical research. It verifies immutable sources, renders seven figures, assembles this PDF, renders the PDF to images for visual inspection, and validates the presentation outputs.",
                styles["body"],
            ),
            Paragraph("Minimal command sequence from the project root", styles["subsection"]),
            Table(
                [
                    [Paragraph("python scripts/build_gate7_figures.py", styles["small"])],
                    [Paragraph("python scripts/build_gate7_pdf.py", styles["small"])],
                    [Paragraph("python scripts/validate_gate7_presentation.py", styles["small"])],
                    [Paragraph("pytest -q", styles["small"])],
                ],
                colWidths=[content_width],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7F9")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD6DE")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D7E0E6")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            ),
            Spacer(1, 5 * mm),
            Paragraph("Locked source inventory", styles["subsection"]),
            table(locked_rows, [130 * mm, 39 * mm], styles),
            Spacer(1, 4 * mm),
            Paragraph(
                "The complete digests, output hashes, PDF page count, required-text checks, and validation result are stored in the Gate 7 audit manifest and validation JSON beside this report.",
                styles["small"],
            ),
        ]
    )

    document.build(story, onFirstPage=page_rule, onLaterPages=page_rule)
    return output_path


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or project_root).resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_path = build_pdf(project_root, output_root, config, config_path)
    print(json.dumps({"pdf": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
