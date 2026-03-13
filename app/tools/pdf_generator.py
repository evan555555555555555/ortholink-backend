"""
OrthoLink PDF export — WeasyPrint + HTML for ROA checklists (PRD G12).
"""

import logging
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)

CHECKLIST_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>OrthoLink Compliance Checklist — {country} {device_class}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; font-size: 11pt; line-height: 1.4; margin: 1in; color: #1a1a1a; }}
    h1 {{ font-size: 16pt; margin-bottom: 0.5em; }}
    .meta {{ color: #555; margin-bottom: 1em; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.5em; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    .role-MANUFACTURER {{ background: #e8f4fc; }}
    .role-IMPORTER {{ background: #fef3e8; }}
    .role-BOTH {{ background: #f0f0f0; }}
    .disclaimer {{ margin-top: 1.5em; font-size: 9pt; color: #666; }}
  </style>
</head>
<body>
  <h1>Compliance Checklist</h1>
  <p class="meta">Country: {country} | Device class: {device_class}</p>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Item</th>
        <th>Role</th>
        <th>Regulation</th>
        <th>Days</th>
        <th>Apostille</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <p class="disclaimer">{disclaimer}</p>
</body>
</html>
"""


def render_checklist_pdf(
    country: str,
    device_class: str,
    items: list[dict[str, Any]],
    disclaimer: str = "Reference tool only. Verify with official sources.",
) -> bytes:
    """Render ROA checklist to PDF via WeasyPrint. Returns PDF bytes."""
    rows = []
    for i, it in enumerate(items, 1):
        role = it.get("role", "BOTH")
        rows.append(
            f"<tr class=\"role-{role}\">"
            f"<td>{i}</td>"
            f"<td>{_escape(it.get('item', ''))}</td>"
            f"<td>{_escape(role)}</td>"
            f"<td>{_escape(it.get('regulation_cite', ''))}</td>"
            f"<td>{it.get('deadline_days', 0)}</td>"
            f"<td>{'Yes' if it.get('apostille_required') else 'No'}</td>"
            f"<td>{_escape(it.get('notes', ''))}</td>"
            "</tr>"
        )
    html = CHECKLIST_HTML_TEMPLATE.format(
        country=_escape(country),
        device_class=_escape(device_class),
        rows="\n".join(rows),
        disclaimer=_escape(disclaimer),
    )
    try:
        from weasyprint import HTML

        buf = BytesIO()
        HTML(string=html).write_pdf(buf)
        return buf.getvalue()
    except Exception as e:
        logger.exception(f"WeasyPrint PDF generation failed: {e}")
        raise


STRATEGY_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>OrthoLink Strategy Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; font-size: 11pt; line-height: 1.4; margin: 1in; color: #1a1a1a; }}
    h1 {{ font-size: 16pt; margin-bottom: 0.5em; }}
    h2 {{ font-size: 13pt; margin-top: 1.5em; margin-bottom: 0.3em; }}
    .meta {{ color: #555; margin-bottom: 1em; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.5em; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    .disclaimer {{ margin-top: 1.5em; font-size: 9pt; color: #666; }}
  </style>
</head>
<body>
  <h1>Regulatory Strategy Report</h1>
  <p class="meta">{meta}</p>
  {content}
  <p class="disclaimer">{disclaimer}</p>
</body>
</html>
"""


def render_strategy_pdf(data: dict[str, Any]) -> bytes:
    """Render strategy report to PDF via WeasyPrint. Returns PDF bytes."""
    device_name = data.get("device_name", data.get("summary", "Strategy Report"))
    markets = data.get("target_markets", [])
    meta = f"Device: {_escape(str(device_name))}"
    if markets:
        meta += f" | Markets: {_escape(', '.join(str(m) for m in markets))}"

    sections: list[str] = []

    entry_seq = data.get("entry_sequence", [])
    if entry_seq:
        rows = []
        for i, e in enumerate(entry_seq, 1):
            rows.append(
                f"<tr>"
                f"<td>{i}</td>"
                f"<td>{_escape(str(e.get('country', '')))}</td>"
                f"<td>{_escape(str(e.get('pathway', '')))}</td>"
                f"<td>{e.get('reuse_pct', 0):.0f}%</td>"
                f"<td>{e.get('timeline_months', 0)} mo</td>"
                f"<td>${e.get('cost_usd', 0):,.0f}</td>"
                f"</tr>"
            )
        sections.append(
            "<h2>Entry Sequence</h2><table><thead><tr>"
            "<th>#</th><th>Country</th><th>Pathway</th><th>Reuse</th><th>Timeline</th><th>Cost</th>"
            "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"
        )

    cost_est = data.get("cost_estimates", {})
    if cost_est:
        rows = []
        for c, costs in cost_est.items():
            total = costs.get("total_estimate_usd", 0) if isinstance(costs, dict) else 0
            rows.append(f"<tr><td>{_escape(str(c))}</td><td>${total:,.0f}</td></tr>")
        sections.append(
            "<h2>Cost Estimates</h2><table><thead><tr><th>Country</th><th>Total</th>"
            "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"
        )

    disclaimer = data.get("disclaimer", "Reference tool only. Verify with official sources.")
    html = STRATEGY_HTML_TEMPLATE.format(
        meta=meta,
        content="\n".join(sections) if sections else "<p>No detailed data available.</p>",
        disclaimer=_escape(str(disclaimer)),
    )
    try:
        from weasyprint import HTML

        buf = BytesIO()
        HTML(string=html).write_pdf(buf)
        return buf.getvalue()
    except Exception as e:
        logger.exception(f"WeasyPrint strategy PDF generation failed: {e}")
        raise


def _escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


RISK_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>OrthoLink Risk Management Report — {country}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; font-size: 11pt; line-height: 1.4; margin: 1in; color: #1a1a1a; }}
    h1 {{ font-size: 16pt; margin-bottom: 0.25em; }}
    h2 {{ font-size: 13pt; margin-top: 1.5em; margin-bottom: 0.3em; }}
    .meta {{ color: #555; margin-bottom: 0.5em; font-size: 10pt; }}
    .verdict-badge {{ display: inline-block; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 13pt; margin: 0.5em 0 1em; letter-spacing: 0.05em; }}
    .verdict-ACCEPTABLE {{ background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }}
    .verdict-ALARP {{ background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }}
    .verdict-UNACCEPTABLE {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.5em; font-size: 9.5pt; }}
    th, td {{ border: 1px solid #ccc; padding: 5px 7px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; font-weight: 600; font-size: 9pt; }}
    .risk-ACCEPTABLE {{ color: #065f46; font-weight: 700; }}
    .risk-ALARP {{ color: #92400e; font-weight: 700; }}
    .risk-UNACCEPTABLE {{ color: #991b1b; font-weight: 700; }}
    .counts-row {{ display: flex; gap: 2em; margin: 0.75em 0 1.25em; font-size: 11pt; }}
    .count-item {{ text-align: center; }}
    .count-num {{ font-size: 22pt; font-weight: 900; line-height: 1.1; }}
    .count-label {{ font-size: 8.5pt; color: #666; text-transform: uppercase; letter-spacing: 0.06em; }}
    .count-acceptable {{ color: #065f46; }}
    .count-alarp {{ color: #92400e; }}
    .count-unacceptable {{ color: #991b1b; }}
    .count-total {{ color: #1a1a1a; }}
    .disclaimer {{ margin-top: 1.5em; font-size: 9pt; color: #666; border-top: 1px solid #e5e7eb; padding-top: 0.75em; }}
    .section-label {{ font-size: 8.5pt; font-weight: 700; color: #666; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 1.25em; margin-bottom: 0.25em; }}
  </style>
</head>
<body>
  <h1>Risk Management Report</h1>
  <p class="meta">Country: {country} | Generated by OrthoLink RMA — ISO 14971:2019</p>

  <p class="section-label">Device</p>
  <p class="meta">{device_description}</p>

  <p class="section-label">Intended Use</p>
  <p class="meta">{intended_use}</p>

  <p class="section-label">Overall Verdict</p>
  <span class="verdict-badge verdict-{overall_verdict}">{overall_verdict}</span>

  <div class="counts-row">
    <div class="count-item"><div class="count-num count-total">{total}</div><div class="count-label">Hazards</div></div>
    <div class="count-item"><div class="count-num count-unacceptable">{unacceptable}</div><div class="count-label">Unacceptable</div></div>
    <div class="count-item"><div class="count-num count-alarp">{alarp}</div><div class="count-label">ALARP</div></div>
    <div class="count-item"><div class="count-num count-acceptable">{acceptable}</div><div class="count-label">Acceptable</div></div>
  </div>

  <h2>Hazard Analysis — ISO 14971 §4 Risk Assessment Table</h2>
  <table>
    <thead>
      <tr>
        <th>Hazard</th>
        <th>Harm</th>
        <th>Sev</th>
        <th>Prob</th>
        <th>Initial Risk</th>
        <th>Mitigation / Controls</th>
        <th>Res. Sev</th>
        <th>Res. Prob</th>
        <th>Residual Risk</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <p class="disclaimer">{disclaimer}</p>
</body>
</html>
"""


def render_risk_pdf(report_data: dict) -> bytes:
    """Render ISO 14971 RMA risk report to PDF via WeasyPrint. Returns PDF bytes."""
    hazards = report_data.get("hazard_analysis", [])
    counts = report_data.get("counts", {})
    overall_verdict = str(report_data.get("overall_verdict", "ALARP"))
    # Sanitise verdict so CSS class is always one of the three valid values
    if overall_verdict not in ("ACCEPTABLE", "ALARP", "UNACCEPTABLE"):
        overall_verdict = "ALARP"

    rows = []
    for h in hazards:
        rl  = str(h.get("risk_level", "ALARP"))
        rrl = str(h.get("residual_risk_level", "ALARP"))
        if rl  not in ("ACCEPTABLE", "ALARP", "UNACCEPTABLE"): rl  = "ALARP"
        if rrl not in ("ACCEPTABLE", "ALARP", "UNACCEPTABLE"): rrl = "ALARP"

        mitigation = h.get("mitigation", "") or ""
        # control_measures may be present instead of mitigation (field name varies by version)
        if not mitigation:
            measures = h.get("control_measures", [])
            if isinstance(measures, list):
                mitigation = "; ".join(str(m) for m in measures)
            else:
                mitigation = str(measures)

        rows.append(
            f"<tr>"
            f"<td>{_escape(h.get('hazard_name', h.get('hazard', '')))}</td>"
            f"<td>{_escape(h.get('harm', ''))}</td>"
            f"<td>{_escape(str(h.get('severity', '')))}</td>"
            f"<td>{_escape(str(h.get('probability', '')))}</td>"
            f"<td class=\"risk-{rl}\">{rl}</td>"
            f"<td>{_escape(mitigation)}</td>"
            f"<td>{_escape(str(h.get('residual_severity', '')))}</td>"
            f"<td>{_escape(str(h.get('residual_probability', '')))}</td>"
            f"<td class=\"risk-{rrl}\">{rrl}</td>"
            f"</tr>"
        )

    html = RISK_HTML_TEMPLATE.format(
        country=_escape(str(report_data.get("country", ""))),
        device_description=_escape(str(report_data.get("device_description", ""))),
        intended_use=_escape(str(report_data.get("intended_use", ""))),
        overall_verdict=overall_verdict,
        total=int(counts.get("total", len(hazards))),
        acceptable=int(counts.get("acceptable", 0)),
        alarp=int(counts.get("alarp", 0)),
        unacceptable=int(counts.get("unacceptable", 0)),
        rows="\n".join(rows),
        disclaimer=_escape(str(report_data.get(
            "disclaimer",
            "Reference tool only. Not a substitute for qualified risk management review.",
        ))),
    )
    try:
        from weasyprint import HTML

        buf = BytesIO()
        HTML(string=html).write_pdf(buf)
        return buf.getvalue()
    except Exception as e:
        logger.exception(f"WeasyPrint risk PDF generation failed: {e}")
        raise
