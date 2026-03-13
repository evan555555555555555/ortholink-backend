"""
OrthoLink Global Compliance Orchestrator (GCO) — Multi-Agent Parallel Analysis
POST /api/v1/gco-analysis

Runs TDA + PMS + ROA + CAPA in parallel (ThreadPoolExecutor max_workers=4).
Phase 2: GPT-4o Chief Regulatory Officer synthesis cross-validates all agent outputs,
detects amplifying interactions, and flags notification risk.

Readiness scoring: 8-tier content-aware (notification_risk × high_risk_count × gap_count).
"""

import concurrent.futures
import logging
import uuid
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Global Compliance Orchestrator synthesizes outputs from multiple AI agents. "
    "All results require review by qualified regulatory affairs professionals."
)


class AgentResult(BaseModel):
    """Result from a single agent in the GCO pipeline."""

    agent_name: str
    agent_id: str = Field(default="")
    status: str = Field(..., description="success | failed | skipped")
    summary: str = Field(default="")
    key_findings: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    high_risk_flags: list[str] = Field(default_factory=list)
    citation_count: int = 0
    error: str = Field(default="")


class GcoReport(BaseModel):
    """Aggregated multi-agent regulatory analysis."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    country: str
    device_class: str
    device_type: Optional[str] = None
    agents_run: int = 0
    agents_succeeded: int = 0
    critical_gaps: list[str] = Field(
        default_factory=list, description="High-priority items identified across all agents"
    )
    immediate_actions: list[str] = Field(
        default_factory=list, description="Actions that should be taken immediately"
    )
    overall_readiness: str = Field(
        default="",
        description="NOT_READY | PARTIAL | READY_WITH_CONDITIONS | READY",
    )
    readiness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    agent_results: list[AgentResult] = Field(default_factory=list)
    synthesis: str = Field(default="", description="Cross-agent synthesis narrative")
    disclaimer: str = Field(default=DISCLAIMER)


def run_gco_analysis(
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
    problem_statement: Optional[str] = None,
    max_workers: int = 4,
) -> GcoReport:
    """
    Run TDA, PMS, ROA, and (optionally) CAPA in parallel.
    Phase 2 synthesis: LLM cross-validates all agent outputs.
    Returns synthesized GcoReport.
    """
    from app.crews.technical_dossier import run_technical_dossier
    from app.crews.pms_plan import run_pms_plan
    from app.crews.generate_checklist import run_roa_checklist
    from app.crews.capa_analysis import run_capa_analysis

    agent_tasks = [
        ("Technical Documentation Agent", "tda", run_technical_dossier, (country, device_class, device_type)),
        ("Post-Market Surveillance Agent", "pms", run_pms_plan, (country, device_class, device_type)),
        ("Regulatory Obligation Agent", "roa", run_roa_checklist, (country, device_class, device_type)),
    ]
    if problem_statement:
        agent_tasks.append((
            "CAPA Analysis Agent", "capa", run_capa_analysis,
            (problem_statement, country, device_class, device_type),
        ))

    agent_results: list[AgentResult] = []

    def _run_agent(name: str, agent_id: str, fn, args: tuple) -> AgentResult:
        try:
            result = fn(*args)
            return _summarize_agent_result(name, agent_id, result)
        except Exception as e:
            logger.warning("GCO agent %s failed: %s", name, e)
            return AgentResult(
                agent_name=name,
                agent_id=agent_id,
                status="failed",
                error=str(e)[:200],
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_agent, name, aid, fn, args): name
            for name, aid, fn, args in agent_tasks
        }
        for future in concurrent.futures.as_completed(futures, timeout=120):
            agent_name = futures[future]
            try:
                result = future.result(timeout=5)
                agent_results.append(result)
            except Exception as e:
                logger.warning("GCO future %s error: %s", agent_name, e)
                agent_results.append(
                    AgentResult(agent_name=agent_name, agent_id="", status="failed", error=str(e)[:200])
                )

    # ── Aggregate cross-agent findings ────────────────────────────────────────
    all_flags: list[str] = []
    all_actions: list[str] = []
    succeeded = sum(1 for r in agent_results if r.status == "success")
    total_high_risk = 0

    for r in agent_results:
        all_flags.extend(r.high_risk_flags)
        all_actions.extend(r.action_items)
        total_high_risk += len(r.high_risk_flags)

    # Deduplicate preserving order
    critical_gaps = list(dict.fromkeys(all_flags))[:10]
    immediate_actions = list(dict.fromkeys(all_actions))[:10]

    # ── 8-tier content-aware readiness scoring ────────────────────────────────
    # Detect regulatory notification risk (serious incident / FSCA triggers)
    _notification_kws = (
        "notification", "serious incident", "mandatory report",
        "fsca", "recall", "critical severity", "regulatory notification",
        "15 days", "2-day", "immediate hazard",
    )
    notification_risk = any(
        any(kw in flag.lower() for kw in _notification_kws)
        for flag in all_flags
    )

    if succeeded == 0:
        readiness_score = 0.0
        overall_readiness = "NOT_READY"
    elif succeeded < len(agent_tasks) // 2:
        readiness_score = 0.15
        overall_readiness = "PARTIAL"
    elif notification_risk and total_high_risk >= 4:
        readiness_score = 0.25
        overall_readiness = "PARTIAL"
    elif notification_risk:
        readiness_score = 0.35
        overall_readiness = "PARTIAL"
    elif total_high_risk >= 6 or len(critical_gaps) > 5:
        readiness_score = 0.45
        overall_readiness = "PARTIAL"
    elif total_high_risk >= 3 or len(critical_gaps) > 2:
        readiness_score = 0.60
        overall_readiness = "READY_WITH_CONDITIONS"
    elif total_high_risk >= 1 or len(critical_gaps) > 0:
        readiness_score = 0.75
        overall_readiness = "READY_WITH_CONDITIONS"
    else:
        readiness_score = 0.95
        overall_readiness = "READY"

    synthesis = _generate_synthesis(
        country, device_class, succeeded, len(agent_tasks), critical_gaps,
        agent_results, notification_risk, total_high_risk
    )

    return GcoReport(
        country=country,
        device_class=device_class,
        device_type=device_type,
        agents_run=len(agent_tasks),
        agents_succeeded=succeeded,
        critical_gaps=critical_gaps,
        immediate_actions=immediate_actions,
        overall_readiness=overall_readiness,
        readiness_score=round(readiness_score, 2),
        agent_results=agent_results,
        synthesis=synthesis,
    )


def _summarize_agent_result(name: str, agent_id: str, result) -> AgentResult:
    """Convert any agent result Pydantic model into a standardized AgentResult."""
    key_findings: list[str] = []
    action_items: list[str] = []
    high_risk_flags: list[str] = []
    citation_count = 0
    summary = ""

    # TDA result
    if hasattr(result, "sections"):
        sections = result.sections or []
        required_sections = [s for s in sections if s.required]
        key_findings = [
            f"{s.section_title} [{s.regulation_cite}]"
            for s in sections[:6]
        ]
        action_items = [
            f"Prepare {s.section_title} — est. {s.estimated_effort_days}d ({s.regulation_cite})"
            for s in required_sections
            if s.estimated_effort_days > 0
        ][:6]
        # Flag clinical, biocompatibility, sterility, software as high-risk missing
        risk_keywords = ("clinical", "biocompatibility", "sterility", "software", "risk management", "pmcf")
        high_risk_flags = [
            f"Required section missing or incomplete: {s.section_title}"
            for s in required_sections
            if any(kw in s.section_title.lower() for kw in risk_keywords)
        ]
        # Flag high-effort items
        high_effort = [s for s in required_sections if s.estimated_effort_days >= 90]
        for s in high_effort[:2]:
            high_risk_flags.append(
                f"Long-lead item ({s.estimated_effort_days}d): {s.section_title}"
            )
        citation_count = len(sections)
        total_days = sum(s.estimated_effort_days for s in required_sections)
        summary = (
            result.executive_summary
            or f"{len(sections)} dossier sections identified ({len(required_sections)} required). "
               f"Estimated total effort: {total_days} days."
        )

    # PMS result
    elif hasattr(result, "activities"):
        activities = result.activities or []
        reporting = getattr(result, "reporting_requirements", []) or []
        continuous = [a for a in activities if a.frequency.lower() == "continuous"]
        key_findings = [
            f"{a.activity} ({a.frequency}) [{a.regulation_cite}]"
            for a in activities[:6]
        ]
        action_items = [
            f"Implement {a.activity} — {a.responsible_party}"
            for a in activities[:6]
        ]
        # Flag short-deadline reporting requirements
        short_deadline_kws = ("2 day", "15 day", "immediate", "serious incident", "adverse")
        high_risk_flags = [
            f"Mandatory report: {r.report_type} within {r.timeline} → {r.recipient}"
            for r in reporting
            if any(kw in (r.trigger + r.timeline).lower() for kw in short_deadline_kws)
        ]
        if not continuous:
            high_risk_flags.append("No continuous PMS activity defined — potential regulatory gap")
        citation_count = len(activities) + len(reporting)
        summary = (
            result.executive_summary
            or f"PMS: {len(activities)} surveillance activities, {len(reporting)} reporting obligations. "
               f"Level: {getattr(result, 'pms_level', 'N/A')}."
        )

    # ROA checklist result
    elif hasattr(result, "items"):
        items = result.items or []
        apostille_items = [i for i in items if i.apostille_required]
        manufacturer_items = [i for i in items if i.role == "MANUFACTURER"]
        importer_items = [i for i in items if i.role == "IMPORTER"]
        key_findings = [
            f"{i.item} [{i.role}] — {i.regulation_cite}"
            for i in items[:6]
        ]
        action_items = [
            f"[{i.role}] {i.item} — {i.deadline_days}d deadline"
            for i in items
            if i.role in ("MANUFACTURER", "BOTH") and i.deadline_days > 0
        ][:6]
        high_risk_flags = [
            f"Apostille/notarization required: {i.item}"
            for i in apostille_items[:4]
        ]
        if len(manufacturer_items) == 0:
            high_risk_flags.append("No manufacturer-specific obligations found — checklist may be incomplete")
        citation_count = len(items)
        summary = (
            f"{len(items)} regulatory obligations: {len(manufacturer_items)} manufacturer, "
            f"{len(importer_items)} importer. {len(apostille_items)} apostille requirements."
        )

    # CAPA result
    elif hasattr(result, "corrective_actions"):
        root_causes = result.root_cause_categories or []
        actions = result.corrective_actions or []
        obligations = getattr(result, "regulatory_obligations", []) or []
        containment = [a for a in actions if a.action_type == "Containment"]
        key_findings = [
            f"Root cause: {rc.category} — {rc.likelihood} likelihood"
            for rc in root_causes
        ]
        key_findings += [
            f"{a.action_id} [{a.action_type}]: {a.description[:70]}"
            for a in actions[:3]
        ]
        action_items = [
            f"{a.action_id} ({a.action_type}): {a.description[:80]} [{a.target_completion_days}d]"
            for a in actions[:6]
        ]
        if result.requires_regulatory_notification:
            high_risk_flags.append(
                f"[!] Regulatory notification required -- {result.notification_rationale[:120]}"
            )
        if result.severity == "Critical":
            high_risk_flags.append(
                f"[!] Critical severity -- {result.recommended_timeline_days}-day resolution mandatory"
            )
        if not containment:
            high_risk_flags.append("No containment action defined — immediate patient protection may be unaddressed")
        # Flag high-likelihood root causes
        for rc in root_causes:
            if rc.likelihood == "High":
                high_risk_flags.append(f"High-likelihood root cause: {rc.category}")
        citation_count = len(actions) + len(obligations)
        summary = (
            f"CAPA: {result.severity} severity. {len(actions)} actions ({len(containment)} containment). "
            f"Regulatory notification: {'YES' if result.requires_regulatory_notification else 'No'}. "
            f"Target: {result.recommended_timeline_days} days."
        )

    return AgentResult(
        agent_name=name,
        agent_id=agent_id,
        status="success",
        summary=summary,
        key_findings=key_findings[:8],
        action_items=action_items[:8],
        high_risk_flags=high_risk_flags[:6],
        citation_count=citation_count,
    )


def _generate_synthesis(
    country: str,
    device_class: str,
    succeeded: int,
    total: int,
    critical_gaps: list[str],
    agent_results: list[AgentResult],
    notification_risk: bool,
    total_high_risk: int,
) -> str:
    """
    Phase 2: GPT-4o Chief Regulatory Officer synthesis.
    Cross-validates all agent outputs, detects amplifying interactions,
    flags notification risk. Falls back to structured template if LLM fails.
    """
    from app.tools.llm import chat_completion

    # Build structured per-agent context blocks
    agent_blocks = []
    for r in agent_results:
        if r.status != "success":
            agent_blocks.append(f"[{r.agent_name}] STATUS: FAILED — {r.error[:100]}")
            continue
        block_lines = [
            f"[{r.agent_name}]",
            f"  Summary: {r.summary}",
        ]
        if r.key_findings:
            block_lines.append("  Key findings:")
            for f_ in r.key_findings[:4]:
                block_lines.append(f"    • {f_}")
        if r.high_risk_flags:
            block_lines.append("  High-risk flags:")
            for f_ in r.high_risk_flags[:3]:
                block_lines.append(f"    [!] {f_}")
        agent_blocks.append("\n".join(block_lines))

    agent_context = "\n\n".join(agent_blocks)

    notification_line = (
        "NOTIFICATION RISK DETECTED: One or more agents flagged a potential mandatory regulatory "
        "reporting obligation (serious incident / FSCA / MDR). Investigate immediately."
        if notification_risk
        else "No mandatory notification triggers detected across agents."
    )

    system_prompt = (
        "You are the Chief Regulatory Officer at a global Class III medical device manufacturer. "
        "You receive summary reports from four regulatory AI agents (TDA, PMS, ROA, CAPA) and "
        "produce an executive synthesis that: (1) identifies where agent findings AMPLIFY each other "
        "(e.g. a CAPA with Critical severity AND a PMS report flagging a serious incident reporting "
        "deadline — these compound the risk), (2) flags the top 3 actions that must happen THIS WEEK, "
        "(3) gives a one-sentence overall regulatory posture assessment, "
        "(4) cites the single most important regulation number applicable. "
        "ZERO-INFERENCE: Only reference findings that appear in the agent outputs below. "
        "Never invent findings or cite regulations not mentioned by the agents. "
        "Write in direct, boardroom-level prose. Max 5 sentences. No bullet lists. No markdown."
    )

    user_prompt = (
        f"Country: {country} | Device Class: {device_class}\n"
        f"Agents completed: {succeeded}/{total}\n"
        f"Total high-risk flags: {total_high_risk}\n"
        f"{notification_line}\n\n"
        f"AGENT OUTPUTS:\n{agent_context}\n\n"
        f"Critical gaps (deduplicated): {'; '.join(critical_gaps[:5]) if critical_gaps else 'None identified'}\n\n"
        "Write the executive synthesis now."
    )

    try:
        synthesis = chat_completion(system_prompt, user_prompt)
        if synthesis and len(synthesis.strip()) > 40:
            return synthesis.strip()
    except Exception as e:
        logger.warning("GCO synthesis LLM call failed: %s", e)

    # ── Structured fallback (no LLM) ─────────────────────────────────────────
    lines = [
        f"GCO analysis for {country} Class {device_class}: {succeeded}/{total} agents completed."
    ]
    if notification_risk:
        lines.append(
            "ALERT: Regulatory notification triggers detected — mandatory reporting timelines may apply. "
            "Engage regulatory affairs counsel immediately."
        )
    if critical_gaps:
        top_gaps = critical_gaps[:3]
        gap_text = "; ".join(top_gaps)
        lines.append(
            f"{len(critical_gaps)} critical gaps identified requiring immediate attention: "
            + gap_text
            + ("." if not gap_text.endswith(".") else "")
        )
    else:
        lines.append("No critical regulatory gaps detected across all agents.")
    lines.append(
        f"Total high-risk flags across agents: {total_high_risk}. "
        "Review each agent's detailed output and initiate gap remediation before regulatory submission."
    )
    return " ".join(lines)
