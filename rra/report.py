from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .agent import RunResult
from .tools import EvidenceTools


def overall_decision(decisions: list[str]) -> str:
    """Deterministic policy (not model-chosen): any FAIL -> BLOCKED; else any INSUFFICIENT -> NEEDS_REVIEW; else READY."""
    if "FAIL" in decisions:
        return "BLOCKED"
    if "INSUFFICIENT_EVIDENCE" in decisions or not decisions:
        return "NEEDS_REVIEW"
    return "READY"


def _criterion(tools: EvidenceTools, rid: str) -> str:
    from .textutil import adf_to_text
    raw = tools._issues.get(rid)
    if not raw:
        return ""
    m = re.search(r"Acceptance criterion:\s*(.*)", adf_to_text(raw["fields"].get("description")))
    return m.group(1).strip() if m else ""


def build_report(tools: EvidenceTools, run: RunResult, project: str) -> tuple[dict, str]:
    reqs = sorted(tools.requirements) if tools.requirements else sorted(tools.decisions)
    rows = []
    for rid in reqs:
        d = tools.decisions.get(rid)
        if d is None:  # never silently drop: an undecided requirement is reported as unresolved
            d = {"requirement_id": rid, "decision": "INSUFFICIENT_EVIDENCE",
                 "explanation": f"NOT ASSESSED: the agent run ended ({run.status}) before this requirement was decided. "
                                f"Reason(s): {'; '.join(run.failures) or 'unknown'}. This is a tooling gap, not a verdict on the evidence.",
                 "evidence_refs": [], "outstanding": ["Re-run the assessment to complete this requirement"],
                 "excluded_evidence": [], "not_assessed": True}
        d = {**d, "title": tools.requirements.get(rid, {}).get("summary", ""),
             "acceptance_criterion": _criterion(tools, rid),
             "jira_key": tools.requirements.get(rid, {}).get("jira_key")}
        rows.append(d)
    overall = overall_decision([r["decision"] for r in rows])
    if run.status != "COMPLETED" and overall == "READY":
        overall = "NEEDS_REVIEW"

    def ref_line(ref: str) -> str:
        m = tools.seen_refs.get(ref, {})
        if "comment_id" in m:
            who = f"{m['actor']}, {m['role']}, {m['recorded_at']}"
            return f"[{ref}]({m['url']}) ({m['jira_key']}; {who}): “{m['excerpt'][:160].strip()}”"
        return f"[{ref}]({m.get('url', '')}) ({m.get('jira_key', '?')})"

    L = [f"# Release readiness report: {tools.manifest['target']['service'] if tools.manifest else '?'} {tools.release}", ""]
    L += [f"- **Overall decision: {overall}**",
          f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
          f"- Jira project: `{project}` | dataset scope label: `{tools.label}`",
          f"- Target build: `{json.dumps(tools.manifest['target']) if tools.manifest else 'unknown'}`",
          f"- Agent run status: **{run.status}**" + (f" ({'; '.join(run.failures)})" if run.failures else ""), ""]
    if tools.final_summary:
        L += ["## Summary", "", tools.final_summary, ""]
    counts = {k: sum(1 for r in rows if r["decision"] == k) for k in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE")}
    L += [f"Overall policy: any FAIL → BLOCKED; else any INSUFFICIENT_EVIDENCE → NEEDS_REVIEW; all PASS → READY. "
          f"Result: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['INSUFFICIENT_EVIDENCE']} INSUFFICIENT_EVIDENCE.", ""]
    L += ["| Requirement | Title | Decision |", "|---|---|---|"]
    L += [f"| {r['requirement_id']} | {r['title']} | **{r['decision']}** |" for r in rows]
    L += ["", "## Requirement decisions", ""]
    for r in rows:
        L += [f"### {r['requirement_id']}: {r['title']} — {r['decision']}", ""]
        if r["acceptance_criterion"]:
            L += [f"**Acceptance criterion:** {r['acceptance_criterion']}", ""]
        L += [f"**Explanation:** {r['explanation']}", ""]
        if r["evidence_refs"]:
            L += ["**Supporting evidence:**"] + [f"- {ref_line(x)}" for x in r["evidence_refs"]] + [""]
        if r.get("excluded_evidence"):
            L += ["**Evidence considered and excluded:**"] + [f"- {ref_line(e['ref'])} — {e['reason']}" for e in r["excluded_evidence"]] + [""]
        if r["outstanding"]:
            L += ["**Outstanding verification / clarification:**"] + [f"- {o}" for o in r["outstanding"]] + [""]
    notes = list(tools.integrity_notes)
    if tools.flagged:
        L += ["## Integrity notes", "",
              "Ticket text addressed to reviewers/automation was detected and **ignored**; it did not influence any decision:", ""]
        L += [f"- {ref_line(ref)}" for ref in tools.flagged]
        L += [f"- Agent note: {n}" for n in notes] + [""]
    elif notes:
        L += ["## Integrity notes", ""] + [f"- {n}" for n in notes] + [""]
    u = run.usage
    L += ["## Run statistics", "",
          f"- Model: `{run.model}` | turns: {run.turns} | tool calls: {run.tool_calls} (errors: {run.tool_errors})",
          f"- Wall-clock: {run.seconds:.1f}s",
          f"- Tokens: prompt {u['input_tokens']:,} (cached {u['cached_input_tokens']:,}), completion {u['output_tokens']:,}",
          f"- Jira API requests: {tools.jira.request_count}", "",
          "Full tool-call trace: `trace.md` / `trace.jsonl`.", ""]
    data = {"overall_decision": overall, "run_status": run.status, "failures": run.failures,
            "target": tools.manifest["target"] if tools.manifest else None, "label": tools.label, "project": project,
            "decisions": rows, "integrity_flagged_refs": list(tools.flagged),
            "stats": {"model": run.model, "turns": run.turns, "tool_calls": run.tool_calls, "tool_errors": run.tool_errors,
                      "seconds": round(run.seconds, 1), "usage": run.usage, "jira_requests": tools.jira.request_count}}
    return data, "\n".join(L)
