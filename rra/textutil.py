"""Pure helpers: ADF <-> text, recorded-event parsing, scope extraction, safety flags."""
from __future__ import annotations

import re
from typing import Any

# ---------- Atlassian Document Format ----------

def text_to_adf(text: str) -> dict:
    """One paragraph per line; text is preserved verbatim (no markdown interpretation)."""
    content: list[dict] = []
    for line in text.split("\n"):
        para: dict[str, Any] = {"type": "paragraph", "content": []}
        if line:
            para["content"].append({"type": "text", "text": line})
        content.append(para)
    return {"version": 1, "type": "doc", "content": content}


def adf_to_text(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    t = node.get("type")
    if t == "text":
        return node.get("text", "")
    if t == "hardBreak":
        return "\n"
    children = node.get("content", []) or []
    if t == "doc":
        return "\n".join(adf_to_text(c) for c in children)
    return "".join(adf_to_text(c) for c in children)


# ---------- Recorded events ----------

_EVENT_RE = re.compile(
    r"^Recorded event (?P<ts>\S+) \| actor (?P<actor>.+?) \| role (?P<role>.+?)\n(?P<body>.*)$", re.S
)


def parse_recorded_event(text: str) -> dict:
    """Split 'Recorded event <ts> | actor <name> | role <role>\\n<body>' into parts.

    The fictional actor/time live in the text because native Jira authors/timestamps
    belong to the importer.
    """
    m = _EVENT_RE.match(text.strip())
    if not m:
        return {"recorded_at": None, "actor": None, "role": None, "body": text.strip()}
    return {"recorded_at": m["ts"], "actor": m["actor"], "role": m["role"], "body": m["body"].strip()}


def template_of(body: str) -> str:
    """Normalise a body so near-identical boilerplate ('note 17: ...') groups together."""
    return re.sub(r"\d+", "#", body.lower()).strip()


# ---------- Scope extraction / comparison ----------

_SCOPE_PATTERNS = {
    "release": re.compile(r"\brelease\s+(R\d+(?:\.\d+)*)", re.I),
    "commit": re.compile(r"\bcommit\s+([0-9a-f]{7,40})\b", re.I),
    "service": re.compile(r"\bservice\s+([a-z0-9][a-z0-9-]*)", re.I),
    "environment": re.compile(r"\benvironment\s+([a-z][a-z-]*)", re.I),
    "prompt": re.compile(r"\bprompt(?:\s+bundle)?\s+(p\d+)\b", re.I),
}
_APPROVAL_SCOPE = re.compile(r"\bApproval scope:\s*([a-z-]+)", re.I)
CORE_SCOPE_FIELDS = ("release", "commit", "service", "environment")


def extract_scope(body: str) -> dict:
    out: dict[str, list[str]] = {}
    for name, rx in _SCOPE_PATTERNS.items():
        vals = []
        for v in rx.findall(body):
            v = v.rstrip(".;,").lower() if name != "release" else v.rstrip(".;,")
            if v not in vals:
                vals.append(v)
        out[name] = vals
    out["approval_scope"] = [v.lower() for v in _APPROVAL_SCOPE.findall(body)]
    return out


def compare_scope(found: dict, target: dict) -> dict:
    """Per-field verdict: match / mismatch / not_stated. 'match' needs every stated value to equal target."""
    result: dict[str, Any] = {}
    for fld in ("release", "commit", "service", "environment", "prompt"):
        vals = found.get(fld, [])
        tgt = str(target.get(fld, "")).lower() if target.get(fld) is not None else ""
        if not vals:
            verdict = "not_stated"
        elif all(v.lower() == tgt for v in vals):
            verdict = "match"
        else:
            verdict = "mismatch"
        result[fld] = {"stated": vals, "target": target.get(fld), "verdict": verdict}
    result["approval_scope"] = found.get("approval_scope", [])
    core = [result[f]["verdict"] for f in CORE_SCOPE_FIELDS]
    result["core_scope_applies"] = (
        "yes" if all(v == "match" for v in core)
        else "no" if any(v == "mismatch" for v in core)
        else "unknown_missing_fields"
    )
    return result


# ---------- Safety flag ----------

_INSTRUCTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|override)\b.{0,40}\b(rules?|instructions?|policy|previous)\b", re.I),
    re.compile(r"\bnote for (automated|ai|llm|agent|release)", re.I),
    re.compile(r"\b(automated|ai|llm)\s+(release\s+)?(reviewers?|agents?|assistants?)\b", re.I),
    re.compile(r"\b(report|mark|record|set|decide)\b.{0,50}\bas\s+(pass|fail|insufficient)", re.I),
    re.compile(r"\bdo not (mention|report|flag|disclose)\b", re.I),
]


def looks_like_instruction_to_reviewer(text: str) -> bool:
    """Heuristic WARNING flag only. Ticket text is data; it can never change the policy."""
    return any(p.search(text) for p in _INSTRUCTION_PATTERNS)


def adf_or_str(v: Any) -> str:
    return adf_to_text(v) if isinstance(v, dict) else (v or "")
