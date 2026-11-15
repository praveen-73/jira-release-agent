from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import JiraConfig, load_dotenv


def _jira():
    from .jira_client import JiraClient
    return JiraClient(JiraConfig.from_env())


def cmd_import(a) -> int:
    from .importer import run_import
    cfg = JiraConfig.from_env()
    from .jira_client import JiraClient
    st = run_import(JiraClient(cfg), a.project, a.dataset, cfg.issue_type, cfg.link_type, a.workers)
    print(f"created {st.issues_created} issues ({st.issues_existing} already existed), +{st.comments_added} comments, "
          f"+{st.links_added} links, {st.transitions_done} status transitions in {st.seconds:.0f}s")
    for w in st.warnings[:20]:
        print("WARNING:", w)
    return 0 if not st.warnings else 2


def cmd_verify(a) -> int:
    from .importer import verify_import
    probs = verify_import(_jira(), a.project, a.dataset)
    print("import verified OK" if not probs else "\n".join(probs[:50]))
    return 0 if not probs else 1


def cmd_assess(a) -> int:
    from .agent import Budget, run_agent
    from .llm import DEFAULT_MODEL, GeminiChat
    from .report import build_report
    from .tools import EvidenceTools
    from .trace import Trace
    import os
    jira = _jira()
    model = a.model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    print(f"Gemini model: {model}")
    out = Path(a.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    tools = EvidenceTools(jira, a.project, a.label, a.release)
    trace = Trace(out)
    budget = Budget(a.max_turns, a.max_tool_calls, a.max_tokens, a.max_seconds)
    run = run_agent(tools, trace, GeminiChat.from_env(), model, budget)
    data, md = build_report(tools, run, a.project)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out / "trace.md").write_text(trace.render_md(), encoding="utf-8")
    print(md.split("## Requirement decisions")[0])
    print(f"Artifacts written to {out}/")
    return 0 if run.status == "COMPLETED" else 3


def cmd_compare(a) -> int:
    got = {d["requirement_id"]: d["decision"] for d in json.loads(Path(a.report).read_text(encoding="utf-8"))["decisions"]}
    exp = json.loads(Path(a.expected).read_text(encoding="utf-8"))["decisions"]
    bad = 0
    for rid, want in sorted(exp.items()):
        ok = got.get(rid) == want
        bad += not ok
        print(f"{'OK ' if ok else 'DIFF'} {rid}: got {got.get(rid)} expected {want}")
    return 1 if bad else 0


def cmd_models(a) -> int:
    from .llm import GeminiChat
    rows = [m for m in GeminiChat.from_env().list_models() if "generateContent" in m.get("supportedGenerationMethods", [])]
    if not a.all:
        rows = [m for m in rows if "flash" in m["name"].lower()]
    for m in sorted(rows, key=lambda m: m["name"]):
        print(f"{m['name'].removeprefix('models/'):40s} {m.get('displayName', '')}")
    print(f"\n{len(rows)} model(s). Put one in .env as GEMINI_MODEL=<id>  (use --all to list every model)")
    return 0


def cmd_mock(a) -> int:
    from .mock_jira import serve
    serve(a.port)
    return 0


def cmd_web(a) -> int:
    from .web import serve
    serve(a.host, a.port)
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy codepage
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    load_dotenv()
    p = argparse.ArgumentParser(prog="rra", description="Jira Release Evidence Agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, h in (("import", cmd_import, "import the dataset into a Jira project"),
                        ("verify", cmd_verify, "verify Jira matches the dataset")):
        s = sub.add_parser(name, help=h)
        s.add_argument("--project", required=True, help="Jira project key")
        s.add_argument("--dataset", default="inputs/dataset.json")
        if name == "import":
            s.add_argument("--workers", type=int, default=4)
        s.set_defaults(fn=fn)
    s = sub.add_parser("assess", help="run the agent and write the report")
    s.add_argument("--project", required=True)
    s.add_argument("--label", default="tyn-release-assessment-v2", help="dataset scope label")
    s.add_argument("--release", required=True, help="target release, e.g. R2.4")
    s.add_argument("--model")
    s.add_argument("--out-dir", default="runs")
    s.add_argument("--max-turns", type=int, default=100)
    s.add_argument("--max-tool-calls", type=int, default=250)
    s.add_argument("--max-tokens", type=int, default=1_500_000)
    s.add_argument("--max-seconds", type=float, default=2400)
    s.set_defaults(fn=cmd_assess)
    s = sub.add_parser("models", help="list Gemini model IDs your API key can use")
    s.add_argument("--all", action="store_true", help="include non-Flash models")
    s.set_defaults(fn=cmd_models)
    s = sub.add_parser("compare", help="diff a report.json against expected decisions")
    s.add_argument("report")
    s.add_argument("--expected", default="tests/expected_decisions.json")
    s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("mock-jira", help="run a local Jira-compatible mock (offline testing)")
    s.add_argument("--port", type=int, default=8099)
    s.set_defaults(fn=cmd_mock)
    s = sub.add_parser("web", help="run the local browser interface for release assessments")
    s.add_argument("--host", default="127.0.0.1", help="localhost only")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_web)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
