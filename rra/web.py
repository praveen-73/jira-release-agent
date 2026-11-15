"""Local web form for running and reviewing Jira release assessments."""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import JiraConfig, load_dotenv
from .jira_client import JiraClient

DEFAULT_PROJECT = "TRA"
DEFAULT_LABEL = "tyn-release-assessment-v2"
DEFAULT_RELEASE = "R2.4"
RUNS_ROOT = Path("runs").resolve()
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ARTIFACTS = {"report.md", "report.json", "trace.md", "trace.jsonl"}

_lock = threading.Lock()
_runs: dict[str, dict[str, Any]] = {}


def validate_inputs(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    fields = {
        "project": payload.get("project"),
        "label": payload.get("label"),
        "release": payload.get("release"),
    }
    for name, value in fields.items():
        if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value.strip()):
            raise ValueError(f"{name.capitalize()} must be 1–64 letters, numbers, dots, underscores, or hyphens")
        fields[name] = value.strip()
    return fields


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in run.items() if k != "out_dir"}


def _perform_assessment(run_id: str, fields: dict[str, str]) -> None:
    out = RUNS_ROOT / f"web-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{run_id[:8]}"
    with _lock:
        run = _runs[run_id]
        run.update(state="running", started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), out_dir=str(out))
    try:
        from .agent import Budget, run_agent
        from .llm import DEFAULT_MODEL as DEFAULT_GEMINI_MODEL, GeminiChat
        from .report import build_report
        from .tools import EvidenceTools
        from .trace import Trace
        import os

        jira = JiraClient(JiraConfig.from_env())
        model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        tools = EvidenceTools(jira, fields["project"], fields["label"], fields["release"])
        trace = Trace(out)
        result = run_agent(tools, trace, GeminiChat.from_env(), model, Budget())
        data, markdown = build_report(tools, result, fields["project"])
        (out / "report.md").write_text(markdown, encoding="utf-8")
        (out / "report.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "trace.md").write_text(trace.render_md(), encoding="utf-8")
        state = "completed" if result.status == "COMPLETED" else "failed"
        summary = {"overall_decision": data["overall_decision"], "run_status": result.status,
                   "stats": data["stats"], "failure_reasons": result.failures}
        with _lock:
            run.update(state=state, summary=summary, artifacts=sorted(ARTIFACTS),
                       finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    except (Exception, SystemExit) as exc:
        with _lock:
            run.update(state="failed", error=f"{type(exc).__name__}: {str(exc)[:500]}",
                       finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


class _Handler(BaseHTTPRequestHandler):
    server_version = "RRA-local/1.0"

    def _is_local_request(self) -> bool:
        host = urlparse("//" + self.headers.get("Host", "")).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        if origin:
            origin_host = urlparse(origin).hostname
            if origin_host not in ("127.0.0.1", "localhost", "::1") or origin_host != host:
                return False
        return True

    def _deny_non_local_request(self) -> bool:
        if self._is_local_request():
            return False
        self._json(403, {"error": "Local requests only"})
        return True

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, body: Any) -> None:
        self._send(status, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self._deny_non_local_request():
            return
        path = urlparse(self.path).path
        if path == "/":
            page = (_PAGE.replace("__PROJECT__", DEFAULT_PROJECT)
                    .replace("__LABEL__", DEFAULT_LABEL)
                    .replace("__RELEASE__", DEFAULT_RELEASE))
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path.startswith("/api/runs/"):
            run_id = path.rsplit("/", 1)[-1]
            with _lock:
                run = _runs.get(run_id)
                data = _public_run(run) if run else None
            self._json(200 if data else 404, data or {"error": "Run not found"})
            return
        if path.startswith("/artifacts/"):
            parts = [unquote(p) for p in path.split("/") if p]
            if len(parts) == 3 and parts[2] in ARTIFACTS:
                with _lock:
                    run = _runs.get(parts[1])
                    out_dir = Path(run["out_dir"]).resolve() if run and run.get("out_dir") else None
                if out_dir and run.get("state") in ("completed", "failed"):
                    candidate = (out_dir / parts[2]).resolve()
                    if candidate.parent == out_dir and candidate.is_file():
                        mime = "application/json; charset=utf-8" if parts[2].endswith(".json") or parts[2].endswith(".jsonl") else "text/markdown; charset=utf-8"
                        self._send(200, candidate.read_bytes(), mime)
                        return
            self._json(404, {"error": "Artifact not found"})
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self._deny_non_local_request():
            return
        if urlparse(self.path).path != "/api/runs":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 8192:
                raise ValueError("Request body must be between 1 and 8192 bytes")
            fields = validate_inputs(json.loads(self.rfile.read(length)))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        with _lock:
            if any(r["state"] in ("queued", "running") for r in _runs.values()):
                self._json(409, {"error": "An assessment is already running"})
                return
            run_id = uuid.uuid4().hex
            _runs[run_id] = {"id": run_id, **fields, "state": "queued", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        worker = threading.Thread(target=_perform_assessment, args=(run_id, fields), daemon=True)
        worker.start()
        self._json(202, _public_run(_runs[run_id]))

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid writing request data to the console; keep only basic operational logging.
        print(f"[rra-web] {self.address_string()} {fmt % args}")


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jira Release Assessment</title><style>
body{font:16px system-ui,sans-serif;max-width:780px;margin:48px auto;padding:0 20px;color:#182230;background:#f5f7fa}main{background:white;border:1px solid #d9e0e8;border-radius:12px;padding:28px;box-shadow:0 4px 20px #18223010}h1{margin-top:0}p{color:#506070}label{display:block;font-weight:600;margin:18px 0 6px}input{box-sizing:border-box;width:100%;padding:11px;border:1px solid #aeb9c6;border-radius:6px;font:inherit}button{margin-top:22px;padding:11px 18px;border:0;border-radius:6px;background:#1459c7;color:white;font:inherit;font-weight:600;cursor:pointer}button:disabled{opacity:.6;cursor:wait}#status{margin-top:22px;padding:14px;background:#f1f5fa;border-radius:6px;white-space:pre-wrap}#links a{margin-right:14px}small{color:#526273}
</style></head><body><main><h1>Jira release assessment</h1><p>Run a read-only evidence assessment against issues in Jira. The dataset is read from Jira; this form does not upload a JSON file.</p>
<form id="form"><label for="project">Jira project key</label><input id="project" name="project" required maxlength="64" pattern="[A-Za-z0-9._-]{1,64}" value="__PROJECT__">
<label for="label">Dataset scope label</label><input id="label" name="label" required maxlength="64" pattern="[A-Za-z0-9._-]{1,64}" value="__LABEL__">
<label for="release">Target release</label><input id="release" name="release" required maxlength="64" pattern="[A-Za-z0-9._-]{1,64}" value="__RELEASE__">
<button id="submit">Start assessment</button></form><div id="status" role="status">Ready. One assessment can run at a time.</div><div id="links"></div>
<script>
const form=document.querySelector('#form'),button=document.querySelector('#submit'),status=document.querySelector('#status'),links=document.querySelector('#links');
async function poll(id){for(;;){const r=await fetch('/api/runs/'+encodeURIComponent(id));const d=await r.json();status.textContent='Assessment '+d.state+(d.summary?'\nDecision: '+d.summary.overall_decision+'\nRun status: '+d.summary.run_status:'')+(d.error?'\n'+d.error:'')+(d.summary?.failure_reasons?.length?'\n'+d.summary.failure_reasons.join('\n'):'');if(d.state==='completed'||d.state==='failed'){button.disabled=false;links.replaceChildren();for(const f of (d.artifacts||[])){const a=document.createElement('a');a.href='/artifacts/'+encodeURIComponent(id)+'/'+encodeURIComponent(f);a.textContent=f;a.target='_blank';a.rel='noopener';links.append(a,document.createTextNode(' '));}return;}await new Promise(resolve=>setTimeout(resolve,2000));}}
form.addEventListener('submit',async e=>{e.preventDefault();button.disabled=true;links.replaceChildren();status.textContent='Starting assessment…';try{const body=Object.fromEntries(new FormData(form));const r=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw new Error(d.error||'Could not start assessment');await poll(d.id);}catch(err){status.textContent='Could not start assessment: '+err.message;button.disabled=false;}});
</script></main></body></html>"""


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("The assessment web app can only bind to localhost")
    load_dotenv()
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"Jira Release Assessment is available at http://{host}:{server.server_port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
