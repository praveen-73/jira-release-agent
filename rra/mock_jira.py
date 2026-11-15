"""In-memory mock of the Jira Cloud REST v3 subset this project uses. For tests/offline demos only."""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .textutil import adf_to_text

STATUSES = ["To Do", "In Progress", "Done"]


class Store:
    def __init__(self, project: str = "TYN"):
        self.project = project
        self.issues: dict[str, dict] = {}
        self.comments: dict[str, list[dict]] = {}
        self.next_issue = 1
        self.next_comment = 10000
        self.lock = threading.Lock()


def _matches(issue: dict, jql: str, store: Store) -> bool:
    jql = re.split(r"\s+ORDER BY\s+", jql, flags=re.I)[0]
    for clause in re.split(r"\s+AND\s+", jql, flags=re.I):
        clause = clause.strip()
        if not clause:
            continue
        f = issue["fields"]
        if m := re.fullmatch(r'project\s*=\s*"?([^"]+)"?', clause):
            ok = issue["key"].startswith(m[1] + "-")
        elif m := re.fullmatch(r'labels\s*=\s*"([^"]+)"', clause):
            ok = m[1] in f["labels"]
        elif m := re.fullmatch(r"key\s+in\s*\(([^)]*)\)", clause, flags=re.I):
            ok = issue["key"] in [k.strip().strip('"') for k in m[1].split(",")]
        elif m := re.fullmatch(r'summary\s*~\s*"([^"]*)"', clause):
            ok = m[1].lower() in f["summary"].lower()
        elif m := re.fullmatch(r'status\s*=\s*"([^"]+)"', clause):
            ok = f["status"]["name"].lower() == m[1].lower()
        else:
            raise ValueError(f"mock cannot parse JQL clause: {clause}")
        if not ok:
            return False
    return True


def _view(issue: dict, fields: list[str], store: Store) -> dict:
    f = issue["fields"]
    out = {k: f[k] for k in fields if k in f}
    if "issuelinks" in fields:
        out["issuelinks"] = f["issuelinks"]
    return {"id": issue["id"], "key": issue["key"], "fields": out}


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body=None):
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n)) if n else {}

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            parts = u.path.strip("/").split("/")
            with store.lock:
                if parts[3:4] == ["project"]:
                    return self._send(200, {"key": parts[4]}) if parts[4] == store.project else self._send(404, {"error": "no project"})
                if parts[3:4] == ["issue"] and len(parts) >= 5:
                    key = parts[4]
                    if key not in store.issues:
                        return self._send(404, {"error": "not found"})
                    if len(parts) == 5:
                        fields = q.get("fields", [""])[0].split(",")
                        return self._send(200, _view(store.issues[key], fields, store))
                    if parts[5] == "comment" and len(parts) == 6:
                        cs = store.comments.get(key, [])
                        s, m = int(q.get("startAt", ["0"])[0]), int(q.get("maxResults", ["50"])[0])
                        return self._send(200, {"startAt": s, "maxResults": m, "total": len(cs), "comments": cs[s:s + m]})
                    if parts[5] == "comment":
                        c = next((c for c in store.comments.get(key, []) if c["id"] == parts[6]), None)
                        return self._send(200, c) if c else self._send(404, {})
                    if parts[5] == "transitions":
                        cur = store.issues[key]["fields"]["status"]["name"]
                        return self._send(200, {"transitions": [{"id": str(i), "to": {"name": s}} for i, s in enumerate(STATUSES) if s != cur]})
                if parts[3:4] == ["myself"]:
                    return self._send(200, {"accountId": "mock"})
            self._send(404, {"error": f"mock: unhandled GET {u.path}"})

        def do_POST(self):
            u = urlparse(self.path)
            parts = u.path.strip("/").split("/")
            b = self._body()
            with store.lock:
                if u.path.endswith("/rest/api/3/issue"):
                    n = store.next_issue
                    store.next_issue += 1
                    key = f"{store.project}-{n}"
                    f = b["fields"]
                    store.issues[key] = {"id": str(10000 + n), "key": key, "fields": {
                        "summary": f["summary"], "description": f.get("description"), "labels": list(f.get("labels", [])),
                        "status": {"name": "To Do"}, "issuelinks": []}}
                    store.comments[key] = []
                    return self._send(201, {"id": store.issues[key]["id"], "key": key})
                if parts[3:4] == ["issue"] and parts[-1] == "comment":
                    key = parts[4]
                    cid = str(store.next_comment)
                    store.next_comment += 1
                    c = {"id": cid, "body": b["body"], "created": f"2026-11-14T00:00:{len(store.comments[key]):02d}.000+0000",
                         "author": {"displayName": "importer"}}
                    store.comments[key].append(c)
                    return self._send(201, c)
                if u.path.endswith("/rest/api/3/issueLink"):
                    a, z = b["inwardIssue"]["key"], b["outwardIssue"]["key"]
                    def brief(k): return {"key": k, "fields": {"summary": store.issues[k]["fields"]["summary"], "status": store.issues[k]["fields"]["status"]}}
                    store.issues[a]["fields"]["issuelinks"].append({"type": {"name": b["type"]["name"]}, "outwardIssue": brief(z)})
                    store.issues[z]["fields"]["issuelinks"].append({"type": {"name": b["type"]["name"]}, "inwardIssue": brief(a)})
                    return self._send(201)
                if u.path.endswith("/rest/api/3/search/jql"):
                    try:
                        hits = [i for i in store.issues.values() if _matches(i, b["jql"], store)]
                    except ValueError as e:
                        return self._send(400, {"errorMessages": [str(e)]})
                    start = int(b.get("nextPageToken") or 0)
                    size = min(int(b.get("maxResults", 50)), 100)
                    page = hits[start:start + size]
                    res = {"issues": [_view(i, b.get("fields", []), store) for i in page]}
                    if start + size < len(hits):
                        res["nextPageToken"] = str(start + size)
                    else:
                        res["isLast"] = True
                    return self._send(200, res)
                if parts[3:4] == ["issue"] and parts[-1] == "transitions":
                    key = parts[4]
                    store.issues[key]["fields"]["status"] = {"name": STATUSES[int(b["transition"]["id"])]}
                    return self._send(204)
            self._send(404, {"error": f"mock: unhandled POST {u.path}"})

    return H


def start(port: int = 0, project: str = "TYN") -> tuple[ThreadingHTTPServer, Store, str]:
    store = Store(project)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, store, f"http://127.0.0.1:{srv.server_address[1]}"


def serve(port: int = 8099) -> None:
    srv, _, url = start(port)
    print(f"Mock Jira on {url}  (project TYN). Set JIRA_BASE_URL={url}  JIRA_EMAIL=x  JIRA_API_TOKEN=x. Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
