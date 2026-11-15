"""Minimal Jira Cloud REST v3 client: retries on 429/5xx, thread-safe via per-thread sessions."""
from __future__ import annotations

import threading
import time
from typing import Any, Iterator

import requests

from .config import JiraConfig


class JiraError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Jira HTTP {status}: {message}")
        self.status = status


class JiraClient:
    def __init__(self, cfg: JiraConfig, max_retries: int = 5, timeout: float = 30.0):
        self.cfg = cfg
        self.base = cfg.base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._local = threading.local()
        self.request_count = 0
        self._lock = threading.Lock()

    @property
    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.auth = (self.cfg.email, self.cfg.token)
            s.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
            self._local.s = s
        return s

    def _request(self, method: str, path: str, **kw) -> Any:
        url = f"{self.base}{path}"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._lock:
                    self.request_count += 1
                r = self._session.request(method, url, timeout=self.timeout, **kw)
            except requests.RequestException as e:  # network error -> retry
                last = e
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", min(2 ** attempt, 30)))
                last = JiraError(r.status_code, r.text[:200])
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                raise JiraError(r.status_code, r.text[:500])
            if r.status_code == 204 or not r.content:
                return None
            return r.json()
        raise JiraError(0, f"giving up after {self.max_retries + 1} attempts: {last}")

    # ---- reads ----
    def myself(self) -> dict:
        return self._request("GET", "/rest/api/3/myself")

    def get_project(self, key: str) -> dict:
        return self._request("GET", f"/rest/api/3/project/{key}")

    def search(self, jql: str, fields: list[str], max_results: int = 100, limit: int | None = None) -> list[dict]:
        out: list[dict] = []
        token = None
        while True:
            body: dict[str, Any] = {"jql": jql, "fields": fields, "maxResults": min(max_results, 100)}
            if token:
                body["nextPageToken"] = token
            page = self._request("POST", "/rest/api/3/search/jql", json=body)
            out.extend(page.get("issues", []))
            if limit is not None and len(out) >= limit:
                return out[:limit]
            token = page.get("nextPageToken")
            if not token or page.get("isLast"):
                return out

    def get_issue(self, key: str, fields: list[str]) -> dict:
        return self._request("GET", f"/rest/api/3/issue/{key}", params={"fields": ",".join(fields)})

    def iter_comments(self, key: str, page: int = 100) -> Iterator[dict]:
        start = 0
        while True:
            d = self._request("GET", f"/rest/api/3/issue/{key}/comment",
                              params={"startAt": start, "maxResults": page, "orderBy": "created"})
            cs = d.get("comments", [])
            yield from cs
            start += len(cs)
            if not cs or start >= d.get("total", 0):
                return

    def get_comment(self, key: str, comment_id: str) -> dict:
        return self._request("GET", f"/rest/api/3/issue/{key}/comment/{comment_id}")

    # ---- writes (importer only; the agent never receives these) ----
    def create_issue(self, fields: dict) -> dict:
        return self._request("POST", "/rest/api/3/issue", json={"fields": fields})

    def add_comment(self, key: str, adf_body: dict) -> dict:
        return self._request("POST", f"/rest/api/3/issue/{key}/comment", json={"body": adf_body})

    def link_issues(self, link_type: str, inward_key: str, outward_key: str) -> None:
        self._request("POST", "/rest/api/3/issueLink", json={
            "type": {"name": link_type},
            "inwardIssue": {"key": inward_key},
            "outwardIssue": {"key": outward_key},
        })

    def transitions(self, key: str) -> list[dict]:
        return self._request("GET", f"/rest/api/3/issue/{key}/transitions")["transitions"]

    def transition(self, key: str, transition_id: str) -> None:
        self._request("POST", f"/rest/api/3/issue/{key}/transitions", json={"transition": {"id": transition_id}})
