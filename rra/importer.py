"""Repeatable dataset importer.

Idempotent: every issue carries labels  <dataset_label>, rr-<kind>, rel-<release>, src-<source_id>.
Re-running finds existing issues by those labels and only adds what is missing
(issues, then comments by position, then links, then status transitions).
Scenario actor names, roles and timestamps stay verbatim inside the comment text.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .jira_client import JiraClient, JiraError
from .textutil import adf_to_text, text_to_adf

STATE_FILE = ".import_state.json"


@dataclass
class ImportStats:
    issues_created: int = 0
    issues_existing: int = 0
    comments_added: int = 0
    links_added: int = 0
    transitions_done: int = 0
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def load_dataset(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("dataset_label", "target", "records"):
        if k not in data:
            raise SystemExit(f"dataset missing '{k}'")
    ids = [r["source_id"] for r in data["records"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("dataset contains duplicate source_id values")
    return data


def _labels(label: str, rec: dict) -> list[str]:
    return [label, f"rr-{rec['kind']}", f"rel-{rec['release']}", f"src-{rec['source_id']}"]


def _description(rec: dict) -> str:
    # Provenance header first, then the original description verbatim.
    return (f"Source ID: {rec['source_id']}\nSource release: {rec['release']}\n"
            f"Source recorded status: {rec['recorded_status']}\n\n{rec['description']}")


def _existing_by_source(jira: JiraClient, project: str, label: str) -> dict[str, dict]:
    issues = jira.search(f'project = "{project}" AND labels = "{label}"', ["labels", "summary", "status", "issuelinks"])
    out = {}
    for i in issues:
        src = next((l[4:] for l in i["fields"]["labels"] if l.startswith("src-")), None)
        if src:
            out[src] = i
    return out


def run_import(jira: JiraClient, project: str, dataset_path: str, issue_type: str = "Task",
               link_type: str = "Relates", workers: int = 4, log=print) -> ImportStats:
    t0 = time.time()
    data = load_dataset(dataset_path)
    label = data["dataset_label"]
    recs = data["records"]
    by_id = {r["source_id"]: r for r in recs}
    st = ImportStats()
    jira.get_project(project)  # fail early if project is wrong / inaccessible

    # ---- phase 1: issues ----
    existing = _existing_by_source(jira, project, label)
    keys: dict[str, str] = {s: i["key"] for s, i in existing.items()}
    st.issues_existing = len(existing)
    todo = [r for r in recs if r["source_id"] not in existing]
    log(f"[1/4] issues: {len(existing)} already imported, creating {len(todo)}")

    def create(r: dict) -> tuple[str, str]:
        res = jira.create_issue({
            "project": {"key": project},
            "issuetype": {"name": issue_type},
            "summary": f"[{r['source_id']}] {r['summary']}"[:255],
            "description": text_to_adf(_description(r)),
            "labels": _labels(label, r),
        })
        return r["source_id"], res["key"]

    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(create, r) for r in todo]):
            try:
                sid, key = fut.result()
                keys[sid] = key
                st.issues_created += 1
            except JiraError as e:
                st.warnings.append(f"create failed: {e}")
    missing = [r["source_id"] for r in recs if r["source_id"] not in keys]
    if missing:
        raise SystemExit(f"{len(missing)} issues could not be created (e.g. {missing[:3]}); re-run to resume. "
                         f"Warnings: {st.warnings[:3]}")

    # ---- phase 2: comments (resume by position) ----
    log("[2/4] comments")

    def add_comments(r: dict) -> int:
        key = keys[r["source_id"]]
        have = sum(1 for _ in jira.iter_comments(key))
        n = 0
        for text in r["comments"][have:]:
            jira.add_comment(key, text_to_adf(text))
            n += 1
        return n

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(add_comments, r): r["source_id"] for r in recs if r["comments"]}
        for fut in as_completed(futs):
            try:
                st.comments_added += fut.result()
            except JiraError as e:
                st.warnings.append(f"comments failed for {futs[fut]}: {e}")

    # ---- phase 3: links ----
    log("[3/4] links")
    have_links: set[tuple[str, str]] = set()
    for sid, i in _existing_by_source(jira, project, label).items():
        for l in i["fields"].get("issuelinks", []):
            other = (l.get("outwardIssue") or l.get("inwardIssue") or {}).get("key")
            if other:
                have_links.add(frozenset((i["key"], other)))  # type: ignore[arg-type]
    for r in recs:
        for target in r["links"]:
            if target not in by_id:
                st.warnings.append(f"{r['source_id']} links to unknown {target}; skipped")
                continue
            pair = frozenset((keys[r["source_id"]], keys[target]))
            if pair in have_links:
                continue
            try:
                jira.link_issues(link_type, keys[r["source_id"]], keys[target])
                have_links.add(pair)  # type: ignore[arg-type]
                st.links_added += 1
            except JiraError as e:
                st.warnings.append(f"link {r['source_id']}->{target} failed: {e}")

    # ---- phase 4: recorded status (best effort; workflow may differ) ----
    log("[4/4] statuses")
    current = {s: i["fields"]["status"]["name"] for s, i in _existing_by_source(jira, project, label).items()}
    for r in recs:
        want, key = r["recorded_status"], keys[r["source_id"]]
        if current.get(r["source_id"], "").lower() == want.lower() or want.lower() == "to do":
            continue
        try:
            tr = next((t for t in jira.transitions(key) if t["to"]["name"].lower() == want.lower()), None)
            if tr:
                jira.transition(key, tr["id"])
                st.transitions_done += 1
            else:
                st.warnings.append(f"{r['source_id']}: no transition to '{want}'")
        except JiraError as e:
            st.warnings.append(f"{r['source_id']}: transition failed: {e}")

    Path(STATE_FILE).write_text(json.dumps({"project": project, "label": label, "keys": keys}, indent=1), encoding="utf-8")
    st.seconds = time.time() - t0
    return st


def verify_import(jira: JiraClient, project: str, dataset_path: str) -> list[str]:
    """Compare Jira against the dataset: issue count, per-issue comment count and text, links."""
    data = load_dataset(dataset_path)
    label = data["dataset_label"]
    problems: list[str] = []
    existing = _existing_by_source(jira, project, label)
    for r in data["records"]:
        i = existing.get(r["source_id"])
        if not i:
            problems.append(f"missing issue for {r['source_id']}")
            continue
        got = [adf_to_text(c["body"]) for c in jira.iter_comments(i["key"])]
        if sorted(got) != sorted(r["comments"]):
            problems.append(f"{r['source_id']}: comments differ ({len(got)} in Jira vs {len(r['comments'])})")
        nlinks = len({(l.get("outwardIssue") or l.get("inwardIssue") or {}).get("key")
                      for l in i["fields"].get("issuelinks", [])})
        if nlinks < len(r["links"]):
            problems.append(f"{r['source_id']}: {nlinks} links in Jira < {len(r['links'])} expected")
    extra = set(existing) - {r["source_id"] for r in data["records"]}
    if extra:
        problems.append(f"unexpected issues in scope: {sorted(extra)[:5]}")
    return problems
