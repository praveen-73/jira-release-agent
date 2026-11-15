from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader (no dependency). Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class JiraConfig:
    base_url: str
    email: str
    token: str
    link_type: str = "Relates"
    issue_type: str = "Task"

    @classmethod
    def from_env(cls) -> "JiraConfig":
        missing = [k for k in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN") if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"Missing environment variables: {', '.join(missing)} (see .env.example)")
        return cls(
            base_url=os.environ["JIRA_BASE_URL"].rstrip("/"),
            email=os.environ["JIRA_EMAIL"],
            token=os.environ["JIRA_API_TOKEN"],
            link_type=os.environ.get("JIRA_LINK_TYPE", "Relates"),
            issue_type=os.environ.get("JIRA_ISSUE_TYPE", "Task"),
        )
