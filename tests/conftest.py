import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rra import mock_jira
from rra.config import JiraConfig
from rra.importer import run_import
from rra.jira_client import JiraClient

DATASET = str(Path(__file__).resolve().parents[1] / "inputs" / "dataset.json")
LABEL = "tyn-release-assessment-v2"


@pytest.fixture(scope="session")
def env():
    srv, store, url = mock_jira.start(0, "TYN")
    jira = JiraClient(JiraConfig(url, "x", "x"), max_retries=1)
    stats = run_import(jira, "TYN", DATASET, workers=4, log=lambda *_: None)
    # a foreign, un-labelled issue that must never be visible to the agent
    jira.create_issue({"project": {"key": "TYN"}, "issuetype": {"name": "Task"}, "summary": "[REQ-99] foreign",
                       "labels": ["some-other-dataset", "src-REQ-99"], "description": None})
    yield {"jira": jira, "store": store, "stats": stats, "url": url}
    srv.shutdown()


@pytest.fixture(scope="session")
def dataset():
    return json.loads(Path(DATASET).read_text())
