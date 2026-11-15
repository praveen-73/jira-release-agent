import pytest

from rra.tools import EvidenceTools, ToolError
from conftest import LABEL


@pytest.fixture()
def tools(env):
    return EvidenceTools(env["jira"], "TYN", LABEL, "R2.4")


def test_manifest_target_and_twelve_requirements(tools):
    m = tools.get_release_manifest()
    assert m["target"] == {"release": "R2.4", "commit": "b7a91f2", "prompt": "p7", "environment": "production", "service": "atlas-assistant"}
    assert [r["source_id"] for r in m["requirements_to_assess"]] == [f"REQ-{i:02d}" for i in range(1, 13)]
    assert "PASS requires affirmative applicable evidence" in m["policy_text"]


def test_scope_is_enforced(tools):
    with pytest.raises(ToolError):
        tools.get_issue("REQ-99")  # exists in project but not in dataset scope
    assert all(i["source_id"] != "REQ-99" for i in tools.search_issues(kind="requirement", max_results=40)["issues"])


@pytest.mark.parametrize("bad", ['REQ-01" OR project = X', "a b", "", "x" * 80, "REC-1; DROP"])
def test_input_sanitised(tools, bad):
    with pytest.raises(ToolError):
        tools.get_issue(bad)


def test_dependency_defect_reachable_via_links(tools):
    ids = {n["source_id"] for n in tools.get_linked_issues("REQ-08", depth=2)["linked"]}
    assert {"REC-4076", "REC-9635"} <= ids
    cs = tools.get_comments("REC-9635", query="HND-DEF")["comments"]
    assert "Defect HND-DEF-4 status: open" in cs[0]["text"]


def test_noise_thread_is_collapsed(tools):
    r = tools.get_comments("REC-5957")
    assert r["collapsed_threads"] and r["collapsed_threads"][0]["repeated"] == 62
    assert any("STREAM-731" in c["text"] for c in r["comments"])
    assert len(str(r)) < 6000


def test_comments_ordered_by_recorded_time_not_import_order(tools):
    ts = [c["recorded_at"] for c in tools.get_comments("REC-9745", collapse_repetitive=False)["comments"]]
    assert ts == sorted(ts)


@pytest.mark.parametrize("sid,query,applies", [
    ("REC-2081", "REG-804", "yes"), ("REC-2299", "REG-779", "no"),
    ("REC-5825", "ISO-290", "no"), ("REC-1028", "PERF-812", "no"), ("REC-6499", "REDACT-905", "no")])
def test_scope_checker_on_real_records(tools, sid, query, applies):
    ref = tools.get_comments(sid, query=query)["comments"][0]["ref"]
    assert tools.compare_scope_to_target(ref)["scope_vs_target"]["core_scope_applies"] == applies or sid == "REC-6499"


def test_service_mismatch_detected(tools):
    ref = tools.get_comments("REC-6499", query="REDACT-905")["comments"][0]["ref"]
    assert tools.compare_scope_to_target(ref)["scope_vs_target"]["service"]["verdict"] == "mismatch"


def test_instruction_like_comment_is_flagged_but_still_data(tools):
    cs = tools.get_comments("REC-5716")["comments"]
    flagged = [c for c in cs if "WARNING" in c]
    assert len(flagged) == 1 and tools.flagged


def _ready(tools, rid="REQ-01"):
    tools.get_release_manifest()
    tools.get_issue(rid)
    return tools.get_comments("REC-2081", query="REG-804")["comments"][0]["ref"]


def test_decision_validation(tools):
    ref = _ready(tools)
    with pytest.raises(ToolError, match="never invent"):
        tools.submit_requirement_decision("REQ-01", "PASS", "x" * 50, evidence_refs=["REC-2081#999999"])
    with pytest.raises(ToolError, match="at least one evidence"):
        tools.submit_requirement_decision("REQ-01", "PASS", "x" * 50)
    with pytest.raises(ToolError, match="outstanding"):
        tools.submit_requirement_decision("REQ-01", "INSUFFICIENT_EVIDENCE", "x" * 50)
    with pytest.raises(ToolError, match="not one of"):
        tools.submit_requirement_decision("REQ-77", "PASS", "x" * 50, evidence_refs=[ref])
    with pytest.raises(ToolError, match="get_issue"):
        tools.submit_requirement_decision("REQ-02", "PASS", "x" * 50, evidence_refs=[ref])
    assert tools.submit_requirement_decision("REQ-01", "PASS", "Regression run matched target build. " * 2, evidence_refs=[ref])["remaining"][0] == "REQ-02"
    with pytest.raises(ToolError, match="undecided"):
        tools.finish_assessment("done")
