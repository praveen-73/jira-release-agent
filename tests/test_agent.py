import json

from rra.agent import Budget, run_agent
from rra.llm import LLMResponse, ToolCall
from rra.report import build_report, overall_decision
from rra.tools import EvidenceTools
from rra.trace import Trace
from conftest import LABEL

EXPECTED = json.load(open("tests/expected_decisions.json"))["decisions"]
USAGE = {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 40}


def resp(*calls, text=""):
    tcs = [ToolCall(f"call_{n}_{id(c)}", c[0], c[1], json.dumps(c[1])) for n, c in enumerate(calls)]
    msg = {"role": "assistant", "content": text or None}
    if tcs:
        msg["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.raw_arguments}} for t in tcs]
    return LLMResponse(text, tcs, "tool_calls" if tcs else "stop", dict(USAGE), msg)


class ScriptedModel:
    """Stand-in for the LLM: drives the real tools/loop deterministically. Tests plumbing, not reasoning."""

    def __init__(self):
        self.reqs, self.i, self.stage, self.linked, self.sent = [], 0, "start", [], []

    @staticmethod
    def _last_results(messages):
        out = []
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            out.append(json.loads(m["content"]))
        return out[::-1]

    def chat(self, model, messages, tools):
        self.sent.append(json.loads(json.dumps(messages)))     # snapshot what the model would have seen
        results = self._last_results(messages)
        if self.stage == "start":
            self.stage = "manifest"
            return resp(("get_release_manifest", {}), text="Loading the manifest.")
        if self.stage == "manifest":
            self.reqs = [r["source_id"] for r in results[0]["requirements_to_assess"]]
            self.stage = "issue"
        if self.stage == "issue":
            if self.i >= len(self.reqs):
                self.stage = "done"
                return resp(("finish_assessment", {"summary": "scripted", "integrity_notes": []}))
            self.stage = "links"
            return resp(("get_issue", {"source_id": self.reqs[self.i]}))
        if self.stage == "links":
            self.stage = "comments"
            return resp(("get_linked_issues", {"source_id": self.reqs[self.i], "depth": 2}))
        if self.stage == "comments":
            self.linked = [n["source_id"] for n in results[0]["linked"] if n["kind"] != "requirement"]
            self.stage = "decide"
            return resp(*[("get_comments", {"source_id": s}) for s in self.linked])
        if self.stage == "decide":
            refs = [c["ref"] for r in results for c in r["comments"]
                    if c["role"] in ("QA Lead", "Security Owner", "Developer", "Engineering Manager")]
            rid, want = self.reqs[self.i], EXPECTED[self.reqs[self.i]]
            args = {"requirement_id": rid, "decision": want, "explanation": f"Scripted decision for {rid} based on retrieved comments.",
                    "evidence_refs": refs[:2] or [self.linked[0]], "outstanding": ["scripted follow-up"] if want == "INSUFFICIENT_EVIDENCE" else []}
            self.i += 1
            self.stage = "issue"
            return resp(("submit_requirement_decision", args))
        raise AssertionError(self.stage)


def _run(env, tmp_path, client, budget=None):
    tools = EvidenceTools(env["jira"], "TYN", LABEL, "R2.4")
    trace = Trace(tmp_path)
    return tools, trace, run_agent(tools, trace, client, "test-model", budget or Budget())


def test_full_loop_matches_expected_and_blocks(env, tmp_path):
    model = ScriptedModel()
    tools, trace, run = _run(env, tmp_path, model)
    assert run.status == "COMPLETED" and run.tool_errors == 0
    data, md = build_report(tools, run, "TYN")
    assert {d["requirement_id"]: d["decision"] for d in data["decisions"]} == EXPECTED
    assert data["overall_decision"] == "BLOCKED"
    assert all(f"REQ-{i:02d}" in md for i in range(1, 13)) and "Run statistics" in md
    assert run.usage["input_tokens"] == 100 * run.turns and run.usage["cached_input_tokens"] == 40 * run.turns
    events = [e for e in trace.events if e["type"] == "tool_call"]
    assert len(events) == run.tool_calls > 12
    assert {"get_release_manifest", "get_linked_issues", "get_comments", "submit_requirement_decision"} <= {e["tool"] for e in events}
    assert "| `get_comments` |" in trace.render_md()
    # protocol: every assistant tool_call is answered by a tool message with the same id
    last = model.sent[-1]
    asked = {c["id"] for m in last if m["role"] == "assistant" for c in m.get("tool_calls", [])}
    answered = {m["tool_call_id"] for m in last if m["role"] == "tool"}
    assert asked - answered <= {c["id"] for c in last[-1].get("tool_calls", [])}   # only the very last call may be pending
    # context management: earlier evidence results were pruned, manifest kept
    assert sum(m.get("content") == "[pruned after decision to save context; call the tool again if needed]" for m in last) > 5
    assert any(m["role"] == "tool" and "requirements_to_assess" in m["content"] for m in last)


def test_budget_exhaustion_reported_honestly(env, tmp_path):
    tools, trace, run = _run(env, tmp_path, ScriptedModel(), Budget(max_turns=6))
    assert run.status == "INCOMPLETE" and "turn budget" in run.failures[0]
    data, md = build_report(tools, run, "TYN")
    assert data["overall_decision"] in ("NEEDS_REVIEW", "BLOCKED") and "NOT ASSESSED" in md
    assert len(data["decisions"]) == 12


def test_token_budget(env, tmp_path):
    _, _, run = _run(env, tmp_path, ScriptedModel(), Budget(max_total_tokens=500))
    assert run.status == "INCOMPLETE" and "token budget" in run.failures[0]


def test_model_api_failure_reported(env, tmp_path):
    class Boom:
        def chat(self, *a, **k): raise RuntimeError("503 overloaded")
    tools, trace, run = _run(env, tmp_path, Boom())
    assert run.status == "FAILED" and "503" in run.failures[0]
    data, md = build_report(tools, run, "TYN")
    assert data["overall_decision"] == "NEEDS_REVIEW" and "FAILED" in md


def test_tool_errors_and_bad_json_returned_to_model(env, tmp_path):
    class Bad(ScriptedModel):
        def chat(self, model, messages, tools):
            if self.stage == "start":
                self.stage = "manifest"
                r = resp(("get_issue", {"source_id": "NOPE-1"}), ("get_release_manifest", {}))
                r.tool_calls.append(ToolCall("call_bad", "get_issue", None, "{not json"))
                r.assistant_message["tool_calls"].append({"id": "call_bad", "type": "function", "function": {"name": "get_issue", "arguments": "{not json"}})
                return r
            return super().chat(model, messages, tools)
    tools, trace, run = _run(env, tmp_path, Bad(), Budget(max_turns=2))
    assert run.tool_errors == 2
    errs = [e["error"] for e in trace.events if e["type"] == "tool_call" and e.get("error")]
    assert any("not found" in e for e in errs) and any("not valid JSON" in e for e in errs)


def test_overall_decision_policy():
    assert overall_decision(["PASS", "FAIL", "INSUFFICIENT_EVIDENCE"]) == "BLOCKED"
    assert overall_decision(["PASS", "INSUFFICIENT_EVIDENCE"]) == "NEEDS_REVIEW"
    assert overall_decision(["PASS"] * 12) == "READY"
