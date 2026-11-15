"""Gemini client vs a fake generateContent endpoint; also drives the real agent loop through it."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rra.agent import Budget, run_agent
from rra.llm import GeminiChat, function_declarations, to_contents
from rra.tools import TOOL_SCHEMAS, EvidenceTools
from rra.trace import Trace
from conftest import LABEL


def _serve(responses, seen):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            seen.append({"path": self.path, "key": self.headers.get("x-goog-api-key")})
            self._reply(200, {"models": [{"name": "models/gemini-x-flash", "displayName": "X Flash", "supportedGenerationMethods": ["generateContent"]},
                                         {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]}]})
        def do_POST(self):
            seen.append({"path": self.path, "key": self.headers.get("x-goog-api-key"),
                         "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            code, body = responses[min(len([s for s in seen if "body" in s]) - 1, len(responses) - 1)]
            self._reply(code, body)
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1beta"


def ok(parts, prompt=100, cand=10, thoughts=0, cached=0):
    return 200, {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
                 "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": cand, "thoughtsTokenCount": thoughts, "cachedContentTokenCount": cached}}


def fc(name, args, sig=None, cid=None):
    p = {"functionCall": {"name": name, "args": args}}
    if cid: p["functionCall"]["id"] = cid
    if sig: p["thoughtSignature"] = sig
    return p


def client(url, **kw):
    return GeminiChat("AIza-test", url, timeout=10, rpm=0, **kw)


def test_schema_conversion_is_gemini_safe():
    decls = function_declarations(TOOL_SCHEMAS)
    assert len(decls) == 8
    manifest = next(d for d in decls if d["name"] == "get_release_manifest")
    assert "parameters" not in manifest                      # empty-properties OBJECT is rejected by Gemini
    txt = json.dumps(decls)
    assert '"default"' not in txt and '"additionalProperties"' not in txt
    search = next(d for d in decls if d["name"] == "search_issues")["parameters"]
    assert search["type"] == "OBJECT" and search["properties"]["kind"]["type"] == "STRING"
    assert "(default: 20)" in search["properties"]["max_results"]["description"]
    sub = next(d for d in decls if d["name"] == "submit_requirement_decision")["parameters"]
    assert sub["properties"]["evidence_refs"]["items"] == {"type": "STRING"} and sub["required"][0] == "requirement_id"


def test_message_conversion_groups_parallel_results_and_replays_raw_parts():
    raw = [fc("get_issue", {"source_id": "A"}, sig="SIG1", cid="id-1"), fc("get_issue", {"source_id": "B"}, cid="id-2")]
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "_gemini_parts": raw, "tool_calls": [
                {"id": "id-1", "type": "function", "function": {"name": "get_issue", "arguments": "{}"}},
                {"id": "id-2", "type": "function", "function": {"name": "get_issue", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "id-1", "content": json.dumps({"x": 1})},
            {"role": "tool", "tool_call_id": "id-2", "content": "plain text"},
            {"role": "user", "content": "nudge"}]
    system, contents = to_contents(msgs)
    assert system == "SYS" and [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][0]["thoughtSignature"] == "SIG1"           # signature replayed untouched
    resp = contents[2]["parts"]
    assert [p["functionResponse"]["id"] for p in resp[:2]] == ["id-1", "id-2"]
    assert resp[0]["functionResponse"]["response"] == {"x": 1} and resp[1]["functionResponse"]["response"] == {"result": "plain text"}
    assert resp[2] == {"text": "nudge"}                                    # text merged into the same user turn


def test_roundtrip_through_agent_loop_with_thought_signature(env, tmp_path):
    seen = []
    srv, url = _serve([
        ok([{"text": "loading"}, fc("get_release_manifest", {}, sig="SIG-A")], 100, 10, thoughts=30),
        ok([fc("get_issue", {"source_id": "REQ-01"}, sig="SIG-B")], 300, 12, cached=128),
        ok([{"text": "I am done."}]),
    ], seen)
    tools = EvidenceTools(env["jira"], "TYN", LABEL, "R2.4")
    run = run_agent(tools, Trace(tmp_path), client(url), "gemini-test-flash", Budget(max_turns=8))
    srv.shutdown()
    posts = [s for s in seen if "body" in s]
    assert posts[0]["path"].endswith("/models/gemini-test-flash:generateContent") and posts[0]["key"] == "AIza-test"
    b0 = posts[0]["body"]
    assert b0["systemInstruction"]["parts"][0]["text"].startswith("You are a release-readiness investigator")
    assert b0["tools"][0]["functionDeclarations"][0]["name"] == "get_release_manifest" and "temperature" not in b0["generationConfig"]
    third = posts[2]["body"]["contents"]
    assert [c["role"] for c in third][:5] == ["user", "model", "user", "model", "user"]
    assert third[1]["parts"][1]["thoughtSignature"] == "SIG-A" and third[3]["parts"][0]["thoughtSignature"] == "SIG-B"
    assert "requirements_to_assess" in json.dumps(third[2]["parts"][0]["functionResponse"]["response"])
    assert "REQ-01" in tools.fetched_requirements and run.tool_errors == 0
    assert run.usage["cached_input_tokens"] == 128 and run.usage["output_tokens"] >= 10 + 30 + 12   # thinking tokens counted
    assert run.status == "INCOMPLETE"                                    # model quit without finishing -> nudged, then reported


@pytest.mark.parametrize("code,body,expect", [
    (400, {"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid API key.", "details": [{"reason": "API_KEY_INVALID"}]}}, "rejected the API key"),
    (403, {"error": {"status": "PERMISSION_DENIED", "message": "API has not been used"}}, "denied access"),
    (404, {"error": {"status": "NOT_FOUND", "message": "models/foo is not found"}}, "rra models"),
    (429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}, "DAILY quota"),
    (429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded, limit: 0, model: gemini-x"}}, "no free-tier quota"),
    (400, {"error": {"status": "INVALID_ARGUMENT", "message": "Function call is missing a thought_signature"}}, "thought signatures")])
def test_errors_are_clear_and_not_retried(env, tmp_path, code, body, expect):
    seen = []
    srv, url = _serve([(code, body)], seen)
    run = run_agent(EvidenceTools(env["jira"], "TYN", LABEL, "R2.4"), Trace(tmp_path), client(url), "m", Budget(max_turns=2))
    srv.shutdown()
    assert run.status == "FAILED" and expect in run.failures[0] and len([s for s in seen if "body" in s]) == 1


def test_per_minute_429_waits_and_retries(monkeypatch):
    slept = []
    monkeypatch.setattr("rra.llm.time.sleep", lambda s: slept.append(s))
    seen = []
    srv, url = _serve([(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "rate limited", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}}), ok([{"text": "ok"}])], seen)
    r = client(url).chat("m", [{"role": "user", "content": "x"}], TOOL_SCHEMAS)
    srv.shutdown()
    assert r.text == "ok" and slept == [8.0] and len(seen) == 2


def test_503_is_retried(monkeypatch):
    monkeypatch.setattr("rra.llm.time.sleep", lambda s: None)
    seen = []
    srv, url = _serve([(503, {"error": {"status": "UNAVAILABLE", "message": "overloaded"}}), ok([{"text": "ok"}])], seen)
    assert client(url).chat("m", [{"role": "user", "content": "x"}], []).text == "ok"
    srv.shutdown()


def test_client_side_pacing(monkeypatch):
    t = {"now": 0.0}; slept = []
    monkeypatch.setattr("rra.llm.time.monotonic", lambda: t["now"])
    monkeypatch.setattr("rra.llm.time.sleep", lambda s: (slept.append(s), t.__setitem__("now", t["now"] + s)))
    c = GeminiChat("k", rpm=10)
    c._pace(); c._pace()
    assert slept == [pytest.approx(6.0)]


def test_blocked_and_empty_responses_do_not_crash():
    c = GeminiChat("k", rpm=0)
    r = c._parse({"promptFeedback": {"blockReason": "SAFETY"}})
    assert r.finish_reason == "BLOCKED:SAFETY" and r.tool_calls == []
    _, contents = to_contents([{"role": "user", "content": "x"}, r.assistant_message])
    assert contents[1]["parts"] == [{"text": "(no output)"}]


def test_list_models_and_key_header():
    seen = []
    srv, url = _serve([], seen)
    models = client(url).list_models()
    srv.shutdown()
    assert [m["name"] for m in models][0] == "models/gemini-x-flash" and seen[0]["key"] == "AIza-test"


def test_from_env(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        GeminiChat.from_env()
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert GeminiChat.from_env().key == "g-key"
