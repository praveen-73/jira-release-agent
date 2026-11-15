"""Google Gemini (Gemini API / Google AI Studio key) client with function calling.

Plain REST via `requests` (no vendor SDK). The agent loop keeps a neutral message list
(system / user / assistant(+tool_calls) / tool); this module converts it to Gemini's
`contents` format on every call.

Gemini 3+/4 models REQUIRE the model's `thoughtSignature` parts to be sent back unchanged
during function calling, so each assistant message keeps the raw response parts in
`_gemini_parts` and they are replayed verbatim.

Env: GEMINI_API_KEY (or GOOGLE_API_KEY), GEMINI_MODEL, GEMINI_RPM (client-side pacing), GEMINI_BASE_URL.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-flash-latest"      # alias for the newest Flash; run `python -m rra models` to see exact IDs
SYNTH_ID = "gcall_"                         # prefix for tool-call ids we create when Gemini returns none


class GeminiError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict[str, int]       # input_tokens (prompt), output_tokens (answer + thinking), cached_input_tokens
    assistant_message: dict = field(default_factory=dict)   # append this to `messages` as-is


# ---------------------------------------------------------------- request conversion
def _clean_schema(s):
    """JSON-schema -> Gemini's OpenAPI subset: UPPERCASE types, no `default`, defaults moved into the description."""
    if isinstance(s, list):
        return [_clean_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k in ("default", "additionalProperties", "$schema", "title"):
            continue
        if k == "type" and isinstance(v, str):
            out[k] = v.upper()
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _clean_schema(pv) for pk, pv in v.items()}
        else:
            out[k] = _clean_schema(v)
    if "default" in s:
        out["description"] = (out.get("description", "") + f" (default: {s['default']})").strip()
    return out


def function_declarations(tools: list[dict]) -> list[dict]:
    decls = []
    for t in tools:
        fn = t["function"]
        d = {"name": fn["name"], "description": fn["description"]}
        params = _clean_schema(fn.get("parameters") or {})
        if params.get("properties"):          # Gemini rejects an OBJECT with no properties
            d["parameters"] = params
        decls.append(d)
    return decls


def _as_object(content: str) -> dict:
    try:
        v = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {"result": content}
    return v if isinstance(v, dict) else {"result": v}


def to_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    system, contents, id2name = "", [], {}

    def add(role: str, parts: list[dict]) -> None:
        if contents and contents[-1]["role"] == role:     # parallel tool results must share one user turn
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for m in messages:
        r = m["role"]
        if r == "system":
            system = m["content"]
        elif r == "user":
            add("user", [{"text": m["content"]}])
        elif r == "assistant":
            for tc in m.get("tool_calls", []):
                id2name[tc["id"]] = tc["function"]["name"]
            if m.get("_gemini_parts"):
                parts = copy.deepcopy(m["_gemini_parts"])     # includes thoughtSignature, untouched
            else:
                parts = [{"text": m["content"]}] if m.get("content") else []
                for tc in m.get("tool_calls", []):
                    parts.append({"functionCall": {"name": tc["function"]["name"],
                                                   "args": json.loads(tc["function"]["arguments"] or "{}")}})
            add("model", parts or [{"text": "(no output)"}])
        elif r == "tool":
            fr = {"name": id2name.get(m["tool_call_id"], "unknown"), "response": _as_object(m["content"])}
            if not m["tool_call_id"].startswith(SYNTH_ID):
                fr["id"] = m["tool_call_id"]
            add("user", [{"functionResponse": fr}])
    return system, contents


# ---------------------------------------------------------------- client
class GeminiChat:
    def __init__(self, api_key: str, base_url: str = BASE_URL, timeout: float = 180.0,
                 max_retries: int = 6, rpm: float = 10.0):
        self.key, self.base = api_key, base_url.rstrip("/")
        self.timeout, self.max_retries = timeout, max_retries
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last = float("-inf")
        self._n = 0

    @classmethod
    def from_env(cls) -> "GeminiChat":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set (add it to .env; get one at https://aistudio.google.com/apikey)")
        return cls(key, os.environ.get("GEMINI_BASE_URL", BASE_URL), rpm=float(os.environ.get("GEMINI_RPM", "10")))

    # -- low-level HTTP with pacing, retry/backoff and precise errors
    def _pace(self) -> None:
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _request(self, method: str, path: str, **kw) -> dict:
        headers = {"x-goog-api-key": self.key, "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                r = requests.request(method, f"{self.base}{path}", headers=headers, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                last = e
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code < 400:
                return r.json()
            wait = self._classify(r)             # raises for non-retryable errors, else returns seconds to wait
            last = GeminiError(f"HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(wait)
        raise GeminiError(f"Gemini request failed after {self.max_retries + 1} attempts: {last}")

    @staticmethod
    def _classify(r: requests.Response) -> float:
        try:
            err = r.json().get("error", {}) or {}
        except ValueError:
            err = {}
        msg, status = err.get("message") or r.text[:300], err.get("status", "")
        details = err.get("details") or []
        reasons = [d.get("reason", "") for d in details]
        quota_ids = [v.get("quotaId", "") for d in details for v in d.get("violations", [])]
        delay = None
        for d in details:
            if str(d.get("@type", "")).endswith("RetryInfo"):
                m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
                delay = float(m.group(1)) if m else None
        code, low = r.status_code, msg.lower()
        if "API_KEY_INVALID" in reasons or "api key not valid" in low or code == 401:
            raise GeminiError(f"Gemini rejected the API key ({code}): {msg}. Create a key at https://aistudio.google.com/apikey and set GEMINI_API_KEY.")
        if code == 403:
            raise GeminiError(f"Gemini denied access (403): {msg}. Check the key's restrictions, that the Generative Language API is enabled, and that your region is supported.")
        if code == 404:
            raise GeminiError(f"Model not found or not available to your key (404): {msg}. Run `python -m rra models` to list valid model IDs, then set GEMINI_MODEL.")
        if code == 429:
            if any("PerDay" in q for q in quota_ids) or "per day" in low or "daily" in low:
                raise GeminiError(f"Gemini DAILY quota used up (429): {msg}. It resets daily; wait, enable billing in Google AI Studio, or set GEMINI_MODEL to another model.")
            if re.search(r"limit:\s*0\b", msg):
                raise GeminiError(f"This model has no free-tier quota for your key (429): {msg}. Choose another model (`python -m rra models`) or enable billing.")
            return min(delay + 1 if delay is not None else 20.0, 120.0)      # per-minute limit: wait as instructed
        if code >= 500:
            return 5.0
        hint = " (thought signatures must be replayed unchanged; please report this)" if "thought" in low else ""
        raise GeminiError(f"Gemini HTTP {code} {status}: {msg}{hint}")

    # -- public API
    def chat(self, model: str, messages: list[dict], tools: list[dict], max_output_tokens: int = 8192) -> LLMResponse:
        system, contents = to_contents(messages)
        body: dict = {"contents": contents, "tools": [{"functionDeclarations": function_declarations(tools)}],
                      "generationConfig": {"maxOutputTokens": max_output_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return self._parse(self._request("POST", f"/models/{model}:generateContent", json=body))

    def list_models(self) -> list[dict]:
        out, token = [], None
        while True:
            d = self._request("GET", "/models", params={"pageSize": 100, **({"pageToken": token} if token else {})})
            out += d.get("models", [])
            token = d.get("nextPageToken")
            if not token:
                return out

    def _parse(self, d: dict) -> LLMResponse:
        cand = (d.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought")).strip()
        calls, raw_calls = [], []
        for p in parts:
            fc = p.get("functionCall")
            if not fc:
                continue
            self._n += 1
            cid = fc.get("id") or f"{SYNTH_ID}{self._n}"
            args = fc.get("args") or {}
            raw = json.dumps(args)
            calls.append(ToolCall(cid, fc.get("name", ""), args if isinstance(args, dict) else None, raw))
            raw_calls.append({"id": cid, "type": "function", "function": {"name": fc.get("name", ""), "arguments": raw}})
        finish = cand.get("finishReason", "")
        if not cand and (d.get("promptFeedback") or {}).get("blockReason"):
            finish = f"BLOCKED:{d['promptFeedback']['blockReason']}"
        assistant: dict = {"role": "assistant", "content": text or None, "_gemini_parts": parts}
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        u = d.get("usageMetadata") or {}
        usage = {"input_tokens": u.get("promptTokenCount", 0) or 0,
                 "output_tokens": (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0),
                 "cached_input_tokens": u.get("cachedContentTokenCount", 0) or 0}
        return LLMResponse(text, calls, finish, usage, assistant)

