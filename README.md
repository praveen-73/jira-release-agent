# Jira Release Evidence Agent

A tool-using AI agent (Google Gemini function calling) that decides, requirement by requirement, whether the evidence **recorded in Jira**
satisfies a release's acceptance criteria, and produces one evidence-backed release-readiness report.

```
inputs/dataset.json ──(rra import)──▶ your Jira Cloud project ──(read-only tools)──▶ agent (Gemini) ──▶ report.md / report.json / trace.md
```

* The dataset JSON is **only import input**. The agent never sees it: it investigates through Jira-backed tools.
* The agent chooses which tools to call (manifest → requirement → linked records → comments → scope check → decision).
* The overall decision (READY / NEEDS_REVIEW / BLOCKED) is computed **in code** from the per-requirement decisions.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # one dependency: requests==2.34.2
cp .env.example .env                       # then fill in credentials (never commit .env)
```

1. **Jira**: in your own Jira Cloud site create a *dedicated* software project (e.g. key `TYN`; team-managed or
   company-managed). It needs the issue type `Task`, a link type `Relates`, and the standard To Do / In Progress / Done
   statuses (all defaults). Create an API token at <https://id.atlassian.com/manage-profile/security/api-tokens>.
2. **Gemini**: create a free API key in Google AI Studio (<https://aistudio.google.com/apikey>) and set `GEMINI_API_KEY`. Then run `python -m rra models` to list the model IDs your key can use and set `GEMINI_MODEL` (default `gemini-flash-latest`).

## Run

### Browser form (local)

Start the local-only web form with:

```bash
python -m rra web
```

Open <http://127.0.0.1:8765>. The form defaults to project `TRA`, dataset scope label
`tyn-release-assessment-v2`, and release `R2.4`. Submit starts one assessment at a time; the page shows its
status, decision, failure details, and links to `report.md`, `report.json`, `trace.md`, and `trace.jsonl`.
It reads Jira and Gemini credentials from `.env`; it does not accept or upload credentials or dataset files.
The server binds only to localhost. Use Ctrl+C in the terminal to stop it.

### Command line

```bash
# 1. Import (repeatable/idempotent; safe to re-run after a failure)
python -m rra import  --project TYN --dataset inputs/dataset.json
python -m rra verify  --project TYN                    # checks counts, comment text and links against the dataset

# 2. Assess
python -m rra assess  --project TYN --label tyn-release-assessment-v2 --release R2.4
# -> runs/<timestamp>/report.md  report.json  trace.md  trace.jsonl

# 3. (optional) diff against my own reading of the dataset
python -m rra compare runs/<timestamp>/report.json
```

`assess` inputs are exactly: Jira project, dataset scope (the label the importer applied), target release.
Exit codes: `0` completed, `3` run incomplete/failed (the report still says so honestly).
Budgets (all overridable): `--max-turns 100 --max-tool-calls 250 --max-tokens 1500000 --max-seconds 2400`.

### What the importer does
Per record: one issue titled `[<source_id>] <summary>`; labels `tyn-release-assessment-v2`, `rr-<kind>`,
`rel-<release>`, `src-<source_id>`; description = provenance header + original text verbatim; every comment added
verbatim (the `Recorded event <time> | actor <name> | role <role>` header stays in the text because native Jira
authors/timestamps belong to the importer); `Relates` links; and the recorded status via workflow transitions
(best effort, warnings listed if your workflow lacks a transition). Re-runs find issues by label and add only what is missing.

## Notes on the Gemini backend
* `rra/llm.py` calls the Gemini REST API directly (`generateContent` with function calling; plain `requests`, no SDK).
* **Free tier is rate-limited.** The client paces itself (`GEMINI_RPM`, default 10 requests/minute), waits as instructed on per-minute 429s,
  and fails fast with a clear message on daily-quota or "no free quota for this model" errors. A full run takes a few minutes
  because of this pacing. Free-tier prompts may be used by Google to improve its products; the dataset here is fictional.
* Gemini 3+/4 models require "thought signatures" to be sent back unchanged during function calling; the client replays the raw
  model parts verbatim, so this works without configuration.
* Errors are explicit: invalid key, 403 access, 404 unknown model (run `python -m rra models`), daily quota, no free quota.

## Test without Jira or an API key

```bash
pip install -r requirements-dev.txt && pytest -q          # 49 tests
python -m rra mock-jira --port 8099                       # local Jira-compatible mock, then point JIRA_BASE_URL at it
```
Tests import the full dataset into the mock, check fidelity and idempotency, exercise every tool on the tricky
records, drive the agent loop with a scripted stand-in model (budgets, failures, pruning), and verify the Gemini
request/response round trip (thought signatures, parallel tool results, quota errors) against a local fake endpoint. **They do not test the LLM's reasoning**; that is
what `assess` + `compare` are for.

## Repository layout
```
rra/importer.py     idempotent importer + verifier        rra/tools.py   scoped read-only tools + decision validation
rra/jira_client.py  REST v3 client (retry/backoff)         rra/agent.py   bounded tool-use loop, pruning, budgets
rra/textutil.py     ADF, recorded-event & scope parsing    rra/report.py  report + deterministic overall decision
rra/trace.py        JSONL + markdown trace                 rra/llm.py     Gemini client (function calling)
rra/mock_jira.py    offline Jira mock
tests/              pytest suite + expected_decisions.json (regression oracle, never read by the agent)
docs/DESIGN.md      design choices, limits, AI-assistance note
```
