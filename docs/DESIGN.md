# Design notes

## Architecture
Importer → Jira → read-only tools → Gemini tool-calling loop → validated decision ledger → report.

**Scope isolation in code, not in the prompt.** The model never writes JQL. Every query is prefixed with
`project = P AND labels = <dataset label>`, inputs are whitelisted (`[A-Za-z0-9._-]`), and out-of-scope linked issues
simply disappear from results. A tested foreign issue with a colliding source id is invisible to the agent.

## Tools (all read-only)
| Tool | Purpose |
|---|---|
| `get_release_manifest` | target build, **authoritative policy text**, the list of requirements to assess |
| `search_issues` | filtered, compact discovery (kind / release / status / text) |
| `get_issue` | description (acceptance criterion), status, comment count, direct links |
| `get_linked_issues(depth≤3)` | follow links to evidence **and dependencies** (a failure can sit two hops away) |
| `get_comments` | parsed scenario actor/role/recorded time, ordered by recorded time, with a filter, paging and **collapsing of repetitive threads** |
| `compare_scope_to_target` | mechanical release/commit/service/environment/prompt comparison for one comment |
| `submit_requirement_decision`, `finish_assessment` | validated output channel |

## Correctness guards (why the output can be trusted)
* **No invented evidence**: `submit_requirement_decision` rejects any ref the run did not actually retrieve.
  PASS/FAIL need evidence; INSUFFICIENT_EVIDENCE must state what is outstanding; a requirement must have been read first.
* **Missing proof ≠ failure** is built into the decision vocabulary and the system prompt.
* **Policy comes from Jira** (the manifest), not from my code, so the rules the evidence is judged against are themselves evidence-backed.
* **Applicability is a first-class step**: the system prompt makes the model check release, commit, service, environment and prompt;
  the scope tool does the mechanical comparison but is labelled advisory (a field absent from one comment may appear in another,
  and the model is told not to assume it).
* **Recorded vs native metadata**: actors/roles/timestamps are parsed from the comment text; Jira authors are never shown to the model.
* **Untrusted text**: ticket content is treated as data. A heuristic flags comments that address reviewers or try to dictate
  a verdict; the model sees a warning, ignores them, and the report lists them under *Integrity notes*. Decision rules cannot be changed from ticket text.
* **Overall decision is deterministic** (`report.overall_decision`).

## Context management
Bounded outputs (≤9k chars per tool result, comment text capped), repetitive comment threads collapsed (a 62-comment noise thread
costs one summary line), per-requirement working: after each decision, earlier evidence results are replaced by stubs (the manifest is kept),
and any provider-side context caching is reported in the run statistics.

## Bounded execution and failure handling
Budgets on turns, tool calls, total tokens (prompt + output incl. thinking) and wall clock; the Gemini client paces requests (free-tier friendly), honours the server's retry delay on per-minute limits, retries 5xx, and fails fast on key, access, model and daily-quota errors; Jira client retries 429/5xx with `Retry-After`.
Tool failures are returned to the model as error results (and counted). If the run ends early, undecided requirements are listed as
`INSUFFICIENT_EVIDENCE` flagged **NOT ASSESSED** with the reason, the run status is `INCOMPLETE`/`FAILED`, and the overall result can never be READY.

## Known limits
* Jira `Relates` links are undirected, so link direction is not preserved (the dataset gives none beyond "linked").
* Status transitions depend on your project's workflow; unmatched statuses are reported as import warnings, not failures.
* The scope extractor is regex-based on this dataset's phrasing; it is advisory only.
* LLM output varies run to run. Run `assess` a few times and use `compare` to check stability; the validator prevents fabricated
  references but cannot prove the *reasoning* is right.
* The injection detector is a heuristic warning, not a security boundary; the real defence is that policy only comes from the manifest and decisions are evidence-validated.
* The mock server implements only the Jira subset used here.

## Measured results (fill in after your run)
Add the actual numbers from `runs/<ts>/report.md` ("Run statistics"): wall-clock, turns, tool calls, token usage, model, and `compare` output.

## How AI assistance was used
An AI assistant helped design and write the code and tests; I reviewed the code and verified behaviour with the test suite (49 tests, mock Jira,
fake Gemini endpoint). **Edit this paragraph to describe your own use accurately.** At runtime the agent calls a Gemini Flash model through function calling.
