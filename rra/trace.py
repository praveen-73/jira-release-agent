from __future__ import annotations

import json
import time
from pathlib import Path


class Trace:
    """Append-only JSONL trace + a human-readable markdown rendering."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = out_dir / "trace.jsonl"
        self.jsonl.write_text("", encoding="utf-8")
        self.events: list[dict] = []
        self.t0 = time.time()

    def add(self, **ev) -> None:
        ev = {"t": round(time.time() - self.t0, 2), **ev}
        self.events.append(ev)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")

    def render_md(self) -> str:
        lines = ["# Investigation trace", "", "| # | t(s) | turn | tool | input | result | ms |", "|---|---|---|---|---|---|---|"]
        n = 0
        for e in self.events:
            if e["type"] != "tool_call":
                continue
            n += 1
            inp = json.dumps(e["input"], ensure_ascii=False)[:110].replace("|", "\\|")
            res = ("ERROR: " + e["error"] if e.get("error") else f"{e['output_chars']} chars").replace("|", "\\|")[:120]
            lines.append(f"| {n} | {e['t']} | {e['turn']} | `{e['tool']}` | `{inp}` | {res} | {e['ms']} |")
        lines += ["", "## Model turns", "", "| turn | t(s) | prompt tok | completion tok | cached | finish | assistant text (truncated) |", "|---|---|---|---|---|---|---|"]
        for e in self.events:
            if e["type"] == "model_call":
                txt = (e.get("text") or "").replace("\n", " ").replace("|", "\\|")[:140]
                u = e.get("usage", {})
                lines.append(f"| {e['turn']} | {e['t']} | {u.get('input_tokens', 0)} | {u.get('output_tokens', 0)} | {u.get('cached_input_tokens', 0)} | {e.get('stop_reason')} | {txt} |")
        events = [e for e in self.events if e["type"] in ("budget", "failure", "prune", "nudge")]
        if events:
            lines += ["", "## Notable events", ""] + [f"- t={e['t']}s **{e['type']}**: {e.get('detail')}" for e in events]
        return "\n".join(lines) + "\n"
