from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from rra import web


def test_validate_inputs_accepts_scope_fields_and_trims_whitespace():
    assert web.validate_inputs({"project": " TRA ", "label": "tyn-release-assessment-v2", "release": "R2.4"}) == {
        "project": "TRA", "label": "tyn-release-assessment-v2", "release": "R2.4"
    }


@pytest.mark.parametrize("payload", [
    None,
    {"project": "TRA", "label": "scope", "release": "R2.4/../../secret"},
    {"project": "", "label": "scope", "release": "R2.4"},
    {"project": 3, "label": "scope", "release": "R2.4"},
])
def test_validate_inputs_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        web.validate_inputs(payload)


def test_web_api_starts_run_and_serves_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path)
    web._runs.clear()

    def fake_run(run_id, fields):
        out = tmp_path / "fake-run"
        out.mkdir()
        (out / "report.md").write_text("# Test report", encoding="utf-8")
        with web._lock:
            web._runs[run_id].update(state="completed", out_dir=str(out), artifacts=["report.md"],
                                     summary={"overall_decision": "NEEDS_REVIEW", "run_status": "COMPLETED"})

    monkeypatch.setattr(web, "_perform_assessment", fake_run)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        req = Request(base + "/api/runs", data=json.dumps({"project": "TRA", "label": "scope", "release": "R2.4"}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req) as response:
            assert response.status == 202
            run_id = json.load(response)["id"]
        with urlopen(base + f"/api/runs/{run_id}") as response:
            assert json.load(response)["state"] == "completed"
        with urlopen(base + f"/artifacts/{run_id}/report.md") as response:
            assert response.read().decode() == "# Test report"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        web._runs.clear()


def test_serve_refuses_non_local_bind():
    with pytest.raises(ValueError, match="localhost"):
        web.serve("0.0.0.0", 8765)
