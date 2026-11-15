from rra.importer import run_import, verify_import
from conftest import DATASET


def test_import_counts_and_fidelity(env, dataset):
    s = env["stats"]
    assert s.issues_created == 400 and s.comments_added == 2059 and not s.warnings
    assert verify_import(env["jira"], "TYN", DATASET) == []


def test_import_is_idempotent(env):
    s = run_import(env["jira"], "TYN", DATASET, log=lambda *_: None)
    assert (s.issues_created, s.comments_added, s.links_added) == (0, 0, 0)
    assert s.issues_existing == 400


def test_status_and_labels_and_source_ids(env, dataset):
    store = env["store"]
    iss = next(i for i in store.issues.values() if "src-REQ-08" in i["fields"]["labels"])
    assert iss["fields"]["status"]["name"] == "Done"
    assert {"tyn-release-assessment-v2", "rr-requirement", "rel-R2.4"} <= set(iss["fields"]["labels"])
    assert iss["fields"]["summary"].startswith("[REQ-08]")


def test_native_author_is_not_the_scenario_actor(env):
    store = env["store"]
    key = next(i["key"] for i in store.issues.values() if "src-REQ-02" in i["fields"]["labels"])
    c = store.comments[key][0]
    assert c["author"]["displayName"] == "importer"  # scenario actor lives only in text
