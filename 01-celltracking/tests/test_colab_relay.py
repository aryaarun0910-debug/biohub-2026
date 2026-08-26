"""Software contracts for the Colab relay (scripts/core/colab_relay.py, colab_factory.py).

The rule-8 gate is the load-bearing piece: a job must be REFUSED unless the host recorded a budget
whose scope and remaining units admit it. Nothing here touches the network.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))
import colab_relay as relay  # noqa: E402

_spec = importlib.util.spec_from_file_location("colab_factory", ROOT / "scripts" / "core" / "colab_factory.py")
factory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(factory)

NOW = "2026-08-26T22:00:00Z"


def active_session(units=30.0, kinds=("smoke", "notebook"), hours=12.0):
    return relay.new_session("30 units on L4, inference only", units, list(kinds), hours, gpu="L4", now_utc=NOW)


def notebook_job(max_hours=3.0):
    return relay.new_job("p20-test", "notebook", datasets=["pilkwang/biohub-tracking-support-pack-50ep-v1"],
                         max_hours=max_hours)


def test_inactive_session_refuses_everything():
    job = relay.new_job("smoke-1", "smoke")
    ok, reason = relay.job_allowed(job, dict(relay.SESSION_SCHEMA), 0.0, "Tesla T4", NOW)
    assert not ok and "inactive" in reason


def test_active_session_needs_host_words_and_budget():
    s = active_session()
    s["host_approval"] = ""
    assert any("host_approval" in e for e in relay.validate_session(s))
    s = active_session(units=0.0)
    assert any("budget_units" in e for e in relay.validate_session(s))


def test_scope_and_budget_gate():
    s = active_session(units=30.0, kinds=("smoke",))
    ok, reason = relay.job_allowed(notebook_job(), s, 0.0, "NVIDIA L4", NOW)
    assert not ok and "scope" in reason
    s = active_session(units=30.0)
    ok, reason = relay.job_allowed(notebook_job(max_hours=3.0), s, 0.0, "NVIDIA L4", NOW)   # 15 units worst case
    assert ok, reason
    ok, reason = relay.job_allowed(notebook_job(max_hours=3.0), s, 20.0, "NVIDIA L4", NOW)  # 20 + 15 > 30
    assert not ok and "exceeds budget" in reason
    ok, reason = relay.job_allowed(notebook_job(max_hours=3.0), s, 0.0, "NVIDIA A100-SXM4-40GB", NOW)  # 36 > 30
    assert not ok and "A100" in reason


def test_expiry_and_stop_flag():
    s = active_session(hours=1.0)
    ok, reason = relay.job_allowed(relay.new_job("smoke-2", "smoke"), s, 0.0, "T4", "2026-08-26T23:30:00Z")
    assert not ok and "expired" in reason
    s = active_session()
    s["stop"] = True
    ok, reason = relay.job_allowed(relay.new_job("smoke-3", "smoke"), s, 0.0, "T4", NOW)
    assert not ok and "stop" in reason


def test_rates_override_and_estimate():
    assert relay.gpu_family("NVIDIA A100-SXM4-40GB") == "A100"
    assert relay.gpu_family("Tesla T4") == "T4"
    assert relay.gpu_family("something") == "unknown"
    assert relay.estimate_units("NVIDIA L4", 3600, {"L4": 4.82}) == pytest.approx(4.82)
    assert relay.estimate_units("NVIDIA L4", 1800) == pytest.approx(2.5)


def test_job_validation():
    bad = relay.new_job("Bad ID!", "notebook")
    assert any("id" in e for e in relay.validate_job(bad))
    bad = relay.new_job("ok-id", "notebook", datasets=["notaslug"])
    assert any("owner/slug" in e for e in relay.validate_job(bad))
    good = notebook_job()
    assert relay.validate_job(good) == []
    smoke = relay.new_job("smoke-4", "smoke", max_hours=5.0)
    assert smoke["max_hours"] <= 0.25 and smoke["datasets"] == [] and smoke["competition"] == ""


def test_kaggle_layout_plan_matches_notebook_conventions():
    job = notebook_job()
    job["input_root"] = relay.KAGGLE_INPUT   # the plan honours whatever root the job names
    plan = relay.kaggle_layout_plan(job)
    paths = {p["path"] for p in plan}
    assert "/kaggle/input/biohub-tracking-support-pack-50ep-v1" in paths
    assert "/kaggle/input/biohub-cell-tracking-during-development" in paths
    aliases = {p["alias"] for p in plan}
    assert "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1" in aliases
    assert "/kaggle/input/competitions/biohub-cell-tracking-during-development" in aliases


def test_select_relay_outputs_respects_size_cap(tmp_path):
    (tmp_path / "submission.csv").write_bytes(b"x" * 10)
    (tmp_path / "big.csv").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "metrics.json").write_text("{}")
    picked = relay.select_relay_outputs(tmp_path, ["*.csv", "*.json"], max_bytes=50)
    names = sorted(p.name for p in picked)
    assert names == ["metrics.json", "submission.csv"]


def test_worker_notebook_generates_valid_python_and_no_secrets():
    nb = factory.build_worker_notebook("owner/relay")
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4
    for c in code_cells:
        compile("".join(c["source"]), "<cell>", "exec")
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "owner/relay" in text
    assert "ghp_" not in text and "gho_" not in text
    assert 'userdata.get("GH_TOKEN")' in text
    assert "job_allowed" in text


def test_queue_writes_job_locally_without_push(tmp_path):
    relay_dir = tmp_path / "relay"
    relay_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=relay_dir, check=True)
    for d in ("jobs/pending", "runs"):
        (relay_dir / d).mkdir(parents=True)
    relay.write_json(relay_dir / "session.json", dict(relay.SESSION_SCHEMA))
    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    job = relay.new_job("t-job", "notebook", datasets=["a/b"], max_hours=1.0)
    jdir = factory.write_job(relay_dir, job, nb)
    assert (jdir / "job.json").exists() and (jdir / "notebook.ipynb").exists()
    written = json.loads((jdir / "job.json").read_text())
    assert written["notebook_sha256"] and written["id"] == "t-job"
    with pytest.raises(SystemExit):
        factory.write_job(relay_dir, job, nb)   # duplicate id refused


def test_layout_plan_uses_input_root_and_rewrite_counts_literals():
    job = notebook_job()
    job["input_root"] = "/content/kaggle/input"
    plan = relay.kaggle_layout_plan(job)
    assert {p["path"] for p in plan} == {"/content/kaggle/input/biohub-tracking-support-pack-50ep-v1",
                                          "/content/kaggle/input/biohub-cell-tracking-during-development"}
    nb = {"cells": [
        {"cell_type": "code", "source": ['x = Path("/kaggle/input/a")\n', 'y = "/kaggle/input/b"\n']},
        {"cell_type": "markdown", "source": ["/kaggle/input stays in prose\n"]},
        {"cell_type": "code", "source": ['w = "/kaggle/working"\n']},
    ]}
    n = relay.rewrite_input_root(nb, "/content/kaggle/input")
    assert n == 2
    assert "".join(nb["cells"][0]["source"]) == 'x = Path("/content/kaggle/input/a")\ny = "/content/kaggle/input/b"\n'
    assert "".join(nb["cells"][1]["source"]) == "/kaggle/input stays in prose\n"
    assert "".join(nb["cells"][2]["source"]) == 'w = "/kaggle/working"\n'


def test_write_job_rewrites_notebook_and_records_count(tmp_path):
    relay_dir = tmp_path / "relay"
    (relay_dir / "jobs" / "pending").mkdir(parents=True)
    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                                          "source": ['p = "/kaggle/input/x"\n']}],
                              "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    job = relay.new_job("rw-job", "notebook", datasets=["a/b"], max_hours=1.0)
    jdir = factory.write_job(relay_dir, job, nb)
    written = json.loads((jdir / "job.json").read_text())
    assert written["input_root"] == "/content/kaggle/input" and written["input_rewrites"] == 1
    text = (jdir / "notebook.ipynb").read_text()
    assert 'p = \\"/content/kaggle/input/x\\"' in text      # JSON-escaped source line
    assert 'p = \\"/kaggle/input/x\\"' not in text
