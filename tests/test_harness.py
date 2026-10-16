import csv

import torch

from anatomy import harness
from anatomy.units import UNITS


def rows(out):
    with open(out / "passes.csv") as f:
        return list(csv.DictReader(f))


def test_dry_run_full_grid_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PROBE_S", 0.002)
    expected = len(harness.tiny_runs()) * len(UNITS) * harness.DRY_PROTOCOL["passes"]

    harness.profile(resume=False, dry_run=True, out_dir=tmp_path)
    first = rows(tmp_path)
    assert len(first) == expected and all(r["status"] in ("ok", "short_window") for r in first)
    assert (tmp_path / "idle.csv").exists()

    harness.profile(resume=True, dry_run=True, out_dir=tmp_path)
    assert len(rows(tmp_path)) == expected  # nothing redone

    # Simulate a session killed mid-unit: drop the last pass, resume redoes exactly that one.
    with open(tmp_path / "passes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, harness.FIELDS)
        w.writeheader()
        w.writerows(first[:-1])
    harness.profile(resume=True, dry_run=True, out_dir=tmp_path)
    after = rows(tmp_path)
    assert len(after) == expected
    assert {k: after[-1][k] for k in ("unit", "pass_idx")} == {k: first[-1][k] for k in ("unit", "pass_idx")}


def test_oom_is_logged_and_run_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PROBE_S", 0.002)
    real_build = harness.build

    def build(unit, run, device):
        if unit == "mlp":
            raise torch.OutOfMemoryError("fake")
        return real_build(unit, run, device)

    monkeypatch.setattr(harness, "build", build)
    harness.profile(resume=False, dry_run=True, out_dir=tmp_path)
    rs = rows(tmp_path)
    oom = [r for r in rs if r["status"] == "oom"]
    assert {r["unit"] for r in oom} == {"mlp"}
    assert len(oom) == len(harness.tiny_runs()) * harness.DRY_PROTOCOL["passes"]
    assert any(r["unit"] == "block" for r in rs)  # later units still ran
