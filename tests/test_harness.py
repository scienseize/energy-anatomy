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


def test_gpu_window_samples_clocks_and_reasons(monkeypatch):
    """Gpu.window with stub NVML/Zeus: clocks averaged, temp maxed, reason bits OR-ed over the window."""
    import itertools
    import time
    from types import SimpleNamespace

    monkeypatch.setattr(harness, "SAMPLE_S", 0.01)
    clocks = itertools.cycle([1500, 600])
    reasons = itertools.cycle([0x4, 0x20])
    nvml = SimpleNamespace(
        NVML_CLOCK_SM=0, NVML_CLOCK_MEM=1, NVML_TEMPERATURE_GPU=0,
        nvmlDeviceGetClockInfo=lambda h, c: next(clocks) if c == 0 else 5000,
        nvmlDeviceGetTemperature=lambda h, t: 75,
        nvmlDeviceGetEnforcedPowerLimit=lambda h: 70000,
    )
    monitor = SimpleNamespace(begin_window=lambda k: None,
                              end_window=lambda k: SimpleNamespace(gpu_energy={0: 140.0}, time=2.0))
    gpu = object.__new__(harness.Gpu)
    gpu.nvml, gpu.h, gpu.monitor = nvml, None, monitor
    gpu._get_reasons = lambda h: next(reasons)

    e, t, ctx = gpu.window(lambda: time.sleep(0.1))
    assert (e, t) == (140.0, 2.0)
    assert 600 <= ctx["sm_clock_mhz"] <= 1500 and ctx["mem_clock_mhz"] == 5000 and ctx["temp_c"] == 75
    assert ctx["throttle_reasons"] == "sw_power_cap|sw_thermal" and ctx["power_limit_w"] == 70.0


def test_refuses_to_append_under_a_different_header(tmp_path):
    import pytest

    (tmp_path / "passes.csv").write_text("run_id,ts,model\n")
    with pytest.raises(ValueError, match="different column layout"):
        harness.profile(resume=True, dry_run=True, out_dir=tmp_path)
    assert (tmp_path / "passes.csv").read_text() == "run_id,ts,model\n"  # untouched
