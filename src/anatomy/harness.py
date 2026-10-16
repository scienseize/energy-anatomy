"""Profiling harness (DESIGN §6): per (config, unit) build once, warm up, calibrate R, then
measure N passes in energy windows, appending each pass to passes.csv as soon as it ends."""
import csv
import gc
import math
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Make CUDA, NVML and Zeus agree on device 0 (Kaggle's T4 x2). Must precede CUDA init.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from anatomy.config import DATA, ROOT, ModelSpec, RunSpec, load_grid, load_yaml
from anatomy.units import UNITS, build, kind, make_inputs, step

FIELDS = ["run_id", "ts", "model", "seq_len", "precision", "tokens", "unit", "unit_kind", "pass_idx",
          "reps", "energy_j", "time_s", "energy_per_rep_j", "avg_power_w", "sm_clock_mhz",
          "mem_clock_mhz", "temp_c", "power_limit_w", "throttled", "status", "torch", "cuda",
          "driver", "git_sha"]
IDLE_FIELDS = ["run_id", "ts", "energy_j", "time_s", "avg_power_w", "sm_clock_mhz", "mem_clock_mhz",
               "temp_c", "power_limit_w"]
PROBE_S = 0.2  # calibration probe length


class Gpu:
    """Zeus energy windows + NVML context on CUDA device 0."""

    device = "cuda"

    def __init__(self):
        import pynvml
        from zeus.monitor import ZeusMonitor

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        bus = getattr(torch.cuda.get_device_properties(0), "pci_bus_id", None)
        assert bus is None or pynvml.nvmlDeviceGetPciInfo(self.h).bus == bus, "NVML device 0 is not CUDA device 0"
        self.monitor = ZeusMonitor(gpu_indices=[0])
        self.driver = pynvml.nvmlSystemGetDriverVersion()

    def sync(self):
        torch.cuda.synchronize()

    def window(self, fn) -> tuple[float, float]:
        self.monitor.begin_window("w")  # syncs CUDA before reading the counter
        fn()
        m = self.monitor.end_window("w")
        return sum(m.gpu_energy.values()), m.time

    def context(self) -> dict:
        n, h = self.nvml, self.h
        return {"sm_clock_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                "mem_clock_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
                "temp_c": n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                "power_limit_w": n.nvmlDeviceGetEnforcedPowerLimit(h) / 1000}


class FakeGpu:
    """CPU stand-in for dry runs: energy = wall time x a constant power. Never real numbers."""

    device = "cpu"
    driver = "fake"
    POWER_W = 50.0

    def sync(self):
        pass

    def window(self, fn):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        return dt * self.POWER_W, dt

    def context(self):
        return {"sm_clock_mhz": 1590, "mem_clock_mhz": 5000, "temp_c": 40, "power_limit_w": 70.0}


def _git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _append(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fields)
        if new:
            w.writeheader()
        w.writerow(row)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(model, seq, prec, unit):
    return model, int(seq), prec, unit


class Profiler:
    def __init__(self, gpu, protocol: dict, out_dir: Path, resume: bool):
        self.gpu, self.p = gpu, protocol
        self.passes_csv, self.idle_csv = out_dir / "passes.csv", out_dir / "idle.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:8]
        self.static = {"run_id": self.run_id, "torch": torch.__version__, "cuda": torch.version.cuda or "",
                       "driver": gpu.driver, "git_sha": _git_sha()}
        self.done: dict[tuple, set[int]] = {}  # (model, seq, prec, unit) -> pass indices recorded
        self.ref_sm: dict[tuple, float] = {}  # pass-0 SM clock, the throttle reference
        if resume and self.passes_csv.exists():
            with open(self.passes_csv) as f:
                for r in csv.DictReader(f):
                    k = _key(r["model"], r["seq_len"], r["precision"], r["unit"])
                    self.done.setdefault(k, set()).add(int(r["pass_idx"]))
                    if r["pass_idx"] == "0" and r["sm_clock_mhz"]:
                        self.ref_sm[k] = float(r["sm_clock_mhz"])
        self.last_idle = -math.inf

    def idle_if_due(self) -> None:
        if time.monotonic() - self.last_idle < self.p["idle_every_min"] * 60:
            return
        self.gpu.sync()
        e, t = self.gpu.window(lambda: time.sleep(self.p["idle_window_s"]))
        _append(self.idle_csv, IDLE_FIELDS, {"run_id": self.run_id, "ts": _now(), "energy_j": e,
                                             "time_s": t, "avg_power_w": e / t, **self.gpu.context()})
        self.last_idle = time.monotonic()

    def _buffers(self, unit: str, run: RunSpec):
        """K rotating (inputs, grads) sets totalling > buffer_l2_multiple x L2."""
        mod, inputs, grads = build(unit, run, self.gpu.device)
        sets = [(inputs, grads)]
        nbytes = sum(t.numel() * t.element_size() for t in (*inputs, *(grads or ())))
        need = self.p["buffer_l2_multiple"] * self.p["l2_bytes"]
        k = max(self.p["min_buffers"], math.ceil(need / nbytes))
        for _ in range(k - 1):
            g = None if grads is None else tuple(torch.randn_like(t) for t in grads)
            sets.append((make_inputs(unit, run, self.gpu.device), g))
        return mod, sets

    def _reps(self, mod, sets, n: int) -> None:
        for i in range(n):
            step(mod, *sets[i % len(sets)])

    def _calibrate(self, mod, sets) -> int:
        n = 1
        while True:
            self.gpu.sync()
            t = time.perf_counter()
            self._reps(mod, sets, n)
            self.gpu.sync()
            dt = time.perf_counter() - t
            if dt >= PROBE_S:
                return math.ceil(1.1 * self.p["min_window_s"] / (dt / n))
            n *= 2

    def unit(self, run: RunSpec, unit: str) -> None:
        k = _key(run.model.name, run.seq_len, run.precision, unit)
        todo = [j for j in range(self.p["passes"]) if j not in self.done.get(k, set())]
        if not todo:
            return
        base = {**self.static, "model": run.model.name, "seq_len": run.seq_len, "precision": run.precision,
                "tokens": run.tokens, "unit": unit, "unit_kind": kind(unit)}
        mod = sets = None
        try:
            mod, sets = self._buffers(unit, run)
            self._reps(mod, sets, self.p["warmup_iters"])
            reps = self._calibrate(mod, sets)
            for j in todo:
                self.idle_if_due()
                e, t = self.gpu.window(lambda: self._reps(mod, sets, reps))
                ctx = self.gpu.context()
                ref = self.ref_sm.setdefault(k, ctx["sm_clock_mhz"]) if j == 0 else self.ref_sm.get(k)
                throttled = ref is not None and ctx["sm_clock_mhz"] < (1 - self.p["throttle_drop"]) * ref
                _append(self.passes_csv, FIELDS, {
                    **base, "ts": _now(), "pass_idx": j, "reps": reps, "energy_j": e, "time_s": t,
                    "energy_per_rep_j": e / reps, "avg_power_w": e / t, **ctx, "throttled": throttled,
                    "status": "ok" if t >= self.p["min_window_s"] else "short_window"})
                self.done.setdefault(k, set()).add(j)
        except torch.OutOfMemoryError:
            # Logged, never silently dropped; resume treats these passes as done.
            for j in todo:
                if j not in self.done.get(k, set()):
                    _append(self.passes_csv, FIELDS, {**base, "ts": _now(), "pass_idx": j, "status": "oom"})
                    self.done.setdefault(k, set()).add(j)
        finally:
            del mod, sets
            gc.collect()
            if self.gpu.device == "cuda":
                torch.cuda.empty_cache()

    def run(self, runs: list[RunSpec], units=UNITS) -> None:
        # Config-major in grid order (160M first): each finished config is analyzable on its own.
        for run in runs:
            for u in units:
                self.unit(run, u)


def tiny_runs() -> list[RunSpec]:
    """Dry-run grid: same structure as the real one at toy dims (CPU, seconds)."""
    m = ModelSpec("tiny", d=64, heads=4, layers=2, ffn=256, vocab=512, rotary_pct=0.25, rotary_base=10000, eps=1e-5)
    return [RunSpec(m, s, p, 256) for s in (64, 128) for p in ("fp32", "fp16")]


DRY_PROTOCOL = {"warmup_iters": 1, "passes": 2, "min_window_s": 0.01, "idle_window_s": 0.01}


def profile(resume: bool, dry_run: bool = False, out_dir: Path | None = None, max_configs: int | None = None) -> Path:
    p = load_yaml("protocol.yaml") | {"l2_bytes": load_yaml("hardware/t4.yaml")["l2_bytes"]}
    if dry_run:
        p |= DRY_PROTOCOL
    out_dir = out_dir or DATA / "raw"
    runs = (tiny_runs() if dry_run else load_grid())[:max_configs]
    Profiler(FakeGpu() if dry_run else Gpu(), p, out_dir, resume).run(runs)
    return out_dir
