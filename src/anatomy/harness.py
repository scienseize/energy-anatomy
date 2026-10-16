"""Profiling harness (DESIGN §6): per (config, unit) build once, warm up, calibrate R, then
measure N passes in energy windows, appending each pass to passes.csv as soon as it ends."""
import csv
import gc
import math
import os
import subprocess
import threading
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
          "mem_clock_mhz", "temp_c", "power_limit_w", "throttled", "throttle_reasons", "status", "torch",
          "cuda", "driver", "git_sha"]
IDLE_FIELDS = ["run_id", "ts", "energy_j", "time_s", "avg_power_w", "sm_clock_mhz", "mem_clock_mhz",
               "temp_c", "power_limit_w", "throttle_reasons"]
PROBE_S = 0.2  # calibration probe length
SAMPLE_S = 0.1  # NVML sampling period inside a window
# NVML clocks-event (throttle) reason bits, stable across driver versions
REASONS = {0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
           0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clock"}


def _reasons(mask: int) -> str:
    return "|".join(name for bit, name in REASONS.items() if mask & bit) or "none"


class Gpu:
    """Zeus energy windows + NVML context on CUDA device 0. Clocks, temperature and throttle
    reasons are sampled every SAMPLE_S during each window, not read once afterwards."""

    device = "cuda"

    def __init__(self):
        import pynvml
        from zeus.monitor import ZeusMonitor

        self.nvml = n = pynvml
        n.nvmlInit()
        self.h = n.nvmlDeviceGetHandleByIndex(0)
        bus = getattr(torch.cuda.get_device_properties(0), "pci_bus_id", None)
        assert bus is None or n.nvmlDeviceGetPciInfo(self.h).bus == bus, "NVML device 0 is not CUDA device 0"
        self.monitor = ZeusMonitor(gpu_indices=[0])
        self.driver = n.nvmlSystemGetDriverVersion()
        self.max_sm_mhz = n.nvmlDeviceGetMaxClockInfo(self.h, n.NVML_CLOCK_SM)
        self._get_reasons = getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            n.nvmlDeviceGetCurrentClocksThrottleReasons

    def sync(self):
        torch.cuda.synchronize()

    def _temp(self) -> int:
        return self.nvml.nvmlDeviceGetTemperature(self.h, self.nvml.NVML_TEMPERATURE_GPU)

    def _sample(self) -> tuple:
        n, h = self.nvml, self.h
        return (n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM), n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
                self._temp(), self._get_reasons(h))

    def window(self, fn) -> tuple[float, float, dict]:
        samples, stop = [], threading.Event()

        def sampler():
            while not stop.is_set():
                samples.append(self._sample())
                stop.wait(SAMPLE_S)

        th = threading.Thread(target=sampler, daemon=True)
        self.monitor.begin_window("w")  # syncs CUDA before reading the counter
        th.start()
        fn()
        m = self.monitor.end_window("w")
        stop.set()
        th.join()
        sm, mem, temp, mask = zip(*samples)
        mask_or = 0
        for x in mask:
            mask_or |= x
        ctx = {"sm_clock_mhz": sum(sm) / len(sm), "mem_clock_mhz": sum(mem) / len(mem), "temp_c": max(temp),
               "power_limit_w": self.nvml.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000,
               "throttle_reasons": _reasons(mask_or)}
        return sum(m.gpu_energy.values()), m.time, ctx

    def precondition(self, max_s: float, step_s: float, tol_c: float) -> tuple[int, float]:
        """Run matmuls until the temperature changes by < tol_c over step_s (or max_s), so every
        measured unit, including the first of a session, sees the hot steady state."""
        x = torch.randn(4096, 4096, device="cuda")
        t0, last = time.monotonic(), self._temp()
        while time.monotonic() - t0 < max_s:
            end = time.monotonic() + step_s
            while time.monotonic() < end:
                for _ in range(10):
                    x = x @ x
                    x = x / x.norm()
                torch.cuda.synchronize()
            now = self._temp()
            if abs(now - last) < tol_c:
                break
            last = now
        del x
        torch.cuda.empty_cache()
        return self._temp(), time.monotonic() - t0


class FakeGpu:
    """CPU stand-in for dry runs: energy = wall time x a constant power. Never real numbers."""

    device = "cpu"
    driver = "fake"
    POWER_W = 50.0
    max_sm_mhz = 1590

    def sync(self):
        pass

    def window(self, fn):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        ctx = {"sm_clock_mhz": 1590, "mem_clock_mhz": 5000, "temp_c": 40, "power_limit_w": 70.0,
               "throttle_reasons": "none"}
        return dt * self.POWER_W, dt, ctx

    def precondition(self, max_s, step_s, tol_c):
        return 40, 0.0


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
        if resume and self.passes_csv.exists():
            with open(self.passes_csv) as f:
                for r in csv.DictReader(f):
                    k = _key(r["model"], r["seq_len"], r["precision"], r["unit"])
                    self.done.setdefault(k, set()).add(int(r["pass_idx"]))
        self.last_idle = -math.inf

    def idle_if_due(self) -> None:
        if time.monotonic() - self.last_idle < self.p["idle_every_min"] * 60:
            return
        if self.last_idle == -math.inf:  # session start: reach thermal steady state first
            temp, secs = self.gpu.precondition(self.p["precondition_max_s"], self.p["precondition_step_s"],
                                               self.p["precondition_tol_c"])
            print(f"preconditioned: {temp} C after {secs:.0f} s", flush=True)
        self.gpu.sync()
        e, t, ctx = self.gpu.window(lambda: time.sleep(self.p["idle_window_s"]))
        _append(self.idle_csv, IDLE_FIELDS, {"run_id": self.run_id, "ts": _now(), "energy_j": e,
                                             "time_s": t, "avg_power_w": e / t, **ctx})
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
                e, t, ctx = self.gpu.window(lambda: self._reps(mod, sets, reps))
                # Against the GPU's max SM clock: pass 0 is already power-capped on the T4 (smoke test).
                throttled = ctx["sm_clock_mhz"] < (1 - self.p["throttle_drop"]) * self.gpu.max_sm_mhz
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
