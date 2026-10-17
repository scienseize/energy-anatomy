"""Dashboard export (DESIGN section 9, Phase 7): JSON copied straight from data/derived/ and the
tables, so the dashboard computes nothing. Writes data/export/{configs,profile,ranking,meta}.json."""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

from anatomy.config import DATA, load_yaml
from anatomy.harness import _git_sha

DERIVED, TABLES, EXPORT = DATA / "derived", DATA / "tables", DATA / "export"
PROFILE_FIELDS = ("c1", "c2", "memory_bound", "c3_med", "c3_iqr", "c3_dyn_med", "energy_per_flop_j", "power_w_med")
RANKING_FIELDS = ("c4_med", "c4_iqr", "c4_dyn_med", "rank_c4")


def _value(v: str):
    """CSV text -> JSON value: '' -> null, True/False -> bool, numbers -> int/float, else string."""
    if v in ("", "None"):
        return None
    if v in ("True", "False"):
        return v == "True"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _read(path) -> list[dict]:
    with open(path) as f:
        return [{k: _value(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _config_id(r: dict) -> str:
    return f"{r['model']}/{r['seq_len']}/{r['precision']}"


def build() -> dict:
    summary = _read(DERIVED / "shares_summary.csv")
    by_cfg = defaultdict(dict)
    for r in summary:
        by_cfg[_config_id(r)][r["component"]] = r

    configs = [{"id": _config_id(r), **r} for r in _read(TABLES / "T2.csv")]
    profile = [{"config": cid, "components": {c: {k: r[k] for k in PROFILE_FIELDS} for c, r in comps.items()
                                              if c != "unattributed"},
                "unattributed": {k: comps["unattributed"][k] for k in ("c3_med", "c3_iqr")}}
               for cid, comps in by_cfg.items()]
    capability = _read(DERIVED / "capability.csv") if (DERIVED / "capability.csv").exists() else []
    ranking = {
        "per_config": [{"config": cid, "components": {c: {k: r[k] for k in RANKING_FIELDS} for c, r in comps.items()
                                                      if c != "unattributed"}}
                       for cid, comps in by_cfg.items()],
        "decision_rule": _read(TABLES / "T6.csv"),
        "capability": capability,
        "rank_shifts": _read(DERIVED / "rank_shifts.csv"),
    }
    with open(DATA / "raw" / "passes.csv") as f:
        rows = list(csv.DictReader(f))
    first = rows[0]
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_git_sha": _git_sha(),
        "gpu": load_yaml("hardware/t4.yaml")["name"],
        "measurement": {"run_ids": sorted({r["run_id"] for r in rows}), "gpu_uuids": sorted({r["gpu_uuid"] for r in rows}),
                        "git_sha": sorted({r["git_sha"] for r in rows}), "torch": first["torch"], "cuda": first["cuda"],
                        "driver": first["driver"], "first_pass": min(r["ts"] for r in rows),
                        "last_pass": max(r["ts"] for r in rows), "passes": len(rows)},
        "notes": "Shares and waste from the main run in data/raw/; absolute joules are session-specific (docs/methods.md section 17).",
    }
    return {"configs": configs, "profile": profile, "ranking": ranking, "meta": meta}


def write() -> list:
    EXPORT.mkdir(parents=True, exist_ok=True)
    out = []
    for name, obj in build().items():
        path = EXPORT / f"{name}.json"
        path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
        out.append(path)
    return out
