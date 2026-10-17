import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="anatomy")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("analytical", help="write data/derived/analytical.csv")
    sub.add_parser("report", help="write T2 and F6 (needs analytical.csv)")
    sub.add_parser("analyze", help="shares, C4, decision rule: T4-T7 (needs passes.csv, analytical.csv)")
    prof = sub.add_parser("profile", help="measure the grid on the GPU, appending to data/raw/passes.csv")
    prof.add_argument("--resume", action="store_true", help="skip (config, unit, pass) rows already recorded")
    prof.add_argument("--dry-run", action="store_true", help="tiny dims on CPU with a fake monitor (no real energy)")
    prof.add_argument("--out", type=Path, help="output dir (default data/raw; dry run: a temp dir)")
    prof.add_argument("--max-configs", type=int, help="only the first N grid configs (smoke test)")
    args = p.parse_args()

    if args.cmd == "analytical":
        from anatomy.analytical import write

        print(f"{len(write())} rows -> data/derived/analytical.csv")
    elif args.cmd == "report":
        from anatomy import report

        print(f"T2: {len(report.t2())} configs")
        report.f6()
        print("F6 -> data/figures/F6.{png,svg}")
    elif args.cmd == "analyze":
        from anatomy.analysis import run

        res = run()
        from anatomy import report

        report.f5()
        report.f7()
        report.f10()
        if res["t8"]:
            report.f8()
        report.f9()
        for r in res["t6"]:
            print(f"{r['component']:13s} {r['both']}/{r['memory_bound']} memory-bound configs -> {r['verdict']}")
    elif args.cmd == "profile":
        import tempfile

        from anatomy.harness import profile

        out = args.out or (Path(tempfile.mkdtemp(prefix="anatomy-dry-")) if args.dry_run else None)
        print(f"passes -> {profile(args.resume, args.dry_run, out, args.max_configs)}")


if __name__ == "__main__":
    main()
