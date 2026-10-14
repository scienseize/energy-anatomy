import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="anatomy")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("analytical", help="write data/derived/analytical.csv")
    sub.add_parser("report", help="write T2 and F6 (needs analytical.csv)")
    args = p.parse_args()

    if args.cmd == "analytical":
        from anatomy.analytical import write

        print(f"{len(write())} rows -> data/derived/analytical.csv")
    elif args.cmd == "report":
        from anatomy import report

        print(f"T2: {len(report.t2())} configs")
        report.f6()
        print("F6 -> data/figures/F6.{png,svg}")


if __name__ == "__main__":
    main()
