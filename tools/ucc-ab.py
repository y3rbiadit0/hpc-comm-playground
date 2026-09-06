#!/usr/bin/env python3
"""Join two benchscribe CSVs (UCC on vs off) cell by cell.

Benchscribe compares backends *within* one results tree against cuda_mpi; it has
no notion of two trees, so the A/B join lives here.

  python3 tools/benchscribe results-ucc-on  --format csv > ucc-on.csv
  python3 tools/benchscribe results-ucc-off --format csv > ucc-off.csv
  python3 ucc-ab.py ucc-on.csv ucc-off.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

KEY = ("benchmark", "case", "topology", "backend", "n")


def load(path):
    rows = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows[tuple(row[k] for k in KEY)] = row
    return rows


def number(row, field):
    value = row.get(field, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("on_csv", help="benchscribe CSV from the UCC-on tree")
    parser.add_argument("off_csv", help="benchscribe CSV from the UCC-off tree")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="percent change below which a cell counts as no effect")
    args = parser.parse_args(argv)

    on, off = load(args.on_csv), load(args.off_csv)

    only_on = sorted(set(on) - set(off))
    only_off = sorted(set(off) - set(on))
    for key in only_on:
        print(f"warning: only in on-arm: {'/'.join(key)}", file=sys.stderr)
    for key in only_off:
        print(f"warning: only in off-arm: {'/'.join(key)}", file=sys.stderr)

    records = []
    for key in sorted(set(on) & set(off), key=lambda k: (k[0], k[1], k[2], k[3], int(k[4]))):
        on_row, off_row = on[key], off[key]
        on_value, off_value = number(on_row, "value_mean"), number(off_row, "value_mean")
        valid = on_row["valid"] == "PASS" and off_row["valid"] == "PASS"
        if on_value is None or off_value is None or on_value == 0:
            ratio = change = None
        else:
            # Latency: lower is better, so off/on > 1 means UCC is the faster arm.
            ratio = off_value / on_value
            change = (on_value - off_value) / off_value * 100.0
        records.append((key, on_row, on_value, off_value, ratio, change, valid))

    if args.format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow([*KEY, "unit", "ucc_on", "ucc_off", "ratio_off_over_on",
                         "pct_change_from_ucc", "valid"])
        for key, on_row, on_value, off_value, ratio, change, valid in records:
            writer.writerow([*key, on_row["unit"],
                             "" if on_value is None else f"{on_value:.9g}",
                             "" if off_value is None else f"{off_value:.9g}",
                             "" if ratio is None else f"{ratio:.4f}",
                             "" if change is None else f"{change:.2f}",
                             "PASS" if valid else "FAIL"])
        return 0

    print("# UCC A/B\n")
    print("`ratio` = off / on for a latency metric, so **above 1.00 means UCC is faster**. "
          f"`change` is the percent shift attributable to UCC. Cells within "
          f"±{args.threshold:.0f}% are marked `~`.\n")
    current = None
    for key, on_row, on_value, off_value, ratio, change, valid in records:
        benchmark, case, topology, backend, n = key
        heading = (benchmark, case, topology)
        if heading != current:
            current = heading
            label = f"{benchmark} / {topology}" + (f" / {case}" if case else "")
            print(f"\n## {label}\n")
            print(f"| n | backend | on ({on_row['unit']}) | off ({on_row['unit']}) "
                  "| ratio | change | |")
            print("| ---: | --- | ---: | ---: | ---: | ---: | :-: |")
        if ratio is None:
            verdict = "?"
        elif change is None or abs(change) < args.threshold:
            verdict = "~"
        elif change < 0:
            verdict = "UCC"
        else:
            verdict = "no-UCC"
        if not valid:
            verdict += " (FAIL)"
        print(f"| {n} | `{backend}` "
              f"| {'-' if on_value is None else f'{on_value:.4g}'} "
              f"| {'-' if off_value is None else f'{off_value:.4g}'} "
              f"| {'-' if ratio is None else f'{ratio:.2f}x'} "
              f"| {'-' if change is None else f'{change:+.1f}%'} | {verdict} |")

    graded = [r for r in records if r[5] is not None and r[6]]
    if graded:
        wins = sum(1 for r in graded if r[5] < -args.threshold)
        losses = sum(1 for r in graded if r[5] > args.threshold)
        print(f"\n---\n\n{len(graded)} valid cells: UCC faster in {wins}, slower in {losses}, "
              f"within ±{args.threshold:.0f}% in {len(graded) - wins - losses}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
