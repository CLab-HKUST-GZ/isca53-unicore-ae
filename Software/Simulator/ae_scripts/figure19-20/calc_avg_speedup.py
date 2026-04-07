#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import geometric_mean


PRECISIONS = ["w4a4", "w4a8", "w8a8"]
MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-8B",
]
METHODS = ["ant", "olive", "tender", "mant", "unicore"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute average decode speedup of a target method relative to other "
            "methods from uni_sim CSV result folders."
        )
    )
    parser.add_argument(
        "--results-dir",
        dest="results_dirs",
        action="append",
        required=True,
        help="Decode result directory. Repeat this option to include multiple folders.",
    )
    parser.add_argument(
        "--cxt-len",
        type=int,
        default=8192,
        help="Context length to use. Default: 8192",
    )
    parser.add_argument(
        "--target",
        default="unicore",
        choices=METHODS,
        help="Target method in the numerator denominator relation other/target. Default: unicore",
    )
    parser.add_argument(
        "--others",
        nargs="+",
        default=["ant", "olive", "tender", "mant"],
        choices=METHODS,
        help="Methods to compare against the target. Default: ant olive tender mant",
    )
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print per-case speedups before the average.",
    )
    return parser.parse_args()


def load_cycles(results_dir, cxt_len):
    data = {}
    results_dir = Path(results_dir)
    for csv_path in sorted(results_dir.glob("*.csv")):
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["cxt_len"]) != cxt_len:
                    continue
                if int(row.get("is_generation", 0)) != 1:
                    continue

                config_name = row["config_name"]
                prefix, precision = config_name.rsplit("_", 1)
                model_name = row["model_name"]
                data[(prefix, precision, model_name)] = float(row["total_cycle"])
    return data


def collect_speedups(data, target, other_methods):
    values = {method: [] for method in other_methods}
    cases = {method: [] for method in other_methods}

    for precision in PRECISIONS:
        for model_name in MODELS:
            target_key = (target, precision, model_name)
            if target_key not in data:
                raise ValueError(f"Missing target entry: {target_key}")
            target_cycle = data[target_key]

            for method in other_methods:
                other_key = (method, precision, model_name)
                if other_key not in data:
                    raise ValueError(f"Missing comparison entry: {other_key}")
                speedup = data[other_key] / target_cycle
                values[method].append(speedup)
                cases[method].append((precision, model_name, speedup))

    return values, cases


def print_report(title, values, cases, show_cases):
    print(title)
    for method, speedups in values.items():
        arith = sum(speedups) / len(speedups)
        geo = geometric_mean(speedups)
        print(
            f"  {method:>6s}: arithmetic_mean={arith:.4f}x, "
            f"geometric_mean={geo:.4f}x, count={len(speedups)}"
        )
        if show_cases:
            for precision, model_name, speedup in cases[method]:
                print(
                    f"           - {precision:>4s}, {model_name}: {speedup:.4f}x"
                )


def main():
    args = parse_args()

    combined_values = {method: [] for method in args.others}
    combined_cases = {method: [] for method in args.others}

    for results_dir in args.results_dirs:
        data = load_cycles(results_dir, args.cxt_len)
        values, cases = collect_speedups(data, args.target, args.others)
        print_report(
            title=f"[results_dir={results_dir}] cxt_len={args.cxt_len}",
            values=values,
            cases=cases,
            show_cases=args.show_cases,
        )
        print()

        for method in args.others:
            combined_values[method].extend(values[method])
            combined_cases[method].extend(cases[method])

    if len(args.results_dirs) > 1:
        print_report(
            title=f"[combined] cxt_len={args.cxt_len}",
            values=combined_values,
            cases=combined_cases,
            show_cases=args.show_cases,
        )


if __name__ == "__main__":
    main()
