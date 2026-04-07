#!/usr/bin/env python3

import argparse
import configparser
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List


DEFAULT_CONFIGS = [
    "configs/DDR4_8Gb_x8_3200.ini",
    "configs/HBM2_8Gb_x128.ini",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    dramsim_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Run DRAMsim3 for multiple DRAM configs and generate a summary CSV "
            "containing dram_rw_bw / r_cost / w_cost / dram_bg_power for uni_sim."
        )
    )
    parser.add_argument(
        "--dramsim-root",
        type=Path,
        default=dramsim_root,
        help="Path to DRAMsim3 root directory.",
    )
    parser.add_argument(
        "--exe",
        type=str,
        default="dramsim3main.out",
        help="DRAMsim3 executable name or path.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=dramsim_root / "tests" / "example.trace",
        help="Trace file path.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=200000,
        help="Simulation cycles for each DRAM config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=dramsim_root / "runs_by_type",
        help="Output root for per-config runs and summary.csv.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config paths (relative to --dramsim-root or absolute).",
    )
    return parser.parse_args()


def load_ini(cfg_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg.read_file(f)
    return cfg


def infer_dram_type(cfg_path: Path) -> str:
    # e.g., DDR4_8Gb_x8_3200.ini -> DDR4, HBM2_8Gb_x128.ini -> HBM2
    return cfg_path.stem.split("_", 1)[0]


def sum_nested_values(value) -> float:
    if isinstance(value, dict):
        return float(sum(float(v) for v in value.values()))
    if value is None:
        return 0.0
    return float(value)


def parse_dramsim_json(json_path: Path) -> Dict[str, float]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    channels: List[Dict] = list(data.values())

    if not channels:
        raise RuntimeError(f"No channel stats found in {json_path}")

    agg: Dict[str, float] = {}
    agg["channels"] = float(len(channels))
    agg["num_cycles"] = float(max(float(ch.get("num_cycles", 0.0)) for ch in channels))

    scalar_sum_keys = [
        "num_read_cmds",
        "num_write_cmds",
        "read_energy",
        "write_energy",
        "act_energy",
        "ref_energy",
        "refb_energy",
        "total_energy",
        "average_power",
    ]
    for key in scalar_sum_keys:
        agg[key] = float(sum(float(ch.get(key, 0.0)) for ch in channels))

    background_energy = 0.0
    for ch in channels:
        background_energy += sum_nested_values(ch.get("act_stb_energy", {}))
        background_energy += sum_nested_values(ch.get("pre_stb_energy", {}))
        background_energy += sum_nested_values(ch.get("sref_energy", {}))
    agg["background_energy"] = background_energy

    return agg


def resolve_path(root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()

    dramsim_root = args.dramsim_root.resolve()
    exe_path = resolve_path(dramsim_root, args.exe).resolve()
    trace_path = resolve_path(dramsim_root, str(args.trace)).resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not exe_path.exists():
        raise FileNotFoundError(
            f"Cannot find executable: {exe_path}. Build first with `make -j -C {dramsim_root}`"
        )
    if not trace_path.exists():
        raise FileNotFoundError(f"Cannot find trace file: {trace_path}")

    rows = []

    for cfg_arg in args.configs:
        cfg_path = resolve_path(dramsim_root, cfg_arg).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Cannot find config: {cfg_path}")

        ini = load_ini(cfg_path)
        protocol = ini.get("dram_structure", "protocol", fallback="UNKNOWN")
        bus_width = float(ini.get("system", "bus_width"))
        burst_length = float(ini.get("dram_structure", "BL"))

        dram_type = infer_dram_type(cfg_path)
        run_dir = output_root / dram_type / cfg_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(exe_path),
            str(cfg_path),
            "-c",
            str(args.cycles),
            "-t",
            str(trace_path),
            "-o",
            str(run_dir),
        ]
        subprocess.run(cmd, cwd=str(dramsim_root), check=True)

        stats = parse_dramsim_json(run_dir / "dramsim3.json")

        dram_rw_bw = bus_width * burst_length
        num_read_cmds = stats["num_read_cmds"]
        num_write_cmds = stats["num_write_cmds"]

        r_cost = stats["read_energy"] / num_read_cmds if num_read_cmds else 0.0
        w_cost = stats["write_energy"] / num_write_cmds if num_write_cmds else 0.0

        num_cycles = stats["num_cycles"]
        dram_bg_power = stats["background_energy"] / num_cycles if num_cycles else 0.0

        current_wr_cost = dram_rw_bw / 64.0 * 1200.0
        r_ratio = (current_wr_cost / r_cost) if r_cost else 0.0
        w_ratio = (current_wr_cost / w_cost) if w_cost else 0.0

        rows.append(
            {
                "config": cfg_path.name,
                "protocol": protocol,
                "channels": int(stats["channels"]),
                "dram_rw_bw": dram_rw_bw,
                "r_cost": r_cost,
                "w_cost": w_cost,
                "dram_bg_power": dram_bg_power,
                "current_wr_cost": current_wr_cost,
                "r_ratio_current_over_new": r_ratio,
                "w_ratio_current_over_new": w_ratio,
                "num_read_cmds": int(num_read_cmds),
                "num_write_cmds": int(num_write_cmds),
                "total_energy_pJ": stats["total_energy"],
                "average_power_mW": stats["average_power"],
            }
        )

    summary_path = output_root / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary: {summary_path}")
    for row in rows:
        print(
            f"{row['config']}: bw={row['dram_rw_bw']:.0f}, r_cost={row['r_cost']:.1f}, "
            f"w_cost={row['w_cost']:.1f}, bg={row['dram_bg_power']:.1f}, "
            f"current={row['current_wr_cost']:.1f}, "
            f"ratio(current/r)={row['r_ratio_current_over_new']:.2f}x, "
            f"ratio(current/w)={row['w_ratio_current_over_new']:.2f}x"
        )


if __name__ == "__main__":
    main()
