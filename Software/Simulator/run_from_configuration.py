import os
import csv
import argparse

from accelerator_os2d import AcceleratorOS2D
from accelerator_ws import AcceleratorWS
import configuration as cfg


RESULTS_DIR = "results"

# 映射 configuration 中的硬件配置到数据流类型
# OS: 使用 AcceleratorOS2D
# WS: 使用 AcceleratorWS
CONFIGS = {
    "unicore_w4a4": ("ws", cfg.unicore_w4a4),
    "unicore_w4a8": ("ws", cfg.unicore_w4a8),
    "unicore_w8a8": ("ws", cfg.unicore_w8a8),
    "ant_w4a4": ("os", cfg.ant_w4a4),
    "ant_w4a8": ("os", cfg.ant_w4a8),
    "ant_w8a8": ("os", cfg.ant_w8a8),
    "olive_w4a4": ("os", cfg.olive_w4a4),
    "olive_w4a8": ("os", cfg.olive_w4a8),
    "olive_w8a8": ("os", cfg.olive_w8a8),
    "tender_w4a4": ("os", cfg.tender_w4a4),
    "tender_w4a8": ("os", cfg.tender_w4a8),
    "tender_w8a8": ("os", cfg.tender_w8a8),
    "mant_w4a4": ("ws", cfg.mant_w4a4),
    "mant_w4a8": ("ws", cfg.mant_w4a8),
    "mant_w8a8": ("ws", cfg.mant_w8a8),
}

PREFILL_ONLY_CONFIGS = {
    "unicore_w16a16": ("ws", cfg.unicore_w16a16),
    "tender_w16a16": ("os", cfg.tender_w16a16),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run uni_sim configs and export CSV results with optional DRAM parameter overrides."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Output directory for result CSV files.",
    )
    parser.add_argument(
        "--dram-rw-bw",
        type=float,
        default=None,
        help="Override DRAM per-channel rw bandwidth in bits/transfer.",
    )
    parser.add_argument(
        "--dram-channels",
        type=int,
        default=None,
        help=(
            "Override DRAM channel count. Effective DRAM interface scales as "
            "dram_rw_bw * dram_channels."
        ),
    )
    parser.add_argument(
        "--dram-r-cost",
        type=float,
        default=None,
        help="Override DRAM read energy cost (pJ per transfer).",
    )
    parser.add_argument(
        "--dram-w-cost",
        type=float,
        default=None,
        help="Override DRAM write energy cost (pJ per transfer).",
    )
    parser.add_argument(
        "--dram-bg-power",
        type=float,
        default=None,
        help="Override DRAM background energy slope (pJ per cycle).",
    )
    parser.add_argument(
        "--is-generation",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Override generation/decode mode: 0 for prefill, 1 for decode. "
            "If omitted, use configuration.py is_generation value."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override batch size. If omitted, use configuration.py batch_size value."
        ),
    )
    parser.add_argument(
        "--fused-attn",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Enable a prefill-only fused-attention DRAM approximation that "
            "keeps the score matrix off DRAM between attn_qk and attn_v."
        ),
    )
    return parser.parse_args()


def ensure_results_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)


def build_hw_cfg_with_overrides(hw_cfg, args):
    merged = dict(hw_cfg)

    if args.dram_rw_bw is not None:
        merged["dram_rw_bw"] = args.dram_rw_bw
    if args.dram_channels is not None:
        merged["dram_channels"] = args.dram_channels
    if args.dram_r_cost is not None:
        merged["dram_r_cost"] = args.dram_r_cost
    if args.dram_w_cost is not None:
        merged["dram_w_cost"] = args.dram_w_cost
    if args.dram_bg_power is not None:
        merged["dram_bg_power"] = args.dram_bg_power
    if args.fused_attn is not None:
        merged["fused_attn"] = bool(args.fused_attn)

    return merged


def run_single_case(acc):
    """运行一次仿真并返回全局统计信息。能量统一转为 uJ。"""
    total_compute_cycle, total_cycle = acc.calc_cycle()
    # Linear vs attention breakdown (cycles)
    linear_compute_cycle = getattr(acc, "total_cycle_compute_linear", 0)
    linear_dram_cycle = getattr(acc, "total_cycle_dram_linear", 0)
    attn_compute_cycle = getattr(acc, "total_cycle_compute_attn", 0)
    attn_dram_cycle = getattr(acc, "total_cycle_dram_attn", 0)

    compute_energy = acc.calc_compute_energy() / 1e6
    sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
    sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
    dram_energy = acc.calc_dram_energy() / 1e6
    onchip_energy = compute_energy + sram_rd_energy + sram_wr_energy
    total_energy = onchip_energy + dram_energy

    return {
        "total_compute_cycle": total_compute_cycle,
        "total_cycle": total_cycle,
        "linear_compute_cycle": linear_compute_cycle,
        "linear_dram_cycle": linear_dram_cycle,
        "attn_compute_cycle": attn_compute_cycle,
        "attn_dram_cycle": attn_dram_cycle,
        "compute_uJ": compute_energy,
        "sram_rd_uJ": sram_rd_energy,
        "sram_wr_uJ": sram_wr_energy,
        "sram_uJ": sram_rd_energy + sram_wr_energy,
        "dram_uJ": dram_energy,
        "onchip_uJ": onchip_energy,
        "total_uJ": total_energy,
    }


def run_config(
    config_name,
    dataflow,
    hw_cfg,
    results_dir,
    is_generation_override=None,
    batch_size_override=None,
):
    """对一个硬件配置跑 model_list × cxt_len 全部组合，并写入一个 CSV。"""
    if batch_size_override is None:
        batch_size = cfg.batch_size
    else:
        batch_size = int(batch_size_override)

    if is_generation_override is None:
        is_generation = cfg.is_generation
    else:
        is_generation = bool(is_generation_override)
    effective_fused_attn = bool(hw_cfg.get("fused_attn", False)) and not is_generation
    model_list = cfg.model_list
    cxt_len_list = cfg.cxt_len

    csv_path = os.path.join(results_dir, f"{config_name}_{dataflow}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # 统一列格式
        writer.writerow([
            "config_name",
            "dataflow",  # OS / WS
            "model_name",
            "cxt_len",
            "batch_size",
            "is_generation",
            "fused_attn",
            "total_compute_cycle",
            "total_cycle",
            "linear_compute_cycle",
            "linear_dram_cycle",
            "attn_compute_cycle",
            "attn_dram_cycle",
            "compute_uJ",
            "sram_uJ",
            "dram_uJ",
            "onchip_uJ",
            "total_uJ",
        ])

        for model_name in model_list:
            for cur_cxt_len in cxt_len_list:
                print(
                    f"Running config={config_name}, dataflow={dataflow}, "
                    f"model={model_name}, cxt_len={cur_cxt_len}, "
                    f"batch_size={batch_size}, "
                    f"is_generation={int(is_generation)}"
                )

                if dataflow == "os":
                    AccelCls = AcceleratorOS2D
                else:
                    AccelCls = AcceleratorWS

                acc = AccelCls(
                    model_name=model_name,
                    i_prec=hw_cfg["i_prec"],
                    kv_prec=hw_cfg["kv_prec"],
                    w_prec=hw_cfg["w_prec"],
                    batch_size=batch_size,
                    is_bit_serial=hw_cfg["is_bit_serial"],
                    pe_dp_size=hw_cfg["pe_dp_size"],
                    pe_energy=hw_cfg["pe_energy"],
                    pe_area=hw_cfg["pe_area"],
                    pe_array_dim=hw_cfg["pe_array_dim"],
                    cxt_len=cur_cxt_len,
                    is_generation=is_generation,
                    fused_attn=effective_fused_attn,
                    dram_rw_bw=hw_cfg.get("dram_rw_bw", 512.0),
                    dram_channels=hw_cfg.get("dram_channels", 1),
                    dram_r_cost=hw_cfg.get("dram_r_cost", None),
                    dram_w_cost=hw_cfg.get("dram_w_cost", None),
                    dram_bg_power=hw_cfg.get("dram_bg_power", 0.0),
                )

                stats = run_single_case(acc)

                writer.writerow([
                    config_name,
                    dataflow,
                    model_name,
                    cur_cxt_len,
                    batch_size,
                    int(is_generation),
                    int(effective_fused_attn),
                    stats["total_compute_cycle"],
                    stats["total_cycle"],
                    stats["linear_compute_cycle"],
                    stats["linear_dram_cycle"],
                    stats["attn_compute_cycle"],
                    stats["attn_dram_cycle"],
                    stats["compute_uJ"],
                    stats["sram_uJ"],
                    stats["dram_uJ"],
                    stats["onchip_uJ"],
                    stats["total_uJ"],
                ])


def main():
    args = parse_args()
    ensure_results_dir(args.results_dir)

    if args.is_generation is None:
        effective_is_generation = bool(cfg.is_generation)
    else:
        effective_is_generation = bool(args.is_generation)

    configs_to_run = dict(CONFIGS)
    if not effective_is_generation:
        configs_to_run.update(PREFILL_ONLY_CONFIGS)

    for config_name, (dataflow, hw_cfg) in configs_to_run.items():
        effective_hw_cfg = build_hw_cfg_with_overrides(hw_cfg, args)
        run_config(
            config_name,
            dataflow,
            effective_hw_cfg,
            args.results_dir,
            is_generation_override=args.is_generation,
            batch_size_override=args.batch_size,
        )


if __name__ == "__main__":
    main()
