import math
from typing import List, Optional

from accelerator import Accelerator


class AcceleratorOS2D(Accelerator):
    """OS accelerator with 2D systolic compute-cycle approximation.

    This class keeps the original OS memory / refetch modelling in
    `Accelerator`, but replaces `_calc_compute_cycle` with a 2D
    systolic-style pipeline estimate similar to `AcceleratorWS`.

    The PE array is interpreted as:
      - pe_array_dim[0] (R): PE rows, mapped to output-channel (N) tile.
      - pe_array_dim[1] (C): PE cols, mapped to input-channel (K) tile.
      - M dimension (batch_size * num_token) is temporal.
    """

    def __init__(
        self,
        model_name: str,
        i_prec: int = 16,
        kv_prec: int = 8,
        w_prec: int = 8,
        batch_size: int = 1,
        is_bit_serial: bool = False,
        pe_dp_size: int = 1,
        pe_energy: float = 0,
        pe_area: float = 0,
        pe_array_dim: List[int] = [],
        init_mem: bool = True,
        cxt_len: int = 256,
        is_generation: bool = False,
        fused_attn: bool = False,
        dram_rw_bw: float = 512.0,
        dram_r_cost: Optional[float] = None,
        dram_w_cost: Optional[float] = None,
        dram_bg_power: float = 0.0,
        dram_channels: int = 1,
    ) -> None:
        # Force OS-style memory behaviour by keeping dataflow="os".
        super().__init__(
            model_name=model_name,
            i_prec=i_prec,
            kv_prec=kv_prec,
            w_prec=w_prec,
            batch_size=batch_size,
            is_bit_serial=is_bit_serial,
            pe_dp_size=pe_dp_size,
            pe_energy=pe_energy,
            pe_area=pe_area,
            pe_array_dim=pe_array_dim,
            dataflow="os",
            init_mem=init_mem,
            cxt_len=cxt_len,
            is_generation=is_generation,
            fused_attn=fused_attn,
            dram_rw_bw=dram_rw_bw,
            dram_r_cost=dram_r_cost,
            dram_w_cost=dram_w_cost,
            dram_bg_power=dram_bg_power,
            dram_channels=dram_channels,
        )

    def _calc_compute_cycle(self):
        """2D systolic compute-cycle approximation for OS.

        Same GEMM abstraction as WS:
          - M = batch_size * num_token
          - K = cin
          - N = cout

        For each weight tile of size K_t x N_t mapped to an R x C array,
        we use the classical estimate:

            cycle_per_tile ~= M + K_t + N_t - 2
        """
        self._layer_cycle_compute = {}
        self._layer_tile_stats_2d = {}

        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is None:
                continue
            R, C = self._effective_pe_array_dim(name)

            is_attn = ("attn_qk" in name) or ("attn_v" in name)
            if is_attn:
                pe_latency = self.pe_latency["attn"]
            else:
                pe_latency = self.pe_latency["linear"]

            batch_kv, cout, cin = w_dim
            batch_size, num_token, _ = o_dim

            # For attention layers, o_dim[0] already folds (batch * num_kv_heads),
            # so M should be per (batch, kv_head) group to avoid double counting.
            M = num_token if is_attn else (batch_size * num_token)
            K = cin
            N = cout

            tile_cout = math.ceil(N / R)
            tile_cin = math.ceil(K / C)

            num_weight_tiles = batch_kv * tile_cout * tile_cin

            K_t = min(C, K)
            N_t = min(R, N)

            cycle_per_tile = max(0, M + K_t + N_t - 2)
            cycle_layer_compute = num_weight_tiles * cycle_per_tile * pe_latency

            self._layer_cycle_compute[name] = int(cycle_layer_compute)
            self._layer_tile_stats_2d[name] = {
                "M": M,
                "K": K,
                "N": N,
                "tile_cin": tile_cin,
                "tile_cout": tile_cout,
                "num_weight_tiles": num_weight_tiles,
                "K_t": K_t,
                "N_t": N_t,
                "cycle_per_tile": cycle_per_tile,
                "pe_latency": pe_latency,
            }
