import math
from typing import List, Optional

import numpy as np

from mem.mem_instance import MemoryInstance
from pe_array import PE_Array


class AcceleratorWS(PE_Array):
    """Systolic weight-stationary accelerator model.

    This class reuses PE_Array's model-shape profiling, but remaps the PE
    array dimensions to a 2D systolic WS architecture:

    - pe_array_dim[0] (R): PE rows, mapped to the output-channel (N) tile.
    - pe_array_dim[1] (C): PE cols, mapped to the input-channel (K) tile.
    - M dimension (batch_size * num_token) is temporal.

    We assume weights are stationary at PEs across K and N tiles, activations
    stream spatially across one array dimension, and partial sums propagate
    along the orthogonal dimension to complete accumulation over K.
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
            cxt_len=cxt_len,
            is_generation=is_generation,
        )

        self.is_generation = bool(is_generation)
        self.fused_attn = bool(fused_attn)
        # Fused attention keeps score/softmax scratch traffic on-chip at a
        # fixed 16-bit precision, while vector-unit compute is still ignored.
        self.attn_score_prec = 16
        self.dram_rw_bw = float(dram_rw_bw)
        self.dram_r_cost = None if dram_r_cost is None else float(dram_r_cost)
        self.dram_w_cost = None if dram_w_cost is None else float(dram_w_cost)
        self.dram_channels = int(dram_channels)
        if self.dram_channels < 1:
            raise ValueError(f"dram_channels must be >= 1, got {self.dram_channels}")
        # DRAM background energy slope in pJ/cycle.
        self.dram_bg_power = float(dram_bg_power)
        self.cycle_compute = None
        self.cycle_total = None
        if init_mem:
            self._init_mem()
            self._check_layer_mem_size()
            self._calc_num_mem_refetch()

    # ------------------------------------------------------------------
    # Cycle modelling
    # ------------------------------------------------------------------
    def calc_cycle(self):
        self._calc_compute_cycle()
        self._calc_dram_cycle()

        total_cycle = 0
        total_cycle_compute = 0
        for name in self.layer_name_list:
            cycle_layer_compute = self._layer_cycle_compute.get(name, 0)
            cycle_layer_dram = self._layer_cycle_dram.get(name, 0)
            print(
                f"layer name: {name}, compute: {cycle_layer_compute}, dram: {cycle_layer_dram}"
            )
            total_cycle_compute += cycle_layer_compute
            total_cycle += max(cycle_layer_compute, cycle_layer_dram)

        self.cycle_compute = total_cycle_compute
        self.cycle_total = total_cycle

        total_cycle_compute_linear = 0
        total_cycle_dram_linear = 0
        total_cycle_compute_attn = 0
        total_cycle_dram_attn = 0

        for name in self.layer_name_list:
            if ("attn_qk" in name) or ("attn_v" in name):
                total_cycle_compute_attn += self._layer_cycle_compute.get(name, 0)
                total_cycle_dram_attn += self._layer_cycle_dram.get(name, 0)
            else:
                total_cycle_compute_linear += self._layer_cycle_compute.get(name, 0)
                total_cycle_dram_linear += self._layer_cycle_dram.get(name, 0)

        # Store per-type breakdown for reuse (e.g., plotting, logging)
        self.total_cycle_compute_linear = total_cycle_compute_linear
        self.total_cycle_dram_linear = total_cycle_dram_linear
        self.total_cycle_compute_attn = total_cycle_compute_attn
        self.total_cycle_dram_attn = total_cycle_dram_attn

        print(
            f"Linear Compute: {total_cycle_compute_linear}, Linear DRAM: {total_cycle_dram_linear}"
        )
        print(
            f"Attn Compute:   {total_cycle_compute_attn}, Attn DRAM:   {total_cycle_dram_attn}"
        )
        print("\n")

        return total_cycle_compute, total_cycle

    def _fused_attn_role(self, layer_name):
        if not self.fused_attn or self.is_generation:
            return None
        if "attn_qk" in layer_name:
            return "producer"
        if "attn_v" in layer_name:
            return "consumer"
        return None

    def _uses_global_input_buffer(self, layer_name):
        # Fused attention bypasses the modeled *global* input SRAM. Local
        # score-tile scratch traffic is accounted separately as on-chip SRAM.
        return self._fused_attn_role(layer_name) != "consumer"

    def _uses_global_output_buffer(self, layer_name):
        # Fused attention bypasses the modeled *global* output SRAM/DRAM
        # hierarchy. QK score tiles are still charged as local on-chip SRAM
        # scratch writes.
        return self._fused_attn_role(layer_name) != "producer"

    def _output_prec(self, layer_name):
        # Simplified consistent KV-cache model: k_proj/v_proj outputs are
        # assumed to be quantized and written back at KV-cache precision.
        if (".self_attn.k_proj" in layer_name) or (".self_attn.v_proj" in layer_name):
            return self.kv_prec
        return self.i_prec

    def _fused_score_tensor_dim(self, layer_name):
        role = self._fused_attn_role(layer_name)
        if role == "producer":
            return self.output_dim[layer_name]
        if role == "consumer":
            return self.input_dim[layer_name]
        return None

    def _calc_fused_score_sram_accesses(self, layer_name, is_write):
        score_dim = self._fused_score_tensor_dim(layer_name)
        if score_dim is None:
            return 0

        batch_size, num_token, hidden = score_dim
        if is_write:
            bw_min = self.i_sram.w_bw_min
        else:
            bw_min = self.i_sram.r_bw_min

        accesses_per_row = math.ceil(hidden * self.attn_score_prec / bw_min)
        return accesses_per_row * num_token * batch_size

    def _calc_fused_score_sram_rd_energy(self, layer_name):
        if self._fused_attn_role(layer_name) != "consumer":
            return 0
        return self._calc_fused_score_sram_accesses(layer_name, is_write=False) * self.i_sram.r_cost_min

    def _calc_fused_score_sram_wr_energy(self, layer_name):
        if self._fused_attn_role(layer_name) != "producer":
            return 0
        return self._calc_fused_score_sram_accesses(layer_name, is_write=True) * self.i_sram.w_cost_min

    def _calc_compute_cycle(self):
        """Compute cycles for each layer under systolic WS dataflow.

        We approximate a 2D systolic array where one weight tile of
        size K_t x N_t is mapped onto an R x C array (R rows, C cols),
        and the M dimension (batch * sequence) is temporal. For each
        weight tile we use the classical pipeline estimate:

            cycle_per_tile ~= M + K_t + N_t - 2
        """
        self._layer_cycle_compute = {}
        self._layer_tile_stats = {}

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
            self._layer_tile_stats[name] = {
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

    def _calc_dram_cycle(self):
        self._layer_cycle_dram = {}
        dram_bandwidth = self.dram.rw_bw * 2  # DDR

        for name in self.layer_name_list:
            i_prec = self.i_prec
            if ("attn_qk" in name) or ("attn_v" in name):
                w_prec = self.kv_prec
            else:
                w_prec = self.w_prec

            num_dram_fetch_w, num_dram_fetch_i = self._layer_mem_refetch[name]

            cycle_dram_load_w = self._w_mem_required[name] * 8 / dram_bandwidth
            cycle_dram_load_w *= num_dram_fetch_w

            cycle_dram_load_i = self._i_mem_required[name] * 8 / dram_bandwidth
            cycle_dram_load_i *= num_dram_fetch_i

            cycle_dram_write_o = self._o_mem_required[name] * 8 / dram_bandwidth
            if not self._uses_global_output_buffer(name):
                cycle_dram_write_o = 0
            if not self._uses_global_input_buffer(name):
                cycle_dram_load_i = 0

            cycle_layer_dram = cycle_dram_load_w + cycle_dram_write_o + cycle_dram_load_i
            self._layer_cycle_dram[name] = int(cycle_layer_dram)

    # ------------------------------------------------------------------
    # Energy modelling
    # ------------------------------------------------------------------
    def calc_compute_energy(self):
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        compute_energy = self.pe_energy * self.total_pe_count * self.cycle_compute
        return compute_energy

    def calc_sram_rd_energy(self):
        """SRAM read energy under WS dataflow.

        We approximate weight SRAM reads proportional to the number of
        weight tiles, and input SRAM reads proportional to the product
        of weight tiles and M (since activations stream for each tile).
        """
        w_sram_rd_cost = self.w_sram.r_cost
        i_sram_rd_cost = self.i_sram.r_cost

        total_w_reads = 0
        total_i_reads = 0

        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is None:
                continue
            R, C = self._effective_pe_array_dim(name)

            is_attn = ("attn_qk" in name) or ("attn_v" in name)
            batch_kv, cout, cin = w_dim
            batch_size, num_token, _ = o_dim
            # Keep SRAM-read activation streaming factor aligned with cycle model.
            M = num_token if is_attn else (batch_size * num_token)
            K = cin
            N = cout

            tile_cout = math.ceil(N / R)
            tile_cin = math.ceil(K / C)
            num_weight_tiles = batch_kv * tile_cout * tile_cin

            total_w_reads += num_weight_tiles
            if self._uses_global_input_buffer(name):
                total_i_reads += num_weight_tiles * M

        sram_rd_energy = total_w_reads * w_sram_rd_cost + total_i_reads * i_sram_rd_cost
        for name in self.layer_name_list:
            sram_rd_energy += self._calc_fused_score_sram_rd_energy(name)
        return sram_rd_energy

    def calc_sram_wr_energy(self):
        total_energy = 0
        for name in self.layer_name_list:
            total_energy += self._calc_sram_wr_energy_fc(name)
        return total_energy

    def _calc_sram_wr_energy_fc(self, layer_name):
        w_dim = self.weight_dim[layer_name]
        i_dim = self.input_dim[layer_name]
        o_dim = self.output_dim[layer_name]

        i_prec = self.i_prec
        o_prec = self._output_prec(layer_name)
        if ("attn_qk" in layer_name) or ("attn_v" in layer_name):
            w_prec = self.kv_prec
        else:
            w_prec = self.w_prec

        w_sram_wr_cost = self.w_sram.w_cost_min
        i_sram_wr_cost = self.i_sram.w_cost_min
        w_sram_min_wr_bw = self.w_sram.w_bw_min
        i_sram_min_wr_bw = self.i_sram.w_bw_min
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        batch_kv, cout_w, cin_w = w_dim
        batch_size_in, num_token_in, cin_i = i_dim
        batch_size_out, num_token_out, cin_o = o_dim

        num_w_sram_wr = math.ceil(cin_w * w_prec / w_sram_min_wr_bw) * cout_w * batch_kv
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w

        num_i_sram_wr = (
            math.ceil(cin_i * i_prec / i_sram_min_wr_bw)
            * num_token_in
            * batch_size_in
        )
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i

        num_o_sram_wr = (
            math.ceil(cin_o * o_prec / i_sram_min_wr_bw)
            * num_token_out
            * batch_size_out
        )
        energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost
        if not self._uses_global_input_buffer(layer_name):
            energy_i_sram_wr = 0
        if not self._uses_global_output_buffer(layer_name):
            energy_o_sram_wr = 0

        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        total_energy += self._calc_fused_score_sram_wr_energy(layer_name)
        return total_energy

    def calc_dram_energy(self):
        energy = 0
        for name in self.layer_name_list:
            energy += self._calc_dram_energy_fc(name)
        if self.dram_bg_power:
            if self.cycle_total is None:
                _, total_cycle = self.calc_cycle()
            else:
                total_cycle = self.cycle_total
            energy += self.dram_bg_power * total_cycle
        return energy

    def _calc_dram_energy_fc(self, layer_name):
        bus_width = self.dram.rw_bw
        rd_cost = self.dram.r_cost
        wr_cost = self.dram.w_cost
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        w_mem_required = self._w_mem_required[layer_name]
        energy_weight = w_mem_required * 8 / bus_width * rd_cost

        i_mem_required = self._i_mem_required[layer_name]
        energy_input = i_mem_required * 8 / bus_width * rd_cost

        o_mem_required = self._o_mem_required[layer_name]
        energy_output = o_mem_required * 8 / bus_width * wr_cost
        if not self._uses_global_output_buffer(layer_name):
            energy_output = 0
        if not self._uses_global_input_buffer(layer_name):
            energy_input = 0

        energy_weight *= num_fetch_w
        energy_input *= num_fetch_i
        total_energy = energy_weight + energy_input + energy_output
        return total_energy

    # ------------------------------------------------------------------
    # Memory size / refetch
    # ------------------------------------------------------------------
    def _check_layer_mem_size(self):
        self._w_mem_required = {}
        self._i_mem_required = {}
        self._o_mem_required = {}

        for _, name in enumerate(self.layer_name_list):
            i_prec = self.i_prec
            o_prec = self._output_prec(name)
            if ("attn_qk" in name) or ("attn_v" in name):
                w_prec = self.kv_prec
            else:
                w_prec = self.w_prec

            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]

            batch_kv, cout_w, cin_w = w_dim
            batch_size_in, num_token_in, cin_i = i_dim
            batch_size_out, num_token_out, cin_o = o_dim

            assert cin_w == cin_i
            assert cout_w == cin_o
            assert num_token_in == num_token_out
            assert batch_size_in == batch_size_out

            self._w_mem_required[name] = math.ceil(cin_w * w_prec / 8) * cout_w * batch_kv
            self._i_mem_required[name] = (
                math.ceil(cin_i * i_prec / 8) * num_token_in * batch_size_in
            )
            self._o_mem_required[name] = (
                math.ceil(cin_o * o_prec / 8) * num_token_out * batch_size_out
            )

    def _calc_num_mem_refetch(self):
        """Hybrid DRAM refetch modelling for WS.

        - If only one tensor overflows SRAM: stream that tensor once from DRAM
          while keeping the other tensor resident in SRAM. No extra DRAM
          *refetch* multiplier is needed because `_w_mem_required`/
          `_i_mem_required` already represent full-layer traffic volumes.
        - If both overflow: fall back to the OS-style heuristic that chooses
          the refetch pattern with minimal total DRAM traffic.
        """
        self._layer_mem_refetch = {}

        size_sram_w = self.w_sram.size / 8
        size_sram_i = self.i_sram.size / 8

        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            if w_dim is None:
                continue

            w_mem_required = self._w_mem_required[name]
            if self._uses_global_input_buffer(name):
                i_mem_required = self._i_mem_required[name]
            else:
                i_mem_required = 0

            w_fit = w_mem_required <= size_sram_w
            i_fit = i_mem_required <= size_sram_i

            if w_fit and i_fit:
                # Both fit in SRAM: no refetch needed.
                num_fetch_w = 1
                num_fetch_i = 1
            elif w_fit and not i_fit:
                # Inputs overflow but weights fit: stream full input once while
                # keeping weights stationary in SRAM.
                num_fetch_w = 1
                num_fetch_i = 1
            elif (not w_fit) and i_fit:
                # Weights overflow but inputs fit: stream full weight once while
                # keeping inputs stationary in SRAM.
                num_fetch_w = 1
                num_fetch_i = 1
            else:
                # Both overflow: fall back to OS-style minimal-DRAM heuristic.
                num_refetch_input = math.ceil(w_mem_required / size_sram_w)
                num_refetch_weight = math.ceil(i_mem_required / size_sram_i)
                total_fetch_weight = num_refetch_weight * w_mem_required
                total_fetch_input = num_refetch_input * i_mem_required
                if (total_fetch_weight + i_mem_required) < (
                    total_fetch_input + w_mem_required
                ):
                    # refetch all weight for every input tile
                    num_fetch_w = num_refetch_weight
                    num_fetch_i = 1
                else:
                    # refetch all input for every weight tile
                    num_fetch_w = 1
                    num_fetch_i = num_refetch_input

            self._layer_mem_refetch[name] = (num_fetch_w, num_fetch_i)

    # ------------------------------------------------------------------
    # Memory instances (new bandwidth semantics)
    # ------------------------------------------------------------------
    def _init_mem(self):
        R = self.pe_array_dim["h"]
        C = self.pe_array_dim["w"]

        if self.is_bit_serial:
            w_bandwidth = self.pe_dp_size * math.ceil(self.w_prec / 4) * 4 * R / 2
        else:
            w_bandwidth = self.pe_dp_size * math.ceil(self.w_prec / 4) * 4 * R

        w_sram_bank = 8
        w_sram_config = {
            "technology": 0.028,
            "mem_type": "ram",
            "size": 512 * 1024 * 8,
            "bank_count": w_sram_bank,
            "rw_bw": w_bandwidth,
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
        }
        self.w_sram = MemoryInstance(
            w_sram_config,
            r_cost=0,
            w_cost=0,
            latency=1,
            min_r_granularity=None,
            min_w_granularity=64,
            get_cost_from_cacti=True,
        )

        if self.is_bit_serial:
            i_bandwidth = C * self.i_prec / 2
        else:
            i_bandwidth = C * self.i_prec

        i_sram_bank = 8
        i_sram_config = {
            "technology": 0.028,
            "mem_type": "ram",
            "size": 512 * 1024 * 8,
            "bank_count": i_sram_bank,
            "rw_bw": i_bandwidth,
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
        }
        self.i_sram = MemoryInstance(
            i_sram_config,
            r_cost=0,
            w_cost=0,
            latency=1,
            min_r_granularity=64,
            min_w_granularity=64,
            get_cost_from_cacti=True,
        )

        channel_scale = self.dram_channels
        # `dram_rw_bw` and DRAM per-transfer costs are interpreted as per-channel
        # quantities. Aggregate interface characteristics by channel count.
        dram_rw_bw = self.dram_rw_bw * channel_scale
        dram_config = {
            "technology": 0.028,
            "mem_type": "dram",
            "size": 1e9 * 8,
            "bank_count": 1,
            "rw_bw": dram_rw_bw,
            "r_port": 0,
            "w_port": 0,
            "rw_port": 1,
        }
        default_wr_cost = dram_rw_bw / 64 * 1200
        rd_cost = (
            self.dram_r_cost * channel_scale
            if self.dram_r_cost is not None
            else default_wr_cost
        )
        wr_cost = (
            self.dram_w_cost * channel_scale
            if self.dram_w_cost is not None
            else default_wr_cost
        )
        self.dram = MemoryInstance(
            dram_config,
            r_cost=rd_cost,
            w_cost=wr_cost,
            latency=1,
            min_r_granularity=dram_rw_bw,
            min_w_granularity=dram_rw_bw,
            get_cost_from_cacti=False,
        )
