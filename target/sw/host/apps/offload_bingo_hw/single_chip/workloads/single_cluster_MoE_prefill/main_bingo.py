# Fanchen Kong <fanchen.kong@kuleuven.be>

# This file is the main entry point for the bingo offload application
# Users will create the dfg in this file
# And then the mini-compiler will emit the WORKLOAD.h file
import os
import sys
import argparse
import numpy as np
import pathlib
import hjson
import re
import struct
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")

NODE37_DIAG_MODE = os.environ.get("BINGO_NODE37_DIAG", "none").strip().lower()
VALID_NODE37_DIAG_MODES = {
    "none",
    "mirror_node36_src",
    "bypass_node37_dep",
    "mirror_and_bypass",
}

if NODE37_DIAG_MODE not in VALID_NODE37_DIAG_MODES:
    raise ValueError(
        f"Unsupported BINGO_NODE37_DIAG mode: {NODE37_DIAG_MODE}. "
        f"Expected one of {sorted(VALID_NODE37_DIAG_MODES)}"
    )

print(f"BINGO_NODE37_DIAG: {NODE37_DIAG_MODE}")


def parse_inorder_completion_core_ids() -> set[int]:
    raw_core_ids = os.environ.get("BINGO_INORDER_CORE_IDS", "0").strip().lower()
    if raw_core_ids in {"", "none", "off", "false", "no"}:
        return set()
    if raw_core_ids in {"all", "*"}:
        return {0, 1, 2}
    return {
        int(core_id.strip()) for core_id in raw_core_ids.split(",") if core_id.strip()
    }


INORDER_COMPLETION_CORE_IDS = parse_inorder_completion_core_ids()

LEGACY_ENFORCE_INORDER_PER_CORE = os.environ.get(
    "BINGO_ENFORCE_INORDER_PER_CORE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
if LEGACY_ENFORCE_INORDER_PER_CORE and "BINGO_INORDER_CORE_IDS" not in os.environ:
    INORDER_COMPLETION_CORE_IDS = {0, 1, 2}

print(
    "BINGO_INORDER_CORE_IDS: "
    + (
        ",".join(str(core_id) for core_id in sorted(INORDER_COMPLETION_CORE_IDS))
        if INORDER_COMPLETION_CORE_IDS
        else "none"
    )
)

from bingo_dfg import BingoDFG
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_kernel_args import (
    BingoKernelArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs,
    SnaxBingoKernelGemmMinimalArgs,
    HostBingoKernelMoERouterScheduleArgs,
    HostBingoKernelComputeDelayedSoftmaxArgs,
    HostBingoKernelBuildScatterMetadataArgs,
    HostBingoKernelExpertsResultAccumulateArgs,
    HostBingoKernelScatterAndPadArgs,
)
from typing import Union, Dict


class HostBingoKernelComputeHwSiluGluArgs(BingoKernelArgs):
    def __init__(
        self,
        gate_silu_hw_data_addr,
        up_projection_hw_data_addr,
        activated_out_data_addr,
        valid_elements,
        swish_glu_scale_in_raw,
        swish_glu_scale_out_raw,
    ):
        self.gate_silu_hw_data_addr = gate_silu_hw_data_addr
        self.up_projection_hw_data_addr = up_projection_hw_data_addr
        self.activated_out_data_addr = activated_out_data_addr
        self.valid_elements = valid_elements
        self.swish_glu_scale_in_raw = swish_glu_scale_in_raw
        self.swish_glu_scale_out_raw = swish_glu_scale_out_raw

    def get_struct_name(self):
        return "__host_bingo_kernel_compute_hw_silu_glu_args_t"

    def get_c_field_assignments(self, handle_name_map):
        assignments = {}
        self._process_addr(
            self.gate_silu_hw_data_addr,
            "gate_silu_hw_data_addr",
            assignments,
            handle_name_map,
            split_64bit=False,
            as_64bit=True,
        )
        self._process_addr(
            self.up_projection_hw_data_addr,
            "up_projection_hw_data_addr",
            assignments,
            handle_name_map,
            split_64bit=False,
            as_64bit=True,
        )
        self._process_addr(
            self.activated_out_data_addr,
            "activated_out_data_addr",
            assignments,
            handle_name_map,
            split_64bit=False,
            as_64bit=True,
        )
        assignments["valid_elements"] = str(self.valid_elements)
        assignments["swish_glu_scale_in_raw"] = str(self.swish_glu_scale_in_raw)
        assignments["swish_glu_scale_out_raw"] = str(self.swish_glu_scale_out_raw)
        return assignments


class HostBingoKernelPrefillAccumulateArgs(BingoKernelArgs):
    def __init__(
        self,
        shared_expert_hw_output_addr,
        individual_experts_hw_output_addr,
        reverse_original_token_flat_idx_addr,
        global_calculated_probability_addr,
        expert_token_counts_addr,
        expert_memory_offsets_addr,
        final_layer_output_addr,
        actual_total_tokens,
        output_N2,
        per_tile_cols,
        max_tokens_per_expert,
        expert_number_each_layer_val,
        shared_expert_number_k_val,
        individual_expert_number_k_val,
        softmax_scale_raw,
        final_shift_step_val,
    ):
        self.shared_expert_hw_output_addr = shared_expert_hw_output_addr
        self.individual_experts_hw_output_addr = individual_experts_hw_output_addr
        self.reverse_original_token_flat_idx_addr = reverse_original_token_flat_idx_addr
        self.global_calculated_probability_addr = global_calculated_probability_addr
        self.expert_token_counts_addr = expert_token_counts_addr
        self.expert_memory_offsets_addr = expert_memory_offsets_addr
        self.final_layer_output_addr = final_layer_output_addr
        self.actual_total_tokens = actual_total_tokens
        self.output_N2 = output_N2
        self.per_tile_cols = per_tile_cols
        self.max_tokens_per_expert = max_tokens_per_expert
        self.expert_number_each_layer_val = expert_number_each_layer_val
        self.shared_expert_number_k_val = shared_expert_number_k_val
        self.individual_expert_number_k_val = individual_expert_number_k_val
        self.softmax_scale_raw = softmax_scale_raw
        self.final_shift_step_val = final_shift_step_val

    def get_struct_name(self):
        return "__host_bingo_kernel_prefill_accumulate_args_t"

    def get_c_field_assignments(self, handle_name_map):
        assignments = {}
        for field in [
            "shared_expert_hw_output_addr",
            "individual_experts_hw_output_addr",
            "reverse_original_token_flat_idx_addr",
            "global_calculated_probability_addr",
            "expert_token_counts_addr",
            "expert_memory_offsets_addr",
            "final_layer_output_addr",
        ]:
            self._process_addr(
                getattr(self, field),
                field,
                assignments,
                handle_name_map,
                split_64bit=False,
                as_64bit=True,
            )
        assignments["actual_total_tokens"] = str(self.actual_total_tokens)
        assignments["output_N2"] = str(self.output_N2)
        assignments["per_tile_cols"] = str(self.per_tile_cols)
        assignments["max_tokens_per_expert"] = str(self.max_tokens_per_expert)
        assignments["expert_number_each_layer_val"] = str(
            self.expert_number_each_layer_val
        )
        assignments["shared_expert_number_k_val"] = str(self.shared_expert_number_k_val)
        assignments["individual_expert_number_k_val"] = str(
            self.individual_expert_number_k_val
        )
        assignments["softmax_scale_raw"] = str(self.softmax_scale_raw)
        assignments["final_shift_step_val"] = str(self.final_shift_step_val)
        return assignments


# Override GemmFull to add silu_enable field
_OrigSnaxBingoKernelGemmFullArgs = SnaxBingoKernelGemmFullArgs


class SnaxBingoKernelGemmFullArgs(_OrigSnaxBingoKernelGemmFullArgs):
    def __init__(
        self,
        input_A_addr,
        input_B_addr,
        input_C_addr,
        output_D_addr,
        M,
        K,
        N,
        array_shape_idx,
        transpose_A,
        transpose_B,
        accumPrevC,
        silu_enable=0,
    ):
        super().__init__(
            input_A_addr,
            input_B_addr,
            input_C_addr,
            output_D_addr,
            M,
            K,
            N,
            array_shape_idx,
            transpose_A,
            transpose_B,
            accumPrevC,
        )
        self.silu_enable = silu_enable

    def get_c_field_assignments(self, handle_name_map):
        assignments = super().get_c_field_assignments(handle_name_map)
        assignments["silu_enable"] = str(self.silu_enable)
        return assignments


# GemmMinimalSiLU: writes only 4 address CSRs + 1 silu_enable extension CSR (5 total).
# Use for up[n=0] in individual experts: gate[n=0] already configured M/K/N/strides,
# only B_addr, D_addr, and silu_enable differ between gate and up.
class SnaxBingoKernelGemmMinimalSiluArgs(BingoKernelArgs):
    def __init__(
        self,
        input_A_addr,
        input_B_addr,
        input_C_addr,
        output_D_addr,
        silu_enable=0,
    ):
        self.input_A_addr = input_A_addr
        self.input_B_addr = input_B_addr
        self.input_C_addr = input_C_addr
        self.output_D_addr = output_D_addr
        self.silu_enable = silu_enable

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_gemm_minimal_silu_args_t"

    def get_c_field_assignments(self, handle_name_map):
        assignments = {}
        self._process_addr(
            self.input_A_addr,
            "input_A_addr",
            assignments,
            handle_name_map,
            split_64bit=False,
        )
        self._process_addr(
            self.input_B_addr,
            "input_B_addr",
            assignments,
            handle_name_map,
            split_64bit=False,
        )
        self._process_addr(
            self.input_C_addr,
            "input_C_addr",
            assignments,
            handle_name_map,
            split_64bit=False,
        )
        self._process_addr(
            self.output_D_addr,
            "output_D_addr",
            assignments,
            handle_name_map,
            split_64bit=False,
        )
        assignments["silu_enable"] = str(self.silu_enable)
        return assignments


# =========================================================================
# Post-processing: inject dynamic GEMM M patch table into generated C code
# =========================================================================
def inject_dynamic_patch_table(output_path, gemm_full_node_ids):
    """
    After bingo_compile_dfg() generates offload_bingo_hw.h, inject code to
    populate g_gemm_M_l1_ptrs[][] with the L1 addresses of GEMM M fields.
    Also injects MOE_OPERATOR_CUSTOM define before host.h to prevent
    host_kernel_lib.h from including the original MoE_operator.h.

    gemm_full_node_ids: dict[expert_id] -> {"gate": node_id, "up": node_id, "down": node_id}
    """
    with open(output_path, "r") as f:
        content = f.read()

    # Inject #define MOE_OPERATOR_CUSTOM before #include "host.h"
    content = content.replace(
        '#include "host.h"',
        '#define MOE_OPERATOR_CUSTOM\n#include "host.h"',
        1,  # only first occurrence
    )

    # Also ensure MoE_operator.h is included BEFORE host.h
    # Move MoE_operator.h include to before MOE_OPERATOR_CUSTOM
    content = content.replace('#include "MoE_operator.h"\n', "")
    content = content.replace(
        '#define MOE_OPERATOR_CUSTOM\n#include "host.h"',
        '#include "MoE_operator.h"\n#define MOE_OPERATOR_CUSTOM\n#include "host.h"',
    )

    # Build patch table init code
    patch_lines = [
        "\n        // ===== Dynamic M Patch Table Init (injected by main_bingo.py) ====="
    ]
    for eid, projections in sorted(gemm_full_node_ids.items()):
        for proj_idx, proj_name in enumerate(["gate", "up", "down"]):
            if proj_name not in projections:
                continue
            nid = projections[proj_name]
            patch_lines.append(
                f"        g_gemm_M_l1_ptrs[{eid}][{proj_idx}] = &(args_dev_chip00_{nid}->M);"
            )
    patch_lines.append("        // ===== End Dynamic M Patch Table Init =====\n")
    patch_code = "\n".join(patch_lines)

    # Insert BEFORE bingo_hw_scheduler_init
    marker = 'printf_safe("Chip(%x, %x): [Host] Init HW Bingo Scheduler'
    idx = content.find(marker)
    if idx < 0:
        raise RuntimeError("Cannot find scheduler_init marker in generated code")
    content = content[:idx] + patch_code + "\n        " + content[idx:]

    with open(output_path, "w") as f:
        f.write(content)


# =========================================================================
# Multi-Phase Dynamic Scheduling: Build sub-DFGs and generate per-phase code
# =========================================================================


def _classify_nodes_by_phase(bingo_dfg, indiv_nodes, node_accumulate):
    """Assign each user node to a phase: 'router_shared', 'expert_N', or 'accumulate'."""
    # Build reverse index: node_id → expert_id for individual expert nodes
    indiv_nid_to_expert = {}
    for category, entries in indiv_nodes.items():
        for key, node in entries.items():
            expert_id = key[0]
            indiv_nid_to_expert[node.node_id] = expert_id

    phase_map = {}  # node_id → phase_name
    for node in bingo_dfg.node_list:
        nid = node.node_id
        if nid in indiv_nid_to_expert:
            phase_map[nid] = f"expert_{indiv_nid_to_expert[nid]}"
        elif nid == node_accumulate.node_id:
            phase_map[nid] = "accumulate"
        else:
            phase_map[nid] = "router_shared"
    return phase_map


def _create_sub_dfg(bingo_dfg, phase_map, phase_name):
    """Create a sub-DFG for a given phase. Returns (sub_dfg, orig_to_sub_map)."""
    sub_dfg = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=1,
        num_cores_per_cluster=2,
        is_host_as_acc=True,
        chiplet_ids=[0x00],
    )
    orig_to_sub = {}  # original_node → sub_node
    phase_nodes = [
        n for n in bingo_dfg.node_list if phase_map.get(n.node_id) == phase_name
    ]

    for orig_node in phase_nodes:
        new_node = BingoNode(
            assigned_chiplet_id=orig_node.assigned_chiplet_id,
            assigned_cluster_id=orig_node.assigned_cluster_id,
            assigned_core_id=orig_node.assigned_core_id,
            kernel_name=orig_node.kernel_name,
            kernel_args=orig_node.kernel_args,  # share the same args object
        )
        sub_dfg.bingo_add_node(new_node)
        orig_to_sub[orig_node] = new_node

    # Add intra-phase edges
    for src, dst in bingo_dfg.edges():
        if src in orig_to_sub and dst in orig_to_sub:
            sub_dfg.bingo_add_edge(orig_to_sub[src], orig_to_sub[dst])

    return sub_dfg, orig_to_sub


def _build_dep_cleanup_descs(sub_dfg, chiplet_id=0):
    """Build 3 packed dummy task descriptors that saturate the dep_matrix.

    The HW Manager's dep_matrix is NOT reset on start=1. Residual bits from a
    previous phase corrupt dependency resolution. We inject:
        3 dummy tasks: dep_set=ALL on each core column → all matrix bits = 1
    With all bits set, every dep_check in the new phase passes immediately.
    The dep_check of the first real task on each core then clears the bits
    that were set, and subsequent tasks depend only on fresh dep_set ops.
    NOTE: RTL drops dummy tasks with dep_check_en=1 (line 666), so we
    cannot use dummies to CLEAR the matrix; instead we saturate it.
    """
    import math

    nc = sub_dfg.num_cores_per_cluster  # typically 3 (2 device + host)
    nclust = sub_dfg.num_clusters_per_chiplet
    cid_w = math.ceil(math.log2(nclust)) if nclust > 1 else 1
    core_w = math.ceil(math.log2(nc)) if nc > 1 else 1
    chip_id_w = 8
    task_id_w = 12
    dep_all = (1 << nc) - 1  # e.g. 0b111 for nc=3

    max_node_id = max(n.node_id for n in sub_dfg.node_list)
    base_id = max_node_id + 1

    descs = []
    for core in range(nc):
        nid = base_id + core
        s = 0
        packed = 0
        packed |= 1 << s
        s += 1  # task_type = 1 (dummy)
        packed |= nid << s
        s += task_id_w  # task_id
        packed |= chiplet_id << s
        s += chip_id_w  # chiplet_id
        packed |= 0 << s
        s += cid_w  # cluster_id = 0
        packed |= core << s
        s += core_w  # core_id
        packed |= 0 << s
        s += 1  # dep_check_en = 0
        packed |= 0 << s
        s += nc  # dep_check_code = 0
        packed |= 1 << s
        s += 1  # dep_set_en = 1
        packed |= 0 << s
        s += 1  # dep_set_all_chiplet = 0
        packed |= 0 << s
        s += chip_id_w  # dep_set_chiplet_id = 0
        packed |= 0 << s
        s += cid_w  # dep_set_cluster_id = 0
        packed |= dep_all << s
        s += nc  # dep_set_code = ALL
        descs.append((nid, packed, f"CLEANUP dep_set ALL, core {core}"))
    return descs


def _emit_phase_code(
    phase_name,
    sub_dfg,
    orig_to_sub,
    chiplet_id=0,
    is_final=True,
    is_first=True,
    alloc_ndev=None,
    alloc_ntasks=None,
):
    """Generate C code for one scheduler phase. Returns list of indented C lines.

    is_final: True → exit_code=0 (final EXIT). False → exit_code=3 (PHASE_EXIT).
    is_first: True → bingo_hw_scheduler_init(). False → bingo_hw_scheduler_reinit().
    alloc_ndev/alloc_ntasks: override L3 alloc sizes (first phase uses max across phases).
    """
    lines = []
    lines.append(f"// ===== Phase: {phase_name} =====")

    # --- Task descriptor list ---
    topo_nodes = list(nx.topological_sort(sub_dfg))
    local_nodes = [n for n in topo_nodes if n.assigned_chiplet_id == chiplet_id]
    num_local = len(local_nodes)
    all_nodes_sorted = sorted(sub_dfg.node_list, key=lambda n: n.node_id)

    # --- dep_matrix cleanup disabled (proven ineffective) ---
    cleanup_descs = []
    num_cleanup = 0

    # num_all must cover all node_ids including cleanup task ids
    max_real_id = max(n.node_id for n in all_nodes_sorted) if all_nodes_sorted else 0
    if cleanup_descs:
        max_cleanup_id = max(nid for nid, _, _ in cleanup_descs)
        num_all = max(max_real_id, max_cleanup_id) + 1
    else:
        num_all = len(all_nodes_sorted)

    total_task_desc = num_local + num_cleanup

    lines.append(f"uint32_t {phase_name}_num_tasks = {num_all};")
    lines.append(f"uint32_t {phase_name}_num_task_desc = {total_task_desc};")
    lines.append(
        f"uint64_t* {phase_name}_task_desc = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {max(total_task_desc,1)} * sizeof(uint64_t));"
    )
    # Cleanup dummy tasks first (prepended before real tasks)
    for i, (nid, packed, comment) in enumerate(cleanup_descs):
        lines.append(f"{phase_name}_task_desc[{i}] = 0x{packed:016X}; // {comment}")
    # Real tasks after cleanup
    for idx, node in enumerate(local_nodes):
        packed = sub_dfg.bingo_pack_node(node)
        lines.append(
            f"{phase_name}_task_desc[{idx + num_cleanup}] = 0x{packed:016X}; // Node {node.node_id} ({node.kernel_name or node.node_type})"
        )

    # --- Mapping lists (first phase uses max sizes for TCDM reservation) ---
    g2d_alloc_size = alloc_ntasks if (alloc_ntasks and is_first) else num_all
    lines.append(
        f"int32_t* {phase_name}_g2d = (int32_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {g2d_alloc_size} * sizeof(int32_t));"
    )
    dev_counter = 0
    for idx, node in enumerate(all_nodes_sorted):
        if node.kernel_name and node.kernel_name.startswith("__snax"):
            lines.append(f"{phase_name}_g2d[{idx}] = {dev_counter};")
            dev_counter += 1
        else:
            lines.append(f"{phase_name}_g2d[{idx}] = -1;")
    # Cleanup task entries in g2d (auto-completed by HW, never reach cores)
    for nid, _, _ in cleanup_descs:
        lines.append(f"{phase_name}_g2d[{nid}] = -1;")
    lines.append(f"uint32_t {phase_name}_ndev = {dev_counter};")

    lines.append(
        f"int32_t* {phase_name}_g2h = (int32_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {num_all} * sizeof(int32_t));"
    )
    host_counter = 0
    for idx, node in enumerate(all_nodes_sorted):
        if node.kernel_name and node.kernel_name.startswith("__host"):
            lines.append(f"{phase_name}_g2h[{idx}] = {host_counter};")
            host_counter += 1
        else:
            lines.append(f"{phase_name}_g2h[{idx}] = -1;")
    # Cleanup task entries in g2h
    for nid, _, _ in cleanup_descs:
        lines.append(f"{phase_name}_g2h[{nid}] = -1;")
    lines.append(f"uint32_t {phase_name}_nhost = {host_counter};")

    # --- Arg/kernel list allocation (first phase uses max sizes for TCDM reservation) ---
    da_alloc = alloc_ndev if (alloc_ndev and is_first) else max(dev_counter, 1)
    g2d_alloc = alloc_ntasks if (alloc_ntasks and is_first) else num_all
    lines.append(
        f"uint32_t* {phase_name}_da = (uint32_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {da_alloc} * sizeof(uint32_t));"
    )
    lines.append(
        f"uint32_t* {phase_name}_dk = (uint32_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {da_alloc} * sizeof(uint32_t));"
    )
    lines.append(
        f"uint64_t* {phase_name}_ha = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {max(host_counter,1)} * sizeof(uint64_t));"
    )
    lines.append(
        f"uint64_t* {phase_name}_hk = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {max(host_counter,1)} * sizeof(uint64_t));"
    )

    # --- Collect memory handle name map from phase nodes ---
    handle_name_map = {}
    for node in all_nodes_sorted:
        if node.kernel_args:
            for attr, value in node.kernel_args.__dict__.items():
                if isinstance(value, BingoMemAlloc):
                    handle_name_map[value] = value.get_c_var_name()

    # --- Arg init ---
    dev_idx = 0
    host_idx = 0
    gemm_full_vars = (
        {}
    )  # (sub_node_id) → arg_var_name for GemmFull nodes (have M field)

    for node in all_nodes_sorted:
        kn = node.kernel_name
        is_dev = kn and kn.startswith("__snax")
        is_host = kn and kn.startswith("__host")
        if not (is_dev or is_host):
            continue

        if is_dev:
            av = f"{phase_name}_ad{node.node_id}"
            if node.kernel_args:
                st = node.kernel_args.get_struct_name()
                lines.append(
                    f"{st}* {av} = ({st}*)bingo_l1_alloc(0x{chiplet_id:02x}, {node.assigned_cluster_id}, sizeof({st}));"
                )
                for field, val in node.kernel_args.get_c_field_assignments(
                    handle_name_map
                ).items():
                    lines.append(f"{av}->{field} = {val};")
                lines.append(f"{phase_name}_da[{dev_idx}] = (uint32_t)(uintptr_t){av};")
                # Track GemmFull nodes (they have M field)
                if isinstance(node.kernel_args, SnaxBingoKernelGemmFullArgs):
                    gemm_full_vars[node.node_id] = av
            elif "exit" in kn:
                lines.append(
                    f"__snax_bingo_kernel_exit_args_t* {av} = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x{chiplet_id:02x}, {node.assigned_cluster_id}, sizeof(__snax_bingo_kernel_exit_args_t));"
                )
                _exit_val = 0 if is_final else 3  # BINGO_RET_PHASE_EXIT
                lines.append(f"{av}->exit_code = {_exit_val};")
                lines.append(f"{phase_name}_da[{dev_idx}] = (uint32_t)(uintptr_t){av};")
            else:
                lines.append(f"{phase_name}_da[{dev_idx}] = 0;")
            lines.append(
                f'{phase_name}_dk[{dev_idx}] = (uint32_t)(uintptr_t)get_device_function("{kn}");'
            )
            dev_idx += 1

        elif is_host:
            av = f"{phase_name}_ah{node.node_id}"
            if node.kernel_args:
                st = node.kernel_args.get_struct_name()
                lines.append(
                    f"{st}* {av} = ({st}*)bingo_l3_alloc(0x{chiplet_id:02x}, sizeof({st}));"
                )
                for field, val in node.kernel_args.get_c_field_assignments(
                    handle_name_map
                ).items():
                    lines.append(f"{av}->{field} = {val};")
                lines.append(
                    f"{phase_name}_ha[{host_idx}] = (uint64_t)(uintptr_t){av};"
                )
            elif "exit" in kn:
                lines.append(
                    f"__host_bingo_kernel_exit_args_t* {av} = (__host_bingo_kernel_exit_args_t*)bingo_l3_alloc(0x{chiplet_id:02x}, sizeof(__host_bingo_kernel_exit_args_t));"
                )
                _exit_val = 0 if is_final else 3  # BINGO_RET_PHASE_EXIT
                lines.append(f"{av}->exit_code = {_exit_val};")
                lines.append(
                    f"{phase_name}_ha[{host_idx}] = (uint64_t)(uintptr_t){av};"
                )
            else:
                lines.append(f"{phase_name}_ha[{host_idx}] = 0;")
            lines.append(f"{phase_name}_hk[{host_idx}] = (uint64_t)(uintptr_t)&{kn};")
            host_idx += 1

    # --- Scheduler launch ---
    lines.append(
        f'printf_safe("Chip(%x, %x): [Host] Phase {phase_name} Init HW Scheduler\\r\\n", get_current_chip_loc_x(), get_current_chip_loc_y());'
    )
    if is_first:
        # First phase: full init with max sizes for TCDM reservation
        init_ndev = alloc_ndev if alloc_ndev else f"{phase_name}_ndev"
        init_ntasks = alloc_ntasks if alloc_ntasks else f"{phase_name}_num_tasks"
        lines.append(f"bingo_hw_scheduler_init((uint64_t)(uintptr_t){phase_name}_da,")
        lines.append(f"    (uint64_t)(uintptr_t){phase_name}_dk,")
        lines.append(f"    {init_ndev},")
        lines.append(f"    (uint64_t)(uintptr_t){phase_name}_g2d,")
        lines.append(f"    {init_ntasks},")
        lines.append(f"    (uint64_t)(uintptr_t){phase_name}_task_desc,")
        lines.append(f"    {phase_name}_num_task_desc);")
    else:
        # Subsequent phases: reinit reuses TCDM in-place
        lines.append(f"_bingo_phase_counter++;")
        lines.append(f"bingo_hw_scheduler_reinit((uint64_t)(uintptr_t){phase_name}_da,")
        lines.append(f"    (uint64_t)(uintptr_t){phase_name}_dk,")
        lines.append(f"    {phase_name}_ndev,")
        lines.append(f"    (uint64_t)(uintptr_t){phase_name}_g2d,")
        lines.append(f"    {phase_name}_num_tasks,")
        lines.append(f"    (uint64_t)(uintptr_t){phase_name}_task_desc,")
        lines.append(f"    {phase_name}_num_task_desc,")
        lines.append(f"    _bingo_phase_counter);")
    lines.append(
        f"err = bingo_hw_scheduler({phase_name}_ha, {phase_name}_hk, {phase_name}_g2h);"
    )
    lines.append(
        f'if (err) {{ printf_safe("Phase {phase_name} failed err=%u\\r\\n", err); return err; }}'
    )

    return lines, gemm_full_vars


def postprocess_multi_phase(
    output_path, bingo_dfg, indiv_nodes, node_accumulate, params, gemm_full_node_ids
):
    """
    Replace the single-phase generated offload_bingo_hw.h with multi-phase
    dynamic dispatch code.
    """
    expert_count = params["expert_number_each_layer"]

    # 1. Classify nodes
    phase_map = _classify_nodes_by_phase(bingo_dfg, indiv_nodes, node_accumulate)
    phase_names = ["router_shared"]
    for eid in range(expert_count):
        phase_names.append(f"expert_{eid}")
    phase_names.append("accumulate")

    # 2. Create sub-DFGs and run transforms
    phase_data = {}  # phase_name → (sub_dfg, orig_to_sub)
    for pname in phase_names:
        phase_node_ids = {nid for nid, p in phase_map.items() if p == pname}
        if not phase_node_ids:
            continue
        sub_dfg, orig_to_sub = _create_sub_dfg(bingo_dfg, phase_map, pname)
        enforce_in_order_completion_per_core(sub_dfg)
        sub_dfg.bingo_transform_dfg_add_entry_node()
        sub_dfg.bingo_transform_dfg_add_exit_nodes()
        sub_dfg.bingo_transform_dfg_add_dummy_set_nodes()
        sub_dfg.bingo_transform_dfg_add_dummy_check_nodes()
        sub_dfg.bingo_assign_normal_node_dep_set_info()
        sub_dfg.bingo_assign_normal_node_dep_check_info()
        phase_data[pname] = (sub_dfg, orig_to_sub)

    # 3. Collect ALL unique memory handles for global allocation
    all_handles = set()
    for node in sorted(bingo_dfg.node_list, key=lambda n: n.node_id):
        if node.kernel_args:
            for attr, value in node.kernel_args.__dict__.items():
                if isinstance(value, BingoMemAlloc):
                    all_handles.add(value)
    sorted_handles = sorted(list(all_handles), key=lambda h: h.name)
    handle_name_map = {h: h.get_c_var_name() for h in sorted_handles}

    # Helper: compute M pointer patch lines for an expert phase
    def _compute_m_patches(eid, orig_to_sub_e, gemm_full_vars, phase_prefix):
        """Return list of C lines setting g_gemm_M_l1_ptrs for this expert."""
        patch_lines = []
        if eid not in gemm_full_node_ids:
            return patch_lines
        orig_gemm_ids = gemm_full_node_ids[eid]
        for proj_idx, proj_name in enumerate(["gate", "up", "down"]):
            orig_nid = orig_gemm_ids.get(proj_name)
            if orig_nid is None:
                continue
            orig_node = None
            for n in bingo_dfg.node_list:
                if n.node_id == orig_nid:
                    orig_node = n
                    break
            if orig_node and orig_node in orig_to_sub_e:
                sub_node = orig_to_sub_e[orig_node]
                if sub_node.node_id in gemm_full_vars:
                    av = gemm_full_vars[sub_node.node_id]
                    patch_lines.append(
                        f"g_gemm_M_l1_ptrs[{eid}][{proj_idx}] = &({av}->M);"
                    )
        return patch_lines

    # 4. Pre-compute max sizes across all phases (for TCDM reservation in Phase 1)
    # Non-first phases get 6 cleanup dummy tasks prepended (dep_matrix reset).
    # Their task-ID namespace expands by 6, so max_ntasks must cover that.
    NUM_CLEANUP_TASKS = 0  # cleanup disabled (dep_matrix not the issue)
    phase_sizes = {}
    for pname, (sub_dfg_p, _) in phase_data.items():
        all_sorted_p = sorted(sub_dfg_p.node_list, key=lambda n: n.node_id)
        ndev_p = sum(
            1
            for n in all_sorted_p
            if n.kernel_name and n.kernel_name.startswith("__snax")
        )
        ntasks_p = len(all_sorted_p)
        if pname != "router_shared":
            ntasks_p += NUM_CLEANUP_TASKS  # cleanup IDs extend array
        phase_sizes[pname] = {"ndev": ndev_p, "ntasks": ntasks_p}
    max_ndev = max(s["ndev"] for s in phase_sizes.values())
    max_ntasks = max(s["ntasks"] for s in phase_sizes.values())

    # 5. Generate the combined multi-phase header file
    with open(output_path, "w") as f:
        # --- Headers ---
        f.write(
            "// Auto-generated offload_hw_bingo.h (multi-phase dynamic scheduling)\n"
        )
        f.write("#pragma once\n")
        f.write('#include "libbingo/bingo_api.h"\n')
        f.write('#include "MoE_operator.h"\n')
        f.write("#define MOE_OPERATOR_CUSTOM\n")
        f.write('#include "host.h"\n')
        data_header = f"{params['app_name']}_data.h"
        f.write(f'#include "{data_header}"\n\n')

        # --- kernel_execution function ---
        f.write("int kernel_execution(){\n")
        f.write("    check_kernel_tab_ready();\n")
        f.write(
            f'    printf_safe("Chip(%x, %x): [Host] Multi-Phase MoE Prefill\\r\\n", get_current_chip_loc_x(), get_current_chip_loc_y());\n'
        )
        f.write("    uint32_t current_chip_id = get_current_chip_id();\n")
        f.write("    if (current_chip_id == 0x00) {\n")
        f.write("        uint32_t err = 0;\n")
        f.write(f"        uint32_t _bingo_phase_counter = 1;\n\n")

        # --- Global memory allocations ---
        f.write("        // ===== Global Memory Allocations =====\n")
        for h in sorted_handles:
            c_var = handle_name_map[h]
            if h.mem_level == "L1":
                f.write(
                    f"        uint64_t {c_var} = bingo_l1_alloc(0x{h.chip_id:02x}, {h.cluster_id}, {h.size});\n"
                )
            elif h.mem_level == "L2":
                f.write(
                    f"        uint64_t {c_var} = bingo_l2_alloc(0x{h.chip_id:02x}, {h.size});\n"
                )
            else:
                f.write(
                    f"        uint64_t {c_var} = bingo_l3_alloc(0x{h.chip_id:02x}, {h.size});\n"
                )
        f.write("\n")

        # ===== PREBUILD OPTIMIZATION =====
        # Pre-generate ALL expert + accumulate phase data BEFORE RS phase.
        # This eliminates the ~108K CC gap between RS exit and first expert dispatch.
        # At dispatch time, only M patches + scatter_pad + scheduler reinit are needed.

        # Pre-build expert phases
        expert_prebuild = {}  # eid → (setup_lines, launch_lines, gemm_full_vars, orig_to_sub)
        for eid in range(expert_count):
            pname = f"expert_{eid}"
            if pname not in phase_data:
                continue
            sub_dfg_e, orig_to_sub_e = phase_data[pname]
            code_lines, gemm_full_vars = _emit_phase_code(
                f"e{eid}",
                sub_dfg_e,
                orig_to_sub_e,
                is_final=False,
                is_first=False,
            )
            # Split at "Init HW Scheduler" marker: setup before, launch after
            launch_idx = None
            for i, line in enumerate(code_lines):
                if "Init HW Scheduler" in line:
                    launch_idx = i
                    break
            if launch_idx is not None:
                setup_lines = code_lines[:launch_idx]
                launch_lines = code_lines[launch_idx:]
            else:
                setup_lines = code_lines
                launch_lines = []
            expert_prebuild[eid] = (setup_lines, launch_lines, gemm_full_vars, orig_to_sub_e)

        # Pre-build accumulate phase
        acc_setup_lines = []
        acc_launch_lines = []
        if "accumulate" in phase_data:
            sub_dfg_a, orig_to_sub_a = phase_data["accumulate"]
            code_lines, _ = _emit_phase_code(
                "acc",
                sub_dfg_a,
                orig_to_sub_a,
                is_final=True,
                is_first=False,
            )
            launch_idx = None
            for i, line in enumerate(code_lines):
                if "Init HW Scheduler" in line:
                    launch_idx = i
                    break
            if launch_idx is not None:
                acc_setup_lines = code_lines[:launch_idx]
                acc_launch_lines = code_lines[launch_idx:]
            else:
                acc_setup_lines = code_lines
                acc_launch_lines = []

        # --- Emit pre-built expert setup code (before RS phase) ---
        f.write("        // ===== Pre-build Expert Phase Data (before RS phase) =====\n")
        for eid in sorted(expert_prebuild.keys()):
            setup_lines, _, _, _ = expert_prebuild[eid]
            f.write(f"        // --- Expert {eid} setup ---\n")
            for line in setup_lines:
                f.write(f"        {line}\n")
            f.write("\n")

        # --- Emit pre-built accumulate setup code ---
        if acc_setup_lines:
            f.write("        // ===== Pre-build Accumulate Phase Data =====\n")
            for line in acc_setup_lines:
                f.write(f"        {line}\n")
            f.write("\n")

        # --- Phase 1: Router + Shared (first phase, intermediate exit) ---
        if "router_shared" in phase_data:
            sub_dfg_rs, orig_to_sub_rs = phase_data["router_shared"]
            code_lines, _ = _emit_phase_code(
                "rs",
                sub_dfg_rs,
                orig_to_sub_rs,
                is_final=False,
                is_first=True,
                alloc_ndev=max_ndev,
                alloc_ntasks=max_ntasks,
            )
            for line in code_lines:
                f.write(f"        {line}\n")
            f.write("\n")

        # --- Read token counts from L3 histogram ---
        hist_var = handle_name_map.get(
            next((h for h in sorted_handles if "exp_hist" in h.name), None),
            "alloc_l3_exp_hist_chip_00",
        )
        f.write(f"        // ===== Read Expert Token Counts =====\n")
        f.write(
            f"        volatile uint32_t* _hist = (volatile uint32_t*)(uintptr_t){hist_var};\n"
        )
        f.write(f"        uint32_t _tcounts[{expert_count}];\n")
        f.write(
            f"        for (int _e = 0; _e < {expert_count}; _e++) _tcounts[_e] = _hist[_e];\n"
        )
        f.write(
            f'        printf_safe("Chip(%x, %x): [Host] Token counts:", get_current_chip_loc_x(), get_current_chip_loc_y());\n'
        )
        f.write(
            f'        for (int _e = 0; _e < {expert_count}; _e++) printf_safe(" E%d=%u", _e, _tcounts[_e]);\n'
        )
        f.write(f'        printf_safe("\\r\\n");\n\n')

        # --- Expert dispatch: only M patches + scatter_pad + launch (data pre-built) ---
        for eid in range(expert_count):
            if eid not in expert_prebuild:
                continue
            _, launch_lines, gemm_full_vars, orig_to_sub_e = expert_prebuild[eid]

            f.write(f"        // ===== Expert {eid}: dispatch if tokens > 0 =====\n")
            f.write(f"        if (_tcounts[{eid}] > 0) {{\n")

            # M pointer patches (dynamic based on token counts)
            m_patches = _compute_m_patches(
                eid, orig_to_sub_e, gemm_full_vars, f"e{eid}"
            )
            for patch_line in m_patches:
                f.write(f"            {patch_line}\n")

            # Pre-execute scatter_pad_dynamic BEFORE scheduler to avoid
            # SoC DMA (host scatter) + iDMA (device weight load) AXI deadlock.
            pfx = f"e{eid}"
            f.write(
                f"            // Pre-execute scatter_pad to avoid SoC DMA + iDMA deadlock\n"
            )
            f.write(
                f"            __host_bingo_kernel_scatter_pad_dynamic((void*)(uintptr_t){pfx}_ha[0]);\n"
            )
            f.write(
                f"            // Replace scatter_pad with NOP in scheduler (already executed)\n"
            )
            f.write(
                f"            {pfx}_hk[0] = (uint64_t)(uintptr_t)&__host_bingo_kernel_entry;\n"
            )

            # Scheduler launch lines only (reinit + bingo_hw_scheduler)
            for line in launch_lines:
                f.write(f"            {line}\n")

            f.write(f"        }}\n\n")

        # --- Phase Final: Accumulate (launch only, data pre-built) ---
        if acc_launch_lines:
            f.write("        // ===== Phase: Accumulate (final) =====\n")
            for line in acc_launch_lines:
                f.write(f"        {line}\n")
            f.write("\n")

        f.write("    }\n")
        f.write("    return 0;\n")
        f.write("}\n")

    print(f"[MultiPhase] Generated multi-phase dynamic scheduling → {output_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Bingo HW Manager")
    parser.add_argument(
        "--cfg",
        type=pathlib.Path,
        default=pathlib.Path(current_dir) / "params.hjson",
        help="Workload parameter config file",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        default=pathlib.Path(ROOT_DIR)
        / "deps"
        / "snitch_cluster"
        / "target"
        / "snitch_cluster"
        / "cfg"
        / "snax_versacore_to_cluster.hjson",
        help="Hardware config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--output_offload_file_name",
        type=str,
        default="offload_bingo_hw.h",
        help="Output filename for the offload header file",
    )
    parser.add_argument(
        "--emit_mini_golden",
        action="store_true",
        help="Emit mini golden data for verification",
    )
    return parser.parse_args()


head_path = os.path.join(current_dir, "MoE_common_variable.h")


def parse_header_config(head_path):
    MoE_config = {}
    pattern = re.compile(r"#define\s+(\w+)\s+([-+]?[\d\.]+)")
    with open(head_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                key = match.group(1)
                value_str = match.group(2).strip()
                MoE_config[key] = eval(value_str)
    return MoE_config


def load_workload_config(args):
    with args.cfg.open() as cfg_file:
        param_cfg = hjson.loads(cfg_file.read())

    with args.hwcfg.open() as hw_file:
        hw_cfg = hjson.loads(hw_file.read())

    merged_config = {**param_cfg, **hw_cfg}
    merged_config["emit_mini_golden"] = args.emit_mini_golden
    return merged_config


def define_workload_params(**kwargs):
    """Defines the GeMM workload parameters."""
    MoE_config = parse_header_config(head_path)
    num_double_buffers = 2
    data_type = 0  # int8 data type
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    array_shape = kwargs["array_shape"]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        0
    ]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        1
    ]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        2
    ]

    params = {
        "app_name": "single_cluster_MoE_prefill",
        "num_double_buffers": num_double_buffers,
        "router_M1": kwargs["router_M1"],
        "router_N1": kwargs["router_N1"],
        "router_K1": kwargs["router_K1"],
        "router_M2": kwargs["router_M2"],
        "router_N2": kwargs["router_N2"],
        "router_K2": kwargs["router_K2"],
        "shared_swish_glu_M1": kwargs["shared_swish_glu_M1"],
        "shared_swish_glu_N1": kwargs["shared_swish_glu_N1"],
        "shared_swish_glu_K1": kwargs["shared_swish_glu_K1"],
        "shared_swish_glu_M2": kwargs["shared_swish_glu_M2"],
        "shared_swish_glu_N2": kwargs["shared_swish_glu_N2"],
        "shared_swish_glu_K2": kwargs["shared_swish_glu_K2"],
        "shared_down_projection_M1": kwargs["shared_down_projection_M1"],
        "shared_down_projection_N1": kwargs["shared_down_projection_N1"],
        "shared_down_projection_K1": kwargs["shared_down_projection_K1"],
        "shared_down_projection_M2": kwargs["shared_down_projection_M2"],
        "shared_down_projection_N2": kwargs["shared_down_projection_N2"],
        "shared_down_projection_K2": kwargs["shared_down_projection_K2"],
        "individual_swish_glu_M1": kwargs["individual_swish_glu_M1"],
        "individual_swish_glu_N1": kwargs["individual_swish_glu_N1"],
        "individual_swish_glu_K1": kwargs["individual_swish_glu_K1"],
        "individual_swish_glu_M2": kwargs["individual_swish_glu_M2"],
        "individual_swish_glu_N2": kwargs["individual_swish_glu_N2"],
        "individual_swish_glu_K2": kwargs["individual_swish_glu_K2"],
        "individual_down_projection_M1": kwargs["individual_down_projection_M1"],
        "individual_down_projection_N1": kwargs["individual_down_projection_N1"],
        "individual_down_projection_K1": kwargs["individual_down_projection_K1"],
        "individual_down_projection_M2": kwargs["individual_down_projection_M2"],
        "individual_down_projection_N2": kwargs["individual_down_projection_N2"],
        "individual_down_projection_K2": kwargs["individual_down_projection_K2"],
        "meshRow": snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][
            array_shape
        ][0],
        "tileSize": snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][
            array_shape
        ][1],
        "meshCol": snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][
            array_shape
        ][2],
        "array_shape": array_shape,
        "snax_acc_cfg": kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0],
    }
    # Router stage
    params["router_A_tileSize"] = (
        params["router_M1"]
        * params["router_K1"]
        * params["meshRow"]
        * params["tileSize"]
        * 1
    )
    params["router_B_tileSize"] = (
        params["router_K1"]
        * params["router_N1"]
        * params["meshCol"]
        * params["tileSize"]
        * 1
    )
    params["router_D_tileSize"] = (
        params["router_M1"]
        * params["router_N1"]
        * params["meshRow"]
        * params["meshCol"]
        * 4
    )

    # shared expert stage
    params["shared_swish_glu_A_tileSize"] = (
        params["shared_swish_glu_M1"]
        * params["shared_swish_glu_K1"]
        * params["meshRow"]
        * params["tileSize"]
        * 1
    )
    params["shared_swish_glu_B_tileSize"] = (
        params["shared_swish_glu_K1"]
        * params["shared_swish_glu_N1"]
        * params["meshCol"]
        * params["tileSize"]
        * 1
    )
    params["shared_swish_glu_D_tileSize"] = (
        params["shared_swish_glu_M1"]
        * params["shared_swish_glu_N1"]
        * params["meshRow"]
        * params["meshCol"]
        * 4
    )
    params["shared_down_projection_A_tileSize"] = (
        params["shared_down_projection_M1"]
        * params["shared_down_projection_K1"]
        * params["meshRow"]
        * params["tileSize"]
        * 1
    )
    params["shared_down_projection_B_tileSize"] = (
        params["shared_down_projection_K1"]
        * params["shared_down_projection_N1"]
        * params["meshCol"]
        * params["tileSize"]
        * 1
    )
    params["shared_down_projection_D_tileSize"] = (
        params["shared_down_projection_M1"]
        * params["shared_down_projection_N1"]
        * params["meshRow"]
        * params["meshCol"]
        * 4
    )

    # individual expert stage
    params["individual_swish_glu_A_tileSize"] = (
        params["individual_swish_glu_M1"]
        * params["individual_swish_glu_K1"]
        * params["meshRow"]
        * params["tileSize"]
        * 1
    )
    params["individual_swish_glu_B_tileSize"] = (
        params["individual_swish_glu_K1"]
        * params["individual_swish_glu_N1"]
        * params["meshCol"]
        * params["tileSize"]
        * 1
    )
    params["individual_swish_glu_D_tileSize"] = (
        params["individual_swish_glu_M1"]
        * params["individual_swish_glu_N1"]
        * params["meshRow"]
        * params["meshCol"]
        * 4
    )
    params["individual_down_projection_A_tileSize"] = (
        params["individual_down_projection_M1"]
        * params["individual_down_projection_K1"]
        * params["meshRow"]
        * params["tileSize"]
        * 1
    )
    params["individual_down_projection_B_tileSize"] = (
        params["individual_down_projection_K1"]
        * params["individual_down_projection_N1"]
        * params["meshCol"]
        * params["tileSize"]
        * 1
    )
    params["individual_down_projection_D_tileSize"] = (
        params["individual_down_projection_M1"]
        * params["individual_down_projection_N1"]
        * params["meshRow"]
        * params["meshCol"]
        * 4
    )

    params["input_dimension"] = MoE_config["input_dimension"]
    params["expert_number_each_layer"] = MoE_config["expert_number_each_layer"]
    params["individual_expert_number_k"] = MoE_config["individual_expert_number_k"]
    params["shared_expert_number_k"] = MoE_config["shared_expert_number_k"]
    params["final_shift_step"] = MoE_config["final_shift_step"]
    params["swish_glu_scale_in"] = MoE_config["swish_glu_scale_in"]
    params["swish_glu_scale_out"] = MoE_config["swish_glu_scale_out"]
    params["softmax_scale"] = MoE_config["softmax_scale"]

    return params


def define_memory_handles(params):
    """Defines memory symbols and handles."""
    mem_handles = {}
    mem_handles["L3_Sym_Input_A"] = BingoMemSymbol("input_A")

    # Define memory Symbol
    # =========================================================================
    # 1. L3 静态数据符号挂载 (L3_Sym)
    # 物理意义：编译期确定的 C 语言全局数组，利用 M2/N2 静态展开 Tile Offset
    # =========================================================================
    assert (
        params["router_N2"] == 1
    ), "Current Router DFG implementation does not support N2 > 1 dimension tiling."
    # ------------------ Router 阶段 ------------------
    for m in range(params["router_M2"]):
        mem_handles[f"L3_Sym_Router_Input_A_tile_{m}"] = BingoMemSymbol(
            "input_A", offset=m * params["router_A_tileSize"]
        )

    for n in range(params["router_N2"]):
        mem_handles[f"L3_Sym_Router_Weight_B_tile_{n}"] = BingoMemSymbol(
            "router_B", offset=n * params["router_B_tileSize"]
        )

    # ------------------ Shared Expert 阶段 ------------------
    for m in range(params["shared_swish_glu_M2"]):
        # Gate 和 Up 共享同一个物理输入 A
        mem_handles[f"L3_Sym_Shared_Swish_Input_A_tile_{m}"] = BingoMemSymbol(
            "input_A", offset=m * params["shared_swish_glu_A_tileSize"]
        )

    for n in range(params["shared_swish_glu_N2"]):
        mem_handles[f"L3_Sym_Shared_Up_Weight_B_tile_{n}"] = BingoMemSymbol(
            "shared_experts_up_projection_B",
            offset=n * params["shared_swish_glu_B_tileSize"],
        )
        mem_handles[f"L3_Sym_Shared_Gate_Weight_B_tile_{n}"] = BingoMemSymbol(
            "shared_experts_gate_B", offset=n * params["shared_swish_glu_B_tileSize"]
        )

    for n in range(params["shared_down_projection_N2"]):
        mem_handles[f"L3_Sym_Shared_Down_Weight_B_tile_{n}"] = BingoMemSymbol(
            "shared_experts_down_projection_B",
            offset=n * params["shared_down_projection_B_tileSize"],
        )

    # ------------------ Individual Expert 阶段 ------------------
    # A 和 D 依赖 Router 动态选路，不在编译期展开 Tile，留由 L3_Alloc 动态池处理
    for n in range(params["individual_swish_glu_N2"]):
        mem_handles[f"L3_Sym_Indiv_Up_Weight_B_tile_{n}"] = BingoMemSymbol(
            "individual_experts_up_projection_B",
            offset=n * params["individual_swish_glu_B_tileSize"],
        )
        mem_handles[f"L3_Sym_Indiv_Gate_Weight_B_tile_{n}"] = BingoMemSymbol(
            "individual_experts_gate_B",
            offset=n * params["individual_swish_glu_B_tileSize"],
        )

    for n in range(params["individual_down_projection_N2"]):
        mem_handles[f"L3_Sym_Indiv_Down_Weight_B_tile_{n}"] = BingoMemSymbol(
            "individual_experts_down_projection_B",
            offset=n * params["individual_down_projection_B_tileSize"],
        )

    # ------------------ 黄金验证数据 ------------------
    mem_handles["L3_Sym_Golden_Layer_Output"] = BingoMemSymbol("layer_output")

    # =========================================================================
    # 2. L1 统一动态缓冲池 (L1_Buf)
    # 物理意义：TCDM 中的时分复用空间，根据 M2/N2 探针动态决定是否 Ping-Pong
    # =========================================================================
    # 获取全流程对 L1 SRAM 容量的最大极值需求
    max_A_l1_size = max(
        params["router_A_tileSize"],
        params["shared_swish_glu_A_tileSize"],
        params["shared_down_projection_A_tileSize"],
        params["individual_swish_glu_A_tileSize"],
        params["individual_down_projection_A_tileSize"],
    )
    max_B_l1_size = max(
        params["router_B_tileSize"],
        params["shared_swish_glu_B_tileSize"],
        params["shared_down_projection_B_tileSize"],
        params["individual_swish_glu_B_tileSize"],
        params["individual_down_projection_B_tileSize"],
    )
    max_D_l1_size = max(
        params["router_D_tileSize"],
        params["shared_swish_glu_D_tileSize"],
        params["shared_down_projection_D_tileSize"],
        params["individual_swish_glu_D_tileSize"],
        params["individual_down_projection_D_tileSize"],
    )

    # 动态架构探针
    A_needs_pingpong = (
        max(
            params["router_M2"],
            params["shared_swish_glu_M2"],
            params["shared_down_projection_M2"],
            params["individual_swish_glu_M2"],
            params["individual_down_projection_M2"],
        )
        > 1
    )
    B_needs_pingpong = (
        max(
            params["router_N2"],
            params["shared_swish_glu_N2"],
            params["shared_down_projection_N2"],
            params["individual_swish_glu_N2"],
            params["individual_down_projection_N2"],
        )
        > 1
    )
    D_needs_pingpong = A_needs_pingpong or B_needs_pingpong

    chip_id = 0
    cluster_id = 0

    if B_needs_pingpong:
        mem_handles["L1_Buf_Unified_B_ping"] = BingoMemAlloc(
            "l1_buf_B_ping",
            size=max_B_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
        mem_handles["L1_Buf_Unified_B_pong"] = BingoMemAlloc(
            "l1_buf_B_pong",
            size=max_B_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
    else:
        mem_handles["L1_Buf_Unified_B"] = BingoMemAlloc(
            "l1_buf_B",
            size=max_B_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )

    if A_needs_pingpong:
        mem_handles["L1_Buf_Unified_A_ping"] = BingoMemAlloc(
            "l1_buf_A_ping",
            size=max_A_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
        mem_handles["L1_Buf_Unified_A_pong"] = BingoMemAlloc(
            "l1_buf_A_pong",
            size=max_A_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
    else:
        mem_handles["L1_Buf_Unified_A"] = BingoMemAlloc(
            "l1_buf_A",
            size=max_A_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )

    if D_needs_pingpong:
        mem_handles["L1_Buf_Unified_D_ping"] = BingoMemAlloc(
            "l1_buf_D_ping",
            size=max_D_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
        mem_handles["L1_Buf_Unified_D_pong"] = BingoMemAlloc(
            "l1_buf_D_pong",
            size=max_D_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
    else:
        mem_handles["L1_Buf_Unified_D"] = BingoMemAlloc(
            "l1_buf_D",
            size=max_D_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )

    # =========================================================================
    # 3. L3 运行时动态中间池 (L3_Alloc)
    # 物理意义：CVA6 与硬件之间交接中间特征图 (Feature Maps) 和控制表的内存
    # =========================================================================

    # --- 控制面表格 ---
    metadata_size = (
        params["router_M1"]
        * params["meshRow"]
        * params["router_M2"]
        * params["individual_expert_number_k"]
        * 4
    )
    # store the chosen experts for all tokens
    mem_handles["L3_Alloc_Router_TopK_Indices"] = BingoMemAlloc(
        "l3_router_topk_idx", size=metadata_size, mem_level="L3"
    )
    # 2. 【新增】存选出来的原始分数 (Fashion 1: 按 Token 顺排，给后台 Softmax 用)
    mem_handles["L3_Alloc_Router_TopK_Scores"] = BingoMemAlloc(
        "l3_router_topk_scores", size=metadata_size, mem_level="L3"
    )
    # store the probability of chosen experts for all tokens
    mem_handles["L3_Alloc_Router_Probabilities"] = BingoMemAlloc(
        "l3_router_prob", size=metadata_size, mem_level="L3"
    )
    # record the tokens for each expert
    expert_num = params["expert_number_each_layer"]
    mem_handles["L3_Alloc_Expert_Histogram"] = BingoMemAlloc(
        "l3_exp_hist", size=expert_num * 4, mem_level="L3"
    )
    # record the offset in the input matrix of each expert
    # so that we can separate them with padding data after down projection
    mem_handles["L3_Alloc_Expert_Offsets"] = BingoMemAlloc(
        "l3_exp_offset", size=expert_num * 4, mem_level="L3"
    )

    # 4. 【注释升级】导航票表 (Fashion 2: 按专家物理坑位排)
    # 记录该位置的数据对应的 flat_idx (token_id * K + k_index)
    # 用于归约时精准反推数据该加到哪一行、该乘哪个概率
    mem_handles["L3_Alloc_Reverse_Index"] = BingoMemAlloc(
        "l3_rev_idx", size=metadata_size, mem_level="L3"
    )

    # --- 数据面张量池 (按各阶段的宏观满尺寸分配) ---
    router_D_total = (
        params["router_D_tileSize"] * params["router_M2"] * params["router_N2"]
    )
    mem_handles["L3_Alloc_Router_HW_Output"] = BingoMemAlloc(
        "l3_router_hw_out", size=router_D_total, mem_level="L3"
    )

    # Shared Expert 中间态
    shared_swish_D_total = (
        params["shared_expert_number_k"]  # <--- 补齐专家数
        * params["shared_swish_glu_D_tileSize"]
        * params["shared_swish_glu_M2"]
        * params["shared_swish_glu_N2"]
    )
    mem_handles["L3_Alloc_Shared_Up_HW_Output"] = BingoMemAlloc(
        "l3_shared_up_out", size=shared_swish_D_total, mem_level="L3"
    )
    mem_handles["L3_Alloc_Shared_Gate_HW_Output"] = BingoMemAlloc(
        "l3_shared_gate_out", size=shared_swish_D_total, mem_level="L3"
    )

    # [HW SiLU] Swish 浮点暂存池不再需要，SiLU 在硬件中完成

    # 这个 Buffer 既是 CVA6 算完 Swish 的落盘处，也是后续 Down 的输入起点
    shared_down_A_total = (
        params["shared_expert_number_k"]  # <--- 补齐专家数
        * params["shared_down_projection_A_tileSize"]
        * params["shared_down_projection_M2"]
    )

    mem_handles["L3_Alloc_Shared_Activated_A"] = BingoMemAlloc(
        "l3_shared_act_a", size=shared_down_A_total, mem_level="L3"
    )

    shared_down_D_total = (
        params["shared_expert_number_k"]  # <--- 补齐专家数
        * params["shared_down_projection_D_tileSize"]
        * params["shared_down_projection_M2"]
        * params["shared_down_projection_N2"]
    )

    mem_handles["L3_Alloc_Shared_Down_HW_Output"] = BingoMemAlloc(
        "l3_shared_down_out", size=shared_down_D_total, mem_level="L3"
    )

    # Individual Expert 中间态 (Scatter 池要足够包容所有专家的动态重组)
    indiv_A_total = (
        params["expert_number_each_layer"]  # <--- 修正：必须是总专家数 E，不是 K！
        * params["individual_swish_glu_A_tileSize"]
        * params["individual_swish_glu_M2"]
    )

    mem_handles["L3_Alloc_Indiv_Scatter_Pool"] = BingoMemAlloc(
        "l3_indiv_scatter_pool", size=indiv_A_total, mem_level="L3"
    )

    indiv_swish_D_total = (
        params["expert_number_each_layer"]  # <--- 补齐专家数
        * params["individual_swish_glu_D_tileSize"]
        * params["individual_swish_glu_M2"]
        * params["individual_swish_glu_N2"]
    )

    mem_handles["L3_Alloc_Indiv_Up_HW_Output"] = BingoMemAlloc(
        "l3_indiv_up_out", size=indiv_swish_D_total, mem_level="L3"
    )
    mem_handles["L3_Alloc_Indiv_Gate_HW_Output"] = BingoMemAlloc(
        "l3_indiv_gate_out", size=indiv_swish_D_total, mem_level="L3"
    )

    # [HW SiLU] Individual Swish 浮点暂存池不再需要，SiLU 在硬件中完成

    indiv_down_A_total = (
        params["expert_number_each_layer"]  # <--- 补齐专家数
        * params["individual_down_projection_A_tileSize"]
        * params["individual_down_projection_M2"]
    )

    mem_handles["L3_Alloc_Indiv_Activated_A"] = BingoMemAlloc(
        "l3_indiv_act_a", size=indiv_down_A_total, mem_level="L3"
    )

    indiv_down_D_total = (
        params["individual_down_projection_D_tileSize"]
        * params["individual_down_projection_M2"]
        * params["individual_down_projection_N2"]
        * params["expert_number_each_layer"]  # <--- 必须乘以专家总数
    )

    mem_handles["L3_Alloc_Indiv_Down_HW_Output"] = BingoMemAlloc(
        "l3_indiv_down_out", size=indiv_down_D_total, mem_level="L3"
    )

    final_output_total_size = (
        params["router_M1"]
        * params["meshRow"]
        * params["router_M2"]  # 总 Token 数
        * params["individual_down_projection_N1"]
        * params["meshCol"]
        * params["individual_down_projection_N2"]  # 输出特征维度
        * 4  # int32 或 float32，占 4 字节
    )

    # 最终输出累加池
    mem_handles["L3_Alloc_Final_MoE_Output"] = BingoMemAlloc(
        "l3_final_moe_out", size=final_output_total_size, mem_level="L3"
    )

    return mem_handles


def float_to_uint64_bits(f_val: float) -> int:
    # 先将 float 打包为 4 字节的 bytes，再解包为 32 位无符号整数
    packed_f32 = struct.unpack("<I", struct.pack("<f", float(f_val)))[0]
    return packed_f32  # 自动扩展为 Python 的长整型，传递给 C 时会作为 uint64_t 处理


def addr_with_optional_offset(mem_handle, offset: int):
    """Return the handle object for zero offset to keep allocator metadata intact."""
    if offset == 0:
        return mem_handle
    if isinstance(mem_handle, BingoMemSymbol):
        return BingoMemSymbol(mem_handle.symbol_name, offset=mem_handle.offset + offset)
    return f"{mem_handle.get_c_var_name()} + {offset}"


def is_shared_expert1_first_up_tile(se_id: int, m: int, n: int) -> bool:
    return se_id == 1 and m == 0 and n == 0


def should_mirror_node36_src(se_id: int, m: int, n: int) -> bool:
    return NODE37_DIAG_MODE in {
        "mirror_node36_src",
        "mirror_and_bypass",
    } and is_shared_expert1_first_up_tile(se_id, m, n)


def should_bypass_node37_dep(se_id: int, m: int, n: int) -> bool:
    return NODE37_DIAG_MODE in {
        "bypass_node37_dep",
        "mirror_and_bypass",
    } and is_shared_expert1_first_up_tile(se_id, m, n)


def enforce_in_order_completion_per_core(bingo_dfg: BingoDFG) -> None:
    """Serialize normal tasks per core to match HW checkout/done semantics."""
    if not INORDER_COMPLETION_CORE_IDS:
        return

    topo_order = list(nx.topological_sort(bingo_dfg))
    normal_nodes = [
        node
        for node in topo_order
        if node.node_type == "normal"
        and node.assigned_core_id in INORDER_COMPLETION_CORE_IDS
    ]
    prev_node_per_lane = {}

    for node in normal_nodes:
        lane = (
            node.assigned_chiplet_id,
            node.assigned_cluster_id,
            node.assigned_core_id,
        )
        prev_node = prev_node_per_lane.get(lane)
        if prev_node is not None:
            # 使用 has_path 而非 has_edge：若两节点间已存在传递路径则跳过，
            # 避免插入冗余直接边。冗余直接边会导致前序节点需同时向多个后继
            # 发送 dep_matrix 完成信号（fan-out），而硬件每次完成只能设置
            # 一个 DepSet 槽位，造成其中一个后继永久等待（死锁）。
            if not nx.has_path(bingo_dfg, prev_node, node):
                bingo_dfg.bingo_add_edge(prev_node, node)
        prev_node_per_lane[lane] = node


def create_dfg(params, mem_handles):
    """Creates the Bingo Data Flow Graph with nodes and dependencies."""
    # 1. Initialize DFG
    num_chiplets = 1
    num_clusters_per_chiplet = 1
    num_cores_per_cluster = 2
    is_host_as_acc = True
    chiplet_ids = [0x00]
    bingo_dfg = BingoDFG(
        num_chiplets,
        num_clusters_per_chiplet,
        num_cores_per_cluster,
        is_host_as_acc,
        chiplet_ids,
    )
    cur_chiplet_id = 0
    cur_cluster_id = 0
    gemm_core_id = 0  # Core 0 for Compute
    dma_core_id = 1  # Core 1 for Load
    host_core_id = 2  # Core 2 for Host DMA Store

    # 物理层面：内存池里到底有没有分配 Ping-Pong 空间？
    # 这取决于 define_memory_handles 的分配结果
    has_phys_A_pingpong = "L1_Buf_Unified_A_ping" in mem_handles
    has_phys_B_pingpong = "L1_Buf_Unified_B_ping" in mem_handles
    has_phys_D_pingpong = "L1_Buf_Unified_D_ping" in mem_handles

    # 推导总 Token 数量 (物理行数)
    total_tokens = params["router_M1"] * params["meshRow"] * params["router_M2"]
    scale_in_raw = float_to_uint64_bits(params["swish_glu_scale_in"])
    scale_out_raw = float_to_uint64_bits(params["swish_glu_scale_out"])

    # 2. Define Nodes
    # 2.1 静态加载 Router Weight (B 矩阵)
    l1_B_router = (
        mem_handles["L1_Buf_Unified_B_ping"]
        if has_phys_B_pingpong
        else mem_handles["L1_Buf_Unified_B"]
    )
    node_router_copy_B = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["L3_Sym_Router_Weight_B_tile_0"],
            dst_addr=l1_B_router,
            size=params["router_B_tileSize"],
        ),
    )
    bingo_dfg.bingo_add_node(node_router_copy_B)

    router_copy_A_nodes = []
    router_gemm_nodes = []
    router_copy_D_nodes = []

    for m in range(params["router_M2"]):
        if has_phys_A_pingpong and params["router_M2"] > 1:
            current_l1_A = (
                mem_handles["L1_Buf_Unified_A_ping"]
                if m % 2 == 0
                else mem_handles["L1_Buf_Unified_A_pong"]
            )
        elif has_phys_A_pingpong:
            current_l1_A = mem_handles["L1_Buf_Unified_A_ping"]  # 有空间但不翻转
        else:
            current_l1_A = mem_handles["L1_Buf_Unified_A"]  # 根本没分配 Ping-Pong

        if has_phys_D_pingpong and params["router_M2"] > 1:
            current_l1_D = (
                mem_handles["L1_Buf_Unified_D_ping"]
                if m % 2 == 0
                else mem_handles["L1_Buf_Unified_D_pong"]
            )
        elif has_phys_D_pingpong:
            current_l1_D = mem_handles["L1_Buf_Unified_D_ping"]  # 有空间但不翻转
        else:
            current_l1_D = mem_handles["L1_Buf_Unified_D"]  # 根本没分配 Ping-Pong

        # DMA 搬运 Input A
        node_copy_A = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles[f"L3_Sym_Router_Input_A_tile_{m}"],
                dst_addr=current_l1_A,
                size=params["router_A_tileSize"],
            ),
        )
        router_copy_A_nodes.append(node_copy_A)
        bingo_dfg.bingo_add_node(node_copy_A)

        # 硬件 GEMM 计算 (注意：传入的是 TCDM 中的物理尺寸 M1, K1, N1)
        if m == 0:
            node_gemm = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=gemm_core_id,
                kernel_name="__snax_bingo_kernel_gemm_full",
                kernel_args=SnaxBingoKernelGemmFullArgs(
                    input_A_addr=current_l1_A,
                    input_B_addr=l1_B_router,
                    input_C_addr=0,
                    output_D_addr=current_l1_D,
                    M=params["router_M1"],
                    K=params["router_K1"],
                    N=params["router_N1"],
                    array_shape_idx=params["array_shape"],
                    transpose_A=0,
                    transpose_B=0,
                    accumPrevC=0,
                ),
            )
        else:
            node_gemm = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=gemm_core_id,
                kernel_name="__snax_bingo_kernel_gemm_minimal",
                kernel_args=SnaxBingoKernelGemmMinimalArgs(
                    input_A_addr=current_l1_A,
                    input_B_addr=l1_B_router,
                    input_C_addr=0,
                    output_D_addr=current_l1_D,
                ),
            )
        router_gemm_nodes.append(node_gemm)
        bingo_dfg.bingo_add_node(node_gemm)

        # DMA 写回 L3 张量池
        node_copy_D = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=current_l1_D,
                dst_addr=addr_with_optional_offset(
                    mem_handles["L3_Alloc_Router_HW_Output"],
                    m * params["router_D_tileSize"],
                ),
                size=params["router_D_tileSize"],
            ),
        )
        router_copy_D_nodes.append(node_copy_D)
        bingo_dfg.bingo_add_node(node_copy_D)

    # -------------------------------------------------------------------------
    # 阶段 2: 异构控制与调度 (CVA6 CPU 处理)
    # -------------------------------------------------------------------------

    # 节点: 提取 Top-K
    node_host_router_topk = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,
        kernel_name="__host_bingo_kernel_moe_router_schedule",
        kernel_args=HostBingoKernelMoERouterScheduleArgs(
            total_tokens=total_tokens,
            hardware_output_buffer_addr=mem_handles[
                "L3_Alloc_Router_HW_Output"
            ],  # 是一个大矩阵？
            global_indices_out_addr=mem_handles["L3_Alloc_Router_TopK_Indices"],
            global_scores_out_addr=mem_handles["L3_Alloc_Router_TopK_Scores"],
            expert_number_each_layer=params["expert_number_each_layer"],
            individual_expert_number_k=params["individual_expert_number_k"],
            mesh_row=params["meshRow"],
            mesh_col=params["meshCol"],
            router_m1=params["router_M1"],
            router_n1=params["router_N1"],
        ),
    )
    bingo_dfg.bingo_add_node(node_host_router_topk)

    # 节点: 计算 Softmax
    node_host_compute_softmax = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,
        kernel_name="__host_bingo_kernel_compute_delayed_softmax",
        kernel_args=HostBingoKernelComputeDelayedSoftmaxArgs(
            global_top_k_scores_ptr_addr=mem_handles["L3_Alloc_Router_TopK_Scores"],
            global_calculated_probability_ptr_addr=mem_handles[
                "L3_Alloc_Router_Probabilities"
            ],
            actual_total_tokens=total_tokens,
            individual_expert_number_k=params["individual_expert_number_k"],
            softmax_scale_raw=float_to_uint64_bits(params["softmax_scale"]),
        ),
    )
    bingo_dfg.bingo_add_node(node_host_compute_softmax)

    # 节点: 生成 Scatter 元数据
    node_host_build_scatter_meta = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,
        kernel_name="__host_bingo_kernel_build_scatter_metadata",
        kernel_args=HostBingoKernelBuildScatterMetadataArgs(
            global_top_k_indices_addr=mem_handles["L3_Alloc_Router_TopK_Indices"],
            actual_total_tokens=total_tokens,
            expert_token_counts_addr=mem_handles["L3_Alloc_Expert_Histogram"],
            expert_memory_offsets_addr=mem_handles["L3_Alloc_Expert_Offsets"],
            reverse_original_token_flat_idx_addr=mem_handles["L3_Alloc_Reverse_Index"],
            expert_number_each_layer=params["expert_number_each_layer"],
            individual_expert_number_k=params["individual_expert_number_k"],
        ),
    )
    bingo_dfg.bingo_add_node(node_host_build_scatter_meta)

    # =========================================================================
    # 辅助变量：算出每个专家占据的权重总字节数 (用于 B 矩阵的全局地址跨步)
    # =========================================================================
    shared_gate_B_expert_stride = (
        params["shared_swish_glu_N2"] * params["shared_swish_glu_B_tileSize"]
    )
    shared_up_B_expert_stride = (
        params["shared_swish_glu_N2"] * params["shared_swish_glu_B_tileSize"]
    )
    shared_down_B_expert_stride = (
        params["shared_down_projection_N2"]
        * params["shared_down_projection_B_tileSize"]
    )

    indiv_gate_B_expert_stride = (
        params["individual_swish_glu_N2"] * params["individual_swish_glu_B_tileSize"]
    )
    indiv_up_B_expert_stride = (
        params["individual_swish_glu_N2"] * params["individual_swish_glu_B_tileSize"]
    )
    indiv_down_B_expert_stride = (
        params["individual_down_projection_N2"]
        * params["individual_down_projection_B_tileSize"]
    )

    # =========================================================================
    # 3. 阶段 3: Shared Expert 任务定义 (引入 shared_expert_number_k 循环)
    # =========================================================================
    shared_nodes = {
        "load_A": {},  # Key: (se_id, m)
        "load_gate_B": {},  # Key: (se_id, m, n)
        "gate_gemm": {},  # Key: (se_id, m, n)
        "store_gate_D": {},  # Key: (se_id, m, n)
        "load_up_B": {},  # Key: (se_id, m, n)
        "up_gemm": {},  # Key: (se_id, m, n)
        "store_up_D": {},  # Key: (se_id, m, n)
        "host_hw_silu_glu": {},  # Key: (se_id, m) — HW SiLU 输出 + GLU 合并为单节点
        "load_act_A": {},  # Key: (se_id, m)
        "load_down_B": {},  # Key: (se_id, m, n)
        "down_gemm": {},  # Key: (se_id, m, n)
        "store_down_D": {},  # Key: (se_id, m, n)
    }

    for se_id in range(params["shared_expert_number_k"]):
        for m in range(params["shared_swish_glu_M2"]):
            # Input A 只与 M 有关，每个 M 块只需要搬运一次
            if has_phys_A_pingpong and params["shared_swish_glu_M2"] > 1:
                current_l1_A = (
                    mem_handles["L1_Buf_Unified_A_ping"]
                    if m % 2 == 0
                    else mem_handles["L1_Buf_Unified_A_pong"]
                )
            elif has_phys_A_pingpong:
                current_l1_A = mem_handles["L1_Buf_Unified_A_ping"]
            else:
                current_l1_A = mem_handles["L1_Buf_Unified_A"]

            # 节点：DMA Load Shared Input A
            node_shared_load_A = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=dma_core_id,
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=mem_handles[f"L3_Sym_Shared_Swish_Input_A_tile_{m}"],
                    dst_addr=current_l1_A,
                    size=params["shared_swish_glu_A_tileSize"],
                ),
            )
            shared_nodes["load_A"][(se_id, m)] = node_shared_load_A
            bingo_dfg.bingo_add_node(node_shared_load_A)

            # ------------------ Gate 计算循环 (N2) ------------------
            for n in range(params["shared_swish_glu_N2"]):
                current_l1_B = (
                    mem_handles["L1_Buf_Unified_B_ping"]
                    if n % 2 == 0
                    else mem_handles["L1_Buf_Unified_B_pong"]
                )
                # 细粒度 D 缓冲切换：确保每一次 GEMM 都在切换 D 缓冲
                current_l1_D = (
                    mem_handles["L1_Buf_Unified_D_ping"]
                    if (m * params["shared_swish_glu_N2"] + n) % 2 == 0
                    else mem_handles["L1_Buf_Unified_D_pong"]
                )

                # 节点：DMA Load Gate Weight B
                gate_B_src_addr = BingoMemSymbol(
                    "shared_experts_gate_B",
                    offset=se_id * shared_gate_B_expert_stride
                    + n * params["shared_swish_glu_B_tileSize"],
                )
                node_shared_load_gate_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=gate_B_src_addr,
                        dst_addr=current_l1_B,
                        size=params["shared_swish_glu_B_tileSize"],
                    ),
                )
                shared_nodes["load_gate_B"][(se_id, m, n)] = node_shared_load_gate_B
                bingo_dfg.bingo_add_node(node_shared_load_gate_B)

                # 节点：GEMM Gate (silu_enable=1: 硬件在 D32 输出做 SiLU)
                node_shared_gate_gemm = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name=(
                        "__snax_bingo_kernel_gemm_full"
                        if n == 0
                        else "__snax_bingo_kernel_gemm_minimal"
                    ),
                    kernel_args=(
                        SnaxBingoKernelGemmFullArgs(
                            input_A_addr=current_l1_A,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                            M=params["shared_swish_glu_M1"],
                            K=params["shared_swish_glu_K1"],
                            N=params["shared_swish_glu_N1"],
                            array_shape_idx=params["array_shape"],
                            transpose_A=0,
                            transpose_B=0,
                            accumPrevC=0,
                            silu_enable=1,
                        )
                        if n == 0
                        else SnaxBingoKernelGemmMinimalArgs(
                            input_A_addr=current_l1_A,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                        )
                    ),
                )
                shared_nodes["gate_gemm"][(se_id, m, n)] = node_shared_gate_gemm
                bingo_dfg.bingo_add_node(node_shared_gate_gemm)

                # 节点：Host DMA Store Gate D
                gate_D_offset = (
                    se_id
                    * (
                        params["shared_swish_glu_M2"]
                        * params["shared_swish_glu_N2"]
                        * params["shared_swish_glu_D_tileSize"]
                    )
                    + (m * params["shared_swish_glu_N2"] + n)
                    * params["shared_swish_glu_D_tileSize"]
                )
                node_shared_store_gate_D = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=current_l1_D,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Shared_Gate_HW_Output"],
                            gate_D_offset,
                        ),
                        size=params["shared_swish_glu_D_tileSize"],
                    ),
                )
                shared_nodes["store_gate_D"][(se_id, m, n)] = node_shared_store_gate_D
                bingo_dfg.bingo_add_node(node_shared_store_gate_D)

            # ------------------ Up 计算循环 (N2) ------------------
            for n in range(params["shared_swish_glu_N2"]):
                current_l1_B = (
                    mem_handles["L1_Buf_Unified_B_ping"]
                    if n % 2 == 0
                    else mem_handles["L1_Buf_Unified_B_pong"]
                )
                current_l1_D = (
                    mem_handles["L1_Buf_Unified_D_ping"]
                    if (m * params["shared_swish_glu_N2"] + n) % 2 == 0
                    else mem_handles["L1_Buf_Unified_D_pong"]
                )

                # 节点：DMA Load Up Weight B
                up_weight_offset = (
                    se_id * shared_up_B_expert_stride
                    + n * params["shared_swish_glu_B_tileSize"]
                )
                if should_mirror_node36_src(se_id, m, n):
                    up_weight_offset = n * params["shared_swish_glu_B_tileSize"]
                up_B_src_addr = BingoMemSymbol(
                    "shared_experts_up_projection_B",
                    offset=up_weight_offset,
                )
                node_shared_load_up_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=up_B_src_addr,
                        dst_addr=current_l1_B,
                        size=params["shared_swish_glu_B_tileSize"],
                    ),
                )
                shared_nodes["load_up_B"][(se_id, m, n)] = node_shared_load_up_B
                bingo_dfg.bingo_add_node(node_shared_load_up_B)

                # 节点：GEMM Up
                node_shared_up_gemm = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name=(
                        "__snax_bingo_kernel_gemm_full"
                        if n == 0
                        else "__snax_bingo_kernel_gemm_minimal"
                    ),
                    kernel_args=(
                        SnaxBingoKernelGemmFullArgs(
                            input_A_addr=current_l1_A,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                            M=params["shared_swish_glu_M1"],
                            K=params["shared_swish_glu_K1"],
                            N=params["shared_swish_glu_N1"],
                            array_shape_idx=params["array_shape"],
                            transpose_A=0,
                            transpose_B=0,
                            accumPrevC=0,
                        )
                        if n == 0
                        else SnaxBingoKernelGemmMinimalArgs(
                            input_A_addr=current_l1_A,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                        )
                    ),
                )
                shared_nodes["up_gemm"][(se_id, m, n)] = node_shared_up_gemm
                bingo_dfg.bingo_add_node(node_shared_up_gemm)

                # 节点：Host DMA Store Up D
                up_D_offset = (
                    se_id
                    * (
                        params["shared_swish_glu_M2"]
                        * params["shared_swish_glu_N2"]
                        * params["shared_swish_glu_D_tileSize"]
                    )
                    + (m * params["shared_swish_glu_N2"] + n)
                    * params["shared_swish_glu_D_tileSize"]
                )
                node_shared_store_up_D = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=current_l1_D,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Shared_Up_HW_Output"],
                            up_D_offset,
                        ),
                        size=params["shared_swish_glu_D_tileSize"],
                    ),
                )
                shared_nodes["store_up_D"][(se_id, m, n)] = node_shared_store_up_D
                bingo_dfg.bingo_add_node(node_shared_store_up_D)

            # ------------------ Host 激活计算 (SwishGLU) ------------------
            valid_elements_shared = (
                params["shared_swish_glu_M1"]
                * params["shared_swish_glu_N1"]
                * params["shared_swish_glu_N2"]
                * params["meshRow"]
                * params["meshCol"]
            )
            # 一整个expert的offset
            expert_D_total_offset = se_id * (
                params["shared_swish_glu_M2"]
                * params["shared_swish_glu_N2"]
                * params["shared_swish_glu_D_tileSize"]
            )
            # 一个硬件tile的offset
            m_row_offset = (
                m
                * params["shared_swish_glu_N2"]
                * params["shared_swish_glu_D_tileSize"]
            )

            gate_out_row_addr = addr_with_optional_offset(
                mem_handles["L3_Alloc_Shared_Gate_HW_Output"],
                expert_D_total_offset + m_row_offset,
            )
            up_out_row_addr = addr_with_optional_offset(
                mem_handles["L3_Alloc_Shared_Up_HW_Output"],
                expert_D_total_offset + m_row_offset,
            )
            act_a_row_addr = addr_with_optional_offset(
                mem_handles["L3_Alloc_Shared_Activated_A"],
                se_id
                * (
                    params["shared_down_projection_M2"]
                    * params["shared_down_projection_A_tileSize"]
                )
                + m * params["shared_down_projection_A_tileSize"],
            )

            # [HW SiLU] Gate GEMM 输出已包含 SiLU 激活，直接与 Up 做 GLU 乘法
            node_shared_hw_silu_glu = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=host_core_id,
                kernel_name="__host_bingo_kernel_compute_hw_silu_glu",
                kernel_args=HostBingoKernelComputeHwSiluGluArgs(
                    gate_silu_hw_data_addr=gate_out_row_addr,
                    up_projection_hw_data_addr=up_out_row_addr,
                    activated_out_data_addr=act_a_row_addr,
                    valid_elements=valid_elements_shared,
                    swish_glu_scale_in_raw=scale_in_raw,
                    swish_glu_scale_out_raw=scale_out_raw,
                ),
            )
            shared_nodes["host_hw_silu_glu"][(se_id, m)] = node_shared_hw_silu_glu
            bingo_dfg.bingo_add_node(node_shared_hw_silu_glu)

            # ------------------ Down 计算循环 (N2) ------------------
            if has_phys_A_pingpong and params["shared_down_projection_M2"] > 1:
                current_l1_A_down = (
                    mem_handles["L1_Buf_Unified_A_ping"]
                    if m % 2 == 0
                    else mem_handles["L1_Buf_Unified_A_pong"]
                )
            elif has_phys_A_pingpong:
                current_l1_A_down = mem_handles["L1_Buf_Unified_A_ping"]
            else:
                current_l1_A_down = mem_handles["L1_Buf_Unified_A"]
            # 节点：DMA Load Activated A back to L1
            node_shared_load_act_A = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=dma_core_id,
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=act_a_row_addr,
                    dst_addr=current_l1_A_down,
                    size=params["shared_down_projection_A_tileSize"],
                ),
            )
            shared_nodes["load_act_A"][(se_id, m)] = node_shared_load_act_A
            bingo_dfg.bingo_add_node(node_shared_load_act_A)

            for n in range(params["shared_down_projection_N2"]):
                current_l1_B = (
                    mem_handles["L1_Buf_Unified_B_ping"]
                    if n % 2 == 0
                    else mem_handles["L1_Buf_Unified_B_pong"]
                )
                if (
                    has_phys_D_pingpong
                    and params["shared_down_projection_M2"]
                    * params["shared_down_projection_N2"]
                    > 1
                ):
                    current_l1_D = (
                        mem_handles["L1_Buf_Unified_D_ping"]
                        if (m * params["shared_down_projection_N2"] + n) % 2 == 0
                        else mem_handles["L1_Buf_Unified_D_pong"]
                    )
                elif has_phys_D_pingpong:
                    current_l1_D = mem_handles["L1_Buf_Unified_D_ping"]
                else:
                    current_l1_D = mem_handles["L1_Buf_Unified_D"]

                # 节点：DMA Load Down Weight B
                down_B_src_addr = BingoMemSymbol(
                    "shared_experts_down_projection_B",
                    offset=se_id * shared_down_B_expert_stride
                    + n * params["shared_down_projection_B_tileSize"],
                )
                node_shared_load_down_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=down_B_src_addr,
                        dst_addr=current_l1_B,
                        size=params["shared_down_projection_B_tileSize"],
                    ),
                )
                shared_nodes["load_down_B"][(se_id, m, n)] = node_shared_load_down_B
                bingo_dfg.bingo_add_node(node_shared_load_down_B)

                # 节点：GEMM Down
                node_shared_down_gemm = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name=(
                        "__snax_bingo_kernel_gemm_full"
                        if n == 0
                        else "__snax_bingo_kernel_gemm_minimal"
                    ),
                    kernel_args=(
                        SnaxBingoKernelGemmFullArgs(
                            input_A_addr=current_l1_A_down,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                            M=params["shared_down_projection_M1"],
                            K=params["shared_down_projection_K1"],
                            N=params["shared_down_projection_N1"],
                            array_shape_idx=params["array_shape"],
                            transpose_A=0,
                            transpose_B=0,
                            accumPrevC=0,
                        )
                        if n == 0
                        else SnaxBingoKernelGemmMinimalArgs(
                            input_A_addr=current_l1_A_down,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                        )
                    ),
                )
                shared_nodes["down_gemm"][(se_id, m, n)] = node_shared_down_gemm
                bingo_dfg.bingo_add_node(node_shared_down_gemm)

                # 节点：Host DMA Store Down D
                down_D_offset = (
                    se_id
                    * (
                        params["shared_down_projection_M2"]
                        * params["shared_down_projection_N2"]
                        * params["shared_down_projection_D_tileSize"]
                    )
                    + (m * params["shared_down_projection_N2"] + n)
                    * params["shared_down_projection_D_tileSize"]
                )
                node_shared_store_down_D = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=current_l1_D,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Shared_Down_HW_Output"],
                            down_D_offset,
                        ),
                        size=params["shared_down_projection_D_tileSize"],
                    ),
                )
                shared_nodes["store_down_D"][(se_id, m, n)] = node_shared_store_down_D
                bingo_dfg.bingo_add_node(node_shared_store_down_D)

    # =========================================================================
    # 4. 阶段 4: Individual Expert 任务定义 (引入 expert_number_each_layer 循环)
    # =========================================================================
    indiv_nodes = {
        "scatter_pad": {},  # Key: (expert_id,) — 直接 L3→L1 scatter
        "load_gate_B": {},  # Key: (expert_id, m, n)
        "gate_gemm": {},  # Key: (expert_id, m, n)
        "store_gate_D": {},  # Key: (expert_id, m, n)
        "load_up_B": {},  # Key: (expert_id, m, n)
        "up_gemm": {},  # Key: (expert_id, m, n)
        "store_up_D": {},  # Key: (expert_id, m, n)
        "host_hw_silu_glu": {},  # Key: (expert_id, m) — HW SiLU 输出 + GLU 合并为单节点
        "load_act_A": {},  # Key: (expert_id, m)
        "load_down_B": {},  # Key: (expert_id, m, n)
        "down_gemm": {},  # Key: (expert_id, m, n)
        "store_down_D": {},  # Key: (expert_id, m, n)
    }

    # Track individual expert GEMM full node IDs for dynamic M patching
    gemm_full_node_ids = {}  # {expert_id: {"gate": nid, "up": nid, "down": nid}}

    for expert_id in range(params["expert_number_each_layer"]):

        gemm_full_node_ids[expert_id] = {}

        # 定义单个专家输入所需的绝对行数 (M1 * meshRow * M2)
        max_tokens_per_expert = (
            params["individual_swish_glu_M1"]
            * params["meshRow"]
            * params["individual_swish_glu_M2"]
        )

        # 确定 L1 A 缓冲区地址 (m=0 的 tile, 用于 scatter 直接写入)
        if has_phys_A_pingpong and params["individual_swish_glu_M2"] > 1:
            scatter_l1_A = mem_handles["L1_Buf_Unified_A_ping"]
        elif has_phys_A_pingpong:
            scatter_l1_A = mem_handles["L1_Buf_Unified_A_ping"]
        else:
            scatter_l1_A = mem_handles["L1_Buf_Unified_A"]

        # 节点：Host scatter — 直接从 L3 全局输入通过 SoC DMA 搬运到 L1
        # 消除原有的 L3 scatter pool 中间跳: L3_input → L3_pool → L1
        # 现在: L3_input → L1 (一跳完成)
        node_indiv_scatter = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            kernel_name="__host_bingo_kernel_scatter_pad_dynamic",
            kernel_args=HostBingoKernelScatterAndPadArgs(
                expert_id=expert_id,
                global_input_A_addr=mem_handles["L3_Sym_Input_A"],
                padded_scatter_pool_addr=scatter_l1_A,  # 直接目标: L1 buffer
                expert_token_counts_addr=mem_handles["L3_Alloc_Expert_Histogram"],
                expert_memory_offsets_addr=mem_handles["L3_Alloc_Expert_Offsets"],
                reverse_original_token_flat_idx_addr=mem_handles[
                    "L3_Alloc_Reverse_Index"
                ],
                input_dimension=params["input_dimension"],
                max_padded_tokens=max_tokens_per_expert,
                individual_expert_number_k=params["individual_expert_number_k"],
            ),
        )
        indiv_nodes["scatter_pad"][(expert_id,)] = node_indiv_scatter
        bingo_dfg.bingo_add_node(node_indiv_scatter)

        # load_A 已被消除: scatter_pad 已通过 SoC DMA 直接将数据写入 L1,
        # 无需再通过 cluster DMA 从 L3 scatter pool 搬到 L1

        for m in range(params["individual_swish_glu_M2"]):
            if has_phys_A_pingpong and params["individual_swish_glu_M2"] > 1:
                current_l1_A = (
                    mem_handles["L1_Buf_Unified_A_ping"]
                    if m % 2 == 0
                    else mem_handles["L1_Buf_Unified_A_pong"]
                )
            elif has_phys_A_pingpong:
                current_l1_A = mem_handles["L1_Buf_Unified_A_ping"]
            else:
                current_l1_A = mem_handles["L1_Buf_Unified_A"]
            # load_A 已被消除: scatter_pad 已将数据直接写入 L1

            # ------------------ Gate 计算循环 (N2) ------------------
            for n in range(params["individual_swish_glu_N2"]):
                current_l1_B = (
                    mem_handles["L1_Buf_Unified_B_ping"]
                    if n % 2 == 0
                    else mem_handles["L1_Buf_Unified_B_pong"]
                )
                current_l1_D = (
                    mem_handles["L1_Buf_Unified_D_ping"]
                    if (m * params["individual_swish_glu_N2"] + n) % 2 == 0
                    else mem_handles["L1_Buf_Unified_D_pong"]
                )

                # 节点：DMA Load Gate Weight B
                gate_B_src_addr = BingoMemSymbol(
                    "individual_experts_gate_B",
                    offset=expert_id * indiv_gate_B_expert_stride
                    + n * params["individual_swish_glu_B_tileSize"],
                )
                node_indiv_load_gate_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=gate_B_src_addr,
                        dst_addr=current_l1_B,
                        size=params["individual_swish_glu_B_tileSize"],
                    ),
                )
                indiv_nodes["load_gate_B"][(expert_id, m, n)] = node_indiv_load_gate_B
                bingo_dfg.bingo_add_node(node_indiv_load_gate_B)

                # 节点：GEMM Gate (silu_enable=1: 硬件在 D32 输出做 SiLU)
                node_indiv_gate_gemm = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name=(
                        "__snax_bingo_kernel_gemm_full"
                        if n == 0
                        else "__snax_bingo_kernel_gemm_minimal"
                    ),
                    kernel_args=(
                        SnaxBingoKernelGemmFullArgs(
                            input_A_addr=current_l1_A,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                            M=params["individual_swish_glu_M1"],
                            K=params["individual_swish_glu_K1"],
                            N=params["individual_swish_glu_N1"],
                            array_shape_idx=params["array_shape"],
                            transpose_A=0,
                            transpose_B=0,
                            accumPrevC=0,
                            silu_enable=1,
                        )
                        if n == 0
                        else SnaxBingoKernelGemmMinimalArgs(
                            input_A_addr=current_l1_A,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                        )
                    ),
                )
                indiv_nodes["gate_gemm"][(expert_id, m, n)] = node_indiv_gate_gemm
                bingo_dfg.bingo_add_node(node_indiv_gate_gemm)
                if n == 0 and m == 0:
                    gemm_full_node_ids[expert_id]["gate"] = node_indiv_gate_gemm.node_id

                # 节点：Host DMA Store Gate D
                gate_D_offset = (
                    expert_id
                    * (
                        params["individual_swish_glu_M2"]
                        * params["individual_swish_glu_N2"]
                        * params["individual_swish_glu_D_tileSize"]
                    )
                    + (m * params["individual_swish_glu_N2"] + n)
                    * params["individual_swish_glu_D_tileSize"]
                )
                node_indiv_store_gate_D = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=current_l1_D,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Indiv_Gate_HW_Output"],
                            gate_D_offset,
                        ),
                        size=params["individual_swish_glu_D_tileSize"],
                    ),
                )
                indiv_nodes["store_gate_D"][(expert_id, m, n)] = node_indiv_store_gate_D
                bingo_dfg.bingo_add_node(node_indiv_store_gate_D)

            # ------------------ Up 计算循环 (N2) ------------------
            for n in range(params["individual_swish_glu_N2"]):
                current_l1_B = (
                    mem_handles["L1_Buf_Unified_B_ping"]
                    if n % 2 == 0
                    else mem_handles["L1_Buf_Unified_B_pong"]
                )
                current_l1_D = (
                    mem_handles["L1_Buf_Unified_D_ping"]
                    if (m * params["individual_swish_glu_N2"] + n) % 2 == 0
                    else mem_handles["L1_Buf_Unified_D_pong"]
                )

                # 节点：DMA Load Up Weight B
                up_B_src_addr = BingoMemSymbol(
                    "individual_experts_up_projection_B",
                    offset=expert_id * indiv_up_B_expert_stride
                    + n * params["individual_swish_glu_B_tileSize"],
                )
                node_indiv_load_up_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=up_B_src_addr,
                        dst_addr=current_l1_B,
                        size=params["individual_swish_glu_B_tileSize"],
                    ),
                )
                indiv_nodes["load_up_B"][(expert_id, m, n)] = node_indiv_load_up_B
                bingo_dfg.bingo_add_node(node_indiv_load_up_B)

                # 节点：GEMM Up
                # n=0: 使用 gemm_minimal_silu（比 gemm_full 少~45个CSR写入）
                #   - gate[n=0] 已完整配置了 M/K/N/strides/versacore_csr
                #   - gate 与 up 的 M 相同（处理同一批 tokens），K=128, N=1 也相同
                #   - 唯一需要更新的是：B_addr, D_addr, silu_enable(0 vs 1)
                # n>0: 使用 gemm_minimal（addresses only，shape 与 n=0 相同）
                node_indiv_up_gemm = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name=(
                        "__snax_bingo_kernel_gemm_minimal_silu"
                        if n == 0
                        else "__snax_bingo_kernel_gemm_minimal"
                    ),
                    kernel_args=(
                        SnaxBingoKernelGemmMinimalSiluArgs(
                            input_A_addr=current_l1_A,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                            silu_enable=0,
                        )
                        if n == 0
                        else SnaxBingoKernelGemmMinimalArgs(
                            input_A_addr=current_l1_A,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                        )
                    ),
                )
                indiv_nodes["up_gemm"][(expert_id, m, n)] = node_indiv_up_gemm
                bingo_dfg.bingo_add_node(node_indiv_up_gemm)
                # up[n=0] 改用 gemm_minimal_silu，不再（也无需）patch M：
                # gate[n=0] 的 gemm_full 已将 M 写入 Versacore 的 temporal-bound CSR，
                # 由于 gate 与 up 处理相同数量的 tokens，该 M 值对 up 同样有效。
                # if n == 0 and m == 0: ← 注意：此行故意注释掉，不再加入 gemm_full_node_ids

                # 节点：Host DMA Store Up D
                up_D_offset = (
                    expert_id
                    * (
                        params["individual_swish_glu_M2"]
                        * params["individual_swish_glu_N2"]
                        * params["individual_swish_glu_D_tileSize"]
                    )
                    + (m * params["individual_swish_glu_N2"] + n)
                    * params["individual_swish_glu_D_tileSize"]
                )
                node_indiv_store_up_D = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=current_l1_D,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Indiv_Up_HW_Output"],
                            up_D_offset,
                        ),
                        size=params["individual_swish_glu_D_tileSize"],
                    ),
                )
                indiv_nodes["store_up_D"][(expert_id, m, n)] = node_indiv_store_up_D
                bingo_dfg.bingo_add_node(node_indiv_store_up_D)

            # ------------------ Host 激活计算 (SwishGLU) ------------------
            valid_elements_indiv = (
                params["individual_swish_glu_M1"]
                * params["individual_swish_glu_N1"]
                * params["individual_swish_glu_N2"]
                * params["meshRow"]
                * params["meshCol"]
            )
            indiv_expert_D_total_offset = expert_id * (
                params["individual_swish_glu_M2"]
                * params["individual_swish_glu_N2"]
                * params["individual_swish_glu_D_tileSize"]
            )
            indiv_m_row_offset = (
                m
                * params["individual_swish_glu_N2"]
                * params["individual_swish_glu_D_tileSize"]
            )

            gate_out_row_addr = addr_with_optional_offset(
                mem_handles["L3_Alloc_Indiv_Gate_HW_Output"],
                indiv_expert_D_total_offset + indiv_m_row_offset,
            )
            up_out_row_addr = addr_with_optional_offset(
                mem_handles["L3_Alloc_Indiv_Up_HW_Output"],
                indiv_expert_D_total_offset + indiv_m_row_offset,
            )
            act_a_row_addr = addr_with_optional_offset(
                mem_handles["L3_Alloc_Indiv_Activated_A"],
                expert_id
                * (
                    params["individual_down_projection_M2"]
                    * params["individual_down_projection_A_tileSize"]
                )
                + m * params["individual_down_projection_A_tileSize"],
            )

            # [HW SiLU] Gate GEMM 输出已包含 SiLU 激活，直接与 Up 做 GLU 乘法
            node_indiv_hw_silu_glu = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=host_core_id,
                kernel_name="__host_bingo_kernel_compute_hw_silu_glu",
                kernel_args=HostBingoKernelComputeHwSiluGluArgs(
                    gate_silu_hw_data_addr=gate_out_row_addr,
                    up_projection_hw_data_addr=up_out_row_addr,
                    activated_out_data_addr=act_a_row_addr,
                    valid_elements=valid_elements_indiv,
                    swish_glu_scale_in_raw=scale_in_raw,
                    swish_glu_scale_out_raw=scale_out_raw,
                ),
            )
            indiv_nodes["host_hw_silu_glu"][(expert_id, m)] = node_indiv_hw_silu_glu
            bingo_dfg.bingo_add_node(node_indiv_hw_silu_glu)

            # ------------------ Down 计算循环 (N2) ------------------
            if has_phys_A_pingpong and params["individual_down_projection_M2"] > 1:
                current_l1_A_down = (
                    mem_handles["L1_Buf_Unified_A_ping"]
                    if m % 2 == 0
                    else mem_handles["L1_Buf_Unified_A_pong"]
                )
            elif has_phys_A_pingpong:
                current_l1_A_down = mem_handles["L1_Buf_Unified_A_ping"]
            else:
                current_l1_A_down = mem_handles["L1_Buf_Unified_A"]

            # 节点：DMA Load Activated A back to L1
            node_indiv_load_act_A = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=dma_core_id,
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=act_a_row_addr,
                    dst_addr=current_l1_A_down,
                    size=params["individual_down_projection_A_tileSize"],
                ),
            )
            indiv_nodes["load_act_A"][(expert_id, m)] = node_indiv_load_act_A
            bingo_dfg.bingo_add_node(node_indiv_load_act_A)

            for n in range(params["individual_down_projection_N2"]):
                current_l1_B = (
                    mem_handles["L1_Buf_Unified_B_ping"]
                    if n % 2 == 0
                    else mem_handles["L1_Buf_Unified_B_pong"]
                )
                if (
                    has_phys_D_pingpong
                    and params["individual_down_projection_M2"]
                    * params["individual_down_projection_N2"]
                    > 1
                ):
                    current_l1_D = (
                        mem_handles["L1_Buf_Unified_D_ping"]
                        if (m * params["individual_down_projection_N2"] + n) % 2 == 0
                        else mem_handles["L1_Buf_Unified_D_pong"]
                    )
                elif has_phys_D_pingpong:
                    current_l1_D = mem_handles["L1_Buf_Unified_D_ping"]
                else:
                    current_l1_D = mem_handles["L1_Buf_Unified_D"]

                # 节点：DMA Load Down Weight B
                down_B_src_addr = BingoMemSymbol(
                    "individual_experts_down_projection_B",
                    offset=expert_id * indiv_down_B_expert_stride
                    + n * params["individual_down_projection_B_tileSize"],
                )
                node_indiv_load_down_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=down_B_src_addr,
                        dst_addr=current_l1_B,
                        size=params["individual_down_projection_B_tileSize"],
                    ),
                )
                indiv_nodes["load_down_B"][(expert_id, m, n)] = node_indiv_load_down_B
                bingo_dfg.bingo_add_node(node_indiv_load_down_B)

                # 节点：GEMM Down
                node_indiv_down_gemm = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name=(
                        "__snax_bingo_kernel_gemm_full"
                        if n == 0
                        else "__snax_bingo_kernel_gemm_minimal"
                    ),
                    kernel_args=(
                        SnaxBingoKernelGemmFullArgs(
                            input_A_addr=current_l1_A_down,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                            M=params["individual_down_projection_M1"],
                            K=params["individual_down_projection_K1"],
                            N=params["individual_down_projection_N1"],
                            array_shape_idx=params["array_shape"],
                            transpose_A=0,
                            transpose_B=0,
                            accumPrevC=0,
                        )
                        if n == 0
                        else SnaxBingoKernelGemmMinimalArgs(
                            input_A_addr=current_l1_A_down,
                            input_B_addr=current_l1_B,
                            input_C_addr=0,
                            output_D_addr=current_l1_D,
                        )
                    ),
                )
                indiv_nodes["down_gemm"][(expert_id, m, n)] = node_indiv_down_gemm
                bingo_dfg.bingo_add_node(node_indiv_down_gemm)
                if n == 0 and m == 0:
                    gemm_full_node_ids[expert_id]["down"] = node_indiv_down_gemm.node_id

                # 节点：Host DMA Store Down D
                down_D_offset = (
                    expert_id
                    * (
                        params["individual_down_projection_M2"]
                        * params["individual_down_projection_N2"]
                        * params["individual_down_projection_D_tileSize"]
                    )
                    + (m * params["individual_down_projection_N2"] + n)
                    * params["individual_down_projection_D_tileSize"]
                )
                node_indiv_store_down_D = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=current_l1_D,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Indiv_Down_HW_Output"],
                            down_D_offset,
                        ),
                        size=params["individual_down_projection_D_tileSize"],
                    ),
                )
                indiv_nodes["store_down_D"][(expert_id, m, n)] = node_indiv_store_down_D
                bingo_dfg.bingo_add_node(node_indiv_store_down_D)
    # =========================================================================
    # 5. 阶段 5: Final Accumulate (归约与加权) — Prefill Tiled Version
    # =========================================================================
    # per_tile_cols: output tile width = N1 × meshRow × meshCol
    per_tile_cols = (
        params["individual_down_projection_N1"] * params["meshRow"] * params["meshCol"]
    )
    max_tokens_per_expert = (
        params["individual_swish_glu_M1"]
        * params["meshRow"]
        * params["individual_swish_glu_M2"]
    )

    node_accumulate = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,
        kernel_name="__host_bingo_kernel_prefill_accumulate",
        kernel_args=HostBingoKernelPrefillAccumulateArgs(
            shared_expert_hw_output_addr=mem_handles["L3_Alloc_Shared_Down_HW_Output"],
            individual_experts_hw_output_addr=mem_handles[
                "L3_Alloc_Indiv_Down_HW_Output"
            ],
            reverse_original_token_flat_idx_addr=mem_handles["L3_Alloc_Reverse_Index"],
            global_calculated_probability_addr=mem_handles[
                "L3_Alloc_Router_Probabilities"
            ],
            expert_token_counts_addr=mem_handles["L3_Alloc_Expert_Histogram"],
            expert_memory_offsets_addr=mem_handles["L3_Alloc_Expert_Offsets"],
            final_layer_output_addr=mem_handles["L3_Alloc_Final_MoE_Output"],
            actual_total_tokens=total_tokens,
            output_N2=params["individual_down_projection_N2"],
            per_tile_cols=per_tile_cols,
            max_tokens_per_expert=max_tokens_per_expert,
            expert_number_each_layer_val=params["expert_number_each_layer"],
            shared_expert_number_k_val=params["shared_expert_number_k"],
            individual_expert_number_k_val=params["individual_expert_number_k"],
            softmax_scale_raw=float_to_uint64_bits(params["softmax_scale"]),
            final_shift_step_val=params.get("final_shift_step", 24),
        ),
    )
    bingo_dfg.bingo_add_node(node_accumulate)

    # define dependency
    # 1. router
    # Keep Router startup single-rooted: B must be loaded before first A tile launch.
    bingo_dfg.bingo_add_edge(node_router_copy_B, router_copy_A_nodes[0])
    bingo_dfg.bingo_add_edge(node_router_copy_B, router_gemm_nodes[0])
    for m in range(params["router_M2"]):
        bingo_dfg.bingo_add_edge(router_copy_A_nodes[m], router_gemm_nodes[m])
        bingo_dfg.bingo_add_edge(router_gemm_nodes[m], router_copy_D_nodes[m])

        if m < params["router_M2"] - 1:
            bingo_dfg.bingo_add_edge(router_copy_A_nodes[m], router_copy_A_nodes[m + 1])
            bingo_dfg.bingo_add_edge(router_gemm_nodes[m], router_gemm_nodes[m + 1])
            bingo_dfg.bingo_add_edge(router_copy_D_nodes[m], router_copy_D_nodes[m + 1])

            bingo_dfg.bingo_add_edge(router_copy_A_nodes[m + 1], router_copy_D_nodes[m])

        if m < params["router_M2"] - 2:
            bingo_dfg.bingo_add_edge(router_gemm_nodes[m], router_copy_A_nodes[m + 2])
            bingo_dfg.bingo_add_edge(router_copy_D_nodes[m], router_gemm_nodes[m + 2])

        if m < params["router_M2"] - 3:
            bingo_dfg.bingo_add_edge(router_copy_D_nodes[m], router_copy_A_nodes[m + 3])

    bingo_dfg.bingo_add_edge(router_copy_D_nodes[-1], node_host_router_topk)
    bingo_dfg.bingo_add_edge(node_host_router_topk, node_host_build_scatter_meta)
    bingo_dfg.bingo_add_edge(node_host_build_scatter_meta, node_host_compute_softmax)

    # Shared 阶段入口门控：仅第一个 shared expert 的 A tile 由 softmax 启动。
    # Gate/Up 的首个 B tile 统一通过阶段内数据流接力，避免把 host 入口
    # 直接扇出到多个 DMA 首节点，触发首块 descriptor 的双前驱膨胀。
    bingo_dfg.bingo_add_edge(node_host_compute_softmax, shared_nodes["load_A"][(0, 0)])

    # Shared expert 串行：后一个 expert 依赖前一个 expert 的 down projection 完成
    last_m_shared = params["shared_swish_glu_M2"] - 1
    last_n_shared = params["shared_down_projection_N2"] - 1
    for se_id in range(1, params["shared_expert_number_k"]):
        bingo_dfg.bingo_add_edge(
            shared_nodes["store_down_D"][(se_id - 1, last_m_shared, last_n_shared)],
            shared_nodes["load_A"][(se_id, 0)],
        )

    # 2. share expert
    # =========================================================================
    # 定义依赖关系: Shared Expert 模块 (统一向前看 Router 风格)
    # =========================================================================

    M2 = params["shared_swish_glu_M2"]
    N2 = params["shared_swish_glu_N2"]
    N2_down = params["shared_down_projection_N2"]

    for se_id in range(params["shared_expert_number_k"]):
        for m in range(M2):
            idx_m = (se_id, m)

            # -----------------------------------------------------------------
            # 1. Gate 计算循环 (N 维度流水)
            # -----------------------------------------------------------------
            for n in range(N2):
                idx_mn = (se_id, m, n)

                # A. 基础数据流 (RAW)
                if n == 0:
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_A"][idx_m], shared_nodes["gate_gemm"][idx_mn]
                    )
                    # Gate B first tile should not run ahead of its corresponding A tile readiness.
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_A"][idx_m],
                        shared_nodes["load_gate_B"][idx_mn],
                    )

                bingo_dfg.bingo_add_edge(
                    shared_nodes["load_gate_B"][idx_mn],
                    shared_nodes["gate_gemm"][idx_mn],
                )
                bingo_dfg.bingo_add_edge(
                    shared_nodes["gate_gemm"][idx_mn],
                    shared_nodes["store_gate_D"][idx_mn],
                )

                # B. 同核心保序与流水交错 (n < N2 - 1)
                if n < N2 - 1:
                    next_mn = (se_id, m, n + 1)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_gate_B"][idx_mn],
                        shared_nodes["load_gate_B"][next_mn],
                    )

                # C. 缓冲防御 (WAR) (n < N2 - 2)
                if n < N2 - 2:
                    next2_mn = (se_id, m, n + 2)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["gate_gemm"][idx_mn],
                        shared_nodes["load_gate_B"][next2_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_gate_D"][idx_mn],
                        shared_nodes["gate_gemm"][next2_mn],
                    )

                # D. 预取限制 (n < N2 - 3)
                if n < N2 - 3:
                    next3_mn = (se_id, m, n + 3)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_gate_D"][idx_mn],
                        shared_nodes["load_gate_B"][next3_mn],
                    )

            # -----------------------------------------------------------------
            # 2. 阶段切换屏障 (Gate 结束 -> Up 启动)
            # -----------------------------------------------------------------
            bingo_dfg.bingo_add_edge(
                shared_nodes["store_gate_D"][(se_id, m, N2 - 1)],
                shared_nodes["load_up_B"][(se_id, m, 0)],
            )
            # Make the gate->up handoff wait for the last gate GEMM consumer on
            # core0 as well. This avoids relying on a host-only transition token
            # while keeping load_up_B free of same-core duplicate predecessors.
            bingo_dfg.bingo_add_edge(
                shared_nodes["gate_gemm"][(se_id, m, N2 - 1)],
                shared_nodes["load_up_B"][(se_id, m, 0)],
            )
            # -----------------------------------------------------------------
            # 3. Up 计算循环 (N 维度流水, 逻辑完全同 Gate)
            # -----------------------------------------------------------------
            for n in range(N2):
                idx_mn = (se_id, m, n)

                if not should_bypass_node37_dep(se_id, m, n):
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_up_B"][idx_mn],
                        shared_nodes["up_gemm"][idx_mn],
                    )
                bingo_dfg.bingo_add_edge(
                    shared_nodes["up_gemm"][idx_mn], shared_nodes["store_up_D"][idx_mn]
                )

                if n < N2 - 1:
                    next_mn = (se_id, m, n + 1)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_up_B"][idx_mn],
                        shared_nodes["load_up_B"][next_mn],
                    )

                if n < N2 - 2:
                    next2_mn = (se_id, m, n + 2)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["up_gemm"][idx_mn],
                        shared_nodes["load_up_B"][next2_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_up_D"][idx_mn],
                        shared_nodes["up_gemm"][next2_mn],
                    )

                if n < N2 - 3:
                    next3_mn = (se_id, m, n + 3)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_up_D"][idx_mn],
                        shared_nodes["load_up_B"][next3_mn],
                    )

            # -----------------------------------------------------------------
            # 4. 阶段切换屏障 (Up 结束 -> Host 激活 -> Act_A 回填 L1)
            # -----------------------------------------------------------------
            # [HW SiLU] Swish 已在硬件中完成，只需 store_up_D → hw_silu_glu → load_act_A
            bingo_dfg.bingo_add_edge(
                shared_nodes["store_up_D"][(se_id, m, N2 - 1)],
                shared_nodes["host_hw_silu_glu"][idx_m],
            )
            bingo_dfg.bingo_add_edge(
                shared_nodes["host_hw_silu_glu"][idx_m],
                shared_nodes["load_act_A"][idx_m],
            )

            # [关键保护] 防止 Act_A 覆盖正被 Up_GEMM 读取的 L1_A
            bingo_dfg.bingo_add_edge(
                shared_nodes["up_gemm"][(se_id, m, N2 - 1)],
                shared_nodes["load_act_A"][idx_m],
            )

            # -----------------------------------------------------------------
            # 5. Down 计算循环 (N 维度流水)
            # -----------------------------------------------------------------
            for n in range(N2_down):
                idx_mn = (se_id, m, n)

                if n == 0:
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_act_A"][idx_m],
                        shared_nodes["down_gemm"][idx_mn],
                    )
                    # Down B first tile should wait for activated A tile to avoid root DMA bursts.
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_act_A"][idx_m],
                        shared_nodes["load_down_B"][idx_mn],
                    )
                bingo_dfg.bingo_add_edge(
                    shared_nodes["load_down_B"][idx_mn],
                    shared_nodes["down_gemm"][idx_mn],
                )
                bingo_dfg.bingo_add_edge(
                    shared_nodes["down_gemm"][idx_mn],
                    shared_nodes["store_down_D"][idx_mn],
                )

                if n < N2_down - 1:
                    next_mn = (se_id, m, n + 1)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_down_B"][idx_mn],
                        shared_nodes["load_down_B"][next_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["down_gemm"][idx_mn],
                        shared_nodes["down_gemm"][next_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_down_D"][idx_mn],
                        shared_nodes["store_down_D"][next_mn],
                    )

                if n < N2_down - 2:
                    next2_mn = (se_id, m, n + 2)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["down_gemm"][idx_mn],
                        shared_nodes["load_down_B"][next2_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_down_D"][idx_mn],
                        shared_nodes["down_gemm"][next2_mn],
                    )

                if n < N2_down - 3:
                    next3_mn = (se_id, m, n + 3)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_down_D"][idx_mn],
                        shared_nodes["load_down_B"][next3_mn],
                    )

            # -----------------------------------------------------------------
            # 6. M 维度的向前看防御 (M-level Forward-Looking WAR)
            # -----------------------------------------------------------------
            if m < M2 - 1:
                # 下一个 M 块的 Load A，必须等当前 M 块的最后一块 Down GEMM 彻底让出 L1_A 缓冲
                bingo_dfg.bingo_add_edge(
                    shared_nodes["down_gemm"][(se_id, m, N2_down - 1)],
                    shared_nodes["load_A"][(se_id, m + 1)],
                )

            # NOTE: Do not add m+2 look-ahead dependency on shared load_A.
            # It may create dual-consumer dependency patterns in dep_matrix and
            # block the following gemm task from becoming ready.

    # 3. individual expert
    # ==================================================================================
    E = params["expert_number_each_layer"]
    M2 = params["individual_swish_glu_M2"]
    N2_gate = params["individual_swish_glu_N2"]
    N2_down = params["individual_down_projection_N2"]

    # [全局入口] Shared 严格串行后，只需最后一个 Shared Expert 收工来启动 Individual。
    last_shared_expert = params["shared_expert_number_k"] - 1
    last_m_shared = params["shared_swish_glu_M2"] - 1
    last_n_shared = params["shared_down_projection_N2"] - 1
    last_shared_store_down_D = shared_nodes["store_down_D"][
        (last_shared_expert, last_m_shared, last_n_shared)
    ]
    bingo_dfg.bingo_add_edge(
        last_shared_store_down_D,
        indiv_nodes["scatter_pad"][(0,)],
    )

    # [元数据入口] build_scatter_meta → scatter_pad[(0,)] 这条直接边已被移除。
    # 原因：build_scatter_meta 已通过 topk→build_scatter_meta→softmax→shared_load_A→
    # ...→last_shared_store_down_D→scatter_pad 这条串行路径，先于 scatter_pad 完成。
    # 若再加一条 build_scatter_meta→scatter_pad 的直接边，scatter_pad 有两个 Core2
    # 前驱（build_scatter_meta 和 last_shared_store_down_D），触发 dummy_check 插入
    # WAIT_NO_SIG dummy 节点，从而消耗 dep_matrix 中的单比特令牌，造成死锁。
    for expert_id in range(E):

        # ---------------------------------------------------------------------
        # 0. 启动与 E 维度前向防御 (Inter-Expert Defense)
        # ---------------------------------------------------------------------
        # scatter 完成后启动该专家的 load_gate_B (load_A 已被消除, scatter 直接写入 L1)
        # 仅保留单扇出: scatter_pad(Core2) → load_gate_B(Core1): dep_matrix[1][2]
        # gate_gemm 通过 scatter_pad → load_gate_B → gate_gemm 传递路径保序
        # 令牌流: Core2 → Core1 → Core0 (与 advance 版本一致, 无 Core2→Core0 直接信号)

        if expert_id < E - 1:
            # [L1_A WAR 防御 + dep_matrix 令牌保护]
            # 下一个专家的 scatter 直接写 L1_A 缓冲, 必须等当前专家的
            # 最后一块 store_down_D 完成 (而非 down_gemm)。
            #
            # 关键修正: 原先使用 down_gemm → scatter_pad 边 (Core0→Core2),
            # 但 down_gemm 同时还有 store_down_D 后继 (也在 Core2)。
            # 这导致 down_gemm 产生两个 Core0→Core2 DummySet 节点,
            # 由于两者之间无拓扑约束, NetworkX 的拓扑排序可能交换其顺序,
            # 使 store_down_D 消费了错误的令牌 (本应属于 scatter_pad 的),
            # 从而在 down_gemm 完成前就执行 store_down_D, 导致数据不一致。
            #
            # 修正方案: 改用 store_down_D → scatter_pad (Core2→Core2 同核边),
            # 消除 Core0 → Core2 的双 DummySet 歧义。store_down_D 本身
            # 已通过 dep_check 确认 down_gemm 完成, 所以 L1_A WAR 仍然成立。
            # 同时, store_down_D 严格在 host_hw_silu_glu 之后执行 (pipeline 保序:
            #   hw_silu_glu → load_act_A → down_gemm → store_down_D),
            # 因此隐含了原 host_hw_silu_glu → scatter_pad 边的令牌窃取防护:
            # 到 store_down_D 完成时, Core1 已消耗了 host_hw_silu_glu 的所有信号。
            bingo_dfg.bingo_add_edge(
                indiv_nodes["store_down_D"][(expert_id, M2 - 1, N2_down - 1)],
                indiv_nodes["scatter_pad"][(expert_id + 1,)],
            )

        for m in range(M2):
            idx_m = (expert_id, m)

            # -----------------------------------------------------------------
            # 1. Gate 计算循环 (N 维度流水)
            # -----------------------------------------------------------------
            for n in range(N2_gate):
                idx_mn = (expert_id, m, n)

                # A. 基础数据流 (RAW)
                if n == 0:
                    # scatter_pad(Core2) 已将 A 数据直接 DMA 到 L1
                    # 仅保留 scatter_pad → load_gate_B (Core2→Core1) 单扇出边。
                    # gate_gemm 通过传递路径 scatter_pad → load_gate_B → gate_gemm
                    # 隐式保序, 不需要 scatter_pad → gate_gemm 直接边。
                    # 移除该直接边消除了 scatter_pad 的 Core2→Core0 信号,
                    # 使 dep_matrix 令牌流与 advance 版本一致 (Core2→Core1→Core0)。
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["scatter_pad"][(expert_id,)],
                        indiv_nodes["load_gate_B"][idx_mn],
                    )
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["load_gate_B"][idx_mn], indiv_nodes["gate_gemm"][idx_mn]
                )
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["gate_gemm"][idx_mn],
                    indiv_nodes["store_gate_D"][idx_mn],
                )

                # B. 严格串行保序 (n < N2 - 1)
                # 原流水优化（load_gate_B[n]→load_gate_B[n+1]）会导致 load_gate_B[n+1]
                # 写 B_pong 与 gate_gemm[n] 读 B_ping 并发，而 B_ping/B_pong 的 TCDM bank
                # 完全重叠（偏移 65536B = 8192 * 8B，mod 64 banks = 0），在某些调度时序
                # 下（如 Expert1 中两任务被同时 pop）会导致 SNAX streamer 挂起。
                # 改为 store_gate_D[n]→load_gate_B[n+1]：确保 n+1 的 B 矩阵加载在
                # gate_gemm[n] 及其 store 完全结束后才启动，消除 TCDM bank 冲突。
                # gate_gemm[n+1] 的正确顺序通过
                #   store_gate_D[n] → load_gate_B[n+1] → gate_gemm[n+1] 传递保证，
                # 不再需要单独的 gate_gemm[n]→gate_gemm[n+1] 或
                # store_gate_D[n]→store_gate_D[n+1] 边。
                if n < N2_gate - 1:
                    next_mn = (expert_id, m, n + 1)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["store_gate_D"][idx_mn],
                        indiv_nodes["load_gate_B"][next_mn],
                    )

                # C/D 流水控制边已随 B 分支重构一并覆盖，故不再单独添加。

            # -----------------------------------------------------------------
            # 2. 阶段切换屏障 (Gate 结束 -> Up 启动)
            # -----------------------------------------------------------------
            # 使用 store_gate_D[n=0]（而非 n=N2_gate-1）作为 load_up_B[n=0] 的触发前驱。
            # 原因：当 N2_gate >= 2 时，store_gate_D[N2_gate-1] → load_up_B[0] 及
            #   gate_gemm[N2_gate-1] → load_up_B[0] 都会形成有向环（因为 load_up_B[0]
            #   通过 up loop 最终会到达 gate_gemm/store_gate_D[N2_gate-1]），所以这
            #   两条边实际上从未被有效地添加到 DFG。
            # 改为通过 store_gate_D[n=0] 屏障，保证 load_up_B[n=0] 在 gate_gemm[n=0]
            # （及其紧随的 host DMA store）完成后才启动，消除并发 SNAX + iDMA 同时
            # 访问相同 TCDM bank 组的竞态条件，解决 Expert1 gate_gemm 挂起问题。
            # 对 N2_gate=1 时等同于原始行为（store_gate_D[0] == store_gate_D[N2_gate-1]）。
            bingo_dfg.bingo_add_edge(
                indiv_nodes["store_gate_D"][(expert_id, m, 0)],
                indiv_nodes["load_up_B"][(expert_id, m, 0)],
            )

            # -----------------------------------------------------------------
            # 3. Up 计算循环 (N 维度流水, 逻辑完全同 Gate)
            # -----------------------------------------------------------------
            for n in range(N2_gate):
                idx_mn = (expert_id, m, n)

                # scatter_pad→up_gemm[n=0] 这条直接边不需要。
                # up_gemm[n=0] 通过传递路径 scatter_pad→gate_gemm→load_up_B[n=0]→up_gemm[n=0]
                # 已隐式保序，直接边冗余。保留该边会使 scatter_pad 扇出到
                # gate_gemm/load_gate_B/up_gemm 三个跨核目标，
                # minicompiler 需插入多层 dummy 节点消耗 dep_matrix token slot，
                # 在 NODE_TIMING 开启时 slot 耗尽导致死锁。
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["load_up_B"][idx_mn], indiv_nodes["up_gemm"][idx_mn]
                )
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["up_gemm"][idx_mn], indiv_nodes["store_up_D"][idx_mn]
                )

                # B. 严格串行保序 (n < N2 - 1)
                # 与 gate loop 的修复完全相同：
                # 原流水 load_up_B[n] → load_up_B[n+1] 会使 load_up_B[n+1] 写
                # up_B_pong 与 up_gemm[n] 读 up_B_ping 并发，而
                # up_B_ping/up_B_pong 偏移同样是 65536 B（= 8192 × 8B，mod 64 banks
                # = 0），Expert1 中两任务被同时 pop → SNAX streamer 死锁。
                # 改为 store_up_D[n] → load_up_B[n+1]：确保 n+1 的 B 矩阵加载在
                # up_gemm[n] 及其 store 完全结束后才启动，消除 bank 冲突。
                # C/D 流水控制边随 B 分支重构一并覆盖，故不再单独添加。
                if n < N2_gate - 1:
                    next_mn = (expert_id, m, n + 1)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["store_up_D"][idx_mn],
                        indiv_nodes["load_up_B"][next_mn],
                    )

            # -----------------------------------------------------------------
            # 4. 阶段切换屏障 (Up 结束 -> Host 激活 -> Act_A 回填 L1)
            # -----------------------------------------------------------------
            # [HW SiLU] Swish 已在硬件中完成，只需 store_up_D → hw_silu_glu → load_act_A
            bingo_dfg.bingo_add_edge(
                indiv_nodes["store_up_D"][(expert_id, m, N2_gate - 1)],
                indiv_nodes["host_hw_silu_glu"][idx_m],
            )
            bingo_dfg.bingo_add_edge(
                indiv_nodes["host_hw_silu_glu"][idx_m], indiv_nodes["load_act_A"][idx_m]
            )

            # [WAR 防御] 防止 Act_A 覆写正被 Up_GEMM 读的 A_ping/pong
            bingo_dfg.bingo_add_edge(
                indiv_nodes["up_gemm"][(expert_id, m, N2_gate - 1)],
                indiv_nodes["load_act_A"][idx_m],
            )

            # -----------------------------------------------------------------
            # 5. Down 计算循环 (N 维度流水)
            # -----------------------------------------------------------------
            for n in range(N2_down):
                idx_mn = (expert_id, m, n)

                if n == 0:
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["load_act_A"][idx_m],
                        indiv_nodes["down_gemm"][idx_mn],
                    )
                    # Down B first tile should wait for activated A tile to avoid root DMA bursts.
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["load_act_A"][idx_m],
                        indiv_nodes["load_down_B"][idx_mn],
                    )
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["load_down_B"][idx_mn], indiv_nodes["down_gemm"][idx_mn]
                )
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["down_gemm"][idx_mn],
                    indiv_nodes["store_down_D"][idx_mn],
                )

                if n < N2_down - 1:
                    next_mn = (expert_id, m, n + 1)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["load_down_B"][idx_mn],
                        indiv_nodes["load_down_B"][next_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["down_gemm"][idx_mn],
                        indiv_nodes["down_gemm"][next_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["store_down_D"][idx_mn],
                        indiv_nodes["store_down_D"][next_mn],
                    )

                if n < N2_down - 2:
                    next2_mn = (expert_id, m, n + 2)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["down_gemm"][idx_mn],
                        indiv_nodes["load_down_B"][next2_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["store_down_D"][idx_mn],
                        indiv_nodes["down_gemm"][next2_mn],
                    )

                if n < N2_down - 3:
                    next3_mn = (expert_id, m, n + 3)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["store_down_D"][idx_mn],
                        indiv_nodes["load_down_B"][next3_mn],
                    )

            # -----------------------------------------------------------------
            # 6. M 维度前向防御 (M-level Forward-Looking WAR)
            # -----------------------------------------------------------------
            # [已移除] 直接 L3→L1 scatter 架构下, scatter_pad 为 per-expert 单次
            # 节点 (非 per-M-tile), load_A 已被消除; 当 M2=1 时无 M 维度前向依赖。
            # 若未来 M2>1, 需重新设计 scatter 分片与 L1_A 互斥保护。

    # =========================================================================
    # 5. 全局同步屏障：Accumulation (最终结果归约)
    # =========================================================================
    # 依赖条件：等待最后一个 Individual Expert 收工（即全部任务的最末端节点）。
    # 以下冗余边已被移除，以防止 node_accumulate 出现多个 Core2 前驱：
    #   - node_host_compute_softmax → node_accumulate
    #     （softmax 通过 softmax→shared_load_A→...→last_shared_store_down_D→scatter_pad
    #       →...→last_indiv_store_down_D→accumulate 路径先于 accumulate，冗余）
    #   - shared_nodes["store_down_D"][*] → node_accumulate
    #     （shared expert 串行完成后才启动 individual，indiv 必然更晚完成，冗余）
    #   - indiv_nodes["store_down_D"][(expert_id<E-1), *] → node_accumulate
    #     （individual expert 串行执行，expert E-1 必然最晚，其他均冗余）
    # 保留多个 Core2 前驱会触发 bingo_transform_dfg_add_dummy_check_nodes 插入
    # WAIT_NO_SIG dummy 节点，消耗 dep_matrix 单比特令牌，造成 token aliasing 死锁。
    last_indiv_expert = params["expert_number_each_layer"] - 1
    last_m_indiv = params["individual_swish_glu_M2"] - 1
    last_n_indiv = params["individual_down_projection_N2"] - 1
    bingo_dfg.bingo_add_edge(
        indiv_nodes["store_down_D"][(last_indiv_expert, last_m_indiv, last_n_indiv)],
        node_accumulate,
    )

    enforce_in_order_completion_per_core(bingo_dfg)

    return bingo_dfg, gemm_full_node_ids, indiv_nodes, node_accumulate


def main():
    args = get_args()
    output_dir = args.output_dir
    output_file_name = args.output_offload_file_name
    print(f"Output DIR: {output_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merged_config = load_workload_config(args)
    params = define_workload_params(**merged_config)

    # =========================================================================
    # Build full DFG (Router + Shared + all Experts + Accumulate)
    # Multi-phase: each phase dispatched dynamically via bingo_hw_scheduler_reinit
    # =========================================================================
    mem_handles = define_memory_handles(params)
    bingo_dfg, gemm_full_node_ids, indiv_nodes, node_accumulate = create_dfg(
        params, mem_handles
    )

    enforce_in_order_completion_per_core(bingo_dfg)

    output_path = os.path.join(output_dir, output_file_name)

    # =========================================================================
    # DARTS CERF: Mark individual expert nodes for conditional execution.
    # Each individual expert's task nodes get cond_exec_en=1 with group_id
    # equal to the expert_id (0..N-1). Router, shared experts, and accumulate
    # nodes keep cond_exec_en=0 (always execute).
    # At runtime, host writes CERF bitmask to activate/skip expert groups.
    # Default CERF=0xFFFFFFFF means all experts active.
    # =========================================================================
    for category, nodes_dict in indiv_nodes.items():
        for key, node in nodes_dict.items():
            expert_id = key[0]  # First element of tuple key is always expert_id
            node.cond_exec_en = True
            node.cond_exec_group_id = expert_id
            node.cond_exec_invert = False

    # Single-phase compilation: all experts in ONE scheduler call.
    # Eliminates all inter-expert phase transitions (Exit/Entry/Dummy).
    # The full DFG already has correct dependency edges:
    #   Router → Shared → scatter_pad[E0] → E0_work → scatter_pad[E1] → ... → Accumulate
    # bingo_compile_dfg() auto-applies: entry, exit, dummy_set/check, dep assignment.
    data_header_name = "single_cluster_MoE_prefill_data.h"
    bingo_dfg.bingo_compile_dfg(
        "single_cluster_MoE_prefill",
        output_dir,
        output_file_name,
        extra_include_header_list=[data_header_name, "MoE_operator.h"],
    )

    # Post-process: inject MOE_OPERATOR_CUSTOM define and GEMM M patch table
    inject_dynamic_patch_table(output_path, gemm_full_node_ids)


if __name__ == "__main__":
    main()
