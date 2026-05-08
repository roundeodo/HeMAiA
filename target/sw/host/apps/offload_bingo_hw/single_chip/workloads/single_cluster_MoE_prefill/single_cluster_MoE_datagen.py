#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add data utility path
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/")
)
from data_utils import (
    format_scalar_definition,
    format_vector_definition,
    format_scalar_define,
    format_vector_define,
)  # noqa E402

# # Add golden model path
from snax_utils import block_gemm_golden_model  # noqa E402

np.random.seed(320)


def log_progress(message):
    print(f"[single_cluster_MoE_datagen] {message}", file=sys.stderr, flush=True)


def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += ["#include <stddef.h>"]
    emit_str += emit_MoE_data(**kwargs)
    return "\n\n".join(emit_str)


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


def format_ptr_array(type_name, array_name, names_list):
    elements = ", ".join([f"(uintptr_t){name}" for name in names_list])
    return f"{type_name} {array_name}[] = {{ {elements} }};"


def find_top_k_golden_model(scores_2D, expert_number_k):
    top_k_indices = np.argsort(scores_2D, axis=1)[:, ::-1][:, :expert_number_k]
    return top_k_indices.astype(np.uint16)


def softmax_golden_model(raw_scores_2D, expert_number_k, softmax_scale):
    M_total = raw_scores_2D.shape[0]
    top_k_indices = find_top_k_golden_model(raw_scores_2D, expert_number_k)
    top_k_probability = np.zeros((M_total, expert_number_k), dtype=np.uint32)
    for m in range(M_total):
        selected_scores = raw_scores_2D[m, top_k_indices[m]]
        max_score = np.max(selected_scores)
        exp_values = np.exp(selected_scores - max_score).astype(np.float32)
        exp_sum = np.sum(exp_values)

        probability = ((exp_values / exp_sum) * softmax_scale).astype(np.uint32)
        top_k_probability[m] = probability
    return top_k_indices, top_k_probability


def swish_glu_golden_model(gate_data, up_data, swish_glu_scale_in, swish_glu_scale_out):
    gate_data_float = gate_data.astype(np.float32) * swish_glu_scale_in
    up_data_float = up_data.astype(np.float32) * swish_glu_scale_in
    sigmoid_gate_data_float = 1.0 / (1.0 + np.exp(-gate_data_float).astype(np.float32))
    swish_gate_data = sigmoid_gate_data_float * gate_data_float

    result_float = swish_gate_data * up_data_float

    result_int32 = (result_float * swish_glu_scale_out).astype(np.int32)

    activated_data = np.clip(result_int32, -128, 127).astype(np.int8)
    return activated_data


def emit_MoE_data(**kwargs):
    MoE_config = parse_header_config(os.path.join(CURRENT_DIR, "MoE_common_variable.h"))
    data_str = []
    full_golden_data = kwargs.get("full_golden_data", False)
    router_M1 = kwargs["router_M1"]
    router_K1 = kwargs["router_K1"]
    router_N1 = kwargs["router_N1"]
    router_M2 = kwargs["router_M2"]
    router_K2 = kwargs["router_K2"]
    router_N2 = kwargs["router_N2"]

    shared_swish_glu_M1 = kwargs["shared_swish_glu_M1"]
    shared_swish_glu_K1 = kwargs["shared_swish_glu_K1"]
    shared_swish_glu_N1 = kwargs["shared_swish_glu_N1"]
    shared_swish_glu_M2 = kwargs["shared_swish_glu_M2"]
    shared_swish_glu_K2 = kwargs["shared_swish_glu_K2"]
    shared_swish_glu_N2 = kwargs["shared_swish_glu_N2"]

    shared_down_projection_M1 = kwargs["shared_down_projection_M1"]
    shared_down_projection_K1 = kwargs["shared_down_projection_K1"]
    shared_down_projection_N1 = kwargs["shared_down_projection_N1"]
    shared_down_projection_M2 = kwargs["shared_down_projection_M2"]
    shared_down_projection_K2 = kwargs["shared_down_projection_K2"]
    shared_down_projection_N2 = kwargs["shared_down_projection_N2"]

    individual_swish_glu_M1 = kwargs["individual_swish_glu_M1"]
    individual_swish_glu_K1 = kwargs["individual_swish_glu_K1"]
    individual_swish_glu_N1 = kwargs["individual_swish_glu_N1"]
    individual_swish_glu_M2 = kwargs["individual_swish_glu_M2"]
    individual_swish_glu_K2 = kwargs["individual_swish_glu_K2"]
    individual_swish_glu_N2 = kwargs["individual_swish_glu_N2"]

    individual_down_projection_M1 = kwargs["individual_down_projection_M1"]
    individual_down_projection_K1 = kwargs["individual_down_projection_K1"]
    individual_down_projection_N1 = kwargs["individual_down_projection_N1"]
    individual_down_projection_M2 = kwargs["individual_down_projection_M2"]
    individual_down_projection_K2 = kwargs["individual_down_projection_K2"]
    individual_down_projection_N2 = kwargs["individual_down_projection_N2"]

    # add the params of shared experts
    data_str += [
        format_scalar_definition("uint32_t", "shared_swish_glu_M1", shared_swish_glu_M1)
    ]
    data_str += [
        format_scalar_definition("uint32_t", "shared_swish_glu_K1", shared_swish_glu_K1)
    ]
    data_str += [
        format_scalar_definition("uint32_t", "shared_swish_glu_N1", shared_swish_glu_N1)
    ]
    data_str += [
        format_scalar_definition("uint32_t", "shared_swish_glu_M2", shared_swish_glu_M2)
    ]
    data_str += [
        format_scalar_definition("uint32_t", "shared_swish_glu_K2", shared_swish_glu_K2)
    ]
    data_str += [
        format_scalar_definition("uint32_t", "shared_swish_glu_N2", shared_swish_glu_N2)
    ]

    data_str += [
        format_scalar_definition(
            "uint32_t", "shared_down_projection_M1", shared_down_projection_M1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "shared_down_projection_K1", shared_down_projection_K1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "shared_down_projection_N1", shared_down_projection_N1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "shared_down_projection_M2", shared_down_projection_M2
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "shared_down_projection_K2", shared_down_projection_K2
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "shared_down_projection_N2", shared_down_projection_N2
        )
    ]

    # add the params of individual experts
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_swish_glu_M1", individual_swish_glu_M1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_swish_glu_K1", individual_swish_glu_K1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_swish_glu_N1", individual_swish_glu_N1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_swish_glu_M2", individual_swish_glu_M2
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_swish_glu_K2", individual_swish_glu_K2
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_swish_glu_N2", individual_swish_glu_N2
        )
    ]

    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_down_projection_M1", individual_down_projection_M1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_down_projection_K1", individual_down_projection_K1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_down_projection_N1", individual_down_projection_N1
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_down_projection_M2", individual_down_projection_M2
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_down_projection_K2", individual_down_projection_K2
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "individual_down_projection_N2", individual_down_projection_N2
        )
    ]

    array_shape = kwargs["array_shape"]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]

    data_type = 0  # int8 data type
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        0
    ]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        1
    ]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        2
    ]
    M_total = router_M2 * router_M1 * meshRow
    N_total_shared_expert_swish = shared_swish_glu_N1 * shared_swish_glu_N2 * meshCol
    N_total_shared_expert_down_projection = (
        shared_down_projection_N1 * shared_down_projection_N2 * meshCol
    )
    N_total_out = (
        individual_down_projection_N1 * individual_down_projection_N2 * meshCol
    )
    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]

    # transposed_A = kwargs["transposed_A"]
    # data_str += [format_scalar_definition("uint32_t", "transposed_A", transposed_A)]

    # transposed_B = kwargs["transposed_B"]
    # data_str += [format_scalar_definition("uint32_t", "transposed_B", transposed_B)]

    data_str += [
        format_scalar_definition("uint32_t", "addNonZeroC", kwargs["addNonZeroC"])
    ]
    data_str += [format_scalar_definition("uint32_t", "addZeroC", kwargs["addZeroC"])]
    data_str += [
        format_scalar_definition("uint32_t", "accumPrevC", kwargs["accumPrevC"])
    ]
    assert (
        sum([kwargs["addNonZeroC"], kwargs["addZeroC"], kwargs["accumPrevC"]]) == 1
    ), "Only one of addNonZeroC, addZeroC, accumPrevC can be set to 1."

    if kwargs["accumPrevC"] == 1:
        all_inner_dims_are_one = all(
            dim == 1
            for dim in (
                router_M1,
                router_N1,
                shared_swish_glu_M1,
                shared_swish_glu_N1,
                shared_down_projection_M1,
                shared_down_projection_N1,
                individual_swish_glu_M1,
                individual_swish_glu_N1,
                individual_down_projection_M1,
                individual_down_projection_N1,
            )
        )
        assert (
            all_inner_dims_are_one
        ), "When accumPrevC=1, all stage-local M1/N1 dimensions must be 1."

    log_progress("generating input tensor A and weight tensors B")

    # test data generation
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127
    C_MIN, C_MAX = -2147483648, 2147483647

    # generate input A
    # =========================================================================
    # 1. 修正输入 A 的生成 (提取 M2)
    # =========================================================================
    # 假设 K 不分块 (K2=1)。A 是按 M2 切块搬运的，所以 M2 必须是最外层！
    input_A_tiled = np.random.randint(
        A_MIN,
        A_MAX,
        size=(router_M2, router_M1, router_K2 * router_K1, meshRow, tileSize),
        dtype=np.int8,
    )
    input_A = input_A_tiled.reshape(-1)  # 展平后，就是按 M2 顺序排好的 1D DMA 可读数据
    # 假设输入特征维度统一为 K_total
    K_total = router_K2 * router_K1 * tileSize
    M_total = router_M2 * router_M1 * meshRow

    # 将 5D 的物理存储结构还原为 [M_total, K_total] 的逻辑 2D 矩阵
    # 物理 shape: (M2, M1, K, meshRow, tileSize)
    # 转换逻辑：将 M 相关维度合并，K 相关维度合并
    input_A_matrix = input_A_tiled.transpose(0, 1, 3, 2, 4).reshape(M_total, K_total)
    # padding 逻辑保持不变...
    length = input_A.size
    pad_len = (-length) % 64
    if pad_len > 0:
        A_padded = np.pad(input_A, (0, pad_len), mode="constant", constant_values=0)
    else:
        A_padded = input_A
    data_str += [format_vector_definition("int8_t", "input_A", A_padded)]

    # generate router B
    # B 是按 N2 切块搬运的，所以 N2 必须在 K 维度之前（或者说在最外层）！
    router_B_tiled = np.random.randint(
        B_MIN,
        B_MAX,
        size=(
            router_N2,
            router_K2 * router_K1,
            router_N1,
            tileSize,
            meshCol,
        ),  # 修改N和K的顺序？
        dtype=np.int8,
    )
    router_B = router_B_tiled.reshape(-1)
    data_str += [format_vector_definition("int8_t", "router_B", router_B)]

    ## shared expert weight generation
    # generate up projection B
    shared_experts_up_projection_B_tiled = np.random.randint(
        B_MIN,
        B_MAX,
        size=(
            MoE_config["shared_expert_number_k"],
            shared_swish_glu_N2,
            shared_swish_glu_K2 * shared_swish_glu_K1,
            shared_swish_glu_N1,
            tileSize,
            meshCol,
        ),
        dtype=np.int8,
    )
    shared_experts_up_projection_B = shared_experts_up_projection_B_tiled.reshape(-1)
    data_str += [
        format_vector_definition(
            "int8_t", "shared_experts_up_projection_B", shared_experts_up_projection_B
        )
    ]
    # gate B
    shared_experts_gate_B_tiled = np.random.randint(
        B_MIN,
        B_MAX,
        size=(
            MoE_config["shared_expert_number_k"],
            shared_swish_glu_N2,
            shared_swish_glu_K2 * shared_swish_glu_K1,
            shared_swish_glu_N1,
            tileSize,
            meshCol,
        ),
        dtype=np.int8,
    )
    shared_experts_gate_B = shared_experts_gate_B_tiled.reshape(-1)
    data_str += [
        format_vector_definition(
            "int8_t", "shared_experts_gate_B", shared_experts_gate_B
        )
    ]

    # down projection B
    shared_experts_down_projection_B_tiled = np.random.randint(
        B_MIN,
        B_MAX,
        size=(
            MoE_config["shared_expert_number_k"],
            shared_down_projection_N2,
            shared_down_projection_K2 * shared_down_projection_K1,
            shared_down_projection_N1,
            tileSize,
            meshCol,
        ),
        dtype=np.int8,
    )
    shared_experts_down_projection_B = shared_experts_down_projection_B_tiled.reshape(
        -1
    )
    data_str += [
        format_vector_definition(
            "int8_t",
            "shared_experts_down_projection_B",
            shared_experts_down_projection_B,
        )
    ]

    ## individual expert weight generation
    # generate up projection B
    individual_experts_up_projection_B_tiled = np.random.randint(
        B_MIN,
        B_MAX,
        size=(
            MoE_config["expert_number_each_layer"],
            individual_swish_glu_N2,
            individual_swish_glu_K2 * individual_swish_glu_K1,
            individual_swish_glu_N1,
            tileSize,
            meshCol,
        ),
        dtype=np.int8,
    )
    individual_experts_up_projection_B = (
        individual_experts_up_projection_B_tiled.reshape(-1)
    )
    data_str += [
        format_vector_definition(
            "int8_t",
            "individual_experts_up_projection_B",
            individual_experts_up_projection_B,
        )
    ]

    # generate gate B
    individual_experts_gate_B_tiled = np.random.randint(
        B_MIN,
        B_MAX,
        size=(
            MoE_config["expert_number_each_layer"],
            individual_swish_glu_N2,
            individual_swish_glu_K2 * individual_swish_glu_K1,
            individual_swish_glu_N1,
            tileSize,
            meshCol,
        ),
        dtype=np.int8,
    )
    individual_experts_gate_B = individual_experts_gate_B_tiled.reshape(-1)
    data_str += [
        format_vector_definition(
            "int8_t", "individual_experts_gate_B", individual_experts_gate_B
        )
    ]

    # down projection B
    individual_experts_down_projection_B_tiled = np.random.randint(
        B_MIN,
        B_MAX,
        size=(
            MoE_config["expert_number_each_layer"],
            individual_down_projection_N2,
            individual_down_projection_K2 * individual_down_projection_K1,
            individual_down_projection_N1,
            tileSize,
            meshCol,
        ),
        dtype=np.int8,
    )
    individual_experts_down_projection_B = (
        individual_experts_down_projection_B_tiled.reshape(-1)
    )
    data_str += [
        format_vector_definition(
            "int8_t",
            "individual_experts_down_projection_B",
            individual_experts_down_projection_B,
        )
    ]

    ## C generation
    if kwargs["addNonZeroC"] == 1:
        router_C_tiled = np.random.randint(
            C_MIN,
            C_MAX,
            size=(router_M2, router_N2, router_M1, router_N1, meshRow, meshCol),
            dtype=np.int32,
        )
        router_C = router_C_tiled.reshape(-1)
        data_str += [format_vector_definition("int32_t", "router_C", router_C)]

        shared_experts_up_projection_C_tiled = np.random.randint(
            C_MIN,
            C_MAX,
            size=(
                shared_swish_glu_M2,
                shared_swish_glu_N2,
                shared_swish_glu_M1,
                shared_swish_glu_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        shared_experts_up_projection_C = shared_experts_up_projection_C_tiled.reshape(
            -1
        )
        data_str += [
            format_vector_definition(
                "int32_t",
                "shared_experts_up_projection_C",
                shared_experts_up_projection_C,
            )
        ]

        shared_experts_gate_C_tiled = np.random.randint(
            C_MIN,
            C_MAX,
            size=(
                shared_swish_glu_M2,
                shared_swish_glu_N2,
                shared_swish_glu_M1,
                shared_swish_glu_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        shared_experts_gate_C = shared_experts_gate_C_tiled.reshape(-1)
        data_str += [
            format_vector_definition(
                "int32_t", "shared_experts_gate_C", shared_experts_gate_C
            )
        ]

        shared_experts_down_projection_C_tiled = np.random.randint(
            C_MIN,
            C_MAX,
            size=(
                shared_down_projection_M2,
                shared_down_projection_N2,
                shared_down_projection_M1,
                shared_down_projection_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        shared_experts_down_projection_C = (
            shared_experts_down_projection_C_tiled.reshape(-1)
        )
        data_str += [
            format_vector_definition(
                "int32_t",
                "shared_experts_down_projection_C",
                shared_experts_down_projection_C,
            )
        ]

        individual_experts_up_projection_C_tiled = np.random.randint(
            C_MIN,
            C_MAX,
            size=(
                individual_swish_glu_M2,
                individual_swish_glu_N2,
                individual_swish_glu_M1,
                individual_swish_glu_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        individual_experts_up_projection_C = (
            individual_experts_up_projection_C_tiled.reshape(-1)
        )
        data_str += [
            format_vector_definition(
                "int32_t",
                "individual_experts_up_projection_C",
                individual_experts_up_projection_C,
            )
        ]

        individual_experts_gate_C_tiled = np.random.randint(
            C_MIN,
            C_MAX,
            size=(
                individual_swish_glu_M2,
                individual_swish_glu_N2,
                individual_swish_glu_M1,
                individual_swish_glu_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        individual_experts_gate_C = individual_experts_gate_C_tiled.reshape(-1)
        data_str += [
            format_vector_definition(
                "int32_t", "individual_experts_gate_C", individual_experts_gate_C
            )
        ]

        individual_experts_down_projection_C_tiled = np.random.randint(
            C_MIN,
            C_MAX,
            size=(
                individual_down_projection_M2,
                individual_down_projection_N2,
                individual_down_projection_M1,
                individual_down_projection_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        individual_experts_down_projection_C = (
            individual_experts_down_projection_C_tiled.reshape(-1)
        )
        data_str += [
            format_vector_definition(
                "int32_t",
                "individual_experts_down_projection_C",
                individual_experts_down_projection_C,
            )
        ]

    else:  # use accumPrevC (Zero initialized)
        router_C_tiled = np.zeros(
            (router_M2, router_N2, router_M1, router_N1, meshRow, meshCol),
            dtype=np.int32,
        )
        shared_experts_up_projection_C_tiled = np.zeros(
            (
                shared_swish_glu_M2,
                shared_swish_glu_N2,
                shared_swish_glu_M1,
                shared_swish_glu_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        shared_experts_gate_C_tiled = np.zeros(
            (
                shared_swish_glu_M2,
                shared_swish_glu_N2,
                shared_swish_glu_M1,
                shared_swish_glu_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        shared_experts_down_projection_C_tiled = np.zeros(
            (
                shared_down_projection_M2,
                shared_down_projection_N2,
                shared_down_projection_M1,
                shared_down_projection_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        individual_experts_up_projection_C_tiled = np.zeros(
            (
                individual_swish_glu_M2,
                individual_swish_glu_N2,
                individual_swish_glu_M1,
                individual_swish_glu_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        individual_experts_gate_C_tiled = np.zeros(
            (
                individual_swish_glu_M2,
                individual_swish_glu_N2,
                individual_swish_glu_M1,
                individual_swish_glu_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )
        individual_experts_down_projection_C_tiled = np.zeros(
            (
                individual_down_projection_M2,
                individual_down_projection_N2,
                individual_down_projection_M1,
                individual_down_projection_N1,
                meshRow,
                meshCol,
            ),
            dtype=np.int32,
        )

        data_str += [
            format_scalar_definition("int32_t *", "router_C", "(int32_t *)NULL")
        ]
        data_str += [
            format_scalar_definition(
                "int32_t *", "shared_experts_up_projection_C", "(int32_t *)NULL"
            )
        ]
        data_str += [
            format_scalar_definition(
                "int32_t *", "shared_experts_gate_C", "(int32_t *)NULL"
            )
        ]
        data_str += [
            format_scalar_definition(
                "int32_t *", "shared_experts_down_projection_C", "(int32_t *)NULL"
            )
        ]
        data_str += [
            format_scalar_definition(
                "int32_t *", "individual_experts_up_projection_C", "(int32_t *)NULL"
            )
        ]
        data_str += [
            format_scalar_definition(
                "int32_t *", "individual_experts_gate_C", "(int32_t *)NULL"
            )
        ]
        data_str += [
            format_scalar_definition(
                "int32_t *", "individual_experts_down_projection_C", "(int32_t *)NULL"
            )
        ]

    if not full_golden_data:
        log_progress(
            "skipping intermediate golden tensors and emitting placeholder layer_output"
        )
        data_str += [
            format_vector_definition(
                "int8_t",
                "layer_output",
                np.zeros(1, dtype=np.int8),
            )
        ]
        return data_str

    subtraction_a = 0
    subtraction_b = 0

    ## golden model generation
    log_progress("generating full golden tensors")
    ############# routing ###############
    # router D
    router_D_tiles = []
    for m in range(router_M2):
        tile_A = input_A_tiled[m].reshape(-1)
        for n in range(router_N2):
            tile_B = router_B_tiled[n].reshape(-1)
            tile_C = (
                router_C_tiled[m, n].reshape(-1)
                if kwargs["addNonZeroC"] == 1
                else np.zeros(
                    (router_M1, router_N1, meshRow, meshCol), dtype=np.int32
                ).reshape(-1)
            )

            tile_D = block_gemm_golden_model(
                router_M1,
                router_K1 * router_K2,
                router_N1,
                meshRow,
                tileSize,
                meshCol,
                tile_A,
                tile_B,
                subtraction_a,
                subtraction_b,
                tile_C,
            )
            router_D_tiles.append(tile_D)

    router_D_1D = np.concatenate(router_D_tiles)

    # Un-tile to Global 2D Matrix for Softmax
    router_D_tiled_6D = router_D_1D.reshape(
        router_M2, router_N2, router_M1, router_N1, meshRow, meshCol
    )
    router_D_matrix = router_D_tiled_6D.transpose(0, 2, 1, 3, 4, 5).reshape(
        router_M1 * router_M2 * meshRow, -1
    )

    top_k_indices, top_k_probability = softmax_golden_model(
        router_D_matrix,
        MoE_config["individual_expert_number_k"],
        MoE_config["softmax_scale"],
    )

    unique_selected_experts = np.unique(top_k_indices)
    expert_to_tokens = {expert_id: [] for expert_id in unique_selected_experts}
    token_expert_probability_map = {}
    M_total = router_M1 * router_M2 * meshRow
    for m in range(M_total):
        for k_idx in range(MoE_config["individual_expert_number_k"]):
            expert_id = top_k_indices[m, k_idx]
            expert_to_tokens[expert_id].append(m)
            token_expert_probability_map[(m, expert_id)] = top_k_probability[m, k_idx]

    ############# activating and down projection ##############
    all_experts_down_projection_D_2d = []

    ######### shared experts #########
    total_shared_experts_up_projection_D = []
    total_shared_experts_gate_D = []
    total_shared_experts_activated_A = []
    total_shared_experts_down_projection_D = []

    act_A_chunk_size_shared = (
        shared_swish_glu_N2
        * shared_swish_glu_M1
        * shared_swish_glu_N1
        * meshRow
        * meshCol
    )

    for shared_expert_index in range(MoE_config["shared_expert_number_k"]):
        expert_up_D_tiles = []
        expert_gate_D_tiles = []

        for m in range(shared_swish_glu_M2):
            # Shared A is directly input_A
            tile_A = input_A_tiled[m].reshape(-1)
            for n in range(shared_swish_glu_N2):
                tile_up_B = shared_experts_up_projection_B_tiled[
                    shared_expert_index, n
                ].reshape(-1)
                tile_up_C = (
                    shared_experts_up_projection_C_tiled[m, n].reshape(-1)
                    if kwargs["addNonZeroC"] == 1
                    else np.zeros(
                        (shared_swish_glu_M1, shared_swish_glu_N1, meshRow, meshCol),
                        dtype=np.int32,
                    ).reshape(-1)
                )
                tile_up_D = block_gemm_golden_model(
                    shared_swish_glu_M1,
                    shared_swish_glu_K1 * shared_swish_glu_K2,
                    shared_swish_glu_N1,
                    meshRow,
                    tileSize,
                    meshCol,
                    tile_A,
                    tile_up_B,
                    subtraction_a,
                    subtraction_b,
                    tile_up_C,
                )
                expert_up_D_tiles.append(tile_up_D)

                tile_gate_B = shared_experts_gate_B_tiled[
                    shared_expert_index, n
                ].reshape(-1)
                tile_gate_C = (
                    shared_experts_gate_C_tiled[m, n].reshape(-1)
                    if kwargs["addNonZeroC"] == 1
                    else np.zeros(
                        (shared_swish_glu_M1, shared_swish_glu_N1, meshRow, meshCol),
                        dtype=np.int32,
                    ).reshape(-1)
                )
                tile_gate_D = block_gemm_golden_model(
                    shared_swish_glu_M1,
                    shared_swish_glu_K1 * shared_swish_glu_K2,
                    shared_swish_glu_N1,
                    meshRow,
                    tileSize,
                    meshCol,
                    tile_A,
                    tile_gate_B,
                    subtraction_a,
                    subtraction_b,
                    tile_gate_C,
                )
                expert_gate_D_tiles.append(tile_gate_D)

        expert_up_D_1d = np.concatenate(expert_up_D_tiles)
        expert_gate_D_1d = np.concatenate(expert_gate_D_tiles)
        total_shared_experts_up_projection_D.append(expert_up_D_1d)
        total_shared_experts_gate_D.append(expert_gate_D_1d)

        # SwishGLU calculates element-wise, shape remains intact
        expert_act_A_1d = swish_glu_golden_model(
            expert_gate_D_1d,
            expert_up_D_1d,
            MoE_config["swish_glu_scale_in"],
            MoE_config["swish_glu_scale_out"],
        )
        total_shared_experts_activated_A.append(expert_act_A_1d)

        # Down Projection Loop
        expert_down_D_tiles = []
        for m in range(shared_down_projection_M2):
            # Extract the correct activated A chunk for this M iteration
            tile_act_A = expert_act_A_1d[
                m * act_A_chunk_size_shared : (m + 1) * act_A_chunk_size_shared
            ]
            for n in range(shared_down_projection_N2):
                tile_down_B = shared_experts_down_projection_B_tiled[
                    shared_expert_index, n
                ].reshape(-1)
                tile_down_C = (
                    shared_experts_down_projection_C_tiled[m, n].reshape(-1)
                    if kwargs["addNonZeroC"] == 1
                    else np.zeros(
                        (
                            shared_down_projection_M1,
                            shared_down_projection_N1,
                            meshRow,
                            meshCol,
                        ),
                        dtype=np.int32,
                    ).reshape(-1)
                )
                tile_down_D = block_gemm_golden_model(
                    shared_down_projection_M1,
                    shared_down_projection_K1 * shared_down_projection_K2,
                    shared_down_projection_N1,
                    meshRow,
                    tileSize,
                    meshCol,
                    tile_act_A,
                    tile_down_B,
                    subtraction_a,
                    subtraction_b,
                    tile_down_C,
                )
                expert_down_D_tiles.append(tile_down_D)

        expert_down_D_1d = np.concatenate(expert_down_D_tiles)
        total_shared_experts_down_projection_D.append(expert_down_D_1d)
        all_experts_down_projection_D_2d.append(expert_down_D_1d)

    data_str += [
        format_vector_definition(
            "int32_t",
            "shared_experts_up_projection_D",
            np.concatenate(total_shared_experts_up_projection_D),
        )
    ]
    data_str += [
        format_vector_definition(
            "int32_t",
            "total_shared_experts_gate_D",
            np.concatenate(total_shared_experts_gate_D),
        )
    ]
    data_str += [
        format_vector_definition(
            "int32_t",
            "total_shared_experts_activated_A",
            np.concatenate(total_shared_experts_activated_A),
        )
    ]
    data_str += [
        format_vector_definition(
            "int32_t",
            "total_shared_experts_down_projection_D",
            np.concatenate(total_shared_experts_down_projection_D),
        )
    ]

    ######### Individual experts #########
    total_individual_experts_up_projection_D = []
    total_individual_experts_gate_D = []
    total_individual_experts_activated_A = []
    total_individual_experts_down_projection_D = []

    act_A_chunk_size_indiv = (
        individual_swish_glu_N2
        * individual_swish_glu_M1
        * individual_swish_glu_N1
        * meshRow
        * meshCol
    )

    # 计算单个硬件专家最大的容纳 Token 数量 (M 维度物理总行数)
    max_tokens_per_expert = individual_swish_glu_M2 * individual_swish_glu_M1 * meshRow

    for individual_expert_index in range(MoE_config["expert_number_each_layer"]):
        expert_up_D_tiles = []
        expert_gate_D_tiles = []

        # =====================================================================
        # 核心：完美模拟硬件的 Scatter & Pad
        # =====================================================================
        # 1. 查找分发给该专家的所有 token ID
        token_ids = expert_to_tokens.get(individual_expert_index, [])
        actual_tokens_num = len(token_ids)

        # 2. 从全局 A 矩阵中把这些 Token 抠出来
        if actual_tokens_num > 0:
            # 硬件通常有最大容量限制，超出的会被截断 (Truncate)
            safe_tokens = min(actual_tokens_num, max_tokens_per_expert)
            scattered_A_2d = input_A_matrix[token_ids[:safe_tokens]]
        else:
            scattered_A_2d = np.empty((0, K_total), dtype=np.int8)

        # 3. 零填充 (Pad) 到硬件要求的完美物理边界
        pad_rows = max_tokens_per_expert - scattered_A_2d.shape[0]
        if pad_rows > 0:
            padded_expert_A_2d = np.pad(
                scattered_A_2d,
                ((0, pad_rows), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            padded_expert_A_2d = scattered_A_2d

        # 4. 把 2D 矩阵重新切成 Tile，模拟 DMA 视角
        # 从 [M_total, K_total] 转回 [M2, M1, K, meshRow, tileSize]
        expert_A_tiled_5D = padded_expert_A_2d.reshape(
            individual_swish_glu_M2,
            individual_swish_glu_M1,
            meshRow,
            individual_swish_glu_K1,
            tileSize,
        ).transpose(0, 1, 3, 2, 4)

        # =====================================================================

        for m in range(individual_swish_glu_M2):
            # 现在！我们提取的是经过 Scatter & Pad 后，属于这个专家的专属物理 Tile！
            tile_A = expert_A_tiled_5D[m].reshape(-1)

            for n in range(individual_swish_glu_N2):
                tile_up_B = individual_experts_up_projection_B_tiled[
                    individual_expert_index, n
                ].reshape(-1)
                tile_up_C = (
                    individual_experts_up_projection_C_tiled[m, n].reshape(-1)
                    if kwargs["addNonZeroC"] == 1
                    else np.zeros(
                        (
                            individual_swish_glu_M1,
                            individual_swish_glu_N1,
                            meshRow,
                            meshCol,
                        ),
                        dtype=np.int32,
                    ).reshape(-1)
                )
                tile_up_D = block_gemm_golden_model(
                    individual_swish_glu_M1,
                    individual_swish_glu_K1 * individual_swish_glu_K2,
                    individual_swish_glu_N1,
                    meshRow,
                    tileSize,
                    meshCol,
                    tile_A,
                    tile_up_B,
                    subtraction_a,
                    subtraction_b,
                    tile_up_C,
                )
                expert_up_D_tiles.append(tile_up_D)

                tile_gate_B = individual_experts_gate_B_tiled[
                    individual_expert_index, n
                ].reshape(-1)
                tile_gate_C = (
                    individual_experts_gate_C_tiled[m, n].reshape(-1)
                    if kwargs["addNonZeroC"] == 1
                    else np.zeros(
                        (
                            individual_swish_glu_M1,
                            individual_swish_glu_N1,
                            meshRow,
                            meshCol,
                        ),
                        dtype=np.int32,
                    ).reshape(-1)
                )
                tile_gate_D = block_gemm_golden_model(
                    individual_swish_glu_M1,
                    individual_swish_glu_K1 * individual_swish_glu_K2,
                    individual_swish_glu_N1,
                    meshRow,
                    tileSize,
                    meshCol,
                    tile_A,
                    tile_gate_B,
                    subtraction_a,
                    subtraction_b,
                    tile_gate_C,
                )
                expert_gate_D_tiles.append(tile_gate_D)

        expert_up_D_1d = np.concatenate(expert_up_D_tiles)
        expert_gate_D_1d = np.concatenate(expert_gate_D_tiles)
        total_individual_experts_up_projection_D.append(expert_up_D_1d)
        total_individual_experts_gate_D.append(expert_gate_D_1d)

        # SwishGLU calculates element-wise
        expert_act_A_1d = swish_glu_golden_model(
            expert_gate_D_1d,
            expert_up_D_1d,
            MoE_config["swish_glu_scale_in"],
            MoE_config["swish_glu_scale_out"],
        )
        total_individual_experts_activated_A.append(expert_act_A_1d)

        # Down Projection Loop
        expert_down_D_tiles = []
        for m in range(individual_down_projection_M2):
            tile_act_A = expert_act_A_1d[
                m * act_A_chunk_size_indiv : (m + 1) * act_A_chunk_size_indiv
            ]
            for n in range(individual_down_projection_N2):
                tile_down_B = individual_experts_down_projection_B_tiled[
                    individual_expert_index, n
                ].reshape(-1)
                tile_down_C = (
                    individual_experts_down_projection_C_tiled[m, n].reshape(-1)
                    if kwargs["addNonZeroC"] == 1
                    else np.zeros(
                        (
                            individual_down_projection_M1,
                            individual_down_projection_N1,
                            meshRow,
                            meshCol,
                        ),
                        dtype=np.int32,
                    ).reshape(-1)
                )
                tile_down_D = block_gemm_golden_model(
                    individual_down_projection_M1,
                    individual_down_projection_K1 * individual_down_projection_K2,
                    individual_down_projection_N1,
                    meshRow,
                    tileSize,
                    meshCol,
                    tile_act_A,
                    tile_down_B,
                    subtraction_a,
                    subtraction_b,
                    tile_down_C,
                )
                expert_down_D_tiles.append(tile_down_D)

        expert_down_D_1d = np.concatenate(expert_down_D_tiles)
        total_individual_experts_down_projection_D.append(expert_down_D_1d)
        all_experts_down_projection_D_2d.append(expert_down_D_1d)

    data_str += [
        format_vector_definition(
            "int32_t",
            "individual_experts_up_projection_D",
            np.concatenate(total_individual_experts_up_projection_D),
        )
    ]
    data_str += [
        format_vector_definition(
            "int32_t",
            "individual_experts_gate_D",
            np.concatenate(total_individual_experts_gate_D),
        )
    ]
    data_str += [
        format_vector_definition(
            "int32_t",
            "individual_experts_down_projection_D",
            np.concatenate(total_individual_experts_down_projection_D),
        )
    ]

    # =========================================================================
    # 最终的 Accumulation 归约计算
    # =========================================================================
    N_total_out = (
        individual_down_projection_N1 * individual_down_projection_N2 * meshCol
    )
    final_layer_output_2D = np.zeros((M_total, N_total_out), dtype=np.float64)
    shared_weight = MoE_config["softmax_scale"]

    # 1. 共享专家的归约 (Shared Expert 没有做 Scatter，输入是全量的 input_A，所以全局行号可以直接用)
    for shared_idx in range(MoE_config["shared_expert_number_k"]):
        shared_D_1D = all_experts_down_projection_D_2d[shared_idx]
        shared_D_tiled_6D = shared_D_1D.reshape(
            shared_down_projection_M2,
            shared_down_projection_N2,
            shared_down_projection_M1,
            shared_down_projection_N1,
            meshRow,
            meshCol,
        )
        # Shared Expert 输出的行数与总 Token 数 (M_total) 是一致的
        shared_D_2D = shared_D_tiled_6D.transpose(0, 2, 1, 3, 4, 5).reshape(
            M_total, N_total_out
        )
        final_layer_output_2D += shared_D_2D * shared_weight

    # 2. 独立专家的归约 (Individual Expert 经过了 Scatter，必须进行反向映射 Gather)
    offset = MoE_config["shared_expert_number_k"]

    # max_tokens_per_expert 是该专家最大的行数容量
    max_tokens_per_expert = individual_swish_glu_M2 * individual_swish_glu_M1 * meshRow

    # 为了加快效率，依然只遍历被选中的专家
    for list_idx, individual_expert_index in enumerate(unique_selected_experts):

        # 注意：all_experts_down_projection_D_2d 里装的是按 expert_id 顺序排好的全量列表
        # 所以直接用 offset + individual_expert_index 取出对应的专家数据
        individual_D_1D = all_experts_down_projection_D_2d[
            offset + individual_expert_index
        ]

        # Un-tile 恢复为数学视角的 2D 矩阵
        individual_D_tiled_6D = individual_D_1D.reshape(
            individual_down_projection_M2,
            individual_down_projection_N2,
            individual_down_projection_M1,
            individual_down_projection_N1,
            meshRow,
            meshCol,
        )
        # 此时的 2D 矩阵行数是 max_tokens_per_expert，而不是 M_total！
        individual_D_2D = individual_D_tiled_6D.transpose(0, 2, 1, 3, 4, 5).reshape(
            max_tokens_per_expert, N_total_out
        )

        # 取出路由给该专家的所有全局 Token ID
        token_ids = expert_to_tokens[individual_expert_index]

        # 【核心修正】：用 enumerate 产生 local_row_idx
        # local_row_idx 对应于该专家内部矩阵的第 0, 1, 2... 行
        for local_row_idx, token_id in enumerate(token_ids):
            # 硬件防护：如果路由分配给该专家的 Token 数超过了它的最大容量，超出部分在 Scatter 时被丢弃了，这里就不加了
            if local_row_idx >= max_tokens_per_expert:
                break

            probability = token_expert_probability_map[
                (token_id, individual_expert_index)
            ]

            # 【反向映射 (Gather)】:
            # 将专家矩阵的第 local_row_idx 行的数据，乘以概率后，累加到最终输出矩阵的第 token_id 行！
            final_layer_output_2D[token_id] += (
                probability * individual_D_2D[local_row_idx]
            )

    # =========================================================================
    # Final Layer Output 处理与落盘
    # =========================================================================
    layer_output_float = final_layer_output_2D / (2 ** MoE_config["final_shift_step"])
    layer_output_2D = np.clip(layer_output_float, -128, 127).astype(np.int8)

    # Re-tile to 6D memory layout before flattening for C array
    layer_output_tiled_6D = layer_output_2D.reshape(
        router_M2,
        router_M1,
        individual_down_projection_N2,
        individual_down_projection_N1,
        meshRow,
        meshCol,
    ).transpose(
        0, 2, 1, 3, 4, 5
    )  # Swaps to (M2, N2, M1, N1, meshRow, meshCol)

    layer_output = layer_output_tiled_6D.flatten()

    data_str += [
        format_vector_definition(
            "int8_t",
            "layer_output",
            layer_output,
        )
    ]

    return data_str


def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generating data for kernels")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
        help="Select hardware config file kernel",
    )
    parser.add_argument(
        "--emit_mini_golden",
        action="store_true",
        help="Accepted for compatibility with copied workload Makefiles.",
    )
    parser.add_argument(
        "--num_double_buffers",
        type=int,
        default=None,
        help="Accepted for compatibility with copied workload Makefiles.",
    )
    parser.add_argument(
        "--full_golden_data",
        action="store_true",
        help="Emit all intermediate and final golden tensors for verification.",
    )
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Load hardware config file
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    # Merge dictionaries (hw overrides param in case of conflicts)
    merged_config = {**param, **hw}
    merged_config["full_golden_data"] = args.full_golden_data

    # Emit header file
    print(emit_header_file(**merged_config))


if __name__ == "__main__":
    main()
