# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# 这个 Makefile 只做两件事：
#   1. 调 datagen 生成 multi_cluster_MoE_data.h
#   2. 调 main_bingo.py 生成 offload_bingo_hw.h
#
# 也就是说，这里不负责编译 RTL，也不负责跑仿真；它只是把 workload 需要的
# “静态数据头文件”和“DFG offload 头文件”准备好。
#
# 注意：外部 include 这个 Makefile 时要用绝对路径，否则 MK_DIR 解析会出错。
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = multi_cluster_MoE
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1 # True to include the bingo runtime
INCL_DEVICE_BINARY ?= 1 # True to include the device binary

# DATA: 由 datagen 生成运行时输入、权重和 fused L15 配置。
DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/multi_cluster_MoE_data.h
DATAGEN_FLAGS ?=

BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
HW_CFG ?= $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_rebalanced_cluster.hjson

$(DATA_H): $(MK_DIR)/multi_cluster_MoE_datagen.py $(MK_DIR)/moe_l15_layout.py $(DATA_CFG) $(HW_CFG)
	python3 $< -c $(DATA_CFG) --hwcfg $(HW_CFG) $(DATAGEN_FLAGS) > $@

# Offload: 由 main_bingo.py 生成 DFG 对应的 offload 头文件
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
CONFIG_H = $(MK_DIR)/multi_cluster_MoE_config.h
$(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(MK_DIR)/moe_l15_layout.py $(DATA_CFG) $(HW_CFG)
	python3 $< --cfg $(DATA_CFG) --hwcfg $(HW_CFG) --output_dir $(MK_DIR) --output_offload_file_name $(notdir $(OFFLOAD_H))
	rm -f $(MK_DIR)/*.png $(MK_DIR)/*.csv


###############
# Build settings
############### 
# Scheduler implementation selector. There are exactly two supported builds:
#   MOE_HW_SCHEDULER=1: RTL MMIO scheduler + direct lowering (production path)
#   MOE_HW_SCHEDULER=0: pure C scheduler (paper comparison path)
#
# 在 Thesis 根目录的 hemaia_main_dev 容器内执行：
#   make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
#     WORKLOAD=multi_cluster_MoE DEV_APP=snax-bingo-offload \
#     CFG_OVERRIDE=target/rtl/cfg/hemaia_4cluster_singlechip.hjson \
#     MOE_HW_SCHEDULER=1
#
# 可选开关：
#   DEBUG_LEVEL=1       打印 router/scheduler 调试值，会显著扰动周期（默认 0）。
#   NODE_TIMING=1       打印每个 device task 的 Start/End mcycle，用于检查任务是否停滞。
#   PERF_TRACING=0      关闭默认启用的 NOP trace marker；仿真 trace 需要保持 1。
#   MOE_MCYCLE_DETAIL=1 在最终节点打印细粒度 mcycle 统计。
#   MOE_INDIV_BRINGUP=1 打印 individual descriptor、真实 token 搬运和各计算阶段。
#   MOE_RUNTIME_TIMING=1 将 individual 执行时间写入各自 scratchpad，DFG
#                        结束后由 CVA6 统一打印紧凑原始记录，离线脚本分析。
# 不再支持 HW/SW check、legacy scheduler 或 runtime fallback 开关。
MOE_HW_SCHEDULER ?= 0
MOE_MCYCLE_DETAIL ?= 0
ifeq ($(MOE_HW_SCHEDULER), 1)
override USER_FLAGS += -DMOE_ENABLE_HW_SCHEDULER
endif
ifeq ($(MOE_MCYCLE_DETAIL), 1)
override USER_FLAGS += -DMOE_MCYCLE_DETAIL=1
endif

SRCS += $(MK_DIR)/moe_scheduler.c
INCDIRS += $(MK_DIR)
# Add the DATA_H to partial outputs
PARTIAL_OUTPUTS += $(DATA_H)
# Add the OFFLOAD_H to partial outputs
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: clean-data clean-offload clean

clean-data:
	rm -f $(DATA_H)
	
clean-offload:
	rm -rf $(OFFLOAD_H)
	rm -f $(CONFIG_H)
	rm -rf *.png
	rm -rf *.csv
clean: clean-data clean-offload

gen-data: $(DATA_H)

gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
