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
DEBUG_LEVEL ?= 0

# DATA: 生成 dense individual、padded shared token 输入和 packed weights/config。
DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/multi_cluster_MoE_data.h
DATAGEN_FLAGS ?=
OFFLOAD_FLAGS ?=
MOE_SCHED_BENCH_CASE ?=
MOE_WORKLOAD_MODE ?=
MOE_TOTAL_TOKENS ?=
SCHED_BENCH_CASES_PY = $(MK_DIR)/moe_scheduler_bench_cases.py
ROUTING_CASES_PY = $(MK_DIR)/moe_routing_cases.py
SCHED_BENCH_STAMP = $(MK_DIR)/build/.scheduler_bench_case
ROUTING_DEBUG_STAMP = $(MK_DIR)/build/.routing_debug_level
WORKLOAD_MODE_STAMP = $(MK_DIR)/build/.workload_mode
TOTAL_TOKENS_STAMP = $(MK_DIR)/build/.total_tokens

ifneq ($(strip $(MOE_SCHED_BENCH_CASE)),)
DATAGEN_FLAGS += --scheduler-bench-case $(MOE_SCHED_BENCH_CASE)
OFFLOAD_FLAGS += --scheduler-bench-case $(MOE_SCHED_BENCH_CASE)
endif

ifeq ($(strip $(MOE_WORKLOAD_MODE)),scheduler_routed)
DATAGEN_FLAGS += --scheduler-bench-run-routed
endif

ifneq ($(strip $(MOE_WORKLOAD_MODE)),)
OFFLOAD_FLAGS += --workload-mode $(MOE_WORKLOAD_MODE)
endif

ifneq ($(strip $(MOE_TOTAL_TOKENS)),)
DATAGEN_FLAGS += --total-tokens $(MOE_TOTAL_TOKENS)
OFFLOAD_FLAGS += --total-tokens $(MOE_TOTAL_TOKENS)
endif

BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
HW_CFG ?= $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_rebalanced_cluster.hjson

$(DATA_H): $(MK_DIR)/multi_cluster_MoE_datagen.py $(MK_DIR)/moe_layout.py \
		$(SCHED_BENCH_CASES_PY) $(ROUTING_CASES_PY) $(DATA_CFG) $(HW_CFG) \
		$(SCHED_BENCH_STAMP) $(ROUTING_DEBUG_STAMP) $(TOTAL_TOKENS_STAMP)
	python3 $< -c $(DATA_CFG) --hwcfg $(HW_CFG) \
		--debug-level $(DEBUG_LEVEL) $(DATAGEN_FLAGS) > $@

# Offload: 由 main_bingo.py 生成 DFG 对应的 offload 头文件
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
CONFIG_H = $(MK_DIR)/multi_cluster_MoE_config.h
BINGO_KERNEL_ARGS_PY = $(MK_DIR)/../../../../../runtime/libbingo/mini_compiler/bingo_kernel_args.py
BINGO_DFG_PY = $(MK_DIR)/../../../../../runtime/libbingo/mini_compiler/bingo_dfg.py
MOE_DYNAMIC_SLOT_DFG_PY = $(MK_DIR)/../moe_dynamic_slot_dfg.py
$(OFFLOAD_H) $(CONFIG_H) &: $(MK_DIR)/main_bingo.py $(MK_DIR)/moe_layout.py \
		$(SCHED_BENCH_CASES_PY) $(SCHED_BENCH_STAMP) \
		$(MOE_DYNAMIC_SLOT_DFG_PY) $(BINGO_DFG_PY) $(BINGO_KERNEL_ARGS_PY) \
		$(DATA_CFG) $(HW_CFG) $(WORKLOAD_MODE_STAMP) $(TOTAL_TOKENS_STAMP)
	python3 $< --cfg $(DATA_CFG) --hwcfg $(HW_CFG) --output_dir $(MK_DIR) \
		--output_offload_file_name $(notdir $(OFFLOAD_H)) $(OFFLOAD_FLAGS)
	rm -f $(MK_DIR)/*.png $(MK_DIR)/*.csv

.PHONY: FORCE_SCHED_BENCH_STAMP
FORCE_SCHED_BENCH_STAMP:

$(SCHED_BENCH_STAMP): FORCE_SCHED_BENCH_STAMP
	mkdir -p $(dir $@)
	@printf '%s\n' '$(strip $(MOE_SCHED_BENCH_CASE))' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi

.PHONY: FORCE_ROUTING_DEBUG_STAMP
FORCE_ROUTING_DEBUG_STAMP:

$(ROUTING_DEBUG_STAMP): FORCE_ROUTING_DEBUG_STAMP
	mkdir -p $(dir $@)
	@printf '%s\n' '$(DEBUG_LEVEL)' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi

.PHONY: FORCE_WORKLOAD_MODE_STAMP
FORCE_WORKLOAD_MODE_STAMP:

$(WORKLOAD_MODE_STAMP): FORCE_WORKLOAD_MODE_STAMP
	mkdir -p $(dir $@)
	@printf '%s\n' '$(strip $(MOE_WORKLOAD_MODE))' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi

.PHONY: FORCE_TOTAL_TOKENS_STAMP
FORCE_TOTAL_TOKENS_STAMP:

$(TOTAL_TOKENS_STAMP): FORCE_TOTAL_TOKENS_STAMP
	mkdir -p $(dir $@)
	@printf '%s\n' '$(strip $(MOE_TOTAL_TOKENS))' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi


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
#   MOE_RUNTIME_TIMING=1 只记录两个cluster的begin/end，用于低扰动global时间。
#   MOE_RUNTIME_TIMING=2 记录全部slot/stage/task时间，供离线脚本详细分析。
#   MOE_SCHED_BENCH_CASE=<name> 仅生成所选 Prepare/Execute 小测试。
# 不再支持 HW/SW check、legacy scheduler 或 runtime fallback 开关。
# Workload generation selectors:
#   MOE_WORKLOAD_MODE=router_shared|scheduler_prepare|scheduler_routed|full
#   scheduler_routed requires MOE_SCHED_BENCH_CASE and measures Prepare through
#   completion of both routed-expert cluster chains as one continuous interval.
#   MOE_TOTAL_TOKENS=8|32|64|128 (empty means params.hjson)
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
# main_bingo.py emits the workload config together with OFFLOAD_H.
PARTIAL_OUTPUTS += $(CONFIG_H)

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

# common.mk does not consume workload-generated headers from PARTIAL_OUTPUTS as
# ELF prerequisites. Attach them explicitly so changing benchmark cases cannot
# silently reuse an ELF compiled from the previous generated DFG.
$(PARTIAL_ELF) $(ELFS): $(DATA_H) $(OFFLOAD_H) $(CONFIG_H)
