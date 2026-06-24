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

# DATA: 由 multi_cluster_MoE_datagen.py 生成输入/权重/Golden 数据头文件
DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/multi_cluster_MoE_data.h
DATAGEN_FLAGS ?=

BENDER   = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
HW_CFG ?= $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_rebalanced_cluster.hjson

$(DATA_H): $(MK_DIR)/multi_cluster_MoE_datagen.py $(DATA_CFG) $(HW_CFG)
	python3 $< -c $(DATA_CFG) --hwcfg $(HW_CFG) $(DATAGEN_FLAGS) > $@

# Offload: 由 main_bingo.py 生成 DFG 对应的 offload 头文件
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
$(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(DATA_CFG) $(HW_CFG)
	python3 $< --cfg $(DATA_CFG) --hwcfg $(HW_CFG) --output_dir $(MK_DIR) --output_offload_file_name $(notdir $(OFFLOAD_H)) --emit_mini_golden


###############
# Build settings
############### 
# 启用 O(E) 快速调度路径（去掉此行则退回原版 O(E³×N⁴) 完整搜索）
# The top-level HeMAiA Makefile forwards USER_FLAGS as a command-line
# variable to recursive makes.  Plain "USER_FLAGS +=" assignments in this
# workload Makefile would then be ignored by GNU make, so use "override" for
# workload-local compile switches that must survive top-level builds.
override USER_FLAGS += -DMOE_SCHEDULE_FAST

# Optional production path: use the RTL scheduler MMIO slave for compact-plan
# generation, then lower that compact plan on CVA6 into the existing MoE API
# schedule.  Keep disabled by default so software-vs-hardware A/B builds can
# use the same workload.
MOE_HW_SCHEDULER ?= 0
MOE_HW_SCHEDULER_CHECK ?= 0
ifeq ($(MOE_HW_SCHEDULER), 1)
override USER_FLAGS += -DMOE_ENABLE_HW_SCHEDULER
endif
ifeq ($(MOE_HW_SCHEDULER_CHECK), 1)
override USER_FLAGS += -DMOE_ENABLE_HW_SCHEDULER_CHECK
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
	rm -rf *.png
	rm -rf *.csv
clean: clean-data clean-offload

gen-data: $(DATA_H)

gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
