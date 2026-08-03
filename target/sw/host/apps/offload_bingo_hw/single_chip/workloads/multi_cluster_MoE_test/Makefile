# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE = single_chip
WORKLOAD = multi_cluster_MoE_test
DEV_APP = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

DATA_CFG ?= $(MK_DIR)/params.hjson
DATA_H = $(MK_DIR)/multi_cluster_MoE_test_data.h
OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h
# Choices: baseline, s2pf_both, static_desc, dynamic_desc, high_to_low, low_to_high,
# ends_inward, c_tail_smoke,
# s1_stage_smoke.
MOE_TEST_SCHEDULE ?= baseline
SCHEDULE_STAMP = $(MK_DIR)/.schedule-$(MOE_TEST_SCHEDULE)
MOE_DYNAMIC_SLOT_DFG_PY = $(MK_DIR)/../moe_dynamic_slot_dfg.py
MOE_RUNTIME_TIMING_H = $(MK_DIR)/../multi_cluster_MoE/moe_runtime_timing.h

BENDER = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
HW_CFG ?= $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_bgran2_highgran_search_2.hjson

$(SCHEDULE_STAMP):
	rm -f $(MK_DIR)/.schedule-*
	touch $@

$(DATA_H): $(MK_DIR)/multi_cluster_MoE_test_datagen.py $(MK_DIR)/moe_test_layout.py \
		$(MK_DIR)/moe_test_schedule.py $(SCHEDULE_STAMP) $(DATA_CFG) $(HW_CFG)
	python3 $< -c $(DATA_CFG) --hwcfg $(HW_CFG) \
		--schedule-profile $(MOE_TEST_SCHEDULE) > $@

$(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(MK_DIR)/moe_test_layout.py \
		$(MK_DIR)/moe_test_schedule.py $(MK_DIR)/moe_high_to_low_workload.py \
		$(MOE_DYNAMIC_SLOT_DFG_PY) $(MOE_RUNTIME_TIMING_H) $(DATA_H) $(DATA_CFG) $(HW_CFG)
	python3 $< --cfg $(DATA_CFG) --hwcfg $(HW_CFG) \
		--schedule-profile $(MOE_TEST_SCHEDULE) \
		--output_dir $(MK_DIR) --output_offload_file_name $(notdir $(OFFLOAD_H))
	rm -f $(MK_DIR)/*.png $(MK_DIR)/*.csv

INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H) $(OFFLOAD_H)

# This static test only uses the entry/exit host kernels.  The shared host
# header also defines unrelated software-MoE helpers; discard those unreachable
# function sections so their libm/libgcc dependencies do not enter this image.
RISCV_LDFLAGS += -Wl,--gc-sections

.PHONY: clean-data clean-offload clean gen-data gen-offload
clean-data:
	rm -f $(DATA_H) $(MK_DIR)/.schedule-*
clean-offload:
	rm -f $(OFFLOAD_H) $(MK_DIR)/*.png $(MK_DIR)/*.csv
clean: clean-data clean-offload
gen-data: $(DATA_H)
gen-offload: $(OFFLOAD_H)

include ../../../../common.mk

# common.mk emits a dependency file but does not include it. Keep profile
# switches correct even when an ELF from the other profile already exists.
$(PARTIAL_ELF) $(ELFS): $(SCHEDULE_STAMP) $(DATA_H) $(OFFLOAD_H)
