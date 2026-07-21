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
MOE_DYNAMIC_SLOT_DFG_PY = $(MK_DIR)/../moe_dynamic_slot_dfg.py

BENDER = bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
HW_CFG ?= $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_bgran2_highgran_search_2.hjson

$(DATA_H): $(MK_DIR)/multi_cluster_MoE_test_datagen.py $(MK_DIR)/moe_test_layout.py $(DATA_CFG) $(HW_CFG)
	python3 $< -c $(DATA_CFG) --hwcfg $(HW_CFG) > $@

$(OFFLOAD_H): $(MK_DIR)/main_bingo.py $(MK_DIR)/moe_test_layout.py \
	$(MOE_DYNAMIC_SLOT_DFG_PY) $(DATA_H) $(DATA_CFG) $(HW_CFG)
	python3 $< --cfg $(DATA_CFG) --hwcfg $(HW_CFG) \
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
	rm -f $(DATA_H)
clean-offload:
	rm -f $(OFFLOAD_H) $(MK_DIR)/*.png $(MK_DIR)/*.csv
clean: clean-data clean-offload
gen-data: $(DATA_H)
gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
