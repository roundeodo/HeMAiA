# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Host API test workload — verifies MoE operator C functions against
# Python golden models without requiring SNAX accelerator computation.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = host_api_test
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

# DATA — self-contained datagen, no external config needed
DATA_H = $(MK_DIR)/host_api_test_data.h

$(DATA_H): $(MK_DIR)/host_api_test_datagen.py
	python3 $< > $@

###############
# Build settings
###############
INCDIRS += $(MK_DIR)
PARTIAL_OUTPUTS += $(DATA_H)

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)

clean: clean-data

include ../../../../common.mk
