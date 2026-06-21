# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal CVA6 MMIO smoke test for the MoE scheduler AXI-to-reg slave.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HOST_APP_TYPE = offload_bingo_hw
CHIP_TYPE     = single_chip
WORKLOAD      = scheduler_mmio_smoke
DEV_APP       = snax-bingo-offload
SRCS = ../../src/offload_bingo_hw.c
BINGO_HOST ?= 1
INCL_DEVICE_BINARY ?= 1

MOE_SCHED_DIR = $(MK_DIR)/../multi_cluster_MoE
SRCS += $(MOE_SCHED_DIR)/moe_scheduler.c

OFFLOAD_H = $(MK_DIR)/offload_bingo_hw.h

###############
# Build settings
###############
INCDIRS += $(MK_DIR)
INCDIRS += $(MOE_SCHED_DIR)
PARTIAL_OUTPUTS += $(OFFLOAD_H)

.PHONY: gen-offload

gen-offload: $(OFFLOAD_H)

include ../../../../common.mk
