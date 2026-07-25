// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

// Dynamic MoE device-kernel umbrella. Keep includes in dependency order:
// definitions -> hardware CSR helpers -> layouts/protocol -> DMA -> stage APIs.
#pragma once

#include "../macros.h"
#include "moe_runtime_timing_record.h"
#include "moe_dynamic_s4_schedule.h"
#include <snax_versacore_lib.h>
#include <snax_xdma_lib.h>

#include "moe_dynamic_defs.h"
#include "moe_versacore_hw.h"
#include "moe_router_kernels.h"
#include "moe_swiglu_config.h"
#include "moe_dynamic_protocol.h"
#include "moe_dynamic_bank.h"
#include "moe_down_config.h"
#include "moe_dynamic_dma.h"
#include "moe_dynamic_generic_kernels.h"
#include "moe_dynamic_stage_s1.h"
#include "moe_dynamic_stage_s3.h"
#include "moe_dynamic_stage_prefetch.h"
#include "moe_dynamic_stage_compute.h"
#include "moe_dynamic_stage_store.h"
#include "moe_dynamic_discrete.h"
#include "moe_dynamic_l15.h"
