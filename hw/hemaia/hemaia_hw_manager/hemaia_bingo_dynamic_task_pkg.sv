// Copyright 2026 KU Leuven.
// Licensed under the Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package hemaia_bingo_dynamic_task_pkg;

  localparam int unsigned HEMAIA_BINGO_DYN_SEQ_WIDTH  = 16;

  // Area-optimized dynamic path: the FIFO carries exactly the 64-bit Bingo
  // task descriptor consumed by bingo_hw_manager_top. Large per-expert context
  // lives in software-programmed slot argument records, not in RTL flops.
  typedef logic [63:0] hemaia_bingo_dynamic_task_desc_t;

  typedef struct packed {
    logic [HEMAIA_BINGO_DYN_SEQ_WIDTH-1:0] seq_id;
    logic [1:0]                            cluster_id;
    logic [7:0]                            status;
    logic                                  error;
  } hemaia_bingo_dynamic_done_t;

endpackage
