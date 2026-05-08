// Copyright 2026 KU Leuven.
// Licensed under the Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

module hemaia_bingo_dynamic_task_csr
  import hemaia_bingo_dynamic_task_pkg::*;
#(
  parameter int unsigned CSR_ADDR_WIDTH = 48,
  parameter int unsigned TASK_QUEUE_DEPTH = 8,
  parameter int unsigned DONE_QUEUE_DEPTH = 8,
  parameter logic [CSR_ADDR_WIDTH-1:0] DYNAMIC_CSR_BASE_ADDR = '0,
  parameter logic [CSR_ADDR_WIDTH-1:0] DYNAMIC_CSR_ADDR_MASK = {CSR_ADDR_WIDTH{1'b1}},
  parameter type csr_req_t = logic,
  parameter type csr_rsp_t = logic
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  csr_req_t csr_req_i,
  input  logic     csr_req_valid_i,
  output logic     csr_req_ready_o,
  output csr_rsp_t csr_rsp_o,
  output logic     csr_rsp_valid_o,
  input  logic     csr_rsp_ready_i,

  output csr_req_t manager_csr_req_o,
  output logic     manager_csr_req_valid_o,
  input  logic     manager_csr_req_ready_i,
  input  csr_rsp_t manager_csr_rsp_i,
  input  logic     manager_csr_rsp_valid_i,
  output logic     manager_csr_rsp_ready_o,

  output hemaia_bingo_dynamic_task_desc_t task_pop_desc_o,
  output logic                            task_pop_valid_o,
  input  logic                            task_pop_ready_i,

  input  hemaia_bingo_dynamic_done_t done_push_i,
  input  logic                       done_push_valid_i,
  output logic                       done_push_ready_o
);

  localparam int unsigned TaskDescWidth = $bits(hemaia_bingo_dynamic_task_desc_t);
  localparam int unsigned DoneWidth     = $bits(hemaia_bingo_dynamic_done_t);
  localparam int unsigned TaskDescWords = (TaskDescWidth + 31) / 32;
  localparam int unsigned DoneWords     = (DoneWidth + 31) / 32;

  localparam int unsigned RegStatus      = 8'h00;
  localparam int unsigned RegControl     = 8'h01;
  localparam int unsigned RegTaskCredit  = 8'h02;
  localparam int unsigned RegDoneCredit  = 8'h03;
  localparam int unsigned RegDoneBase    = 8'h04;
  localparam int unsigned RegTaskBase    = 8'h10;

  logic [TaskDescWidth-1:0] task_shadow_q;
  logic [DoneWidth-1:0]     done_pop_bits;
  logic [CSR_ADDR_WIDTH-1:0] dynamic_addr_delta;
  logic [7:0] csr_word_offset;

  logic dynamic_selected;
  logic dynamic_read;
  logic dynamic_write;
  logic dynamic_ready;
  logic dynamic_task_commit;
  logic dynamic_flush;
  logic dynamic_done_pop;
  logic dynamic_task_word_write;

  logic [31:0] dynamic_read_data;
  logic [31:0] dynamic_status;
  logic [31:0] task_credit;
  logic [31:0] done_credit;

  logic task_push_ready;
  logic task_push_valid;
  logic done_pop_valid;
  logic done_pop_ready;
  hemaia_bingo_dynamic_done_t done_pop;

  function automatic logic [31:0] read_task_shadow_word(
    input logic [TaskDescWidth-1:0] bits,
    input logic [7:0]               word_idx
  );
    int unsigned src_idx;
    begin
      read_task_shadow_word = '0;
      for (int unsigned bit_idx = 0; bit_idx < 32; bit_idx++) begin
        src_idx = (word_idx * 32) + bit_idx;
        if (src_idx < TaskDescWidth) begin
          read_task_shadow_word[bit_idx] = bits[src_idx];
        end
      end
    end
  endfunction

  function automatic logic [31:0] read_done_word(
    input logic [DoneWidth-1:0] bits,
    input logic [7:0]           word_idx
  );
    int unsigned src_idx;
    begin
      read_done_word = '0;
      for (int unsigned bit_idx = 0; bit_idx < 32; bit_idx++) begin
        src_idx = (word_idx * 32) + bit_idx;
        if (src_idx < DoneWidth) begin
          read_done_word[bit_idx] = bits[src_idx];
        end
      end
    end
  endfunction

  assign dynamic_selected = csr_req_valid_i &&
      ((csr_req_i.addr & DYNAMIC_CSR_ADDR_MASK) == (DYNAMIC_CSR_BASE_ADDR & DYNAMIC_CSR_ADDR_MASK));
  assign dynamic_addr_delta = csr_req_i.addr - DYNAMIC_CSR_BASE_ADDR;
  assign csr_word_offset = dynamic_addr_delta[9:2];
  assign dynamic_read  = dynamic_selected && !csr_req_i.write;
  assign dynamic_write = dynamic_selected &&  csr_req_i.write;

  assign dynamic_task_commit = dynamic_write && (csr_word_offset == RegControl) && csr_req_i.data[0];
  assign dynamic_flush       = dynamic_write && (csr_word_offset == RegControl) && csr_req_i.data[1] && dynamic_ready;
  assign dynamic_done_pop    = dynamic_write && (csr_word_offset == RegControl) && csr_req_i.data[2] && dynamic_ready;
  assign dynamic_task_word_write = dynamic_write &&
      (csr_word_offset >= RegTaskBase) && (csr_word_offset < (RegTaskBase + TaskDescWords));

  assign dynamic_ready = dynamic_read ? csr_rsp_ready_i :
                         dynamic_task_commit ? task_push_ready : 1'b1;

  assign task_push_valid = dynamic_task_commit && csr_req_valid_i && dynamic_ready;
  assign done_pop_ready  = dynamic_done_pop && done_pop_valid;

  assign manager_csr_req_o       = csr_req_i;
  assign manager_csr_req_valid_o = csr_req_valid_i && !dynamic_selected;
  assign manager_csr_rsp_ready_o = csr_rsp_ready_i && !dynamic_selected;

  assign csr_req_ready_o = dynamic_selected ? dynamic_ready : manager_csr_req_ready_i;
  assign csr_rsp_valid_o = dynamic_selected ? dynamic_read : manager_csr_rsp_valid_i;
  assign csr_rsp_o       = dynamic_selected ? csr_rsp_t'{data: dynamic_read_data} : manager_csr_rsp_i;

  assign done_pop_bits = done_pop;

  assign dynamic_status[0]    = task_push_ready;
  assign dynamic_status[1]    = task_pop_valid_o;
  assign dynamic_status[2]    = done_pop_valid;
  assign dynamic_status[3]    = done_push_ready_o;
  assign dynamic_status[7:4]  = '0;
  assign dynamic_status[15:8] = TaskDescWords[7:0];
  assign dynamic_status[23:16] = DoneWords[7:0];
  assign dynamic_status[31:24] = '0;

  always_comb begin
    dynamic_read_data = '0;
    unique case (csr_word_offset)
      RegStatus:     dynamic_read_data = dynamic_status;
      RegTaskCredit: dynamic_read_data = task_credit;
      RegDoneCredit: dynamic_read_data = done_credit;
      default: begin
        if ((csr_word_offset >= RegDoneBase) && (csr_word_offset < (RegDoneBase + DoneWords))) begin
          dynamic_read_data = read_done_word(done_pop_bits, csr_word_offset - RegDoneBase);
        end else if ((csr_word_offset >= RegTaskBase) &&
                     (csr_word_offset < (RegTaskBase + TaskDescWords))) begin
          dynamic_read_data = read_task_shadow_word(task_shadow_q, csr_word_offset - RegTaskBase);
        end
      end
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      task_shadow_q <= '0;
    end else if (dynamic_flush) begin
      task_shadow_q <= '0;
    end else if (dynamic_task_word_write && dynamic_ready) begin
      for (int unsigned bit_idx = 0; bit_idx < 32; bit_idx++) begin
        int unsigned dst_idx;
        dst_idx = ((csr_word_offset - RegTaskBase) * 32) + bit_idx;
        if (dst_idx < TaskDescWidth) begin
          task_shadow_q[dst_idx] <= csr_req_i.data[bit_idx];
        end
      end
    end
  end

  hemaia_bingo_dynamic_task_queue #(
    .TASK_QUEUE_DEPTH(TASK_QUEUE_DEPTH),
    .DONE_QUEUE_DEPTH(DONE_QUEUE_DEPTH)
  ) i_dynamic_task_queue (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .flush_i(dynamic_flush),
    .task_push_desc_i(hemaia_bingo_dynamic_task_desc_t'(task_shadow_q)),
    .task_push_valid_i(task_push_valid),
    .task_push_ready_o(task_push_ready),
    .task_pop_desc_o(task_pop_desc_o),
    .task_pop_valid_o(task_pop_valid_o),
    .task_pop_ready_i(task_pop_ready_i),
    .task_credit_o(task_credit),
    .done_push_i(done_push_i),
    .done_push_valid_i(done_push_valid_i),
    .done_push_ready_o(done_push_ready_o),
    .done_pop_o(done_pop),
    .done_pop_valid_o(done_pop_valid),
    .done_pop_ready_i(done_pop_ready),
    .done_credit_o(done_credit)
  );

endmodule
