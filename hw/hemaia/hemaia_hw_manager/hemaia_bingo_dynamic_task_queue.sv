// Copyright 2026 KU Leuven.
// Licensed under the Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

module hemaia_bingo_dynamic_task_queue
  import hemaia_bingo_dynamic_task_pkg::*;
#(
  parameter int unsigned TASK_QUEUE_DEPTH = 8,
  parameter int unsigned DONE_QUEUE_DEPTH = 8
) (
  input  logic                             clk_i,
  input  logic                             rst_ni,
  input  logic                             flush_i,

  input  hemaia_bingo_dynamic_task_desc_t  task_push_desc_i,
  input  logic                             task_push_valid_i,
  output logic                             task_push_ready_o,
  output hemaia_bingo_dynamic_task_desc_t  task_pop_desc_o,
  output logic                             task_pop_valid_o,
  input  logic                             task_pop_ready_i,
  output logic [31:0]                      task_credit_o,

  input  hemaia_bingo_dynamic_done_t       done_push_i,
  input  logic                             done_push_valid_i,
  output logic                             done_push_ready_o,
  output hemaia_bingo_dynamic_done_t       done_pop_o,
  output logic                             done_pop_valid_o,
  input  logic                             done_pop_ready_i,
  output logic [31:0]                      done_credit_o
);

  localparam int unsigned TaskPtrWidth = (TASK_QUEUE_DEPTH > 1) ? $clog2(TASK_QUEUE_DEPTH) : 1;
  localparam int unsigned DonePtrWidth = (DONE_QUEUE_DEPTH > 1) ? $clog2(DONE_QUEUE_DEPTH) : 1;
  localparam int unsigned TaskCntWidth = $clog2(TASK_QUEUE_DEPTH + 1);
  localparam int unsigned DoneCntWidth = $clog2(DONE_QUEUE_DEPTH + 1);
  localparam logic [TaskCntWidth-1:0] TaskQueueDepth = TASK_QUEUE_DEPTH;
  localparam logic [DoneCntWidth-1:0] DoneQueueDepth = DONE_QUEUE_DEPTH;
  localparam logic [31:0] TaskQueueDepth32 = TASK_QUEUE_DEPTH;
  localparam logic [31:0] DoneQueueDepth32 = DONE_QUEUE_DEPTH;

  hemaia_bingo_dynamic_task_desc_t task_mem_q [TASK_QUEUE_DEPTH];
  hemaia_bingo_dynamic_done_t      done_mem_q [DONE_QUEUE_DEPTH];

  logic [TaskPtrWidth-1:0] task_wr_q, task_rd_q;
  logic [DonePtrWidth-1:0] done_wr_q, done_rd_q;
  logic [TaskCntWidth-1:0] task_count_q;
  logic [DoneCntWidth-1:0] done_count_q;

  logic task_push_fire, task_pop_fire;
  logic done_push_fire, done_pop_fire;

  function automatic logic [TaskPtrWidth-1:0] task_ptr_inc(
    input logic [TaskPtrWidth-1:0] ptr
  );
    if (ptr == TASK_QUEUE_DEPTH - 1) begin
      task_ptr_inc = '0;
    end else begin
      task_ptr_inc = ptr + 1'b1;
    end
  endfunction

  function automatic logic [DonePtrWidth-1:0] done_ptr_inc(
    input logic [DonePtrWidth-1:0] ptr
  );
    if (ptr == DONE_QUEUE_DEPTH - 1) begin
      done_ptr_inc = '0;
    end else begin
      done_ptr_inc = ptr + 1'b1;
    end
  endfunction

  assign task_push_ready_o = (task_count_q < TaskQueueDepth);
  assign task_pop_valid_o  = (task_count_q != '0);
  assign task_push_fire    = task_push_valid_i & task_push_ready_o;
  assign task_pop_fire     = task_pop_valid_o & task_pop_ready_i;
  assign task_pop_desc_o   = task_mem_q[task_rd_q];
  assign task_credit_o     = TaskQueueDepth32 - {{(32-TaskCntWidth){1'b0}}, task_count_q};

  assign done_push_ready_o = (done_count_q < DoneQueueDepth);
  assign done_pop_valid_o  = (done_count_q != '0);
  assign done_push_fire    = done_push_valid_i & done_push_ready_o;
  assign done_pop_fire     = done_pop_valid_o & done_pop_ready_i;
  assign done_pop_o        = done_mem_q[done_rd_q];
  assign done_credit_o     = DoneQueueDepth32 - {{(32-DoneCntWidth){1'b0}}, done_count_q};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      task_wr_q    <= '0;
      task_rd_q    <= '0;
      task_count_q <= '0;
      done_wr_q    <= '0;
      done_rd_q    <= '0;
      done_count_q <= '0;
    end else if (flush_i) begin
      task_wr_q    <= '0;
      task_rd_q    <= '0;
      task_count_q <= '0;
      done_wr_q    <= '0;
      done_rd_q    <= '0;
      done_count_q <= '0;
    end else begin
      if (task_push_fire) begin
        task_mem_q[task_wr_q] <= task_push_desc_i;
        task_wr_q <= task_ptr_inc(task_wr_q);
      end
      if (task_pop_fire) begin
        task_rd_q <= task_ptr_inc(task_rd_q);
      end
      unique case ({task_push_fire, task_pop_fire})
        2'b10: task_count_q <= task_count_q + 1'b1;
        2'b01: task_count_q <= task_count_q - 1'b1;
        default: task_count_q <= task_count_q;
      endcase

      if (done_push_fire) begin
        done_mem_q[done_wr_q] <= done_push_i;
        done_wr_q <= done_ptr_inc(done_wr_q);
      end
      if (done_pop_fire) begin
        done_rd_q <= done_ptr_inc(done_rd_q);
      end
      unique case ({done_push_fire, done_pop_fire})
        2'b10: done_count_q <= done_count_q + 1'b1;
        2'b01: done_count_q <= done_count_q - 1'b1;
        default: done_count_q <= done_count_q;
      endcase
    end
  end

endmodule
