open_project hemaia_system/hemaia_system.xpr
open_run impl_1

report_timing_summary \
  -nworst 20 \
  -file current_route_timing_summary.rpt

report_timing \
  -delay_type max \
  -max_paths 100 \
  -sort_by slack \
  -file current_route_timing_paths.rpt

report_high_fanout_nets \
  -fanout_greater_than 50 \
  -max_nets 300 \
  -file current_route_high_fanout.rpt

report_design_analysis \
  -congestion \
  -complexity \
  -file current_route_design_analysis.rpt
