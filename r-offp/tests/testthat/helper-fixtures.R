# Shared test fixtures.

# The cortical, layer-agnostic, spatial analysis frame, the same restriction
# `run_cx_homeostasis_analysis()` applies internally, via the same helper, so a
# schema change lands in one place instead of at every call site.
cx_offs_summary <- function(off_type = "llas") {
  offp:::filter_layer_agnostic_cx(load_offs_summary(off_type))
}
