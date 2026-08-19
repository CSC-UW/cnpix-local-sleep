#!/usr/bin/env Rscript
#
# Run the locality (Local vs Overlapping) analyses for ONE response variable.
#
# Usage:
#   Rscript scripts/run_locality_analysis.R <response_var> [kind] [dataset]
#     [params_file]
#
# Arguments:
#   response_var: a measure (median_duration / median_span / median_area), or
#                 "mean_overlap_degree" for the request-1 condition contrasts
#   kind:         "all" (default), "per_condition", "interaction", "state",
#                 or "state_interaction"; filters which analysis entries to run
#                 (ignored for mean_overlap_degree)
#   dataset:      "llas" (default; only llas is exported)
#   params_file:  path to YAML config (default: config/locality.yaml)
#
# Examples:
#   Rscript scripts/run_locality_analysis.R mean_overlap_degree
#   Rscript scripts/run_locality_analysis.R median_duration per_condition
#   Rscript scripts/run_locality_analysis.R median_span state

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop(
    "Usage: Rscript scripts/run_locality_analysis.R <response_var> ",
    "[kind] [dataset] [params_file]"
  )
}

response_var <- args[1]
kind_arg <- if (length(args) >= 2) args[2] else "all"
dataset <- if (length(args) >= 3) args[3] else "llas"
params_file <- if (length(args) >= 4) {
  args[4]
} else {
  file.path("config", "locality.yaml")
}

if (!file.exists(params_file)) {
  stop("Parameter file not found: ", params_file)
}

library(offp)
source(file.path("scripts", "locality_runner.R"))

config <- yaml::read_yaml(params_file)
dataset_config <- config$datasets[[dataset]]
if (is.null(dataset_config)) {
  stop(
    "Dataset '", dataset, "' not found in ", params_file, "\n",
    "Available: ", paste(names(config$datasets), collapse = ", ")
  )
}

# Request 1: the mean # overlapping structures (condition contrasts).
if (response_var == config$overlap_degree$response_variable) {
  data_overlap <- offp::load_locality_overlap_summary()
  run_overlap_degree(config, dataset, data = data_overlap)
  message("Done.")
  quit(save = "no")
}

rv_entry <- NULL
for (rv in dataset_config$response_variables) {
  if (rv$response_variable == response_var) {
    rv_entry <- rv
    break
  }
}
if (is.null(rv_entry)) {
  available <- vapply(
    dataset_config$response_variables,
    function(rv) rv$response_variable, character(1)
  )
  stop(
    "Response variable '", response_var, "' not found for dataset '",
    dataset, "' in ", params_file, "\n",
    "Available: ", paste(c(available, config$overlap_degree$response_variable),
                         collapse = ", ")
  )
}

# Optionally filter the analysis entries by kind.
if (kind_arg != "all") {
  rv_entry$analyses <- Filter(
    function(a) a$kind == kind_arg, rv_entry$analyses
  )
  if (length(rv_entry$analyses) == 0) {
    stop("No analyses of kind '", kind_arg, "' for '", response_var, "'")
  }
}

data_pc <- offp::prepare_locality_data(
  offp::load_locality_per_condition_summary(dataset)
)
data_state <- offp::prepare_locality_data(
  offp::load_locality_full48h_summary(dataset)
)
run_locality_rv(config, dataset, rv_entry,
                data_pc = data_pc, data_state = data_state)

message("Done.")
