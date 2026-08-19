#!/usr/bin/env Rscript
#
# Run all NOD-activity vs NREM.Rebound correlation analyses defined in
# config/nod_rebound_correlation.yaml. Writes to _output_correlations/.
#
# Usage:
#   Rscript scripts/run_all_correlation_analyses.R [dataset]
#
# Arguments:
#   dataset:    "llas", "clas", "blas", or "all" (default)

args <- commandArgs(trailingOnly = TRUE)
dataset_arg <- if (length(args) >= 1) args[1] else "all"

params_path <- file.path("config", "nod_rebound_correlation.yaml")
if (!file.exists(params_path)) {
  stop("Config file not found: ", params_path)
}

library(offp)
source(file.path("scripts", "correlation_runner.R"))

config <- yaml::read_yaml(params_path)

if (dataset_arg == "all") {
  datasets_to_run <- names(config$datasets)
} else {
  if (is.null(config$datasets[[dataset_arg]])) {
    stop(
      "Dataset '", dataset_arg, "' not found in ", params_path, "\n",
      "Available: ", paste(names(config$datasets), collapse = ", ")
    )
  }
  datasets_to_run <- dataset_arg
}

# Load the band-power condition means once (shared response source) iff any
# metric entry requests a cross-metric response.
needs_response_data <- any(vapply(
  datasets_to_run,
  function(ds) any(vapply(
    config$datasets[[ds]]$metrics,
    function(me) !is.null(me$response_metric) &&
      !identical(me$response_metric, me$metric),
    logical(1)
  )),
  logical(1)
))
response_data <- if (needs_response_data) {
  offp::load_bandpower_condition_means()
} else {
  NULL
}

for (dataset in datasets_to_run) {
  metrics <- config$datasets[[dataset]]$metrics
  message("=== Dataset: ", dataset, " (", length(metrics), " metrics) ===")
  # Load the correlation input once per dataset.
  dat <- offp::load_nod_rebound_correlation_data(dataset)
  for (me in metrics) {
    run_correlation_metric(config, dataset, me,
                           data = dat, response_data = response_data)
  }
}

message("All correlation analyses complete.")
