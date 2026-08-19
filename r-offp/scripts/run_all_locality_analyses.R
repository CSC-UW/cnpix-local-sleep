#!/usr/bin/env Rscript
#
# Run all locality (Local vs Overlapping) analyses defined in
# config/locality.yaml. Writes to _output_locality/.
#
# Usage:
#   Rscript scripts/run_all_locality_analyses.R [dataset]
#
# Arguments:
#   dataset:    "llas" or "all" (default; only llas is exported)
#
# Produces:
#   request 1: _output_locality/<ds>/overlap_degree/mean_overlap_degree/
#   request 2: .../cond-<condition>/<measure>/  and  .../interaction-{six,nrem,wake}/<measure>/
#   request 3: .../state-{NREM,Wake}/<measure>/  and  .../interaction-state/<measure>/

args <- commandArgs(trailingOnly = TRUE)
dataset_arg <- if (length(args) >= 1) args[1] else "all"

params_path <- file.path("config", "locality.yaml")
if (!file.exists(params_path)) {
  stop("Config file not found: ", params_path)
}

library(offp)
source(file.path("scripts", "locality_runner.R"))

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

for (dataset in datasets_to_run) {
  rvs <- config$datasets[[dataset]]$response_variables
  message("=== Dataset: ", dataset, " (", length(rvs),
          " measures + overlap degree) ===")

  # Load + prepare each locality dataset once per OFF type.
  data_overlap <- offp::load_locality_overlap_summary()
  data_pc <- offp::prepare_locality_data(
    offp::load_locality_per_condition_summary(dataset)
  )
  data_state <- offp::prepare_locality_data(
    offp::load_locality_full48h_summary(dataset)
  )

  # Request 1: condition contrasts on the mean # of overlapping structures.
  run_overlap_degree(config, dataset, data = data_overlap)

  # Requests 2 & 3 for each measure.
  for (rv in rvs) {
    run_locality_rv(config, dataset, rv,
                    data_pc = data_pc, data_state = data_state)
  }
}

message("All locality analyses complete.")
