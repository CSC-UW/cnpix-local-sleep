#!/usr/bin/env Rscript
#
# Run every configured band-power condition-homeostasis analysis.
#
# Iterates all response variables x analyses in config/bandpower_homeostasis.yaml
# and writes to _output_bandpower/. See run_bandpower_analysis.R for details.
#
# Usage:
#   Rscript scripts/run_all_bandpower_analyses.R [params_file]

args <- commandArgs(trailingOnly = TRUE)
params_file <- if (length(args) >= 1) {
  args[1]
} else {
  file.path("config", "bandpower_homeostasis.yaml")
}

if (!file.exists(params_file)) {
  stop("Parameter file not found: ", params_file)
}

library(offp)

BANDPOWER_OUTPUT_ROOT <- "_output_bandpower"
DATASET <- "bandpower"

config <- yaml::read_yaml(params_file)
dataset_config <- config$datasets[[DATASET]]
if (is.null(dataset_config)) {
  stop("Dataset '", DATASET, "' not found in ", params_file)
}

data_all <- offp::load_bandpower_condition_means()
data_all <- dplyr::filter(data_all, .data[["clade"]] == "Cx")

for (rv_entry in dataset_config$response_variables) {
  response_var <- rv_entry$response_variable
  for (analysis in rv_entry$analyses) {
    cs <- analysis$condition_set
    model_def <- analysis$model
    output_dir <- file.path(
      BANDPOWER_OUTPUT_ROOT, DATASET, cs$name, response_var
    )
    label <- paste0(response_var, " [", cs$name, "/", model_def$name, "]")
    message("[bandpower] ", label, " -> ", output_dir)
    tryCatch(
      offp::run_cx_homeostasis_analysis(
        response_var = response_var,
        output_dir = output_dir,
        model_def = model_def,
        off_type = DATASET,
        condition_set = cs$name,
        conditions = cs$conditions,
        posthocs = cs$posthocs,
        data = data_all
      ),
      error = function(e) {
        message("  ERROR [", label, "]: ", conditionMessage(e))
      }
    )
  }
}

message("Done.")
