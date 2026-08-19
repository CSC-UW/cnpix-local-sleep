#!/usr/bin/env Rscript
#
# Run all cx_homeostasis analyses defined in the YAML config file.
# Each response variable is analyzed under its specified model definitions.
#
# Usage:
#   Rscript scripts/run_all_analyses.R [dataset]
#
# Arguments:
#   dataset:    "llas", "clas", "blas", or "all" (default)
#
# Examples:
#   Rscript scripts/run_all_analyses.R       # all datasets
#   Rscript scripts/run_all_analyses.R llas  # just LLAS
#   Rscript scripts/run_all_analyses.R blas  # just BLAS

args <- commandArgs(trailingOnly = TRUE)
dataset_arg <- if (length(args) >= 1) args[1] else "all"

params_path <- file.path("config", "cx_homeostasis.yaml")

if (!file.exists(params_path)) {
  stop("Config file not found: ", params_path)
}

library(offp)

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
  dataset_config <- config$datasets[[dataset]]
  response_vars <- dataset_config$response_variables

  n_analyses <- sum(vapply(
    response_vars,
    function(rv) length(rv$analyses),
    integer(1)
  ))
  message(
    "Dataset: ", dataset, " - ",
    length(response_vars), " response variables, ",
    n_analyses, " analyses"
  )

  counter <- 0
  for (rv in response_vars) {
    for (analysis in rv$analyses) {
      cs <- analysis$condition_set
      model_def <- analysis$model
      output_dir <- file.path(
        "_output", dataset, cs$name, rv$response_variable
      )

      counter <- counter + 1
      message(
        sprintf(
          "[%d/%d] %s [%s/%s] (%s) -> %s/%s",
          counter, n_analyses, rv$response_variable,
          cs$name, model_def$name, dataset,
          output_dir, model_def$name
        )
      )

      # Log and continue, as every sibling batch runner does: a sweep of this
      # size hits the occasional non-converging fit, and losing the rest of the
      # run to it means re-running everything that already succeeded.
      tryCatch(
        offp::run_cx_homeostasis_analysis(
          response_var = rv$response_variable,
          output_dir = output_dir,
          model_def = model_def,
          off_type = dataset,
          condition_set = cs$name,
          conditions = cs$conditions,
          posthocs = cs$posthocs
        ),
        error = function(e) {
          message(
            "  ERROR [", rv$response_variable, " ", cs$name, "/",
            model_def$name, "]: ", conditionMessage(e)
          )
        }
      )
    }
  }
}

message("All analyses complete.")
