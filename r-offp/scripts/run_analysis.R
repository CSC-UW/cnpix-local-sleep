#!/usr/bin/env Rscript
#
# Run a cx_homeostasis analysis for one response variable.
# Model definitions and dataset configs are read from the YAML config file.
#
# Usage:
#   Rscript scripts/run_analysis.R <response_var> [model_name] [dataset]
#     [condition_set] [params_file]
#
# Arguments:
#   response_var:  column name of the response variable
#   model_name:    name of a model definition from the config file,
#                  or "all" to run all models for this variable (default)
#   dataset:       "llas" (default), "clas", or "blas"
#   condition_set: name of a condition set (e.g. "six", "nrem", "wake"),
#                  or "all" to run every set configured for this variable
#                  (default)
#   params_file:   path to YAML config file
#                  (default: config/cx_homeostasis.yaml)
#
# Examples:
#   Rscript scripts/run_analysis.R rate
#   Rscript scripts/run_analysis.R rate crossed
#   Rscript scripts/run_analysis.R rate all clas
#   Rscript scripts/run_analysis.R rate all llas wake
#   Rscript scripts/run_analysis.R rate crossed blas nrem config/custom.yaml

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop(
    "Usage: Rscript scripts/run_analysis.R ",
    "<response_var> [model_name] [dataset] ",
    "[condition_set] [params_file]\n",
    "  response_var:  column name of the response variable\n",
    "  model_name:    model name or 'all' (default)\n",
    "  dataset:       'llas' (default), 'clas', or 'blas'\n",
    "  condition_set: 'six'/'nrem'/'wake' or 'all' (default)\n",
    "  params_file:   path to YAML config file (default: auto)"
  )
}

response_var <- args[1]
model_name <- if (length(args) >= 2) args[2] else "all"
dataset <- if (length(args) >= 3) args[3] else "llas"
condition_set_name <- if (length(args) >= 4) args[4] else "all"
params_file <- if (length(args) >= 5) {
  args[5]
} else {
  file.path("config", "cx_homeostasis.yaml")
}

if (!file.exists(params_file)) {
  stop("Parameter file not found: ", params_file)
}

library(offp)

config <- yaml::read_yaml(params_file)
dataset_config <- config$datasets[[dataset]]
if (is.null(dataset_config)) {
  stop(
    "Dataset '", dataset, "' not found in ", params_file, "\n",
    "Available: ", paste(names(config$datasets), collapse = ", ")
  )
}

response_vars <- dataset_config$response_variables

# Find the entry for this response variable
rv_entry <- NULL
for (rv in response_vars) {
  if (rv$response_variable == response_var) {
    rv_entry <- rv
    break
  }
}
if (is.null(rv_entry)) {
  available <- vapply(response_vars, function(rv) rv$response_variable,
                      character(1))
  stop(
    "Response variable '", response_var,
    "' not found for dataset '", dataset, "' in ", params_file, "\n",
    "Available: ", paste(available, collapse = ", ")
  )
}

# Select analyses (condition_set x model) to run, filtering by the optional
# model_name and condition_set args.
analyses_to_run <- rv_entry$analyses
if (model_name != "all") {
  analyses_to_run <- Filter(
    function(a) a$model$name == model_name, analyses_to_run
  )
}
if (condition_set_name != "all") {
  analyses_to_run <- Filter(
    function(a) a$condition_set$name == condition_set_name, analyses_to_run
  )
}
if (length(analyses_to_run) == 0) {
  avail <- vapply(
    rv_entry$analyses,
    function(a) paste0(a$condition_set$name, "/", a$model$name),
    character(1)
  )
  stop(
    "No analyses match model='", model_name, "', condition_set='",
    condition_set_name, "' for response variable '", response_var, "'\n",
    "Available (condition_set/model): ", paste(avail, collapse = ", ")
  )
}

for (analysis in analyses_to_run) {
  cs <- analysis$condition_set
  model_def <- analysis$model
  output_dir <- file.path(
    "_output", dataset, cs$name, response_var
  )
  message(
    "Running: ", response_var, " [", cs$name, "/", model_def$name, "] ",
    "(", dataset, ") -> ", output_dir
  )
  offp::run_cx_homeostasis_analysis(
    response_var = response_var,
    output_dir = output_dir,
    model_def = model_def,
    off_type = dataset,
    condition_set = cs$name,
    conditions = cs$conditions,
    posthocs = cs$posthocs
  )
}

message("Done.")
