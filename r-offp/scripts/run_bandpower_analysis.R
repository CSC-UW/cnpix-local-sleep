#!/usr/bin/env Rscript
#
# Run a band-power condition-homeostasis analysis for one response variable.
#
# Separable companion to run_analysis.R: the measure is a per-condition mean of
# the z-scored log10 instantaneous bipolar band power (not an OFF property), but
# it is fit through the identical condition-homeostasis model. Loads
# summarized_full48h_bandpower_offs.parquet, passes it in as pre-loaded `data`,
# and writes to _output_bandpower/. Delete this script (plus
# run_all_bandpower_analyses.R, config/bandpower_homeostasis.yaml, the
# bandpower plot/summary configs, and _output_bandpower/) to remove the feature.
#
# Usage:
#   Rscript scripts/run_bandpower_analysis.R <response_var> [model_name]
#     [condition_set] [params_file]
#
# Arguments:
#   response_var:  column name of the response variable (e.g. mean_zlog_delta)
#   model_name:    model name from the config, or "all" (default)
#   condition_set: "six"/"nrem"/"wake" or "all" (default)
#   params_file:   path to YAML config (default: config/bandpower_homeostasis.yaml)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop(
    "Usage: Rscript scripts/run_bandpower_analysis.R ",
    "<response_var> [model_name] [condition_set] [params_file]"
  )
}

response_var <- args[1]
model_name <- if (length(args) >= 2) args[2] else "all"
condition_set_name <- if (length(args) >= 3) args[3] else "all"
params_file <- if (length(args) >= 4) {
  args[4]
} else {
  file.path("config", "bandpower_homeostasis.yaml")
}

if (!file.exists(params_file)) {
  stop("Parameter file not found: ", params_file)
}

library(offp)

# Provenance label recorded in outputs; the band power is a property of the
# whole 48h recording, and the dataset has no llas/clas/blas split.
BANDPOWER_OUTPUT_ROOT <- "_output_bandpower"
DATASET <- "bandpower"

config <- yaml::read_yaml(params_file)
dataset_config <- config$datasets[[DATASET]]
if (is.null(dataset_config)) {
  stop("Dataset '", DATASET, "' not found in ", params_file)
}

# Find the entry for this response variable.
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
    "Response variable '", response_var, "' not found in ", params_file, "\n",
    "Available: ", paste(available, collapse = ", ")
  )
}

# Select analyses (condition_set x model), filtering by optional args.
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
  stop(
    "No analyses match model='", model_name, "', condition_set='",
    condition_set_name, "' for response variable '", response_var, "'"
  )
}

# Load the band-power condition means once; restrict to cortical rows (all rows
# are cortical by construction, but mirror the OFF Cx filter for parity).
data_all <- offp::load_bandpower_condition_means()
data_all <- dplyr::filter(data_all, .data[["clade"]] == "Cx")

for (analysis in analyses_to_run) {
  cs <- analysis$condition_set
  model_def <- analysis$model
  output_dir <- file.path(
    BANDPOWER_OUTPUT_ROOT, DATASET, cs$name, response_var
  )
  message(
    "Running: ", response_var, " [", cs$name, "/", model_def$name, "] ",
    "(", DATASET, ") -> ", output_dir
  )
  offp::run_cx_homeostasis_analysis(
    response_var = response_var,
    output_dir = output_dir,
    model_def = model_def,
    off_type = DATASET,
    condition_set = cs$name,
    conditions = cs$conditions,
    posthocs = cs$posthocs,
    data = data_all
  )
}

message("Done.")
