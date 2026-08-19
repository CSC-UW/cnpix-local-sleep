#!/usr/bin/env Rscript
#
# Run the NOD-activity vs NREM.Rebound correlation analyses for ONE metric.
# Writes to _output_correlations/.
#
# Usage:
#   Rscript scripts/run_correlation_analysis.R <metric> [model] [dataset]
#     [params_file] [--response-metric=<name>]
#
# Arguments:
#   metric:      predictor (x) metric column name (e.g. "total_area_norm", "rate")
#   model:       "all" (default), "subject", or "subject_structure"
#   dataset:     "llas" (default), "clas", or "blas"
#   params_file: path to YAML config (default: config/nod_rebound_correlation.yaml)
#   --response-metric=<name>: response (y) metric. Selects the cross-metric entry
#     whose response is <name> (e.g. mean_zlog_delta); defaults to <metric>
#     (self-rebound). Order-independent flag, so existing positional calls are
#     unaffected.
#
# Examples:
#   Rscript scripts/run_correlation_analysis.R total_area_norm
#   Rscript scripts/run_correlation_analysis.R rate subject clas
#   Rscript scripts/run_correlation_analysis.R rate all clas --response-metric=mean_zlog_delta

args <- commandArgs(trailingOnly = TRUE)

# Pull the order-independent --response-metric[=/ ]<name> flag out of args before
# positional parsing so it does not shift the positional slots.
response_metric_arg <- NULL
flag_idx <- grep("^--response[-_]metric(=.*)?$", args)
if (length(flag_idx) > 0) {
  fi <- flag_idx[1]
  if (grepl("=", args[fi])) {
    response_metric_arg <- sub("^--response[-_]metric=", "", args[fi])
    args <- args[-fi]
  } else {
    # "--response-metric <name>" form
    assertthat::assert_that(length(args) >= fi + 1,
      msg = "--response-metric requires a value")
    response_metric_arg <- args[fi + 1]
    args <- args[-c(fi, fi + 1)]
  }
}

if (length(args) < 1) {
  stop(
    "Usage: Rscript scripts/run_correlation_analysis.R <metric> ",
    "[model] [dataset] [params_file] [--response-metric=<name>]"
  )
}

metric <- args[1]
model_arg <- if (length(args) >= 2) args[2] else "all"
dataset <- if (length(args) >= 3) args[3] else "llas"
params_file <- if (length(args) >= 4) {
  args[4]
} else {
  file.path("config", "nod_rebound_correlation.yaml")
}
# Default response = predictor metric (self-rebound).
response_metric <- if (is.null(response_metric_arg)) metric else response_metric_arg

if (!file.exists(params_file)) {
  stop("Parameter file not found: ", params_file)
}

library(offp)
source(file.path("scripts", "correlation_runner.R"))

config <- yaml::read_yaml(params_file)
dataset_config <- config$datasets[[dataset]]
if (is.null(dataset_config)) {
  stop(
    "Dataset '", dataset, "' not found in ", params_file, "\n",
    "Available: ", paste(names(config$datasets), collapse = ", ")
  )
}

# Select the entry matching BOTH the predictor metric and the effective response
# metric (an omitted `response_metric:` in the config means self-rebound).
entry_label <- function(me) {
  rm <- if (is.null(me$response_metric)) me$metric else me$response_metric
  if (identical(rm, me$metric)) me$metric else paste0(me$metric, " -> ", rm)
}
metric_entry <- NULL
for (me in dataset_config$metrics) {
  me_response <- if (is.null(me$response_metric)) me$metric else me$response_metric
  if (me$metric == metric && identical(me_response, response_metric)) {
    metric_entry <- me
    break
  }
}
if (is.null(metric_entry)) {
  available <- vapply(dataset_config$metrics, entry_label, character(1))
  stop(
    "No entry with predictor metric '", metric, "' and response metric '",
    response_metric, "' for dataset '", dataset, "' in ", params_file, "\n",
    "Available: ", paste(available, collapse = ", ")
  )
}

# Optionally filter the analysis entries by model name.
if (model_arg != "all") {
  metric_entry$analyses <- Filter(
    function(a) a$model$name == model_arg, metric_entry$analyses
  )
  if (length(metric_entry$analyses) == 0) {
    stop("No analyses with model '", model_arg, "' for metric '", metric, "'")
  }
}

dat <- offp::load_nod_rebound_correlation_data(dataset)
# Load the band-power condition means only if this entry requests a cross-metric
# response (response_metric != metric).
response_data <- if (!is.null(metric_entry$response_metric) &&
                     !identical(metric_entry$response_metric, metric_entry$metric)) {
  offp::load_bandpower_condition_means()
} else {
  NULL
}
run_correlation_metric(config, dataset, metric_entry,
                       data = dat, response_data = response_data)

message("Done.")
