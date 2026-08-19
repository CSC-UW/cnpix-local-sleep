#!/usr/bin/env Rscript
#
# Run the cross-structure "excess globality" (observed vs windowed null)
# analysis. A single-analysis separable companion: it loads the
# summarized_excess_globality_offs.parquet data, fits the paired observed-vs-null
# main-effect model for whole-recording NREM, and writes to
# _output_excess_globality/. Delete this script (plus the excess-globality R
# files, config, test, and _output_excess_globality/) to remove the feature.
#
# Usage:
#   Rscript scripts/run_excess_globality_analysis.R [dataset] [params_file]
#
# Arguments:
#   dataset:     "llas" (default; only llas is exported)
#   params_file: path to YAML config (default: config/excess_globality.yaml)

args <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[1] else "llas"
params_file <- if (length(args) >= 2) {
  args[2]
} else {
  file.path("config", "excess_globality.yaml")
}

if (!file.exists(params_file)) {
  stop("Parameter file not found: ", params_file)
}

library(offp)

EXCESS_OUTPUT_ROOT <- "_output_excess_globality"

config <- yaml::read_yaml(params_file)
dataset_config <- config$datasets[[dataset]]
if (is.null(dataset_config)) {
  stop(
    "Dataset '", dataset, "' not found in ", params_file, "\n",
    "Available: ", paste(names(config$datasets), collapse = ", ")
  )
}

contrast <- if (is.null(config$quantity_contrast)) {
  "observed - null = 0"
} else {
  config$quantity_contrast
}
states <- if (is.null(config$states)) "NREM" else config$states

data_state <- offp::prepare_excess_globality_data(
  offp::load_excess_globality_summary()
)

for (rv_entry in dataset_config$response_variables) {
  response_var <- rv_entry$response_variable
  for (analysis in rv_entry$analyses) {
    if (analysis$kind != "state") {
      stop("Unknown analysis kind: ", analysis$kind)
    }
    model_def <- analysis$model
    for (st in states) {
      ak <- paste0("state-", st)
      output_dir <- file.path(EXCESS_OUTPUT_ROOT, dataset, ak,
                              response_var)
      label <- paste(dataset, response_var, ak, model_def$name)
      message("[excess-globality] ", label)
      tryCatch(
        offp::run_excess_globality_analysis(
          response_var = response_var, output_dir = output_dir,
          model_def = model_def, off_type = dataset,
          analysis_kind = ak, conditions = st,
          excess_posthocs = contrast, data = data_state
        ),
        error = function(e) {
          message("  ERROR [", label, "]: ", conditionMessage(e))
        }
      )
    }
  }
}

message("Done.")
