#!/usr/bin/env Rscript
# Export exact mixed-model inputs and marginal expectations for the Python Figma
# renderer. The companion plot_correlation_xtransform_publication.py script
# consumes these JSON files and writes the publication SVGs.

suppressWarnings(suppressMessages({
  library(arrow)
  library(lme4)
  library(modelbased)
}))

CORR_ROOT <- "_output_correlations"
PREDICTORS <- c("NOD", "NOD.Wake")
METRICS <- c("rate", "total_area_norm")
OFF_TYPES <- c("llas", "clas")
TRANSFORMS <- list(
  raw = list(fn = function(x) x, label = "raw"),
  rank = list(fn = function(x) rank(x), label = "rank"),
  log = list(fn = function(x) log(x), label = "log")
)

# Matplotlib/seaborn's tab20 palette, used by incline_magnitudes.ipynb.
TAB20 <- c(
  "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
  "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
  "#17becf", "#9edae5"
)

fit_panel <- function(result, transform) {
  d <- result$data
  d$subject <- as.character(d$subject)
  d$structure <- as.character(d$structure)
  d$x <- transform$fn(d$x)
  re_terms <- result$model_def$re_terms
  full_formula <- reformulate(c("x", re_terms), response = "y")
  null_formula <- reformulate(c("1", re_terms), response = "y")
  model <- suppressWarnings(lme4::lmer(full_formula, d, REML = FALSE))
  null_model <- suppressWarnings(lme4::lmer(null_formula, d, REML = FALSE))
  p <- suppressWarnings(stats::anova(model, null_model))[["Pr(>Chisq)"]][2]
  prediction <- as.data.frame(
    modelbased::estimate_expectation(model, by = "x", length = 100)
  )
  list(data = d, prediction = prediction, p = p, transform = transform)
}

build_structure_colors <- function() {
  metadata <- arrow::read_parquet(
    "inst/extdata/nod_rebound_correlation_llas_offs.parquet",
    col_select = c("structure", "AP.Coord")
  )
  metadata <- unique(metadata)
  metadata <- metadata[order(metadata[["AP.Coord"]]), ]
  structures <- as.character(metadata$structure)
  stopifnot(length(structures) <= length(TAB20))
  stats::setNames(TAB20[seq_along(structures)], structures)
}

build_subject_order <- function() {
  files <- Sys.glob(file.path(CORR_ROOT, "*", "*", "*", "*", "results.rds"))
  sort(unique(unlist(lapply(files, function(path) {
    as.character(readRDS(path)$data$subject)
  }), use.names = FALSE)))
}

serialize_panel <- function(panel, metric, off_type, transform_name) {
  list(
    metric = metric,
    off_type = off_type,
    transform = transform_name,
    pval = panel$p,
    data = panel$data[, c("x", "y", "structure", "subject")],
    prediction = panel$prediction[, c("x", "Predicted", "CI_low", "CI_high")]
  )
}

structure_colors <- build_structure_colors()
subject_order <- build_subject_order()
output_dir <- file.path(CORR_ROOT, "xtransform_sensitivity")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (predictor in PREDICTORS) {
  panels <- list()
  for (metric in METRICS) {
    for (off_type in OFF_TYPES) {
      result_path <- file.path(CORR_ROOT, off_type, metric, predictor, "subject", "results.rds")
      stopifnot(file.exists(result_path))
      result <- readRDS(result_path)
      stopifnot(identical(result$metric, metric), identical(result$response_metric, metric))
      for (transform_name in names(TRANSFORMS)) {
        panel <- fit_panel(result, TRANSFORMS[[transform_name]])
        panels[[length(panels) + 1]] <- serialize_panel(
          panel, metric, off_type, transform_name
        )
      }
    }
  }
  output_path <- file.path(output_dir, paste0("publication_grid_", predictor, ".json"))
  jsonlite::write_json(
    list(
      predictor = predictor,
      structure_colors = as.list(structure_colors),
      subject_order = subject_order,
      panels = panels
    ),
    output_path, auto_unbox = TRUE, pretty = TRUE, dataframe = "rows"
  )
  cat("wrote", output_path, "\n")
}
