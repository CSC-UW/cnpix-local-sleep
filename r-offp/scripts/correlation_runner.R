#' Shared helpers for the NOD-rebound correlation analysis scripts.
#'
#' Sourced by run_correlation_analysis.R and run_all_correlation_analyses.R.
#' Expands the `metrics` entries of config/nod_rebound_correlation.yaml into calls
#' on offp::run_nod_rebound_correlation(), writing to the _output_correlations/
#' root.

CORRELATION_OUTPUT_ROOT <- "_output_correlations"
# The config's `alternatives` fits. They sit outside the four-level tree that the
# robustness diagnostics glob, because those refit y ~ x alone.
CORRELATION_ALTERNATIVES_ROOT <- file.path(CORRELATION_OUTPUT_ROOT, "alternatives")

# Run an expression, reporting (but not aborting on) errors so one bad fit does
# not stop the whole sweep.
correlation_try <- function(label, expr) {
  tryCatch(
    force(expr),
    error = function(e) message("  ERROR [", label, "]: ", conditionMessage(e)),
    warning = function(w) {
      message("  warning [", label, "]: ", conditionMessage(w))
      suppressWarnings(force(expr))
    }
  )
}

# Resolve a config value with a default.
.cfg_or <- function(value, default) if (is.null(value)) default else value

#' Run every analysis entry for one metric in one dataset.
#'
#' @param config Parsed correlation config (list).
#' @param dataset OFF type ("llas"/"clas"/"blas").
#' @param metric_entry One `metrics` entry (list with `metric`, optional
#'   `response_metric`, `rebound_scale`, `covariates`, `spearman`, and
#'   `analyses`).
#' @param data Optional pre-loaded correlation input for `dataset`.
#' @param response_data Optional pre-loaded response frame (band-power condition
#'   means) used when a metric entry sets a `response_metric` != `metric`.
#' @param output_root Root of the `<dataset>/<metric>/<predictor>/<model>/` tree.
run_correlation_metric <- function(config, dataset, metric_entry,
                                   data = NULL,
                                   response_data = NULL,
                                   output_root = CORRELATION_OUTPUT_ROOT) {
  metric <- metric_entry$metric
  # `response_metric` (optional) makes y the NREM.Rebound of a DIFFERENT quantity
  # than the predictor x (e.g. cortical delta power `mean_zlog_delta`, sourced
  # from the band-power condition means). Defaults to `metric` (self-rebound).
  response_metric <- .cfg_or(metric_entry$response_metric, metric)
  rebound_scale <- .cfg_or(metric_entry$rebound_scale, "difference")
  covariates <- as.character(unlist(metric_entry$covariates))
  spearman <- as.character(unlist(metric_entry$spearman))
  # `predictor_conditions` (list) is the current form; a scalar
  # `predictor_condition` is still honored for backward compatibility. The whole
  # metric is fit once per predictor, and outputs are separated by predictor at
  # the .../<metric>/<predictor>/ level so predictors never collide.
  pcs <- config$predictor_conditions
  if (is.null(pcs)) pcs <- .cfg_or(config$predictor_condition, "NOD")
  pcs <- as.character(unlist(pcs))
  rp <- .cfg_or(config$rebound_post, "Early.REC.NREM")
  rb <- .cfg_or(config$rebound_baseline, "Early.REC.NREM.Match")

  # Keep the fixed 4-level output depth (.../<dataset>/<metric>/<predictor>/
  # <model>/) that the diagnostic scripts glob: fold a cross-metric response into
  # the metric segment as `<metric>__vs__<response_metric>` (self-rebound is
  # unchanged), then a non-default rebound scale as `__<rebound_scale>` and each
  # covariate as `__adj__<covariate>`.
  path_metric <- metric
  if (!identical(response_metric, metric)) {
    path_metric <- paste0(path_metric, "__vs__", response_metric)
  }
  if (!identical(rebound_scale, "difference")) {
    path_metric <- paste0(path_metric, "__", rebound_scale)
  }
  for (covariate in covariates) {
    path_metric <- paste0(path_metric, "__adj__", covariate)
  }

  for (pc in pcs) {
    for (analysis in metric_entry$analyses) {
      model_def <- analysis$model
      # run_nod_rebound_correlation appends the <model_def$name>/ subdir.
      output_dir <- file.path(output_root, dataset, path_metric, pc)
      label <- paste(dataset, path_metric, pc, model_def$name)
      message("[correlation] ", label)
      correlation_try(label, offp::run_nod_rebound_correlation(
        metric = metric, off_type = dataset, model_def = model_def,
        output_dir = output_dir,
        predictor_condition = pc, rebound_post = rp, rebound_baseline = rb,
        response_metric = response_metric, rebound_scale = rebound_scale,
        covariates = covariates, spearman = spearman, data = data,
        response_data = response_data
      ))
    }
  }
}
