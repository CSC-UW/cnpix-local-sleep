# NOD-activity vs NREM.Rebound correlation pipeline.
#
# A self-contained companion to the `cx_homeostasis` (`R/analysis.R`) and
# locality (`R/locality_analysis.R`) pipelines. It tests whether a whole-period
# NOD OFF metric (the predictor `x`) predicts the same metric's `NREM.Rebound`
# (the response `y = Early.REC.NREM - Early.REC.NREM.Match`), across the
# `(subject, probe, structure)` combos.
#
# Motivation: the exploratory scatter this replaces uses an ordinary Pearson
# correlation that pools every `(subject, probe, structure)` point as independent
# (~165 points from 15 subjects). That is pseudoreplication.
# Here the slope of `x` is estimated in a linear mixed model with `subject` (and
# optionally `structure`) as random intercepts, so the p-value respects the
# nesting.
#
# All statistical machinery is reused from the layer-agnostic code (the LRT via
# [get_anova_pval()], Cohen's f^2 via [subtract_ranef_get_fsquared()], which
# already subtracts every grouping factor so crossed `(1|subject)+(1|structure)`
# works unchanged). Delete this file, `config/nod_rebound_correlation.yaml`,
# `scripts/*correlation*.R`, `tests/testthat/test-correlation_analysis.R`, the
# `nod_rebound_correlation_*` extdata parquets, and `_output_correlations/` to
# remove the feature entirely; nothing in the existing files is modified.

# -------------------- Defaults (the NOD -> NREM.Rebound case) --------------------

#' Default predictor / rebound conditions for the correlation analysis
#'
#' `PREDICTOR_CONDITION` is the whole-period NOD window used as the x-axis;
#' `REBOUND_POST` minus `REBOUND_BASELINE` is the `NREM.Rebound` response (the
#' same `NREM.Rebound` contrast defined in cnpix_local_sleep's `const.py`). All three are
#' exported by cnpix_local_sleep's `off-analysis export-nod-rebound-correlation`.
#' @export
PREDICTOR_CONDITION <- "NOD"
#' @rdname PREDICTOR_CONDITION
#' @export
REBOUND_POST <- "Early.REC.NREM"
#' @rdname PREDICTOR_CONDITION
#' @export
REBOUND_BASELINE <- "Early.REC.NREM.Match"

# -------------------- Data loading + reshaping --------------------

#' Load the NOD-rebound correlation input data
#'
#' Reads `nod_rebound_correlation_<off_type>_offs.parquet` from the package
#' `extdata`. Produced by cnpix_local_sleep's `off-analysis export-nod-rebound-correlation`,
#' it carries the same summarized schema as [load_offs_summary()] but for the
#' three correlation conditions (`NOD`, `Early.REC.NREM`, `Early.REC.NREM.Match`),
#' one row per `(subject, probe, structure, condition)`. A dedicated loader
#' rather than an option on [load_offs_summary()], so the pipeline stays fully
#' separable.
#'
#' @param off_type One of [OFF_TYPES].
#' @return Data frame with the summarized metrics for the correlation conditions.
#' @export
load_nod_rebound_correlation_data <- function(off_type = "llas") {
  assertthat::assert_that(off_type %in% OFF_TYPES)
  fname <- paste0("nod_rebound_correlation_", off_type, "_offs.parquet")
  path <- system.file("extdata", fname, package = "offp")
  assertthat::assert_that(
    nzchar(path),
    msg = paste0(
      "Correlation input file not found in extdata: ", fname,
      " (off_type='", off_type, "'). ",
      "Run cnpix_local_sleep's `off-analysis export-nod-rebound-correlation` first."
    )
  )
  arrow::read_parquet(path)
}

#' Build the per-(subject, probe, structure) correlation frame
#'
#' Reshapes the long correlation input into one row per
#' `(subject, probe, structure)` with `x` (the predictor metric in the NOD
#' window) and `y` (a metric's `NREM.Rebound`, i.e.
#' `response_metric@REBOUND_POST - response_metric@REBOUND_BASELINE`). Restricts
#' to `clade == "Cx"` (when present) and drops combos missing any of the required
#' conditions. Mirrors f25a's `get_correlation_data` plus the rebound difference.
#'
#' By default `response_metric == metric` and `response_data == NULL`, so `x` and
#' `y` are the *same* metric taken from the same frame `d` (the original
#' self-rebound case). To correlate an OFF predictor with the `NREM.Rebound` of a
#' *different* quantity (e.g. `mean_zlog_delta` cortical delta power), pass a
#' distinct `response_metric` and a `response_data` frame carrying that column at
#' the rebound conditions (the band-power condition means from
#' [load_bandpower_condition_means()]); it is joined to `x` on
#' `(subject, probe, structure)`.
#'
#' @param d Data frame from [load_nod_rebound_correlation_data()]; supplies `x`.
#' @param metric Predictor (x) metric column name (e.g. `"total_area_norm"`,
#'   `"rate"`).
#' @param predictor_condition Predictor (x) condition (default
#'   [PREDICTOR_CONDITION]).
#' @param rebound_post,rebound_baseline The two conditions whose difference is the
#'   response `y` (defaults [REBOUND_POST], [REBOUND_BASELINE]).
#' @param response_metric Response (y) metric column name (default `metric`).
#' @param response_data Optional frame supplying `response_metric` at the rebound
#'   conditions. `NULL` (default) means take the response from `d` too.
#' @return Data frame with columns `subject` (factor), `probe`, `structure`
#'   (factor), `x`, `y`.
#' @export
build_correlation_frame <- function(d, metric,
                                    predictor_condition = PREDICTOR_CONDITION,
                                    rebound_post = REBOUND_POST,
                                    rebound_baseline = REBOUND_BASELINE,
                                    response_metric = metric,
                                    response_data = NULL) {
  assertthat::assert_that(metric %in% names(d),
    msg = paste0("Metric '", metric, "' not found in correlation input data"))
  response_source <- if (is.null(response_data)) d else response_data
  assertthat::assert_that(response_metric %in% names(response_source),
    msg = paste0("Response metric '", response_metric,
                 "' not found in the response input data"))
  id_cols <- c("subject", "probe", "structure")

  # Restrict each source to cortical rows independently (they may be different
  # frames), then reshape.
  cx_only <- function(df) {
    if ("clade" %in% names(df)) df[as.character(df$clade) == "Cx", , drop = FALSE] else df
  }
  d <- cx_only(d)
  response_source <- cx_only(response_source)

  # One narrow frame per (source, metric, condition), renamed to a temp column.
  take <- function(src, col, cond, newname) {
    s <- src[as.character(src$condition) == cond, c(id_cols, col), drop = FALSE]
    assertthat::assert_that(nrow(s) > 0,
      msg = paste0("No rows for condition '", cond, "' in correlation input"))
    names(s)[names(s) == col] <- newname
    s
  }
  x_df <- take(d, metric, predictor_condition, ".x")
  post_df <- take(response_source, response_metric, rebound_post, ".post")
  base_df <- take(response_source, response_metric, rebound_baseline, ".base")

  m <- merge(x_df, post_df, by = id_cols)
  m <- merge(m, base_df, by = id_cols)
  m$x <- as.double(m$.x)
  m$y <- as.double(m$.post) - as.double(m$.base)
  m <- m[stats::complete.cases(m[, c("x", "y")]), , drop = FALSE]

  m$subject <- factor(as.character(m$subject))
  m$structure <- factor(as.character(m$structure))
  m[, c(id_cols, "x", "y"), drop = FALSE]
}

# -------------------- Model fitting + slope test --------------------

#' Fit full and null mixed models for the correlation
#'
#' Full: `y ~ x + <re_terms>`. Null: `y ~ 1 + <re_terms>`. A likelihood ratio
#' test of full vs null is the test of the slope of `x`. Generalizes f25a's
#' `fit_basic_models` to an arbitrary set of crossed random-intercept terms (e.g.
#' `c("(1|subject)")` or `c("(1|subject)","(1|structure)")`).
#'
#' @param d Data frame with `x`, `y`, and the grouping columns named in
#'   `re_terms`.
#' @param re_terms Character vector of random-effect terms.
#' @return Named list with `full` and `null` `lmerMod` objects.
#' @export
fit_correlation_models <- function(d, re_terms) {
  assertthat::assert_that(length(re_terms) >= 1)
  f_full <- stats::reformulate(c("x", re_terms), response = "y")
  f_null <- stats::reformulate(c("1", re_terms), response = "y")
  m_full <- lme4::lmer(f_full, data = d, REML = FALSE)
  m_null <- lme4::lmer(f_null, data = d, REML = FALSE)
  list(full = m_full, null = m_null)
}

#' Test the slope of the correlation predictor
#'
#' Likelihood ratio test of the `x` slope (full vs null), the fixed-effect slope
#' estimate with a Wald confidence interval, and the omnibus Cohen's f^2 via
#' [subtract_ranef_get_fsquared()] (reused unchanged; it subtracts every grouping
#' factor, so single and crossed random intercepts are both handled).
#'
#' @param models List from [fit_correlation_models()].
#' @param d The data frame the models were fit on.
#' @param conf_level Confidence level for the Wald slope interval.
#' @return List with `anova`, `pval`, `significant`, `slope`, `ci_lower`,
#'   `ci_upper`, `fsquared`, `n`, `n_subjects`.
#' @export
test_correlation_slope <- function(models, d, conf_level = 0.95) {
  aov <- stats::anova(models$full, models$null)
  pval <- get_anova_pval(aov)
  slope <- unname(lme4::fixef(models$full)["x"])

  alpha <- 1 - conf_level
  ci <- tryCatch(
    stats::confint(models$full, parm = "x", method = "Wald", level = conf_level),
    error = function(e) {
      # Fall back to a normal-approximation interval from the coefficient SE.
      se <- sqrt(as.matrix(stats::vcov(models$full))["x", "x"])
      z <- stats::qnorm(1 - alpha / 2)
      matrix(c(slope - z * se, slope + z * se), nrow = 1)
    }
  )

  fsq <- subtract_ranef_get_fsquared(d, models$full, models$null)$fsquared

  list(
    anova = aov,
    pval = pval,
    significant = isTRUE(pval < 0.05),
    slope = slope,
    ci_lower = unname(ci[1, 1]),
    ci_upper = unname(ci[1, 2]),
    fsquared = fsq,
    n = nrow(d),
    n_subjects = nlevels(factor(d$subject))
  )
}

# -------------------- Plot --------------------

#' Plot the NOD-activity vs NREM.Rebound correlation
#'
#' Scatter of `x` (predictor metric in the NOD window) vs `y` (the metric's
#' `NREM.Rebound`), coloured by `subject` so the within/between-subject structure
#' the mixed model accounts for is visible. Overlays the population-level
#' (fixed-effect) expectation line and its confidence ribbon, both estimated with
#' the marginal-effects method [modelbased::estimate_expectation()] (random
#' effects held at their population value), matching f25a's correlation plots.
#' The line is drawn solid if the slope p < 0.05, dashed if p < 0.1, else dotted.
#'
#' @param result Result list from [run_nod_rebound_correlation()].
#' @param fig_path If non-NULL, save the plot there (PNG/SVG by extension).
#' @return The ggplot object.
#' @export
plot_nod_rebound_correlation <- function(result, fig_path = NULL) {
  d <- result$data
  m <- result$models$full

  # Population-level (fixed-effect) expectation + CI over a 100-point grid in x,
  # via the easystats marginal-effects method (same as f25a). The grid does not
  # contain subject/structure, so random effects are held at their population
  # value; the ribbon is the CI of the expected mean (residual excluded).
  pred <- as.data.frame(
    modelbased::estimate_expectation(m, by = "x", length = 100)
  )

  p <- result$test$pval
  lt <- if (isTRUE(p < 0.05)) "solid" else if (isTRUE(p < 0.1)) "dashed" else "dotted"

  metric_lab <- get_display_label(result$metric)
  response_metric <- if (is.null(result$response_metric)) {
    result$metric
  } else {
    result$response_metric
  }
  response_lab <- get_display_label(response_metric)
  gg <- ggplot2::ggplot(d, ggplot2::aes(x = .data$x, y = .data$y)) +
    ggplot2::geom_ribbon(
      data = pred,
      ggplot2::aes(x = .data$x, ymin = .data$CI_low, ymax = .data$CI_high),
      inherit.aes = FALSE, alpha = 0.15
    ) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey60") +
    ggplot2::geom_point(
      ggplot2::aes(colour = .data$subject), size = 2, alpha = 0.85
    ) +
    ggplot2::geom_line(
      data = pred, ggplot2::aes(x = .data$x, y = .data$Predicted),
      inherit.aes = FALSE, linewidth = 1, linetype = lt
    ) +
    ggplot2::labs(
      x = paste0(metric_lab, ", ", result$predictor_condition),
      y = paste0(response_lab, ", NREM.Rebound"),
      colour = "subject",
      title = sprintf(
        "%s -> %s / %s [%s]: slope=%.3g, p=%.3g, f2=%.3g, n=%d (%d subj)",
        result$metric, response_metric, result$off_type, result$model_def$name,
        result$test$slope, result$test$pval, result$test$fsquared,
        result$test$n, result$test$n_subjects
      )
    ) +
    ggplot2::theme_classic() +
    ggplot2::theme(legend.position = "right")

  if (!is.null(fig_path)) {
    if (identical(tolower(tools::file_ext(fig_path)), "svg")) {
      grDevices::svg(fig_path, width = 8, height = 5, bg = "white")
      print(gg)
      grDevices::dev.off()
    } else {
      ggplot2::ggsave(fig_path, gg, width = 8, height = 5, bg = "white",
                      create.dir = TRUE)
    }
  }
  gg
}

# -------------------- Orchestrator + serializers --------------------

#' Run one NOD-activity vs NREM.Rebound correlation analysis
#'
#' Loads (or accepts) the correlation input, builds the per-combo frame, fits the
#' full/null mixed models for `model_def$re_terms`, tests the slope, plots, and
#' writes `results.rds`, `results.json`, `summary.txt`, `variance_components.csv`,
#' and a figure into `output_dir/<model_def$name>/`.
#'
#' @param metric Predictor (x) metric column name (`"total_area_norm"`,
#'   `"rate"`, ...).
#' @param off_type One of [OFF_TYPES].
#' @param model_def Model definition list with `name` and `re_terms`.
#' @param output_dir Base directory; a `<model_def$name>/` subdir is created.
#' @param predictor_condition,rebound_post,rebound_baseline Condition names (see
#'   [build_correlation_frame()]).
#' @param response_metric Response (y) metric column name (default `metric`, the
#'   self-rebound case). When it differs, `y` is that metric's `NREM.Rebound`.
#' @param data Optional pre-loaded correlation input for `off_type`.
#' @param response_data Optional pre-loaded response frame carrying
#'   `response_metric` at the rebound conditions. When `response_metric` differs
#'   from `metric` and this is `NULL`, the band-power condition means are loaded
#'   via [load_bandpower_condition_means()].
#' @param save_ggplot_rds If `TRUE`, also save the ggplot object as `.rds`.
#' @return Invisibly, the full result list (also saved as `results.rds`).
#' @export
run_nod_rebound_correlation <- function(
  metric,
  off_type,
  model_def,
  output_dir,
  predictor_condition = PREDICTOR_CONDITION,
  rebound_post = REBOUND_POST,
  rebound_baseline = REBOUND_BASELINE,
  response_metric = metric,
  data = NULL,
  response_data = NULL,
  save_ggplot_rds = TRUE
) {
  assertthat::assert_that(assertthat::is.string(metric))
  assertthat::assert_that(assertthat::is.string(response_metric))
  assertthat::assert_that(assertthat::is.string(output_dir))
  assertthat::assert_that(assertthat::is.string(model_def$name))
  assertthat::assert_that(!is.null(model_def$re_terms))
  assertthat::assert_that(off_type %in% OFF_TYPES)

  if (is.null(data)) {
    data <- load_nod_rebound_correlation_data(off_type)
  }
  cross_metric <- !identical(response_metric, metric)
  if (cross_metric && is.null(response_data)) {
    response_data <- load_bandpower_condition_means()
  }
  d <- build_correlation_frame(data, metric, predictor_condition,
                               rebound_post, rebound_baseline,
                               response_metric = response_metric,
                               response_data = if (cross_metric) response_data else NULL)
  assertthat::assert_that(
    nrow(d) >= 3,
    msg = paste0("Too few complete (subject, probe, structure) combos (",
                 nrow(d), ") to fit a correlation for '", metric, "'")
  )

  models <- fit_correlation_models(d, model_def$re_terms)
  test <- test_correlation_slope(models, d)

  result <- list(
    metric = metric,
    response_metric = response_metric,
    response = "NREM.Rebound",
    predictor_condition = predictor_condition,
    rebound_conditions = c(rebound_post, rebound_baseline),
    off_type = off_type,
    model_def = model_def,
    data = d,
    models = models,
    test = test
  )

  spec_dir <- file.path(output_dir, model_def$name)
  fig_dir <- file.path(spec_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  fig_stem <- if (cross_metric) {
    paste0(metric, "__vs__", response_metric, "_NREM.Rebound")
  } else {
    paste0(metric, "_NOD_vs_NREM.Rebound")
  }
  gg <- plot_nod_rebound_correlation(
    result, fig_path = file.path(fig_dir, paste0(fig_stem, ".svg"))
  )
  if (save_ggplot_rds) {
    saveRDS(gg, file.path(fig_dir, paste0(fig_stem, ".rds")))
  }

  saveRDS(result, file.path(spec_dir, "results.rds"))
  writeLines(
    jsonlite::toJSON(build_correlation_json_summary(result),
                     auto_unbox = TRUE, pretty = TRUE, digits = NA),
    file.path(spec_dir, "results.json")
  )
  writeLines(build_correlation_text_summary(result),
             file.path(spec_dir, "summary.txt"))

  write_variance_components(models, spec_dir)

  invisible(result)
}

#' JSON summary for a correlation result
#' @keywords internal
build_correlation_json_summary <- function(result) {
  t <- result$test
  list(
    metric = result$metric,
    response_metric = if (is.null(result$response_metric)) result$metric else result$response_metric,
    response = result$response,
    predictor_condition = result$predictor_condition,
    rebound_conditions = I(result$rebound_conditions),
    off_type = result$off_type,
    model = result$model_def$name,
    re_terms = I(result$model_def$re_terms),
    slope = t$slope,
    ci_lower = t$ci_lower,
    ci_upper = t$ci_upper,
    pval = t$pval,
    significant = t$significant,
    cohens_f2 = t$fsquared,
    n = t$n,
    n_subjects = t$n_subjects
  )
}

#' Text summary for a correlation result
#' @keywords internal
build_correlation_text_summary <- function(result) {
  t <- result$test
  response_metric <- if (is.null(result$response_metric)) {
    result$metric
  } else {
    result$response_metric
  }
  c(
    paste("NOD-activity vs NREM.Rebound correlation:", result$metric,
          "->", response_metric),
    paste("Model name:", result$model_def$name),
    paste("RE terms:", paste(result$model_def$re_terms, collapse = ", ")),
    paste("OFF type:", result$off_type),
    paste0("Predictor (x): ", result$metric, " @ ", result$predictor_condition),
    paste0("Response  (y): ", response_metric, " NREM.Rebound (",
           result$rebound_conditions[1], " - ", result$rebound_conditions[2], ")"),
    "",
    "--- slope of x (LRT full vs null) ---",
    paste("slope:", format(t$slope, digits = 4)),
    paste0("95% CI: [", format(t$ci_lower, digits = 4), ", ",
           format(t$ci_upper, digits = 4), "]"),
    paste("p-value:", format(t$pval, digits = 4)),
    paste("Significant:", t$significant),
    paste("Cohen's f^2:", format(round(t$fsquared, 4), nsmall = 4)),
    paste0("n = ", t$n, " (subject x probe x structure) combos, ",
           t$n_subjects, " subjects")
  )
}
