# Cross-structure "excess globality" analysis pipeline.
#
# A self-contained companion to the layer-agnostic `cx_homeostasis` pipeline
# (`R/analysis.R`), structurally identical to the locality pipeline
# (`R/locality_analysis.R`) with `quantity in {null, observed}` in place of
# `overlap_status in {Local, Overlapping}`. It answers, for whole-recording NREM
# cross-structure OFFs: is the observed cross-structure overlap degree greater
# than the duration-matched windowed-shift null ("more global than chance")?
#
# The unit of analysis is the per-(subject, structure) mean overlap degree, with
# two rows per cell (`quantity` = `observed` / `null`). The LRT main effect of
# `quantity` (full `value ~ quantity + RE` vs null `value ~ 1 + RE`) is the
# observed-vs-chance test; the `"observed - null"` posthoc contrast gives the
# direction and Cohen's d. This is the r-offp port of
# `cnpix_local_sleep`'s `cross_structure_offs.test_excess_above_chance` (the subject-level
# paired test + intercept-only mixed model).
#
# All statistical machinery is reused from the layer-agnostic code by calling the
# generic helpers ([fit_models()], [test_main_effect()],
# [subtract_ranef_get_fsquared()], [build_json_summary()], etc.); nothing in the
# existing files is modified. Delete this file, [excess_globality_data_loading.R][offp],
# the excess-globality config/script/test, and `_output_excess_globality/` to
# fully remove the feature.

#' Build a quantity contrast matrix from posthoc strings
#'
#' The excess-globality analogue of [build_locality_comparisons()]: wraps the
#' strings in `multcomp::mcp(quantity = ...)`. Defaults to the single
#' observed-vs-null contrast.
#'
#' @param posthoc_strings Character vector of contrast strings (e.g.
#'   `"observed - null = 0"`). If `NULL`/empty, returns `NULL`.
#' @return A [multcomp::mcp] object, or `NULL`.
#' @export
build_excess_comparisons <- function(posthoc_strings = c("observed - null = 0")) {
  if (is.null(posthoc_strings) || length(posthoc_strings) == 0) {
    return(NULL)
  }
  multcomp::mcp(quantity = posthoc_strings)
}

#' Two-color palette for quantities
#' @keywords internal
excess_palette <- function() {
  c(null = "#7570b3", observed = "#e7298a")
}

#' Text diagnostics for an excess-globality (observed vs null) fit
#' @keywords internal
generate_excess_diagnostics <- function(model, data, response_var) {
  resids <- stats::residuals(model)
  n <- min(length(resids), 5000)
  sw <- stats::shapiro.test(resids[seq_len(n)])
  lines <- c(
    "=== Model Diagnostic Summary ===",
    "",
    "Shapiro-Wilk normality test on residuals:",
    paste("  W =", format(sw$statistic, digits = 4)),
    paste("  p =", format(sw$p.value, digits = 4)),
    "",
    "=== Response by quantity ==="
  )
  for (q in levels(data$quantity)) {
    vals <- data[[response_var]][data$quantity == q]
    lines <- c(lines, paste0(
      "  ", q, ": median=", format(stats::median(vals), digits = 4),
      ", IQR=", format(stats::IQR(vals), digits = 4),
      ", n=", length(vals)
    ))
  }
  lines
}

#' Run the excess-globality observed-vs-null analysis
#'
#' Mirrors [run_locality_main_effect_analysis()] but tests `quantity` (null vs
#' observed) as the fixed effect. The default `conditions = "NREM"` selects the
#' whole-recording NREM rows (the only window exported).
#'
#' @param response_var Response variable column name (default `"value"`).
#' @param output_dir Base directory; a `<model_def$name>/` subdir is created.
#' @param model_def Model definition (`name`, `fe_terms = "quantity"`, `re_terms`,
#'   `weighted`, optional `transform`).
#' @param off_type Provenance label (default `"llas"`).
#' @param analysis_kind Provenance label recorded in the `condition_set` slot
#'   (default `"state-NREM"`).
#' @param conditions Character vector of condition/state levels to include.
#' @param excess_posthocs Contrast strings for the observed-vs-null posthoc.
#' @param data Pre-loaded + prepared excess-globality data frame.
#' @param save_ggplot_rds If `TRUE`, also save ggplot objects as `.rds`.
#' @return Invisibly, the full result list (also saved as `results.rds`).
#' @export
run_excess_globality_analysis <- function(
  response_var = "value",
  output_dir,
  model_def,
  off_type = "llas",
  analysis_kind = "state-NREM",
  conditions = "NREM",
  excess_posthocs = c("observed - null = 0"),
  data = NULL,
  save_ggplot_rds = TRUE
) {
  assertthat::assert_that(assertthat::is.string(response_var))
  assertthat::assert_that(assertthat::is.string(output_dir))
  validate_model_def(model_def)
  assertthat::assert_that(identical(model_def$fe_terms, "quantity"),
    msg = "Excess-globality models must have fe_terms = 'quantity'")

  if (is.null(data)) {
    d <- prepare_excess_globality_data(load_excess_globality_summary())
  } else {
    d <- data
  }

  if (!is.null(conditions) && "condition" %in% names(d)) {
    d <- d[d$condition %in% conditions, , drop = FALSE]
  }

  assertthat::assert_that(
    response_var %in% names(d),
    msg = paste0("Response variable '", response_var, "' not found in data")
  )
  assertthat::assert_that(
    nlevels(d$quantity) == 2 && all(c("null", "observed") %in% d$quantity),
    msg = "Excess data must contain both null and observed rows after filtering"
  )

  d[[response_var]] <- as.double(d[[response_var]])
  transform <- if (is.null(model_def$transform)) "identity" else model_def$transform
  d[[response_var]] <- apply_response_transform(d[[response_var]], transform)

  models <- fit_models(
    d, response_var, model_def$fe_terms, model_def$re_terms, weights = NULL
  )

  # -------------------- Diagnostic plots (grouped by quantity) --------------------
  spec_dir <- file.path(output_dir, model_def$name)
  fig_dir <- file.path(spec_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  pal <- excess_palette()
  p_rvf <- plot_rvf(models$full, d$quantity, pal, weighted = FALSE)
  p_qq <- plot_qqline(models$full, d$quantity, pal, weighted = FALSE)
  p_vio <- plot_distributions_by_condition(
    d, quantity, response_var, pal, geom = "violin",
    xlabel = "Quantity",
    ylabel = transformed_display_label(response_var, transform)
  )

  d_adjusted <- subtract_random_effects(d, models$full)
  utils::write.csv(d_adjusted, file.path(spec_dir, "adjusted_data.csv"),
                   row.names = FALSE)
  p_vio_adj <- plot_distributions_by_condition(
    d_adjusted, quantity, response_var, pal, geom = "violin",
    xlabel = "Quantity",
    ylabel = paste0(
      transformed_display_label(response_var, transform), " (RE removed)"
    )
  )

  ggplot2::ggsave(file.path(fig_dir, paste0(response_var, "_residuals.png")),
                  plot = p_rvf, width = 10, height = 6)
  ggplot2::ggsave(file.path(fig_dir, paste0(response_var, "_qq.png")),
                  plot = p_qq, width = 10, height = 6)
  ggplot2::ggsave(file.path(fig_dir, paste0(response_var, "_violins.png")),
                  plot = p_vio, width = 10, height = 6)
  ggplot2::ggsave(file.path(fig_dir, paste0(response_var, "_violins_adjusted.png")),
                  plot = p_vio_adj, width = 10, height = 6)
  if (save_ggplot_rds) {
    saveRDS(p_rvf, file.path(fig_dir, paste0(response_var, "_residuals.rds")))
    saveRDS(p_qq, file.path(fig_dir, paste0(response_var, "_qq.rds")))
    saveRDS(p_vio, file.path(fig_dir, paste0(response_var, "_violins.rds")))
    saveRDS(p_vio_adj,
            file.path(fig_dir, paste0(response_var, "_violins_adjusted.rds")))
  }

  diag_lines <- generate_excess_diagnostics(models$full, d, response_var)
  writeLines(diag_lines, file.path(fig_dir, "diagnostics.txt"))

  # Statistical tests (reuses the generic main-effect path)
  contrast_matrix <- build_excess_comparisons(excess_posthocs)
  main_effect <- test_main_effect(d, models, contrast_matrix, weights = NULL)
  sig_main_effect <- main_effect$pval < 0.05

  result <- list(
    response_var = response_var,
    model_def = model_def,
    off_type = off_type,
    condition_set = analysis_kind,
    transform = transform,
    conditions = if (!is.null(conditions)) {
      conditions
    } else {
      sort(unique(as.character(d$condition)))
    },
    data = d,
    models = models,
    main_effect = main_effect,
    sig_main_effect = sig_main_effect
  )

  dir.create(spec_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(result, file.path(spec_dir, "results.rds"))

  # The result shape matches the layer-agnostic one, so reuse its serializers.
  json_summary <- build_json_summary(result)
  writeLines(
    jsonlite::toJSON(json_summary, auto_unbox = TRUE, pretty = TRUE, digits = NA),
    file.path(spec_dir, "results.json")
  )
  writeLines(build_text_summary(result), file.path(spec_dir, "summary.txt"))

  write_variance_components(models, spec_dir)

  invisible(result)
}
