# High-level analysis workflow
#
# Functions that orchestrate the full cx_homeostasis analysis pipeline:
# load data, filter, fit models, run tests, generate plots, and save results.

#' Run a cx_homeostasis analysis for one response variable
#'
#' Executes the full pipeline: load -> filter -> fit -> test -> plot -> save.
#' Results are saved to `<output_dir>/<model_def$name>/` as RDS, JSON, PNG,
#' and text files.
#'
#' @param response_var Column name of the response variable (string).
#' @param output_dir Base directory for results. A subdirectory named
#'   after the model definition's `name` is created within it.
#' @param model_def Model definition list with elements `name`,
#'   `fe_terms`, `re_terms`, and `weighted`. See
#'   [validate_model_def()] for details.
#' @param off_type Type of OFFs to analyze: one of [OFF_TYPES].
#' @param condition_set Optional name of the condition set this analysis
#'   belongs to (e.g. `"six"`, `"nrem"`, `"wake"`). Recorded in the result /
#'   JSON / text outputs for provenance and consumed by the plot/summary
#'   tooling; does not itself change which rows are fit (use `conditions` /
#'   `posthocs` for that).
#' @param conditions Character vector of conditions to include, or `NULL`
#'   for all conditions present in the data.
#' @param posthocs Character vector of contrast strings for post-hoc tests,
#'   e.g. `c("Early.REC.NREM - Early.BSL.NREM = 0", ...)`. If `NULL`, no
#'   post-hoc tests are performed.
#' @param data Optional pre-loaded data frame. If `NULL`, calls
#'   [load_offs_summary()] and applies the standard Cx/None/spatial filter.
#' @param save_ggplot_rds If `TRUE`, save ggplot objects as `.rds` alongside
#'   PNGs for programmatic inspection.
#' @return Invisibly, the full result list (also saved as `results.rds`).
#' @export
run_cx_homeostasis_analysis <- function(
  response_var,
  output_dir,
  model_def,
  off_type = "llas",
  condition_set = NULL,
  conditions = NULL,
  posthocs = NULL,
  data = NULL,
  save_ggplot_rds = TRUE
) {
  # -------------------- Validate inputs --------------------
  assertthat::assert_that(assertthat::is.string(response_var))
  assertthat::assert_that(assertthat::is.string(output_dir))
  validate_model_def(model_def)
  # `off_type` is an OFF-summary selector only when this function loads the
  # data itself. When `data` is supplied (e.g. the separable `bandpower`
  # companion, whose measure is not an OFF property), it is just a provenance
  # label recorded in the outputs, so don't constrain it to the OFF enum.
  if (is.null(data)) {
    assertthat::assert_that(off_type %in% OFF_TYPES)
  }
  assertthat::assert_that(
    is.null(condition_set) || assertthat::is.string(condition_set)
  )

  weighted <- model_def$weighted

  # -------------------- Load and filter data --------------------
  if (is.null(data)) {
    d <- filter_layer_agnostic_cx(load_offs_summary(off_type))
  } else {
    d <- data
  }

  # Filter to specified conditions
  if (!is.null(conditions)) {
    d <- d[d$condition %in% conditions, ]
    d$condition <- droplevels(factor(d$condition, levels = conditions))
  }

  assertthat::assert_that(
    response_var %in% names(d),
    msg = paste0("Response variable '", response_var, "' not found in data")
  )

  # Validate one structure entry per subject-condition
  assertthat::assert_that(
    !any(d |>
      dplyr::group_by(.data[["subject"]], .data[["condition"]]) |>
      dplyr::count(.data[["structure"]]) |>
      dplyr::pull("n") > 1),
    msg = "Each subject-condition combination should have only one structure entry"
  )

  # Cast response to double (needed for random effects subtraction)
  d[[response_var]] <- as.double(d[[response_var]])

  # Drop cells with a missing response *before* fitting, so that `d` and the
  # fitted model share one row set. lme4 drops them silently, which leaves the
  # diagnostic plots, the random-effect-subtracted data and the effect sizes
  # (all built from `d`) misaligned with `fitted()`/`residuals()`. Responses
  # exported by a companion pipeline need not cover every cell the base summary
  # has (e.g. the size-adjusted edge statistics skip cells with too few events
  # to standardize), so this is a normal condition, not an error.
  n_before <- nrow(d)
  d <- d[!is.na(d[[response_var]]), , drop = FALSE]
  if (nrow(d) < n_before) {
    message(sprintf(
      "Dropped %d of %d rows with a missing '%s'",
      n_before - nrow(d), n_before, response_var
    ))
  }
  assertthat::assert_that(
    nrow(d) > 0,
    msg = paste0("No non-missing values of '", response_var, "'")
  )
  if (!is.null(conditions)) {
    missing_conditions <- setdiff(conditions, unique(as.character(d$condition)))
    assertthat::assert_that(
      length(missing_conditions) == 0,
      msg = paste0(
        "Condition(s) with no non-missing '", response_var, "': ",
        paste(missing_conditions, collapse = ", ")
      )
    )
    d$condition <- droplevels(d$condition)
  }

  # Apply optional response transform (default identity)
  # The transform is applied to the response column in place, before weighting
  # and fitting, so that weights, random-effect subtraction, effect sizes, and
  # all diagnostic plots operate on a single consistent (transformed) scale.
  transform <- if (is.null(model_def$transform)) {
    "identity"
  } else {
    model_def$transform
  }
  d[[response_var]] <- apply_response_transform(d[[response_var]], transform)

  # -------------------- Compute weights (if requested) --------------------
  w <- if (weighted) {
    compute_condition_weights(d, response_var)
  } else {
    NULL
  }

  # -------------------- Fit models --------------------
  models <- fit_models(
    d, response_var, model_def$fe_terms, model_def$re_terms, weights = w
  )

  # -------------------- Generate diagnostic plots --------------------
  spec_dir <- file.path(output_dir, model_def$name)
  fig_dir <- file.path(spec_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  condition_palette <- get_condition_palette()

  p_vio <- plot_distributions_by_condition(
    d, condition, response_var, condition_palette, geom = "violin",
    ylabel = transformed_display_label(response_var, transform)
  )
  ggplot2::ggsave(
    file.path(fig_dir, paste0(response_var, "_violins.png")),
    plot = p_vio, width = 10, height = 6
  )
  if (save_ggplot_rds) {
    saveRDS(p_vio, file.path(fig_dir, paste0(response_var, "_violins.rds")))
  }

  p_rvf <- plot_rvf(models$full, d$condition, condition_palette,
                    weighted = weighted)
  p_qq <- plot_qqline(models$full, d$condition, condition_palette,
                      weighted = weighted)

  d_adjusted <- subtract_random_effects(d, models$full)
  utils::write.csv(d_adjusted, file.path(spec_dir, "adjusted_data.csv"),
                   row.names = FALSE)
  adj_ylabel <- paste0(
    transformed_display_label(response_var, transform), " (RE removed)"
  )
  p_vio_adj <- plot_distributions_by_condition(
    d_adjusted, condition, response_var, condition_palette,
    geom = "violin", ylabel = adj_ylabel
  )

  ggplot2::ggsave(
    file.path(fig_dir, paste0(response_var, "_residuals.png")),
    plot = p_rvf, width = 10, height = 6
  )
  ggplot2::ggsave(
    file.path(fig_dir, paste0(response_var, "_qq.png")),
    plot = p_qq, width = 10, height = 6
  )
  ggplot2::ggsave(
    file.path(fig_dir, paste0(response_var, "_violins_adjusted.png")),
    plot = p_vio_adj, width = 10, height = 6
  )

  if (save_ggplot_rds) {
    saveRDS(p_rvf, file.path(fig_dir, paste0(response_var, "_residuals.rds")))
    saveRDS(p_qq, file.path(fig_dir, paste0(response_var, "_qq.rds")))
    saveRDS(
      p_vio_adj,
      file.path(fig_dir, paste0(response_var, "_violins_adjusted.rds"))
    )
  }

  diag_lines <- generate_figure_diagnostics(models$full, d, response_var,
                                            weighted = weighted)
  writeLines(diag_lines, file.path(fig_dir, "diagnostics.txt"))

  # -------------------- Statistical tests --------------------
  contrast_matrix <- build_condition_comparisons(posthocs)
  main_effect <- test_main_effect(d, models, contrast_matrix, weights = w)
  sig_main_effect <- main_effect$pval < 0.05

  # -------------------- Assemble result --------------------
  result <- list(
    response_var = response_var,
    model_def = model_def,
    off_type = off_type,
    condition_set = condition_set,
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

  # -------------------- Save outputs --------------------
  dir.create(spec_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(result, file.path(spec_dir, "results.rds"))

  json_summary <- build_json_summary(result)
  writeLines(
    jsonlite::toJSON(json_summary, auto_unbox = TRUE, pretty = TRUE, digits = NA),
    file.path(spec_dir, "results.json")
  )

  txt_summary <- build_text_summary(result)
  writeLines(txt_summary, file.path(spec_dir, "summary.txt"))

  write_variance_components(models, spec_dir)

  invisible(result)
}

#' Build a machine-readable JSON summary of analysis results
#'
#' @param result Result list from [run_cx_homeostasis_analysis()].
#' @return A list suitable for [jsonlite::toJSON()].
#' @keywords internal
build_json_summary <- function(result) {
  summary <- list(
    response_var = result$response_var,
    model_def = result$model_def,
    off_type = result$off_type,
    condition_set = result$condition_set,
    transform = result$transform,
    # Wrap in I() so jsonlite keeps it a JSON array even when length 1.
    conditions = I(result$conditions),
    main_effect = list(
      pval = result$main_effect$pval,
      significant = result$sig_main_effect
    )
  )

  if (result$sig_main_effect && !is.null(result$main_effect$effect_size)) {
    summary$main_effect$cohens_f2 <- result$main_effect$effect_size$fsquared

    if (!is.null(result$main_effect$posthoc)) {
      ph <- result$main_effect$posthoc
      # Wrap each in I() so jsonlite emits JSON arrays even for a single
      # contrast (e.g. the one-contrast `wake` condition set); otherwise
      # auto_unbox collapses length-1 vectors to scalars and breaks the
      # downstream plot/summary consumers.
      summary$main_effect$posthoc <- list(
        contrasts = I(names(stats::coef(ph$glht))),
        pvalues = I(as.numeric(summary(ph$glht)$test$pvalues)),
        estimates = I(as.numeric(stats::coef(ph$glht))),
        cohens_d = I(as.numeric(ph$effect_size[, 1])),
        ci_lower = I(as.numeric(ph$ci$confint[, "lwr"])),
        ci_upper = I(as.numeric(ph$ci$confint[, "upr"]))
      )
    }
  }

  summary
}

#' Build a human-readable text summary of analysis results
#'
#' @param result Result list from [run_cx_homeostasis_analysis()].
#' @return Character vector of text lines.
#' @keywords internal
build_text_summary <- function(result) {
  model_def <- result$model_def
  lines <- character()
  lines <- c(lines, paste("Analysis:", result$response_var))
  lines <- c(lines, paste("Model name:", model_def$name))
  lines <- c(lines, paste("OFF type:", result$off_type))
  if (!is.null(result$condition_set)) {
    lines <- c(lines, paste("Condition set:", result$condition_set))
  }
  lines <- c(lines, paste(
    "Conditions:", paste(result$conditions, collapse = ", ")
  ))
  lines <- c(lines, paste("Weighted:", model_def$weighted))
  lines <- c(lines, paste(
    "Transform:",
    if (is.null(result$transform)) "identity" else result$transform
  ))
  lines <- c(lines, paste("FE terms:", paste(model_def$fe_terms,
                                             collapse = ", ")))
  lines <- c(lines, paste("RE terms:", paste(model_def$re_terms,
                                             collapse = ", ")))
  lines <- c(lines, "")

  lines <- c(lines, "--- Main Effect Test ---")
  lines <- c(lines, paste("p-value:", format(result$main_effect$pval,
                                             digits = 4)))
  lines <- c(lines, paste("Significant:", result$sig_main_effect))

  if (result$sig_main_effect && !is.null(result$main_effect$effect_size)) {
    lines <- c(lines, paste(
      "Cohen's f^2:",
      format(round(result$main_effect$effect_size$fsquared, 4), nsmall = 4)
    ))

    if (!is.null(result$main_effect$posthoc)) {
      lines <- c(lines, "")
      lines <- c(lines, "--- Post-hoc Contrasts ---")
      lines <- c(lines, format_posthoc_summary(result$main_effect$posthoc))
    }
  }

  lines
}

#' Generate text-based figure diagnostics
#'
#' Produces a text summary of model diagnostics: Shapiro-Wilk test on
#' residuals and group-level summary statistics.
#'
#' @param model A fitted lmer model.
#' @param data The data frame used for fitting.
#' @param response_var The response variable name.
#' @param weighted If `TRUE`, compute diagnostics on weighted residuals.
#'   Default: `FALSE`.
#' @return Character vector of diagnostic lines.
#' @keywords internal
generate_figure_diagnostics <- function(model, data, response_var,
                                        weighted = FALSE) {
  resids <- stats::residuals(model)
  resid_label <- "residuals"
  if (weighted) {
    resids <- resids * sqrt(stats::weights(model))
    resid_label <- "weighted residuals"
  }
  # Shapiro-Wilk only supports n <= 5000
  n <- min(length(resids), 5000)
  sw <- stats::shapiro.test(resids[seq_len(n)])

  lines <- c(
    "=== Model Diagnostic Summary ===",
    "",
    paste0("Shapiro-Wilk normality test on ", resid_label, ":"),
    paste("  W =", format(sw$statistic, digits = 4)),
    paste("  p =", format(sw$p.value, digits = 4)),
    ""
  )

  lines <- c(lines, "=== Response by Condition ===")
  conditions <- sort(unique(data$condition))
  for (cond in conditions) {
    vals <- data[[response_var]][data$condition == cond]
    lines <- c(lines, paste0(
      "  ", cond, ": ",
      "median=", format(stats::median(vals), digits = 4), ", ",
      "IQR=", format(stats::IQR(vals), digits = 4), ", ",
      "n=", length(vals)
    ))
  }

  lines
}
