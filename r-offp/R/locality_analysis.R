# Locality (Local vs Overlapping) analysis pipeline.
#
# A self-contained companion to the layer-agnostic `cx_homeostasis` pipeline
# (`R/analysis.R`), splitting OFFs by `overlap_status in {Local, Overlapping}`
# rather than by condition alone. It answers, for cross-structure OFFs:
#   - request 2 (per condition): within each single statistical condition, do
#     Local and Overlapping OFFs differ on each measure?
#   - request 2 (interaction): does that Local-vs-Overlapping difference depend on
#     condition (condition × overlap_status / difference-in-differences)?
#   - request 3: within all NREM / all Wake (whole-recording OFFs), do Local and
#     Overlapping OFFs differ? And does that gap differ between NREM and Wake?
#
# (Request 1, condition contrasts on `mean_overlap_degree`, is the ordinary
# condition main-effect path and is run directly through
# [run_cx_homeostasis_analysis()] by the runner, not here.)
#
# All statistical machinery is reused from the layer-agnostic code by calling the
# generic helpers ([fit_models()], [test_main_effect()],
# [subtract_ranef_get_fsquared()], [build_json_summary()], etc.); nothing in the
# existing files is modified. Delete this file, [locality_data_loading.R][offp],
# the locality config/scripts, and `_output_locality/` to fully remove the
# feature.

# -------------------- Helpers --------------------

#' Build an overlap_status contrast matrix from posthoc strings
#'
#' The locality analogue of [build_condition_comparisons()]: wraps the strings in
#' `multcomp::mcp(overlap_status = ...)`. Defaults to the single
#' Overlapping-vs-Local contrast.
#'
#' @param posthoc_strings Character vector of contrast strings (e.g.
#'   `"Overlapping - Local = 0"`). If `NULL`/empty, returns `NULL`.
#' @return A [multcomp::mcp] object, or `NULL`.
#' @export
build_locality_comparisons <- function(
  posthoc_strings = c("Overlapping - Local = 0")
) {
  if (is.null(posthoc_strings) || length(posthoc_strings) == 0) {
    return(NULL)
  }
  multcomp::mcp(overlap_status = posthoc_strings)
}

#' Inverse-variance weights over one or more grouping columns
#'
#' Private to this pipeline, so locality stays deletable in one piece.
#' @keywords internal
.locality_group_weights <- function(d, response_var, group_vars) {
  assertthat::assert_that(response_var %in% names(d))
  for (g in group_vars) {
    assertthat::assert_that(g %in% names(d))
  }
  group_var_vals <- do.call(
    stats::ave,
    c(
      list(d[[response_var]]),
      unname(lapply(group_vars, function(g) d[[g]])),
      list(FUN = stats::var)
    )
  )
  assertthat::assert_that(
    all(group_var_vals > 0),
    msg = "All weighting groups must have non-zero variance for weighting"
  )
  1.0 / group_var_vals
}

#' Two-color palette for overlap statuses
#' @keywords internal
locality_palette <- function() {
  c(Local = "#7570b3", Overlapping = "#e7298a")
}

# Main-effect: Local-vs-Overlapping within a condition / state

#' Run a locality Local-vs-Overlapping analysis for one response variable
#'
#' Mirrors [run_cx_homeostasis_analysis()] but tests `overlap_status` (Local vs
#' Overlapping) as the fixed effect. Use `conditions` to select the comparison
#' window: a single statistical condition (request 2) or a single whole-recording
#' state, `"NREM"` or `"Wake"` (request 3). When more than one condition is pooled,
#' supply a model whose `re_terms` include a `(1 | subject:structure:condition)`
#' cell so that `overlap_status` is a pure within-cell (paired) contrast.
#'
#' @param response_var Response variable column name.
#' @param output_dir Base directory; a `<model_def$name>/` subdir is created.
#' @param model_def Model definition (`name`, `fe_terms = "overlap_status"`,
#'   `re_terms`, `weighted`, optional `transform`).
#' @param off_type One of [OFF_TYPES] (provenance label; default `"llas"`).
#' @param analysis_kind Provenance label recorded in the `condition_set` slot
#'   (e.g. `"cond-Early.REC.NREM"`, `"NREM"`, `"Wake"`).
#' @param conditions Character vector of condition/state levels to include.
#' @param locality_posthocs Contrast strings for the Local-vs-Overlapping posthoc.
#' @param data Pre-loaded + prepared locality data frame.
#' @param save_ggplot_rds If `TRUE`, also save ggplot objects as `.rds`.
#' @return Invisibly, the full result list (also saved as `results.rds`).
#' @export
run_locality_main_effect_analysis <- function(
  response_var,
  output_dir,
  model_def,
  off_type = "llas",
  analysis_kind = NULL,
  conditions = NULL,
  locality_posthocs = c("Overlapping - Local = 0"),
  data = NULL,
  save_ggplot_rds = TRUE
) {
  assertthat::assert_that(assertthat::is.string(response_var))
  assertthat::assert_that(assertthat::is.string(output_dir))
  validate_model_def(model_def)
  assertthat::assert_that(identical(model_def$fe_terms, "overlap_status"),
    msg = "Locality main-effect models must have fe_terms = 'overlap_status'")
  assertthat::assert_that(off_type %in% OFF_TYPES)

  weighted <- model_def$weighted

  if (is.null(data)) {
    d <- prepare_locality_data(load_locality_per_condition_summary(off_type))
  } else {
    d <- data
  }

  if (!is.null(conditions)) {
    d <- d[d$condition %in% conditions, , drop = FALSE]
    # Force an UNORDERED factor: the parquet may store condition as an ordered
    # categorical, and factor() would preserve the ordering, yielding polynomial
    # contrasts.
    d$condition <- factor(as.character(d$condition), levels = conditions)
  }

  assertthat::assert_that(
    response_var %in% names(d),
    msg = paste0("Response variable '", response_var, "' not found in data")
  )
  assertthat::assert_that(
    nlevels(d$overlap_status) == 2 &&
      all(c("Local", "Overlapping") %in% d$overlap_status),
    msg = "Locality data must contain both Local and Overlapping rows after filtering"
  )

  d[[response_var]] <- as.double(d[[response_var]])

  transform <- if (is.null(model_def$transform)) "identity" else model_def$transform
  d[[response_var]] <- apply_response_transform(d[[response_var]], transform)

  # Weight by the pooled cells when conditions are pooled, else by overlap status.
  weight_groups <- if (!is.null(conditions) && length(conditions) > 1) {
    c("condition", "overlap_status")
  } else {
    "overlap_status"
  }
  w <- if (weighted) {
    .locality_group_weights(d, response_var, weight_groups)
  } else {
    NULL
  }

  models <- fit_models(
    d, response_var, model_def$fe_terms, model_def$re_terms, weights = w
  )

  # Diagnostic plots (grouped by overlap status)
  spec_dir <- file.path(output_dir, model_def$name)
  fig_dir <- file.path(spec_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  pal <- locality_palette()
  p_rvf <- plot_rvf(models$full, d$overlap_status, pal, weighted = weighted)
  p_qq <- plot_qqline(models$full, d$overlap_status, pal, weighted = weighted)
  p_vio <- plot_distributions_by_condition(
    d, overlap_status, response_var, pal, geom = "violin",
    xlabel = "Overlap status",
    ylabel = transformed_display_label(response_var, transform)
  )

  d_adjusted <- subtract_random_effects(d, models$full)
  utils::write.csv(d_adjusted, file.path(spec_dir, "adjusted_data.csv"),
                   row.names = FALSE)
  p_vio_adj <- plot_distributions_by_condition(
    d_adjusted, overlap_status, response_var, pal, geom = "violin",
    xlabel = "Overlap status",
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

  diag_lines <- generate_locality_diagnostics(models$full, d, response_var,
                                              weighted = weighted)
  writeLines(diag_lines, file.path(fig_dir, "diagnostics.txt"))

  # Statistical tests (reuses the generic main-effect path)
  contrast_matrix <- build_locality_comparisons(locality_posthocs)
  main_effect <- test_main_effect(d, models, contrast_matrix, weights = w)
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

#' Text diagnostics for a locality (Local vs Overlapping) fit
#' @keywords internal
generate_locality_diagnostics <- function(model, data, response_var,
                                          weighted = FALSE) {
  resids <- stats::residuals(model)
  resid_label <- "residuals"
  if (weighted) {
    resids <- resids * sqrt(stats::weights(model))
    resid_label <- "weighted residuals"
  }
  n <- min(length(resids), 5000)
  sw <- stats::shapiro.test(resids[seq_len(n)])
  lines <- c(
    "=== Model Diagnostic Summary ===",
    "",
    paste0("Shapiro-Wilk normality test on ", resid_label, ":"),
    paste("  W =", format(sw$statistic, digits = 4)),
    paste("  p =", format(sw$p.value, digits = 4)),
    "",
    "=== Response by overlap status ==="
  )
  for (os in levels(data$overlap_status)) {
    vals <- data[[response_var]][data$overlap_status == os]
    lines <- c(lines, paste0(
      "  ", os, ": median=", format(stats::median(vals), digits = 4),
      ", IQR=", format(stats::IQR(vals), digits = 4),
      ", n=", length(vals)
    ))
  }
  lines
}

# Interaction: condition (or state) x overlap_status (difference-in-differences)

#' Fit full and null models for a condition x overlap_status interaction
#'
#' Full: `response ~ condition * overlap_status + re_terms`.
#' Null: `response ~ condition + overlap_status + re_terms`.
#' A likelihood ratio test of full vs null is a test of the interaction.
#'
#' @param d Data frame with `condition`, `overlap_status`, and the response.
#' @param response_var Response variable name (string).
#' @param re_terms Character vector of random-effect terms.
#' @param weights Optional prior weights.
#' @return Named list with `full` and `null` `lmerMod` objects.
#' @export
fit_locality_interaction_models <- function(d, response_var, re_terms,
                                            weights = NULL) {
  f_full <- stats::reformulate(
    c("condition * overlap_status", re_terms), response = response_var
  )
  f_null <- stats::reformulate(
    c("condition", "overlap_status", re_terms), response = response_var
  )
  if (!is.null(weights)) {
    m_full <- lme4::lmer(f_full, data = d, REML = FALSE, weights = weights)
    m_null <- lme4::lmer(f_null, data = d, REML = FALSE, weights = weights)
  } else {
    m_full <- lme4::lmer(f_full, data = d, REML = FALSE)
    m_null <- lme4::lmer(f_null, data = d, REML = FALSE)
  }
  list(full = m_full, null = m_null)
}

#' Locate the interaction coefficient for a condition level
#' @keywords internal
.locality_interaction_coef <- function(fe_names, condition_level) {
  cand <- c(
    paste0("condition", condition_level, ":overlap_statusOverlapping"),
    paste0("overlap_statusOverlapping:condition", condition_level)
  )
  hit <- cand[cand %in% fe_names]
  if (length(hit) == 0) NA_character_ else hit[1]
}

#' Build difference-in-differences interaction contrasts
#'
#' For each condition contrast `"A - B"`, builds the linear combination of fixed
#' effects equal to `(A - B | Overlapping) - (A - B | Local)`, i.e. whether the
#' A-vs-B condition difference depends on overlap status. With `Local` as the
#' reference, this is `coef(condition<A>:overlap_statusOverlapping) -
#' coef(condition<B>:overlap_statusOverlapping)`.
#'
#' @param posthoc_strings Condition (or state) contrast strings.
#' @param full_model The fitted interaction model.
#' @return A numeric contrast matrix (rows = contrasts), or `NULL`.
#' @export
build_locality_did_contrasts <- function(posthoc_strings, full_model) {
  if (is.null(posthoc_strings) || length(posthoc_strings) == 0) {
    return(NULL)
  }
  fe <- names(lme4::fixef(full_model))
  rows <- list()
  rn <- character()
  for (s in posthoc_strings) {
    lhs <- trimws(strsplit(s, "=", fixed = TRUE)[[1]][1])
    parts <- trimws(strsplit(lhs, "-", fixed = TRUE)[[1]])
    assertthat::assert_that(
      length(parts) == 2,
      msg = paste0("Locality DiD supports only two-condition contrasts: ", s)
    )
    k <- stats::setNames(rep(0, length(fe)), fe)
    coef_a <- .locality_interaction_coef(fe, parts[1])
    coef_b <- .locality_interaction_coef(fe, parts[2])
    if (!is.na(coef_a)) k[coef_a] <- k[coef_a] + 1
    if (!is.na(coef_b)) k[coef_b] <- k[coef_b] - 1
    if (all(k == 0)) {
      next
    }
    rows[[length(rows) + 1]] <- k
    rn <- c(rn, gsub("\\s+", " ", lhs))
  }
  if (length(rows) == 0) {
    return(NULL)
  }
  K <- do.call(rbind, rows)
  rownames(K) <- rn
  K
}

#' Build per-condition simple-effect contrasts for overlap_status
#'
#' For each condition in the set, builds the linear combination of fixed effects
#' equal to the simple effect of `overlap_status` *within that condition*
#' (`Overlapping - Local | condition`). This is the pooled-model analogue of the
#' per-condition [run_locality_main_effect_analysis] fits: instead of fitting one
#' model per condition, the simple effects are read off the single
#' `condition * overlap_status` interaction model as `multcomp` contrasts, so
#' they share one pooled residual/RE variance and a single family-wise
#' multiplicity adjustment. With `Local` as the overlap_status reference and
#' treatment contrasts, the simple effect in the reference condition is
#' `coef(overlap_statusOverlapping)`, and in any other condition C it is
#' `coef(overlap_statusOverlapping) + coef(condition<C>:overlap_statusOverlapping)`.
#'
#' @param conditions Condition (or state) levels in the set; the first is the
#'   factor reference level.
#' @param full_model The fitted interaction model.
#' @return A numeric contrast matrix (one row per condition), or `NULL`.
#' @export
build_locality_simple_effect_contrasts <- function(conditions, full_model) {
  if (is.null(conditions) || length(conditions) == 0) {
    return(NULL)
  }
  fe <- names(lme4::fixef(full_model))
  base_coef <- "overlap_statusOverlapping"
  assertthat::assert_that(
    base_coef %in% fe,
    msg = "Interaction model is missing the overlap_statusOverlapping coefficient"
  )
  ref <- conditions[1]
  rows <- list()
  rn <- character()
  for (cond in conditions) {
    k <- stats::setNames(rep(0, length(fe)), fe)
    k[base_coef] <- 1
    if (!identical(cond, ref)) {
      ic <- .locality_interaction_coef(fe, cond)
      assertthat::assert_that(
        !is.na(ic),
        msg = paste0("Missing interaction coefficient for condition: ", cond)
      )
      k[ic] <- 1
    }
    rows[[length(rows) + 1]] <- k
    rn <- c(rn, paste0("Overlapping - Local | ", cond))
  }
  K <- do.call(rbind, rows)
  rownames(K) <- rn
  K
}

#' Run a locality condition x overlap_status interaction analysis
#'
#' Fits the interaction model, LR-tests the interaction, computes the omnibus
#' Cohen's f^2, and computes a difference-in-differences contrast for each
#' condition posthoc. It additionally reads off, from the SAME pooled model, the
#' per-condition simple effects of `overlap_status` (the pooled-variance
#' alternative to the separate per-condition [run_locality_main_effect_analysis]
#' fits) and the marginal main effect of `overlap_status` (averaged across
#' conditions), so the two approaches can be compared head-to-head.
#'
#' @param response_var Response variable column name.
#' @param output_dir Base directory; a `<model_def$name>/` subdir is created.
#' @param model_def Model definition (`name`, `re_terms`, `weighted`, optional
#'   `transform`; `fe_terms` is descriptive only here).
#' @param off_type One of [OFF_TYPES].
#' @param condition_set Name of the condition/state set (e.g. `"nrem"`, `"state"`).
#' @param conditions Condition/state levels in the set.
#' @param posthocs Condition (or state) contrast strings for the DiD.
#' @param data Pre-loaded + prepared locality data frame.
#' @param save_ggplot_rds If `TRUE`, also save ggplot objects as `.rds`.
#' @return Invisibly, the full result list (also saved as `results.rds`).
#' @export
run_locality_interaction_analysis <- function(
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
  assertthat::assert_that(assertthat::is.string(response_var))
  assertthat::assert_that(assertthat::is.string(output_dir))
  assertthat::assert_that(assertthat::is.string(model_def$name))
  assertthat::assert_that(off_type %in% OFF_TYPES)

  weighted <- isTRUE(model_def$weighted)
  re_terms <- model_def$re_terms
  transform <- if (is.null(model_def$transform)) "identity" else model_def$transform

  if (is.null(data)) {
    d <- prepare_locality_data(load_locality_per_condition_summary(off_type))
  } else {
    d <- data
  }

  assertthat::assert_that(!is.null(conditions))
  d <- d[d$condition %in% conditions, , drop = FALSE]
  # Unordered factor -> treatment contrasts -> `condition<level>:overlap_statusOverlapping`
  # coefficient names that build_locality_did_contrasts() matches against.
  d$condition <- factor(as.character(d$condition), levels = conditions)
  assertthat::assert_that(
    response_var %in% names(d),
    msg = paste0("Response variable '", response_var, "' not found in data")
  )
  d[[response_var]] <- as.double(d[[response_var]])
  d[[response_var]] <- apply_response_transform(d[[response_var]], transform)

  w <- if (weighted) {
    .locality_group_weights(d, response_var, c("condition", "overlap_status"))
  } else {
    NULL
  }

  models <- fit_locality_interaction_models(d, response_var, re_terms, weights = w)

  aov <- stats::anova(models$full, models$null)
  pval <- get_anova_pval(aov)
  sig <- pval < 0.05
  fsq <- subtract_ranef_get_fsquared(d, models$full, models$null,
                                     weights = w)$fsquared

  did <- NULL
  K <- build_locality_did_contrasts(posthocs, models$full)
  if (!is.null(K)) {
    glht <- multcomp::glht(models$full, linfct = K)
    ci <- stats::confint(glht)
    eff <- cohens_d_analogue(glht, models$full)
    did <- list(
      contrasts = rownames(K),
      pvalues = as.numeric(summary(glht)$test$pvalues),
      estimates = as.numeric(stats::coef(glht)),
      cohens_d = as.numeric(eff[, 1]),
      ci_lower = as.numeric(ci$confint[, "lwr"]),
      ci_upper = as.numeric(ci$confint[, "upr"])
    )
  }

  # Per-condition simple effects of overlap_status read off the SAME pooled
  # interaction model: the pooled-variance counterpart to the separate
  # per-condition main-effect fits. Computed unconditionally (so a non-sig
  # interaction does not mask them); glht applies one family-wise adjustment
  # across the set.
  simple <- NULL
  Ks <- build_locality_simple_effect_contrasts(conditions, models$full)
  if (!is.null(Ks)) {
    glht_s <- multcomp::glht(models$full, linfct = Ks)
    ci_s <- stats::confint(glht_s)
    eff_s <- cohens_d_analogue(glht_s, models$full)
    simple <- list(
      contrasts = rownames(Ks),
      pvalues = as.numeric(summary(glht_s)$test$pvalues),
      estimates = as.numeric(stats::coef(glht_s)),
      cohens_d = as.numeric(eff_s[, 1]),
      ci_lower = as.numeric(ci_s$confint[, "lwr"]),
      ci_upper = as.numeric(ci_s$confint[, "upr"])
    )
  }

  # Marginal main effect of overlap_status (averaged across conditions, the
  # interaction excluded): LRT of (condition + overlap_status) vs (condition).
  # This is the "test for a main effect of overlap_status" step of the pooled
  # alternative; `simple` above is its per-condition decomposition.
  f_reduced <- stats::reformulate(c("condition", re_terms), response = response_var)
  m_reduced <- if (!is.null(w)) {
    lme4::lmer(f_reduced, data = d, REML = FALSE, weights = w)
  } else {
    lme4::lmer(f_reduced, data = d, REML = FALSE)
  }
  aov_om <- stats::anova(models$null, m_reduced)
  pval_om <- get_anova_pval(aov_om)
  overlap_main <- list(
    anova = aov_om,
    pval = pval_om,
    significant = pval_om < 0.05,
    fsquared = subtract_ranef_get_fsquared(d, models$null, m_reduced,
                                           weights = w)$fsquared
  )

  interaction <- list(
    anova = aov, pval = pval, significant = sig, fsquared = fsq, did = did,
    simple = simple, overlap_main = overlap_main
  )

  result <- list(
    response_var = response_var,
    model_def = model_def,
    off_type = off_type,
    condition_set = condition_set,
    analysis_kind = paste0("interaction-", condition_set),
    transform = transform,
    conditions = conditions,
    data = d,
    models = models,
    interaction = interaction
  )

  spec_dir <- file.path(output_dir, model_def$name)
  fig_dir <- file.path(spec_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  # Diagnostics colored by the condition x overlap_status interaction cell.
  cell <- interaction(d$condition, d$overlap_status, drop = TRUE,
                      sep = " : ", lex.order = TRUE)
  cell_pal <- stats::setNames(
    grDevices::hcl.colors(nlevels(cell), "Dark 3"), levels(cell)
  )
  p_rvf <- plot_rvf(models$full, cell, cell_pal, weighted = weighted)
  p_qq <- plot_qqline(models$full, cell, cell_pal, weighted = weighted)
  ggplot2::ggsave(file.path(fig_dir, paste0(response_var, "_residuals.png")),
                  plot = p_rvf, width = 10, height = 6)
  ggplot2::ggsave(file.path(fig_dir, paste0(response_var, "_qq.png")),
                  plot = p_qq, width = 10, height = 6)
  if (save_ggplot_rds) {
    saveRDS(p_rvf, file.path(fig_dir, paste0(response_var, "_residuals.rds")))
    saveRDS(p_qq, file.path(fig_dir, paste0(response_var, "_qq.rds")))
  }

  # Residual-normality diagnostics (Shapiro-Wilk on the full model's residuals,
  # weighted residuals when weighted) so transform/weighting variants can be
  # compared on the same metric the main-effect path reports.
  writeLines(
    generate_locality_diagnostics(models$full, d, response_var,
                                  weighted = weighted),
    file.path(fig_dir, "diagnostics.txt")
  )

  dir.create(spec_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(result, file.path(spec_dir, "results.rds"))
  writeLines(
    jsonlite::toJSON(build_locality_interaction_json_summary(result),
                     auto_unbox = TRUE, pretty = TRUE, digits = NA),
    file.path(spec_dir, "results.json")
  )
  writeLines(build_locality_interaction_text_summary(result),
             file.path(spec_dir, "summary.txt"))

  write_variance_components(models, spec_dir)

  invisible(result)
}

#' JSON summary for a locality interaction result
#' @keywords internal
build_locality_interaction_json_summary <- function(result) {
  s <- list(
    response_var = result$response_var,
    model_def = result$model_def,
    off_type = result$off_type,
    condition_set = result$condition_set,
    analysis_kind = result$analysis_kind,
    transform = result$transform,
    conditions = I(result$conditions),
    interaction = list(
      pval = result$interaction$pval,
      significant = result$interaction$significant,
      cohens_f2 = result$interaction$fsquared
    )
  )
  did <- result$interaction$did
  if (!is.null(did)) {
    s$interaction$did <- lapply(did, function(x) I(x))
  }
  simple <- result$interaction$simple
  if (!is.null(simple)) {
    s$interaction$simple <- lapply(simple, function(x) I(x))
  }
  om <- result$interaction$overlap_main
  if (!is.null(om)) {
    s$interaction$overlap_main <- list(
      pval = om$pval, significant = om$significant, cohens_f2 = om$fsquared
    )
  }
  s
}

#' Text summary for a locality interaction result
#' @keywords internal
build_locality_interaction_text_summary <- function(result) {
  it <- result$interaction
  lines <- c(
    paste("Locality interaction analysis:", result$response_var),
    paste("Model name:", result$model_def$name),
    paste("OFF type:", result$off_type),
    paste("Condition set:", result$condition_set),
    paste("Conditions:", paste(result$conditions, collapse = ", ")),
    paste("Transform:", result$transform),
    paste("RE terms:", paste(result$model_def$re_terms, collapse = ", ")),
    "",
    "--- condition x overlap_status interaction (LRT) ---",
    paste("p-value:", format(it$pval, digits = 4)),
    paste("Significant:", it$significant),
    paste("Cohen's f^2:", format(round(it$fsquared, 4), nsmall = 4)),
    ""
  )
  om <- it$overlap_main
  if (!is.null(om)) {
    lines <- c(lines,
      "--- Main effect of overlap_status (averaged across conditions) ---",
      "(LRT of condition + overlap_status vs condition; interaction excluded)",
      paste("p-value:", format(om$pval, digits = 4)),
      paste("Significant:", om$significant),
      paste("Cohen's f^2:", format(round(om$fsquared, 4), nsmall = 4)),
      "")
  }
  simple <- it$simple
  if (!is.null(simple)) {
    lines <- c(lines,
      "--- Per-condition simple effects of overlap_status (pooled model) ---",
      "(estimate = Overlapping - Local within each condition; FWER-adjusted)")
    for (i in seq_along(simple$contrasts)) {
      lines <- c(lines, paste0(
        "  ", simple$contrasts[i],
        ": est=", format(simple$estimates[i], digits = 4),
        ", p=", format(simple$pvalues[i], digits = 4),
        ", d=", format(simple$cohens_d[i], digits = 3)
      ))
    }
    lines <- c(lines, "")
  }
  did <- it$did
  if (!is.null(did)) {
    lines <- c(lines,
      "--- Difference-in-differences per condition contrast ---",
      "(estimate = how the condition difference shifts Overlapping - Local)")
    for (i in seq_along(did$contrasts)) {
      lines <- c(lines, paste0(
        "  ", did$contrasts[i],
        ": est=", format(did$estimates[i], digits = 4),
        ", p=", format(did$pvalues[i], digits = 4),
        ", d=", format(did$cohens_d[i], digits = 3)
      ))
    }
  }
  lines
}
