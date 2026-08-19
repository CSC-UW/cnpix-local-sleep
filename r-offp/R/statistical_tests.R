# Statistical testing functions
#
# General functions for statistical hypothesis testing.

#' Get ANOVA p-value
#'
#' @param anova_result Result from anova() function
#' @return P-value from likelihood ratio test
#' @export
get_anova_pval <- function(anova_result) {
  anova_result["Pr(>Chisq)"][[1]][2]
}

#' Test for main effect of condition
#'
#' Compares the full model (with `condition` as a fixed effect) to the null
#' model (intercept-only fixed effects) via likelihood ratio test. If the
#' main effect is significant and a contrast matrix is provided, performs
#' post-hoc tests via [multcomp::glht()].
#'
#' @param dat Data frame.
#' @param models Named list with elements `full` and `null`.
#' @param contrast_matrix Optional contrast matrix for post-hoc tests (a
#'   `multcomp::mcp` object).
#' @param weights Optional numeric vector of prior weights, passed through
#'   to [subtract_ranef_get_fsquared()].
#' @return List with main effect test results: `anova`, `pval`, and
#'   optionally `effect_size` and `posthoc`.
#' @export
test_main_effect <- function(dat, models, contrast_matrix = NULL,
                             weights = NULL) {
  main_effect <- list()
  main_effect$anova <- stats::anova(models$full, models$null)
  main_effect$pval <- get_anova_pval(main_effect$anova)

  if (main_effect$pval < 0.05) {
    main_effect$effect_size <- subtract_ranef_get_fsquared(
      dat,
      models$full,
      models$null,
      weights = weights
    )

    if (!is.null(contrast_matrix)) {
      ph <- list()
      ph$glht <- multcomp::glht(models$full, contrast_matrix)
      ph$ci <- stats::confint(ph$glht)
      ph$effect_size <- cohens_d_analogue(ph$glht, models$full)
      main_effect$posthoc <- ph
    }
  }
  main_effect
}
