#' Subtract random effects from data
#'
#' Subtracts estimated random intercepts from the response variable so that
#' standard linear models can be fit on the adjusted data for effect size
#' estimation (Selya et al., 2012).
#'
#' Iterates over every grouping factor in the model's random effects
#' structure and subtracts each factor's estimated random intercept from the
#' response variable. Handles single, crossed, nested, and interaction
#' random effects uniformly.
#'
#' @param d Data frame used to fit model `m`.
#' @param m A fitted `lmerMod` with random intercepts to subtract.
#' @return Data frame with random effects subtracted from the response.
#' @export
subtract_random_effects <- function(d, m) {
  fixed <- d
  random_effects <- lme4::ranef(m)
  response_var <- all.vars(stats::formula(m))[1]

  for (re_name in names(random_effects)) {
    re_table <- random_effects[[re_name]]

    if (grepl(":", re_name)) {
      # Interaction term (e.g., "subject:structure")
      var_names <- strsplit(re_name, ":")[[1]]
      for (level_id in rownames(re_table)) {
        intercept <- re_table[level_id, 1]
        id_parts <- strsplit(level_id, ":")[[1]]
        mask <- rep(TRUE, nrow(fixed))
        for (j in seq_along(var_names)) {
          mask <- mask & (as.character(fixed[[var_names[j]]]) == id_parts[j])
        }
        fixed[mask, response_var] <- fixed[mask, response_var] - intercept
      }
    } else {
      # Simple term (e.g., "subject" or "structure")
      for (level_id in rownames(re_table)) {
        intercept <- re_table[level_id, 1]
        mask <- (as.character(fixed[[re_name]]) == level_id)
        fixed[mask, response_var] <- fixed[mask, response_var] - intercept
      }
    }
  }

  fixed
}

#' Strip random effects from formula
#'
#' Reconstruct a formula with only fixed effects.
#' Example A: y ~ x * z + (1|subject) -> y ~ x * z
#' Example B: y ~ x + (1|subject) -> y ~ x
#' Example C: y ~ (1|subject) -> y ~ 1
#' Example D: y ~ (1|subject) + (1|group) -> y ~ 1
#' For most formulas, this will be equivalent to nobars(formula_obj).
#'
#' @param formula_obj Formula object with random effects
#' @return Formula object with only fixed effects
#' @export
strip_random_effects <- function(formula_obj) {
  terms <- as.character(formula_obj)
  lhs <- terms[2]
  rhs <- terms[3]

  # Random-effect terms are the ones bracketing a "|".
  fixed_terms <- unlist(strsplit(rhs, " \\+ "))
  fixed_terms <- fixed_terms[!grepl(" \\| ", fixed_terms)]

  # A model that was all random effects still needs an intercept-only RHS.
  fixed_rhs <- if (length(fixed_terms) == 0) {
    "1"
  } else {
    paste(fixed_terms, collapse = " + ")
  }

  stats::as.formula(paste(lhs, "~", fixed_rhs))
}

#' Calculate Cohen's local f-squared
#'
#' For estimation statistics, Cohen's f2 (f-squared) is a measure of effect
#' size. The local version of the f2 statistic quantifies the amount of variance
#' accounted for by variables of interest, relative to the amount of variance
#' left unaccounted for by the model.
#'
#' @param model_a The model with the effect of interest included
#' @param model_b The model with the effect of interest removed
#' @return Cohen's f-squared value
#' @export
cohens_local_fsquared <- function(model_a, model_b) {
  r2_a <- summary(model_a)$r.squared
  r2_b <- summary(model_b)$r.squared
  f2 <- (r2_a - r2_b) / (1 - r2_a)
  f2
}

#' Subtract random effects and calculate f-squared
#'
#' There is a caveat to computing f2 for LME models; when we remove the effect
#' of interest, it is possible that the reduced model is estimated with a
#' drastically different random effect structure, misrepresenting the variance
#' accounted for by the fixed effects (Selya et al., 2012). To circumvent this
#' issue, we first fit the full model to estimate a random effect structure
#' and subtract the estimated random effects from the data. We then fit two
#' standard linear models to the adjusted data (one full, one reduced) and use
#' their R2 values to calculate f2. In this way, we enforce that the full and
#' reduced models have identical random effect structure, and that f2 explicitly
#' captures the variance accounted for by the fixed effects.
#'
#' @param d Data frame
#' @param model_a The model with the effect of interest included, and the source
#' of random effects to be subtracted from the data
#' @param model_b The model with the effect of interest removed
#' @param weights Optional numeric vector of prior weights. If provided,
#'   passed to [stats::lm()] for weighted least-squares estimation of the
#'   adjusted models. Should be the same weights used in the original
#'   `lmerMod` fit.
#' @return List with adjusted data, models, and f-squared value
#' @export
subtract_ranef_get_fsquared <- function(d, model_a, model_b,
                                        weights = NULL) {
  d_fixef <- subtract_random_effects(d, model_a)

  # Remove random effects structure from the formulas, and refit using the
  # data that also has random effects structure removed.
  # Setting the model's call attribute ensures that it prints nicely.
  fa_fixef <- strip_random_effects(stats::formula(model_a))
  fb_fixef <- strip_random_effects(stats::formula(model_b))

  if (!is.null(weights)) {
    # Add weights to data frame so lm()'s model.frame can find them via NSE
    d_fixef[[".lm_weights"]] <- weights
    ma_fixef <- stats::lm(fa_fixef, data = d_fixef, weights = .lm_weights)
    ma_fixef$call <- call("lm", formula = fa_fixef, data = quote(d_fixef))

    mb_fixef <- stats::lm(fb_fixef, data = d_fixef, weights = .lm_weights)
    mb_fixef$call <- call("lm", formula = fb_fixef, data = quote(d_fixef))

    d_fixef[[".lm_weights"]] <- NULL
  } else {
    ma_fixef <- stats::lm(fa_fixef, data = d_fixef)
    ma_fixef$call <- call("lm", formula = fa_fixef, data = quote(d_fixef))

    mb_fixef <- stats::lm(fb_fixef, data = d_fixef)
    mb_fixef$call <- call("lm", formula = fb_fixef, data = quote(d_fixef))
  }

  fsquared <- cohens_local_fsquared(ma_fixef, mb_fixef)

  list(
    data = d_fixef,
    models = list(a = ma_fixef, b = mb_fixef),
    fsquared = fsquared
  )
}

#' Calculate Cohen's d analogue for mixed effects models
#'
#' The ratio of mean to variance is used as a measure of effect size, with the
#' combined residual variance and random effect variance in the denominator
#' (analogous to Cohen's D but for LME models).
#'
#' @param glht General linear hypothesis test object
#' @param model Mixed effects model
#' @return Matrix of Cohen's d analogue values
#' @export
cohens_d_analogue <- function(glht, model) {
  # We are being very conservative here.
  # lme4::sigma(model) would also be fine as a measure of variance.
  # There is no standard.
  denom <- sqrt(sum(as.data.frame(lme4::VarCorr(model))$vcov))
  eff_size <- as.matrix(stats::coef(glht)) / denom
  colnames(eff_size) <- "Cohen's d analogue"
  eff_size
}
