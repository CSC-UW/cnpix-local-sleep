# Core functions for fitting models

#' Supported response-variable transforms
#'
#' Named list mapping a transform name (the optional `transform` field of a
#' model definition) to the function applied to the response column *before*
#' weighting and fitting. `"identity"` is the default no-op. The transform is
#' applied to the response column in place (not as a `log(y)` term in the model
#' formula) so that weights, random-effect subtraction, effect sizes, and
#' diagnostic plots all operate on a single consistent scale.
#'
#' @export
RESPONSE_TRANSFORMS <- list(
  identity = function(x) x,
  log = function(x) log(x),
  log10 = function(x) log10(x),
  log1p = function(x) log1p(x),
  sqrt = function(x) sqrt(x)
)

#' Apply a response-variable transform with domain validation
#'
#' Validates that `transform` is one of [RESPONSE_TRANSFORMS] and that the
#' values lie in the transform's domain, then returns the transformed vector.
#'
#' @param x Numeric vector of response values.
#' @param transform Transform name; one of `names(RESPONSE_TRANSFORMS)`.
#'   Default `"identity"`.
#' @return The transformed numeric vector.
#' @export
apply_response_transform <- function(x, transform = "identity") {
  assertthat::assert_that(
    assertthat::is.string(transform),
    transform %in% names(RESPONSE_TRANSFORMS),
    msg = paste0(
      "Unknown transform '", transform, "'. Supported: ",
      paste(names(RESPONSE_TRANSFORMS), collapse = ", ")
    )
  )
  if (transform %in% c("log", "log10")) {
    assertthat::assert_that(
      all(x > 0, na.rm = TRUE),
      msg = paste0("transform '", transform,
                   "' requires strictly positive values; found values <= 0")
    )
  } else if (transform == "log1p") {
    assertthat::assert_that(
      all(x > -1, na.rm = TRUE),
      msg = "transform 'log1p' requires values > -1; found values <= -1"
    )
  } else if (transform == "sqrt") {
    assertthat::assert_that(
      all(x >= 0, na.rm = TRUE),
      msg = "transform 'sqrt' requires non-negative values; found values < 0"
    )
  }
  RESPONSE_TRANSFORMS[[transform]](x)
}

#' Validate a model definition list
#'
#' Checks that a model definition list contains the required fields
#' with correct types. Stops with an informative error if validation
#' fails.
#'
#' A model definition must have:
#' - `name`: a single string identifying the model (used for output
#'   directory naming)
#' - `fe_terms`: a length-1 character vector of fixed effect terms.
#'   Only a single fixed effect term is supported because the null
#'   model is derived by removing all `fe_terms`; with multiple FE
#'   terms, the LRT would test all of them simultaneously.
#' - `re_terms`: a character vector of random effect terms (e.g.,
#'   `c("(1 | subject)", "(1 | structure)")`)
#' - `weighted`: a logical flag for inverse-variance weighting
#' - `transform` (optional): name of a response transform applied to the
#'   response column before weighting/fitting; one of
#'   `names(RESPONSE_TRANSFORMS)`. Defaults to `"identity"` when absent.
#'
#' @param model_def A list with elements `name`, `fe_terms`,
#'   `re_terms`, `weighted`, and optionally `transform`.
#' @return `model_def` invisibly (called for side effects).
#' @export
validate_model_def <- function(model_def) {
  assertthat::assert_that(is.list(model_def))

  required <- c("name", "fe_terms", "re_terms", "weighted")
  missing <- setdiff(required, names(model_def))
  if (length(missing) > 0) {
    stop(
      "Model definition missing required fields: ",
      paste(missing, collapse = ", ")
    )
  }

  assertthat::assert_that(
    assertthat::is.string(model_def$name),
    msg = "model_def$name must be a single string"
  )
  assertthat::assert_that(
    is.character(model_def$fe_terms) && length(model_def$fe_terms) == 1,
    msg = "model_def$fe_terms must be a length-1 character vector"
  )
  assertthat::assert_that(
    is.character(model_def$re_terms) && length(model_def$re_terms) >= 1,
    msg = "model_def$re_terms must be a character vector with >= 1 element"
  )
  assertthat::assert_that(
    assertthat::is.flag(model_def$weighted),
    msg = "model_def$weighted must be TRUE or FALSE"
  )
  if (!is.null(model_def$transform)) {
    assertthat::assert_that(
      assertthat::is.string(model_def$transform),
      model_def$transform %in% names(RESPONSE_TRANSFORMS),
      msg = paste0(
        "model_def$transform must be one of: ",
        paste(names(RESPONSE_TRANSFORMS), collapse = ", ")
      )
    )
  }

  invisible(model_def)
}

#' Compute per-condition inverse-variance weights
#'
#' Computes a weight for each observation based on the inverse of the
#' sample variance of the response variable within its condition group.
#' These weights are suitable for use as prior weights in [lme4::lmer()],
#' where `Var(y_i) = sigma^2 / w_i`.
#'
#' @param d Data frame containing the response variable and a `condition`
#'   column.
#' @param response_var Response variable name (string).
#' @return Numeric vector of weights, one per row of `d`, in the same
#'   order as the rows of `d`.
#' @export
compute_condition_weights <- function(d, response_var) {
  assertthat::assert_that(response_var %in% names(d))
  assertthat::assert_that("condition" %in% names(d))

  condition_vars <- stats::ave(
    d[[response_var]],
    d[["condition"]],
    FUN = stats::var
  )
  assertthat::assert_that(
    all(condition_vars > 0),
    msg = "All conditions must have non-zero variance for weighting"
  )

  1.0 / condition_vars
}

#' Fit full and null models for given fixed and random effect terms
#'
#' Fits a full model (with the specified fixed effect) and a null model
#' (intercept-only fixed effects) using the provided random effects
#' structure. Both models are fit with ML estimation (`REML = FALSE`)
#' to enable likelihood ratio testing.
#'
#' @param d Data frame containing the response variable and all
#'   columns referenced in `fe_terms` and `re_terms`.
#' @param response_var Response variable name (string).
#' @param fe_terms Length-1 character vector specifying the fixed
#'   effect term (e.g., `"condition"`).
#' @param re_terms Character vector of random effect terms (e.g.,
#'   `c("(1 | subject)", "(1 | structure)")`).
#' @param weights Optional numeric vector of prior weights for
#'   weighted estimation. Passed directly to [lme4::lmer()]. If
#'   `NULL` (default), unweighted estimation is used.
#' @return Named list with elements `full` and `null`, each a fitted
#'   `lmerMod` object.
#' @export
fit_models <- function(d, response_var, fe_terms, re_terms,
                       weights = NULL) {
  assertthat::assert_that(
    is.character(fe_terms) && length(fe_terms) == 1,
    msg = "fe_terms must be a length-1 character vector"
  )
  assertthat::assert_that(
    is.character(re_terms) && length(re_terms) >= 1,
    msg = "re_terms must be a character vector with >= 1 element"
  )

  # Extract grouping variable names from RE terms for column validation.
  # E.g., "(1 | subject)" -> "subject", "(1 | subject:structure)" ->
  # c("subject", "structure")
  re_grouping <- gsub(".*\\|\\s*", "", re_terms)
  re_grouping <- gsub("\\).*", "", re_grouping)
  re_grouping <- trimws(re_grouping)
  re_vars <- unique(unlist(strsplit(re_grouping, ":")))

  required_cols <- unique(c(response_var, fe_terms, re_vars))
  missing_cols <- setdiff(required_cols, names(d))
  if (length(missing_cols) > 0) {
    stop(
      "Column(s) not found in data: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  f_full <- stats::reformulate(
    c(fe_terms, re_terms),
    response = response_var
  )
  f_null <- stats::reformulate(re_terms, response = response_var)

  if (!is.null(weights)) {
    m_full <- lme4::lmer(f_full, data = d, REML = FALSE, weights = weights)
    m_null <- lme4::lmer(f_null, data = d, REML = FALSE, weights = weights)
  } else {
    m_full <- lme4::lmer(f_full, data = d, REML = FALSE)
    m_null <- lme4::lmer(f_null, data = d, REML = FALSE)
  }

  list(full = m_full, null = m_null)
}


#' Write the full/null variance components of a fitted pair to CSV
#'
#' Every analysis pipeline emits the same `variance_components.csv` beside its
#' `results.rds`, so the column set lives here rather than in five places.
#'
#' @param models Fitted `full`/`null` pair.
#' @param spec_dir Directory to write `variance_components.csv` into.
#' @return The written data frame, invisibly.
#' @keywords internal
write_variance_components <- function(models, spec_dir) {
  vc_full <- as.data.frame(lme4::VarCorr(models$full))
  vc_full$model <- "full"
  vc_null <- as.data.frame(lme4::VarCorr(models$null))
  vc_null$model <- "null"
  vc <- rbind(vc_full, vc_null)
  vc <- vc[, c("model", "grp", "var1", "var2", "vcov", "sdcor")]
  utils::write.csv(vc, file.path(spec_dir, "variance_components.csv"),
                   row.names = FALSE)
  invisible(vc)
}
