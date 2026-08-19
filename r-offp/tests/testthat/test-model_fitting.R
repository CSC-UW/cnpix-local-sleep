# Convenience constants for common RE term vectors
crossed_re <- c("(1 | subject)", "(1 | structure)")
crossed_interaction_re <- c(
  "(1 | subject)", "(1 | structure)", "(1 | subject:structure)"
)

test_that("fit_models returns correct structure for crossed RE terms", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  models <- fit_models(d, "rate", "condition", crossed_re)

  expect_type(models, "list")
  expect_named(models, c("full", "null"))
  expect_s4_class(models$full, "lmerMod")
  expect_s4_class(models$null, "lmerMod")
})

test_that("fit_models returns correct structure for crossed_interaction RE terms", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  models <- fit_models(d, "rate", "condition", crossed_interaction_re)

  expect_type(models, "list")
  expect_named(models, c("full", "null"))
  expect_s4_class(models$full, "lmerMod")
  expect_s4_class(models$null, "lmerMod")
})

test_that("fit_models errors on invalid fe_terms", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  expect_error(fit_models(d, "rate", c("a", "b"), crossed_re))
  expect_error(fit_models(d, "rate", 123, crossed_re))
})

test_that("fit_models errors on missing column", {
  d <- data.frame(x = 1:10, y = rnorm(10))
  expect_error(
    fit_models(d, "y", "condition", crossed_re),
    "not found in data"
  )
})

test_that("compute_condition_weights returns correct length and positivity", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  w <- compute_condition_weights(d, "rate")

  expect_length(w, nrow(d))
  expect_true(all(w > 0))
  expect_true(all(is.finite(w)))
})

test_that("compute_condition_weights assigns same weight within condition", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  w <- compute_condition_weights(d, "rate")

  for (cond in unique(d$condition)) {
    mask <- d$condition == cond
    expect_equal(length(unique(w[mask])), 1)
  }
})

test_that("fit_models accepts weights parameter", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  w <- compute_condition_weights(d, "rate")
  models <- fit_models(d, "rate", "condition", crossed_re, weights = w)

  expect_type(models, "list")
  expect_named(models, c("full", "null"))
  expect_s4_class(models$full, "lmerMod")
  expect_s4_class(models$null, "lmerMod")
  expect_equal(weights(models$full), w)
})

test_that("VarCorr returns expected columns for fitted models", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  models <- fit_models(d, "rate", "condition", crossed_re)
  vc <- as.data.frame(lme4::VarCorr(models$full))

  expect_s3_class(vc, "data.frame")
  expect_true(all(c("grp", "var1", "var2", "vcov", "sdcor") %in% names(vc)))
  expect_true(nrow(vc) >= 3) # subject, structure, residual at minimum
  expect_true(all(vc$vcov >= 0))
})

test_that("validate_model_def accepts valid definition", {
  model_def <- list(
    name = "crossed",
    fe_terms = "condition",
    re_terms = c("(1 | subject)", "(1 | structure)"),
    weighted = FALSE
  )
  expect_invisible(validate_model_def(model_def))
})

test_that("validate_model_def errors on missing fields", {
  expect_error(
    validate_model_def(list(name = "crossed")),
    "missing required fields"
  )
})

test_that("validate_model_def errors on multiple fe_terms", {
  model_def <- list(
    name = "bad",
    fe_terms = c("condition", "extra"),
    re_terms = "(1 | subject)",
    weighted = FALSE
  )
  expect_error(validate_model_def(model_def), "length-1")
})

test_that("validate_model_def errors on wrong types", {
  expect_error(
    validate_model_def(list(
      name = 123, fe_terms = "condition",
      re_terms = "(1 | subject)", weighted = FALSE
    )),
    "single string"
  )
  expect_error(
    validate_model_def(list(
      name = "ok", fe_terms = "condition",
      re_terms = "(1 | subject)", weighted = "yes"
    )),
    "TRUE or FALSE"
  )
})

test_that("validate_model_def accepts a valid optional transform", {
  model_def <- list(
    name = "crossed_log",
    fe_terms = "condition",
    re_terms = c("(1 | subject)", "(1 | structure)"),
    weighted = FALSE,
    transform = "log"
  )
  expect_invisible(validate_model_def(model_def))
})

test_that("validate_model_def errors on unknown transform", {
  model_def <- list(
    name = "bad",
    fe_terms = "condition",
    re_terms = "(1 | subject)",
    weighted = FALSE,
    transform = "boxcox"
  )
  expect_error(validate_model_def(model_def), "transform must be one of")
})

test_that("apply_response_transform applies known transforms", {
  x <- c(1, exp(1), exp(2))
  expect_equal(apply_response_transform(x, "identity"), x)
  expect_equal(apply_response_transform(x, "log"), c(0, 1, 2))
  expect_equal(apply_response_transform(c(1, 100), "log10"), c(0, 2))
  expect_equal(apply_response_transform(c(0, 3), "log1p"), log1p(c(0, 3)))
  expect_equal(apply_response_transform(c(0, 4, 9), "sqrt"), c(0, 2, 3))
})

test_that("apply_response_transform defaults to identity", {
  x <- c(0.5, 1.5, 2.5)
  expect_equal(apply_response_transform(x), x)
})

test_that("apply_response_transform validates domain and name", {
  expect_error(apply_response_transform(c(1, 2), "nope"), "Unknown transform")
  expect_error(
    apply_response_transform(c(-1, 1), "log"), "strictly positive"
  )
  expect_error(apply_response_transform(c(0, 1), "log"), "strictly positive")
  expect_error(apply_response_transform(c(-2, 1), "log1p"), "log1p")
  expect_error(apply_response_transform(c(-1, 4), "sqrt"), "non-negative")
})
