test_that("strip_random_effects removes random terms", {
  # Example A: y ~ x * z + (1|subject) -> y ~ x * z
  f <- y ~ x * z + (1 | subject)
  result <- strip_random_effects(f)
  expect_equal(deparse(result), "y ~ x * z")

  # Example C: y ~ (1|subject) -> y ~ 1
  f2 <- y ~ (1 | subject)
  result2 <- strip_random_effects(f2)
  expect_equal(deparse(result2), "y ~ 1")
})

test_that("strip_random_effects handles crossed random effects", {
  f <- y ~ condition + (1 | subject) + (1 | structure)
  result <- strip_random_effects(f)
  expect_equal(deparse(result), "y ~ condition")

  f2 <- y ~ (1 | subject) + (1 | structure)
  result2 <- strip_random_effects(f2)
  expect_equal(deparse(result2), "y ~ 1")

  f3 <- y ~ condition + (1 | subject) + (1 | structure) + (1 | subject:structure)
  result3 <- strip_random_effects(f3)
  expect_equal(deparse(result3), "y ~ condition")
})

test_that("cohens_local_fsquared computes correct value", {
  # Create simple lm models with known R^2
  set.seed(42)
  x <- 1:50
  y <- 2 * x + rnorm(50)
  z <- rep(letters[1:2], 25)

  model_a <- stats::lm(y ~ x + z)
  model_b <- stats::lm(y ~ z)

  f2 <- cohens_local_fsquared(model_a, model_b)
  expect_true(is.numeric(f2))
  expect_true(f2 > 0, info = "f^2 should be positive when model_a is better")
})

test_that("subtract_random_effects works with crossed random effects", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_re <- c("(1 | subject)", "(1 | structure)")
  models <- fit_models(d, "rate", "condition", crossed_re)
  d_fixed <- subtract_random_effects(d, models$full)

  expect_s3_class(d_fixed, "data.frame")
  expect_equal(nrow(d_fixed), nrow(d))
  expect_true("rate" %in% names(d_fixed))
})

test_that("subtract_random_effects works with crossed_interaction", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_interaction_re <- c(
    "(1 | subject)", "(1 | structure)", "(1 | subject:structure)"
  )
  models <- fit_models(d, "rate", "condition", crossed_interaction_re)
  d_fixed <- subtract_random_effects(d, models$full)

  expect_s3_class(d_fixed, "data.frame")
  expect_equal(nrow(d_fixed), nrow(d))
})

test_that("subtract_ranef_get_fsquared works with crossed models", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_re <- c("(1 | subject)", "(1 | structure)")
  models <- fit_models(d, "rate", "condition", crossed_re)
  result <- subtract_ranef_get_fsquared(d, models$full, models$null)

  expect_type(result, "list")
  expect_named(result, c("data", "models", "fsquared"))
  expect_true(is.numeric(result$fsquared))
  expect_s3_class(result$models$a, "lm")
  expect_s3_class(result$models$b, "lm")
})

test_that("subtract_ranef_get_fsquared works with weights", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_re <- c("(1 | subject)", "(1 | structure)")
  w <- compute_condition_weights(d, "rate")
  models <- fit_models(d, "rate", "condition", crossed_re, weights = w)
  result <- subtract_ranef_get_fsquared(
    d, models$full, models$null, weights = w
  )

  expect_type(result, "list")
  expect_named(result, c("data", "models", "fsquared"))
  expect_true(is.numeric(result$fsquared))
  expect_s3_class(result$models$a, "lm")
  expect_s3_class(result$models$b, "lm")
})
