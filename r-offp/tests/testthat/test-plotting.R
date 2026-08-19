test_that("plot_rvf returns a ggplot", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_re <- c("(1 | subject)", "(1 | structure)")
  models <- fit_models(d, "rate", "condition", crossed_re)
  palette <- get_condition_palette()

  p <- plot_rvf(models$full, d$condition, palette)
  expect_s3_class(p, "ggplot")
})

test_that("plot_qqline returns a ggplot", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_re <- c("(1 | subject)", "(1 | structure)")
  models <- fit_models(d, "rate", "condition", crossed_re)
  palette <- get_condition_palette()

  p <- plot_qqline(models$full, d$condition, palette)
  expect_s3_class(p, "ggplot")
})

test_that("plot_rvf works with weighted = TRUE", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  w <- compute_condition_weights(d, "rate")
  crossed_re <- c("(1 | subject)", "(1 | structure)")
  models <- fit_models(d, "rate", "condition", crossed_re, weights = w)
  palette <- get_condition_palette()

  p <- plot_rvf(models$full, d$condition, palette, weighted = TRUE)
  expect_s3_class(p, "ggplot")
})

test_that("plot_qqline works with weighted = TRUE", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  w <- compute_condition_weights(d, "rate")
  crossed_re <- c("(1 | subject)", "(1 | structure)")
  models <- fit_models(d, "rate", "condition", crossed_re, weights = w)
  palette <- get_condition_palette()

  p <- plot_qqline(models$full, d$condition, palette, weighted = TRUE)
  expect_s3_class(p, "ggplot")
})

test_that("plot_distributions_by_condition returns a ggplot", {
  d <- cx_offs_summary()
  palette <- get_condition_palette()

  p <- plot_distributions_by_condition(
    d, condition, "rate", palette, geom = "violin"
  )
  expect_s3_class(p, "ggplot")
})
