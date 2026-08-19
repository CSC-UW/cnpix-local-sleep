test_that("get_anova_pval extracts correct value", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_re <- c("(1 | subject)", "(1 | structure)")
  models <- fit_models(d, "rate", "condition", crossed_re)
  anova_result <- stats::anova(models$full, models$null)
  pval <- get_anova_pval(anova_result)

  expect_true(is.numeric(pval))
  expect_true(pval >= 0 && pval <= 1)
})

test_that("test_main_effect returns expected structure", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_re <- c("(1 | subject)", "(1 | structure)")
  models <- fit_models(d, "rate", "condition", crossed_re)
  result <- test_main_effect(d, models, build_condition_comparisons(c(
    "Early.REC.NREM - Early.REC.NREM.Match = 0",
    "Early.REC.NREM - Early.BSL.NREM = 0",
    "Early.REC.NREM - Late.REC.NREM = 0",
    "Late.NOD.Wake - Early.NOD.Wake = 0"
  )))

  expect_type(result, "list")
  expect_true("anova" %in% names(result))
  expect_true("pval" %in% names(result))
  expect_true(is.numeric(result$pval))
})

test_that("test_main_effect works with weights", {
  d <- cx_offs_summary()
  d$rate <- as.double(d$rate)

  crossed_re <- c("(1 | subject)", "(1 | structure)")
  w <- compute_condition_weights(d, "rate")
  models <- fit_models(d, "rate", "condition", crossed_re, weights = w)
  result <- test_main_effect(d, models, build_condition_comparisons(c(
    "Early.REC.NREM - Early.REC.NREM.Match = 0",
    "Early.REC.NREM - Early.BSL.NREM = 0",
    "Early.REC.NREM - Late.REC.NREM = 0",
    "Late.NOD.Wake - Early.NOD.Wake = 0"
  )), weights = w)

  expect_type(result, "list")
  expect_true("anova" %in% names(result))
  expect_true("pval" %in% names(result))
  expect_true(is.numeric(result$pval))
})
