test_that("build_condition_comparisons returns mcp with 4 contrasts", {
  posthocs <- c(
    "Early.REC.NREM - Early.REC.NREM.Match = 0",
    "Early.REC.NREM - Early.BSL.NREM = 0",
    "Early.REC.NREM - Late.REC.NREM = 0",
    "Late.NOD.Wake - Early.NOD.Wake = 0"
  )
  result <- build_condition_comparisons(posthocs)
  expect_s3_class(result, "mcp")
  expect_true("condition" %in% names(result))
  expect_equal(length(result$condition), 4)
})

test_that("build_condition_comparisons returns mcp with 3 contrasts", {
  posthocs <- c(
    "Early.REC.NREM - Early.REC.NREM.Match = 0",
    "Early.REC.NREM - Early.BSL.NREM = 0",
    "Early.REC.NREM - Late.REC.NREM = 0"
  )
  result <- build_condition_comparisons(posthocs)
  expect_s3_class(result, "mcp")
  expect_equal(length(result$condition), 3)
})

test_that("build_condition_comparisons returns NULL for empty input", {
  expect_null(build_condition_comparisons(NULL))
  expect_null(build_condition_comparisons(character(0)))
})
