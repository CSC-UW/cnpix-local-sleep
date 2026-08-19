test_that("OFF_TYPES contains llas, clas, blas, llas_exclusive", {
  expect_equal(OFF_TYPES, c("llas", "clas", "blas", "llas_exclusive"))
})

test_that("load_offs_summary loads every dataset type", {
  for (off_type in OFF_TYPES) {
    fname <- paste0("summarized_full48h_", off_type, "_offs.parquet")
    skip_if_not(nzchar(system.file("extdata", fname, package = "offp")),
                paste0("full48h ", off_type, " extdata not available"))
    expect_s3_class(load_offs_summary(off_type), "data.frame")
  }
})

test_that("load_offs_summary returns expected columns", {
  d <- load_offs_summary()
  # `layer` and `detection_mode` are WNE path components, not summary columns:
  # the cnpix_local_sleep summarizer groups by subject/probe/structure/condition only.
  expected_cols <- c(
    "subject", "probe", "condition", "structure", "clade"
  )
  for (col in expected_cols) {
    expect_true(col %in% names(d), info = paste("Missing column:", col))
  }
})

test_that("load_offs_summary contains expected conditions", {
  d <- load_offs_summary("llas")
  expected_conditions <- c(
    "Early.BSL.NREM", "Early.REC.NREM.Match", "Early.REC.NREM",
    "Late.REC.NREM", "Early.NOD.Wake", "Late.NOD.Wake"
  )
  actual_conditions <- unique(d$condition)
  for (cond in expected_conditions) {
    expect_true(
      cond %in% actual_conditions,
      info = paste("Missing condition:", cond)
    )
  }
})

test_that("load_offs_summary has response variable columns", {
  d <- load_offs_summary()
  response_vars <- c("median_duration", "rate", "total_area_norm")
  for (rv in response_vars) {
    expect_true(rv %in% names(d), info = paste("Missing response var:", rv))
  }
})

test_that("load_offs_summary full48h loads if exported, else errors clearly", {
  # The full48h_* files are exported into extdata by cnpix_local_sleep's
  # `off-analysis export-full48h-offs`; they may be absent in a fresh checkout.
  path <- system.file(
    "extdata", "summarized_full48h_llas_offs.parquet", package = "offp"
  )
  if (nzchar(path)) {
    d <- load_offs_summary("llas")
    expect_s3_class(d, "data.frame")
  } else {
    expect_error(
      load_offs_summary("llas"),
      "export-full48h-offs"
    )
  }
})

test_that("full48h llas_exclusive loads if exported, else errors clearly", {
  # summarized_full48h_llas_exclusive_offs.parquet is exported into extdata by
  # cnpix_local_sleep's `off-analysis export-full48h-exclusive-offs`; it may be absent in a
  path <- system.file(
    "extdata", "summarized_full48h_llas_exclusive_offs.parquet", package = "offp"
  )
  if (nzchar(path)) {
    d <- load_offs_summary("llas_exclusive")
    expect_s3_class(d, "data.frame")
  } else {
    expect_error(
      load_offs_summary("llas_exclusive"),
      "export-full48h"
    )
  }
})
