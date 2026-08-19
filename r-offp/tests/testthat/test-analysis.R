nrem_conditions <- c(
  "Early.BSL.NREM", "Early.REC.NREM.Match", "Early.REC.NREM", "Late.REC.NREM"
)

# A single-contrast condition set (like `wake`) exercises the jsonlite
# auto_unbox edge case: length-1 posthoc vectors must stay JSON arrays.
nrem_one_contrast <- "Early.REC.NREM - Late.REC.NREM = 0"

test_that("run_cx_homeostasis_analysis records condition_set and conditions", {
  skip_if_not(
    nzchar(system.file("extdata", "summarized_full48h_llas_offs.parquet",
                       package = "offp")),
    "full48h LLAS extdata not available"
  )

  out <- file.path(tempdir(), "cset_record_test")
  unlink(out, recursive = TRUE)

  model_def <- list(
    name = "crossed",
    fe_terms = "condition",
    re_terms = c("(1 | subject)", "(1 | structure)"),
    weighted = FALSE
  )

  res <- suppressWarnings(run_cx_homeostasis_analysis(
    response_var = "rate",
    output_dir = out,
    model_def = model_def,
    off_type = "llas",
    condition_set = "nrem",
    conditions = nrem_conditions,
    posthocs = nrem_one_contrast
  ))

  # Recorded on the result list.
  expect_equal(res$condition_set, "nrem")
  expect_setequal(res$conditions, nrem_conditions)
  # Filtering actually restricted the fit to the requested conditions.
  expect_setequal(as.character(unique(res$data$condition)), nrem_conditions)

  # Recorded in the JSON summary.
  js <- jsonlite::fromJSON(
    file.path(out, "crossed", "results.json"),
    simplifyVector = FALSE
  )
  expect_equal(js$condition_set, "nrem")
  expect_setequal(unlist(js$conditions), nrem_conditions)

  # Recorded in the text summary.
  txt <- readLines(file.path(out, "crossed", "summary.txt"))
  expect_true(any(grepl("Condition set: nrem", txt, fixed = TRUE)))
})

test_that("single-contrast posthocs stay JSON arrays (auto_unbox guard)", {
  skip_if_not(
    nzchar(system.file("extdata", "summarized_full48h_llas_offs.parquet",
                       package = "offp")),
    "full48h LLAS extdata not available"
  )

  out <- file.path(tempdir(), "cset_array_test")
  unlink(out, recursive = TRUE)

  model_def <- list(
    name = "crossed",
    fe_terms = "condition",
    re_terms = c("(1 | subject)", "(1 | structure)"),
    weighted = FALSE
  )

  res <- suppressWarnings(run_cx_homeostasis_analysis(
    response_var = "rate",
    output_dir = out,
    model_def = model_def,
    off_type = "llas",
    condition_set = "nrem",
    conditions = nrem_conditions,
    posthocs = nrem_one_contrast
  ))

  skip_if_not(
    isTRUE(res$sig_main_effect) && !is.null(res$main_effect$posthoc),
    "main effect not significant; no posthoc block to check"
  )

  # simplifyVector = FALSE: a JSON array decodes to a list, a scalar does not.
  js <- jsonlite::fromJSON(
    file.path(out, "crossed", "results.json"),
    simplifyVector = FALSE
  )
  ph <- js$main_effect$posthoc
  for (field in c("contrasts", "pvalues", "estimates", "cohens_d",
                  "ci_lower", "ci_upper")) {
    expect_true(
      is.list(ph[[field]]) && length(ph[[field]]) == 1,
      info = paste("posthoc field not a length-1 array:", field)
    )
  }
})
