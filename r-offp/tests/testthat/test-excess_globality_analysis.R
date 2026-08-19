# Tests for the cross-structure "excess globality" (observed vs null) pipeline.
# The synthetic-data tests pass `data=` directly, so they need no extdata (the
# r-offp port of cnpix_local_sleep's test_excess_above_chance_* synthetic tests).

# -------------------- helpers that need no data --------------------

test_that("build_excess_comparisons builds an mcp on quantity", {
  cm <- build_excess_comparisons(c("observed - null = 0"))
  expect_true(inherits(cm, "mcp"))
  expect_named(cm, "quantity")
  expect_null(build_excess_comparisons(NULL))
  expect_null(build_excess_comparisons(character(0)))
})

test_that("prepare_excess_globality_data keeps Cx and sets null as reference", {
  d <- prepare_excess_globality_data(data.frame(
    quantity = c("observed", "null", "observed"),
    clade = c("Cx", "Cx", "Cx"),
    value = 1:3
  ))
  expect_true(is.factor(d$quantity))
  expect_identical(levels(d$quantity), c("null", "observed"))
  expect_true(all(d$clade == "Cx"))
})

# synthetic data generator (mirrors the Python _excess_df_with_offset)

# One mean degree per (subject, structure, quantity): observed = null + offset
# (+ small noise). This is the real parquet's shape (one row per cell per
# quantity), with `(1 | subject:structure)` carrying the pairing.
make_excess_data <- function(offset, n_subjects = 9, n_structs = 2, seed = 1) {
  set.seed(seed)
  rows <- list()
  i <- 1
  for (si in seq_len(n_subjects)) {
    base <- stats::runif(1, 0.3, 0.7)
    for (st in seq_len(n_structs)) {
      null_val <- stats::runif(1, base, base + 0.1)
      obs_val <- null_val + offset + stats::rnorm(1, 0, 0.05)
      rows[[i]] <- data.frame(
        subject = paste0("S", si),
        structure = paste0("st", st),
        quantity = c("null", "observed"),
        value = c(null_val, obs_val),
        clade = "Cx",
        condition = "NREM",
        stringsAsFactors = FALSE
      )
      i <- i + 1
    }
  }
  prepare_excess_globality_data(do.call(rbind, rows))
}

excess_model_def <- function() {
  list(
    name = "excess_paired",
    fe_terms = "quantity",
    re_terms = c("(1 | subject)", "(1 | subject:structure)"),
    weighted = FALSE
  )
}

# significance tests (mirror test_excess_above_chance_*)

test_that("a positive observed-null offset yields a significant quantity effect", {
  d <- make_excess_data(offset = 0.2)
  out <- file.path(tempdir(), "excess_pos_test")
  unlink(out, recursive = TRUE)
  res <- suppressWarnings(run_excess_globality_analysis(
    response_var = "value", output_dir = out, model_def = excess_model_def(),
    conditions = "NREM", data = d, save_ggplot_rds = FALSE
  ))
  expect_true(res$sig_main_effect)
  expect_true(file.exists(file.path(out, "excess_paired", "results.json")))

  # observed is non-reference, so its fixed-effect coefficient is the
  # observed-vs-null contrast; it points positive (observed > null).
  fe <- lme4::fixef(res$models$full)
  expect_gt(unname(fe["quantityobserved"]), 0)

  # The serialized posthoc records the observed-vs-null contrast.
  js <- jsonlite::fromJSON(
    file.path(out, "excess_paired", "results.json"), simplifyVector = FALSE
  )
  expect_equal(unlist(js$main_effect$posthoc$contrasts), "observed - null")
})

test_that("a zero offset yields a non-significant quantity effect", {
  d <- make_excess_data(offset = 0.0)
  out <- file.path(tempdir(), "excess_null_test")
  unlink(out, recursive = TRUE)
  res <- suppressWarnings(run_excess_globality_analysis(
    response_var = "value", output_dir = out, model_def = excess_model_def(),
    conditions = "NREM", data = d, save_ggplot_rds = FALSE
  ))
  expect_false(res$sig_main_effect)
})
