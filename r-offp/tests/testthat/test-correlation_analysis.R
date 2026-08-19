# Tests for the NOD-activity vs NREM.Rebound correlation pipeline. The
# correlation input parquet is exported by cnpix_local_sleep's
# `export-nod-rebound-correlation`, so the data-backed end-to-end test skips when
# it is absent. The reshape and fit tests use synthetic fixtures (no extdata).

correlation_extdata_available <- function(off_type = "llas") {
  nzchar(system.file(
    "extdata",
    paste0("nod_rebound_correlation_", off_type, "_offs.parquet"),
    package = "offp"
  ))
}

# -------------------- build_correlation_frame (synthetic, no data) --------------------

test_that("build_correlation_frame pivots and computes the rebound difference", {
  d <- data.frame(
    subject = c("a", "a", "a", "b", "b", "b"),
    probe = "imec0",
    structure = "X",
    condition = rep(c("NOD", "Early.REC.NREM", "Early.REC.NREM.Match"), 2),
    clade = "Cx",
    total_area_norm = c(5, 8, 3, 7, 10, 4),
    stringsAsFactors = FALSE
  )
  fr <- build_correlation_frame(d, "total_area_norm")
  expect_setequal(as.character(fr$subject), c("a", "b"))
  expect_true(is.factor(fr$subject))
  # a: x = NOD = 5, y = post - base = 8 - 3 = 5
  ra <- fr[fr$subject == "a", ]
  expect_equal(ra$x, 5)
  expect_equal(ra$y, 5)
  # b: x = 7, y = 10 - 4 = 6
  rb <- fr[fr$subject == "b", ]
  expect_equal(rb$x, 7)
  expect_equal(rb$y, 6)
})

test_that("build_correlation_frame honors a custom predictor_condition", {
  # Both NOD and NOD.Wake present; selecting NOD.Wake must use its rows for x
  # while the response (y) still comes from the two REC conditions.
  d <- data.frame(
    subject = "a", probe = "imec0", structure = "X",
    condition = c("NOD", "NOD.Wake", "Early.REC.NREM", "Early.REC.NREM.Match"),
    clade = "Cx",
    rate = c(5, 2, 8, 3),
    stringsAsFactors = FALSE
  )
  fr_nod <- build_correlation_frame(d, "rate", predictor_condition = "NOD")
  fr_wake <- build_correlation_frame(d, "rate", predictor_condition = "NOD.Wake")
  expect_equal(fr_nod$x, 5)        # NOD rate
  expect_equal(fr_wake$x, 2)       # NOD.Wake rate
  expect_equal(fr_nod$y, 5)        # y = 8 - 3, unchanged by predictor choice
  expect_equal(fr_wake$y, 5)
})

test_that("build_correlation_frame drops combos missing a condition and non-Cx", {
  d <- data.frame(
    subject = c("a", "a", "a", "c", "z", "z", "z"),
    probe = "imec0",
    structure = c("X", "X", "X", "X", "Y", "Y", "Y"),
    condition = c(
      "NOD", "Early.REC.NREM", "Early.REC.NREM.Match",  # a: complete
      "NOD",                                            # c: only NOD -> dropped
      "NOD", "Early.REC.NREM", "Early.REC.NREM.Match"   # z: complete but non-Cx
    ),
    clade = c("Cx", "Cx", "Cx", "Cx", "Th", "Th", "Th"),
    rate = c(1, 2, 1, 9, 3, 4, 2),
    stringsAsFactors = FALSE
  )
  fr <- build_correlation_frame(d, "rate")
  # Only subject a survives: c is incomplete, z is non-cortical.
  expect_equal(nrow(fr), 1)
  expect_equal(as.character(fr$subject), "a")
})

test_that("build_correlation_frame joins a distinct response_metric from response_data", {
  # x = OFF predictor (total_area_norm @ NOD) from `d`; y = a DIFFERENT quantity's
  # NREM.Rebound (mean_zlog_delta) taken from a separate `response_data` frame,
  # joined on (subject, probe, structure).
  d <- data.frame(
    subject = c("a", "b", "c"),
    probe = "imec0",
    structure = "X",
    condition = "NOD",
    clade = "Cx",
    total_area_norm = c(5, 7, 9),
    stringsAsFactors = FALSE
  )
  # Response frame carries mean_zlog_delta at the two rebound conditions. Subject
  # c is present in x but absent here -> must be dropped.
  resp <- data.frame(
    subject = rep(c("a", "b"), each = 2),
    probe = "imec0",
    structure = "X",
    condition = rep(c("Early.REC.NREM", "Early.REC.NREM.Match"), 2),
    clade = "Cx",
    mean_zlog_delta = c(1.5, 0.5, 2.0, 0.8),
    stringsAsFactors = FALSE
  )
  fr <- build_correlation_frame(
    d, "total_area_norm",
    response_metric = "mean_zlog_delta", response_data = resp
  )
  expect_setequal(as.character(fr$subject), c("a", "b"))  # c dropped
  ra <- fr[fr$subject == "a", ]
  expect_equal(ra$x, 5)              # total_area_norm @ NOD from d
  expect_equal(ra$y, 1.5 - 0.5)     # mean_zlog_delta rebound from resp
  rb <- fr[fr$subject == "b", ]
  expect_equal(rb$x, 7)
  expect_equal(rb$y, 2.0 - 0.8)
})

test_that("build_correlation_frame errors when response_metric absent from response_data", {
  d <- data.frame(
    subject = "a", probe = "imec0", structure = "X",
    condition = "NOD", clade = "Cx", total_area_norm = 5,
    stringsAsFactors = FALSE
  )
  resp <- data.frame(
    subject = "a", probe = "imec0", structure = "X",
    condition = c("Early.REC.NREM", "Early.REC.NREM.Match"),
    clade = "Cx", mean_zlog_delta = c(1, 0),
    stringsAsFactors = FALSE
  )
  expect_error(
    build_correlation_frame(
      d, "total_area_norm",
      response_metric = "not_a_column", response_data = resp
    ),
    "not_a_column"
  )
})

# -------------------- fit + slope test (synthetic, both re_terms) --------------------

make_fit_fixture <- function() {
  set.seed(1)
  subs <- paste0("s", 1:8)
  structs <- paste0("S", 1:4)
  grid <- expand.grid(
    subject = subs, structure = structs, stringsAsFactors = FALSE
  )
  grid$probe <- "imec0"
  n <- nrow(grid)
  subj_eff <- stats::setNames(stats::rnorm(length(subs), 0, 1), subs)
  grid$x <- stats::rnorm(n, 10, 3)
  grid$y <- 2 - 0.5 * grid$x + subj_eff[grid$subject] + stats::rnorm(n, 0, 0.5)
  grid$subject <- factor(grid$subject)
  grid$structure <- factor(grid$structure)
  grid
}

test_that("fit + slope test runs for (1|subject) and recovers a negative slope", {
  d <- make_fit_fixture()
  models <- suppressWarnings(fit_correlation_models(d, "(1 | subject)"))
  res <- test_correlation_slope(models, d)
  expect_true(is.finite(res$slope))
  expect_lt(res$slope, 0)               # fixture has a true slope of -0.5
  expect_true(res$pval >= 0 && res$pval <= 1)
  expect_true(is.finite(res$fsquared) && res$fsquared >= 0)
  expect_equal(res$n, nrow(d))
  expect_equal(res$n_subjects, 8)
  expect_true(res$ci_lower < res$slope && res$slope < res$ci_upper)
})

test_that("crossed (1|subject)+(1|structure) fits and yields a finite f^2", {
  d <- make_fit_fixture()
  models <- suppressWarnings(
    fit_correlation_models(d, c("(1 | subject)", "(1 | structure)"))
  )
  # Two grouping factors must appear in the full model's random effects.
  expect_setequal(names(lme4::ranef(models$full)), c("subject", "structure"))
  res <- suppressWarnings(test_correlation_slope(models, d))
  expect_true(is.finite(res$fsquared) && res$fsquared >= 0)
  expect_true(is.finite(res$slope))
})

# end-to-end orchestrator (data-backed; skips without extdata)

test_that("run_nod_rebound_correlation writes results.json for both models", {
  skip_if_not(correlation_extdata_available(), "correlation LLAS extdata absent")

  out <- file.path(tempdir(), "corr_test")
  unlink(out, recursive = TRUE)
  dat <- load_nod_rebound_correlation_data("llas")

  for (md in list(
    list(name = "subject", re_terms = "(1 | subject)"),
    list(name = "subject_structure",
         re_terms = c("(1 | subject)", "(1 | structure)"))
  )) {
    res <- suppressWarnings(run_nod_rebound_correlation(
      metric = "total_area_norm", off_type = "llas", model_def = md,
      output_dir = out, data = dat, save_ggplot_rds = FALSE
    ))
    spec_dir <- file.path(out, md$name)
    expect_true(file.exists(file.path(spec_dir, "results.json")))
    expect_true(file.exists(file.path(spec_dir, "summary.txt")))
    expect_true(file.exists(file.path(
      spec_dir, "figures", "total_area_norm_NOD_vs_NREM.Rebound.svg"
    )))
    js <- jsonlite::fromJSON(file.path(spec_dir, "results.json"))
    expect_equal(js$metric, "total_area_norm")
    expect_equal(js$model, md$name)
    expect_true(is.numeric(js$pval) && js$pval >= 0 && js$pval <= 1)
    expect_true(is.finite(js$slope))
    expect_equal(js$n_subjects, length(unique(res$data$subject)))
  }
})
