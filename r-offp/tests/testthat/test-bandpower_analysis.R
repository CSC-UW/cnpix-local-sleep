# Tests for the band-power condition-homeostasis companion pipeline.
# The synthetic-data test passes `data=` directly (bandpower measures are not OFF
# properties), so it needs no extdata. It verifies that the shared
# run_cx_homeostasis_analysis core accepts the bandpower off_type provenance
# label (the assertion relaxation) and fits/writes normally.

# -------------------- synthetic data generator --------------------

# One mean_zlog_delta per (subject, structure, condition), with a rebound signal
# (Early.REC.NREM highest). This is the real parquet's shape.
make_bandpower_df <- function(n_subjects = 6,
                              structures = c("PPC", "M2", "V1"),
                              seed = 1) {
  set.seed(seed)
  conds <- c("Early.BSL.NREM", "Early.REC.NREM.Match",
             "Early.REC.NREM", "Late.REC.NREM")
  subjects <- paste0("S", seq_len(n_subjects))
  d <- expand.grid(subject = subjects, structure = structures,
                   condition = conds, stringsAsFactors = FALSE)
  d$probe <- "imec0"
  d$clade <- "Cx"
  base <- c(1.0, 0.6, 1.7, 1.1)[match(d$condition, conds)]
  subj_off <- stats::rnorm(n_subjects, 0, 0.3)[match(d$subject, subjects)]
  d$mean_zlog_delta <- base + subj_off + stats::rnorm(nrow(d), 0, 0.2)
  d
}

nrem_model_def <- function() {
  list(name = "crossed_interaction", fe_terms = "condition",
       re_terms = c("(1 | subject)", "(1 | structure)",
                    "(1 | subject:structure)"),
       weighted = FALSE)
}

nrem_conds <- c("Early.BSL.NREM", "Early.REC.NREM.Match",
                "Early.REC.NREM", "Late.REC.NREM")
nrem_posthocs <- c("Early.REC.NREM - Early.REC.NREM.Match = 0",
                   "Early.REC.NREM - Early.BSL.NREM = 0",
                   "Early.REC.NREM - Late.REC.NREM = 0")

# -------------------- tests --------------------

test_that("run_cx_homeostasis_analysis accepts the bandpower provenance label", {
  d <- make_bandpower_df()
  outdir <- tempfile("bp_")
  res <- run_cx_homeostasis_analysis(
    response_var = "mean_zlog_delta", output_dir = outdir,
    model_def = nrem_model_def(), off_type = "bandpower",
    condition_set = "nrem",
    conditions = nrem_conds, posthocs = nrem_posthocs, data = d
  )
  expect_identical(res$off_type, "bandpower")
  expect_true(res$sig_main_effect)
  expect_true(file.exists(file.path(outdir, "crossed_interaction",
                                    "results.json")))
})

test_that("OFF-enum assertions still fire when data is loaded internally", {
  # With no `data=`, off_type must be a valid OFF selector.
  expect_error(
    run_cx_homeostasis_analysis(
      response_var = "mean_zlog_delta", output_dir = tempfile(),
      model_def = nrem_model_def(), off_type = "bandpower",
      conditions = nrem_conds
    ),
    "off_type"
  )
})
