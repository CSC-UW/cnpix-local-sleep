# Tests for the locality (Local vs Overlapping) analysis pipeline. The locality
# parquets are exported by cnpix_local_sleep's `export-locality-offs`, so the data-backed
# tests skip when they are absent.

locality_extdata_available <- function(fname) {
  nzchar(system.file("extdata", fname, package = "offp"))
}

per_condition_available <- function(off_type = "llas") {
  locality_extdata_available(
    paste0("summarized_locality_per_condition_", off_type, "_offs.parquet")
  )
}

nrem_conditions <- c(
  "Early.BSL.NREM", "Early.REC.NREM.Match", "Early.REC.NREM", "Late.REC.NREM"
)
nrem_posthocs <- c(
  "Early.REC.NREM - Early.REC.NREM.Match = 0",
  "Early.REC.NREM - Early.BSL.NREM = 0",
  "Early.REC.NREM - Late.REC.NREM = 0"
)

# -------------------- helpers that need no data --------------------

test_that("build_locality_comparisons builds an mcp on overlap_status", {
  cm <- build_locality_comparisons(c("Overlapping - Local = 0"))
  expect_true(inherits(cm, "mcp"))
  expect_named(cm, "overlap_status")
  expect_null(build_locality_comparisons(NULL))
  expect_null(build_locality_comparisons(character(0)))
})

test_that(".locality_group_weights is constant within a group and positive", {
  d <- data.frame(
    y = c(1, 3, 10, 14),
    overlap_status = c("Local", "Local", "Overlapping", "Overlapping")
  )
  w <- offp:::.locality_group_weights(d, "y", "overlap_status")
  expect_length(w, 4)
  expect_true(all(w > 0))
  expect_equal(w[1], w[2])
  expect_equal(w[3], w[4])
})

test_that("prepare_locality_data keeps only Local/Overlapping and sets levels", {
  d <- prepare_locality_data(data.frame(
    overlap_status = c("Local", "Overlapping", "Local"),
    clade = c("Cx", "Cx", "Cx"),
    y = 1:3
  ))
  expect_true(is.factor(d$overlap_status))
  expect_identical(levels(d$overlap_status), c("Local", "Overlapping"))
  expect_true(all(d$clade == "Cx"))
})

test_that(".locality_interaction_coef finds either coefficient ordering", {
  fe <- c("(Intercept)", "conditionWake",
          "conditionWake:overlap_statusOverlapping")
  expect_equal(
    offp:::.locality_interaction_coef(fe, "Wake"),
    "conditionWake:overlap_statusOverlapping"
  )
  expect_true(is.na(offp:::.locality_interaction_coef(fe, "NREM")))
})

# -------------------- data loader --------------------

test_that("prepare_locality_data on real extdata yields a 2-level factor", {
  skip_if_not(per_condition_available(), "locality per-condition extdata absent")
  d <- prepare_locality_data(load_locality_per_condition_summary("llas"))
  expect_identical(levels(d$overlap_status), c("Local", "Overlapping"))
  expect_setequal(as.character(unique(d$overlap_status)),
                  c("Local", "Overlapping"))
  expect_true(all(d$clade == "Cx"))
})

# -------------------- main-effect orchestrator (requests 2 & 3) --------------------

test_that("locality main-effect emits results.json with an array posthoc", {
  skip_if_not(per_condition_available(), "locality per-condition extdata absent")

  out <- file.path(tempdir(), "locality_main_test")
  unlink(out, recursive = TRUE)
  model_def <- list(
    name = "locality_paired",
    fe_terms = "overlap_status",
    re_terms = c("(1 | subject)", "(1 | structure)", "(1 | subject:structure)"),
    weighted = FALSE
  )
  res <- suppressWarnings(run_locality_main_effect_analysis(
    response_var = "median_duration", output_dir = out, model_def = model_def,
    off_type = "llas",
    analysis_kind = "cond-Early.REC.NREM", conditions = "Early.REC.NREM",
    save_ggplot_rds = FALSE
  ))
  expect_equal(res$condition_set, "cond-Early.REC.NREM")
  expect_true(file.exists(file.path(out, "locality_paired", "results.json")))

  skip_if_not(isTRUE(res$sig_main_effect) && !is.null(res$main_effect$posthoc),
              "locality main effect not significant; no posthoc block")
  js <- jsonlite::fromJSON(
    file.path(out, "locality_paired", "results.json"),
    simplifyVector = FALSE
  )
  ph <- js$main_effect$posthoc
  for (field in c("contrasts", "pvalues", "estimates", "cohens_d",
                  "ci_lower", "ci_upper")) {
    expect_true(is.list(ph[[field]]) && length(ph[[field]]) == 1,
                info = paste("posthoc field not a length-1 array:", field))
  }
  expect_equal(unlist(ph$contrasts), "Overlapping - Local")
})

# -------------------- interaction orchestrator --------------------

test_that("fit_locality_interaction_models LRT has df = (#conditions - 1)", {
  skip_if_not(per_condition_available(), "locality per-condition extdata absent")
  d <- prepare_locality_data(load_locality_per_condition_summary("llas"))
  d <- d[d$condition %in% nrem_conditions, ]
  d$condition <- factor(as.character(d$condition), levels = nrem_conditions)
  d$median_duration <- as.double(d$median_duration)
  m <- suppressWarnings(fit_locality_interaction_models(
    d, "median_duration", c("(1 | subject)", "(1 | subject:structure)")
  ))
  aov <- stats::anova(m$full, m$null)
  # Interaction df = (#conditions - 1) * (#overlap levels - 1) = 3 * 1.
  expect_equal(aov$Df[2], length(nrem_conditions) - 1)
})

test_that("DiD interaction contrast sign matches hand-computed cell means", {
  skip_if_not(per_condition_available(), "locality per-condition extdata absent")

  out <- file.path(tempdir(), "locality_did_test")
  unlink(out, recursive = TRUE)
  model_def <- list(
    name = "locality_interaction",
    fe_terms = c("condition", "overlap_status"),
    re_terms = c("(1 | subject)", "(1 | subject:structure)"),
    weighted = FALSE
  )
  res <- suppressWarnings(run_locality_interaction_analysis(
    response_var = "median_duration", output_dir = out, model_def = model_def,
    off_type = "llas", condition_set = "nrem",
    conditions = nrem_conditions, posthocs = nrem_posthocs,
    save_ggplot_rds = FALSE
  ))
  did <- res$interaction$did
  expect_false(is.null(did))
  expect_length(did$contrasts, length(nrem_posthocs))
  expect_true(file.exists(file.path(out, "locality_interaction", "results.json")))

  d <- prepare_locality_data(load_locality_per_condition_summary("llas"))
  d <- d[d$condition %in% nrem_conditions, ]
  cm <- tapply(
    as.double(d$median_duration),
    list(as.character(d$condition), as.character(d$overlap_status)),
    mean
  )
  did_hand <-
    (cm["Early.REC.NREM", "Overlapping"] - cm["Early.BSL.NREM", "Overlapping"]) -
    (cm["Early.REC.NREM", "Local"] - cm["Early.BSL.NREM", "Local"])
  idx <- match("Early.REC.NREM - Early.BSL.NREM", did$contrasts)
  expect_false(is.na(idx))
  expect_equal(sign(did$estimates[idx]), sign(did_hand))
})

test_that("build_locality_simple_effect_contrasts picks base (+ interaction) coef", {
  d <- expand.grid(
    rep = 1:10,
    condition = factor(c("C1", "C2"), levels = c("C1", "C2")),
    overlap_status = factor(c("Local", "Overlapping"),
                            levels = c("Local", "Overlapping"))
  )
  d$subject <- factor(rep(c("s1", "s2"), length.out = nrow(d)))
  set.seed(1)
  d$y <- stats::rnorm(nrow(d))
  m <- suppressWarnings(fit_locality_interaction_models(d, "y", "(1 | subject)"))
  K <- build_locality_simple_effect_contrasts(c("C1", "C2"), m$full)

  expect_equal(nrow(K), 2L)
  expect_identical(rownames(K),
                   c("Overlapping - Local | C1", "Overlapping - Local | C2"))
  fe <- names(lme4::fixef(m$full))
  base_i <- match("overlap_statusOverlapping", colnames(K))
  inter_i <- match(offp:::.locality_interaction_coef(fe, "C2"), colnames(K))
  # Reference condition C1: only the overlap_status main coefficient.
  expect_equal(unname(K[1, base_i]), 1)
  expect_equal(sum(K[1, ]), 1)
  # Non-reference C2: base + its interaction coefficient.
  expect_equal(unname(K[2, base_i]), 1)
  expect_equal(unname(K[2, inter_i]), 1)
  expect_equal(sum(K[2, ]), 2)
})

test_that("simple effects equal fixef linear combinations; overlap_main present", {
  skip_if_not(per_condition_available(), "locality per-condition extdata absent")

  out <- file.path(tempdir(), "locality_simple_test")
  unlink(out, recursive = TRUE)
  model_def <- list(
    name = "locality_interaction",
    fe_terms = c("condition", "overlap_status"),
    re_terms = c("(1 | subject)", "(1 | subject:structure)"),
    weighted = FALSE
  )
  res <- suppressWarnings(run_locality_interaction_analysis(
    response_var = "median_duration", output_dir = out, model_def = model_def,
    off_type = "llas", condition_set = "nrem",
    conditions = nrem_conditions, posthocs = nrem_posthocs,
    save_ggplot_rds = FALSE
  ))
  simple <- res$interaction$simple
  expect_false(is.null(simple))
  expect_length(simple$contrasts, length(nrem_conditions))

  fe <- lme4::fixef(res$models$full)
  ref <- nrem_conditions[1]  # Early.BSL.NREM (reference level)
  idx_ref <- match(paste0("Overlapping - Local | ", ref), simple$contrasts)
  # Reference-condition simple effect == the overlap_status main coefficient.
  expect_equal(simple$estimates[idx_ref],
               unname(fe["overlap_statusOverlapping"]), tolerance = 1e-6)
  # A non-reference condition == base + its interaction coefficient.
  nonref <- "Early.REC.NREM"
  ic <- offp:::.locality_interaction_coef(names(fe), nonref)
  idx_nr <- match(paste0("Overlapping - Local | ", nonref), simple$contrasts)
  expect_equal(simple$estimates[idx_nr],
               unname(fe["overlap_statusOverlapping"] + fe[ic]),
               tolerance = 1e-6)

  # Sign matches hand-computed within-condition Overlapping - Local cell means.
  d <- prepare_locality_data(load_locality_per_condition_summary("llas"))
  d <- d[d$condition %in% nrem_conditions, ]
  cm <- tapply(
    as.double(d$median_duration),
    list(as.character(d$condition), as.character(d$overlap_status)),
    mean
  )
  hand_ref <- cm[ref, "Overlapping"] - cm[ref, "Local"]
  expect_equal(sign(simple$estimates[idx_ref]), sign(hand_ref))

  # Marginal overlap_status main effect computed and serialized.
  expect_false(is.null(res$interaction$overlap_main))
  expect_true(is.numeric(res$interaction$overlap_main$pval))
  expect_true(file.exists(file.path(out, "locality_interaction", "results.json")))
})
