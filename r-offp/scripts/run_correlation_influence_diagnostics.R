#!/usr/bin/env Rscript
# Influence / outlier diagnostics for the NOD -> NREM.Rebound correlation fits.
#
# Reproducible, paper-grade companion to run_correlation_xtransform_sensitivity.R.
# Asks whether the raw-x mixed-model slopes are driven by a few high-leverage
# points (a sparse right tail of the predictor). Covers every predictor
# (NOD, NOD.Wake, ...) in one sweep; per-fit outputs are predictor-scoped
# (influence_diagnostics/<predictor>/<slug>/) and the consolidated summary carries
# a `predictor` column, so predictors never collide. Computes, per existing fit
# (_output_correlations/<off>/<metric>/<predictor>/<model>/results.rds):
#
#   - marginal_fit.csv : the population fixed-effect line (intercept, slope, LRT p)
#     and the same line refit WITHOUT the two most influential subjects, so the
#     leverage story is quantified, not asserted.
#   - loco.csv  : leave-one-COMBO-out (drop each (subject,probe,structure) row).
#     Per combo: marginal residual (raw + SD units), x leverage (SD units),
#     deleted slope + LRT p, DFBETA (raw + SD-standardised), multivariate Cook's D
#     over the 2 fixed effects, and whether significance flips.
#   - loso.csv  : leave-one-SUBJECT-out and (crossed model) leave-one-STRUCTURE-out,
#     with the same Cook's D / slope / p / flip columns.
#   - leave2.csv : leave-two-SUBJECTS-out over all subject pairs (masking check that
#     single-deletion cannot see), with deleted slope + p per pair.
#   - leave3_worst.csv : (raw-significant fits only) leave-three-subjects-out, the
#     trio giving the largest p, plus the full grid.
#
# and a consolidated influence_summary.{csv,md} across all fits (flip counts,
# worst single/pair/trio deletion). Cook's D uses the influence.ME definition
#   D = (1/p_fe) (b - b_drop)^T vcov(full)^{-1} (b - b_drop),  p_fe = 2.
# Cook's-D cutoffs (4/n_groups, or 1) are heuristic; the decision-relevant readout
# is whether a deletion flips significance, not whether D clears a threshold.
#
# Reuses the pipeline's own model machinery (offp::fit_correlation_models); p and
# slope are extracted exactly as the pipeline does (LRT of full vs null).
# Additive: writes only under _output_correlations/influence_diagnostics/
# and touches no existing output. Run from the r-offp package root.

suppressWarnings(suppressMessages({
  library(offp)
  library(lme4)
}))

CORR_ROOT <- "_output_correlations"
OUT_ROOT <- file.path(CORR_ROOT, "influence_diagnostics")
dir.create(OUT_ROOT, recursive = TRUE, showWarnings = FALSE)

# lightweight refit returning just what the loops need (p, slope, intercept)
# p is the LRT of full vs null, identical to the pipeline's get_anova_pval().
fit_ps <- function(d, re_terms) {
  ff <- stats::reformulate(c("x", re_terms), response = "y")
  fn <- stats::reformulate(c("1", re_terms), response = "y")
  mf <- suppressWarnings(suppressMessages(lme4::lmer(ff, d, REML = FALSE)))
  mn <- suppressWarnings(suppressMessages(lme4::lmer(fn, d, REML = FALSE)))
  a <- suppressWarnings(stats::anova(mf, mn))
  b <- lme4::fixef(mf)
  list(p = a[["Pr(>Chisq)"]][2], slope = unname(b[["x"]]),
       intercept = unname(b[["(Intercept)"]]))
}

files <- sort(Sys.glob(file.path(CORR_ROOT, "*", "*", "*", "*", "results.rds")))
stopifnot(length(files) > 0)

summ <- list()

for (f in files) {
  res <- readRDS(f)
  d <- res$data
  d$subject <- as.character(d$subject)
  d$structure <- as.character(d$structure)
  re_terms <- res$model_def$re_terms
  metric <- res$metric; off_type <- res$off_type; model_name <- res$model_def$name
  predictor <- res$predictor_condition
  # Response metric distinguishes self-rebound from cross-metric fits sharing the
  # same predictor metric; fold it into the slug so their outputs never collide.
  response_metric <- if (is.null(res$response_metric)) metric else res$response_metric
  metric_seg <- if (identical(response_metric, metric)) {
    metric
  } else {
    paste0(metric, "__vs__", response_metric)
  }
  slug <- paste(off_type, metric_seg, model_name, sep = "_")
  out_dir <- file.path(OUT_ROOT, predictor, slug)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # baseline full model (for Cook's D vcov + marginal line)
  ff <- stats::reformulate(c("x", re_terms), response = "y")
  m_full <- suppressWarnings(suppressMessages(lme4::lmer(ff, d, REML = FALSE)))
  b0 <- lme4::fixef(m_full)
  V0 <- as.matrix(stats::vcov(m_full)); Vinv <- solve(V0); pfe <- length(b0)
  base <- fit_ps(d, re_terms)
  slope0 <- base$slope; p0 <- base$p; se0 <- sqrt(V0["x", "x"])
  sig0 <- isTRUE(p0 < 0.05)
  cookD <- function(bi) as.numeric(t(b0 - bi) %*% Vinv %*% (b0 - bi)) / pfe

  # marginal fixed-effect line + leverage/residual per combo
  d$x_z <- (d$x - mean(d$x)) / stats::sd(d$x)
  d$yhat_marg <- base$intercept + slope0 * d$x
  d$resid_marg <- d$y - d$yhat_marg
  sdr <- stats::sd(d$resid_marg)
  d$resid_marg_z <- d$resid_marg / sdr

  # -------------------- leave-one-combo-out --------------------
  n <- nrow(d)
  loco <- data.frame(
    i = seq_len(n), subject = d$subject, structure = d$structure,
    x = d$x, x_z = d$x_z, y = d$y, yhat_marg = d$yhat_marg,
    resid_marg = d$resid_marg, resid_marg_z = d$resid_marg_z,
    slope_full = slope0, slope_drop = NA_real_, p_full = p0, p_drop = NA_real_,
    dfbeta = NA_real_, dfbeta_std = NA_real_, cookD = NA_real_
  )
  for (i in seq_len(n)) {
    di <- d[-i, , drop = FALSE]
    r <- tryCatch(fit_ps(di, re_terms), error = function(e) NULL)
    if (is.null(r)) next
    mi <- suppressWarnings(suppressMessages(lme4::lmer(ff, di, REML = FALSE)))
    bi <- lme4::fixef(mi)
    loco$slope_drop[i] <- r$slope; loco$p_drop[i] <- r$p
    loco$dfbeta[i] <- slope0 - r$slope
    loco$dfbeta_std[i] <- (slope0 - r$slope) / se0
    loco$cookD[i] <- cookD(bi)
  }
  loco$flip <- sig0 != (loco$p_drop < 0.05)
  utils::write.csv(loco, file.path(out_dir, "loco.csv"), row.names = FALSE)

  # leave-one-group-out (subject; + structure for crossed)
  group_loo <- function(col) {
    lv <- sort(unique(d[[col]]))
    o <- data.frame(level = col, unit = lv, cookD = NA_real_,
                    slope_drop = NA_real_, p_drop = NA_real_)
    for (k in seq_along(lv)) {
      dk <- d[d[[col]] != lv[k], , drop = FALSE]
      r <- tryCatch(fit_ps(dk, re_terms), error = function(e) NULL)
      if (is.null(r)) next
      mk <- suppressWarnings(suppressMessages(lme4::lmer(ff, dk, REML = FALSE)))
      o$cookD[k] <- cookD(lme4::fixef(mk))
      o$slope_drop[k] <- r$slope; o$p_drop[k] <- r$p
    }
    o
  }
  loso <- group_loo("subject")
  if (length(re_terms) > 1) loso <- rbind(loso, group_loo("structure"))
  loso$flip <- sig0 != (loso$p_drop < 0.05)
  utils::write.csv(loso, file.path(out_dir, "loso.csv"), row.names = FALSE)

  # -------------------- marginal fit + no-top-2-subjects line --------------------
  top2 <- loso[loso$level == "subject", ]
  top2 <- top2$unit[order(-top2$cookD)][1:2]
  d_ex <- d[!d$subject %in% top2, , drop = FALSE]
  ex <- fit_ps(d_ex, re_terms)
  mfit <- data.frame(
    intercept = base$intercept, slope = slope0, p = p0, se_slope = se0,
    resid_sd = sdr, n = n, n_subjects = length(unique(d$subject)),
    top2_subjects = paste(top2, collapse = "+"),
    slope_excl_top2 = ex$slope, intercept_excl_top2 = ex$intercept,
    p_excl_top2 = ex$p
  )
  utils::write.csv(mfit, file.path(out_dir, "marginal_fit.csv"), row.names = FALSE)

  # -------------------- leave-2-subjects-out (all pairs) --------------------
  subs <- sort(unique(d$subject))
  pair_idx <- utils::combn(length(subs), 2)
  l2 <- data.frame(unit_i = subs[pair_idx[1, ]], unit_j = subs[pair_idx[2, ]],
                   slope_drop = NA_real_, p_drop = NA_real_)
  for (k in seq_len(ncol(pair_idx))) {
    dk <- d[!d$subject %in% subs[pair_idx[, k]], , drop = FALSE]
    r <- tryCatch(fit_ps(dk, re_terms), error = function(e) NULL)
    if (is.null(r)) next
    l2$slope_drop[k] <- r$slope; l2$p_drop[k] <- r$p
  }
  utils::write.csv(l2, file.path(out_dir, "leave2.csv"), row.names = FALSE)
  worst2 <- l2[which.max(l2$p_drop), ]

  # leave-3-subjects-out (raw-significant fits only)
  worst3_p <- NA_real_; worst3_units <- NA_character_; n_trios <- NA_integer_
  if (sig0) {
    trio_idx <- utils::combn(length(subs), 3)
    l3 <- data.frame(unit_i = subs[trio_idx[1, ]], unit_j = subs[trio_idx[2, ]],
                     unit_k = subs[trio_idx[3, ]],
                     slope_drop = NA_real_, p_drop = NA_real_)
    for (k in seq_len(ncol(trio_idx))) {
      dk <- d[!d$subject %in% subs[trio_idx[, k]], , drop = FALSE]
      r <- tryCatch(fit_ps(dk, re_terms), error = function(e) NULL)
      if (is.null(r)) next
      l3$slope_drop[k] <- r$slope; l3$p_drop[k] <- r$p
    }
    utils::write.csv(l3[order(-l3$p_drop), ],
                     file.path(out_dir, "leave3.csv"), row.names = FALSE)
    w3 <- l3[which.max(l3$p_drop), ]
    worst3_p <- w3$p_drop
    worst3_units <- paste(w3$unit_i, w3$unit_j, w3$unit_k, sep = "+")
    n_trios <- sum(!is.na(l3$p_drop))
  }

  # Subject and structure deletions are counted separately: only the crossed
  # model has a structure grouping factor, so a pooled counter would be
  # ambiguous (and un-normalizable) for the by-subject model.
  is_subj <- loso$level == "subject"
  is_struct <- loso$level == "structure"
  summ[[paste(predictor, slug, sep = "/")]] <- data.frame(
    predictor = predictor,
    off_type = off_type, metric = metric, response_metric = response_metric,
    model = model_name,
    p0 = p0, sig0 = sig0, slope0 = slope0,
    loco_flips = sum(loco$flip, na.rm = TRUE),
    n_combos = sum(!is.na(loco$p_drop)),
    loso_subject_flips = sum(loso$flip[is_subj], na.rm = TRUE),
    n_subjects = sum(is_subj),
    loso_structure_flips = sum(loso$flip[is_struct], na.rm = TRUE),
    n_structures = sum(is_struct),
    worst_combo_cookD = max(loco$cookD, na.rm = TRUE),
    worst_subject_cookD = max(loso$cookD[loso$level == "subject"], na.rm = TRUE),
    worst_pair = paste(worst2$unit_i, worst2$unit_j, sep = "+"),
    worst_pair_p = worst2$p_drop,
    n_pairs_stay_sig = sum(l2$p_drop < 0.05, na.rm = TRUE),
    n_pairs = sum(!is.na(l2$p_drop)),
    worst_trio = worst3_units, worst_trio_p = worst3_p, n_trios = n_trios,
    top2_subjects = paste(top2, collapse = "+"),
    slope_excl_top2 = ex$slope, p_excl_top2 = ex$p,
    stringsAsFactors = FALSE
  )
  cat(sprintf("done: [%s] %-34s p0=%.4g  loco_flips=%d loso_flips(subj)=%d  worst_pair(%s) p=%.3g\n",
              predictor, slug, p0, sum(loco$flip, na.rm = TRUE),
              sum(loso$flip[is_subj], na.rm = TRUE),
              paste(worst2$unit_i, worst2$unit_j, sep = "+"), worst2$p_drop))
}

tab <- do.call(rbind, summ)
tab <- tab[order(tab$predictor, tab$metric, tab$response_metric, tab$off_type,
                 tab$model), ]
utils::write.csv(tab, file.path(OUT_ROOT, "influence_summary.csv"), row.names = FALSE)

md <- c(
  "# NOD -> NREM.Rebound correlation: influence / leverage diagnostics",
  "",
  "One row per fit. `sig0` = significant at raw x. `*_flips` = how many single",
  "deletions change the significance verdict, out of the matching `n_*` denominator",
  "(`loco_flips`/`n_combos` = subject-structure pairs; `loso_subject_flips`/`n_subjects`;",
  "`loso_structure_flips`/`n_structures`, which is 0 for the by-subject model since it",
  "has no structure grouping factor). `worst_pair_p` = largest LRT p over",
  "all leave-2-subjects-out pairs (masking check). `n_pairs_stay_sig / n_pairs` =",
  "how many subject pairs keep p<0.05 when removed. `slope_excl_top2` / `p_excl_top2`",
  "drop the two highest-Cook's-D subjects (`top2_subjects`).",
  "",
  "| predictor | metric | resp | off | model | p0 | sig0 | loco flips | LOSO subj | LOSO struct | worst subj Cook's D | worst pair (p) | pairs sig | worst trio p | p excl top2 |",
  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)
for (r in seq_len(nrow(tab))) {
  md <- c(md, sprintf("| %s | %s | %s | %s | %s | %.4g | %s | %d/%d | %d/%d | %d/%d | %.3g | %s (%.3g) | %d/%d | %.3g | %.3g |",
    tab$predictor[r], tab$metric[r], tab$response_metric[r], tab$off_type[r], tab$model[r], tab$p0[r],
    ifelse(tab$sig0[r], "**yes**", "no"),
    tab$loco_flips[r], tab$n_combos[r],
    tab$loso_subject_flips[r], tab$n_subjects[r],
    tab$loso_structure_flips[r], tab$n_structures[r],
    tab$worst_subject_cookD[r], tab$worst_pair[r], tab$worst_pair_p[r],
    tab$n_pairs_stay_sig[r], tab$n_pairs[r], tab$worst_trio_p[r], tab$p_excl_top2[r]))
}
writeLines(md, file.path(OUT_ROOT, "influence_summary.md"))

cat("\nWrote per-fit diagnostics under", OUT_ROOT, "\n")
cat("Consolidated: influence_summary.{csv,md}\n")
