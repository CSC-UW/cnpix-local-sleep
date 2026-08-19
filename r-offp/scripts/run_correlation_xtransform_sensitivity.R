#!/usr/bin/env Rscript
# Leverage-robust sensitivity refits for the NOD -> NREM.Rebound correlations.
#
# Motivation: the raw-x fits are identified by a sparse right tail of the NOD
# predictor (a few high-rate subjects), so their slopes are leverage-sensitive
# (see docs/reports on the correlation influence diagnostics). This script refits
# each existing fit with the predictor replaced by rank(x) (fully removes tail
# leverage; a mixed-model analogue of Spearman) and log(x) (compresses the tail),
# WITHOUT modifying or replacing any existing output. It reuses the pipeline's own
# model + effect-size machinery (offp::fit_correlation_models /
# test_correlation_slope), so the LRT and Cohen's f^2 are computed identically.
#
# It reads each _output_correlations/<off>/<metric>/<predictor>/<model>/results.rds
# (for the built frame `d`, re_terms, and predictor) and writes, per fit:
#   <spec_dir>/xtransform_sensitivity/<raw|rank|log>/results.json + summary.txt
# plus a consolidated comparison table at
#   _output_correlations/xtransform_sensitivity_summary.{csv,md}
# All predictors (NOD, NOD.Wake, ...) are covered in one sweep; the consolidated
# table carries a `predictor` column and the per-fit outputs land under each fit's
# own predictor-scoped spec_dir, so predictors never collide.
#
# NB: slope magnitudes are NOT comparable across transforms (slope is per-unit-x,
# per-rank, and per-log-unit respectively). The leverage-robustness verdict rests
# on the p-value, the sign of the slope, and Cohen's f^2 -- not the slope value.

suppressWarnings(suppressMessages({
  library(offp)
  library(arrow)
}))

CORR_ROOT <- "_output_correlations"
TRANSFORMS <- list(
  raw  = list(fn = function(x) x,         label = "raw x (reference refit)"),
  rank = list(fn = function(x) rank(x),   label = "rank(x) -- tail leverage removed"),
  log  = list(fn = function(x) log(x),    label = "log(x) -- tail compressed")
)

files <- sort(Sys.glob(file.path(CORR_ROOT, "*", "*", "*", "*", "results.rds")))
stopifnot(length(files) > 0)

rows <- list()

for (f in files) {
  res <- readRDS(f)
  spec_dir <- dirname(f)
  d0 <- res$data
  re_terms <- res$model_def$re_terms
  # Response metric distinguishes self-rebound from cross-metric fits that share
  # the same predictor metric (e.g. total_area_norm vs total_area_norm ->
  # mean_zlog_delta). NULL for results.rds predating the cross-metric feature.
  response_metric <- if (is.null(res$response_metric)) res$metric else res$response_metric
  stopifnot(all(d0$x > 0))  # log safety; every fit currently satisfies this

  for (tname in names(TRANSFORMS)) {
    tr <- TRANSFORMS[[tname]]
    d <- d0
    d$x <- tr$fn(d0$x)

    models <- offp::fit_correlation_models(d, re_terms)
    test <- offp::test_correlation_slope(models, d)

    out_dir <- file.path(spec_dir, "xtransform_sensitivity", tname)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    summary_json <- list(
      metric = res$metric,
      response_metric = response_metric,
      off_type = res$off_type,
      model = res$model_def$name,
      predictor_transform = tname,
      predictor_transform_label = tr$label,
      predictor_condition = res$predictor_condition,
      response = res$response,
      re_terms = I(re_terms),
      slope = test$slope,
      ci_lower = test$ci_lower,
      ci_upper = test$ci_upper,
      pval = test$pval,
      significant = test$significant,
      cohens_f2 = test$fsquared,
      n = test$n,
      n_subjects = test$n_subjects,
      note = paste(
        "Sensitivity refit; does not replace the pipeline's raw-x result.",
        "Slope is in transformed-x units and NOT comparable across transforms."
      )
    )
    writeLines(
      jsonlite::toJSON(summary_json, auto_unbox = TRUE, pretty = TRUE, digits = NA),
      file.path(out_dir, "results.json")
    )
    writeLines(c(
      paste0("NOD -> NREM.Rebound correlation, predictor transform: ", tname),
      paste0("  (", tr$label, ")"),
      paste("predictor metric:", res$metric, "| response metric:", response_metric,
            "| off_type:", res$off_type, "| model:", res$model_def$name),
      paste("RE terms:", paste(re_terms, collapse = ", ")),
      "",
      paste("slope (transformed-x units):", format(test$slope, digits = 4)),
      paste0("95% CI: [", format(test$ci_lower, digits = 4), ", ",
             format(test$ci_upper, digits = 4), "]"),
      paste("p-value (LRT):", format(test$pval, digits = 4)),
      paste("Significant:", test$significant),
      paste("Cohen's f^2:", format(round(test$fsquared, 4), nsmall = 4)),
      paste0("n = ", test$n, " combos, ", test$n_subjects, " subjects")
    ), file.path(out_dir, "summary.txt"))

    rows[[length(rows) + 1]] <- data.frame(
      predictor = res$predictor_condition,
      off_type = res$off_type, metric = res$metric,
      response_metric = response_metric, model = res$model_def$name,
      transform = tname, slope = test$slope, ci_lower = test$ci_lower,
      ci_upper = test$ci_upper, pval = test$pval, significant = test$significant,
      cohens_f2 = test$fsquared, n = test$n, n_subjects = test$n_subjects,
      stringsAsFactors = FALSE
    )
  }
}

tab <- do.call(rbind, rows)
tab <- tab[order(tab$predictor, tab$metric, tab$response_metric, tab$off_type,
                 tab$model, match(tab$transform, names(TRANSFORMS))), ]

csv_path <- file.path(CORR_ROOT, "xtransform_sensitivity_summary.csv")
utils::write.csv(tab, csv_path, row.names = FALSE)

# Markdown: one block per fit, raw/rank/log rows, so leverage-robustness is legible.
md <- c("# NOD -> NREM.Rebound correlation: predictor-transform sensitivity",
        "",
        paste0("Leverage-robust refits (rank/log of the NOD predictor) alongside ",
               "a raw-x reference refit. Slope units differ per transform; read ",
               "the **p-value, slope sign, and f^2**, not the slope magnitude."),
        "")
fits <- unique(tab[, c("predictor", "metric", "off_type", "model")])
fits <- fits[order(fits$predictor, fits$metric, fits$off_type, fits$model), ]
for (i in seq_len(nrow(fits))) {
  sub <- tab[tab$predictor == fits$predictor[i] & tab$metric == fits$metric[i] &
             tab$off_type == fits$off_type[i] & tab$model == fits$model[i], ]
  md <- c(md, sprintf("## [%s] %s / %s / %s  (n=%d, %d subj)",
                      fits$predictor[i], fits$metric[i], fits$off_type[i],
                      fits$model[i], sub$n[1], sub$n_subjects[1]),
          "",
          "| transform | slope | p | sig | f^2 |",
          "|---|---|---|---|---|")
  for (r in seq_len(nrow(sub))) {
    md <- c(md, sprintf("| %s | %.4g | %.4g | %s | %.3g |",
                        sub$transform[r], sub$slope[r], sub$pval[r],
                        ifelse(sub$significant[r], "**yes**", "no"),
                        sub$cohens_f2[r]))
  }
  md <- c(md, "")
}
md_path <- file.path(CORR_ROOT, "xtransform_sensitivity_summary.md")
writeLines(md, md_path)

cat("Wrote per-fit sensitivity outputs under */xtransform_sensitivity/{raw,rank,log}/\n")
cat("Consolidated:\n  ", csv_path, "\n  ", md_path, "\n\n")
print(tab, row.names = FALSE, digits = 4)
