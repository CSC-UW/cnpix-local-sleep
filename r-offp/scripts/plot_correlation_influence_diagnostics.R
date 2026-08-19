#!/usr/bin/env Rscript
# SVG diagnostic plots for the correlation influence analysis. Reads the CSVs
# written by run_correlation_influence_diagnostics.R (run that first) and draws,
# for the headline rate fits, three plots each:
#
#   A leverage scatter : x vs y, points sized by combo Cook's D, the two tail
#     subjects highlighted, with the full marginal line (solid) and the line refit
#     without the two highest-Cook's-D subjects (dashed) overlaid.
#   B LOSO tornado     : deleted-slope per subject (leave-one-subject-out), sorted,
#     coloured by whether the fit stays significant, baseline slope marked.
#   C leave-2 distribution : sorted LRT p over all leave-two-subjects-out pairs,
#     0.05 line, the worst (masking) pair annotated.
#
# Writes _output_correlations/influence_diagnostics/plots/<predictor>/.
# Additive. Run from the r-offp package root, after
# run_correlation_influence_diagnostics.R.

suppressWarnings(suppressMessages({ library(ggplot2) }))

DIAG_ROOT <- "_output_correlations/influence_diagnostics"
PLOT_DIR <- file.path(DIAG_ROOT, "plots")

# Headline rate fits (each dataset under both the subject and crossed models),
# discovered from the diagnostics tree so every predictor (NOD, NOD.Wake, ...) and
# any future dataset are covered without editing this script. The "*_rate_*" glob
# restricts to the rate fits; the parent dir name is the predictor.
marg_files <- sort(Sys.glob(
  file.path(DIAG_ROOT, "*", "*_rate_*", "marginal_fit.csv")))
stopifnot(length(marg_files) > 0)
HIGHLIGHT <- c("CNPIX4-Doppio" = "#d1495b", "CNPIX7-Giuseppe" = "#00798c")

for (mf_path in marg_files) {
  d <- dirname(mf_path)
  slug <- basename(d)
  pc <- basename(dirname(d))
  out_subdir <- file.path(PLOT_DIR, pc)
  dir.create(out_subdir, recursive = TRUE, showWarnings = FALSE)
  loco <- utils::read.csv(file.path(d, "loco.csv"), stringsAsFactors = FALSE)
  loso <- utils::read.csv(file.path(d, "loso.csv"), stringsAsFactors = FALSE)
  l2   <- utils::read.csv(file.path(d, "leave2.csv"), stringsAsFactors = FALSE)
  mf   <- utils::read.csv(file.path(d, "marginal_fit.csv"), stringsAsFactors = FALSE)

  # -------------------- A: leverage scatter --------------------
  loco$hl <- ifelse(loco$subject %in% names(HIGHLIGHT), loco$subject, "other")
  loco$hl <- factor(loco$hl, levels = c("other", names(HIGHLIGHT)))
  hlrows <- loco[loco$hl != "other", ]
  ggA <- ggplot(loco, aes(x = x, y = y)) +
    geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey70") +
    geom_abline(intercept = mf$intercept, slope = mf$slope,
                linewidth = 0.9, colour = "black") +
    geom_abline(intercept = mf$intercept_excl_top2, slope = mf$slope_excl_top2,
                linewidth = 0.8, colour = "grey45", linetype = "dashed") +
    geom_point(data = loco[loco$hl == "other", ], aes(size = cookD),
               colour = "grey55", alpha = 0.8) +
    geom_point(data = hlrows, aes(size = cookD, colour = hl), alpha = 0.95) +
    geom_text(data = hlrows, aes(label = structure, colour = hl),
              vjust = -0.9, size = 3, show.legend = FALSE) +
    scale_colour_manual(values = HIGHLIGHT, name = "tail subject", drop = FALSE) +
    scale_size_continuous(name = "combo Cook's D", range = c(1.5, 7)) +
    labs(x = sprintf("predictor (raw x, %s)", pc), y = "NREM.Rebound",
         title = sprintf("[%s] %s: leverage scatter", pc, slug),
         subtitle = sprintf(
           "solid: full fit (slope=%.3g, p=%.3g)   dashed: excl. %s (slope=%.3g, p=%.3g)",
           mf$slope, mf$p, mf$top2_subjects, mf$slope_excl_top2, mf$p_excl_top2)) +
    theme_classic(base_size = 11) + theme(legend.position = "right")
  outA <- file.path(out_subdir, sprintf("%s_A_leverage_scatter.svg", slug))
  grDevices::svg(outA, width = 8, height = 5); print(ggA); grDevices::dev.off()

  # -------------------- B: LOSO tornado --------------------
  sj <- loso[loso$level == "subject", ]
  sj$unit <- factor(sj$unit, levels = sj$unit[order(sj$slope_drop)])
  sj$stays_sig <- sj$p_drop < 0.05
  slope0 <- loco$slope_full[1]; p0 <- loco$p_full[1]
  ggB <- ggplot(sj, aes(x = slope_drop, y = unit, colour = stays_sig)) +
    geom_vline(xintercept = 0, linewidth = 0.3, colour = "grey70") +
    geom_vline(xintercept = slope0, linewidth = 0.6, colour = "black", linetype = "dotted") +
    geom_point(size = 3) +
    scale_colour_manual(values = c(`TRUE` = "#2a9d8f", `FALSE` = "#e76f51"),
                        name = "stays p<0.05", drop = FALSE) +
    labs(x = "slope after dropping this subject", y = NULL,
         title = sprintf("[%s] %s: leave-one-subject-out", pc, slug),
         subtitle = sprintf("dotted = baseline slope %.3g (p=%.3g); colour = fit still significant?",
                            slope0, p0)) +
    theme_classic(base_size = 11) + theme(legend.position = "right")
  outB <- file.path(out_subdir, sprintf("%s_B_loso_tornado.svg", slug))
  grDevices::svg(outB, width = 7, height = 5); print(ggB); grDevices::dev.off()

  # -------------------- C: leave-2 p distribution --------------------
  l2 <- l2[!is.na(l2$p_drop), ]
  l2 <- l2[order(l2$p_drop), ]; l2$rank <- seq_len(nrow(l2))
  l2$stays_sig <- l2$p_drop < 0.05
  w <- l2[which.max(l2$p_drop), ]
  ggC <- ggplot(l2, aes(x = rank, y = p_drop, colour = stays_sig)) +
    geom_hline(yintercept = 0.05, linewidth = 0.5, colour = "grey40", linetype = "dashed") +
    geom_point(size = 1.8) +
    annotate("text", x = w$rank, y = w$p_drop,
             label = sprintf("worst pair: %s+%s (p=%.2f)", w$unit_i, w$unit_j, w$p_drop),
             hjust = 1.05, vjust = 0.4, size = 3) +
    scale_colour_manual(values = c(`TRUE` = "#2a9d8f", `FALSE` = "#e76f51"),
                        name = "stays p<0.05", drop = FALSE) +
    labs(x = "subject pair (sorted by p)", y = "LRT p after dropping pair",
         title = sprintf("[%s] %s: leave-two-subjects-out (masking check)", pc, slug),
         subtitle = sprintf("%d of %d pairs keep p<0.05", sum(l2$stays_sig), nrow(l2))) +
    theme_classic(base_size = 11) + theme(legend.position = "right")
  outC <- file.path(out_subdir, sprintf("%s_C_leave2_distribution.svg", slug))
  grDevices::svg(outC, width = 7, height = 5); print(ggC); grDevices::dev.off()

  cat("wrote 3 SVGs for", pc, "/", slug, "\n")
}
