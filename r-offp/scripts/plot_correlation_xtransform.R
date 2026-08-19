#!/usr/bin/env Rscript
# SVG scatter plots for the NOD -> NREM.Rebound predictor-transform sensitivity.
#
# For every fit, one SVG with three panels (raw x | rank(x) | log(x)), each a
# scatter of the (transformed) predictor vs the NREM.Rebound response with the
# population fixed-effect line + CI ribbon (via modelbased::estimate_expectation,
# the same marginal-effects method the pipeline plot uses). The two tail subjects
# that carry the raw-x leverage (CNPIX4-Doppio, CNPIX7-Giuseppe) are highlighted
# so the reader can see where the tail lands under each transform. Line style
# encodes the slope LRT: solid p<.05, dashed p<.1, dotted otherwise.
#
# Reads _output_correlations/<off>/<metric>/<predictor>/<model>/results.rds;
# writes <spec_dir>/xtransform_sensitivity/comparison.svg alongside the raw,
# rank, and log sensitivity-refit results. One SVG per fit; the fit's existing
# path distinguishes self-rebound from cross-metric analyses. Run from the
# r-offp root.

suppressWarnings(suppressMessages({
  library(offp); library(lme4); library(ggplot2); library(modelbased)
}))

CORR_ROOT <- "_output_correlations"

HIGHLIGHT <- c("CNPIX4-Doppio" = "#d1495b", "CNPIX7-Giuseppe" = "#00798c")
TRANSFORMS <- list(
  raw  = list(fn = function(x) x,       lab = "raw x"),
  rank = list(fn = function(x) rank(x), lab = "rank(x)"),
  log  = list(fn = function(x) log(x),  lab = "log(x)")
)
lty_for <- function(p) if (isTRUE(p < 0.05)) "solid" else if (isTRUE(p < 0.1)) "dashed" else "dotted"

files <- sort(Sys.glob(file.path(CORR_ROOT, "*", "*", "*", "*", "results.rds")))

for (f in files) {
  res <- readRDS(f); d0 <- res$data
  pc <- res$predictor_condition
  d0$subject <- as.character(d0$subject)
  re_terms <- res$model_def$re_terms
  pts <- list(); lines <- list(); ribbons <- list(); labs <- character()
  for (tn in names(TRANSFORMS)) {
    tr <- TRANSFORMS[[tn]]
    d <- d0; d$x <- tr$fn(d0$x)
    ff <- reformulate(c("x", re_terms), response = "y")
    fn <- reformulate(c("1", re_terms), response = "y")
    m <- suppressWarnings(suppressMessages(lmer(ff, d, REML = FALSE)))
    mn <- suppressWarnings(suppressMessages(lmer(fn, d, REML = FALSE)))
    p <- suppressWarnings(anova(m, mn))[["Pr(>Chisq)"]][2]
    pred <- as.data.frame(modelbased::estimate_expectation(m, by = "x", length = 100))
    panel <- sprintf("%s   (p=%.3g)", tr$lab, p)
    d$panel <- panel; pred$panel <- panel
    d$hl <- ifelse(d$subject %in% names(HIGHLIGHT), d$subject, "other")
    pts[[tn]] <- d[, c("panel", "x", "y", "hl")]
    lines[[tn]] <- data.frame(panel = panel, x = pred$x, y = pred$Predicted, lty = lty_for(p))
    ribbons[[tn]] <- data.frame(panel = panel, x = pred$x, lo = pred$CI_low, hi = pred$CI_high)
    labs <- c(labs, panel)
  }
  P <- do.call(rbind, pts); L <- do.call(rbind, lines); R <- do.call(rbind, ribbons)
  ord <- labs  # raw, rank, log order
  P$panel <- factor(P$panel, levels = ord); L$panel <- factor(L$panel, levels = ord)
  R$panel <- factor(R$panel, levels = ord)
  P$hl <- factor(P$hl, levels = c("other", names(HIGHLIGHT)))

  gg <- ggplot() +
    geom_ribbon(data = R, aes(x = x, ymin = lo, ymax = hi), alpha = 0.15) +
    geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey70") +
    geom_line(data = L, aes(x = x, y = y, linetype = lty), linewidth = 0.9) +
    geom_point(data = P[P$hl == "other", ], aes(x = x, y = y),
               colour = "grey55", size = 1.8, alpha = 0.8) +
    geom_point(data = P[P$hl != "other", ], aes(x = x, y = y, colour = hl),
               size = 2.6, alpha = 0.95) +
    scale_colour_manual(values = HIGHLIGHT, name = "tail subject", drop = FALSE) +
    scale_linetype_identity() +
    facet_wrap(~panel, scales = "free_x", nrow = 1) +
    labs(x = paste0(res$metric, " @ ", pc, "  (predictor, transformed)"),
         y = paste0(res$metric, ", NREM.Rebound"),
         title = sprintf("[%s] %s / %s / %s: predictor-transform sensitivity (n=%d, %d subj)",
                         pc, res$metric, res$off_type, res$model_def$name,
                         nrow(d0), length(unique(d0$subject)))) +
    theme_classic(base_size = 11) + theme(legend.position = "bottom")

  out <- file.path(dirname(f), "xtransform_sensitivity", "comparison.svg")
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  grDevices::svg(out, width = 11, height = 4.2); print(gg); grDevices::dev.off()
  cat("wrote", out, "\n")
}
