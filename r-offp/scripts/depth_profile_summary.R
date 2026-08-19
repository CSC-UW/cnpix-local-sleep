#!/usr/bin/env Rscript
#
# Laminar-trimodality mechanical-null group analysis.
#
# A small, self-contained companion (NOT part of the offp package pipeline). It
# reads `summarized_depth_profile.parquet` (one tidy row per (subject, probe,
# structure) laminar combo, produced by cnpix_local_sleep's
# `off-analysis export-depth-profile-summary`) and answers the group-level
# question behind `notebooks/figures/laminar_trimodality_null.ipynb`:
#
#   Across structures, once the shape-preserving mechanical null is accounted
#   for, is there a real laminar depth residual, and how big is it?
#
# The inferential unit is the SUBJECT, not the OFF (~10^5 OFFs/structure is
# descriptive scale only; pooling them would be pseudoreplication). For each
# metric we therefore:
#   1. fit an intercept-only random-intercept model  value ~ 1 + (1 | subject)
#      (lme4::lmer) and report the group mean (fixed intercept) with its CI;
#   2. compute a SUBJECT-CLUSTER BOOTSTRAP CI (resample whole subjects with
#      replacement, recompute the across-structure mean), robust to the small
#      number of subjects and to lmer singularity;
#   3. draw a forest plot: every structure's value, the group mean, both CIs,
#      and the reference line (0 = no real structure / no departure from null).
#
# Metrics:
#   w1_com_feasible          earth-mover distance (um) between empirical COM and
#                            the feasible (centroid-contraction) null -> the
#                            robust effect size for the COM depth question
#                            (test: CI above 0). The skill-score attribution is
#                            NOT used for COM: its flat denominator collapses when
#                            empirical COM is near-flat (tall probes), so use this.
#   com_spread_ratio_feasible  std(emp COM)/std(feasible COM) -> DIRECTION of the
#                            COM departure. >1 (CI above 1) = empirical COM reaches
#                            extreme depths MORE than the contraction null predicts.
#   conc_real_feasible       1 - attr_conc_feasible -> concentration "real residual"
#                            (the laminar supra/infra asymmetry the symmetric null
#                            cannot make). Attribution IS well-behaved here (the
#                            concentration support [0,1] keeps emp far from flat).
#   occ_*_w1                 occupancy earth-mover distance vs the depth null (um).
#   occ_*_tv                 occupancy fraction-of-mass-moved vs the depth null.
#
# Usage:
#   Rscript scripts/depth_profile_summary.R [parquet_path] [out_dir]
# Defaults: inst/extdata/summarized_depth_profile.parquet, _output_depth_profile
#
# Delete this script (and `summarized_depth_profile.parquet`) to remove the
# feature; nothing in the offp package depends on it.

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(lme4)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
parquet_path <- if (length(args) >= 1) args[1] else {
  file.path("inst", "extdata", "summarized_depth_profile.parquet")
}
out_dir <- if (length(args) >= 2) args[2] else {
  file.path("_output_depth_profile")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(parquet_path)) {
  stop(
    "Laminar-null summary not found: ", parquet_path, "\n",
    "Run cnpix_local_sleep's `off-analysis export-depth-profile-summary` first."
  )
}

N_BOOT <- 5000
SEED <- 0
set.seed(SEED)

d <- as.data.frame(arrow::read_parquet(parquet_path))
message(sprintf(
  "Loaded %d combos across %d subjects.",
  nrow(d), dplyr::n_distinct(d$subject)
))

# Derive the "real residual" metrics (1 - attribution) alongside the occupancy
# effect sizes. Each named entry is a column on `d` to analyze; `ref` is the
# null reference line and `label` the human-facing axis title.
d <- d |>
  mutate(
    conc_real_feasible = 1 - attr_conc_feasible
  )

metrics <- tibble::tribble(
  ~col,                          ~label,                                     ~ref,
  "w1_com_feasible",             "COM vs feasible null (um, effect size)",    0,
  "com_spread_ratio_feasible",   "COM spread ratio emp/feasible (>1=real)",   1,
  "conc_real_feasible",          "concentration real residual (feasible)",    0,
  # Occupancy effect sizes. The `feasible` (no-clip, size-preserving, in-detection)
  # null is the primary one. The `uniform` null places same-size OFFs over the full
  # ANATOMICAL structure (extends past the detection window) and observes only the
  # detected channels, replacing the feasible null's detection-edge taper with the
  # asymmetric leak-in shape; its out-of-window overhang is divided out by unit-mass
  # renormalization (shape comparison), so both its W1 and TV are deficit-free. Both
  # carried; prefer the `_feasible` rows.
  "occ_count_feasible_w1",       "count-occupancy W1 vs FEASIBLE null (um)",  0,
  "occ_time_feasible_w1",        "time-occupancy W1 vs FEASIBLE null (um)",   0,
  "occ_count_feasible_tv",       "count-occupancy TV vs FEASIBLE null",       0,
  "occ_time_feasible_tv",        "time-occupancy TV vs FEASIBLE null",        0,
  # Signed superficial(+)-vs-deep(-) asymmetry of the empirical occupancy excess
  # (feasible null). asym = fraction of mass; asym_norm = asym/TV in [-1,1].
  "occ_count_feasible_asym",     "count-occ superficial-deep excess (frac)",  0,
  "occ_time_feasible_asym",      "time-occ superficial-deep excess (frac)",   0,
  "occ_count_feasible_asym_norm","count-occ superficial-deep excess / TV",    0,
  "occ_time_feasible_asym_norm", "time-occ superficial-deep excess / TV",     0,
  "occ_count_w1",                "count-occupancy W1 vs whole-structure null (um)", 0,
  "occ_time_w1",                 "time-occupancy W1 vs whole-structure null (um)",  0,
  "occ_count_tv",                "count-occupancy TV vs whole-structure null",      0,
  "occ_time_tv",                 "time-occupancy TV vs whole-structure null",       0
)

# intercept-only random-intercept fit + subject-cluster bootstrap
subject_cluster_boot <- function(df, col, n = N_BOOT) {
  subs <- unique(df$subject)
  by_sub <- split(df[[col]], df$subject)
  means <- vapply(seq_len(n), function(.i) {
    drawn <- sample(subs, length(subs), replace = TRUE)
    # Pool the drawn subjects' structure-level values, then take the grand mean.
    mean(unlist(by_sub[drawn]), na.rm = TRUE)
  }, numeric(1))
  stats::quantile(means, c(0.025, 0.975), na.rm = TRUE, names = FALSE)
}

# Fit the intercept-only random-intercept model and KEEP the fitted object (so it
# can be persisted to disk and used for diagnostics), alongside the summary row.
fit_metric <- function(df, col) {
  df <- df[is.finite(df[[col]]), , drop = FALSE]
  form <- stats::as.formula(paste0(col, " ~ 1 + (1 | subject)"))
  fit <- suppressWarnings(suppressMessages(
    lme4::lmer(form, data = df, REML = TRUE,
               control = lme4::lmerControl(
                 check.conv.singular = "ignore"))
  ))
  intercept <- as.numeric(lme4::fixef(fit)[["(Intercept)"]])
  ci <- tryCatch(
    suppressWarnings(suppressMessages(
      stats::confint(fit, parm = "(Intercept)", method = "Wald")
    )),
    error = function(e) matrix(c(NA_real_, NA_real_), nrow = 1)
  )
  boot <- subject_cluster_boot(df, col)
  row <- data.frame(
    col = col,
    mean = intercept,
    lmer_lo = ci[1, 1], lmer_hi = ci[1, 2],
    boot_lo = boot[1], boot_hi = boot[2],
    singular = lme4::isSingular(fit),
    n_combos = nrow(df),
    n_subjects = dplyr::n_distinct(df$subject)
  )
  list(row = row, fit = fit)
}

fits <- lapply(metrics$col, function(c) fit_metric(d, c))
names(fits) <- metrics$col
models <- lapply(fits, `[[`, "fit")
summ <- do.call(rbind, lapply(fits, `[[`, "row"))
summ <- dplyr::left_join(metrics, summ, by = "col")

# Persist the fitted model objects so a later session (or the Python notebook's
# provenance pointer) can reload the exact models behind the group means/CIs.
saveRDS(
  list(models = models, summary = summ, metrics = metrics,
       n_boot = N_BOOT, seed = SEED, parquet = normalizePath(parquet_path)),
  file.path(out_dir, "depth_profile_models.rds")
)

# Human-readable per-metric model summaries (lme4::summary + the CIs).
summary_txt <- file.path(out_dir, "depth_profile_model_summaries.txt")
con <- file(summary_txt, open = "wt")
writeLines(c(
  "Laminar-trimodality mechanical-null group models",
  sprintf("Source parquet: %s", normalizePath(parquet_path)),
  sprintf("Fitted: metric ~ 1 + (1 | subject), REML; N_BOOT=%d, seed=%d", N_BOOT, SEED),
  ""
), con)
for (m in metrics$col) {
  r <- summ[summ$col == m, ]
  writeLines(c(
    strrep("=", 78),
    sprintf("%s   [%s]", m, r$label),
    sprintf("group mean = %.4f   lmer Wald 95%% = [%.4f, %.4f]   boot 95%% = [%.4f, %.4f]",
            r$mean, r$lmer_lo, r$lmer_hi, r$boot_lo, r$boot_hi),
    sprintf("ref(null) = %g   singular RE = %s   n_combos = %d   n_subjects = %d",
            r$ref, r$singular, r$n_combos, r$n_subjects),
    ""
  ), con)
  capture.output(print(summary(models[[m]])), file = con)
  writeLines("", con)
}
close(con)

# Diagnostic plots: residuals-vs-fitted and a residual QQ per metric (one page
# each), so model adequacy can be eyeballed.
diag_pdf <- file.path(out_dir, "depth_profile_diagnostics.pdf")
grDevices::pdf(diag_pdf, width = 9, height = 4.5)
for (m in metrics$col) {
  fit <- models[[m]]
  graphics::par(mfrow = c(1, 2), oma = c(0, 0, 2, 0))
  rf <- stats::fitted(fit); rr <- stats::resid(fit)
  plot(rf, rr, xlab = "fitted", ylab = "residual",
       main = "residuals vs fitted", pch = 19, col = "grey30")
  graphics::abline(h = 0, lty = 2, col = "red")
  stats::qqnorm(rr, main = "residual Q-Q", pch = 19, col = "grey30")
  stats::qqline(rr, col = "red")
  graphics::mtext(sprintf("%s  [%s]", m, summ$label[summ$col == m]),
                  outer = TRUE, cex = 1.1)
}
grDevices::dev.off()

message("\n=== Group-level summary (per metric) ===")
for (i in seq_len(nrow(summ))) {
  r <- summ[i, ]
  message(sprintf(
    "%-34s mean=%8.3f  lmer95%%=[%7.3f, %7.3f]  boot95%%=[%7.3f, %7.3f]%s",
    r$label, r$mean, r$lmer_lo, r$lmer_hi, r$boot_lo, r$boot_hi,
    if (isTRUE(r$singular)) "  (singular RE)" else ""
  ))
}

write.csv(summ, file.path(out_dir, "depth_profile_group_summary.csv"),
          row.names = FALSE)

# -------------------- forest plots --------------------
# One facet per metric: each structure's value (point), the group mean (diamond)
# with the lmer Wald CI (thick) and subject-cluster bootstrap CI (thin), and the
# reference line at 0.
long <- do.call(rbind, lapply(metrics$col, function(c) {
  data.frame(
    subject = d$subject, probe = d$probe, structure = d$structure,
    col = c, value = d[[c]], stringsAsFactors = FALSE
  )
})) |>
  left_join(metrics, by = "col") |>
  mutate(combo = paste(subject, structure, sep = " / "))

mean_layer <- summ |>
  mutate(combo = "GROUP MEAN")

forest <- ggplot(long, aes(x = value, y = combo)) +
  geom_vline(aes(xintercept = ref), linetype = "dashed", color = "grey40") +
  geom_point(aes(color = subject), size = 2, alpha = 0.8) +
  # bootstrap CI (thin) then lmer CI (thick) for the group mean
  geom_segment(
    data = mean_layer,
    aes(x = boot_lo, xend = boot_hi, y = "GROUP MEAN", yend = "GROUP MEAN"),
    inherit.aes = FALSE, linewidth = 0.5
  ) +
  geom_segment(
    data = mean_layer,
    aes(x = lmer_lo, xend = lmer_hi, y = "GROUP MEAN", yend = "GROUP MEAN"),
    inherit.aes = FALSE, linewidth = 1.4
  ) +
  geom_point(
    data = mean_layer, aes(x = mean, y = "GROUP MEAN"),
    inherit.aes = FALSE, shape = 18, size = 3.5
  ) +
  facet_wrap(~label, scales = "free_x") +
  labs(
    x = "value", y = NULL,
    title = "Laminar-trimodality mechanical-null residuals",
    subtitle = paste(
      "Per-structure points; group mean (diamond) with lmer Wald CI (thick)",
      "and subject-cluster bootstrap CI (thin). Dashed = null reference (0)."
    )
  ) +
  theme_bw(base_size = 10) +
  theme(legend.position = "bottom")

ggsave(file.path(out_dir, "depth_profile_forest.pdf"), forest,
       width = 15, height = 12)
ggsave(file.path(out_dir, "depth_profile_forest.png"), forest,
       width = 15, height = 12, dpi = 150)

message(sprintf("\nWrote to %s/:", out_dir))
message("  depth_profile_group_summary.csv     group means + CIs per metric")
message("  depth_profile_models.rds            fitted lme4 model objects + summary")
message("  depth_profile_model_summaries.txt   human-readable per-metric summaries")
message("  depth_profile_diagnostics.pdf       residual-vs-fitted + QQ per metric")
message("  depth_profile_forest.{pdf,png}      R forest plots (Python notebook draws its own)")
