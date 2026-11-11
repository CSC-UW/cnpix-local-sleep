# Mixed models for the manually annotated wake OFF periods over sleep deprivation.
#
# Numbers for the last paragraph of the Supplementary Methods section "Agreement between
# detectors and manual annotation". Reads offs.parquet and rates.parquet, written by
# annotated_wake_offs.py, and prints each change with its 95% Wald interval and the
# likelihood ratio test against the model without the window.
#
# Rate and cumulative area are modelled per structure on the raw scale and reported as
# a percentage of the first-half mean. Median MUA and area are modelled per OFF period
# on the log scale, in the structures with at least MIN_OFFS OFF periods in each window.
#
# Usage (from r-offp, so that renv loads):
#   Rscript ../scripts/methods/annotated_wake_offs.R [out-dir]

suppressMessages({
  library(lme4)
  library(arrow)
})

MIN_OFFS <- 3
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[1] else "../notebooks/figures/outputs/annotated_wake_offs"
offs <- as.data.frame(read_parquet(file.path(out_dir, "offs.parquet")))
rates <- as.data.frame(read_parquet(file.path(out_dir, "rates.parquet")))
offs$pair <- interaction(offs$subject, offs$probe, offs$structure, drop = TRUE)
rates$pair <- interaction(rates$subject, rates$probe, rates$structure, drop = TRUE)
control <- lmerControl(optimizer = "bobyqa")

report <- function(name, formula, d, percent) {
  fit <- function(f) suppressMessages(lmer(f, data = d, REML = FALSE, control = control))
  full <- fit(formula)
  p <- anova(fit(update(formula, . ~ . - late)), full)$`Pr(>Chisq)`[2]
  b <- fixef(full)[["late"]]
  se <- sqrt(vcov(full)["late", "late"])
  cat(sprintf(
    "%s: %+.1f%% [%.1f, %.1f], p = %.2g (%d structures)\n",
    name, percent(b), percent(b - 1.96 * se), percent(b + 1.96 * se), p, nlevels(droplevels(d$pair))
  ))
}

rates$late <- as.integer(rates$half == "second")
for (y in c("rate", "cumulative_area")) {
  rates$y <- rates[[y]]
  first_mean <- mean(rates$y[rates$late == 0])
  report(
    paste(y, "second vs first half", sep = ", "), y ~ late + (1 | subject) + (1 | pair), rates,
    function(b) 100 * b / first_mean
  )
}

per_off <- function(y, window, first, second) {
  d <- offs[offs[[window]] %in% c(first, second), ]
  d$late <- as.integer(d[[window]] == second)
  n <- table(d$pair, d$late)
  d <- d[d$pair %in% rownames(n)[n[, "0"] >= MIN_OFFS & n[, "1"] >= MIN_OFFS], ]
  d$y <- log(d[[y]])
  report(
    sprintf("%s, %s vs %s %s", y, second, first, window), y ~ late + (1 | subject) + (1 + late | pair), d,
    function(b) 100 * (exp(b) - 1)
  )
}
per_off("median_mua", "half", "first", "second")
per_off("area", "half", "first", "second")
per_off("area", "hour", "first", "last")
