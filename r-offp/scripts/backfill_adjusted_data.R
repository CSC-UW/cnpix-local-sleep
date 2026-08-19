#!/usr/bin/env Rscript
#
# Backfill adjusted_data.csv for existing results.
#
# Iterates over all results.rds files under _output/, loads each one,
# subtracts random effects from the data, and saves adjusted_data.csv
# alongside the RDS file.
#
# Usage:
#   Rscript scripts/backfill_adjusted_data.R

library(offp)

rds_files <- list.files(
  "_output", pattern = "^results\\.rds$", recursive = TRUE, full.names = TRUE
)

if (length(rds_files) == 0) {
  stop("No results.rds files found under _output/")
}

message("Found ", length(rds_files), " results.rds files")

for (rds_path in rds_files) {
  spec_dir <- dirname(rds_path)
  csv_path <- file.path(spec_dir, "adjusted_data.csv")

  result <- readRDS(rds_path)
  d_adjusted <- subtract_random_effects(result$data, result$models$full)
  utils::write.csv(d_adjusted, csv_path, row.names = FALSE)
  message("Wrote ", csv_path)
}

message("Done. Backfilled ", length(rds_files), " files.")
