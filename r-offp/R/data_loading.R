# Functions to load various datasets from the package extdata directory.

#' Valid OFF types for analysis
#'
#' `"llas_exclusive"` is the adjacent-partition complement `llas & ~clas` (OFFs
#' admitted by the LLAS filter but rejected by the CLAS filter), exported by
#' `off-analysis export-full48h-exclusive-offs`.
#'
#' `"blas"` is not reported in the manuscript, but is kept wired up: review may
#' ask to see it.
#'
#' @export
OFF_TYPES <- c("llas", "clas", "blas", "llas_exclusive")

#' Load OFFs summary data
#'
#' Reads `summarized_full48h_<off_type>_offs.parquet` from `inst/extdata`:
#' OFFs detected across the whole recording with state-aware thresholds and
#' subset to the six statistical conditions, exported by cnpix_local_sleep's
#' `off-analysis export-full48h-offs`.
#'
#' @param off_type Type of OFFs to load: one of [OFF_TYPES].
#' @return Data frame with OFFs summary
#' @export
load_offs_summary <- function(off_type = "llas") {
  assertthat::assert_that(off_type %in% OFF_TYPES)
  fname <- paste0("summarized_full48h_", off_type, "_offs.parquet")
  path <- system.file("extdata", fname, package = "offp")
  assertthat::assert_that(
    nzchar(path),
    msg = paste0(
      "Summary file not found in extdata: ", fname,
      " (off_type='", off_type, "'). ",
      "Run cnpix_local_sleep's `off-analysis export-full48h-offs` first."
    )
  )
  d <- arrow::read_parquet(path)
  d <- join_adjusted_edge_statistics(d, off_type)
  d
}

#' Restrict a summary frame to cortical, layer-agnostic, spatial OFFs
#'
#' The analysis frame is always the cortical (`clade == "Cx"`), layer-agnostic,
#' spatially-detected subset. `layer` and `detection_mode` are WNE *path*
#' components rather than summary columns: the cnpix_local_sleep summarizer groups by
#' `subject`/`probe`/`structure`/`condition` only, so current exports omit both
#' columns and every row is `None`/`spatial` by construction. Older exports still
#' carry them, so each filter applies only when its column is present.
#'
#' Single point of truth for that restriction, shared by
#' [run_cx_homeostasis_analysis()] and the test suite.
#'
#' @param d Summary data frame, e.g. from [load_offs_summary()].
#' @return `d`, restricted to the cortical layer-agnostic spatial rows.
#' @keywords internal
filter_layer_agnostic_cx <- function(d) {
  d <- dplyr::filter(d, .data[["clade"]] == "Cx")
  if ("layer" %in% names(d)) {
    d <- dplyr::filter(d, .data[["layer"]] == "None")
  }
  if ("detection_mode" %in% names(d)) {
    d <- dplyr::filter(d, .data[["detection_mode"]] == "spatial")
  }
  d
}

#' Join size-adjusted onset/offset MAD columns, when they have been exported
#'
#' Onset/offset MAD depend on how many channels an OFF period spans for reasons
#' that are mechanical rather than neural (the detector's spatial structuring
#' element forces `MAD == 0` below a 120 um span, and the estimator keeps
#' climbing with span above it), so a condition effect on the raw statistic
#' partly reflects a condition effect on OFF *size*. cnpix_local_sleep's
#' `off-analysis export-adjusted-edge-statistics` re-estimates each cell mean by
#' marginal standardization (the g-formula) and writes
#' `summarized_full48h_<off_type>_edge_adjusted.parquet`.
#'
#' This helper left-joins the `adj_*` columns onto the summary frame when that
#' file is present, and is a no-op when it is not, so nothing breaks if the
#' companion export has not been run.
#'
#' @param d Summary data frame from [load_offs_summary].
#' @param off_type One of [OFF_TYPES].
#' @return `d`, with `adj_*` columns appended where available.
#' @keywords internal
join_adjusted_edge_statistics <- function(d, off_type) {
  fname <- paste0("summarized_full48h_", off_type, "_edge_adjusted.parquet")
  path <- system.file("extdata", fname, package = "offp")
  if (!nzchar(path)) {
    return(d)
  }
  adjusted <- arrow::read_parquet(path)
  keys <- c("subject", "probe", "structure", "condition")
  # The unadjusted mean_* columns are already present in `d`; keep only the
  # adjusted ones plus the bookkeeping, so the join cannot silently overwrite.
  keep <- c(keys, grep("^adj_|_interaction$|^n_events$", names(adjusted),
                       value = TRUE))
  dplyr::left_join(d, adjusted[, keep, drop = FALSE], by = keys)
}

#' Load per-condition band-power condition means
#'
#' Reads `summarized_full48h_bandpower_offs.parquet` from `inst/extdata`,
#' exported by cnpix_local_sleep's `off-analysis export-bandpower-condition-means`. One row
#' per `(subject, probe, structure, condition)`, carrying the mean z-scored log10
#' instantaneous bipolar band-power columns (`mean_zlog_delta`, `mean_zlog_eta`,
#' `mean_log_delta`, ...) plus struct metadata (`clade`, `AP.Coord`,
#' `Cx.AP.group`). Schema-compatible with the summarized OFF parquets, so it
#' feeds the same [run_cx_homeostasis_analysis()] condition-homeostasis model.
#'
#' @return Data frame of band-power condition means.
#' @export
load_bandpower_condition_means <- function() {
  fname <- "summarized_full48h_bandpower_offs.parquet"
  path <- system.file("extdata", fname, package = "offp")
  assertthat::assert_that(
    nzchar(path),
    msg = paste0(
      "Band-power summary not found in extdata: ", fname, ". ",
      "Run cnpix_local_sleep's `off-analysis export-bandpower-condition-means` first."
    )
  )
  arrow::read_parquet(path)
}