# Load and prepare locality (Local vs Overlapping) OFF summary data.
#
# Companion to [load_offs_summary()] for the cross-structure "locality"
# analyses: whether an OFF in one cortical structure temporally overlaps OFFs
# in other structures (`Local` = overlaps none; `Overlapping` = overlaps >= 1).
# These functions are deliberately self-contained so the locality pipeline can
# be removed without touching the layer-agnostic code. The parquets are
# produced by cnpix_local_sleep's `off-analysis export-locality-offs`.

#' Overlap-status levels compared in Local-vs-Overlapping analyses
#'
#' `Local` is the factor reference, so an `"Overlapping - Local"` contrast and the
#' interaction coefficients (`condition<X>:overlap_statusOverlapping`) read
#' relative to Local.
#' @export
OVERLAP_STATUS_LEVELS <- c("Local", "Overlapping")

#' @keywords internal
.read_locality_extdata <- function(fname) {
  path <- system.file("extdata", fname, package = "offp")
  assertthat::assert_that(
    nzchar(path),
    msg = paste0(
      "Locality summary file not found in extdata: ", fname,
      ". Run cnpix_local_sleep's `off-analysis export-locality-offs` first."
    )
  )
  arrow::read_parquet(path)
}

#' Load the overlap-degree summary (request 1)
#'
#' Reads `summarized_locality_overlap_offs.parquet`: one row per
#' (subject, structure, condition) with `mean_overlap_degree` (and the other
#' per-condition overlap fractions). This is the data behind the right subplot of
#' `cross_structure_3_condition_comparison.svg`.
#' @return Data frame with the per-(subject, structure, condition) overlap stats.
#' @export
load_locality_overlap_summary <- function() {
  .read_locality_extdata("summarized_locality_overlap_offs.parquet")
}

#' Load the per-condition Local-vs-Overlapping measure summary (request 2)
#'
#' Reads `summarized_locality_per_condition_<off_type>_offs.parquet`: one row per
#' (subject, probe, structure, condition, overlap_status) with per-group medians
#' of `median_duration` / `median_span` / `median_area`. This is the data behind
#' `cross_structure_4b_local_vs_overlapping_all_conditions.svg`.
#'
#' @param off_type Only `"llas"` is exported (matches the notebook figures).
#' @return Data frame with the per-condition by-overlap_status summarized metrics.
#' @export
load_locality_per_condition_summary <- function(off_type = "llas") {
  assertthat::assert_that(
    off_type == "llas",
    msg = "Only the LLAS locality dataset is exported."
  )
  .read_locality_extdata(
    paste0("summarized_locality_per_condition_", off_type, "_offs.parquet")
  )
}

#' Load the whole-recording NREM/Wake Local-vs-Overlapping summary (request 3)
#'
#' Reads `summarized_locality_full48h_<off_type>_offs.parquet`: one row per
#' (subject, probe, structure, state, overlap_status) with `state in {NREM, Wake}`
#' and the same per-group medians, computed on the whole-recording (not
#' condition-subset) OFFs. The `condition` column equals `state` so the runner's
#' `conditions=` filter works unchanged.
#'
#' @param off_type Only `"llas"` is exported.
#' @return Data frame with the whole-recording by-state, by-overlap_status metrics.
#' @export
load_locality_full48h_summary <- function(off_type = "llas") {
  assertthat::assert_that(
    off_type == "llas",
    msg = "Only the LLAS locality dataset is exported."
  )
  .read_locality_extdata(
    paste0("summarized_locality_full48h_", off_type, "_offs.parquet")
  )
}

#' Prepare locality OFF data for Local-vs-Overlapping modeling
#'
#' Keeps `clade == "Cx"` (when present) and `overlap_status %in%
#' [OVERLAP_STATUS_LEVELS]`, and encodes `overlap_status` as a factor with `Local`
#' as the reference level.
#'
#' @param d Data frame from one of the `load_locality_*_summary()` loaders.
#' @return Filtered data frame with `overlap_status` a 2-level factor.
#' @export
prepare_locality_data <- function(d) {
  if ("clade" %in% names(d)) {
    d <- dplyr::filter(d, .data[["clade"]] == "Cx")
  }
  keep <- as.character(d$overlap_status) %in% OVERLAP_STATUS_LEVELS
  d <- d[keep, , drop = FALSE]
  d$overlap_status <- factor(
    as.character(d$overlap_status),
    levels = OVERLAP_STATUS_LEVELS
  )
  d
}
