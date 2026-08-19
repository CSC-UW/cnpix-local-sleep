# Load and prepare cross-structure "excess globality" summary data.
#
# Companion to [load_locality_per_condition_summary()] for the cross-structure
# "excess globality" test: is an OFF's observed cross-structure overlap degree
# greater than the duration-matched windowed-shift null ("more global than
# chance")? These functions are deliberately self-contained so the
# excess-globality pipeline can be removed without touching the layer-agnostic
# or locality code. The parquet is produced by cnpix_local_sleep's
# `off-analysis export-excess-globality-offs`.

#' Quantity levels compared in the excess-globality test
#'
#' `null` is the factor reference, so an `"observed - null"` contrast and the
#' main effect of `quantity` read as the excess over the windowed-shift null.
#' @export
QUANTITY_LEVELS <- c("null", "observed")

#' @keywords internal
.read_excess_extdata <- function(fname) {
  path <- system.file("extdata", fname, package = "offp")
  assertthat::assert_that(
    nzchar(path),
    msg = paste0(
      "Excess-globality summary file not found in extdata: ", fname,
      ". Run cnpix_local_sleep's `off-analysis export-excess-globality-offs` first."
    )
  )
  arrow::read_parquet(path)
}

#' Load the excess-globality summary
#'
#' Reads `summarized_excess_globality_offs.parquet`: one row per
#' (subject, structure, quantity) with `value` (the cell's mean overlap degree),
#' `count` (scored OFFs), `clade`, `condition == "NREM"`, and provenance columns
#' (`null_scope`, `window`, `n_shuffles`). This is the data behind the *Plot 5
#' statistics* cell of `notebooks/figures/group_cross_structure_offs.ipynb`.
#' @return Data frame with the per-(subject, structure, quantity) summary.
#' @export
load_excess_globality_summary <- function() {
  .read_excess_extdata("summarized_excess_globality_offs.parquet")
}

#' Prepare excess-globality data for the observed-vs-null test
#'
#' Keeps `clade == "Cx"` (when present) and `quantity %in% [QUANTITY_LEVELS]`,
#' and encodes `quantity` as a factor with `null` as the reference level.
#'
#' @param d Data frame from [load_excess_globality_summary()].
#' @return Filtered data frame with `quantity` a 2-level factor.
#' @export
prepare_excess_globality_data <- function(d) {
  if ("clade" %in% names(d)) {
    d <- dplyr::filter(d, .data[["clade"]] == "Cx")
  }
  keep <- as.character(d$quantity) %in% QUANTITY_LEVELS
  d <- d[keep, , drop = FALSE]
  d$quantity <- factor(as.character(d$quantity), levels = QUANTITY_LEVELS)
  d
}
