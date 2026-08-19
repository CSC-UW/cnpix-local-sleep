#' Build a condition contrast matrix from posthoc strings
#'
#' Creates a [multcomp::mcp] object from a character vector of contrast
#' strings (e.g. from a YAML config). Returns `NULL` if no strings are
#' provided.
#'
#' @param posthoc_strings Character vector of contrast strings, e.g.
#'   `c("Early.REC.NREM - Early.BSL.NREM = 0", ...)`. If `NULL` or
#'   length zero, returns `NULL`.
#' @return A [multcomp::mcp] object, or `NULL`.
#' @export
build_condition_comparisons <- function(posthoc_strings) {
  if (is.null(posthoc_strings) || length(posthoc_strings) == 0) {
    return(NULL)
  }
  multcomp::mcp(condition = posthoc_strings)
}