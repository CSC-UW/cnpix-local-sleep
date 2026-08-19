#' Format post-hoc summary
#'
#' @param posthoc Post-hoc test results
#' @return Formatted summary message
#' @importFrom utils capture.output
#' @export
format_posthoc_summary <- function(posthoc) {
  # If posthoc$effect_size is a matrix, it is already in the correct format.
  # Otherwise, it is a list of lists, and we need to extract the f^2 values
  if (is.matrix(posthoc$effect_size)) {
    effect_size <- posthoc$effect_size
  } else {
    effect_size <- list()
    for (lvl in names(posthoc$effect_size)) {
      effect_size[[lvl]] <- posthoc$effect_size[[lvl]]$fsquared
    }
    effect_size <- as.matrix(effect_size)
    colnames(effect_size) <- "Cohen's f^2 analogue"
  }
  xtras <- cbind(effect_size, posthoc$ci$confint)
  msg <- capture.output(
    print(summary(posthoc$glht)),
    cat("Effect sizes and 95% family-wise confidence intervals:\n"),
    print(xtras)
  )
  msg
}
