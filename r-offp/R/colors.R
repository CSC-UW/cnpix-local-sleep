#' Get the condition color palette
#'
#' Returns a named character vector of hex colors for each experimental
#' condition, based on the RColorBrewer "Paired" palette with desaturation
#' applied to every other color.
#'
#' @return Named character vector mapping condition names to hex colors.
#' @export
get_condition_palette <- function() {
  p <- RColorBrewer::brewer.pal(12, "Paired")  # seaborn's default palette
  p_rgb <- col2rgb(p) / 255

  # Brewer "Paired" alternates light/dark; averaging each dark shade toward its
  # lighter partner mutes the pairs enough to read as six conditions, not twelve.
  for (i in seq(2, length(p), by = 2)) {
    new_color <- (p_rgb[, i] + p_rgb[, i - 1]) / 2
    p[i] <- rgb(new_color[1], new_color[2], new_color[3])
  }

  conditions <- c(
    "Early.BSL.NREM",
    "Early.REC.NREM.Match",
    "Early.NOD.Wake",
    "Late.NOD.Wake",
    "Early.REC.NREM",
    "Late.REC.NREM"
  )

  palette <- setNames(p[1:length(conditions)], conditions)

  palette
}
