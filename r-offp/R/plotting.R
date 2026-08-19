#' Plot residuals vs fitted values
#'
#' Create a ggplot2 scatter plot of model residuals against fitted values,
#' with points colored by a grouping variable.
#'
#' @param model A fitted model object (e.g., from lme4::lmer)
#' @param group_var A vector indicating the group membership for each observation
#' @param palette A named vector of colors for the groups
#' @param title Plot title (default: "Residuals vs Fitted")
#' @param weighted If `TRUE`, plot weighted residuals
#'   (`residuals * sqrt(weights)`). Default: `FALSE`.
#'
#' @return A ggplot2 object
#' @export
plot_rvf <- function(model, group_var, palette,
                     title = "Residuals vs Fitted",
                     weighted = FALSE) {
  resids <- residuals(model)
  if (weighted) {
    resids <- resids * sqrt(weights(model))
    if (title == "Residuals vs Fitted") {
      title <- "Weighted Residuals vs Fitted"
    }
  }
  resid_df <- data.frame(
    fitted = fitted(model),
    residuals = resids,
    group = group_var
  )
  ggplot2::ggplot(
    resid_df,
    ggplot2::aes(x = fitted, y = residuals, color = group)
  ) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
    ggplot2::scale_color_manual(values = palette) +
    ggplot2::theme_minimal() +
    ggplot2::labs(x = "Fitted values", y = "Residuals", title = title)
}

#' Plot Q-Q plot with reference line
#'
#' Create a ggplot2 quantile-quantile plot of model residuals against theoretical
#' normal quantiles, with a reference line and points colored by a grouping variable.
#'
#' @param model A fitted model object (e.g., from lme4::lmer)
#' @param group_var A vector indicating the group membership for each observation
#' @param palette A named vector of colors for the groups
#' @param title Plot title (default: "Normal Q-Q Plot")
#' @param weighted If `TRUE`, use weighted residuals
#'   (`residuals * sqrt(weights)`). Default: `FALSE`.
#'
#' @return A ggplot2 object
#' @export
plot_qqline <- function(model, group_var, palette,
                        title = "Normal Q-Q Plot",
                        weighted = FALSE) {
  resids <- residuals(model)
  if (weighted) {
    resids <- resids * sqrt(weights(model))
    if (title == "Normal Q-Q Plot") {
      title <- "Weighted Normal Q-Q Plot"
    }
  }
  qq_df <- data.frame(
    sample = resids,
    group = group_var
  )
  qq_df <- qq_df[order(qq_df$sample), ]
  qq_df$theoretical <- stats::qnorm(stats::ppoints(nrow(qq_df)))

  # Calculate reference line parameters (same method as qqline)
  qq_params <- as.list(stats::qqnorm(resids, plot.it = FALSE))
  slope <- (stats::quantile(qq_params$y, 0.75) -
    stats::quantile(qq_params$y, 0.25)) /
    (stats::quantile(qq_params$x, 0.75) - stats::quantile(qq_params$x, 0.25))
  intercept <- stats::quantile(qq_params$y, 0.25) -
    slope * stats::quantile(qq_params$x, 0.25)

  ggplot2::ggplot(
    qq_df,
    ggplot2::aes(x = theoretical, y = sample, color = group)
  ) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::geom_abline(
      intercept = intercept,
      slope = slope,
      linetype = "dashed"
    ) +
    ggplot2::scale_color_manual(values = palette) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      x = "Theoretical Quantiles",
      y = "Sample Quantiles",
      title = title
    )
}

#' Map from data names to human-readable display labels
#' @export
display_label_map <- c(
  "median_duration" = "Median duration (s)",
  "mean_boxcox_duration" = "Mean Box-Coxed duration (s^lambda)",
  "median_median_duration" = "MoM duration (s)",
  "mean_median_duration" = "Mean MoM duration (s)",
  "mean_boxcox_median_duration" = "Mean Box-Coxed median duration (s^lambda)",
  "median_span" = "Median span (um)",
  "mean_span" = "Mean span (um)",
  "mean_boxcox_span" = "Mean Box-Coxed span (um^lambda)",
  "median_span_rel2max" = "Median relative span (frac)",
  "mean_boxcox_span_rel2max" = "Mean Box-Coxed relative span (frac^lambda)",
  "median_area" = "Median area (s*um)",
  "mean_area" = "Mean area (s*um)",
  "mean_boxcox_area" = "Mean Box-Coxed area ((s*um)^lambda)",
  "median_area_rel2span" = "Median relative area (s*frac)",
  "mean_boxcox_area_rel2span" = "Mean Box-Coxed relative area ((s*frac)^lambda)",
  "mean_grouped_boxcox_duration" = "Mean grouped Box-Coxed duration (s^lambda)",
  "mean_grouped_boxcox_median_duration" = "Mean grouped Box-Coxed median duration (s^lambda)",
  "mean_grouped_boxcox_span" = "Mean grouped Box-Coxed span (um^lambda)",
  "mean_grouped_boxcox_span_rel2max" = "Mean grouped Box-Coxed relative span (frac^lambda)",
  "mean_grouped_boxcox_area" = "Mean grouped Box-Coxed area ((s*um)^lambda)",
  "mean_grouped_boxcox_area_rel2span" = "Mean grouped Box-Coxed relative area ((s*frac)^lambda)",
  "total_area" = "Total area (s*um)",
  "total_area_rel2span" = "Total relative area (s*frac)",
  "rate" = "Rate (Hz)",
  "total_area_norm" = "Total area normalized (au)",
  "mean_median_trace" = "Mean MoM trace (uV)",
  "mean_mad_trace" = "Mean MAD trace (uV)"
)

#' Look up a display label for a data name
#'
#' @param name A string to look up in `display_label_map`.
#' @return The display label if found, otherwise `name` unchanged.
#' @export
get_display_label <- function(name) {
  label <- display_label_map[name]
  if (is.na(label)) name else unname(label)
}

#' Display label reflecting an optional response transform
#'
#' Wraps [get_display_label()] to reflect a response transform on plot axes,
#' e.g. `"log(Rate (Hz))"`. Returns the base label unchanged for `"identity"`
#' or `NULL`.
#'
#' @param name Response variable name.
#' @param transform Transform name (one of `names(RESPONSE_TRANSFORMS)`), or
#'   `NULL`. Default `"identity"`.
#' @return Display label string.
#' @export
transformed_display_label <- function(name, transform = "identity") {
  base <- get_display_label(name)
  if (is.null(transform) || transform == "identity") {
    return(base)
  }
  paste0(transform, "(", base, ")")
}

#' Plot distributions by condition
#'
#' Create a ggplot2 distribution plot (violin or boxplot) with jittered points,
#' showing the distribution of a response variable by condition.
#'
#' @param data A data frame containing the data to plot.
#' @param condition_var Name of the column containing condition groups (unquoted).
#' @param response_var Name of the column containing the response variable (string).
#' @param palette A named vector of colors for the conditions.
#' @param geom Type of distribution geom: `"violin"` (default) or `"boxplot"`.
#' @param conditions Character vector of conditions to include, or `NULL` for all.
#' @param xlabel X-axis label (default: "Condition").
#' @param ylabel Y-axis label. If `NULL`, looked up via `get_display_label()`.
#'
#' @return A ggplot2 object
#' @export
plot_distributions_by_condition <- function(
  data,
  condition_var,
  response_var,
  palette,
  geom = c("violin", "boxplot"),
  conditions = NULL,
  xlabel = "Condition",
  ylabel = NULL
) {
  geom <- match.arg(geom)
  if (is.null(ylabel)) {
    ylabel <- get_display_label(response_var)
  }

  if (!is.null(conditions)) {
    condition_col <- rlang::as_name(rlang::enquo(condition_var))
    data <- data[data[[condition_col]] %in% conditions, ]
    data[[condition_col]] <- droplevels(data[[condition_col]])
    palette <- palette[names(palette) %in% conditions]
  }

  p <- ggplot2::ggplot(
    data,
    ggplot2::aes(
      x = {{ condition_var }},
      y = .data[[response_var]],
      fill = {{ condition_var }}
    )
  )

  if (geom == "violin") {
    p <- p +
      ggplot2::geom_violin(scale = "width", draw_quantiles = c(0.25, 0.5, 0.75))
  } else if (geom == "boxplot") {
    p <- p + ggplot2::geom_boxplot()
  }

  p +
    ggplot2::geom_jitter(width = 0.15, alpha = 0.4, size = 2) +
    ggplot2::scale_fill_manual(values = palette) +
    ggplot2::theme_minimal() +
    ggplot2::labs(x = xlabel, y = ylabel) +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )
}
