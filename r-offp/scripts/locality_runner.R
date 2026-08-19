#' Shared helpers for the locality analysis scripts.
#'
#' Sourced by run_locality_analysis.R and run_all_locality_analyses.R. Expands the
#' config/locality.yaml entries into calls on the exported locality orchestrators
#' (requests 2 & 3) and routes request 1 (mean # overlapping structures) through
#' the layer-agnostic run_cx_homeostasis_analysis. Writes to _output_locality/.

LOCALITY_OUTPUT_ROOT <- "_output_locality"

# Run an expression, reporting (but not aborting on) errors so one bad fit does
# not stop the whole sweep.
locality_try <- function(label, expr) {
  tryCatch(
    force(expr),
    error = function(e) message("  ERROR [", label, "]: ", conditionMessage(e)),
    warning = function(w) {
      message("  warning [", label, "]: ", conditionMessage(w))
      suppressWarnings(force(expr))
    }
  )
}

#' Request 1: condition contrasts on mean_overlap_degree (run once per dataset).
#'
#' @param config Parsed locality config (list).
#' @param dataset OFF type ("llas").
#' @param data Pre-loaded overlap-degree data frame.
run_overlap_degree <- function(config, dataset, data = NULL) {
  od <- config$overlap_degree
  if (is.null(od)) {
    return(invisible(NULL))
  }
  response_var <- od$response_variable
  # `models` (list) is preferred so several models can be compared; fall back to
  # a singular `model` for backward compatibility. Each model nests its own
  # output subdir (run_cx_homeostasis_analysis appends /<model_def$name>/).
  models <- if (!is.null(od$models)) od$models else list(od$model)
  output_dir <- file.path(LOCALITY_OUTPUT_ROOT, dataset,
                          "overlap_degree", response_var)
  for (model_def in models) {
    label <- paste(dataset, response_var, "overlap_degree", model_def$name)
    message("[overlap-degree] ", label)
    locality_try(label, offp::run_cx_homeostasis_analysis(
      response_var = response_var, output_dir = output_dir,
      model_def = model_def, off_type = dataset,
      condition_set = "six", conditions = od$conditions, posthocs = od$posthocs,
      data = data
    ))
  }
}

#' Run every request-2/3 analysis entry for one measure in one dataset.
#'
#' @param config Parsed locality config (list).
#' @param dataset OFF type ("llas").
#' @param rv_entry One `response_variables` entry (`response_variable`, `analyses`).
#' @param data_pc Pre-loaded + prepared per-condition locality data frame (req 2).
#' @param data_state Pre-loaded + prepared whole-recording state data frame (req 3).
run_locality_rv <- function(config, dataset, rv_entry,
                            data_pc = NULL, data_state = NULL) {
  response_var <- rv_entry$response_variable
  windows <- config$condition_windows
  per_cond <- config$per_condition_conditions
  states <- config$states
  contrast <- if (is.null(config$locality_contrast)) {
    "Overlapping - Local = 0"
  } else {
    config$locality_contrast
  }

  for (analysis in rv_entry$analyses) {
    kind <- analysis$kind
    model_def <- analysis$model

    if (kind == "per_condition") {
      for (cond in per_cond) {
        ak <- paste0("cond-", cond)
        output_dir <- file.path(LOCALITY_OUTPUT_ROOT, dataset, ak,
                                response_var)
        label <- paste(dataset, response_var, ak, model_def$name)
        message("[main-effect] ", label)
        locality_try(label, offp::run_locality_main_effect_analysis(
          response_var = response_var, output_dir = output_dir,
          model_def = model_def, off_type = dataset,
          analysis_kind = ak, conditions = cond,
          locality_posthocs = contrast, data = data_pc
        ))
      }

    } else if (kind == "state") {
      for (st in states) {
        ak <- paste0("state-", st)
        output_dir <- file.path(LOCALITY_OUTPUT_ROOT, dataset, ak,
                                response_var)
        label <- paste(dataset, response_var, ak, model_def$name)
        message("[main-effect] ", label)
        locality_try(label, offp::run_locality_main_effect_analysis(
          response_var = response_var, output_dir = output_dir,
          model_def = model_def, off_type = dataset,
          analysis_kind = ak, conditions = st,
          locality_posthocs = contrast, data = data_state
        ))
      }

    } else if (kind == "interaction") {
      cs <- analysis$condition_set
      win <- windows[[cs]]
      posthocs <- config$interaction_posthocs[[cs]]
      output_dir <- file.path(LOCALITY_OUTPUT_ROOT, dataset,
                              paste0("interaction-", cs), response_var)
      label <- paste(dataset, response_var, paste0("interaction-", cs),
                     model_def$name)
      message("[interaction] ", label)
      locality_try(label, offp::run_locality_interaction_analysis(
        response_var = response_var, output_dir = output_dir,
        model_def = model_def, off_type = dataset,
        condition_set = cs, conditions = win$conditions, posthocs = posthocs,
        data = data_pc
      ))

    } else if (kind == "state_interaction") {
      cs <- analysis$condition_set  # "state"
      posthocs <- config$interaction_posthocs[[cs]]
      output_dir <- file.path(LOCALITY_OUTPUT_ROOT, dataset,
                              paste0("interaction-", cs), response_var)
      label <- paste(dataset, response_var, paste0("interaction-", cs),
                     model_def$name)
      message("[interaction] ", label)
      locality_try(label, offp::run_locality_interaction_analysis(
        response_var = response_var, output_dir = output_dir,
        model_def = model_def, off_type = dataset,
        condition_set = cs, conditions = states, posthocs = posthocs,
        data = data_state
      ))

    } else {
      stop("Unknown analysis kind: ", kind)
    }
  }
}
