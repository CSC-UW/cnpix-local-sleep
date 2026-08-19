#!/usr/bin/env bash
# Regenerate the manuscript statistical tables (SLEEP-ready .docx + .xlsx).
#
# Layers, cheapest first. By default only the last (fast) layer runs; pass flags
# to also refresh the upstream data. Run from the r-offp/ root; `python` must be
# the workspace venv interpreter (as for summarize_results.py / plot_results.py).
#
#   ./make_manuscript_tables.sh                 # build tables from current outputs
#   ./make_manuscript_tables.sh --exports       # + refresh cnpix_local_sleep correlation CSVs (needs local cache)
#   ./make_manuscript_tables.sh --rerun-R       # + re-run the R analyses (reads committed extdata; no NFS)
#
# On a clone with no _output* tree, S4a/S4b need --rerun-diagnostics as well; without
# it the supplement builds and warns, but comes out missing those two tables.
#   ./make_manuscript_tables.sh --rerun-diagnostics  # + re-run the S4b leave-out battery (minutes)
#   ./make_manuscript_tables.sh --all           # everything
set -euo pipefail
cd "$(dirname "$0")"

DO_EXPORTS=0; DO_R=0; DO_NFS=0; DO_DIAG=0
for a in "$@"; do
  case "$a" in
    --exports) DO_EXPORTS=1 ;;
    --rerun-R) DO_R=1 ;;
    --nfs-exports) DO_NFS=1 ;;   # size/globality Spearman (Fig 4d-f); needs NFS, slow
    --rerun-diagnostics) DO_DIAG=1 ;;  # S4a transforms + S4b leave-out battery; ~thousands of lmer refits
    --all) DO_EXPORTS=1; DO_R=1; DO_NFS=1; DO_DIAG=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

if [[ "$DO_NFS" == 1 ]]; then
  echo ">> Refreshing size/globality Spearman (Fig 4d-f); requires NFS ..."
  off-analysis export-size-globality-correlations
fi

if [[ "$DO_R" == 1 ]]; then
  echo ">> Re-running R analyses (cx, bandpower, correlations, locality, excess) ..."
  Rscript scripts/run_all_analyses.R all
  Rscript scripts/run_all_bandpower_analyses.R
  Rscript scripts/run_all_correlation_analyses.R all
  Rscript scripts/run_all_locality_analyses.R
  Rscript scripts/run_excess_globality_analysis.R
  # Depth profile (S5), group summary (needs summarized_depth_profile.parquet):
  Rscript scripts/depth_profile_summary.R || echo "   (skipped depth_profile_summary.R)"
fi

if [[ "$DO_DIAG" == 1 ]]; then
  # S4a's rank/log rows and every S4b column. Must follow --rerun-R: both scripts
  # refit from the same data as the base correlation fits, and manuscript_tables.py
  # refuses to build S4a if the transform sweep and the base fits disagree.
  echo ">> Re-running correlation robustness diagnostics (predictor transforms + leave-out battery) ..."
  Rscript scripts/run_correlation_xtransform_sensitivity.R
  Rscript scripts/run_correlation_influence_diagnostics.R
fi

if [[ "$DO_EXPORTS" == 1 ]]; then
  echo ">> Refreshing cnpix_local_sleep correlation inputs (per-event + epoched) ..."
  off-analysis export-manuscript-correlations
fi

echo ">> Building manuscript tables (combined supplement + per-table files) ..."
CFG=config/summary_tables
python manuscript_tables.py --supplement "$CFG/manuscript_supplement.yaml"
for t in s0_models s1a_homeostasis s2a_per_event s2b_epoched s3a_locality s3b_size_globality \
         s4a_sd_rebound s4b_robustness s5_depth; do
  python manuscript_tables.py "$CFG/manuscript_${t}.yaml" >/dev/null
done

echo ">> Building S1b companion data workbook (per-combo values underlying S1a) ..."
python manuscript_s1b_values.py >/dev/null

echo ">> Done. Outputs in _output_manuscript/ :"
ls -1 _output_manuscript/manuscript_supplement.* _output_manuscript/manuscript_s1b_values.* 2>/dev/null || true
