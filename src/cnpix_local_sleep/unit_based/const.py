"""Constants for unit-based (pooled spike-train) OFF detection.

Detection pools the spike trains of all units in a (subject, probe, structure)
into a single multiunit train and runs one of the ``on_off_detection`` methods
on it, separately for each macro-state (NREM, NOD-Wake). See
``cnpix_local_sleep.unit_based.pipeline.detect_full``.
"""

import numpy as np

# Macro-state passes. NREM covers the 4 NREM core conditions; NOD-Wake covers
# the 2 wake core conditions (Early/Late.NOD.Wake).
NREM_STATES = ["NREM"]
WAKE_STATES = ["Wake"]

# Pooled multiunit firing-rate gate for hmmem ONLY (Hz). The Chen 2009 GLM-HMM
# (hmmem) is unreliable below ~100 Hz (morphological-detector heuristics in
# on_off_detection/methods/hmmem.py); below this hmmem falls back to sticky.
# sticky+cap0 and threshold run at ANY firing rate -- a low-FR test (2026-06-17)
# showed sticky+cap0 does not break at low FR and is clearly better than threshold
# at >=~60 Hz, ~tied below. Provenance: the cap-scheme sweep scripts were removed
# 2026-08-12 (recoverable from git history); their figures survive on NFS under
# detection_mode=pooled-sticky/offrate_calibration/ and in unit_based/README.md.
MIN_POOLED_FR = 100.0

# Below this pooled NREM firing rate, unit-based detection is flagged
# low-confidence (few units / sparse population): detection still runs (sticky is
# robust), but the result carries low_confidence=True for downstream filtering.
LOW_CONFIDENCE_POOLED_FR = 100.0

# Default detection algorithm.
DEFAULT_ALGO = "sticky"
ALGOS = ("threshold", "hmmem", "sticky")

# Bin size shared across methods (s). Used both for detection and for the
# area proxy in the Off-schema mapping.
BINSIZE = 0.010

# Per-algorithm parameters passed to on_off_detection.OnOffModel. Keys must be a
# subset of the method's default param dict (DF_PARAMS[method]).
UNIT_BASED_PARAMS = {
    "threshold": {
        "binsize": BINSIZE,
        "smooth_sd_counts": 3,  # units of binsize (~30 ms Gaussian SD)
        "count_threshold": None,  # auto from smoothed-count histogram
        "gap_threshold": None,  # auto from OFF-duration histogram
    },
    # HMM-EM values reuse the previously tuned NREM settings (history window 10,
    # init_state_off_on_fr_ratio_thresh 0.05; see hmmem.py heuristics).
    "hmmem": {
        "binsize": BINSIZE,
        "history_window_nbins": 10,
        "n_iter_EM": 200,
        "n_iter_newton_ralphson": 100,
        "init_A": np.array([[0.1, 0.9], [0.01, 0.99]]),
        "init_state_off_on_fr_ratio_thresh": 0.05,
        "init_mu": None,
        "init_alphaa": None,
        "init_betaa": None,
        "min_off_duration": None,
    },
    # Sticky Poisson-HMM (Li & La Camera 2025). min_dwell sets the
    # self-transition floor delta = 1 - binsize/min_dwell. off_rate_max is the
    # near-silence constraint on the OFF state.
    #
    # off_rate_max=0.0 (cap0): OFF is the silent (zero-count) state. This is
    # PARAMETER-FREE and structure-invariant -- chosen over the old fixed 20 Hz
    # absolute cap and a per-unit cap via the cap-scheme sweep across 21 cortical
    # structures: cap0 gave the best median F1 vs morphological BLAS (0.578) and the
    # deepest, most BLAS-like OFFs. The fixed 20 Hz cap does NOT generalize
    # (P(silent 10 ms bin) spans ~50x across cortex; an absolute Hz cap or a
    # count-percentile occupy wildly different operating points per structure).
    # The sweep script was removed 2026-08-12 (in git history); its figures survive
    # on NFS under detection_mode=pooled-sticky/offrate_calibration/.
    "sticky": {
        "binsize": BINSIZE,
        "min_dwell": 0.050,
        "off_rate_max": 0.0,
        "n_iter_EM": 100,
        "tol": 1e-4,
        "min_off_duration": None,
    },
}
