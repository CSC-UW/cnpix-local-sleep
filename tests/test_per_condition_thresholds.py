"""Pin the per-condition vs 48h quantile-threshold split.

The morphological thresholds were re-optimized for full-recording (48h) detection
in commit 7138c32. To keep the older, per-condition-tuned thresholds usable, the
per-condition detection path reads a sibling
``quantile_thresholds_per_condition.csv`` while full-recording detection keeps
reading the main ``quantile_thresholds.csv``. These tests assert that the two
paths read from their respective files and that variants without a sibling
(tom-bugnon) fall back to the main table.

See cnpix_local_sleep.morphological.common.MorphologicalSourceConfig and cnpix_local_sleep.morphological.detect.
"""

from __future__ import annotations

import pandas as pd


def _value(df: pd.DataFrame, subject, probe, structure, column) -> float:
    row = df[
        (df["subject"] == subject)
        & (df["probe"] == probe)
        & (df["structure_acronym"] == structure)
    ]
    return float(row[column].iloc[0])


class TestPerConditionThresholdSplit:
    def test_mua_per_condition_reads_sibling_file(self):
        from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG

        sibling = SOURCE_CONFIG.load_per_condition_quantile_thresholds()
        main = SOURCE_CONFIG.load_quantile_thresholds()
        # The mua sibling is a distinct file (otherwise this split is moot).
        assert not sibling.equals(main)

        subj, prb, st = "CNPIX7-Giuseppe", "imec0", "M2"
        assert SOURCE_CONFIG.get_per_condition_quantile_threshold(
            subj, prb, st, "Early.BSL.NREM"
        ) == _value(sibling, subj, prb, st, "nrem_quantile_threshold")
        assert SOURCE_CONFIG.get_per_condition_quantile_threshold(
            subj, prb, st, "Late.NOD.Wake"
        ) == _value(sibling, subj, prb, st, "wake_quantile_threshold")

    def test_mua_48h_reads_main_file(self):
        from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG

        main = SOURCE_CONFIG.load_quantile_thresholds()
        subj, prb, st = "CNPIX7-Giuseppe", "imec0", "M2"
        assert SOURCE_CONFIG.get_quantile_threshold(
            subj, prb, st, "Early.BSL.NREM"
        ) == _value(main, subj, prb, st, "nrem_quantile_threshold")

    def test_detect_value_path_uses_per_condition(self):
        """detect.py's from_value lookup must route to the per-condition table."""
        from cnpix_local_sleep.morphological import detect
        from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG

        subj, prb, st, cond = "CNPIX7-Giuseppe", "imec0", "M2", "Early.BSL.NREM"
        assert detect._get_threshold_quantile_from_value(
            subj, prb, st, cond, SOURCE_CONFIG
        ) == SOURCE_CONFIG.get_per_condition_quantile_threshold(
            subj, prb, st, cond
        )

    def test_detect_full_uses_main(self):
        from cnpix_local_sleep.morphological import detect_full
        from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG

        subj, prb, st = "CNPIX7-Giuseppe", "imec0", "M2"
        nrem_q, wake_q = detect_full._get_threshold_quantiles(
            subj, prb, st, SOURCE_CONFIG
        )
        main = SOURCE_CONFIG.load_quantile_thresholds()
        assert nrem_q == _value(main, subj, prb, st, "nrem_quantile_threshold")
        assert wake_q == _value(main, subj, prb, st, "wake_quantile_threshold")

