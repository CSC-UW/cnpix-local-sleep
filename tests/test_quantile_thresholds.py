"""Full-recording detection reads its quantiles from ``quantile_thresholds.csv``."""

from __future__ import annotations

import pandas as pd


def _value(df: pd.DataFrame, subject, probe, structure, column) -> float:
    row = df[
        (df["subject"] == subject)
        & (df["probe"] == probe)
        & (df["structure_acronym"] == structure)
    ]
    return float(row[column].iloc[0])


def test_detect_full_reads_quantile_thresholds_csv():
    from cnpix_local_sleep.morphological import detect_full
    from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG

    subj, prb, st = "CNPIX7-Giuseppe", "imec0", "M2"
    nrem_q, wake_q = detect_full._get_threshold_quantiles(
        subj, prb, st, SOURCE_CONFIG
    )
    main = SOURCE_CONFIG.load_quantile_thresholds()
    assert nrem_q == _value(main, subj, prb, st, "nrem_quantile_threshold")
    assert wake_q == _value(main, subj, prb, st, "wake_quantile_threshold")
