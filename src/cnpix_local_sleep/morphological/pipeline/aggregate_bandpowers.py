from typing import Literal

import wisc_ecephys_tools as wet

from cnpix_local_sleep.files import Files
from cnpix_local_sleep.morphological import agg


# Runtime: 18 minutes for "inst". 2 minutes for "stft".
def do_project(bipolar: bool, kind: Literal["stft", "inst"]):
    nb = wet.get_sglx_project("offproj")

    pwr = agg.aggregate_bandpowers(bipolar, kind)
    c_pwr = agg.aggregated_events_wide_to_long(pwr)
    c_means = (
        c_pwr.groupby(["subject", "probe", "structure", "condition"])
        .mean(numeric_only=True)
        .add_prefix("mean_")
    )
    contrasts = agg.get_contrasts(c_means)

    c_means.to_parquet(nb.get_project_file(Files.BANDPOWER_MEANS(bipolar, kind)))
    contrasts.to_parquet(nb.get_project_file(Files.BANDPOWER_CONTRASTS(bipolar, kind)))