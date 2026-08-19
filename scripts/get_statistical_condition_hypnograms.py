"""CNPIX6-Eugene's imec0 has hippocampal LFP issues that ruin it's circadian match, but
apparently Tom deemed that it can be used for a cortical circadian match. This script
creates a project-specific set of hypnograms for CNPIX6-Eugene, whereas all other
subjects use the shared project-agnostic hypnograms."""

import pandas as pd
import wisc_ecephys_tools as wet
from ecephys import hypnogram as hyp
from wisc_ecephys_tools.rats import cnd_hgs, exp_hgs

from cnpix_local_sleep import const

EXTENDED_WAKE_KWARGS = {
    "minimum_endpoint_bout_duration": 120,
    "maximum_antistate_bout_duration": 95,
    "minimum_fraction_of_final_match": 0.95,
}
CIRCADIAN_MATCH_TOLERANCE = 30 * 60  # 30 minutes


def do_probe(subject: str, probe: str):
    s3 = wet.get_sglx_project("shared")
    sglx_subject = wet.get_sglx_subject(subject)
    lbrl_hg = exp_hgs.get_liberal_hypnogram(s3, const.EXPERIMENT, sglx_subject, probe)

    cons_hg = exp_hgs.load_hypnogram(
        s3,
        const.EXPERIMENT,
        sglx_subject,
        probe,
        include_ephyviewer_edits=True,
        include_sorting_nodata=True,
        include_lf_consolidated_artifacts=False,
        include_ap_consolidated_artifacts=True,
        include_lf_sglx_filetable_nodata=False,
        include_ap_sglx_filetable_nodata=True,
        simplify=True,
        fallback=True,
    )

    return cnd_hgs.compute_statistical_condition_hypnograms(
        lbrl_hg,
        cons_hg,
        const.EXPERIMENT,
        sglx_subject,
        extended_wake_kwargs=EXTENDED_WAKE_KWARGS,
        circadian_match_tolerance=CIRCADIAN_MATCH_TOLERANCE,
    )


def do_subject(
    subject: str,
    probes: tuple[str, ...],
    verbose: bool = True,
    save: bool = True,
) -> tuple[
    dict[str, hyp.FloatHypnogram],
    pd.DataFrame,
    dict[str, dict[str, hyp.FloatHypnogram]],
]:
    prb_hgs = {prb: do_probe(subject, prb) for prb in probes}
    if save:
        nb = wet.get_sglx_project("offproj")
        for prb, hgs in prb_hgs.items():
            fpath = nb.get_experiment_subject_file(
                const.EXPERIMENT,
                subject,
                f"{prb}.condition_hypnograms.parquet",
            )
            cnd_hgs.save_statistical_condition_hypnograms(hgs, fpath)
    if len(prb_hgs) < 2:
        return None, None, prb_hgs

    consensus_hgs, consensus_df = cnd_hgs.get_consensus(prb_hgs)
    if verbose:
        pd.set_option("display.max_rows", 100)
        print(consensus_df)
        pd.reset_option("display.max_rows")
    if save:
        fpath = nb.get_experiment_subject_file(
            const.EXPERIMENT,
            subject,
            "consensus_condition_hypnograms.parquet",
        )
        cnd_hgs.save_statistical_condition_hypnograms(consensus_hgs, fpath)
    return consensus_hgs, consensus_df, prb_hgs


if __name__ == "__main__":
    do_subject("CNPIX6-Eugene", ("imec0", "imec1"), verbose=True, save=True)
