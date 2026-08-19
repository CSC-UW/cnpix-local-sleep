import ecephys
import matplotlib.pyplot as plt
import numpy as np
import wisc_ecephys_tools as wet
import xarray as xr
from cnpix.f25 import ipower as f25_ipower

from cnpix_local_sleep import const, files, hyp


class MissingAnatomyError(AttributeError):
    """Exception raised when anatomical data is missing from a recording."""

    pass


def open_ipwr(subject: str, probe: str, structure: str, band_name: str) -> xr.DataArray:
    nb = wet.get_sglx_project("shared_nobak")
    f = nb.get_experiment_subject_file(
        const.EXPERIMENT, subject, f"{probe}.i{band_name}.zarr"
    )
    ipwr = xr.open_dataarray(f)

    # Assign anatomy
    anatomy_proj = wet.get_sglx_project("shared")
    anatomy_file = anatomy_proj.get_experiment_subject_file(
        const.EXPERIMENT, subject, f"{probe}.structures.htsv"
    )
    if anatomy_file.exists():
        from ecephys.xrsig.core import assign_laminar_coordinate

        structs = ecephys.utils.read_htsv(anatomy_file)
        ipwr = assign_laminar_coordinate(ipwr, structs, sigdim="channel", lamdim="y")
        structs = structs.rename(
            columns={c: f"ref_{c}" for c in structs.columns if c not in ["lo", "hi"]}
        )
        ipwr = assign_laminar_coordinate(
            ipwr, structs, sigdim="channel", lamdim="ref_y"
        )
    else:
        raise MissingAnatomyError(
            f"Could not find anatomy file at: {anatomy_file}. Using dummy structure table"
        )

    if "acronym" not in ipwr.coords:
        raise MissingAnatomyError(f"No 'acronym' coordinate in {f.name}")
    channels = (ipwr["acronym"] == structure) & (ipwr["ref_acronym"] == structure)
    if not any(channels):
        raise MissingAnatomyError(f"No {structure} channels in {f.name}")
    return ipwr.isel({"channel": channels})


def do_structure(
    subject: str,
    probe: str,
    structure: str,
    band_name: str,
    plot_ipower: bool = False,
    overwrite: bool = False,
):
    savefile = files.get_structure_bandpower_path(
        subject, probe, structure, band_name, True, "inst"
    )
    if savefile.exists() and not overwrite:
        print(
            f"Skipping {subject}, {probe}, {structure}, {band_name}:"
            f" {savefile} already exists."
        )
        return None

    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    hg = hgs.pop("Full.Conservative").drop_states({"Artifact", "NoData"})

    try:
        ipwr = open_ipwr(subject, probe, structure, band_name).compute()
    except MissingAnatomyError as e:
        print(f"No {structure} channels in {subject}, {probe}, {band_name}")
        print(e)
        return None

    ipwr = ipwr.where(xr.DataArray(hg.covers_time(ipwr["time"]), dims="time"))
    ipwr = ipwr.median(dim="channel")
    _, ipwr.values = f25_ipower.replace_outliers_kd(ipwr.values, plot_distribution=True)

    savefile.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.suptitle(f"{subject}, {probe}, {structure}, {band_name}")
    fig.savefig(savefile.parent / (savefile.stem + ".threshold.png"))
    plt.close(fig)
    if plot_ipower:
        fs = np.ceil(1 / ipwr["time"].diff(dim="time").median())
        assert fs == 32, f"Sampling rate is {fs} Hz, not 32 Hz"
        sglx_subject = wet.get_sglx_subject(subject)
        ax = f25_ipower.plot_hypnogram(const.EXPERIMENT, sglx_subject, hg)
        ipwr.rolling(time=128, center=True).median().plot(x="time", color="k", ax=ax)
        fig = plt.gcf()
        fig.savefig(savefile.parent / (savefile.stem + ".4s_rolling_median.png"))
        plt.close(fig)
    ipwr.to_zarr(savefile)

    return ipwr


