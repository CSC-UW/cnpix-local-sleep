"""Export the laminar-trimodality mechanical-null summary for r-offp.

An additive, fully separable companion to the other ``off-analysis export-*``
steps. It distils ``notebooks/figures/laminar_trimodality_null.ipynb`` down to one
tidy row per ``(subject, probe, structure)`` laminar combo, so the group-level
question (*across structures, is there real laminar depth structure once the
mechanical null is accounted for?*) can be tested in r-offp with subject as the
random effect.

Per combo we re-run the shape-preserving, depth-randomized null
(:mod:`cnpix_local_sleep.morphological.laminar_null`) and record:

- ``attr_{conc,com}_{uniform,feasible}``: the bin-free Wasserstein skill-score
  attribution ``1 - W1(emp, null)/W1(emp, flat)`` of each measure to the null,
  under both null placements:

  - ``feasible``, the no-clip, size-preserving null (each footprint dropped
    uniformly among its in-*detection*-window positions). This is the clean
    baseline for COM, because clipping cannot deflate the COM extremes; it isolates
    the *centroid-contraction* geometry from real depth occurrence.
  - ``uniform``, the whole-structure null: each footprint is dropped uniformly
    over the full *anatomical* structure (from the registration
    ``structures.htsv``, which typically extends tens of channels past the detection
    window), and only the detection channels are observed, so an OFF centered beyond
    the detected channels is seen only partially (its overhang clipped and lost).
    This says "structure OFFs share the observed size/shape distribution, but a
    limited detection span lets us see only part of them," and removes the
    feasible null's detection-edge taper. Where the structure barely exceeds the
    detection window the two placements nearly coincide.

  Attribution ~ 1 means the measure's structure is mechanical (OFF size + band
  geometry under random depth); ~ 0 means a real depth-occurrence residual. Its
  complement ``1 - attribution`` is the "real residual" the r-offp model tests
  against 0. Caveat: the skill score is only well-behaved when the null sits
  *between* the empirical and a flat reference. For the feasible COM that
  fails: the feasible (centroid-contraction) null is *more* contracted than the
  empirical, and on tall probes the empirical COM is itself near-flat, collapsing
  the ``W1(emp, flat)`` denominator and sending ``attr_com_feasible`` wildly
  negative. Use the robust effect sizes below for COM, not ``attr_com``.
- ``w1_{conc,com}_{uniform,feasible}`` + ``w1_{conc,com}_flat``: the raw
  earth-mover distances behind the attribution (its numerator ``W1(emp, null)``
  and denominator ``W1(emp, flat)``; µm for COM). ``w1_com_feasible`` is the
  always-interpretable effect size for the COM (centroid-contraction) test:
  how far the empirical COM sits from the size-preserving in-bounds-placement
  prediction.
- ``com_std_{emp,feasible,uniform}_um`` and ``com_spread_ratio_feasible`` (=
  ``std_emp / std_feasible``), the *direction* of the COM departure.
  ``> 1`` means the empirical COM reaches extreme depths more than the
  centroid-contraction null predicts (real extreme-depth occurrence); ``< 1`` means
  the empirical COM is more central than contraction predicts.
- ``occ_{time,count}{,_feasible}_{tv,w1,p}``: occupancy effect sizes vs the
  depth null for the time-weighted (total OFF-time per channel) and count-weighted
  (duration-blind, events per channel) readouts: ``tv`` (fraction of occupancy
  mass that must move), ``w1`` (earth-mover distance, µm), and the permutation
  ``p`` (~0 at ~10^5 OFFs; read the effect size, not the p). Two null placements:
  ``occ_{time,count}_feasible_*`` use the no-clip, size-preserving ``feasible``
  (in-detection-window) null, the primary; the bare ``occ_{time,count}_*`` use the
  ``uniform`` whole-structure null, which places same-size OFFs over the full
  anatomical structure and observes only the detection window. The uniform null's
  out-of-window overhang is divided out by unit-mass renormalization, so its f-
  dependent partial-visibility deficit biases neither ``tv`` nor ``w1`` (both are
  shape distances); it replaces the feasible null's detection-edge taper with the
  asymmetric leak-in shape. Prefer the ``feasible`` columns; read ``uniform`` as a
  whole-structure robustness check.
- ``occ_{time,count}_feasible_asym`` and ``occ_{time,count}_feasible_asym_norm``:
  the signed superficial-vs-deep asymmetry of the empirical occupancy excess
  (feasible null), a directional companion to the unsigned ``tv``/``w1``. ``asym``
  = (superficial-half observed mass fraction) - (superficial-half null fraction),
  with the channels split 50/50 at the depth midpoint and flip-corrected
  orientation; ``> 0`` means excess toward the superficial (cortical-surface) half,
  ``< 0`` means toward the deep half. ``asym_norm = asym/tv`` in [-1, 1] is the share of
  the displaced mass that is one-sided. See
  :func:`cnpix_local_sleep.morphological.laminar_null.occupancy_asymmetry`.
- bookkeeping: ``n_offs``, ``n_chans``, ``gap_chans`` (excluded-middle width in
  channels vs ``n_channels_connect=5``), ``clade == "Cx"``, ``condition ==
  "NREM"`` so the R runner's ``conditions=`` filter works unchanged.

One parquet, ``summarized_depth_profile.parquet`` (one row per combo). Writes
only into r-offp, never NFS. Requires NFS mounted: it reads the full-48h
``morphological`` detection (``offs.parquet`` + ``off_label_indices.parquet``).
Delete this module and the ``export-depth-profile-summary`` CLI command to remove
the Python side entirely; nothing else imports it.
"""

import pathlib

import numpy as np
import pandas as pd

from cnpix_local_sleep import atlas, sps_conf
from cnpix_local_sleep.morphological import laminar_null as ln
from cnpix_local_sleep.morphological import mua
from cnpix_local_sleep.morphological.pipeline import postprocess_offs as ppo
from cnpix_local_sleep import channel_anatomy


def _gap_channels(subject: str, probe: str, structure: str, y_coords: np.ndarray) -> float:
    """Excluded-middle (granular gap) width in channels for one structure.

    The supra/infra bands leave a 10%-of-span excluded middle; expressed in
    channels (median pitch) this is the geometry the final spatial closing
    (``n_channels_connect=5``) can bridge to fabricate the central 0.5
    concentration peak. Mirrors the notebook's ``gap_chans``.
    """
    borders = channel_anatomy.get_layer_borders(subject, probe, structure)
    supra = borders[borders["layer"] == "supra"].iloc[0]
    infra = borders[borders["layer"] == "infra"].iloc[0]
    pitch = float(np.median(np.diff(np.sort(y_coords))))
    gap_um = float(supra["lo"] - infra["hi"])
    return gap_um / pitch


def summarize_combo(
    subject: str,
    probe: str,
    structure: str,
    rng: np.random.Generator,
    *,
    n_perm_occ: int = 200,
) -> dict:
    """One tidy summary row for a single laminar combo (see module docstring)."""
    offs, lbl_ixs, y = ln.load_structure_data(subject, probe, structure)
    ref = offs.set_index("label")

    # Empirical measures via the production code (flip-aware concentration SPOT).
    sc_emp, _ic_emp = ppo.laminar_concentrations(
        ref, subject=subject, probe=probe, structure=structure
    )
    sc_emp = sc_emp.to_numpy()
    com_emp = ref["center_of_mass_depth"].to_numpy()
    support_com = (float(y.min()), float(y.max()))

    # Collapse the footprints ONCE and reuse across both placements and both
    # occupancy readouts (the per-OFF collapse is the dominant cost).
    collapsed = ln.collapse_footprints(lbl_ixs)

    # The ``uniform`` null places same-size OFFs over the whole ANATOMICAL structure
    # (which extends past the detection window) and observes only the detection
    # channels; compute that structure window once and pass it to the occupancy
    # test. (``null_measures_per_off`` derives it internally from subject/probe/
    # structure, so the COM/concentration nulls already use it.)
    struct_bounds = ln.structure_index_bounds(y, subject, probe, structure)

    # Null measures per OFF under both placements (fast vectorized kernel).
    null_uniform = ln.null_measures_per_off(
        lbl_ixs, y, subject, probe, structure, rng,
        placement="uniform", collapsed=collapsed,
    )
    null_feasible = ln.null_measures_per_off(
        lbl_ixs, y, subject, probe, structure, rng,
        placement="feasible", collapsed=collapsed,
    )

    com_null_u = null_uniform["center_of_mass_depth"].to_numpy()
    com_null_f = null_feasible["center_of_mass_depth"].to_numpy()

    attr = {}
    for measure, emp, support, null_u, null_f in [
        ("conc", sc_emp, (0.0, 1.0),
         null_uniform["supra_concentration"].to_numpy(),
         null_feasible["supra_concentration"].to_numpy()),
        ("com", com_emp, support_com, com_null_u, com_null_f),
    ]:
        a_u = ln.mechanical_attribution(emp, null_u, support=support, rng=rng)
        a_f = ln.mechanical_attribution(emp, null_f, support=support, rng=rng)
        attr[f"attr_{measure}_uniform"] = a_u["attribution"]
        attr[f"attr_{measure}_feasible"] = a_f["attribution"]
        attr[f"{measure}_uniform_resolvable"] = a_u["resolvable"]
        attr[f"{measure}_feasible_resolvable"] = a_f["resolvable"]
        # Robust, always-interpretable effect size: the raw earth-mover distance
        # between the empirical and null distributions (the attribution's own
        # numerator), plus the distance to flat (its denominator) for transparency.
        # For COM these are microns; for concentration, concentration units.
        attr[f"w1_{measure}_uniform"] = a_u["w_null"]
        attr[f"w1_{measure}_feasible"] = a_f["w_null"]
        attr[f"w1_{measure}_flat"] = a_f["w_flat"]

    # COM spread (direction of the COM departure). The feasible null is the
    # centroid-contraction prediction; std(emp) > std(feasible) means the
    # empirical COM reaches extreme depths *more* than size + in-bounds placement
    # can produce. This signed spread ratio is the honest direction summary for
    # the feasible-COM test (the skill-score attribution is unreliable here
    # because its flat-reference denominator collapses when empirical COM is
    # itself near-flat, as on the tall probes).
    com_std_emp = float(np.nanstd(com_emp))
    com_std_feasible = float(np.nanstd(com_null_f))
    com_std_uniform = float(np.nanstd(com_null_u))
    attr["com_std_emp_um"] = com_std_emp
    attr["com_std_feasible_um"] = com_std_feasible
    attr["com_std_uniform_um"] = com_std_uniform
    attr["com_spread_ratio_feasible"] = (
        com_std_emp / com_std_feasible if com_std_feasible > 0 else np.nan
    )

    # Occupancy effect sizes vs the depth null (both readouts, both placements).
    # ``feasible`` places each OFF uniformly within the DETECTION window (no clip,
    # size preserved), the in-window baseline. ``uniform`` places each OFF
    # uniformly over the whole ANATOMICAL structure and observes only the detection
    # window, so OFFs centered beyond the detected channels are seen only partially.
    # The clipped overhang is divided out by unit-mass renormalization (the test
    # compares depth SHAPE only), so the f-dependent partial-visibility deficit
    # biases neither tv nor W1. Where the structure barely exceeds the detection
    # window the two placements nearly coincide. Both emitted; ``feasible`` is the
    # primary, ``uniform`` the whole-structure robustness.
    occ = {}
    for w in ("time", "count"):
        for placement, suffix in (("uniform", ""), ("feasible", "_feasible")):
            res = ln.occupancy_null_test(
                lbl_ixs, y, rng, n_perm=n_perm_occ, weighting=w,
                placement=placement, struct_bounds=struct_bounds,
                collapsed=collapsed,
            )
            occ[f"occ_{w}{suffix}_tv"] = res["tv"]
            occ[f"occ_{w}{suffix}_w1"] = res["w1_um"]
            occ[f"occ_{w}{suffix}_p"] = res["p_global"]
            if placement == "feasible":
                # Signed superficial-vs-deep asymmetry of the empirical excess (a
                # directional companion to the unsigned tv/w1). Feasible null only.
                a = ln.occupancy_asymmetry(
                    res["obs_p"], res["null_mean"], res["y"],
                    subject, probe, structure,
                )
                occ[f"occ_{w}_feasible_asym"] = a["asym"]
                occ[f"occ_{w}_feasible_asym_norm"] = a["asym_norm"]

    return {
        "subject": subject,
        "probe": probe,
        "structure": structure,
        "clade": "Cx",
        "condition": "NREM",
        "n_offs": int(len(offs)),
        "n_chans": int(y.size),
        "gap_chans": _gap_channels(subject, probe, structure, y),
        **attr,
        **occ,
    }


def summarize_depth_profile(
    *, seed: int = 0, n_perm_occ: int = 200
) -> pd.DataFrame:
    """One-row-per-combo laminar-null summary over all 13 cortical-laminar combos.

    Combos come from the canonical ``morphological`` inclusion list restricted to
    laminar cortical structures. A single seeded RNG threads every combo, so the
    output is reproducible.
    """
    spsl = sps_conf.get_subject_probe_structure_list(
        method=mua.files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
        exclude_nonlaminar=True,
    )
    rng = np.random.default_rng(seed)
    rows = []
    for subject, probe, structure in spsl:
        if not (
            mua.files.get_full_offs_path(subject, probe, structure).exists()
            and mua.files.get_full_off_label_indices_path(
                subject, probe, structure
            ).exists()
        ):
            print(f"  skipping {subject}/{probe}/{structure} (no full-48h detection)")
            continue
        print(f"  {subject}/{probe}/{structure}...")
        rows.append(
            summarize_combo(subject, probe, structure, rng, n_perm_occ=n_perm_occ)
        )
    return pd.DataFrame(rows)


def export_depth_profile_summary(
    output_dir: pathlib.Path | str, *, seed: int = 0, n_perm_occ: int = 200
) -> None:
    """Write ``summarized_depth_profile.parquet`` into ``output_dir``."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Summarizing laminar-trimodality mechanical null (per structure)...")
    summary = summarize_depth_profile(seed=seed, n_perm_occ=n_perm_occ)
    # Interim OFF-analysis structure consolidation (e.g. mPPC -> PPC).
    summary = atlas.consolidate_off_structure_columns(summary)
    out = output_dir / "summarized_depth_profile.parquet"
    summary.to_parquet(out)
    print(f"  {len(summary)} combos -> {out}")
