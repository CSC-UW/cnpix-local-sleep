"""Utilities for loading unit sortings with quality filters applied.

Cell-type classification lives in :mod:`cnpix.units` (project-agnostic, shared
across analyses) -- not here. Use ``cnpix.units.load_cell_types`` and join on
``(subject, experiment, probe, cluster_id)``.

Even when we perform unit-free OFF detection, we often want to visualize
the results alongside actual spike-sorted units. That's why this module is not in the
unit-based OFF detection subpackage.
"""

from collections import OrderedDict

import numpy as np
import wisc_ecephys_tools as wet
from ecephys.units.siks_sorting import SpikeInterfaceKilosortSorting
from ecephys.wne import siutils

from cnpix_local_sleep import const

QUALITY_FILTERS = OrderedDict(
    [
        ("all", ({"quality": {"good", "mua", "unsorted"}}, None)),
        (
            "mu",
            siutils.get_quality_metric_filters(
                "conservative",
                isolation_threshold=None,
                false_negatives_threshold=None,
                presence_threshold=None,
            ),
        ),
        (
            "su2",
            siutils.get_quality_metric_filters(
                "conservative",
                isolation_threshold="permissive",
                false_negatives_threshold="permissive",
                presence_threshold=None,
            ),
        ),
        (
            "su1",
            siutils.get_quality_metric_filters(
                "conservative",
                isolation_threshold="moderate",
                false_negatives_threshold="moderate",
                presence_threshold=None,
            ),
        ),
        (
            "su0",
            siutils.get_quality_metric_filters(
                "conservative",
                isolation_threshold="conservative",
                false_negatives_threshold="conservative",
                presence_threshold=None,
            ),
        ),
    ]
)

# Human-readable names for the per-unit `unit_quality` codes (QUALITY_FILTERS
# keys plus the "???" sentinel), ordered strictest (best isolation) to weakest.
# Used by GUIs that surface per-unit quality. "all" is the residual bucket: a
# unit that passes the broad good/mua/unsorted filter but fails even the MUA
# isolation/rate gate (i.e. below MUA); "???" never matched any filter.
QUALITY_TIER_DISPLAY = OrderedDict(
    [
        ("su0", "conservative SUA"),
        ("su1", "moderate SUA"),
        ("su2", "permissive SUA"),
        ("mu", "MUA"),
        ("all", "low-quality"),
        ("???", "unclassified"),
    ]
)


def assign_unit_quality(
    sorting,
):
    sorting.si_obj.set_property(
        "unit_quality",
        values=np.array(
            ["???" for _ in range(len(sorting.get_unit_ids()))], dtype=object
        ),
    )
    for filter_name, filters in QUALITY_FILTERS.items():
        sub_sorting = sorting.refine_clusters(*filters, verbose=False)
        ids = sub_sorting.get_unit_ids()
        sorting.si_obj.set_property(
            "unit_quality",
            values=np.array([filter_name for _ in range(len(ids))], dtype="object"),
            ids=ids,
        )
    return sorting


def load_sorting(
    subject: str,
    probe: str,
    experiment: str = const.EXPERIMENT,
    unit_quality="all",
    include_nans=True,
) -> SpikeInterfaceKilosortSorting:
    from ecephys.wne.sglx import legacy_sorting

    sorting = legacy_sorting.load_singleprobe_sorting(
        wet.get_sglx_project("shared"),
        subject,
        experiment,
        probe,
        alias="full",
    )
    sorting = assign_unit_quality(sorting)  # refine in sorting.properties
    sorting = sorting.refine_clusters(
        *QUALITY_FILTERS[unit_quality], include_nans=include_nans
    )
    return sorting


def load_structure_sorting(
    subject: str,
    probe: str,
    structure: str,
    experiment: str = const.EXPERIMENT,
    unit_quality="all",
) -> SpikeInterfaceKilosortSorting:
    return load_sorting(
        subject,
        probe,
        experiment=experiment,
        unit_quality=unit_quality,
    ).select_structures([structure])
