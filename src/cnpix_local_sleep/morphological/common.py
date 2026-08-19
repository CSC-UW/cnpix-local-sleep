"""Source-agnostic morphological-detection workflow.

Holds the ``MorphologicalSourceConfig`` dataclass used to plumb file-path,
trace-reader, and quantile-threshold behaviour through the shared detection
code.

Only one variant survives (``morphological``); the legacy ``tom-bugnon`` variant
was removed in the manuscript-relevance prune. The indirection is retained
because it is what keeps detection code from hard-coding its own on-disk
``method=`` segment, and because ``cnpix_local_sleep.harding`` and the unit-based
detectors follow the same shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from types import ModuleType
from typing import Literal

import pandas as pd
import xarray as xr


MorphologicalVariant = Literal["morphological"]

# Sibling of ``quantile_thresholds.csv`` holding the thresholds used by the
# per-condition detection path. Kept separate because the main file's thresholds
# were re-optimized for full-recording (48h) detection (commit 7138c32,
# 2026-05-18) and should not be assumed strictly better for per-condition
# detection.
PER_CONDITION_THRESHOLDS_FILENAME = "quantile_thresholds_per_condition.csv"


@dataclass(frozen=True)
class MorphologicalSourceConfig:
    """Plumbing for a morphological trace source.

    Attributes
    ----------
    variant
        Canonical variant name. Matches the ``method=`` path segment
        that ``files_module`` emits on disk.
    files_module
        Module exposing ``get_path`` and the path helpers
        (``get_offs_path``, ``get_channel_thresholds_path``, ...).
        Detection code reads and writes through this module rather than
        constructing paths itself.
    open_traces_as_xarray
        Callable producing a lazy ``xr.DataArray`` of traces.
    thresholds_package
        Dotted name of the package whose ``data/quantile_thresholds.csv``
        holds the per-(subject, probe, structure) NREM and Wake quantile
        thresholds and the ``include`` flag. Separate from
        ``cnpix_local_sleep.sps_conf`` because both thresholds and inclusion
        decisions are method-specific. Inclusion is dispatched via
        :func:`cnpix_local_sleep.sps_conf.load_method_inclusion` keyed on
        ``self.variant``. The threshold values are hand-set with the
        morphological tuner rather than computed.
    """

    variant: MorphologicalVariant
    files_module: ModuleType
    open_traces_as_xarray: Callable[..., xr.DataArray]
    thresholds_package: str

    def load_quantile_thresholds(self) -> pd.DataFrame:
        """Return this variant's quantile-threshold table.

        Schema: columns ``subject``, ``probe``, ``structure_acronym``,
        ``nrem_quantile_threshold``, ``wake_quantile_threshold``,
        ``include``, ``notes``. Missing threshold cells appear as NaN;
        callers that need a value for a particular row must check for
        that.
        """
        with resources.path(
            f"{self.thresholds_package}.data", "quantile_thresholds.csv"
        ) as f:
            return pd.read_csv(f)

    def load_per_condition_quantile_thresholds(self) -> pd.DataFrame:
        """Return the per-condition detection threshold table.

        Same schema as :meth:`load_quantile_thresholds`. Reads
        ``quantile_thresholds_per_condition.csv`` (frozen
        pre-48h-optimization thresholds).
        """
        with resources.path(
            f"{self.thresholds_package}.data", PER_CONDITION_THRESHOLDS_FILENAME
        ) as f:
            return pd.read_csv(f)

    def get_subject_probe_structure_list(
        self, **kwargs: object
    ) -> list[tuple[str, str, str]]:
        """Variant-aware wrapper around ``sps_conf.get_subject_probe_structure_list``.

        Fills in ``method=self.variant`` so shared morphological pipeline
        code doesn't need to know its own variant string.
        """
        from cnpix_local_sleep import sps_conf

        return sps_conf.get_subject_probe_structure_list(
            method=self.variant, **kwargs  # type: ignore[arg-type]
        )

    def get_excluded_structures(self) -> list[tuple[str, str, str]]:
        """Variant-aware wrapper around ``sps_conf.get_excluded_structures``."""
        from cnpix_local_sleep import sps_conf

        return sps_conf.get_excluded_structures(method=self.variant)

    def get_subject_probe_list(
        self, **kwargs: object
    ) -> list[tuple[str, str]] | list[tuple[str, tuple[str, ...]]]:
        """Variant-aware wrapper around ``sps_conf.get_subject_probe_list``."""
        from cnpix_local_sleep import sps_conf

        return sps_conf.get_subject_probe_list(
            method=self.variant, **kwargs  # type: ignore[arg-type]
        )

    @staticmethod
    def _quantile_column_for_condition(condition: str) -> str:
        """NREM column for NREM conditions, Wake column for Wake / NOD ones."""
        if "NREM" in condition:
            return "nrem_quantile_threshold"
        if "NOD" in condition or "Wake" in condition:
            return "wake_quantile_threshold"
        raise ValueError(
            f"Cannot infer quantile column for condition {condition!r}. "
            "Condition must contain 'NREM', 'NOD', or 'Wake'."
        )

    def _lookup_quantile_threshold(
        self,
        df: pd.DataFrame,
        subject: str,
        probe: str,
        structure: str,
        condition: str,
        source_filename: str,
    ) -> float:
        column = self._quantile_column_for_condition(condition)
        row = df[
            (df["subject"] == subject)
            & (df["probe"] == probe)
            & (df["structure_acronym"] == structure)
        ]
        if row.empty:
            raise KeyError(
                f"No {self.variant} threshold row for "
                f"({subject}, {probe}, {structure}) in {source_filename}"
            )
        value = row[column].iloc[0]
        if pd.isna(value):
            raise ValueError(
                f"{self.variant} {column} is NaN for "
                f"({subject}, {probe}, {structure}); populate "
                f"{self.thresholds_package}.data/{source_filename}"
            )
        return float(value)

    def get_quantile_threshold(
        self,
        subject: str,
        probe: str,
        structure: str,
        condition: str,
    ) -> float:
        """Return the (48h-optimized) quantile threshold for one row+condition.

        Reads the main ``quantile_thresholds.csv``. Used by full-recording
        detection. For the per-condition detection path use
        :meth:`get_per_condition_quantile_threshold`.
        """
        return self._lookup_quantile_threshold(
            self.load_quantile_thresholds(),
            subject,
            probe,
            structure,
            condition,
            "quantile_thresholds.csv",
        )

    def get_per_condition_quantile_threshold(
        self,
        subject: str,
        probe: str,
        structure: str,
        condition: str,
    ) -> float:
        """Return the per-condition-detection quantile threshold for one row+condition.

        Reads ``quantile_thresholds_per_condition.csv`` (falling back to the main
        table for variants without it). Used by the per-condition detection path
        so it is insulated from the 48h-optimized thresholds.
        """
        return self._lookup_quantile_threshold(
            self.load_per_condition_quantile_thresholds(),
            subject,
            probe,
            structure,
            condition,
            PER_CONDITION_THRESHOLDS_FILENAME,
        )
