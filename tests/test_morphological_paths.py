"""Pin morphological path construction.

Two invariants:

1. Detection artifacts land under ``method=morphological``.
2. The annotation grid keeps its own ``method=tom-bugnon`` segment, and
   detection never writes there. The legacy tom-bugnon *detection* variant was
   deleted on 2026-08-11, but the grid identity survived it: the napari stacks
   and every manual OFF label are pinned to those y-coordinates, so Table 1's
   scoring still resolves through that path.
"""

from __future__ import annotations

import pytest


MUA_METHOD = "method=morphological"
ANNOTATION_GRID_METHOD = "method=tom-bugnon"


class TestMorphologicalPaths:
    def test_get_path_injects_mua_method(self):
        from cnpix_local_sleep.morphological.mua import files as mua_files

        path = mua_files.get_path(
            "off_df.parquet",
            subject="CNPIX15-Claude",
            probe="imec0",
            structure="PPC",
            condition="Early.NOD.Wake",
            detection_mode="spatial",
        )
        path_str = str(path)
        assert MUA_METHOD in path_str, (
            f"expected MUA method segment in {path_str!r}"
        )
        assert "probe=imec0" in path_str
        assert "structure=PPC" in path_str

    @pytest.mark.parametrize(
        "fn_name",
        [
            "get_channel_thresholds_path",
            "get_off_label_indices_path",
            "get_offs_path",
        ],
    )
    def test_detection_outputs_under_mua_method(self, fn_name: str):
        from cnpix_local_sleep.morphological.mua import files as mua_files

        path = str(
            getattr(mua_files, fn_name)(
                subject="CNPIX15-Claude",
                probe="imec0",
                structure="PPC",
                condition="Early.NOD.Wake",
                threshold_group=None,
            )
        )
        assert MUA_METHOD in path
        assert ANNOTATION_GRID_METHOD not in path, (
            f"{fn_name} must not write into the annotation grid tree"
        )


class TestAnnotationGrid:
    def test_preprocessed_ap_path_under_grid_method(self):
        """The annotation grid keeps its ``method=tom-bugnon`` segment.

        The function lives in ``cnpix_local_sleep.files`` rather than in a detection
        variant's files module precisely because it outlived one. The path
        itself must not move, or existing manual labels stop resolving.
        """
        from cnpix_local_sleep import files as op_files

        path = op_files.get_preprocessed_ap_path(
            subject="CNPIX15-Claude", probe="imec0"
        )
        path_str = str(path)
        assert ANNOTATION_GRID_METHOD in path_str
        assert MUA_METHOD not in path_str
        assert path_str.endswith("processed_ap.zarr")

    def test_grid_cohort_is_registered(self):
        """The 26-pair stack cohort survived the retirement of the tom variant."""
        from cnpix_local_sleep import sps_conf

        pairs = sps_conf.get_subject_probe_list(method="annotation-grid")
        assert len(pairs) == 26, f"expected 26 annotated pairs, got {len(pairs)}"


class TestSourceConfigWiring:
    def test_mua_source_config(self):
        from cnpix_local_sleep.morphological import mua
        from cnpix_local_sleep.morphological.mua import files as mua_files

        assert mua.SOURCE_CONFIG.variant == "morphological"
        assert mua.SOURCE_CONFIG.files_module is mua_files
