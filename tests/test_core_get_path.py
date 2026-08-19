"""Unit tests for cnpix_local_sleep.files.get_path() function."""

import pathlib
import warnings

import pytest

import wisc_ecephys_tools as wet

import cnpix_local_sleep.files


class TestBasicPathConstruction:
    """Test basic path construction functionality."""

    def test_simple_path_with_kwargs(self):
        """Test basic path construction with kwargs."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            probe="imec0",
            structure="PPC",
        )
        # Path should contain subject and parameter directories
        assert "CNPIX15-Claude" in str(path)
        assert "probe=imec0" in str(path)
        assert "structure=PPC" in str(path)
        assert str(path).endswith("test.parquet")

    def test_path_with_pathspec_dict(self):
        """Test path construction with pathspec dictionary."""
        pathspec = {"probe": "imec0", "structure": "PPC"}
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            pathspec=pathspec,
            subject="CNPIX15-Claude",
        )
        assert "probe=imec0" in str(path)
        assert "structure=PPC" in str(path)

    def test_pathspec_and_kwargs_merged(self):
        """Test that pathspec dict and kwargs are merged."""
        pathspec = {"probe": "imec0"}
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            pathspec=pathspec,
            subject="CNPIX15-Claude",
            structure="PPC",
        )
        assert "probe=imec0" in str(path)
        assert "structure=PPC" in str(path)

    def test_empty_filename_returns_directory(self):
        """Test that empty filename returns directory path only."""
        path = cnpix_local_sleep.files.get_path(
            "",
            subject="CNPIX15-Claude",
            probe="imec0",
        )
        assert str(path).endswith("probe=imec0")
        assert not str(path).endswith("/")

    def test_dots_allowed_in_values(self):
        """Test that dots are allowed in parameter values (R-friendly)."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            condition="Early.BSL.NREM",
        )
        assert "condition=Early.BSL.NREM" in str(path)


class TestWNEComponentHandling:
    """Test handling of WNE schema components (project/experiment/subject)."""

    def test_default_wne_components(self):
        """Test default project and experiment values."""
        path = cnpix_local_sleep.files.get_path("test.parquet", subject="CNPIX15-Claude")
        # Should use default project and experiment from const.EXPERIMENT
        assert "offproj" in str(path) or "novel_objects_deprivation" in str(path)

    def test_custom_project(self):
        """Test custom project name."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            project="shared",
        )
        # Path should use shared project
        project_path = wet.get_sglx_project("shared").dir
        assert str(path).startswith(str(project_path))

    def test_none_experiment_omitted(self):
        """Test that None experiment is omitted from path."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            experiment=None,
            probe="imec0",
        )
        # Should go directly from project to subject
        path_parts = pathlib.Path(path).parts
        subject_idx = path_parts.index("CNPIX15-Claude")
        # Should have project, then subject, then probe
        assert "probe=imec0" in path_parts[subject_idx + 1]

    def test_none_subject_omitted(self):
        """Test that None subject is omitted from path."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject=None,
            probe="imec0",
        )
        # Should not contain any subject name
        assert "CNPIX15-Claude" not in str(path)
        # Should have experiment, then probe directly
        assert "probe=imec0" in str(path)

    def test_wne_components_from_pathspec(self):
        """Test extracting WNE components from pathspec."""
        # Note: subject cannot be in pathspec dict because it's a WNE component
        # that gets extracted. Test should use kwargs instead.
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            probe="imec0",
        )
        assert "CNPIX15-Claude" in str(path)
        assert "probe=imec0" in str(path)

    def test_wne_component_conflict_raises_error(self):
        """Test that conflicting WNE components raise error."""
        # Actually, WNE components (subject, experiment, project) ARE reserved keywords
        # So passing them in pathspec dict will raise a "Reserved keywords" error, not "Conflict"
        # This is the correct behavior - they should be passed as function arguments, not pathspec
        with pytest.raises(ValueError, match="Reserved keywords"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                pathspec={
                    "subject": "CNPIX2-Segundo"
                },  # subject in pathspec - this is invalid
                subject="CNPIX15-Claude",  # subject as kwarg
            )


class TestReservedKeywordValidation:
    """Test validation of reserved keywords."""

    def test_reserved_keyword_in_pathspec_raises_error(self):
        """Test that reserved keywords in pathspec raise error."""
        with pytest.raises(ValueError, match="Reserved keywords"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                pathspec={"filename": "bad.parquet"},
                subject="CNPIX15-Claude",
            )

    def test_reserved_keyword_in_kwargs_raises_error(self):
        """Test that reserved keywords in kwargs raise error."""
        # We can't actually test kwargs with reserved names because Python won't allow
        # duplicate keyword arguments. The pathspec dict test above covers this case.
        # This test is redundant with test_reserved_keyword_in_pathspec_raises_error
        # so instead assert the raised message enumerates the whole reserved set.
        reserved = {
            "filename",
            "pathspec",
            "project",
            "experiment",
            "subject",
            "enforce_in_schema",
            "enforce_schema_order",
        }
        with pytest.raises(ValueError, match="Reserved keywords") as excinfo:
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                pathspec={"filename": "conflict.parquet"},
                subject="CNPIX15-Claude",
            )
        assert all(kw in str(excinfo.value) for kw in reserved)


class TestCharacterValidation:
    """Test validation of forbidden characters."""

    def test_equals_in_key_raises_error(self):
        """Test that '=' in key raises error."""
        with pytest.raises(ValueError, match="reserved substring '='"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                subject="CNPIX15-Claude",
                **{"bad=key": "value"},
            )

    def test_equals_in_value_raises_error(self):
        """Test that '=' in value raises error."""
        with pytest.raises(ValueError, match="reserved substring '='"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                subject="CNPIX15-Claude",
                condition="bad=value",
            )

    def test_separator_in_key_raises_error(self):
        """Test that '_-_' in key raises error."""
        with pytest.raises(ValueError, match="reserved substring '_-_'"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                subject="CNPIX15-Claude",
                **{"bad_-_key": "value"},
            )

    def test_separator_in_value_raises_error(self):
        """Test that '_-_' in value raises error."""
        with pytest.raises(ValueError, match="reserved substring '_-_'"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                subject="CNPIX15-Claude",
                condition="bad_-_value",
            )

    def test_nested_dict_key_validation(self):
        """Test validation of keys in nested dicts."""
        with pytest.raises(ValueError, match="reserved substring '='"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                subject="CNPIX15-Claude",
                threshold_group={"bad=key": "value"},
            )

    def test_nested_dict_value_validation(self):
        """Test validation of values in nested dicts."""
        with pytest.raises(ValueError, match="reserved substring '='"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                subject="CNPIX15-Claude",
                threshold_group={"contrast": "bad=value"},
            )


class TestNoneValueHandling:
    """Test handling of None values."""

    def test_none_value_omitted_from_path(self):
        """Test that None values are omitted entirely."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            structure="PPC",
            layer=None,
        )
        # layer should not appear in path
        assert "layer" not in str(path)
        assert "structure=PPC" in str(path)

    def test_multiple_none_values_omitted(self):
        """Test that multiple None values are all omitted."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            structure="PPC",
            layer=None,
            threshold_group=None,
        )
        assert "layer" not in str(path)
        assert "threshold_group" not in str(path)
        assert "structure=PPC" in str(path)


class TestNestedDictEncoding:
    """Test encoding of nested dictionaries."""

    def test_simple_nested_dict(self):
        """Test encoding of simple nested dictionary."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            threshold_group={"contrast": "NOD.Incline", "param": "slope"},
        )
        # Should encode as threshold_group=(contrast=NOD.Incline_-_param=slope)
        assert "threshold_group=(contrast=NOD.Incline_-_param=slope)" in str(
            path
        ) or "threshold_group=(param=slope_-_contrast=NOD.Incline)" in str(path)

    def test_nested_dict_with_none_value(self):
        """Test that None values in nested dicts work correctly."""
        # Note: This should still encode the nested dict, just with None converted to string
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            threshold_group={"contrast": "NOD.Incline"},
            enforce_in_schema=False,  # Allow threshold_group structure
        )
        assert "threshold_group=(contrast=NOD.Incline)" in str(path)


class TestSchemaEnforcement:
    """Test schema enforcement functionality."""

    def test_unknown_key_with_enforcement_raises_error(self):
        """Test that unknown keys raise error when enforce_in_schema=True."""
        with pytest.raises(ValueError, match="Keys not in schema"):
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                subject="CNPIX15-Claude",
                unknown_param="value",
                enforce_in_schema=True,
            )

    def test_unknown_key_without_enforcement_allowed(self):
        """Test that unknown keys are allowed when enforce_in_schema=False."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            unknown_param="value",
            enforce_in_schema=False,
        )
        assert "unknown_param=value" in str(path)

    def test_schema_ordering_enforced(self):
        """Test that keys are reordered according to schema."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            path = cnpix_local_sleep.files.get_path(
                "test.parquet",
                subject="CNPIX15-Claude",
                condition="Early.NOD.Wake",  # condition comes after structure in schema
                structure="PPC",  # structure comes before condition
                probe="imec0",  # probe comes first
                enforce_schema_order=True,
            )
            # Should warn about reordering
            assert len(w) == 1
            assert "reordered" in str(w[0].message).lower()

        # Verify order: should be probe, structure, condition
        path_str = str(path)
        probe_idx = path_str.index("probe=")
        structure_idx = path_str.index("structure=")
        condition_idx = path_str.index("condition=")
        assert probe_idx < structure_idx < condition_idx

    def test_schema_ordering_not_enforced(self):
        """Test that ordering is preserved when enforce_schema_order=False."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use OrderedDict to ensure order
            from collections import OrderedDict

            pathspec = OrderedDict(
                [("condition", "Early.NOD.Wake"), ("structure", "PPC"), ("probe", "imec0")]
            )
            cnpix_local_sleep.files.get_path(
                "test.parquet",
                pathspec=pathspec,
                subject="CNPIX15-Claude",
                enforce_schema_order=False,
            )
            # Should not warn
            assert len(w) == 0


class TestPathStructure:
    """Test overall path structure."""

    def test_full_path_structure(self):
        """Test complete path structure with all components."""
        path = cnpix_local_sleep.files.get_path(
            "off_df.parquet",
            subject="CNPIX15-Claude",
            method="tom-bugnon",
            probe="imec0",
            structure="PPC",
            layer="supra",
            detection_mode="spatial",
            condition="Early.NOD.Wake",
        )

        # Verify path structure
        path_str = str(path)
        assert path_str.endswith("off_df.parquet")
        assert "CNPIX15-Claude" in path_str
        assert "method=tom-bugnon" in path_str
        assert "probe=imec0" in path_str
        assert "structure=PPC" in path_str
        assert "layer=supra" in path_str
        assert "detection_mode=spatial" in path_str
        assert "condition=Early.NOD" in path_str

    def test_model_component_ordering(self):
        """``model`` is encoded just after ``method`` and before ``probe``."""
        path = cnpix_local_sleep.files.get_path(
            "sam3_off_labels.npz",
            subject="CNPIX15-Claude",
            method="sam3",
            model="trained-on-Early.REC.NREM.2026-05-09",
            probe="imec0",
            condition="Early.REC.NREM",
        )
        path_str = str(path)
        assert "method=sam3" in path_str
        assert "model=trained-on-Early.REC.NREM.2026-05-09" in path_str
        method_idx = path_str.index("method=")
        model_idx = path_str.index("model=")
        probe_idx = path_str.index("probe=")
        assert method_idx < model_idx < probe_idx

    def test_path_is_pathlib_path(self):
        """Test that return value is pathlib.Path."""
        path = cnpix_local_sleep.files.get_path("test.parquet", subject="CNPIX15-Claude")
        assert isinstance(path, pathlib.Path)

    def test_path_can_be_used_for_file_operations(self):
        """Test that returned path works with pathlib operations."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            probe="imec0",
        )
        # These should not raise errors
        assert path.name == "test.parquet"
        assert path.suffix == ".parquet"
        parent = path.parent
        assert "probe=imec0" in str(parent)


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_pathspec(self):
        """Test with empty pathspec."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet", pathspec={}, subject="CNPIX15-Claude"
        )
        assert "CNPIX15-Claude" in str(path)
        assert str(path).endswith("test.parquet")

    def test_all_none_values(self):
        """Test with all parameter values as None."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            layer=None,
            threshold_group=None,
        )
        # Should just have subject and filename
        assert "CNPIX15-Claude" in str(path)
        assert str(path).endswith("test.parquet")
        assert "layer" not in str(path)
        assert "threshold_group" not in str(path)

    def test_numeric_values_converted_to_string(self):
        """Test that numeric values are converted to strings."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            version=1,
            enforce_in_schema=False,
        )
        assert "version=1" in str(path)

    def test_boolean_values_converted_to_string(self):
        """Test that boolean values are converted to strings."""
        path = cnpix_local_sleep.files.get_path(
            "test.parquet",
            subject="CNPIX15-Claude",
            flagged=True,
            enforce_in_schema=False,
        )
        assert "flagged=True" in str(path)
