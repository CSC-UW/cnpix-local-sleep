"""Tests for qc_and_fix_labels in manual_validation."""

from __future__ import annotations

import numpy as np

from cnpix_local_sleep.morphological.manual_validation import qc_and_fix_labels


def _make_labels(*chunks: np.ndarray) -> np.ndarray:
    """Stack 2D chunk arrays into a (n_chunks, n_rows, n_cols) array."""
    return np.stack(chunks, axis=0).astype(np.int32)


class TestSingleChunkSingleComponent:
    """A label with one connected component in one chunk is fine."""

    def test_no_violation(self):
        chunk = np.zeros((5, 10), dtype=np.int32)
        chunk[1:4, 3:7] = 1
        labels = _make_labels(chunk)
        fixed, violations = qc_and_fix_labels(labels)
        assert violations == []
        np.testing.assert_array_equal(fixed, labels)


class TestDiagonalConnectivity:
    """Diagonal-only connections should NOT cause a split (8-conn)."""

    def test_diagonal_pixels_are_connected(self):
        chunk = np.zeros((5, 10), dtype=np.int32)
        # Two blocks connected only diagonally.
        chunk[1, 3] = 1
        chunk[2, 4] = 1
        labels = _make_labels(chunk)
        fixed, violations = qc_and_fix_labels(labels)
        assert violations == []
        np.testing.assert_array_equal(fixed, labels)

    def test_checkerboard_pattern_is_connected(self):
        chunk = np.zeros((5, 5), dtype=np.int32)
        chunk[0, 0] = 1
        chunk[1, 1] = 1
        chunk[2, 2] = 1
        labels = _make_labels(chunk)
        fixed, violations = qc_and_fix_labels(labels)
        assert violations == []


class TestGenuineSplit:
    """Two truly disconnected components in one chunk should split."""

    def test_two_disconnected_blobs(self):
        chunk = np.zeros((20, 40), dtype=np.int32)
        # Blob A: top-left (60 pixels, above threshold).
        chunk[0:6, 0:10] = 1
        # Blob B: bottom-right (80 pixels, above threshold).
        chunk[12:20, 30:40] = 1
        labels = _make_labels(chunk)
        fixed, violations = qc_and_fix_labels(labels)
        assert len(violations) == 1
        v = violations[0]
        assert v["original_label"] == 1
        assert v["n_split"] == 1
        assert v["n_removed"] == 0
        # Original label should be on the larger blob.
        assert (fixed[0, 12:20, 30:40] == 1).all()
        new_lbl = fixed[0, 0, 0]
        assert new_lbl != 1 and new_lbl != 0


class TestSmallIslandRemoval:
    """Tiny islands below min_component_size are removed, not split."""

    def test_island_removed(self):
        chunk = np.zeros((10, 20), dtype=np.int32)
        # Large blob.
        chunk[0:8, 0:15] = 1  # 120 pixels
        # Tiny island.
        chunk[9, 19] = 1  # 1 pixel
        labels = _make_labels(chunk)
        fixed, violations = qc_and_fix_labels(labels, min_component_size=50)
        assert len(violations) == 1
        v = violations[0]
        assert v["n_removed"] == 1
        assert v["n_split"] == 0
        assert v["removed_sizes"] == [1]
        # Island should be set to 0.
        assert fixed[0, 9, 19] == 0
        # Large blob untouched.
        assert (fixed[0, 0:8, 0:15] == 1).all()

    def test_custom_threshold(self):
        chunk = np.zeros((10, 20), dtype=np.int32)
        chunk[0:5, 0:10] = 1  # 50 pixels
        chunk[8:10, 18:20] = 1  # 4 pixels
        labels = _make_labels(chunk)
        # With threshold=5, the 4-pixel island is removed.
        fixed, violations = qc_and_fix_labels(labels, min_component_size=5)
        assert len(violations) == 1
        assert violations[0]["n_removed"] == 1
        assert fixed[0, 8, 18] == 0
        # With threshold=3, both survive and the smaller is split.
        fixed2, violations2 = qc_and_fix_labels(
            labels, min_component_size=3,
        )
        assert len(violations2) == 1
        assert violations2[0]["n_split"] == 1
        assert violations2[0]["n_removed"] == 0


class TestCrossChunkBoundary:
    """Labels spanning consecutive chunks with boundary connections."""

    def test_boundary_connected_no_violation(self):
        """Label at right edge of chunk 0, left edge of chunk 1."""
        c0 = np.zeros((5, 10), dtype=np.int32)
        c1 = np.zeros((5, 10), dtype=np.int32)
        c0[2, 9] = 1  # right edge
        c1[2, 0] = 1  # left edge, same row
        labels = _make_labels(c0, c1)
        fixed, violations = qc_and_fix_labels(labels)
        assert violations == []
        np.testing.assert_array_equal(fixed, labels)

    def test_boundary_diagonal_connected_no_violation(self):
        """Label at (row, last_col) in chunk 0 and (row+1, 0) in chunk 1."""
        c0 = np.zeros((5, 10), dtype=np.int32)
        c1 = np.zeros((5, 10), dtype=np.int32)
        c0[2, 9] = 1  # right edge, row 2
        c1[3, 0] = 1  # left edge, row 3 (diagonal neighbor)
        labels = _make_labels(c0, c1)
        fixed, violations = qc_and_fix_labels(labels)
        assert violations == []
        np.testing.assert_array_equal(fixed, labels)

    def test_boundary_not_connected_splits(self):
        """Label in two consecutive chunks but NOT at boundary."""
        c0 = np.zeros((10, 20), dtype=np.int32)
        c1 = np.zeros((10, 20), dtype=np.int32)
        c0[0:10, 0:6] = 1  # 60 px, left side of chunk 0
        c1[0:10, 14:20] = 1  # 60 px, right side of chunk 1
        labels = _make_labels(c0, c1)
        fixed, violations = qc_and_fix_labels(labels)
        assert len(violations) == 1
        assert violations[0]["n_split"] == 1

    def test_non_consecutive_chunks_split(self):
        """Label in chunks 0 and 2 (skipping 1) always splits."""
        c0 = np.zeros((5, 10), dtype=np.int32)
        c1 = np.zeros((5, 10), dtype=np.int32)
        c2 = np.zeros((5, 10), dtype=np.int32)
        c0[2, 9] = 1
        # c1 is empty for this label.
        c2[2, 0] = 1
        labels = _make_labels(c0, c1, c2)
        fixed, violations = qc_and_fix_labels(labels, min_component_size=1)
        assert len(violations) == 1
        assert violations[0]["n_split"] == 1

    def test_three_consecutive_chunks_chained(self):
        """Label spanning 3 consecutive chunks, all boundary-connected."""
        c0 = np.zeros((5, 10), dtype=np.int32)
        c1 = np.zeros((5, 10), dtype=np.int32)
        c2 = np.zeros((5, 10), dtype=np.int32)
        c0[2, 8:10] = 1  # extends to right edge
        c1[2, :] = 1  # spans full width
        c2[2, 0:3] = 1  # extends from left edge
        labels = _make_labels(c0, c1, c2)
        fixed, violations = qc_and_fix_labels(labels)
        assert violations == []


class TestMixedIslandAndEvent:
    """Large component + tiny island = island removed, no split."""

    def test_island_removed_event_kept(self):
        chunk = np.zeros((10, 20), dtype=np.int32)
        chunk[0:8, 0:15] = 1  # large event (120 px)
        chunk[9, 19] = 1  # tiny island (1 px)
        labels = _make_labels(chunk)
        fixed, violations = qc_and_fix_labels(labels, min_component_size=50)
        assert len(violations) == 1
        v = violations[0]
        assert v["n_removed"] == 1
        assert v["n_split"] == 0
        # Only one label remains.
        assert set(np.unique(fixed)) == {0, 1}


class TestNoLabels:
    """Empty label array produces no violations."""

    def test_all_zeros(self):
        labels = np.zeros((3, 5, 10), dtype=np.int32)
        fixed, violations = qc_and_fix_labels(labels)
        assert violations == []
        np.testing.assert_array_equal(fixed, labels)


class TestLargestGroupKeepsLabel:
    """When splitting, the largest group keeps the original label."""

    def test_largest_keeps_original(self):
        chunk = np.zeros((10, 30), dtype=np.int32)
        # Small blob (9 pixels).
        chunk[0:3, 0:3] = 1
        # Large blob (60 pixels).
        chunk[0:6, 20:30] = 1
        labels = _make_labels(chunk)
        fixed, violations = qc_and_fix_labels(
            labels, min_component_size=1,
        )
        assert len(violations) == 1
        # Large blob keeps original label 1.
        assert (fixed[0, 0:6, 20:30] == 1).all()
        # Small blob gets new label.
        new_lbl = fixed[0, 0, 0]
        assert new_lbl > 1