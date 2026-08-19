"""Unit tests for cnpix_local_sleep.evaluation kernels, reconciliation, and rasterizer.

Pure-computation tests with synthetic arrays; no NFS/production data required.
"""

import numpy as np
import pandas as pd
import scipy.ndimage

from cnpix_local_sleep.evaluation import labels, metrics, rasterize


def _qc_reference(arr, *, min_component_size=50):
    """Slow O(n_labels x full_array) reference for qc_and_fix_labels equivalence.

    Mirrors the pre-optimization implementation (per-label full-array scan) so
    the optimized version can be checked against it.
    """
    struct = np.ones((3, 3), dtype=int)
    fixed = arr.copy()
    unique_labels = np.unique(arr)
    unique_labels = unique_labels[unique_labels > 0]
    next_label = int(unique_labels.max()) + 1 if len(unique_labels) > 0 else 1
    for lbl in unique_labels:
        mask = arr == lbl
        chunks_with_label = np.where(mask.any(axis=(1, 2)))[0]
        components, comp_imgs = [], {}
        for chunk_ix in chunks_with_label:
            ci_img, n = scipy.ndimage.label(mask[chunk_ix], structure=struct)
            comp_imgs[chunk_ix] = ci_img
            for cid in range(1, n + 1):
                components.append((chunk_ix, cid, ci_img, int((ci_img == cid).sum())))
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        keys = [(c, cid) for c, cid, _, _ in components]
        for i in range(len(chunks_with_label) - 1):
            ci, cj = chunks_with_label[i], chunks_with_label[i + 1]
            if cj != ci + 1:
                continue
            li, fj = comp_imgs[ci][:, -1], comp_imgs[cj][:, 0]
            for row in range(comp_imgs[ci].shape[0]):
                if li[row] == 0:
                    continue
                for d in (-1, 0, 1):
                    r = row + d
                    if 0 <= r < len(fj) and fj[r] > 0:
                        ra, rb = find((ci, li[row])), find((cj, fj[r]))
                        if ra != rb:
                            parent[rb] = ra
        groups = {}
        for idx, key in enumerate(keys):
            groups.setdefault(find(key), []).append(idx)
        sizes = {r: sum(components[i][3] for i in m) for r, m in groups.items()}
        if len(groups) <= 1:
            continue
        roots = sorted(groups, key=lambda r: sizes[r], reverse=True)
        largest = roots[0]
        small = [r for r in roots[1:] if sizes[r] < min_component_size]
        large = [r for r in roots if sizes[r] >= min_component_size or r is largest]
        for r in small:
            for idx in groups[r]:
                c, cid, img, _ = components[idx]
                fixed[c][img == cid] = 0
        if len(large) > 1:
            for r in large[1:]:
                for idx in groups[r]:
                    c, cid, img, _ = components[idx]
                    fixed[c][img == cid] = next_label
                next_label += 1
    return fixed


class TestPixelMetrics:
    def test_overlap_counts_and_rates(self):
        manual = np.zeros((2, 4, 10), np.int32)
        manual[0, 1, 2:5] = 7  # 3 manual-positive pixels
        predicted = np.zeros((2, 4, 10), np.int32)
        predicted[0, 1, 3:6] = 3  # 3 predicted-positive, 2 overlap

        m = metrics.compute_pixel_metrics(
            manual, predicted, np.array([0, 1]), np.ones(4, bool)
        )
        assert (m["TP"], m["FP"], m["FN"]) == (2, 1, 1)
        assert m["sensitivity"] == 2 / 3
        # TN = all-but the 4 touched pixels of 80 total -> 76; spec = 76/77
        assert m["TN"] == 80 - 4
        assert m["specificity"] == 76 / 77

    def test_perfect_and_empty(self):
        manual = np.zeros((1, 2, 5), np.int32)
        manual[0, 0, 1:3] = 1
        m_perfect = metrics.compute_pixel_metrics(
            manual, manual.copy(), np.array([0]), np.ones(2, bool)
        )
        assert m_perfect["sensitivity"] == 1.0 and m_perfect["FP"] == 0
        empty = np.zeros_like(manual)
        m_none = metrics.compute_pixel_metrics(
            manual, empty, np.array([0]), np.ones(2, bool)
        )
        assert m_none["sensitivity"] == 0.0 and m_none["specificity"] == 1.0


class TestEventMetrics:
    def test_event_keys_are_method_neutral(self):
        manual = np.zeros((1, 3, 6), np.int32)
        manual[0, 0, 1:3] = 5
        predicted = np.zeros((1, 3, 6), np.int32)
        predicted[0, 0, 2:4] = 9  # overlaps the single manual event
        ev = metrics.compute_event_metrics(
            manual, predicted, np.array([0]), np.ones(3, bool)
        )
        assert ev["n_manual_events"] == 1
        assert ev["n_predicted_events"] == 1
        assert ev["n_manual_detected"] == 1
        assert "n_predicted_unmatched" in ev
        flat = metrics.summarize_event_ious(ev)
        assert "median_event_iou" in flat and "per_event_ious" not in flat


class TestReconcile:
    def test_crops_to_common_top_left(self):
        a = np.ones((3, 5, 10), np.int32)
        b = np.ones((2, 4, 9), np.int32)
        ra, rb = labels.reconcile_to_common_grid(a, b)
        assert ra.shape == rb.shape == (2, 4, 9)

    def test_noop_when_equal(self):
        a = np.ones((2, 4, 10), np.int32)
        b = np.zeros((2, 4, 10), np.int32)
        ra, rb = labels.reconcile_to_common_grid(a, b)
        assert ra.shape == (2, 4, 10) and rb.shape == (2, 4, 10)


class TestSelectChunks:
    def test_labeled_vs_all(self):
        manual = np.zeros((4, 2, 3), np.int32)
        manual[1] = 1
        manual[3] = 2
        np.testing.assert_array_equal(
            labels.select_chunks(manual, "labeled"), np.array([1, 3])
        )
        np.testing.assert_array_equal(
            labels.select_chunks(manual, "all"), np.array([0, 1, 2, 3])
        )

    def test_invalid_mode(self):
        try:
            labels.select_chunks(np.zeros((1, 1, 1), np.int32), "bogus")
        except ValueError:
            return
        raise AssertionError("expected ValueError for invalid mode")


class TestQCEquivalence:
    def test_matches_reference_on_random_arrays(self):
        rng = np.random.default_rng(0)
        for _ in range(8):
            # Sparse multi-chunk instance labels with stray marks + splits.
            arr = np.zeros((6, 8, 12), np.int32)
            for lbl in range(1, rng.integers(3, 8)):
                c = int(rng.integers(0, 6))
                r0, s0 = int(rng.integers(0, 6)), int(rng.integers(0, 9))
                arr[c, r0 : r0 + 2, s0 : s0 + 3] = lbl
                if rng.random() < 0.5 and c + 2 < 6:  # non-adjacent second blob
                    arr[c + 2, r0, s0] = lbl
            fixed_opt, _ = labels.qc_and_fix_labels(arr, min_component_size=3)
            fixed_ref = _qc_reference(arr, min_component_size=3)
            # Optimization is behavior-preserving: arrays must be byte-identical
            # (same removals, same splits, same new-label IDs).
            np.testing.assert_array_equal(fixed_opt, fixed_ref)

    def test_clean_label_unchanged(self):
        arr = np.zeros((3, 5, 10), np.int32)
        arr[1, 1:3, 2:5] = 7
        fixed, viol = labels.qc_and_fix_labels(arr, min_component_size=3)
        np.testing.assert_array_equal(fixed, arr)
        assert viol == []


class TestRasterizer:
    def test_orientation_and_extent(self):
        # spc=10, 2 chunks; ts monotonic; 4 channels ascending depth.
        ts = np.arange(20, dtype=float)
        y = np.array([0.0, 100.0, 200.0, 300.0])
        offs = pd.DataFrame(
            {"start_time": [2.0], "end_time": [4.0], "lo": [100.0], "hi": [200.0]}
        )
        r = rasterize.rasterize_offs(offs, ts, y, (2, 4, 10))
        nz = set(map(tuple, np.argwhere(r > 0)))
        # y in [100,200] -> chan_idx {1,2} -> rows (4-1)-{1,2} = {2,1};
        # ts in [2,4] -> flat indices {2,3,4} (all in chunk 0).
        expected = {
            (0, row, s) for row in (1, 2) for s in (2, 3, 4)
        }
        assert nz == expected
        # unique positive event id assigned
        assert set(np.unique(r)) == {0, 1}

    def test_event_outside_window_is_empty(self):
        ts = np.arange(10, dtype=float)
        y = np.array([0.0, 100.0])
        offs = pd.DataFrame(
            {"start_time": [100.0], "end_time": [200.0], "lo": [0.0], "hi": [100.0]}
        )
        r = rasterize.rasterize_offs(offs, ts, y, (1, 2, 10))
        assert r.sum() == 0

    def test_row_count_mismatch_raises(self):
        ts = np.arange(10, dtype=float)
        y = np.array([0.0, 100.0, 200.0])  # 3 != 2 rows
        offs = pd.DataFrame(
            {"start_time": [1.0], "end_time": [2.0], "lo": [0.0], "hi": [100.0]}
        )
        try:
            rasterize.rasterize_offs(offs, ts, y, (1, 2, 10))
        except ValueError:
            return
        raise AssertionError("expected ValueError for row/depth mismatch")
