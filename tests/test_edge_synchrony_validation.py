"""Tests for the OFF edge-synchrony estimator validation tools."""

import numpy as np
import pandas as pd
import pytest

from cnpix_local_sleep.morphological import edge_synchrony_validation as esv


# -------------------- The analytic MAD floor --------------------


def test_mad_zero_min_ties_matches_brute_force():
    """``n // 2 + 1`` really is the tie count that forces MAD to zero."""
    rng = np.random.default_rng(0)
    for n in range(2, 20):
        required = int(esv.mad_zero_min_ties(n))
        # One fewer tie than required must be able to leave MAD > 0.
        below = np.concatenate(
            [np.zeros(required - 1), rng.uniform(1, 2, size=n - required + 1)]
        )
        assert np.median(np.abs(below - np.median(below))) > 0
        # The required number of ties forces MAD to zero, whatever the rest do.
        at = np.concatenate(
            [np.zeros(required), rng.uniform(1, 2, size=n - required)]
        )
        assert np.median(np.abs(at - np.median(at))) == 0


def test_mad_zero_forced_max_n():
    assert esv.mad_zero_forced_max_n(run_length=4) == 7
    assert esv.mad_zero_forced_max_n(run_length=3) == 5
    assert esv.mad_zero_forced_max_n(run_length=1) == 1


def test_production_clean_opts_match_shipped_yaml():
    esv.assert_production_clean_opts()


def test_simulated_floor_matches_analytic_prediction():
    """The floor gate: two independent routes to the same cliff at n = 7.

    With a latent edge dispersion far larger than the sampling interval, MAD must
    still be identically zero for every event spanning at most
    ``mad_zero_forced_max_n()`` channels, and must become non-zero above it.
    """
    sim = esv.simulate_detector_edges(
        [6, 7, 9, 13], sigma_ms=20.0, duration_ms=80.0, n_events=40, seed=3
    )
    by_n = sim.groupby("n_channels")["onset_mad"].apply(lambda s: (s == 0).mean())
    forced = esv.mad_zero_forced_max_n()
    for n_channels, p_zero in by_n.items():
        if n_channels <= forced:
            assert p_zero == 1.0, f"n={n_channels} should be floored"
        elif n_channels >= forced + 4:
            assert p_zero < 1.0, f"n={n_channels} should escape the floor"

    # And the same for the offset edge, by the mirrored argument.
    off_zero = sim.groupby("n_channels")["offset_mad"].apply(
        lambda s: (s == 0).mean()
    )
    assert all(off_zero.loc[[6, 7]] == 1.0)


def test_simulated_mad_grows_with_span_at_fixed_latent_dispersion():
    """The mechanical claim itself: size moves MAD with the latent value pinned."""
    sim = esv.simulate_detector_edges(
        [6, 11, 26], sigma_ms=8.0, duration_ms=80.0, n_events=60, seed=4
    )
    means = sim.groupby("n_channels")["onset_mad"].mean()
    assert means.loc[6] == 0.0
    assert means.loc[26] > means.loc[11] > means.loc[6]


# -------------------- The full trace chain, filters included --------------------


def test_production_trace_opts_match_shipped_yaml():
    esv.assert_production_trace_opts()


def test_trace_level_simulation_recovers_the_requested_events():
    """Sanity: the synthetic envelope survives the filters as one event per draw."""
    sim = esv.simulate_trace_level_edges(
        [8, 16], sigma_ms=6.0, duration_ms=120.0, n_events=40, seed=5
    )
    intact = sim[sim["n_channels"] == sim["requested_n_channels"]]
    assert (intact.groupby("requested_n_channels").size() == 40).all()
    # And the event's duration comes back roughly as drawn.
    assert intact["median_duration"].mean() * 1e3 == pytest.approx(120, abs=15)


def test_trace_level_simulation_reproduces_the_floor():
    """The filters do not release the structuring element's floor."""
    sim = esv.simulate_trace_level_edges(
        [6, 7, 16], sigma_ms=12.0, duration_ms=120.0, n_events=40, seed=6
    )
    p_zero = sim.groupby("n_channels")["onset_mad"].apply(lambda s: (s == 0).mean())
    forced = esv.mad_zero_forced_max_n()
    assert all(p_zero.loc[[n for n in p_zero.index if n <= forced]] == 1.0)


def test_median_filter_is_applied_and_is_edge_preserving():
    """Both halves of the claim that lets the duration test stand.

    The 30 ms temporal median filter really runs (a dip shorter than its window
    is annihilated by it and survives without it), and yet for events of realistic
    length it leaves the recovered edges bit-identical, which is the textbook
    step-preserving property of a median filter. It therefore cannot be a route
    from event duration to MAD.
    """
    no_clean: esv.CleanOpts = {
        "n_samples_connect": None, "n_samples_clean": None,
        "n_channels_clean": None, "n_channels_connect": None,
    }
    kwargs = dict(sigma_ms=2.0, n_events=12, seed=9, clean_opts=no_clean)

    # Applied: a 20 ms dip is shorter than the 30 ms window and does not survive.
    short_on = esv.simulate_trace_level_edges([16], duration_ms=20.0, **kwargs)
    short_off = esv.simulate_trace_level_edges(
        [16], duration_ms=20.0, apply_median=False, **kwargs
    )
    assert (short_on["span"] >= 200).sum() == 0
    assert (short_off["span"] >= 200).sum() == 12

    # Edge-preserving: at a realistic duration it changes nothing measurable.
    long_on = esv.simulate_trace_level_edges([16], duration_ms=120.0, **kwargs)
    long_off = esv.simulate_trace_level_edges(
        [16], duration_ms=120.0, apply_median=False, **kwargs
    )
    for column in ("onset_mad", "offset_mad", "start_time", "median_duration"):
        assert np.allclose(long_on[column], long_off[column]), column


def test_trace_level_mad_is_flat_in_duration_at_fixed_size():
    """The mediator test: no mechanical route from duration to MAD.

    Latent edge dispersion is pinned; only duration moves, over more than the
    empirical range. Recovered MAD must stay put. Empirically MAD *triples* across
    this range at fixed size, so that relation is generative and duration must not
    be conditioned on.
    """
    means = {}
    for duration_ms in (40.0, 120.0, 240.0):
        sim = esv.simulate_trace_level_edges(
            [26], sigma_ms=6.0, duration_ms=duration_ms, n_events=250, seed=7
        )
        intact = sim[sim["n_channels"] == 26]
        means[duration_ms] = intact["onset_mad"].mean() * 1e3

    values = np.array(list(means.values()))
    assert values.min() > 1.0, means          # the floor has released
    # A sixfold change in duration moves the mean by under a quarter of a sample.
    assert np.ptp(values) < 0.5, means


# -------------------- Direct standardization --------------------


def _toy_events(mix_by_condition, value_by_stratum, n_per_cell=200):
    """Events where the value depends *only* on stratum, not on condition."""
    rng = np.random.default_rng(7)
    rows = []
    for condition, weights in mix_by_condition.items():
        for stratum, weight in weights.items():
            for _ in range(int(round(weight * n_per_cell))):
                rows.append(
                    {
                        "combo": "s|p|st",
                        "condition": condition,
                        "stratum": stratum,
                        "value": value_by_stratum[stratum]
                        + rng.normal(0, 1e-9),
                    }
                )
    return pd.DataFrame(rows)


def test_standardize_single_stratum_reduces_to_raw_mean():
    events = _toy_events({"A": {"x": 1.0}, "B": {"x": 1.0}}, {"x": 5.0})
    out = esv.standardize_cell_means(events, "value", ["stratum"])
    assert np.allclose(out["standardized"], out["raw"])


def test_standardize_is_noop_when_composition_already_balanced():
    mix = {"A": {"x": 0.5, "y": 0.5}, "B": {"x": 0.5, "y": 0.5}}
    events = _toy_events(mix, {"x": 1.0, "y": 9.0})
    out = esv.standardize_cell_means(events, "value", ["stratum"])
    assert np.allclose(out["standardized"], out["raw"], atol=1e-6)


def test_standardize_removes_a_pure_composition_effect():
    """Raw means differ only because the stratum mix moved; standardized must not."""
    mix = {"A": {"x": 0.9, "y": 0.1}, "B": {"x": 0.1, "y": 0.9}}
    events = _toy_events(mix, {"x": 1.0, "y": 9.0})
    out = esv.standardize_cell_means(events, "value", ["stratum"])
    raw = out.set_index("condition")["raw"]
    std = out.set_index("condition")["standardized"]
    assert abs(raw["B"] - raw["A"]) > 5.0
    assert abs(std["B"] - std["A"]) < 1e-6
    assert (out["frac_dropped"] == 0).all()


def test_standardize_common_support_drops_unshared_strata():
    mix = {"A": {"x": 1.0}, "B": {"x": 0.5, "y": 0.5}}
    events = _toy_events(mix, {"x": 1.0, "y": 9.0})
    out = esv.standardize_cell_means(events, "value", ["stratum"])
    # Stratum "y" is absent from condition A, so it leaves the comparison.
    assert np.allclose(out["standardized"], 1.0)
    dropped = out.set_index("condition")["frac_dropped"]
    assert dropped["A"] == 0.0
    assert dropped["B"] == pytest.approx(0.5, abs=0.05)


# -------------------- Regression standardization (g-computation) --------------------


def _toy_gcomp(effect, confound_shift, n=4000, n_combos=4, seed=0):
    """value depends non-linearly on size, plus a true condition effect."""
    rng = np.random.default_rng(seed)
    frames = []
    for combo in range(n_combos):
        for condition, shift, eff in (("A", 0, 0.0), ("B", confound_shift, effect)):
            n_channels = rng.integers(6, 30, size=n) + shift
            value = (
                np.where(n_channels <= 7, 0.0, 3 * np.log(n_channels))
                + eff
                + rng.normal(0, 1, n)
            )
            frames.append(pd.DataFrame({
                "combo": f"c{combo}", "condition": condition,
                "n_channels": n_channels, "value": value,
                "log_median_duration": rng.normal(0, 1, n),
                "start_time": np.sort(rng.uniform(0, 3600, n)),
            }))
    return pd.concat(frames, ignore_index=True)


def _contrast(out, column="standardized"):
    means = out.groupby("condition")[column].mean()
    return means["B"] - means["A"]


def test_gcomp_recovers_effect_through_a_nonlinear_size_confound():
    events = _toy_gcomp(effect=2.0, confound_shift=8)
    out = esv.standardize_by_regression(events, "value")
    assert _contrast(out, "raw") > 3.0          # confounded
    assert _contrast(out) == pytest.approx(2.0, abs=0.2)


def test_gcomp_leaves_an_unconfounded_contrast_alone():
    events = _toy_gcomp(effect=2.0, confound_shift=0)
    out = esv.standardize_by_regression(events, "value")
    assert _contrast(out) == pytest.approx(_contrast(out, "raw"), abs=0.15)


def test_gcomp_removes_a_pure_composition_effect():
    """No true effect, only a size shift: the adjusted contrast must be ~zero."""
    events = _toy_gcomp(effect=0.0, confound_shift=8)
    out = esv.standardize_by_regression(events, "value")
    assert abs(_contrast(out, "raw")) > 1.0
    assert abs(_contrast(out)) < 0.2


def test_gcomp_needs_no_common_support():
    """The wake case: conditions barely overlap in size, yet nothing is dropped."""
    events = _toy_gcomp(effect=1.0, confound_shift=20)
    out = esv.standardize_by_regression(events, "value")
    assert set(out["condition"]) == {"A", "B"}
    assert (out["n_events"] == 4000).all()


def test_gcomp_bootstrap_reports_an_interval_covering_the_estimate():
    events = _toy_gcomp(effect=2.0, confound_shift=8, n=1500, n_combos=2)
    out = esv.standardize_by_regression(events, "value", n_boot=25, seed=1)
    assert {"se", "ci_lo", "ci_hi"} <= set(out.columns)
    assert (out["se"] > 0).all()
    assert ((out["ci_lo"] <= out["standardized"])
            & (out["standardized"] <= out["ci_hi"])).all()


def test_gcomp_uses_exact_size_levels_when_the_combo_can_afford_them():
    events = _toy_gcomp(effect=1.0, confound_shift=0)
    out = esv.standardize_by_regression(
        events, "value", size_term="per_combo_factor"
    )
    assert (out["size_coding"] == "exact").all()


def test_gcomp_falls_back_to_quantile_bins_when_events_are_sparse():
    """A wake-sized combo cannot afford a parameter per channel count."""
    events = _toy_gcomp(effect=1.0, confound_shift=0, n=60, n_combos=1)
    out = esv.standardize_by_regression(
        events, "value", size_term="per_combo_factor"
    )
    assert out["size_coding"].str.startswith("quantile").all()
    assert (out["n_size_bins"] <= 6).all()


# -------------------- The shared size-adjustment curve --------------------


def _toy_shared(
    amplitudes=(1.0, 2.0, 0.5), n=3000, effect=0.0, noise=0.4, seed=0
):
    """One shape, per-combo amplitudes, and optionally a real condition effect."""
    rng = np.random.default_rng(seed)
    frames = []
    for combo, amplitude in enumerate(amplitudes):
        for condition, shift in (("A", 0.0), ("B", effect)):
            n_channels = rng.integers(6, 30, size=n)
            value = (
                amplitude * _toy_shape(n_channels) + shift + rng.normal(0, noise, n)
            )
            frames.append(pd.DataFrame({
                "combo": f"c{combo}", "condition": condition,
                "n_channels": n_channels, "value": value,
                "start_time": np.sort(rng.uniform(0, 3600, n)),
            }))
    events = pd.concat(frames, ignore_index=True)
    events["unit"] = events["combo"]
    events["cell"] = events["combo"] + "@" + events["condition"]
    return events


def _toy_shape(n_channels):
    """A floored, saturating shape, in the spirit of the real MAD-vs-size curve."""
    above_floor = np.clip(np.asarray(n_channels) - 6.0, 1.0, None)
    return np.where(np.asarray(n_channels) <= 7, 0.0, np.log(above_floor))


def test_shared_curve_recovers_an_injected_shape_and_amplitudes():
    events = _toy_shared(amplitudes=(1.0, 2.0, 0.5))
    fitted = esv.fit_shared_size_curve(events, "value", "unit", "cell")

    grid = np.arange(6, 30)
    truth = _toy_shape(grid)
    assert np.corrcoef(fitted.evaluate(grid), truth)[0, 1] > 0.999
    # The (shape, amplitude) split is identified only up to a common constant, so
    # compare the amplitudes against each other rather than to their raw values.
    ratios = fitted.unit_lambda / fitted.unit_lambda["c0"]
    assert ratios["c1"] == pytest.approx(2.0, rel=0.05)
    assert ratios["c2"] == pytest.approx(0.5, rel=0.05)


def test_shared_curve_is_blind_to_condition_effects():
    """Cell means are profiled out, so a condition effect cannot enter the curve."""
    flat = esv.fit_shared_size_curve(_toy_shared(effect=0.0), "value", "unit", "cell")
    shifted = esv.fit_shared_size_curve(
        _toy_shared(effect=25.0), "value", "unit", "cell"
    )
    grid = np.arange(6, 30)
    assert np.allclose(flat.evaluate(grid), shifted.evaluate(grid), atol=0.02)


def test_gcomp_shared_curve_removes_a_pure_composition_effect():
    events = _toy_gcomp(effect=0.0, confound_shift=8)
    out = esv.standardize_by_regression(events, "value")
    assert abs(_contrast(out, "raw")) > 1.0
    assert abs(_contrast(out)) < 0.2
    assert out["size_coding"].str.startswith("shared_curve").all()


def test_gcomp_shared_curve_recovers_a_real_effect():
    events = _toy_gcomp(effect=2.0, confound_shift=8)
    out = esv.standardize_by_regression(events, "value")
    assert _contrast(out) == pytest.approx(2.0, abs=0.2)


def test_gcomp_shared_curve_keeps_combos_too_small_to_fit_their_own_amplitude():
    """The regression guard: a small combo is adjusted, not silently dropped.

    The per-combo factor needs enough events to spend a parameter per channel count
    and drops anything below ``min_events``, which is what shrank the wake panel from
    42 cells to 17. One shared curve costs one parameter, and a combo too small even
    for that is given the pooled amplitude rather than being discarded.
    """
    events = _toy_shared(amplitudes=(1.0, 1.0), n=1500)
    tiny = _toy_shared(amplitudes=(1.0,), n=12, seed=4)
    tiny["combo"] = "small"
    events = pd.concat([events, tiny], ignore_index=True)

    out = esv.standardize_by_regression(events, "value")
    assert set(out["combo"]) == {"c0", "c1", "small"}
    coding = out.set_index("combo")["size_coding"]
    assert coding.loc["small"].eq("shared_curve[pooled]").all()
    assert coding.loc["c0"].eq("shared_curve[free]").all()
    assert out["standardized"].notna().all()

    dropped = esv.standardize_by_regression(
        events, "value", size_term="per_combo_factor"
    )
    assert "small" not in set(dropped["combo"])


def test_gcomp_shared_curve_free_amplitude_beats_a_borrowed_one():
    """Sharing the shape is right; sharing the amplitude is not.

    Only the second combo's conditions differ in size, and its size effect is three
    times the first's. A free amplitude removes the resulting confound; forcing the
    pooled amplitude on it under-corrects, which is the failure mode that flips the
    wake contrast when the shared curve is applied additively.
    """
    rng = np.random.default_rng(2)
    frames = []
    for combo, (amplitude, shift) in enumerate(((1.0, 0), (3.0, 10))):
        for condition in ("A", "B"):
            n_channels = rng.integers(6, 30, size=2000) + (
                shift if condition == "B" else 0
            )
            frames.append(pd.DataFrame({
                "combo": f"c{combo}", "condition": condition,
                "n_channels": n_channels,
                "value": amplitude * _toy_shape(n_channels) + rng.normal(0, 0.4, 2000),
                "start_time": np.sort(rng.uniform(0, 3600, 2000)),
            }))
    events = pd.concat(frames, ignore_index=True)

    free = esv.standardize_by_regression(events, "value")
    borrowed = esv.standardize_by_regression(
        events, "value", free_lambda_min_events=10**9
    )
    confounded = free[free["combo"] == "c1"]
    assert abs(_contrast(confounded, "raw")) > 1.0
    assert abs(_contrast(confounded)) < 0.15
    assert abs(_contrast(borrowed[borrowed["combo"] == "c1"])) > 0.4


def test_gcomp_positivity_is_one_under_shared_support_and_falls_when_it_splits():
    shared = esv.standardize_by_regression(
        _toy_gcomp(effect=1.0, confound_shift=0), "value"
    )
    assert shared["positivity"].min() == pytest.approx(1.0)
    # Disjoint sizes: each condition is predicted over levels it never occupies.
    split = esv.standardize_by_regression(
        _toy_gcomp(effect=1.0, confound_shift=30), "value"
    )
    assert split["positivity"].max() < 0.75


def test_gcomp_does_not_adjust_away_a_mediated_effect_by_default():
    """Duration is a mediator, so conditioning on it would erase a real effect.

    Here condition acts on the response *through* duration and nowhere else, which
    is what the trace-level simulation implies for MAD: no mechanical route from
    duration to MAD, but a strong empirical one. The default specification must
    return the full effect; adding duration as a covariate must destroy it. That
    contrast is the reason duration is not a default covariate.
    """
    rng = np.random.default_rng(3)
    n = 4000
    frames = []
    for combo in range(4):
        for condition, shift in (("A", 0.0), ("B", 5.0)):
            n_channels = rng.integers(6, 30, size=n)
            duration = shift + rng.normal(0, 1, n)
            value = (
                0.4 * duration                      # condition acts only via here
                + np.where(n_channels <= 7, 0.0, 3 * np.log(n_channels))
                + rng.normal(0, 1, n)
            )
            frames.append(pd.DataFrame({
                "combo": f"c{combo}", "condition": condition,
                "n_channels": n_channels, "value": value,
                "log_median_duration": duration,
                "start_time": np.sort(rng.uniform(0, 3600, n)),
            }))
    events = pd.concat(frames, ignore_index=True)

    default = esv.standardize_by_regression(events, "value")
    adjusted = esv.standardize_by_regression(
        events, "value", covariates=("log_median_duration",)
    )
    assert _contrast(default) == pytest.approx(0.4 * 5.0, abs=0.15)
    assert abs(_contrast(adjusted)) < 0.15


# -------------------- Event-level covariate model --------------------


def _toy_event_level(effect, confound_shift, n=4000, n_combos=6):
    """value = effect * (condition == B) + 0.5 * n_channels + noise."""
    rng = np.random.default_rng(11)
    rows = []
    for combo in range(n_combos):
        for condition, shift in (("A", 0.0), ("B", confound_shift)):
            n_channels = rng.integers(6, 30, size=n) + int(shift)
            value = (
                effect * (condition == "B")
                + 0.5 * n_channels
                + rng.normal(0, 1.0, size=n)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "combo": f"combo{combo}",
                        "condition": condition,
                        "n_channels": n_channels,
                        "value": value,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_fit_event_level_recovers_injected_effect_despite_confounding():
    events = _toy_event_level(effect=2.0, confound_shift=6)
    _, adjusted = esv.fit_event_level(events, "value", reference="A")
    _, unadjusted = esv.fit_event_level(
        events, "value", reference="A", n_channel_factor=False
    )
    adj = adjusted.set_index("contrast").loc["B", "estimate"]
    unadj = unadjusted.set_index("contrast").loc["B", "estimate"]
    assert adj == pytest.approx(2.0, abs=0.15)
    # Without the covariate the size shift inflates the estimate substantially.
    assert unadj > 4.0


def test_fit_event_level_agrees_with_raw_when_composition_balanced():
    events = _toy_event_level(effect=2.0, confound_shift=0)
    _, adjusted = esv.fit_event_level(events, "value", reference="A")
    _, unadjusted = esv.fit_event_level(
        events, "value", reference="A", n_channel_factor=False
    )
    adj = adjusted.set_index("contrast").loc["B", "estimate"]
    unadj = unadjusted.set_index("contrast").loc["B", "estimate"]
    assert adj == pytest.approx(unadj, abs=0.1)


# -------------------- Mixture decomposition --------------------


def test_decompose_mixture_reduces_to_observed_when_weights_match_pooled():
    rng = np.random.default_rng(5)
    rows = []
    for condition in ("A", "B"):
        for size_class, value in (("Small", 1.0), ("Medium+Large", 9.0)):
            for _ in range(100):
                rows.append(
                    {
                        "combo": "s|p|st",
                        "condition": condition,
                        "size_class": size_class,
                        "value": value + rng.normal(0, 1e-9),
                    }
                )
    events = pd.DataFrame(rows)
    out = esv.decompose_mixture(events, "value")
    assert np.allclose(out["observed"], out["fixed_weights"], atol=1e-6)
    assert np.allclose(out["observed"], out["fixed_class_means"], atol=1e-6)


def test_decompose_mixture_attributes_a_pure_composition_shift():
    rows = []
    for condition, n_large in (("A", 10), ("B", 90)):
        rows += [
            {"combo": "c", "condition": condition, "size_class": "Medium+Large",
             "value": 9.0}
        ] * n_large
        rows += [
            {"combo": "c", "condition": condition, "size_class": "Small",
             "value": 1.0}
        ] * (100 - n_large)
    out = esv.decompose_mixture(pd.DataFrame(rows), "value").set_index("condition")
    # Holding class means fixed reproduces the whole observed gap ...
    assert out.loc["B", "observed"] - out.loc["A", "observed"] == pytest.approx(6.4)
    assert out.loc["B", "fixed_class_means"] - out.loc[
        "A", "fixed_class_means"
    ] == pytest.approx(6.4)
    # ... while holding weights fixed removes it entirely.
    assert out.loc["B", "fixed_weights"] == pytest.approx(
        out.loc["A", "fixed_weights"]
    )


# Real event-level data (no NFS; Release assets, see release_data)


@pytest.mark.requires_release_data
def test_load_events_n_channel_identity_holds_on_production_events():
    """span/20 + 1 is exactly the channel count MAD is computed over."""
    events = esv.load_events("clas", columns=["onset_mad", "condition"])
    assert (events["n_channels"] >= 11).all()
    assert np.allclose(
        events["span"], (events["n_channels"] - 1) * esv.CHANNEL_PITCH_UM
    )
    assert set(events["size_class"].dropna().unique()) == {"Medium+Large"}


@pytest.mark.requires_release_data
def test_production_small_events_are_floored_at_zero_mad():
    """The floor is not hypothetical: it pins every n <= 7 event in the real data."""
    events = esv.load_events("llas", columns=["onset_mad", "offset_mad"])
    floored = events[events["n_channels"] <= esv.mad_zero_forced_max_n()]
    assert len(floored) > 100_000
    assert floored["onset_mad"].max() == 0.0
    assert floored["offset_mad"].max() == 0.0


@pytest.mark.requires_release_data
def test_adjusted_cell_means_cover_every_published_cell():
    """The adjusted panel must be the published panel, cell for cell.

    Otherwise "adjusted" and "unadjusted" are not comparable: the difference between
    them is partly the adjustment and partly a change of sample. The wake panel is
    where this bites: 42 published cells, of which the per-combo factor could only
    ever reach 17.
    """
    events = esv.load_events("llas", columns=["onset_mad", "offset_mad", "condition"])
    adjusted = esv.compute_adjusted_cell_means(
        events[events["size_class"] == "Medium+Large"]
    )
    published = pd.read_parquet(
        esv.R_OFFP_EXTDATA / "summarized_full48h_clas_offs.parquet",
        columns=["subject", "probe", "structure", "condition"],
    )
    keys = ["subject", "probe", "structure", "condition"]
    missing = set(map(tuple, published[keys].to_numpy())) - set(
        map(tuple, adjusted[keys].to_numpy())
    )
    assert not missing
    assert adjusted["adj_mean_onset_mad"].notna().all()
    assert adjusted["adj_mean_offset_mad"].notna().all()
