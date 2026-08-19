---
title: Sequential-regression estimators of small-OFF added value
updated: 2026-07-26
confirmed_by_user: false
---

# Plan: `sequential_added_value.ipynb`

Quantify how much epoch-level delta power small OFF periods account for on their
own, and how much they account for beyond larger OFF periods, reporting both
on a single common footing so the pair can be read side by side in one figure.

The "beyond" quantity can be scaled three ways. Establishing that the three
scalings carry the same information, and picking one on a principled basis, is
the substance of this notebook.

This document is self-contained: the notebook can be implemented from it without
reading any of the others in this directory.

> Scope and implementation status (2026-07-26). The notebook exists and
> implements everything below. Its job is the partial quantity and the
> demonstration that the three scalings of it are redundant; it is what licenses
> the choice of which one to report. The marginal companion and the published
> marginal-vs-partial panels are owned by `incremental_added_value.ipynb`,
> `incremental_added_value_wake.ipynb` and `added_value_figures.ipynb`, which
> already emit the semipartial; this notebook deliberately does not duplicate
> them. See section 13.

---

## 1. Question

Within a cortical recording, bin NREM (or Wake) time into fixed epochs. For each
epoch, measure:

- `y`: mean log delta power in that epoch
- `x1`: total area of *medium/large* OFF periods starting in that epoch
- `x2`: total area of *small* OFF periods starting in that epoch

Large OFF periods are strongly associated with delta power. Two questions, asked
together:

1. Marginal. How strongly does small-OFF area track delta power on its own?
2. Partial. After medium/large OFF area has been given first claim on every
   scrap of delta variance it can explain, is there signal left over that
   small-OFF area picks up?

This notebook answers (2) and settles how to scale the answer. (1) is fit by the
manuscript notebooks; it appears below only where it is needed to justify the
choice of scaling.

> The framing above ("small OFF periods carrying information beyond larger
> ones") is scientific interpretation, and is not settled.

---

## 2. The quantities

Work with standardized (z-scored, `ddof=0`) `y`, `x1`, `x2` within each
recording. Let `r12 = corr(x1, x2)`.

### Marginal: `beta_M` (reference only; fit elsewhere)

```
y ~ x2        ->  beta_M = slope on x2
```

Because everything is z-scored, `beta_M = corr(y, x2)` exactly, and `beta_M**2`
is the R² of the single-predictor model. Defined here because section 2.1 needs
it; do not fit it in this notebook; the manuscript notebooks already export
pooled marginals, and a second implementation would be a second source of truth.

### Estimator A: residualize the response only

```
step 1:  y  ~ x1        ->  e_y  = residuals
step 2:  e_y ~ x2       ->  beta_A = slope on x2
```

Reads as: strip from delta everything medium/large OFF area predicts, then ask
whether raw small-OFF area tracks what remains.

### Estimator B: Frisch-Waugh-Lovell, residualizing both sides

```
step 1:  y  ~ x1        ->  e_y  = residuals
step 2:  x2 ~ x1        ->  e_x2 = residuals
step 3:  e_y ~ e_x2     ->  beta_B = slope on e_x2
```

The predictor is the *excess* small-OFF area: how much more (or less) small-OFF
area an epoch had than its medium/large area would lead you to expect. `beta_B`
is algebraically identical to the coefficient on `x2` in the joint fit
`y ~ x1 + x2` (this *is* the Frisch-Waugh-Lovell theorem).

### Estimator C: FWL with the residualized predictor rescaled to unit SD

```
step 1:  y  ~ x1                   ->  e_y   = residuals
step 2:  x2 ~ x1                   ->  e_x2  = residuals
step 3:  e_x2n = e_x2 / sd(e_x2)       rescale to unit standard deviation
step 4:  e_y ~ e_x2n               ->  beta_C = slope on e_x2n
```

Motivation: `e_x2` has variance `1 - r12^2 < 1`, so a coefficient on it is not
expressed per one standard deviation *of the regressor actually entering the fit*.
Rescaling before fitting restores that.

`beta_C` is the semipartial (part) correlation of `y` with `x2` given `x1`,
provided `y` is standardized. Note the naming carefully: `beta_C` *is* the
semipartial correlation `sr`; `beta_C**2` is the squared semipartial, which is
the variance share. These are routinely conflated.

### 2.1 Why `beta_C` is the reported partial quantity

`beta_M` and `beta_C` are the pair that belong on one axis, because their squares
are the two variance quantities a reader actually wants:

| figure quantity | coefficient | its square |
|---|---|---|
| marginal | `beta_M` = `corr(y, x2)` | `R2(x2)` |
| partial | `beta_C` = `sr` | `R2(x1, x2) - R2(x1)` = ΔR² |

`beta_B**2` has no such interpretation (it is not a variance share), and
`beta_A**2` has none either. So a marginal/partial pair built from `beta_M` and
`beta_C` is simultaneously a coefficient figure and an incremental-R² figure,
while keeping the sign, which `R2` alone destroys, and which matters here
because small-OFF area is negatively associated with delta in NREM and positively
in Wake.

### 2.2 How the estimators relate (implement the checks, do not take them on faith)

Because OLS residuals are orthogonal to their regressor, `cov(e_y, x2) =
cov(e_y, e_x2)`; all three numerators agree. Only the denominators differ:
`var(x2) = 1`, `var(e_x2) = 1 - r12^2`, `var(e_x2n) = 1`. Hence

```
beta_A = beta_B * (1 - r12^2)
beta_C = beta_B * sqrt(1 - r12^2)
beta_A = beta_C * sqrt(1 - r12^2)
```

so `|beta_A| <= |beta_C| <= |beta_B|`, with equality only at `r12 = 0`. All three
carry the same sign, always. Fit the joint model too, purely as an internal
arithmetic check.

The three estimates differ by a monotone function of `r12` alone and therefore
carry no independent information about the data. Establishing that is one
point of the notebook; it is not a bake-off with a winner. See section 12.

The standard error scales by the same factor, exactly. `sqrt(1 - r12^2)` is a
function of the design matrix only, not of `y`, so conditional on `X` it is a
fixed scalar. Multiplying a coefficient by a fixed scalar multiplies its standard
error by the same scalar, including under HAC:

```
beta_C = beta_B * sqrt(1 - r12^2)
se_C   = se_B   * sqrt(1 - r12^2)      # exact, not approximate
```

Verified to `< 1e-15` relative error on AR(1) predictors with AR(1) errors and
`maxlags = 30`. Consequences: no bootstrap or delta method is needed to get an SE
for the semipartial, and the per-recording p-value is *identical* for A, B and C
(they are positive rescalings of one another, testing the same null). This is
check 10.

Estimator C is not scale-invariant; A and B are. The explicit `sd()` in step
3 does not cancel, which has two consequences an implementer must respect:

- `y` must be z-scored, or the identity `beta_C == sr` fails (you get
  `sr * sd(y)` instead).
- The `ddof` used to z-score `x2` and the `ddof` inside `sd(e_x2)` must match. A
  `ddof=0` / `ddof=1` mismatch introduces a relative error of `sqrt(n/(n-1))`,
  about `1.4e-4` at `n = 3500`, which will fail a `1e-10` check. Contrast with
  checks 1-3 below, which are scale-free and cannot be broken this way.

Where HAC belongs. Newey-West affects standard errors only, never point
estimates. Fit stage-1 regressions with plain OLS (their residuals are what
matter); apply `cov_type="HAC"` only to the final stage of each estimator, to
the marginal fit, and to the joint check model. An implementer who applies HAC
everywhere will get identical coefficients and waste time wondering why.

Inference caveat to record in the notebook. The staged estimators feed
stage-1 *residuals* into a later regression as if they were observed data, so
their standard errors do not propagate stage-1 estimation error. With thousands
of epochs per recording this is expected to be negligible, but the notebook
should state it rather than imply the SEs are exact.

---

## 3. Do not pool R² directly

An obvious-looking simplification is to skip coefficients entirely and pool
`R2(x2)` and `R2(x1,x2) - R2(x1)` across recordings. Do not. The pooling in
section 7 is DerSimonian-Laird, which is effect-size-agnostic and will happily
accept any `(effect, variance)` pair, so this fails silently rather than
erroring. Four reasons it is invalid here:

1. ΔR² is non-negative by construction. Its null distribution is not centered
   at zero, so a random-effects mean over ΔR² is positive and "significant" by
   construction, whatever the data say.
2. Boundary and skew. The ΔR² values here sit near 0 (per-recording
   `beta_C**2` median ~ 0.02 in NREM). A symmetric normal CI there runs below
   zero, into impossible territory.
3. No HAC analog. There is no closed-form standard error for ΔR², and, more
   importantly, no direct analog of the Newey-West serial-correlation
   correction. At 10 s epochs with `maxlags = 30` that correction is not
   cosmetic; dropping it makes intervals badly anticonservative.
4. The sign is destroyed. An R² figure draws NREM and Wake small-OFF
   associations as positive bars in both states, erasing the contrast between
   them.

The `beta_M` / `beta_C` pair sidesteps all four: both are bounded correlations on
an unbounded-enough, roughly symmetric scale, both carry exact HAC standard
errors, both keep their sign, and both square to exactly the R² quantities above.
Square only for display, never for pooling (see the trap in section 11).

---

## 4. Inputs

### 4.1 OFF periods

`./outputs/static_added_value/cache/offs_direct_48h.parquet` (relative to this
directory), written by `static_added_value.ipynb` with `state_mode="direct_48h"`.
If absent, run that notebook first.

Read only these columns: `subject`, `probe`, `structure`, `start_time`, `area`,
`category`, `state`.

- 29 `(subject, probe, structure)` groups.
- `category` in `{"BLAS", "CLAS", "LLAS"}`. These are disjoint tiers, not nested
  sets: `BLAS` = BLAS; `CLAS` = CLAS-but-not-BLAS; `LLAS` = LLAS-but-not-CLAS.
  Summing areas across tiers is therefore legitimate and non-double-counting.
- `state` value counts: `NREM` 4,158,789, `IS` 225,570, `Wake` 213,490,
  `Other` 116,117, `REM` 105,670, `MA` 97,670.

Filter to a single state per run: `offs[offs["state"] == STATE]`. Run the whole
notebook for `STATE = "NREM"` and again for `STATE = "Wake"`.

### 4.2 Predictor construction

```
x1  "Medium/Large"  =  blas_area + clas_excl_area     # the stringent set
x2  "Small"         =  llas_excl_area
```

### 4.3 Delta power (requires NFS)

```python
from cnpix_local_sleep import files, hyp

da = xr.load_dataarray(
    files.get_structure_bandpower_path(subject, probe, structure, "delta", True, "inst")
)
t, v = da["time"].values, da.values
hg = hyp.load_statistical_condition_hypnograms(subject, probe)["Full.Conservative"]
keep = hg.keep_states([STATE]).covers_time(t) & np.isfinite(v)
fs = 1.0 / np.median(np.diff(t[:10000]))
t, logd = t[keep], np.log10(v[keep])
```

Cache per `(subject, probe, structure, state)` in a module-level dict; these
loads dominate runtime and are re-used across the epoch-duration sweep.

---

## 5. Epoch table

Constants:

| name | value |
|---|---|
| `EPOCH_DURATION` | `10.0` s (primary) |
| `EPOCH_SWEEP` | `[4.0, 10.0, 30.0]` |
| `MIN_STATE_FRAC` | `0.8` |
| `MIN_EPOCHS` | `50` per group |
| `HAC_TARGET_S` | `300.0` |

Procedure, per group:

1. `edges = np.arange(t[0], t[-1] + epoch_duration, epoch_duration)`; `n = len(edges) - 1`.
2. Bin delta samples: `s_ep = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n - 1)`,
   accumulate sums and counts with `np.add.at`.
3. Keep an epoch only if `count >= MIN_STATE_FRAC * epoch_duration * fs`; this
   drops epochs straddling a state boundary. The outcome is the mean of `logd`
   over that epoch's in-state samples.
4. Assign each OFF to an epoch by its `start_time`; accumulate `area` per tier
   with `np.add.at`, guarding `(oe >= 0) & (oe < n)`.
5. Drop groups with fewer than `MIN_EPOCHS` surviving epochs, or where `x1` or
   `x2` is constant.

`maxlags = max(1, int(np.ceil(HAC_TARGET_S / epoch_duration)))` -> 30 at 10 s.

---

## 6. Per-group fitting

For each group, after z-scoring `x1`, `x2`, `y`:

```python
def resid(a, b):                       # residualize a on b, plain OLS
    return sm.OLS(a, sm.add_constant(b)).fit().resid

HAC = dict(cov_type="HAC", cov_kwds={"maxlags": maxlags})
r12 = float(np.corrcoef(x1, x2)[0, 1])

e_y  = resid(y, x1)
e_x2 = resid(x2, x1)

# Estimator A
fitA = sm.OLS(e_y, sm.add_constant(x2)).fit(**HAC)
# Estimator B
fitB = sm.OLS(e_y, sm.add_constant(e_x2)).fit(**HAC)
# Estimator C  (same ddof as the z-scoring of x2 -- see section 2)
e_x2n = e_x2 / e_x2.std()
fitC = sm.OLS(e_y, sm.add_constant(e_x2n)).fit(**HAC)
# joint check model
joint = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2]))).fit(**HAC)
# independent semipartial, for check 6 -- must NOT reuse fitC
sr = float(np.cov(y, e_x2n, ddof=0)[0, 1] / (y.std() * e_x2n.std()))
```

Record per group: `n_epochs`, `r12`, `beta_A`, `se_A`, `p_A`, `beta_B`, `se_B`,
`p_B`, `beta_C`, `se_C`, `p_C`, `sr`, `beta_joint`, `se_joint`, and the derived
`ratio_AB = beta_A / beta_B`, `ratio_CB = beta_C / beta_B`,
`one_minus_r12sq = 1 - r12**2`.

Also fit the mirror direction (`x1` residualized on `x2`) so the notebook
reports the medium/large tier on the same footing.

---

## 7. Pooling across recordings

Pool per-group `(beta, se)` with DerSimonian-Laird random effects. Do not
pool raw epochs: recordings differ in length by several-fold and would dominate
by epoch count. And do not pool R²; see section 3.

```python
def random_effects_meta(effects, variances):
    eff, v = np.asarray(effects, float), np.asarray(variances, float)
    w = 1.0 / v
    fe = np.sum(w * eff) / np.sum(w)
    q = np.sum(w * (eff - fe) ** 2)
    k = len(eff)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    wre = 1.0 / (v + tau2)
    pooled = np.sum(wre * eff) / np.sum(wre)
    se = 1.0 / np.sqrt(np.sum(wre))
    i2 = max(0.0, (q - (k - 1)) / q) * 100 if q > 0 else 0.0
    p = 2 * (1 - scipy.stats.norm.cdf(abs(pooled / se)))
    return dict(pooled=pooled, se=se, ci_lo=pooled - 1.96 * se,
                ci_hi=pooled + 1.96 * se, p=p, tau2=tau2, i_squared=i2, k=k)
```

Report pooled estimate, 95% CI, p, I², and k for Estimators A, B and C
separately, per state, per target tier. Mark exactly one row per (state, target)
as `reported = True`: `beta_C`, per section 12.

The A/B/C p-values will be near-identical by construction (identical per
recording; they differ after pooling only because the shrinkage factor varies
across recordings). Present them as one finding under three scalings, not as
three tests.

---

## 8. Validation cells (must pass, and must be visible in the notebook)

These are cheap and they are the point of the exercise: an implementation that
skips them has not demonstrated anything.

| # | check | tolerance |
|---|---|---|
| 1 | `beta_B` == joint model's coefficient on `x2`, every group | `< 1e-10` |
| 2 | `beta_A / beta_B` == `1 - r12**2`, every group | `< 1e-10` |
| 3 | `sign(beta_A) == sign(beta_B) == sign(beta_C)`, every group | exact |
| 4 | `corr(e_x2, x1)` ~ 0 (residualization actually worked) | `< 1e-10` |
| 5 | `corr(e_y, x1)` ~ 0 | `< 1e-10` |
| 6 | `beta_C` == independently computed `sr` | `< 1e-10` |
| 7 | `beta_C / beta_B` == `sqrt(1 - r12**2)`, every group | `< 1e-10` |
| 8 | `beta_C**2` == `R2_joint - corr(y, x1)**2` (ΔR²) | `< 1e-10` |
| 9 | `abs(beta_A) <= abs(beta_C) <= abs(beta_B)`, every group | exact |
| 10 | `se_C / se_B` == `sqrt(1 - r12**2)`, every group (HAC SEs) | `< 1e-10` |

Print the max absolute error across groups for 1, 2, 4-8 and 10 rather than a
bare `assert`, so the notebook shows how tightly the identities hold.

Checks 6-8 and 10 are the ones that catch a botched Estimator C. If 7 fails at
roughly `1.4e-4` relative while 1 and 2 pass, it is the `ddof` mismatch described
in section 2, not a conceptual error.

Note that check 2 is invariant to any affine rescaling of `y`, `x1`, `x2` (both
betas rescale identically and `r12` is scale-free), so it holds whether or not
you standardize, and a `ddof=0` vs `ddof=1` mismatch cannot break it. If check 2
fails while check 1 passes, the cause is almost always that `r12` was computed on
a different set of rows than the regression saw (e.g. computed before an
`NaN`-drop, or on the unfiltered epoch table).

Checks 8 and 10 together are what license section 3's claim that you get the R²
quantities and their standard errors for free.

---

## 9. Figures

Use `pubplots` with `pp.destination("figma")`. Route `figsize`, every explicit
`linewidth`, and `markersize` through `pp.scale(...)`. Do not pass
`fontsize=` / `labelsize=` anywhere; let the pubplots rcParams drive text size.

The marginal-vs-partial publication panel is not built here; it is
`figure_*_dumbbell` in `added_value_figures.ipynb`. These figures are the
estimator-equivalence evidence.

1. Forest plot of per-group `beta_C` with 95% CIs, sorted, with the pooled
   diamond. One panel per state.
2. Estimator scatter: `beta_A` vs `beta_B` and `beta_C` vs `beta_B`, one
   point per group, identity line, plus the predicted curves evaluated at each
   group's own `r12`. This is the visual statement that the scalings are
   redundant.
3. Shrinkage vs collinearity: `ratio_AB` and `ratio_CB` against `r12`,
   overlaid with the analytic `1 - r12^2` and `sqrt(1 - r12^2)` curves. Both
   should lie exactly on their curve; C sits above A everywhere.
4. Epoch-duration sweep: pooled `beta_C` across `EPOCH_SWEEP`, to show the
   conclusion is not an artifact of the 10 s choice.
5. Three-scaling forest: the pooled estimate and CI for A, B and C on one
   axis, per state. Makes the point that the scaling debate moves the number by
   only a few percent and changes nothing. Supplementary, not a main panel.

Write SVGs to `./outputs/sequential_added_value/<state>/`.

---

## 10. Outputs

```
outputs/sequential_added_value/
  <state>/
    group_estimates.parquet     # one row per (subject, probe, structure)
    pooled_estimates.parquet    # one row per (estimator, target); `reported` flag
    identity_checks.csv         # the max-abs-errors from section 8
    *.svg
```

Nothing here feeds the publication figures. `added_value_figures.ipynb` reads
`outputs/added_value_data/*_semipartial_pooled.parquet`, written by the
manuscript notebooks; see section 13.

---

## 11. Optional extension: variance decomposition

Only implement if explicitly asked.

Estimator C already delivers the entry point: `beta_C**2` is the unique variance
contribution of `x2` (validation check 8). The full decomposition needs the mirror
direction too.

Estimator B's stage-3 R² is the squared partial correlation, a fraction of
*residual* `y` variance. Rescale to a share of total `y` variance to get the
squared semipartial correlation, which is the unique contribution:

```
U_small = R2(e_y ~ e_x2) * (1 - r_y1**2)
U_medl  = R2(e_y|x2 ~ e_x1|x2) * (1 - r_y2**2)     # mirror direction
common  = R2_joint - U_medl - U_small
```

Three traps if this is tabled:

- Medians do not sum. `U_medl`, `U_small`, `common` add to `R2_joint` within
  each group, never across group medians. Report means, or show the per-group
  partition.
- `common` can be negative (suppression). It is not a share of a pie and a
  table must not present it as one.
- Squaring discards the sign. Whenever a squared-semipartial column is shown,
  keep the signed coefficient adjacent, or a negative and a positive association
  of equal strength become indistinguishable.

---

## 12. Reporting guidance

This is the convention for the analysis as a whole, not just this notebook.
Report two numbers per tier per state, each with one p-value: the marginal
`beta_M` (from the manuscript notebooks) and the partial `beta_C` (from here).
Do not report A, B and C as three findings:
they differ by a monotone function of `r12` alone, so quoting all three invites a
reader to treat them as three results, or to conclude the finding is
scale-sensitive when it is not.

`beta_C` is preferred over `beta_B` because `beta_C**2` is exactly the
incremental R² while `beta_B**2` is not a variance share at all (section 2.1),
and because the marginal panel it sits next to is already on the correlation
scale. `beta_A` is not recommended for reporting: it is included because it is the
natural first thing to try, and its scaling has no standard name.

Three traps when writing this up.

- Do not describe `beta_C` as "regress `y` on `x1`, then regress the residuals
  on `x2`." That sentence describes Estimator A, a different number
  (NREM small-OFF: -0.143 vs -0.150). `beta_C` must be described as *both* sides
  residualized: "regress `y` on `x1` and `x2` on `x1`, then regress the
  `y`-residuals on the standardized `x2`-residuals", or, more compactly, "the
  semipartial (part) correlation of `x2` with `y`, controlling `x1`."
- `|beta_C| <= |beta_B|` always. Switching from the joint coefficient shrinks
  the reported effect. Here the shrinkage is small (NREM `r12` median -0.229 ->
  factor 0.974, a 2.6% change), but state the direction rather than let a reader
  discover it.
- `(pooled beta_C)**2` is not the pooled ΔR². Pooling happens on the signed
  coefficient scale, where the normal approximation holds; squaring afterwards is
  a display convenience. Say so wherever a squared value is shown.

One caveat for anyone extending this to three or more predictors: the identity
`beta_C_i**2 == ΔR²_i` still holds exactly for each predictor, with the shrinkage
factor becoming `sqrt(1 - R2_i)` where `R2_i` is from regressing `x_i` on all the
others. What breaks is cross-predictor comparison: with exactly two predictors the
factor is the *same* for both (`r12` is symmetric), so C preserves their ratio
within a recording; with three or more the factor differs per predictor and the
ratio changes. Note that even in the two-predictor case the ratio is preserved
only *within* a recording; after DL pooling it is not exact, because the factor
varies across recordings.

---

## 13. Deltas from the version on disk

### 13.1 Manuscript pipeline: done (2026-07-26)

The semipartial is now the reported partial quantity in the manuscript notebooks,
independently of `sequential_added_value.ipynb`:

- `incremental_added_value.ipynb` and `incremental_added_value_wake.ipynb` gained a
  shared `semipartial_scales(df, predictors)` helper; `fit_group`, `fit_model` and
  `fit_collapsed_group` now emit `beta_*_sr` / `se_*_sr` / `sr_scale_*` alongside
  every joint coefficient, for the 3-tier and the collapsed 2-tier models in both
  states. `pool_tiers`, `pool_collapsed` and `pool_predictors` take `suffix="_sr"`.
- New exports: `{nrem,wake}_semipartial_pooled.parquet`,
  `{nrem,wake}_area_semipartial_pooled.parquet`; the `*_group_partial.parquet`
  files carry the `_sr` columns too.
- `added_value_figures.ipynb` reads the semipartial pooled tables for its `partial`
  series and maps the `_sr` per-group columns onto the plain names via
  `_use_semipartial`, so every dumbbell and forest reports one convention. The joint
  coefficients remain available under the `joint` key for sensitivity checks.
- Cross-validated against `sequential_added_value.ipynb`'s independent Estimator C
  code path: per-group max |Δ| = 8.9e-16 (NREM) and 2.4e-14 (Wake) on both
  coefficients and HAC standard errors.

Pooled reported values (marginal -> semipartial, with the semipartial's square):

| state / model | tier | marginal | semipartial | ΔR² |
|---|---|---|---|---|
| NREM amount | Conservative (CLAS set) | +0.732 | +0.666 | 0.443 |
| NREM amount | LLAS-exclusive | -0.288 | -0.150 | 0.022 |
| Wake amount | Conservative (CLAS set) | +0.297 | +0.173 | 0.030 |
| Wake amount | LLAS-exclusive | +0.327 | +0.217 | 0.047 |

### 13.2 `sequential_added_value.ipynb`: two small follow-ups

That notebook already implements every section above for both states, with
`beta_A`, `beta_B`, `beta_C`, `sr`, the mirror direction, checks 1-9, and all
five figures. Its two-predictor model matches the manuscript's collapsed area
model per group to machine precision in both states (verified: NREM max |Δ|
2.3e-15, Wake 2.9e-14, for both tiers).

Only two things trail this revision:

- [ ] Add check 10 (`se_C / se_B == sqrt(1 - r12**2)` under HAC). This is the
      one new claim the revision introduces and the notebook does not yet verify
      it.
- [ ] Flip the `reported` flag in `pooled_estimates.parquet` from `B` to `C`, and
      repoint the per-group forest (figure 1) from `beta_B` to `beta_C`, so the
      notebook does not assert a different reported quantity than section 12.

Optionally, update the closing "Reading the result" prose to section 12,
including the three write-up traps.

Deliberately not added, because the manuscript notebooks already own them and
a second implementation would be a second source of truth: a marginal fit, a
marginal-vs-partial panel, and a `semipartial_pooled.parquet` export.

One pre-existing inconsistency, surfaced by the per-group comparison and not
introduced by this change: for Wake / Conservative (CLAS set),
`incremental_added_value_wake.ipynb` emits `NaN` for `CNPIX19-Otto/imec0/V2`
(its `MIN_NONZERO_EPOCHS` support guard), so `k = 28`, while
`sequential_added_value.ipynb` fits it successfully (`beta_B_medl = 0.078`,
`se = 0.012`, `n_epochs = 9182`, `k = 29`). The Small tier is `k = 29` in both.
The published figure therefore shows the `k = 28` pooled semipartial +0.173
[0.135, 0.210]; including that group would give +0.169 [0.134, 0.205]. Decide
which is correct; do not silently adopt one.

---

## 14. Housekeeping

- Environment. Run through the workspace project so local editable sources are
  used: `cd gfys_workspace && uv run --all-extras --group dev ...`. Never bare
  `pip`.
- NFS required (`/Volumes/npx_nfs/`) for the bandpower zarrs. Verify the mount
  before starting.
- Add a row to `notebooks/README.md` matching the existing table
  format: notebook path, status, description, inputs, outputs, pooling
  note, `requires NFS`.
- Do not commit. Leave the notebook and any edits unstaged in the working
  tree.
- Runtime is dominated by the per-group bandpower loads (~29 groups × 2 states).
  Cache them; the epoch sweep must not re-read NFS.

---

## 15. Definition of done

- [x] Notebook runs top-to-bottom for `STATE = "NREM"` and `STATE = "Wake"`.
- [x] Per-group and pooled estimates written for all three staged estimators,
      both states, both target tiers.
- [x] All five figures rendered and saved.
- [ ] All ten validation checks print max-abs-errors within tolerance (check 10
      outstanding; section 13.2).
- [ ] The notebook names `beta_C` as the reported partial quantity per section
      12, rather than leaving three numbers to the reader.
- [ ] The notebook states, in prose, the relationship between the three
      estimators, the exact SE scaling, the generated-regressor caveat from
      section 2, and why R² is not pooled directly (section 3).
- [ ] The Wake / Conservative `k = 28` vs `k = 29` discrepancy (section 13.2) is
      resolved or explicitly documented.
- [ ] `notebooks/README.md` updated.
