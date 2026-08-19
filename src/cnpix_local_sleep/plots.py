import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubplots as pp
import seaborn as sns


def get_condition_palette():
    p = sns.color_palette("Paired")
    for i in np.arange(len(p))[1::2]:  # Desaturate some colors a bit
        p[i] = tuple((np.array(p[i]) + np.array(p[i - 1])) / 2)

    return {
        k: v
        for k, v in zip(
            [
                "Early.BSL.NREM",
                "Early.REC.NREM.Match",
                "Early.NOD.Wake",
                "Late.NOD.Wake",
                "Early.REC.NREM",
                "Late.REC.NREM",
            ],
            p,
        )
    }  # Assign colors to conditions


def get_category_palette() -> dict[str, tuple]:
    """Return colors for LLAS, CLAS, BLAS using the Set2 palette."""
    colors = sns.color_palette("Set2")
    return {"LLAS": colors[0], "CLAS": colors[1], "BLAS": colors[2]}


def get_smoothed_trace(
    events: pd.DataFrame,
    value_cols: list[str],
    time_col: str = "start_time",
    smoothing: str = "20s",
    rolling_op: str = "mean",
    fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Take a dataframe of events, whose times are given by `events[time_col]`,
    and return smoothed traces of the values in `value_cols`.

    Parameters:
    -----------
    events: pd.DataFrame
        A dataframe of events, whose times are given by `time_col`.
    value_cols: list[str]
        The columns in `events` that give the values to smooth.
    time_col: str
        The column in `events` that gives the event times.
    smoothing: str
        The smoothing window to use for the event rate.
    rolling_op: str
        Aggregation to apply over the rolling window. One of {"mean", "sum", "median"}.
    fill_values: dict[str, float], optional
        If provided, fill gaps longer than `smoothing` with the specified values.
        The keys are the columns to fill, and the values are the fill values.
    """
    df = events[value_cols].copy()
    df.index = pd.to_timedelta(events[time_col], "s")
    df.index.name = time_col
    df = df.sort_index()
    op = rolling_op.lower()
    try:
        agg_func = {
            "mean": "mean",
            "sum": "sum",
            "median": "median",
            "std": "std",
            "sem": "sem",
        }[op]
    except KeyError as exc:
        raise ValueError(
            f"Invalid rolling_op '{rolling_op}'. Expected one of 'mean', 'sum', or 'median'."
        ) from exc
    df = df.rolling(smoothing).aggregate(agg_func)

    fill_values = fill_values or {}
    if fill_values:
        smoothing_td = pd.to_timedelta(smoothing)
        gaps = df.index.to_series().diff()
        large_gaps = gaps[gaps > smoothing_td]
        if not large_gaps.empty:
            fill_rows = []
            for gap_end_idx in large_gaps.index:
                loc = df.index.get_loc(gap_end_idx)
                if isinstance(loc, slice):
                    start_pos = loc.start - 1
                else:
                    start_pos = loc - 1
                if start_pos < 0:
                    continue
                gap_start = df.index[start_pos]
                gap_end = gap_end_idx
                if gap_end - gap_start <= smoothing_td:
                    continue
                fill_times = pd.timedelta_range(
                    start=gap_start + smoothing_td,
                    end=gap_end - smoothing_td,
                    freq=smoothing_td,
                )
                for fill_time in fill_times:
                    fill_rows.append(
                        {
                            time_col: fill_time,
                            **{
                                col: fill_values.get(col, float("nan"))
                                for col in value_cols
                            },
                        }
                    )
            if fill_rows:
                filler = pd.DataFrame(fill_rows).set_index(time_col)
                df = pd.concat([df, filler]).sort_index()

    df = df.reset_index()
    df[time_col] = df[time_col].dt.total_seconds()
    return df


#############
# Publication row-stacked figures (cross-figure row alignment)
#############

# Shared vertical geometry for stacked-row publication figures. Two figures
# built with these same constants (same ``nrows`` and ``height_in``) have their
# rows land at identical absolute y positions, so they line up row-for-row when
# placed side by side -- e.g. the intrusion-sweep (~1.5" wide) and
# incline-magnitude (~3.1" wide) figures share this grid. Figure-fraction
# margins: identical top/bottom/hspace + identical height means identical row boxes.
STACK_HEIGHT_IN: float = 3.3  # shared figure height (inches, pre pp.scale)
STACK_TOP: float = 0.98  # top margin (figure fraction)
STACK_BOTTOM: float = 0.14  # bottom margin (leaves room for bottom-row x labels)
STACK_HSPACE: float = 0.28  # inter-row spacing (fraction of mean axes height)


def stacked_rows(
    width_in: float,
    ncols: int = 1,
    *,
    nrows: int = 3,
    left: float = 0.2,
    right: float = 0.98,
    wspace: float = 0.12,
    width_ratios: list[float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    height_in: float = STACK_HEIGHT_IN,
    top: float = STACK_TOP,
    bottom: float = STACK_BOTTOM,
    hspace: float = STACK_HSPACE,
):
    """Create an ``nrows`` x ``ncols`` publication figure with FIXED vertical geometry.

    Separate figures created with the same ``nrows``/``height_in``/``top``/
    ``bottom``/``hspace`` have their rows aligned when placed side by side; only
    the horizontal margins (``left``/``right``/``wspace``/``width_ratios``) and
    ``width_in`` should differ between such figures. Must be called inside a
    ``with pubplots.destination(...)`` block so figsize and rc scaling match the
    destination. Autolayout is disabled so the explicit margins are respected
    (pubplots sets ``figure.autolayout=True``, which would otherwise re-fit
    margins per-figure and break cross-figure alignment).

    Returns ``(fig, axes)`` where ``axes`` is always a 2-D array (``squeeze=False``).
    """
    gridspec_kw = {"width_ratios": width_ratios} if width_ratios is not None else {}
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=pp.scale(width_in, height_in),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw,
    )
    # Disable pubplots' autolayout (tight_layout) so subplots_adjust holds and
    # the vertical geometry is identical across separately-created figures.
    fig.set_layout_engine("none")
    fig.subplots_adjust(
        top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace
    )
    return fig, axes


#############
# Tom Bugnon: Single structure
#############
