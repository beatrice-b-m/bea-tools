from typing import Optional, Literal
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class BarPlotStyle(ABC):
    """Abstract base class for different bar plot styles."""

    @abstractmethod
    def plot(
        self,
        ax: plt.Axes,  # type: ignore
        x_feature_levels: list[str],
        s_feature_levels: list[str],
        s_feature_counts: dict[str, list[int]],
        s_colors: dict[str, str],
        width: float,
        **kwargs,
    ) -> None:
        """Plot the bars according to the specific style."""
        pass

    @abstractmethod
    def calculate_ymax(
        self, s_feature_counts: dict[str, list[int]], ymax: Optional[float]
    ) -> float:
        """Calculate the appropriate y-axis maximum for this plot style."""
        pass


class StackedBarStyle(BarPlotStyle):
    """Stacked bar plot style where bars are stacked on top of each other."""

    def plot(
        self,
        ax: plt.Axes,  # type: ignore
        x_feature_levels: list[str],
        s_feature_levels: list[str],
        s_feature_counts: dict[str, list[int]],
        s_colors: dict[str, str],
        width: float,
        **kwargs,
    ) -> None:
        x_bottom_vals = np.zeros(len(x_feature_levels))

        for s_level, s_count in s_feature_counts.items():
            color: str = s_colors[s_level]
            p = ax.bar(
                x_feature_levels,
                s_count,
                width,
                label=s_level,
                bottom=x_bottom_vals,
                color=color,
                zorder=2,
            )
            x_bottom_vals += s_count

            s_count_labels: list[str] = [str(i) if (i > 0) else "" for i in s_count]
            ax.bar_label(p, labels=s_count_labels, label_type="edge")

    def calculate_ymax(
        self, s_feature_counts: dict[str, list[int]], ymax: Optional[float]
    ) -> float:
        if ymax is None:
            total_counts: list[int] = [sum(z) for z in zip(*s_feature_counts.values())]
            return round(max(total_counts) * 1.1)
        return ymax


class SideBarStyle(BarPlotStyle):
    """Side-by-side bar plot style where bars are placed next to each other."""

    def __init__(
        self,
        color_bar_labels: bool = True,
        bar_label_percentages: bool = True,
        total_n: Optional[int] = None,
    ):
        self.color_bar_labels = color_bar_labels
        self.bar_label_percentages = bar_label_percentages
        self.total_n = total_n

    def plot(
        self,
        ax: plt.Axes,  # type: ignore
        x_feature_levels: list[str],
        s_feature_levels: list[str],
        s_feature_counts: dict[str, list[int]],
        s_colors: dict[str, str],
        width: float,
        **kwargs,
    ) -> None:
        x_feature_pos = np.arange(len(x_feature_levels))
        n_s_levels: int = len(s_feature_levels)
        offset: float = -((n_s_levels - 1) / 2) * width

        for i, (s_level, s_count) in enumerate(s_feature_counts.items()):
            if i > 0:
                offset += width

            color: str = s_colors[s_level]
            p = ax.bar(
                x_feature_pos + offset,
                s_count,
                width,
                label=s_level,
                color=color,
                zorder=2,
            )

            bar_label_kwargs: dict = {
                "label_type": "edge",
                "padding": 3,
            }

            if self.color_bar_labels:
                bar_label_kwargs["color"] = color

            if self.bar_label_percentages and self.total_n is not None:
                bar_labels = [f"{v:d}\n({v/self.total_n:.1%})" for v in s_count]
                bar_label_kwargs["labels"] = bar_labels

            ax.bar_label(p, **bar_label_kwargs)

        ax.set_xticks(x_feature_pos, x_feature_levels)

    def calculate_ymax(
        self, s_feature_counts: dict[str, list[int]], ymax: Optional[float]
    ) -> float:
        if ymax is None:
            s_feature_count_values: list[int] = []
            for s_f_counts in s_feature_counts.values():
                s_feature_count_values.extend(s_f_counts)
            return round(max(s_feature_count_values) * 1.1)
        return ymax


def bar_plot(
    data: pd.DataFrame,
    x_feature: str,
    s_feature: str,
    s_colors: dict[str, str],
    style: Literal["stacked", "side"] | BarPlotStyle = "stacked",
    x_feature_levels: Optional[list[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,  # type: ignore
    ymax: Optional[float] = None,
    fig_kwargs: Optional[dict] = None,
    width: float = 0.6,
    color_bar_labels: bool = True,
    bar_label_percentages: bool = True,
) -> Optional[float]:
    """
    Create a bar plot with various styles (stacked or side-by-side).

    Args:
        data: DataFrame containing the data to plot
        x_feature: Column name for x-axis categories
        s_feature: Column name for stratification/grouping
        s_colors: Dictionary mapping s_feature values to colors
        style: Plot style - either "stacked", "side", or a custom BarPlotStyle instance
        x_feature_levels: Optional list specifying order of x-axis categories
        title: Optional plot title
        xlabel: Optional x-axis label
        ax: Optional existing axes to plot on
        ymax: Optional y-axis maximum value
        fig_kwargs: Optional kwargs for figure creation
        width: Bar width (default 0.6 for stacked, typically 0.3 for side)
        color_bar_labels: Whether to color bar labels (side style only)
        bar_label_percentages: Whether to show percentages in labels (side style only)

    Returns:
        The calculated ymax if one was not provided, otherwise None
    """
    if ax is None:
        if fig_kwargs is None:
            fig_kwargs = dict()
        fig, ax = plt.subplots(**fig_kwargs)

    assert ax is not None

    _data = data.copy()
    _data[s_feature] = _data[s_feature].astype(str)
    _data[x_feature] = _data[x_feature].astype(str)

    ax.grid(axis="y", zorder=1, color="lightgray")

    if x_feature_levels is None:
        x_feature_levels: list[str] = _data[x_feature].unique().tolist()

    s_feature_levels: list[str] = _data[s_feature].unique().tolist()
    s_feature_counts: dict[str, list[int]] = {
        s_level: list() for s_level in s_feature_levels
    }

    for s_level in s_feature_levels:
        for x_level in x_feature_levels:
            n_s_x: int = len(
                _data[(_data[s_feature] == s_level) & (_data[x_feature] == x_level)]
            )
            s_feature_counts[s_level].append(n_s_x)

    # Get or create the style instance
    if isinstance(style, str):
        if style == "stacked":
            style_instance = StackedBarStyle()
        elif style == "side":
            style_instance = SideBarStyle(
                color_bar_labels=color_bar_labels,
                bar_label_percentages=bar_label_percentages,
                total_n=len(_data),
            )
        else:
            raise ValueError(f"Unknown style: {style}. Use 'stacked' or 'side'.")
    else:
        style_instance = style

    # Calculate ymax using the style's method
    fin_ymax: float = style_instance.calculate_ymax(s_feature_counts, ymax)

    # Plot using the style's method
    style_instance.plot(
        ax=ax,
        x_feature_levels=x_feature_levels,
        s_feature_levels=s_feature_levels,
        s_feature_counts=s_feature_counts,
        s_colors=s_colors,
        width=width,
    )

    ax.set_ylim(0, fin_ymax)

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.set_ylabel("Frequency")

    ax.legend()

    # return the calculated ymax if one was not input
    if ymax is None:
        return fin_ymax


def stratified_bar_plot(
    data: pd.DataFrame,
    color_dict: dict[str, str],
    strat_feature: str,
    x_feature: str,
    s_feature: str,
    style: Literal["stacked", "side"] | BarPlotStyle = "stacked",
    fig_kwargs: Optional[dict] = None,
    xlabel: Optional[str] = None,
    titles: Optional[list[str]] = None,
    suptitle: Optional[str] = None,
    parent: Optional[plt.Figure] = None,  # type: ignore
    x_feature_levels: Optional[list[str]] = None,
    width: float = 0.6,
    color_bar_labels: bool = True,
    bar_label_percentages: bool = True,
) -> plt.Figure:  # type: ignore
    """
    Create stratified bar plots showing overall data and data split by stratification feature.

    Args:
        data: DataFrame containing the data to plot
        color_dict: Dictionary mapping s_feature values to colors
        strat_feature: Column name for stratification (creates separate subplots)
        x_feature: Column name for x-axis categories
        s_feature: Column name for grouping/coloring within each plot
        style: Plot style - either "stacked", "side", or a custom BarPlotStyle instance
        fig_kwargs: Optional kwargs for figure creation
        xlabel: Optional x-axis label
        titles: Optional list of titles for each subplot
        suptitle: Optional overall figure title
        parent: Optional existing figure to plot on
        x_feature_levels: Optional list specifying order of x-axis categories
        width: Bar width
        color_bar_labels: Whether to color bar labels (side style only)
        bar_label_percentages: Whether to show percentages in labels (side style only)

    Returns:
        The matplotlib Figure object
    """
    if fig_kwargs is None:
        fig_kwargs = dict()

    _data = data.copy()
    _data[strat_feature] = _data[strat_feature].astype(str)
    _data[x_feature] = _data[x_feature].astype(str)
    _data[s_feature] = _data[s_feature].astype(str)

    if x_feature_levels is None:
        x_feature_levels: list[str] = _data[x_feature].unique().tolist()

    strat_feature_levels: list[any] = _data[strat_feature].unique().tolist()
    n_plots: int = len(strat_feature_levels) + 1

    if titles is None:
        titles = [None] * n_plots  # type: ignore

    assert titles is not None

    if parent is None:
        parent = plt.figure(constrained_layout=True, **fig_kwargs)

    assert parent is not None
    axes = parent.subplots(n_plots, 1)

    # plot combined strat levels first with the original data
    ax = axes[0]
    ymax = bar_plot(
        data=_data,
        x_feature=x_feature,
        s_feature=s_feature,
        x_feature_levels=x_feature_levels,
        s_colors=color_dict,
        style=style,
        title=titles[0],
        xlabel=xlabel,
        ax=ax,
        width=width,
        color_bar_labels=color_bar_labels,
        bar_label_percentages=bar_label_percentages,
    )

    for i, strat_level in enumerate(strat_feature_levels):
        ax = axes[i + 1]
        plot_data = _data[_data[strat_feature] == strat_level]

        bar_plot(
            data=plot_data,
            x_feature=x_feature,
            s_feature=s_feature,
            x_feature_levels=x_feature_levels,
            s_colors=color_dict,
            style=style,
            title=titles[i + 1],
            xlabel=xlabel,
            ax=ax,
            ymax=ymax,
            width=width,
            color_bar_labels=color_bar_labels,
            bar_label_percentages=bar_label_percentages,
        )

    if suptitle is not None:
        parent.suptitle(suptitle, fontsize=16)

    return parent


def stacked_bar_plot(
    data: pd.DataFrame,
    x_feature: str,
    s_feature: str,
    s_colors: dict[str, str],
    x_feature_levels: Optional[list[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,  # type: ignore
    ymax: Optional[float] = None,
    fig_kwargs: Optional[dict] = None,
    width: float = 0.6,
) -> Optional[float]:
    """
    Create a stacked bar plot. This is a convenience wrapper around bar_plot().

    See bar_plot() for detailed documentation.
    """
    return bar_plot(
        data=data,
        x_feature=x_feature,
        s_feature=s_feature,
        s_colors=s_colors,
        style="stacked",
        x_feature_levels=x_feature_levels,
        title=title,
        xlabel=xlabel,
        ax=ax,
        ymax=ymax,
        fig_kwargs=fig_kwargs,
        width=width,
    )


def stratified_stacked_bar_plot(
    data: pd.DataFrame,
    color_dict: dict[str, str],
    strat_feature: str,
    x_feature: str,
    s_feature: str,
    fig_kwargs: Optional[dict] = None,
    xlabel: Optional[str] = None,
    titles: Optional[list[str]] = None,
    suptitle: Optional[str] = None,
    parent: Optional[plt.Figure] = None,  # type: ignore
    x_feature_levels: Optional[list[str]] = None,
) -> plt.Figure:  # type: ignore
    """
    Create stratified stacked bar plots. This is a convenience wrapper around stratified_bar_plot().

    See stratified_bar_plot() for detailed documentation.
    """
    return stratified_bar_plot(
        data=data,
        color_dict=color_dict,
        strat_feature=strat_feature,
        x_feature=x_feature,
        s_feature=s_feature,
        style="stacked",
        fig_kwargs=fig_kwargs,
        xlabel=xlabel,
        titles=titles,
        suptitle=suptitle,
        parent=parent,
        x_feature_levels=x_feature_levels,
    )


def side_bar_plot(
    data: pd.DataFrame,
    x_feature: str,
    s_feature: str,
    s_colors: dict[str, str],
    x_feature_levels: Optional[list[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,  # type: ignore
    ymax: Optional[float] = None,
    fig_kwargs: Optional[dict] = None,
    width: float = 0.3,
    color_bar_labels: bool = True,
    bar_label_percentages: bool = True,
) -> Optional[float]:
    """
    Create a side-by-side bar plot. This is a convenience wrapper around bar_plot().

    See bar_plot() for detailed documentation.
    """
    return bar_plot(
        data=data,
        x_feature=x_feature,
        s_feature=s_feature,
        s_colors=s_colors,
        style="side",
        x_feature_levels=x_feature_levels,
        title=title,
        xlabel=xlabel,
        ax=ax,
        ymax=ymax,
        fig_kwargs=fig_kwargs,
        width=width,
        color_bar_labels=color_bar_labels,
        bar_label_percentages=bar_label_percentages,
    )


def stratified_side_bar_plot(
    data: pd.DataFrame,
    color_dict: dict[str, str],
    strat_feature: str,
    x_feature: str,
    s_feature: str,
    fig_kwargs: Optional[dict] = None,
    xlabel: Optional[str] = None,
    titles: Optional[list[str]] = None,
    suptitle: Optional[str] = None,
    parent: Optional[plt.Figure] = None,  # type: ignore
    x_feature_levels: Optional[list[str]] = None,
) -> plt.Figure:  # type: ignore
    """
    Create stratified side-by-side bar plots. This is a convenience wrapper around stratified_bar_plot().

    See stratified_bar_plot() for detailed documentation.
    """
    return stratified_bar_plot(
        data=data,
        color_dict=color_dict,
        strat_feature=strat_feature,
        x_feature=x_feature,
        s_feature=s_feature,
        style="side",
        fig_kwargs=fig_kwargs,
        xlabel=xlabel,
        titles=titles,
        suptitle=suptitle,
        parent=parent,
        x_feature_levels=x_feature_levels,
        width=0.3,  # Default width for side-by-side plots
    )
