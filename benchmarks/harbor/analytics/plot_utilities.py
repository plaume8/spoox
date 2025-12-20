import itertools
from collections import Counter
from enum import Enum
from statistics import mean, median
from typing import List, Dict, Any, Set
from typing import Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import MultipleLocator, PercentFormatter, FuncFormatter, MaxNLocator
import matplotlib.patches as mpatches


class Color(Enum):
    """generic color pallet"""

    DARK_BLUE = '#00365e'  # '#00355c'
    LIGHT_BLUE = '#00778A'
    ORANGE = '#99461f'  # '#D64C13' '#ad4317'
    DARK_GREY = '#7F7F7F'
    LIGHT_GREY = '#b5b5b5'
    GREEN = '#036328'
    PURPLE = '#69085A'
    RED = '#9C0D1B'
    LIGHT_ORANGE = '#db7125'
    YELLOW = '#c7970c'

    WHITE = '#FFFFFF'
    BLACK = '#000000'


DEFAULT_COLOR_LIST = [
    Color.DARK_BLUE,
    Color.ORANGE,
    Color.GREEN,
    Color.PURPLE,
    Color.LIGHT_BLUE,
    Color.DARK_GREY,
    Color.RED,
    Color.LIGHT_GREY,
    Color.YELLOW,
]


def color_check_plot() -> None:
    """Dummy plot for testing default color list."""

    data = [[0, 1 * (1 / c)] for c in range(1, len(DEFAULT_COLOR_LIST) + 1)]
    plt.figure()
    for s, c in zip(data, DEFAULT_COLOR_LIST):
        plt.plot(s, color=c.value)
    plt.grid(True)

    plt.show()


def line_plot(
        ax,
        x_labels: list[str],
        y_1_values: dict[str, list[list[float]]],
        y_1_label: str,
        y_1_lim: tuple[float, float],
        y_1_tic_steps: float,
        y_1_colors: list[Color] = None,
        y_1_horizontal_lines: Optional[list[Tuple[str, float, Color]]] = None,
        y_1_decimal_places: bool = True,
        additional_boxes: bool = False,
        close_legend: bool = True,

) -> None:
    """simple line plot"""

    x = range(len(x_labels))

    # colors
    if y_1_colors is None:
        y_1_colors = DEFAULT_COLOR_LIST.copy()

    # use first len(y_1_values) colors
    for (label, dataset), color in zip(y_1_values.items(), y_1_colors):
        label = label.split('-')[-1]
        # scatter all data points for this dataset
        for i, ys in enumerate(dataset):
            ax.scatter(
                [i] * len(ys),
                ys,
                marker='x',
                s=35,
                alpha=0.5,
                color=color.value,
            )

        # add ONE legend entry for all X
        proxy_x = ax.scatter([], [], marker='x', s=35, alpha=0.5, color=color.value, label=f"single runs ({label})")

        # mean line for this dataset
        means = [mean(ys) if ys else None for ys in dataset]
        ax.plot(
            x,
            means,
            color=color.value,
            linestyle='--',
            marker='o',
            linewidth=1.5,
            markersize=4,
            label=f"mean ({label})",
        )

    # horizontal grid lines
    ax.grid(
        which="major",
        axis="y",
        color=Color.LIGHT_GREY.value,
        linestyle="-",
        linewidth=0.5,
    )

    # --- horizontal lines for y1 axis ---
    if y_1_horizontal_lines:
        for name, y_pos, color in y_1_horizontal_lines:
            ax.axhline(
                y=y_pos,
                color=color.value,
                linestyle="-",
                linewidth=1,
                label=name,
                alpha=0.3,
            )

    # formatting left y-axis
    # base ticks from 0 to y_up with given step size ---
    h_ticks = list(np.arange(y_1_lim[0], y_1_lim[1] + 0.0001, y_1_tic_steps))
    # add horizontal-line ticks (if any)
    ticks = h_ticks.copy()
    if y_1_horizontal_lines:
        ticks.extend([y for _, y, _ in y_1_horizontal_lines])
    ticks.sort()
    ax.set_yticks(ticks)
    ax.set_ylim(*y_1_lim)
    ax.set_ylabel(y_1_label)
    # format tick labels
    tick_labels = []
    for t in ticks:
        if not y_1_decimal_places:
            tick_labels.append(f"{t:.0f}")

        elif t in h_ticks:
            tick_labels.append(f"{t:.2f}")  # normal tick → force 2 decimal places
        else:
            tick_labels.append(f"{t:.3f}")  # horizontal line → force 3 decimal place
    ax.set_yticklabels(tick_labels)
    # add light grid for the base ticks ---
    ax.grid(True, axis="y", which="major", linestyle="-", alpha=0.3)
    # color tick labels for horizontal lines ---
    if y_1_horizontal_lines:
        value_to_color = {val: color.value for (_label, val, color) in y_1_horizontal_lines}
        for tick_val, tick_label in zip(ax.get_yticks(), ax.get_yticklabels()):
            color = value_to_color.get(tick_val)
            if color is not None:
                tick_label.set_color(color)

    # --- x-axis ---
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)

    # --- legend (including optional additional boxes) ---
    handles, labels = ax.get_legend_handles_labels()

    if additional_boxes:
        extra_handles = [
            mpatches.Patch(facecolor=y_1_colors[0].value, linewidth=0, alpha=0.5),
            mpatches.Patch(facecolor=y_1_colors[1].value, linewidth=0, alpha=0.5),
        ]
        extra_labels = ["gpt-5-mini", "gpt-5-nano"]
        handles += extra_handles
        labels += extra_labels

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15 if close_legend else -0.25),
        ncol=2,
        borderpad=0.5,
        labelspacing=0.4,
        handletextpad=1,
    )


def line_plot_with_box_plots(
        ax,
        x_labels: list[str],
        y_values: dict[str, dict[str, list[float]]],
        y_label: str,
        y_lim: tuple[float, float],
        y_tic_steps: float,
        y_colors: list[Color] = None,
        y_horizontal_lines: Optional[list[Tuple[str, float, Color]]] = None,
        y_decimal_places: bool = True
) -> None:
    x = range(len(x_labels))

    # colors
    if y_colors is None:
        y_colors = DEFAULT_COLOR_LIST.copy()

    num_datasets = len(y_values)
    # how wide each little boxplot group is (relative to x step = 1)
    box_group_width = 0.6
    if num_datasets > 0:
        box_width = box_group_width / max(num_datasets, 1)
    else:
        box_width = 0.3

    # loop over datasets
    for ds_index, (dataset_label, color) in enumerate(zip(y_values.keys(), y_colors)):
        dataset = y_values[dataset_label]

        # ensure stable order corresponding to x_labels, if keys match
        # fallback: just use dict order
        if set(dataset.keys()) == set(x_labels):
            # order data_points according to x_labels if possible
            data_points = [dataset[label] for label in x_labels]
        else:
            data_points = list(dataset.values())

        # offset each dataset left/right so they don't overlap
        offset = (ds_index - (num_datasets - 1) / 2) * box_width

        # --- mean line (dots centered on boxplots) ---
        means = [median(ys) if ys else None for ys in data_points]
        mean_x_positions = [i + offset for i in range(len(means))]

        ax.plot(
            mean_x_positions,
            means,
            color=color.value,
            linestyle=':',
            marker='o',
            linewidth=1.4,
            markersize=2.0,
            label=f"mean – {dataset_label}",
        )

        # --- small box plots for each x ---
        boxplot_data = []
        boxplot_positions = []

        for i, ys in enumerate(data_points):
            if not ys:
                continue
            boxplot_data.append(ys)
            boxplot_positions.append(i + offset)

        if boxplot_data:
            bp = ax.boxplot(
                boxplot_data,
                positions=boxplot_positions,
                widths=box_width * 0.8,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markersize=0.6, markerfacecolor='black', markeredgecolor='black')
            )

            # color styling for the boxplots
            for box in bp['boxes']:
                box.set_facecolor((*to_rgb(color.value), 0.5))
                box.set_edgecolor('black')
                box.set_linewidth(0.8)
            for med in bp['medians']:
                med.set_color(color.value)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(0.8)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(0.8)

    # horizontal base grid
    ax.grid(
        which="major",
        axis="y",
        color=Color.LIGHT_GREY.value,
        linestyle="-",
        linewidth=0.5,
    )

    # horizontal lines
    if y_horizontal_lines:
        for name, y_pos, color in y_horizontal_lines:
            ax.axhline(
                y=y_pos,
                color=color.value,
                linestyle="-",
                linewidth=1,
                label=name,
                alpha=0.3,
            )

    # y-axis ticks
    h_ticks = list(np.arange(y_lim[0], y_lim[1] + 0.0001, y_tic_steps))
    ticks = h_ticks.copy()
    if y_horizontal_lines:
        ticks.extend([y for _, y, _ in y_horizontal_lines])
    ticks.sort()

    ax.set_yticks(ticks)
    ax.set_ylim(*y_lim)
    ax.set_ylabel(y_label)

    # tick formatting
    tick_labels = []
    for t in ticks:
        if not y_decimal_places:
            tick_labels.append(f"{t:.0f}")
        elif t in h_ticks:
            tick_labels.append(f"{t:.2f}")
        else:
            tick_labels.append(f"{t:.3f}")
    ax.set_yticklabels(tick_labels)

    # recolor tick labels for horizontal lines
    if y_horizontal_lines:
        value_to_color = {val: color.value for (_label, val, color) in y_horizontal_lines}
        for tick_val, tick_label in zip(ax.get_yticks(), ax.get_yticklabels()):
            c = value_to_color.get(tick_val)
            if c is not None:
                tick_label.set_color(c)

    # x-axis labels (still at integer positions)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)

    plt.tight_layout()
    plt.show()


def line_plot_double(
        title: str,
        x_labels: list[str],
        y_1_values: dict[str, list[list[float]]],
        y_1_label: str,
        y_1_lim: tuple[float, float],
        y_1_tic_steps: float,
        y_1_colors: list[Color] = None,
        y_1_horizontal_lines: Optional[list[Tuple[str, float, Color]]] = None,
        y_2_values: Optional[dict[str, list[list[float]]]] = None,
        y_2_label: Optional[str] = None,
        y_2_lim: Optional[tuple[float, float]] = None,
        y_2_tic_steps: Optional[float] = None,
        y_2_colors: list[Color] = None,
        fig_size: tuple[float, float] = (8, 5),
) -> None:
    """simple dual-axis line plot"""

    x = range(len(x_labels))
    fig, ax_left = plt.subplots(figsize=fig_size)
    fig.suptitle(title, fontsize=10)

    # colors
    if y_1_colors is None:
        y_1_colors = DEFAULT_COLOR_LIST.copy()
    if y_2_colors is None:
        y_2_colors = DEFAULT_COLOR_LIST.copy().reverse()

    # --- plot y_1_values on left axis ---
    # use first len(y_1_values) colors
    for (label, dataset), color in zip(y_1_values.items(), y_1_colors):
        # scatter all data points for this dataset
        for i, ys in enumerate(dataset):
            ax_left.scatter(
                [i] * len(ys),
                ys,
                marker='x',
                s=30,
                alpha=0.5,
                color=color.value
            )
        # mean line for this dataset
        means = [mean(ys) if ys else None for ys in dataset]
        ax_left.plot(
            x,
            means,
            color=color.value,
            linestyle='--',
            marker='o',
            markersize=3,
            label=label,
        )

    # formatting left y-axis
    ax_left.set_ylim(*y_1_lim)
    ax_left.yaxis.set_major_locator(MultipleLocator(y_1_tic_steps))
    ax_left.set_ylabel(y_1_label)

    # horizontal grid lines
    ax_left.grid(
        which="major",
        axis="y",
        color=Color.LIGHT_GREY.value,
        linestyle=":",
        linewidth=0.5
    )

    # --- horizontal lines for y1 axis ---
    if y_1_horizontal_lines:
        for name, y_pos, color in y_1_horizontal_lines:
            ax_left.axhline(
                y=y_pos,
                color=color.value,
                linestyle="-",
                linewidth=1,
                label=name,
                alpha=0.3,
            )

    # --- optional: plot y_2_values on right axis ---
    ax_right = None
    if y_2_values is not None:
        if y_2_lim is None or y_2_tic_steps is None or y_2_colors is None:
            raise ValueError("If y_2_values is provided, y_2_lim, y_2_tic_steps and y_2_colors must also be provided.")

        ax_right = ax_left.twinx()
        for (label, dataset), color in zip(y_2_values.items(), y_2_colors):
            # scatter points for this dataset (right axis)
            for i, ys in enumerate(dataset):
                ax_right.scatter(
                    [i] * len(ys),
                    ys,
                    marker='x',
                    s=30,
                    alpha=0.5,
                    color=color.value,
                )
            # mean line for this dataset
            means = [mean(ys) if ys else None for ys in dataset]
            ax_right.plot(
                x,
                means,
                color=color.value,
                linestyle=':',
                marker='s',
                markersize=3,
                label=label,
            )

        # formatting right y-axis
        ax_right.set_ylim(*y_2_lim)
        ax_right.yaxis.set_major_locator(MultipleLocator(y_2_tic_steps))
        ax_right.set_ylabel(y_2_label)

    # --- x-axis ---
    ax_left.set_xticks(list(x))
    ax_left.set_xticklabels(x_labels)

    # --- combined legend (left + right) in bottom-right ---
    handles, labels = ax_left.get_legend_handles_labels()
    if ax_right is not None:
        h2, l2 = ax_right.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax_left.legend(loc="upper left", bbox_to_anchor=(1.15, 1.017))

    plt.tight_layout()
    plt.show()


def double_sided_bar_plot(
        ax,
        x_labels: List[str],
        x_title: str,
        y_values: Dict[str, List[Optional[float]]],
        y_title: str,
        y_tic_steps: float,
        bar_colors: List['Color'],
        y_lim: Optional[Tuple[float, float]] = None,
        legend: bool = True,
) -> None:
    """
    Assumes y_values are in [0.0, 1.0] and formats the axis as percentages.
    None values in y_values are skipped (no bar drawn).
    """

    num_groups = len(x_labels)
    model_names = list(y_values.keys())
    num_models = len(model_names)

    x = range(num_groups)
    bar_width = 0.8 / num_models

    # --- bars per model ---
    for i, (model, values) in enumerate(y_values.items()):
        offset = (i - (num_models - 1) / 2) * bar_width
        bar_color = bar_colors[i].value

        for xi, val in enumerate(values):
            # skip missing values
            if val is None:
                continue

            x_pos = xi + offset
            bar = ax.bar(
                x_pos,
                val,
                width=bar_width,
                label=model if xi == 0 else None,  # only label once per model for legend
                color=bar_color,
                alpha=0.85,
                edgecolor='black',
                linewidth=0.5
            )[0]

            # position label in the middle of the bar or above if bar to small
            height = bar.get_height()
            if abs(val) > 0.02:
                text_y = height / 2.0
            elif val < 0:
                text_y = height - 0.01
            else:
                text_y = height + 0.008
            text_color = 'white' if abs(val) > 0.02 else 'black'

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{height * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    # --- horizontal zero line ---
    ax.axhline(0, color="black", linewidth=1, linestyle="-")

    # --- axes formatting ---
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    # ticks & percentage formatting
    ax.yaxis.set_major_locator(MultipleLocator(y_tic_steps))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))  # adjust xmax if your data is 0–100

    # legend outside on the right
    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            borderpad=0.5
        )

def plot_model_agent_heatmap(
        ax,
        data: List[Dict[str, Any]]
) -> None:

    df = pd.DataFrame(data)

    value_cols = [c for c in df.columns if c not in ("model", "agent")]
    df[value_cols] = df[value_cols].replace({None: np.nan}).astype(float)

    # Preserve row order exactly as given
    separated_rows = []
    last_model = None

    for _, row in df.iterrows():
        model = row["model"]

        # Insert separator when model changes
        if last_model is not None and model != last_model:
            blank = {col: np.nan for col in df.columns}
            blank["model"] = ""
            blank["agent"] = ""
            separated_rows.append(blank)

        separated_rows.append(row.to_dict())
        last_model = model

    df_sep = pd.DataFrame(separated_rows)
    df_sep = df_sep.set_index(["model", "agent"])

    values = df_sep[value_cols].values

    # Create custom two-color colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [Color.WHITE.value, Color.DARK_BLUE.value]
    )

    # Heatmap
    im = ax.imshow(values, aspect="auto", cmap=cmap)

    # --- Add percent labels inside each heatmap cell ---
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i,
                    f"{val:.1%}",
                    ha="center",
                    va="center",
                    color="black" if val < 0.5 else "white",
                    fontsize=8,
                )

    # X labels
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_cols)

    # Y labels (two-line form)
    yticklabels = [
        (model if model else "") + ("\n" + agent if agent else "")
        for model, agent in df_sep.index
    ]
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels)

    # Separator lines
    sep_indices = [
        i for i, (m, a) in enumerate(df_sep.index)
        if m == "" and a == ""
    ]
    for idx in sep_indices:
        ax.axhline(idx - 0.5, color="black", linewidth=0.5)
        ax.axhline(idx + 0.5, color="black", linewidth=0.5)

    # --- Colorbar with percent formatting ---
    cbar = plt.colorbar(im, ax=ax)

    def percent_formatter(x, pos):
        return f"{x:.0%}"

    cbar.formatter = FuncFormatter(percent_formatter)
    cbar.update_ticks()


def plot_model_series(ax, data: dict, y_max: int, x_max: int):
    """
        data (dict): Keys are tuples (model_name, variant),
                     values are lists of numeric lists (each plotted as one line).
    """

    # Define a repeating color cycle (can adjust palette)
    colors = itertools.cycle([Color.YELLOW, Color.ORANGE, Color.DARK_BLUE])

    for key, series_list in data.items():
        color = next(colors)  # one color per dictionary key

        for series in series_list:
            x = list(range(len(series)))
            ax.plot(
                x, series,
                linewidth=0.5,
                alpha=0.4,
                color=color.value,
                label=str(key) if series_list.index(series) == 0 else "_nolabel_"
            )
    ax.set_xlabel("group chat messages count")
    ax.set_ylabel("total group chat text length")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.03),
        borderaxespad=0,
        borderpad=0.5,
        labelspacing=0.4,
        handletextpad=1,
        fontsize=8,
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(0, x_max + 1, 2))


def plot_stacked_histograms(data: dict, title: str = ""):
    if not data:
        raise ValueError("data dict is empty")

    # Global x max
    max_x = max((max(vals) for vals in data.values() if vals), default=0)

    # x positions
    x_vals = np.arange(0, max_x + 1)

    # Count data
    counts_per_key = {}
    for key, values in data.items():
        c = Counter(values)
        counts_per_key[key] = np.array([c.get(x, 0) for x in x_vals])

    # Global max count for shared y-axis
    max_y = max(arr.max() for arr in counts_per_key.values())

    # Create stacked subplots
    n = len(data)
    fig, axes = plt.subplots(
        n, 1,
        sharex=True,
        sharey=True,
        figsize=(4, 1.6 * n),
        constrained_layout=True
    )
    fig.suptitle(title, fontsize=10)

    if n == 1:
        axes = [axes]

    for ax, (name, counts) in zip(axes, counts_per_key.items()):
        bars = ax.bar(
            x_vals,
            counts,
            width=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # Add labels to bars
        for rect, value in zip(bars, counts):
            if value == 0:
                continue

            height = rect.get_height()
            x = rect.get_x() + rect.get_width() / 2

            # Decide position: inside if bar tall enough, else above
            if height >= max_y * 0.15:
                # label inside
                ax.text(
                    x, height / 2,
                    str(value),
                    ha='center', va='center',
                    fontsize=8, color='white'
                )
            else:
                # label above
                ax.text(
                    x, height + max_y * 0.03,
                    str(value),
                    ha='center', va='bottom',
                    fontsize=8, color='black'
                )

        ax.set_ylim(0, max_y + 0.5)
        ax.set_ylabel("count")
        ax.set_title(name, loc="left", fontsize=9)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    axes[-1].set_xlabel("value")
    axes[-1].set_xticks(x_vals)
    axes[-1].set_xlim(-0.5, max_x + 0.5)

    plt.show()


def plot_stacked_bar_from_nested_dict(
        data: Dict[str, Dict[int, Set[str]]],
        title: str,
        fig_size: Tuple[int, int] = (4, 3),
):
    # Handle empty input
    if not data:
        raise ValueError("Data dictionary is empty.")

    categories = list(data.keys())  # x-axis labels

    fig, ax = plt.subplots(figsize=fig_size)
    fig.suptitle(title, fontsize=10)

    x_positions = range(len(categories))
    bar_width = 0.6

    for x, category in zip(x_positions, categories):
        inner = data[category]

        # We sort inner keys so the stacking order is deterministic.
        # Adjust if you want original dict order instead.
        bottom = 0
        for segment_label in sorted(inner.keys(), reverse=True):
            items = inner[segment_label]
            height = len(items)

            if height == 0:
                continue  # nothing to draw

            color = Color.GREEN.value if segment_label == 0 else Color.RED.value

            # Draw the bar segment
            ax.bar(
                x,
                height,
                width=bar_width,
                bottom=bottom,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8 - segment_label * 0.1,
            )

            # Add label centered in the segment
            ax.text(
                x,
                bottom + height / 2,
                str(segment_label),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                clip_on=True,
            )

            bottom += height

    # Create legend entries based on keys in inner dicts
    legend_labels = set()
    for inner in data.values():
        legend_labels.update(inner.keys())

    handles = []
    for lbl in sorted(legend_labels):
        color = Color.GREEN.value if lbl == 0 else Color.RED.value
        alpha = 0.8 - lbl * 0.1
        handles.append(
            mpatches.Patch(color=color, linewidth=0, alpha=alpha, label=str(lbl))
        )

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        borderpad=0.5
    )
    # X-axis labels
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories)

    # Y-axis starts at 0
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0, top=max(ymax, 1))  # ensure non-zero upper bound

    ax.set_ylabel("count")

    fig.tight_layout()
    plt.show()


def plot_stacked_bar_with_line(
        ax,
        data: Dict[str, Dict[int, int]],
        y_max: int,
        color: Color,
        additional_line_values: list[float] = None,
):

    categories = list(data.keys())  # x-axis labels
    x_positions = range(len(categories))
    bar_width = 0.6

    # max label
    max_label = max(l for inner in data.values() for l in inner.keys())

    for x, category in zip(x_positions, categories):
        inner = data[category]

        # We sort inner keys so the stacking order is deterministic.
        # Adjust if you want original dict order instead.
        bottom = 0
        for segment_label in sorted(inner.keys(), reverse=True):
            inner_value = inner[segment_label]

            # ignore the o (unsolved counts)
            if segment_label == 0:
                continue

            if inner_value == 0:
                continue  # nothing to draw

            # Draw the bar segment
            face_alpha = 1 - (max_label - segment_label) * 0.15

            ax.bar(
                x,
                inner_value,
                width=bar_width,
                bottom=bottom,
                color=(color_rgb := matplotlib.colors.to_rgb(color.value)) + (face_alpha,),
                edgecolor='black',  # stays opaque
                linewidth=0.5,
            )

            # Add label centered in the segment
            ax.text(
                x,
                bottom + inner_value / 2,
                str(inner_value),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                clip_on=True,
            )

            bottom += inner_value

        # add max line
        maxs = [sum(v.values()) - v[0] for _, v in data.items()]
        ax.plot(
            list(x_positions),
            maxs,
            color=color.value,
            linestyle='--',
            linewidth=1.5,
            alpha=0,
            marker='o',
            markersize=3,
        )

    # add extra line
    extra_line = None
    if additional_line_values is not None:
        extra_line, = ax.plot(
            list(x_positions),
            additional_line_values,
            color='black',
            linestyle=':',
            linewidth=1.5,
            alpha=1.0,
            marker='x',
            markersize=5,
            label='newly solved'
        )

    # Create legend entries based on keys in inner dicts
    handles = []
    for lbl in range(1, max_label + 1):
        alpha = 1 - (max_label - lbl) * 0.15
        handles.append(
            mpatches.Patch(color=color.value, linewidth=0, alpha=alpha, label=f"{lbl} runs")
        )
    handles.append(extra_line)

    ax.legend(
        handles=handles,
        title="solved by...",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2 if len(handles) <= 4 else 3,
        borderaxespad=0,
        borderpad=0.5,  # inner padding (default is 0.4)
        labelspacing=0.4,  # vertical spacing between entries
        handletextpad=1,  # space between marker and text
    )

    # X-axis labels
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories)

    # Y-axis starts at 0
    ax.set_ylim(bottom=0, top=y_max)
    ax.set_ylabel("count")


def plot_error_code_bars(
    ax,
    ec_data: Dict[str, Any],
    label_colors: Dict[str, Color],
    show_legend: bool = True,
):

    # Categories become X-axis labels (vertical bars)
    categories = ["no errors", "errors"]
    x_positions = range(len(categories))
    bar_width = 0.6

    # ---- NO ERRORS BAR (vertical) ----
    no_errors_count = ec_data["no errors"]
    no_errors_color = label_colors.get("no errors")

    ax.bar(
        x=0,
        height=no_errors_count,
        width=bar_width,
        color=no_errors_color.value,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.text(
        0,
        no_errors_count / 2,
        str(no_errors_count),
        ha="center",
        va="center",
        fontsize=8,
        color="white",
    )

    # ---- STACKED ERROR BAR (vertical) ----
    error_dict = ec_data["errors"]

    bottom = 0  # stacking direction
    legend_handles = []

    for label in error_dict.keys():
        val = error_dict[label]
        if val == 0:
            continue

        bar_color = label_colors.get(label)

        ax.bar(
            x=1,
            height=val,
            width=bar_width,
            bottom=bottom,
            color=bar_color.value,
            edgecolor="black",
            linewidth=0.5,
        )

        if val > 1300:
            ax.text(
                1,
                bottom + val / 2,
                str(val),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

        legend_handles.append(
            mpatches.Patch(color=bar_color.value, label=f"exit {label}" if label != "others" else label)
        )

        bottom += val

    # add legend entry for "no errors"
    legend_handles.insert(
        0, mpatches.Patch(color=no_errors_color.value, label="exit 0")
    )

    # ---- LEGEND ----

    if show_legend:
        ax.legend(
            handles=legend_handles,
            title="exit codes",
            loc="upper right",
            bbox_to_anchor=(0.965, 0.95),
            borderaxespad=0,
            borderpad=0.5,
            labelspacing=0.4,
            handletextpad=1,
            fontsize=8,          # label text size
            title_fontsize=8,    # title text size
        )

    # ---- AXIS SETUP ----
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories)  # labels on X-axis

    ax.set_ylabel("count")

    # nice padding on y-axis
    y_max = max(no_errors_count, sum(error_dict.values())) * 1.15
    ax.set_ylim(0, y_max)


def plot_count_bars(ax, data: dict, total: int, main_color: Color, total_label_right: bool = True):

    labels = list(data.keys())
    counts = list(data.values())

    # Determine bar colors
    colors = [Color.DARK_GREY.value if label == "no loop" else main_color.value for label in labels]

    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.8, width=0.7)

    for bar, count in zip(bars, counts):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        if count < 50:
            # label above the bar
            ax.text(x, y + max(counts) * 0.01, str(count), ha="center", va="bottom", fontsize=8)
        else:
            # label inside the bar
            ax.text(x, y / 2, str(count),ha="center", va="center", color="white", fontsize=8)

    # add count label
    ax.text(
        0.965 if total_label_right else 0.3,
        0.93,
        f"total trials: {total}",
        ha="right", va="top",
        transform=ax.transAxes,
        fontsize=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.5)
    )


    ax.set_ylabel("count", fontsize=8)
    positions = range(len(labels))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
