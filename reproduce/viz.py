import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass
from typing import Iterable, Optional, Dict

@dataclass(frozen=True)
class PlotStyle:
    figsize = (12, 10)
    dpi = 300
    text_size = 40
    number_size = 30
    tick_length = 10
    marker_size = 8
    line_width = 4
    color_map: Optional[Dict[str, str]] = None

    def color_for(self, key):
        if self.color_map is None:
            return None
        return self.color_map.get(key)

DEFAULT_COLORS = {
    "F-test": "blue",
    "Oracle": "gray",
    "gLASSO": "green",
    "sgLASSO": "green",
    "LASSO": "gold",
    "Enet": "black",
    "L-test": "red",
    "MC-free": "gold",
    "PC-test": "orange",   
    r"$\phi$-test": "purple",  
    r"Bonf-$\ell$": "black",
    "L_min": "red",
    "L_1se": "lightsalmon",
    "L_full": "tan",
    "L_proj": "lightsalmon"
}

LEGEND_LABELS = {
    "L_min": "Min rule",
    "L_1se": "1se rule",
    "L_full": r"$(y, X)$",
    "L_proj": r"$(\hat{y}_{1:k}, X)$"
}

def plot_errorbars_from_df(
    mean_df: pd.DataFrame,
    se_df: pd.DataFrame,
    style: PlotStyle,
    y_ticks: Iterable[float],
    y_label: str,
    file_name: str,
    fig_title: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
):
    x = mean_df.iloc[:, 0]
    plt.figure(figsize=style.figsize)
    for col in mean_df.columns[1:]:
        y = mean_df[col]
        y_std = se_df[col]
        label = label_map.get(col, col) if label_map is not None else col
        linestyle = ":" if col == "Oracle" else "-"
        plt.errorbar(
            x,
            y,
            yerr=2*y_std,
            fmt="o",
            linestyle=linestyle,
            markersize=style.marker_size,
            linewidth=style.line_width,
            label=label,
            color=style.color_for(col)
        )

    plt.xlabel(mean_df.columns[0], fontsize=style.text_size)
    plt.ylabel(y_label, fontsize=style.text_size)

    plt.tick_params(axis='both', which='major', length=style.tick_length)
    plt.xticks(fontsize=style.number_size)
    plt.yticks(y_ticks, fontsize=style.number_size)

    if fig_title:
        plt.title(fig_title, fontsize=style.text_size)

    desired_order = ["L-test", "MC-free", "F-test", "gLASSO", r"Bonf-$\ell$", "Oracle"]
    handles, labels = plt.gca().get_legend_handles_labels()
    ordering = [labels.index(lbl) for lbl in desired_order if lbl in labels]
    handles = [handles[i] for i in ordering]
    labels  = [labels[i] for i in ordering]

    plt.legend(handles, labels, fontsize=35)
    plt.tight_layout()
    plt.savefig(f"{file_name}", dpi=style.dpi)
    plt.close()

def plot_test_powers(
    csv_path,
    num_sims = 1000,
    style = PlotStyle(color_map=DEFAULT_COLORS)
):
    power_avgs = pd.read_csv(csv_path)
    power_ses = power_avgs.copy()
    cols = power_avgs.columns[1:]
    power_ses[cols] = np.sqrt(power_avgs[cols] * (1.0 - power_avgs[cols]) / num_sims)
    plot_errorbars_from_df(
        power_avgs,
        power_ses,
        style=style,
        y_ticks=np.arange(0, 1.1, step=0.2),
        y_label="Power",
        file_name = "avg_powers.png",
        label_map=LEGEND_LABELS
    )

def plot_test_type1_errors(
    pickle_path,
    num_sims = 1000,
    style = PlotStyle(color_map=DEFAULT_COLORS)
):
    with open(pickle_path, "rb") as f:
        avg_errors = pickle.load(f)
    grids = {
        "heavy_tail": (r"\nu", np.array([2, 5, 10, 15, 20, 30])),
        "skewed": (r"\alpha", np.array([1, 2, 4, 6, 8, 10])),
        "hetero": (r"\eta^2", np.array([0.01, 0.25, 0.5, 1, 4, 8])),
        "non_linear": (r"\delta", np.array([0.3, 0.5, 1, 2, 3, 4])),
    }
    for violation, (symbol, values) in grids.items():
        for s, val in enumerate(values):
            error_avg = avg_errors[f"{violation}_{s}"]
            error_se = error_avg.copy()
            cols = error_avg.columns[1:]
            error_se[cols] = np.sqrt(error_avg[cols] * (1.0 - error_avg[cols]) / num_sims)
            fig_title = f"${symbol} = {val}$"
            plot_errorbars_from_df(
                error_avg,
                error_se,
                style=style,
                y_ticks=np.arange(0, 0.16, step=0.05),
                y_label="Type I error",
                file_name = f"{violation}_{s}.png",
                fig_title=fig_title,
                label_map=LEGEND_LABELS
            )

"""
To plot average powers of tests, uncomment the following:
"""
# plot_test_powers("avg_powers.csv")

"""
To plot average type I errors associated with the
robustness experiments, uncomment the following:
"""
# plot_test_type1_errors("avg_errors.pkl")
