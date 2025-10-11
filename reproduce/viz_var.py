import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from viz import PlotStyle, DEFAULT_COLORS
style = PlotStyle(color_map=DEFAULT_COLORS)

mean1_df = pd.read_csv("L_means_r=1.csv")
std1_df  = pd.read_csv("L_ses_r=1.csv")
mean2_df = pd.read_csv("L_means_r=5.csv")
std2_df  = pd.read_csv("L_ses_r=5.csv")

lo = float(min(mean1_df[["Overall", "Within"]].min().min(),
               mean2_df[["Overall", "Within"]].min().min()))
hi = float(max(mean1_df[["Overall", "Within"]].max().max(),
               mean2_df[["Overall", "Within"]].max().max()))

plt.figure(figsize=style.figsize)

plt.errorbar(
    mean1_df["Overall"], 
    mean1_df["Within"],
    xerr=2 * std1_df["Overall"], 
    yerr=2 * std1_df["Within"],
    fmt="o", 
    ecolor="gray", 
    capsize=4, 
    color=style.color_for("L"),
    markersize=style.marker_size, 
    label="r=1", 
    linewidth=style.line_width,
)
plt.plot(
    mean1_df["Overall"], 
    mean1_df["Within"],
    linestyle="-", 
    color=style.color_for("L"), 
    linewidth=style.line_width
)

plt.errorbar(
    mean2_df["Overall"], 
    mean2_df["Within"],
    xerr=2 * std2_df["Overall"], 
    yerr=2 * std2_df["Within"],
    fmt="o", 
    ecolor="gray", 
    capsize=4, 
    color="green",
    markersize=style.marker_size, 
    label="r=5", 
    linewidth=style.line_width,
)
plt.plot(
    mean2_df["Overall"], 
    mean2_df["Within"],
    linestyle="-", 
    color="green", 
    linewidth=style.line_width
)

plt.plot([lo, hi], [lo, hi], 
          linestyle="--", 
          color="black",
          label="Identity line", 
          linewidth=style.line_width)

plt.xlabel("log(Overall s.d.)", fontsize=style.text_size)
plt.ylabel("log(Within s.d.)", fontsize=style.text_size)

plt.tick_params(axis="both", which="major", length=style.tick_length)
plt.xticks(fontsize=style.number_size)
plt.yticks(fontsize=style.number_size)
plt.legend(fontsize=35)

plt.tight_layout()
plt.savefig("variability.png", dpi=style.dpi)