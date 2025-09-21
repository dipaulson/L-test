import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from viz import PlotStyle, DEFAULT_COLORS
style = PlotStyle(color_map = DEFAULT_COLORS)

MC = [500, 5000, 50000, 500000]

# Time plot
times_MC_500 = pd.read_csv("times_MC=500.csv", index_col=0)
F_time = np.array([times_MC_500.loc["F", "Avg"]]*4)
R_time = np.array([times_MC_500.loc["R", "Avg"]]*4)

L_time = np.empty(4, dtype=float)
for i in range(4):
    times = pd.read_csv(f"times_MC={MC[i]}.csv", index_col=0)
    L_time[i] = times.loc["L", "Avg"]

plt.figure(figsize=style.figsize)
plt.plot(MC, F_time, label="F", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("F"))
plt.plot(MC, L_time, label="L", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("L"))
plt.plot(MC, R_time, label="R", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("R"))

plt.yscale("log")
plt.xscale("log")
plt.tick_params(axis='both', which='major', length=style.tick_length)
plt.xticks(fontsize=style.number_size)
plt.yticks(fontsize=style.number_size)
plt.xlabel("Number of MC Samples", fontsize=style.text_size)
plt.ylabel("Avg. time", fontsize=style.text_size)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("times.png", dpi=style.dpi)
plt.close()


# Power plot
powers_MC_500 = pd.read_csv("powers_MC=500.csv", index_col=0)
F_holm = np.array([powers_MC_500.loc["F", "Holm (5%) (avg)"]]*4)
F_bhq = np.array([powers_MC_500.loc["F", "BHq (10%) (avg)"]]*4)
R_holm = np.array([powers_MC_500.loc["R", "Holm (5%) (avg)"]]*4)
R_bhq = np.array([powers_MC_500.loc["R", "BHq (10%) (avg)"]]*4)

L_holm = np.empty(4, dtype=float)
L_bhq = np.empty(4, dtype=float)
for i in range(4):
    powers = pd.read_csv(f"powers_MC={MC[i]}.csv", index_col=0)
    L_holm[i] = powers.loc["L", "Holm (5%) (avg)"]
    L_bhq[i] = powers.loc["L", "BHq (10%) (avg)"]

plt.figure(figsize=style.figsize)
plt.plot(MC, F_holm, label="F Holm (5%)", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("F"), linestyle="--")
plt.plot(MC, F_bhq,  label="F BHq (10%)", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("F"))

plt.plot(MC, L_holm, label="L Holm", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("L"), linestyle="--")
plt.plot(MC, L_bhq, label="L BHq", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("L"))

plt.plot(MC, R_holm, label="R Holm", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("R"), linestyle="--")
plt.plot(MC, R_bhq, label="R BHq", marker="o", markersize=style.marker_size, linewidth=style.line_width, color=style.color_for("R"))

plt.xscale("log")
plt.tick_params(axis='both', which='major', length=style.tick_length)
plt.xticks(fontsize=style.number_size)
plt.yticks(fontsize=style.number_size)
plt.xlabel("Number of MC Samples", fontsize=style.text_size)
plt.ylabel("Avg. power", fontsize=style.text_size)
plt.ylim(0, 0.3)
plt.legend(ncol=3, fontsize=25)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("MT_powers.png", dpi=style.dpi)
plt.close()
