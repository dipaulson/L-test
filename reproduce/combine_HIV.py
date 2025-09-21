import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from viz import PlotStyle, DEFAULT_COLORS
style = PlotStyle(color_map = DEFAULT_COLORS)

# We use test significance level of 0.05.
alpha = 0.05

# Dictionary to store test discoveries by null group size
disc = {key: [[], [], [], []] for key in range(1, 12)}

# Dictionary to store number of hypothesis tests of certain null group size
freq = {key: 0 for key in range(1, 12)}

for i in range(1, 17):
    p_vals = pd.read_csv(f"p_vals/p_vals_reg_{i}.csv", header=None).to_numpy()
    num_tests = p_vals.shape[0]
    for j in range(num_tests):
        k = int(p_vals[j][4])
        freq[k] += 1
        disc[k][0].append(float(p_vals[j][0]<alpha))
        disc[k][1].append(float(p_vals[j][1]<alpha))
        disc[k][2].append(float(p_vals[j][2]<alpha))
        disc[k][3].append(float(p_vals[j][3]<alpha))

# Test powers by null group size
powers = {key: tuple([sum(disc[key][i]) / freq[key] for i in range(4)]) for key in disc}

# Total test discoveries
tot_tests = sum(freq[key] for key in freq)
F_disc = sum(sum(value[0]) for value in disc.values())
glasso_disc = sum(sum(value[1]) for value in disc.values())
L_disc = sum(sum(value[2]) for value in disc.values())
R_disc = sum(sum(value[3]) for value in disc.values())

"""
Uncomment to generate barplot of test powers by null group size. 
"""
# keys = list(powers.keys())
# F_powers = [value[0] for value in powers.values()]
# glasso_powers = [value[1] for value in powers.values()]
# L_powers = [value[2] for value in powers.values()]
# R_powers = [value[3] for value in powers.values()]

# x = np.arange(len(keys))
# bar_width = 0.2
# plt.figure(figsize=(10, 5))
# plt.bar(x - 1.5 * bar_width, F_powers, width=bar_width, label='F-test', color=style.color_for("F"))
# plt.bar(x - 0.5 * bar_width, glasso_powers, width=bar_width, label='glasso-test', color=style.color_for("glasso"))
# plt.bar(x + 0.5 * bar_width, L_powers, width=bar_width, label='L-test', color=style.color_for("L"))
# plt.bar(x + 1.5 * bar_width, R_powers, width=bar_width, label='R-test', color=style.color_for("R"))

# plt.title(f"F-test: {F_disc/tot_tests:.2f}, glasso-test: {glasso_disc/tot_tests:.2f}, L-test: {L_disc/tot_tests:.2f}, R-test: {R_disc/tot_tests:.2f}", fontsize=style.title_size)

# plt.xlabel('Group size', fontsize=style.label_size)
# plt.ylabel('Power', fontsize=style.label_size)

# plt.xticks(x, keys, fontsize=style.tick_size)
# plt.yticks(fontsize=style.tick_size)

# plt.legend()
# plt.tight_layout()
# plt.savefig("powers_by_null_group_size.png", dpi=style.dpi, bbox_inches="tight")
# plt.close()

"""
Uncomment to generate bar plot of frequencies of group sizes.
"""
# keys = list(freq.keys())
# freq = list(freq.values())

# x = np.arange(len(keys))
# bar_width = 0.8
# plt.figure(figsize=(10, 5))
# plt.bar(x, freq, width=bar_width, color='skyblue')

# plt.xlabel('Group size', fontsize=style.label_size)
# plt.ylabel('Frequency', fontsize=style.label_size)

# plt.xticks(x, keys, fontsize=style.tick_size)
# plt.yticks(fontsize=style.tick_size)

# plt.tight_layout()
# plt.savefig("frequencies.png", dpi=style.dpi, bbox_inches="tight")
# plt.close()

"""
Uncomment to generate plot of test powers across hypothesis tests of group size
at least a certain value. 
"""
# disc_agg = {key: [0, 0, 0, 0] for key in range(1, 12)}
# for key1 in range(1, 12):
#     discoveries = [0, 0, 0, 0]
#     for key2 in range(key1, 12):  
#         for i in range(4):
#             discoveries[i] += sum(disc[key2][i])
#     disc_agg[key1] = discoveries

# freq_agg = {key: 0 for key in range(1, 12)}
# for key1 in range(1, 12):
#     num_tests = 0
#     for key2 in range(key1, 12):
#         num_tests += freq[key2]
#     freq_agg[key1] = num_tests

# powers_agg = {key: tuple([disc_agg[key][i] / freq_agg[key] for i in range(4)]) for key in disc_agg}

# # Extract data for plotting
# keys = list(powers_agg.keys())
# x = np.arange(len(keys))
# F_powers = [value[0] for value in powers_agg.values()]
# glasso_powers = [value[1] for value in powers_agg.values()]
# L_powers = [value[2] for value in powers_agg.values()]
# R_powers = [value[3] for value in powers_agg.values()]

# # Plot with error bars
# plt.figure(figsize=style.figsize)
# plt.plot(x, F_powers, marker='o', markersize=style.marker_size, label='F-test', color=style.color_for("F"))
# plt.plot(x, glasso_powers, marker='o', markersize=style.marker_size, label='glasso-test', color=style.color_for("glasso"))
# plt.plot(x, L_powers, marker='o', markersize=style.marker_size, label='L-test', color=style.color_for("L"))
# plt.plot(x, R_powers, marker='o', markersize=style.marker_size, label='R-test', color=style.color_for("R"))

# plt.xlabel('Null group size', fontsize=style.label_size)
# plt.ylabel('Average power', fontsize=style.label_size)

# plt.xticks(fontsize=style.tick_size)
# plt.yticks(fontsize=style.tick_size)

# plt.title('Power comparison', fontsize=style.title_size)

# plt.legend()
# plt.tight_layout()
# plt.savefig("power_plot.png", dpi=style.dpi)
# plt.close()