# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize data structures

"""
Dictionaries to store number of discoveries for F- and L-tests 
by null group size for each drug type
"""
disc_1 = {key: [0, 0, 0] for key in range(1, 12)}
disc_2 = {key: [0, 0, 0] for key in range(1, 12)}
disc_3 = {key: [0, 0, 0] for key in range(1, 12)}

"""
# Dictionaries to store number of tests with certain null group size
for each drug type
"""
freq_1 = {key: 0 for key in range(1, 12)}
freq_2 = {key: 0 for key in range(1, 12)}
freq_3 = {key: 0 for key in range(1, 12)}

for i in range(1, 17):
    if (1<=i<=7):
        p_vals = pd.read_csv(f"p_values/p_vals_reg_{i}.csv", header=None).to_numpy()
        num_tests = p_vals.shape[0]
        for j in range(num_tests):
            k = int(p_vals[j][3])
            freq_1[k] += 1
            disc_1[k][0] += float(p_vals[j][0]<0.05)
            disc_1[k][1] += float(p_vals[j][1]<0.05)
            disc_1[k][2] += float(p_vals[j][2]<0.05)
    elif (8<=i<=13):
        p_vals = pd.read_csv(f"p_values/p_vals_reg_{i}.csv", header=None).to_numpy()
        num_tests = p_vals.shape[0]
        for j in range(num_tests):
            k = int(p_vals[j][3])
            freq_2[k] += 1
            disc_2[k][0] += float(p_vals[j][0]<0.05)
            disc_2[k][1] += float(p_vals[j][1]<0.05)
            disc_2[k][2] += float(p_vals[j][2]<0.05)
    else:
        p_vals = pd.read_csv(f"p_values/p_vals_reg_{i}.csv", header=None).to_numpy()
        num_tests = p_vals.shape[0]
        for j in range(num_tests):
            k = int(p_vals[j][3])
            freq_3[k] += 1
            disc_3[k][0] += float(p_vals[j][0]<0.05)
            disc_3[k][1] += float(p_vals[j][1]<0.05)
            disc_3[k][2] += float(p_vals[j][2]<0.05)

# Powers of F- and L-tests by null group size for each drug type
powers_1 = {
    key: (disc_1[key][0] / freq_1[key], disc_1[key][1] / freq_1[key], disc_1[key][2] / freq_1[key])
    for key in disc_1 if freq_1[key] != 0
    }
powers_2 = {
    key: (disc_2[key][0] / freq_2[key], disc_2[key][1] / freq_2[key], disc_2[key][2] / freq_2[key])
    for key in disc_2 if freq_2[key] != 0
    }
powers_3 = {
    key: (disc_3[key][0] / freq_3[key], disc_3[key][1] / freq_3[key], disc_3[key][2] / freq_3[key])
    for key in disc_3 if freq_3[key] != 0
    }
powers = [powers_1, powers_2, powers_3]

# Plot powers by null group size for each drug type
for i in range(3):
    keys = list(powers[i].keys())
    F_powers = [value[0] for value in powers[i].values()]
    L_powers = [value[1] for value in powers[i].values()]
    R_powers = [value[2] for value in powers[i].values()]
    x = np.arange(len(keys))
    bar_width = 0.25
    plt.bar(x - bar_width, F_powers, width=bar_width, label='F-test', color='#264653')
    plt.bar(x, L_powers, width=bar_width, label='L-test', color='#2a9d8f')
    plt.bar(x + bar_width, R_powers, width=bar_width, label='Recentered F-test', color='#8ab17d')
    plt.xlabel('Group size', fontsize=14)
    plt.ylabel('Power', fontsize=14)
    plt.xticks(x, keys)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/powers_drug_type_{i}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Powers of F- and L-tests by null group size overall
disc_overall = {key: (disc_1[key][0]+ disc_2[key][0] + disc_3[key][0], disc_1[key][1]+ disc_2[key][1] + disc_3[key][1], disc_1[key][2]+ disc_2[key][2] + disc_3[key][2]) for key in disc_1}
freq_overall = {key: (freq_1[key]+ freq_2[key] + freq_3[key]) for key in freq_1}
powers_overall = {
    key: (disc_overall[key][0] / freq_overall[key], disc_overall[key][1] / freq_overall[key], disc_overall[key][2] / freq_overall[key])
    for key in disc_overall if freq_overall[key] != 0
    }


# Plot powers by null group size overall
keys = list(powers_overall.keys())
F_powers = [value[0] for value in powers_overall.values()]
L_powers = [value[1] for value in powers_overall.values()]
R_powers = [value[2] for value in powers_overall.values()]
x = np.arange(len(keys))
bar_width = 0.25
plt.bar(x - bar_width, F_powers, width=bar_width, label='F-test', color='#264653')
plt.bar(x, L_powers, width=bar_width, label='L-test', color='#2a9d8f')
plt.bar(x + bar_width, R_powers, width=bar_width, label='Recentered F-test', color='#8ab17d')
plt.xlabel('Group size', fontsize=14)
plt.ylabel('Power', fontsize=14)
plt.xticks(x, keys)
#plt.legend()
plt.tight_layout()
plt.savefig(f"plots/powers_overall.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot frequences of null group sizes overall
keys = list(freq_overall.keys())
freq = list(freq_overall.values())
x = np.arange(len(keys))
bar_width = 0.8
plt.bar(x, freq, width=bar_width, color='skyblue')
plt.xlabel('Group size', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(x, keys)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/frequencies.png", dpi=300, bbox_inches="tight")
plt.close()

# Compute overall F- and L-test powers for each drug type
tot_tests_1 = sum(freq_1.values())
F_disc_1 = sum(value[0] for value in disc_1.values())
L_disc_1 = sum(value[1] for value in disc_1.values())
R_disc_1 = sum(value[2] for value in disc_1.values())
print("Drug type 1")
print(f"F-test power: {F_disc_1/tot_tests_1}")
print(f"L-test power: {L_disc_1/tot_tests_1}")
print(f"Recented F-test power: {R_disc_1/tot_tests_1}")

tot_tests_2 = sum(freq_2.values())
F_disc_2 = sum(value[0] for value in disc_2.values())
L_disc_2 = sum(value[1] for value in disc_2.values())
R_disc_2 = sum(value[2] for value in disc_2.values())
print("Drug type 2")
print(f"F-test power: {F_disc_2/tot_tests_2}")
print(f"L-test power: {L_disc_2/tot_tests_2}")
print(f"Recented F-test power: {R_disc_2/tot_tests_2}")

tot_tests_3 = sum(freq_3.values())
F_disc_3 = sum(value[0] for value in disc_3.values())
L_disc_3 = sum(value[1] for value in disc_3.values())
R_disc_3 = sum(value[2] for value in disc_3.values())
print("Drug type 3")
print(f"F-test power: {F_disc_3/tot_tests_3}")
print(f"L-test power: {L_disc_3/tot_tests_3}")
print(f"Recented F-test power: {R_disc_3/tot_tests_3}")

# Compute overall F- and L-test powers
tot_tests = sum(freq_overall.values())
F_disc = sum(value[0] for value in disc_overall.values())
L_disc = sum(value[1] for value in disc_overall.values())
R_disc = sum(value[2] for value in disc_overall.values())
print("Overall powers")
print(f"F-test power: {F_disc/tot_tests}")
print(f"L-test power: {L_disc/tot_tests}")
print(f"Recented F-test power: {R_disc/tot_tests}")