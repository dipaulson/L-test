# Import packages
import numpy as np
import pickle

N = 1000

violations = ["tail", "skew", "sked", "lin"]

# Setting 1: Heavy-tailed errors
powers_F_tail = np.zeros((6, 7))
powers_L_tail = np.zeros((6, 7))

# Setting 2: Skewed errors
powers_F_skew = np.zeros((6, 7))
powers_L_skew = np.zeros((6, 7))

# Setting 3: Heteroskedastic errors
powers_F_sked = np.zeros((6, 7))
powers_L_sked = np.zeros((6, 7))

# Setting 4: Model non-linearity
powers_F_lin = np.zeros((6, 7))
powers_L_lin = np.zeros((6, 7))

for i in range(1, N + 1):
    with open(f'out_{i}.out', 'r') as file:
        lines = file.readlines()
        for i in range(4):
            for j in range(6):
                for k in range(7):
                    line = lines[42*i+(j*7+k)]
                    F_pow = float(line[1])
                    L_pow = float(line[4])
                    if (violations[i] == "tail"):
                        powers_F_tail[j, k] += F_pow
                        powers_L_tail[j, k] += L_pow
                    elif (violations[i] == "skew"):
                        powers_F_skew[j, k] += F_pow
                        powers_L_skew[j, k] += L_pow
                    elif (violations[i] == "sked"):
                        powers_F_sked[j, k] += F_pow
                        powers_L_sked[j, k] += L_pow
                    else:
                        powers_F_lin[j, k] += F_pow
                        powers_L_lin[j, k] += L_pow

# Save data
powers_F_tail = powers_F_tail / N
powers_L_tail = powers_L_tail / N
with open('F_tail.pkl', 'wb') as f:
    pickle.dump(powers_F_tail, f)
with open('L_tail.pkl', 'wb') as f:
    pickle.dump(powers_L_tail, f)

powers_F_skew = powers_F_skew / N
powers_L_skew = powers_L_skew / N
with open('F_skew.pkl', 'wb') as f:
    pickle.dump(powers_F_skew, f)
with open('L_skew.pkl', 'wb') as f:
    pickle.dump(powers_L_skew, f)

powers_F_sked = powers_F_sked / N
powers_L_sked = powers_L_sked / N
with open('F_sked.pkl', 'wb') as f:
    pickle.dump(powers_F_sked, f)
with open('L_sked.pkl', 'wb') as f:
    pickle.dump(powers_L_sked, f)

powers_F_lin = powers_F_lin / N
powers_L_lin = powers_L_lin / N
with open('F_lin.pkl', 'wb') as f:
    pickle.dump(powers_F_lin, f)
with open('L_lin.pkl', 'wb') as f:
    pickle.dump(powers_L_lin, f)
