# Import packages
import pickle
import numpy as np
import ast

# Set parameters
M = 43 # number of power calcs
N = 1000 # number of simulations

# Create dict with initialized results

avg_powers = {'F_0_5_1_0, O_0_5_1_0, L_0_5_1_0, R1_0_5_1_0, R2_0_5_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_1_5_1_0, O_1_5_1_0, L_1_5_1_0, R1_1_5_1_0, R2_1_5_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_2_5_1_0, O_2_5_1_0, L_2_5_1_0, R1_2_5_1_0, R2_2_5_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_3_5_1_0, O_3_5_1_0, L_3_5_1_0, R1_3_5_1_0, R2_3_5_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_4_5_1_0, O_4_5_1_0, L_4_5_1_0, R1_4_5_1_0, R2_4_5_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_5_5_1_0, O_5_5_1_0, L_5_5_1_0, R1_5_5_1_0, R2_5_5_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_0_5_1_5, O_0_5_1_5, L_0_5_1_5, R1_0_5_1_5, R2_0_5_1_5': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_1_5_1_5, O_1_5_1_5, L_1_5_1_5, R1_1_5_1_5, R2_1_5_1_5': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_2_5_1_5, O_2_5_1_5, L_2_5_1_5, R1_2_5_1_5, R2_2_5_1_5': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_3_5_1_5, O_3_5_1_5, L_3_5_1_5, R1_3_5_1_5, R2_3_5_1_5': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_4_5_1_5, O_4_5_1_5, L_4_5_1_5, R1_4_5_1_5, R2_4_5_1_5': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_5_5_1_5, O_5_5_1_5, L_5_5_1_5, R1_5_5_1_5, R2_5_5_1_5': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_1_0_1_0, O_1_0_1_0, L_1_0_1_0, R1_1_0_1_0, R2_1_0_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_1_1_1_0, O_1_1_1_0, L_1_1_1_0, R1_1_1_1_0, R2_1_1_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_1_2_1_0, O_1_2_1_0, L_1_2_1_0, R1_1_2_1_0, R2_1_2_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_1_3_1_0, O_1_3_1_0, L_1_3_1_0, R1_1_3_1_0, R2_1_3_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_1_4_1_0, O_1_4_1_0, L_1_4_1_0, R1_1_4_1_0, R2_1_4_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_3_0_1_0, O_3_0_1_0, L_3_0_1_0, R1_3_0_1_0, R2_3_0_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_3_1_1_0, O_3_1_1_0, L_3_1_1_0, R1_3_1_1_0, R2_3_1_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_3_2_1_0, O_3_2_1_0, L_3_2_1_0, R1_3_2_1_0, R2_3_2_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_3_3_1_0, O_3_3_1_0, L_3_3_1_0, R1_3_3_1_0, R2_3_3_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_3_4_1_0, O_3_4_1_0, L_3_4_1_0, R1_3_4_1_0, R2_3_4_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_1_2_0_0, O_1_2_0_0, L_1_2_0_0, R1_1_2_0_0, R2_1_2_0_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_1_2_2_0, O_1_2_2_0, L_1_2_2_0, R1_1_2_2_0, R2_1_2_2_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_1_2_3_0, O_1_2_3_0, L_1_2_3_0, R1_1_2_3_0, R2_1_2_3_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_1_2_4_0, O_1_2_4_0, L_1_2_4_0, R1_1_2_4_0, R2_1_2_4_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_1_2_5_0, O_1_2_5_0, L_1_2_5_0, R1_1_2_5_0, R2_1_2_5_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_2_2_0_0, O_2_2_0_0, L_2_2_0_0, R1_2_2_0_0, R2_2_2_0_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 
              'F_2_2_1_0, O_2_2_1_0, L_2_2_1_0, R1_2_2_1_0, R2_2_2_1_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_2_2_2_0, O_2_2_2_0, L_2_2_2_0, R1_2_2_2_0, R2_2_2_2_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 
              'F_2_2_3_0, O_2_2_3_0, L_2_2_3_0, R1_2_2_3_0, R2_2_2_3_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_2_2_4_0, O_2_2_4_0, L_2_2_4_0, R1_2_2_4_0, R2_2_2_4_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 
              'F_2_2_5_0, O_2_2_5_0, L_2_2_5_0, R1_2_2_5_0, R2_2_2_5_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_3_2_0_0, O_3_2_0_0, L_3_2_0_0, R1_3_2_0_0, R2_3_2_0_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_3_2_2_0, O_3_2_2_0, L_3_2_2_0, R1_3_2_2_0, R2_3_2_2_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_3_2_3_0, O_3_2_3_0, L_3_2_3_0, R1_3_2_3_0, R2_3_2_3_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_3_2_4_0, O_3_2_4_0, L_3_2_4_0, R1_3_2_4_0, R2_3_2_4_0': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_3_2_5_0, O_3_2_5_0, L_3_2_5_0, R1_3_2_5_0, R2_3_2_5_0': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_2_2_1_1, O_2_2_1_1, L_2_2_1_1, R1_2_2_1_1, R2_2_2_1_1': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_2_2_1_2, O_2_2_1_2, L_2_2_1_2, R1_2_2_1_2, R2_2_2_1_2': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_2_2_1_3, O_2_2_1_3, L_2_2_1_3, R1_2_2_1_3, R2_2_2_1_3': np.array([0, 0, 0, 0, 0], dtype='float64'), 'F_2_2_1_4, O_2_2_1_4, L_2_2_1_4, R1_2_2_1_4, R2_2_2_1_4': np.array([0, 0, 0, 0, 0], dtype='float64'),
              'F_2_2_1_5, O_2_2_1_5, L_2_2_1_5, R1_2_2_1_5, R2_2_2_1_5': np.array([0, 0, 0, 0, 0], dtype='float64')}

for i in range(1, N + 1):
    with open(f'out_{i}.out', 'r') as file:
        lines = file.readlines()
        for j in range(M):
            line = lines[j]
            powers = np.array(ast.literal_eval(line))
            F_pow = powers[0]
            O_pow = powers[1]
            L_pow = powers[2]
            R1_pow = powers[3]
            R2_pow = powers[4]
            key = list(avg_powers.keys())[j]
            avg_powers[key] += np.array([F_pow, O_pow, L_pow, R1_pow, R2_pow])
for k in range(M):
    key = list(avg_powers.keys())[k]
    val = avg_powers[key]
    F_avg = val[0]
    O_avg = val[1]
    L_avg = val[2]
    R1_avg = val[3]
    R2_avg = val[4]
    avg_powers[key] = (F_avg / N, O_avg / N, L_avg / N, R1_avg / N, R2_avg / N)

# Save powers dict
with open('test_powers.pkl', 'wb') as f:
    pickle.dump(avg_powers, f)
