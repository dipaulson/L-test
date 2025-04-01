# Import packages
import pickle
import numpy as np
import ast

# Set parameters
M = 43 # number of power calcs
N = 1000 # number of simulations

# Create dict with initialized results

avg_powers = {'F_0_5_1_0, O_0_5_1_0, E1_0_5_1_0': np.array([0, 0, 0], dtype='float64'), 'F_1_5_1_0, O_1_5_1_0, E1_1_5_1_0': np.array([0, 0, 0], dtype='float64'),
              'F_2_5_1_0, O_2_5_1_0, E1_2_5_1_0': np.array([0, 0, 0], dtype='float64'), 'F_3_5_1_0, O_3_5_1_0, E1_3_5_1_0': np.array([0, 0, 0], dtype='float64'),
              'F_4_5_1_0, O_4_5_1_0, E1_4_5_1_0': np.array([0, 0, 0], dtype='float64'), 'F_5_5_1_0, O_5_5_1_0, E1_5_5_1_0': np.array([0, 0, 0], dtype='float64'),
              'F_0_5_1_5, O_0_5_1_5, E1_0_5_1_5': np.array([0, 0, 0], dtype='float64'), 'F_1_5_1_5, O_1_5_1_5, E1_1_5_1_5': np.array([0, 0, 0], dtype='float64'),
              'F_2_5_1_5, O_2_5_1_5, E1_2_5_1_5': np.array([0, 0, 0], dtype='float64'), 'F_3_5_1_5, O_3_5_1_5, E1_3_5_1_5': np.array([0, 0, 0], dtype='float64'),
              'F_4_5_1_5, O_4_5_1_5, E1_4_5_1_5': np.array([0, 0, 0], dtype='float64'), 'F_5_5_1_5, O_5_5_1_5, E1_5_5_1_5': np.array([0, 0, 0], dtype='float64'),
              'F_1_0_1_0, O_1_0_1_0, E1_1_0_1_0': np.array([0, 0, 0], dtype='float64'), 'F_1_1_1_0, O_1_1_1_0, E1_1_1_1_0': np.array([0, 0, 0], dtype='float64'),
              'F_1_2_1_0, O_1_2_1_0, E1_1_2_1_0': np.array([0, 0, 0], dtype='float64'), 'F_1_3_1_0, O_1_3_1_0, E1_1_3_1_0': np.array([0, 0, 0], dtype='float64'),
              'F_1_4_1_0, O_1_4_1_0, E1_1_4_1_0': np.array([0, 0, 0], dtype='float64'), 'F_3_0_1_0, O_3_0_1_0, E1_3_0_1_0': np.array([0, 0, 0], dtype='float64'),
              'F_3_1_1_0, O_3_1_1_0, E1_3_1_1_0': np.array([0, 0, 0], dtype='float64'), 'F_3_2_1_0, O_3_2_1_0, E1_3_2_1_0': np.array([0, 0, 0], dtype='float64'),
              'F_3_3_1_0, O_3_3_1_0, E1_3_3_1_0': np.array([0, 0, 0], dtype='float64'), 'F_3_4_1_0, O_3_4_1_0, E1_3_4_1_0': np.array([0, 0, 0], dtype='float64'),
              'F_1_2_0_0, O_1_2_0_0, E1_1_2_0_0': np.array([0, 0, 0], dtype='float64'), 'F_1_2_2_0, O_1_2_2_0, E1_1_2_2_0': np.array([0, 0, 0], dtype='float64'),
              'F_1_2_3_0, O_1_2_3_0, E1_1_2_3_0': np.array([0, 0, 0], dtype='float64'), 'F_1_2_4_0, O_1_2_4_0, E1_1_2_4_0': np.array([0, 0, 0], dtype='float64'),
              'F_1_2_5_0, O_1_2_5_0, E1_1_2_5_0': np.array([0, 0, 0], dtype='float64'), 'F_2_2_0_0, O_2_2_0_0, E1_2_2_0_0': np.array([0, 0, 0], dtype='float64'), 
              'F_2_2_1_0, O_2_2_1_0, E1_2_2_1_0': np.array([0, 0, 0], dtype='float64'), 'F_2_2_2_0, O_2_2_2_0, E1_2_2_2_0': np.array([0, 0, 0], dtype='float64'), 
              'F_2_2_3_0, O_2_2_3_0, E1_2_2_3_0': np.array([0, 0, 0], dtype='float64'), 'F_2_2_4_0, O_2_2_4_0, E1_2_2_4_0': np.array([0, 0, 0], dtype='float64'), 
              'F_2_2_5_0, O_2_2_5_0, E1_2_2_5_0': np.array([0, 0, 0], dtype='float64'), 'F_3_2_0_0, O_3_2_0_0, E1_3_2_0_0': np.array([0, 0, 0], dtype='float64'),
              'F_3_2_2_0, O_3_2_2_0, E1_3_2_2_0': np.array([0, 0, 0], dtype='float64'), 'F_3_2_3_0, O_3_2_3_0, E1_3_2_3_0': np.array([0, 0, 0], dtype='float64'),
              'F_3_2_4_0, O_3_2_4_0, E1_3_2_4_0': np.array([0, 0, 0], dtype='float64'), 'F_3_2_5_0, O_3_2_5_0, E1_3_2_5_0': np.array([0, 0, 0], dtype='float64'),
              'F_2_2_1_1, O_2_2_1_1, E1_2_2_1_1': np.array([0, 0, 0], dtype='float64'), 'F_2_2_1_2, O_2_2_1_2, E1_2_2_1_2': np.array([0, 0, 0], dtype='float64'),
              'F_2_2_1_3, O_2_2_1_3, E1_2_2_1_3': np.array([0, 0, 0], dtype='float64'), 'F_2_2_1_4, O_2_2_1_4, E1_2_2_1_4': np.array([0, 0, 0], dtype='float64'),
              'F_2_2_1_5, O_2_2_1_5, E1_2_2_1_5': np.array([0, 0, 0], dtype='float64')}

for i in range(1, N + 1):
    with open(f'out_{i}.out', 'r') as file:
        lines = file.readlines()
        for j in range(M):
            line = lines[j]
            powers = np.array(ast.literal_eval(line))
            F_pow = powers[0]
            O_pow = powers[1]
            E1_pow = powers[2]
            key = list(avg_powers.keys())[j]
            avg_powers[key] += np.array([F_pow, O_pow, E1_pow])
for k in range(M):
    key = list(avg_powers.keys())[k]
    val = avg_powers[key]
    F_avg = val[0]
    O_avg = val[1]
    E1_avg = val[2]
    avg_powers[key] = (F_avg / N, O_avg / N, E1_avg / N)

# Save powers dict
with open('test_powers.pkl', 'wb') as f:
    pickle.dump(avg_powers, f)
