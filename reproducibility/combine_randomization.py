# Import packages
import numpy as np
import pickle

p_vals_0 = np.zeros((100, 100))
p_vals_1 = np.zeros((100, 100))
p_vals_2 = np.zeros((100, 100))
p_vals_3 = np.zeros((100, 100))
p_vals_4 = np.zeros((100, 100))
p_vals_5 = np.zeros((100, 100))

for i in range(1, 101):
    with open(f'p_vals_{i}.pkl', 'rb') as f:
        p_vals = pickle.load(f)
    p_vals_0[i-1] = p_vals[0]
    p_vals_1[i-1] = p_vals[1]
    p_vals_2[i-1] = p_vals[2]
    p_vals_3[i-1] = p_vals[3]
    p_vals_4[i-1] = p_vals[4]
    p_vals_5[i-1] = p_vals[5]

with open(f'p_vals_set_0.pkl', 'wb') as f:
    pickle.dump(p_vals_0, f)

with open(f'p_vals_set_1.pkl', 'wb') as f:
    pickle.dump(p_vals_1, f)

with open(f'p_vals_set_2.pkl', 'wb') as f:
    pickle.dump(p_vals_2, f)

with open(f'p_vals_set_3.pkl', 'wb') as f:
    pickle.dump(p_vals_3, f)

with open(f'p_vals_set_4.pkl', 'wb') as f:
    pickle.dump(p_vals_4, f)

with open(f'p_vals_set_5.pkl', 'wb') as f:
    pickle.dump(p_vals_5, f)
