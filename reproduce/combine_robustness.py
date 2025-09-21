import pandas as pd
import pickle

num_sims = 1000
num_jobs = 250

violations = ["heavy_tail", "skewed", "hetero", "non_linear"]
num_settings = 6

errors = {}
for vtype in violations:
    for s in range(num_settings):
        with open("rejections_1.pkl", "rb") as f:
            rejections_1 = pickle.load(f)
        total_rejections = rejections_1[f"{vtype}_{s}"]
        for i in range(2, num_jobs+1):
            with open(f"rejections_{i}.pkl", "rb") as f:
                rejections_i = pickle.load(f)
            total_rejections.iloc[:, 1:] += rejections_i[f"{vtype}_{s}"].iloc[:, 1:]

        total_rejections.iloc[:, 1:] /= num_sims
        key = f"{vtype}_{s}"
        errors[key] = total_rejections

with open(f"avg_errors.pkl", "wb") as f:
    pickle.dump(errors, f)